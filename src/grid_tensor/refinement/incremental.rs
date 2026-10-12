//! Per-action delta updates of the row caches and the exact and binned prefix sums.

use super::prefix::{add_statistics, sub_stats};
use super::RefinementStrategy;
use crate::grid_tensor::state::{FittingState, PrefixStats};

impl RefinementStrategy {
    pub fn update_statistics<'a>(
        &self,
        state: FittingState<'a>,
        col: usize,
        start: usize,
        end: usize,
        affected_points_range: &[(usize, usize)],
    ) -> FittingState<'a> {
        self.update_statistics_impl::<true>(state, col, start, end, affected_points_range)
    }

    /// Retain exact incremental curvature before a full-row reference rebuild.
    pub(crate) fn update_curvature_statistics<'a>(
        &self,
        state: FittingState<'a>,
        col: usize,
        start: usize,
        end: usize,
        affected_points_range: &[(usize, usize)],
    ) -> FittingState<'a> {
        assert!(matches!(self, Self::L2Refinement { .. }));
        assert!(state
            .branch_consensus
            .as_ref()
            .is_some_and(|reference| reference.anchors.is_none()));
        self.update_statistics_impl::<false>(state, col, start, end, affected_points_range)
    }

    fn update_statistics_impl<'a, const LINEAR: bool>(
        &self,
        mut state: FittingState<'a>,
        col: usize,
        start: usize,
        end: usize,
        affected_points_range: &[(usize, usize)],
    ) -> FittingState<'a> {
        state.require_current_statistics();
        let updated_points = &state.precomputed_statistics.sorted_indices[col][start..end];
        if updated_points.is_empty() {
            return state;
        }

        // 1) Compute the retained per-point statistics and their deltas vs stored c_*.
        // Precomputing deltas avoids re-loading old c_* values for every affected column.
        let residuals = &state.residuals;

        let (c_s11_new, c_s22_new, c_s12_new, c_t1_new, c_t2_new, d_s11, d_s22, d_s12, d_t1, d_t2) = {
            let old = &state.precomputed_statistics.curvature;
            let old_t1 = &state.precomputed_statistics.c_t1;
            let old_t2 = &state.precomputed_statistics.c_t2;

            let mut c_s11_new = Vec::with_capacity(updated_points.len());
            let mut c_s22_new = Vec::with_capacity(updated_points.len());
            let mut c_s12_new = Vec::with_capacity(updated_points.len());
            let mut c_t1_new = Vec::with_capacity(if LINEAR { updated_points.len() } else { 0 });
            let mut c_t2_new = Vec::with_capacity(if LINEAR { updated_points.len() } else { 0 });
            let mut d_s11 = Vec::with_capacity(updated_points.len());
            let mut d_s22 = Vec::with_capacity(updated_points.len());
            let mut d_s12 = Vec::with_capacity(updated_points.len());
            let mut d_t1 = Vec::with_capacity(if LINEAR { updated_points.len() } else { 0 });
            let mut d_t2 = Vec::with_capacity(if LINEAR { updated_points.len() } else { 0 });

            for &pt in updated_points.iter() {
                let PrefixStats {
                    s11: new_s11,
                    s22: new_s22,
                    s12: new_s12,
                    t1: new_t1,
                    t2: new_t2,
                } = if LINEAR {
                    state.point_statistics(pt, self.weight(residuals[pt]))
                } else {
                    state.point_curvature_statistics(pt, self.weight(residuals[pt]))
                };

                c_s11_new.push(new_s11);
                c_s22_new.push(new_s22);
                c_s12_new.push(new_s12);
                if LINEAR {
                    c_t1_new.push(new_t1);
                    c_t2_new.push(new_t2);
                }

                d_s11.push(new_s11 - old[pt].s11);
                d_s22.push(new_s22 - old[pt].s22);
                d_s12.push(new_s12 - old[pt].s12);
                if LINEAR {
                    d_t1.push(new_t1 - old_t1[pt]);
                    d_t2.push(new_t2 - old_t2[pt]);
                }
            }

            (
                c_s11_new, c_s22_new, c_s12_new, c_t1_new, c_t2_new, d_s11, d_s22, d_s12, d_t1,
                d_t2,
            )
        };

        // 2) For other (non-split) columns, apply delta updates.
        //    Binned columns: scatter deltas into B-length bin buckets and
        //    sweep cumulants over the affected bin range (~B work instead of
        //    ~n work — the main optimization).
        //    Exact columns: original O(span) scatter + sweep over length-n
        //    prefix sums (parity path).
        //
        //    The bucket scatter groups the retained deltas at offset `off`
        //    into one `PrefixStats` per bucket slot so the prefix-sum
        //    sweep touches a single 40-byte struct per index instead of
        //    five independent `Vec<f64>`. Order of additions is unchanged.
        let mut bucket_d: Vec<PrefixStats> = Vec::new();
        for (c, &(min_pos, max_pos)) in affected_points_range
            .iter()
            .enumerate()
            .filter(|&(c, _)| c != col)
        {
            if min_pos > max_pos {
                continue;
            }
            let is_binned = !state.precomputed_statistics.bin_edges[c].is_empty();

            if is_binned {
                // ---- Binned path ----
                let bin_edges = &state.precomputed_statistics.bin_edges[c];
                let min_bin = bin_edges.partition_point(|&e| e <= min_pos);
                let max_bin = bin_edges.partition_point(|&e| e <= max_pos);
                let bin_span = max_bin - min_bin + 1;

                bucket_d.clear();
                bucket_d.resize(bin_span, PrefixStats::ZERO);

                let bin_indices = &state.precomputed_statistics.bin_indices[c];
                for (j, &pt) in updated_points.iter().enumerate() {
                    let bin = bin_indices[pt] as usize;
                    let off = bin - min_bin;
                    let slot = &mut bucket_d[off];
                    slot.s11 += d_s11[j];
                    slot.s22 += d_s22[j];
                    slot.s12 += d_s12[j];
                    if LINEAR {
                        slot.t1 += d_t1[j];
                        slot.t2 += d_t2[j];
                    }
                }

                let b = &mut state.precomputed_statistics.binned_prefix_sums[c];
                let mut acc_d = PrefixStats::ZERO;
                for off in 0..bin_span {
                    add_statistics::<LINEAR>(&mut acc_d, bucket_d[off]);
                    add_statistics::<LINEAR>(&mut b[min_bin + off], acc_d);
                }
                if let Some(tail) = b.get_mut((max_bin + 1)..) {
                    for v in tail {
                        add_statistics::<LINEAR>(v, acc_d);
                    }
                }
            } else {
                // ---- Exact path (length-n prefix sums) ----
                let span = max_pos - min_pos + 1;
                bucket_d.clear();
                bucket_d.resize(span, PrefixStats::ZERO);

                let sort_order = &state.precomputed_statistics.sort_order[c];
                let prefix_sums = &mut state.precomputed_statistics.prefix_sums[c];

                for (j, &pt) in updated_points.iter().enumerate() {
                    let pos_full = sort_order[pt];
                    let off = pos_full - min_pos;
                    let slot = &mut bucket_d[off];
                    slot.s11 += d_s11[j];
                    slot.s22 += d_s22[j];
                    slot.s12 += d_s12[j];
                    if LINEAR {
                        slot.t1 += d_t1[j];
                        slot.t2 += d_t2[j];
                    }
                }

                let mut acc_d = PrefixStats::ZERO;
                let range = &mut prefix_sums[min_pos..=max_pos];
                for off in 0..span {
                    add_statistics::<LINEAR>(&mut acc_d, bucket_d[off]);
                    add_statistics::<LINEAR>(&mut range[off], acc_d);
                }

                if let Some(tail) = prefix_sums.get_mut((max_pos + 1)..) {
                    for v in tail {
                        add_statistics::<LINEAR>(v, acc_d);
                    }
                }
            }
        }

        // 3) Split column: same path-split as non-split columns, but the
        //    affected range is the full interval [start..end).
        //    Binned path uses the same bin-indexed delta scatter as above.
        //    Exact path keeps the original rebuild-from-new-values logic for
        //    parity with the prior implementation.
        let split_is_binned = !state.precomputed_statistics.bin_edges[col].is_empty();

        if split_is_binned {
            // Binned split column: bin-indexed delta scatter, identical to
            // the non-split binned loop above but using (start, end-1).
            let min_pos = start;
            let max_pos = end - 1;

            let bin_edges = &state.precomputed_statistics.bin_edges[col];
            let min_bin = bin_edges.partition_point(|&e| e <= min_pos);
            let max_bin = bin_edges.partition_point(|&e| e <= max_pos);
            let bin_span = max_bin - min_bin + 1;

            bucket_d.clear();
            bucket_d.resize(bin_span, PrefixStats::ZERO);

            let bin_indices = &state.precomputed_statistics.bin_indices[col];
            for (j, &pt) in updated_points.iter().enumerate() {
                let bin = bin_indices[pt] as usize;
                let off = bin - min_bin;
                let slot = &mut bucket_d[off];
                slot.s11 += d_s11[j];
                slot.s22 += d_s22[j];
                slot.s12 += d_s12[j];
                if LINEAR {
                    slot.t1 += d_t1[j];
                    slot.t2 += d_t2[j];
                }
            }

            let b = &mut state.precomputed_statistics.binned_prefix_sums[col];
            let mut acc_d = PrefixStats::ZERO;
            for off in 0..bin_span {
                add_statistics::<LINEAR>(&mut acc_d, bucket_d[off]);
                add_statistics::<LINEAR>(&mut b[min_bin + off], acc_d);
            }
            if let Some(tail) = b.get_mut((max_bin + 1)..) {
                for v in tail {
                    add_statistics::<LINEAR>(v, acc_d);
                }
            }
        } else {
            let prefix_sums = &mut state.precomputed_statistics.prefix_sums[col];

            // Exact split column: rebuild the interval [start..end) and shift the tail.
            // Each per-stat prefix is rebuilt by accumulating the matching
            // c_*_new vector — same order of additions as before.
            let min_pos = start;
            let max_pos = end - 1;
            let prev = if min_pos > 0 {
                prefix_sums[min_pos - 1]
            } else {
                PrefixStats::ZERO
            };
            let old_last = prefix_sums[max_pos];

            let mut acc = prev;
            for (k, v) in c_s11_new.iter().enumerate() {
                acc.s11 += *v;
                prefix_sums[min_pos + k].s11 = acc.s11;
            }
            acc.s22 = prev.s22;
            for (k, v) in c_s22_new.iter().enumerate() {
                acc.s22 += *v;
                prefix_sums[min_pos + k].s22 = acc.s22;
            }
            acc.s12 = prev.s12;
            for (k, v) in c_s12_new.iter().enumerate() {
                acc.s12 += *v;
                prefix_sums[min_pos + k].s12 = acc.s12;
            }
            acc.t1 = prev.t1;
            for (k, v) in c_t1_new.iter().enumerate() {
                acc.t1 += *v;
                prefix_sums[min_pos + k].t1 = acc.t1;
            }
            acc.t2 = prev.t2;
            for (k, v) in c_t2_new.iter().enumerate() {
                acc.t2 += *v;
                prefix_sums[min_pos + k].t2 = acc.t2;
            }

            let diff = if LINEAR {
                sub_stats(prefix_sums[max_pos], old_last)
            } else {
                PrefixStats {
                    s11: prefix_sums[max_pos].s11 - old_last.s11,
                    s22: prefix_sums[max_pos].s22 - old_last.s22,
                    s12: prefix_sums[max_pos].s12 - old_last.s12,
                    t1: 0.0,
                    t2: 0.0,
                }
            };
            if let Some(tail) = prefix_sums.get_mut((max_pos + 1)..) {
                for v in tail {
                    add_statistics::<LINEAR>(v, diff);
                }
            }
        }

        // 4) Store the per-point curvature and linear caches
        {
            let curvature = &mut state.precomputed_statistics.curvature;
            let c_t1 = &mut state.precomputed_statistics.c_t1;
            let c_t2 = &mut state.precomputed_statistics.c_t2;
            for (j, &pt) in updated_points.iter().enumerate() {
                curvature[pt].s11 = c_s11_new[j];
                curvature[pt].s22 = c_s22_new[j];
                curvature[pt].s12 = c_s12_new[j];
                if LINEAR {
                    c_t1[pt] = c_t1_new[j];
                    c_t2[pt] = c_t2_new[j];
                }
            }
        }

        state.linear_statistics_stale = !LINEAR;
        state
    }
}
