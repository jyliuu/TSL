//! Per-action delta updates of the row caches and the exact and binned prefix sums.

use super::prefix::{add_statistics, sub_stats};
use super::RefinementStrategy;
use crate::grid_tensor::state::{CurvatureCache, FittingState, PrecomputedStatistics, PrefixStats};

/// The rows an action rescaled, sorted positions `start..end` of column `col`, with each
/// row's change in retained statistics in the same order.
struct AffectedRows<'d> {
    col: usize,
    start: usize,
    end: usize,
    n: usize,
    deltas: &'d [PrefixStats],
}

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

    /// Without `LINEAR`, only the curvature statistics are retained; the `t1`/`t2` terms of
    /// rows and prefix sums are left as they are.
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

        // 1) Each affected row's retained statistics and their change from the stored
        //    caches, so every column applies the same deltas.
        let (new_values, deltas): (Vec<PrefixStats>, Vec<PrefixStats>) = updated_points
            .iter()
            .map(|&pt| {
                let new = if LINEAR {
                    state.point_statistics(pt, self.weight(state.residuals[pt]))
                } else {
                    state.point_curvature_statistics(pt, self.weight(state.residuals[pt]))
                };
                let statistics = &state.precomputed_statistics;
                let old = statistics.curvature[pt];
                let delta = PrefixStats {
                    s11: new.s11 - old.s11,
                    s22: new.s22 - old.s22,
                    s12: new.s12 - old.s12,
                    t1: if LINEAR {
                        new.t1 - statistics.c_t1[pt]
                    } else {
                        0.0
                    },
                    t2: if LINEAR {
                        new.t2 - statistics.c_t2[pt]
                    } else {
                        0.0
                    },
                };
                (new, delta)
            })
            .unzip();

        let affected = AffectedRows {
            col,
            start,
            end,
            n: state.n,
            deltas: &deltas,
        };
        let statistics = &mut state.precomputed_statistics;

        // 2) Every other column scatters the deltas over the sorted positions its affected
        //    rows span: into length-B bin buckets for binned columns (O(B) work), into the
        //    length-n prefix sums for exact columns.
        let mut buckets = Vec::new();
        for (c, &positions) in affected_points_range
            .iter()
            .enumerate()
            .filter(|&(c, _)| c != col)
        {
            if positions.0 > positions.1 {
                continue;
            }
            scatter_column::<LINEAR>(statistics, &affected, c, positions, &mut buckets);
        }

        // 3) The split column spans the full interval [start, end). A binned split column
        //    scatters the deltas; an exact one rebuilds the interval's prefix sums from the
        //    new row values and shifts the tail.
        if !statistics.bin_edges[col].is_empty() {
            scatter_column::<LINEAR>(statistics, &affected, col, (start, end - 1), &mut buckets);
        } else {
            let prefix_sums = &mut statistics.prefix_sums[col];
            let (min_pos, max_pos) = (start, end - 1);
            let mut acc = if min_pos > 0 {
                prefix_sums[min_pos - 1]
            } else {
                PrefixStats::ZERO
            };
            let old_last = prefix_sums[max_pos];
            for (value, new) in prefix_sums[min_pos..=max_pos].iter_mut().zip(&new_values) {
                add_statistics::<LINEAR>(&mut acc, *new);
                value.s11 = acc.s11;
                value.s22 = acc.s22;
                value.s12 = acc.s12;
                if LINEAR {
                    value.t1 = acc.t1;
                    value.t2 = acc.t2;
                }
            }
            let diff = sub_stats(prefix_sums[max_pos], old_last);
            if let Some(tail) = prefix_sums.get_mut((max_pos + 1)..) {
                for value in tail {
                    add_statistics::<LINEAR>(value, diff);
                }
            }
        }

        // 4) Store the per-point curvature and linear caches.
        let updated_points = &statistics.sorted_indices[col][start..end];
        for (&pt, new) in updated_points.iter().zip(&new_values) {
            statistics.curvature[pt] = CurvatureCache {
                s11: new.s11,
                s22: new.s22,
                s12: new.s12,
            };
            if LINEAR {
                statistics.c_t1[pt] = new.t1;
                statistics.c_t2[pt] = new.t2;
            }
        }

        state.linear_statistics_stale = !LINEAR;
        state
    }
}

/// Add the affected rows' deltas to column `c`, whose affected rows span sorted positions
/// `positions.0..=positions.1`.
fn scatter_column<const LINEAR: bool>(
    statistics: &mut PrecomputedStatistics,
    affected: &AffectedRows,
    c: usize,
    (min_pos, max_pos): (usize, usize),
    buckets: &mut Vec<PrefixStats>,
) {
    let PrecomputedStatistics {
        sorted_indices,
        sort_order,
        bin_edges,
        bin_indices,
        bin_ranks,
        prefix_sums,
        binned_prefix_sums,
        ..
    } = statistics;
    let rows = &sorted_indices[affected.col][affected.start..affected.end];
    let edges = &bin_edges[c];
    if edges.is_empty() {
        let ranks = &sort_order[c];
        scatter_and_sweep::<LINEAR>(
            &mut prefix_sums[c],
            (min_pos, max_pos),
            rows.iter().map(|&row| ranks[row]),
            affected.deltas,
            buckets,
        );
        return;
    }
    let bin_span = (
        edges.partition_point(|&e| e <= min_pos),
        edges.partition_point(|&e| e <= max_pos),
    );
    let prefix = &mut binned_prefix_sums[c];
    match bin_ranks.get(affected.col) {
        // Column `c`'s bins of the affected rows, contiguous in the action column's rank order.
        Some(streams) => {
            let offset = c * affected.n;
            scatter_and_sweep::<LINEAR>(
                prefix,
                bin_span,
                streams[offset + affected.start..offset + affected.end]
                    .iter()
                    .map(|&bin| bin as usize),
                affected.deltas,
                buckets,
            );
        }
        None => {
            let row_bins = &bin_indices[c];
            scatter_and_sweep::<LINEAR>(
                prefix,
                bin_span,
                rows.iter().map(|&row| row_bins[row] as usize),
                affected.deltas,
                buckets,
            );
        }
    }
}

/// Add each delta to its slot of `prefix`, where every slot lies in `lo..=hi`, keeping
/// `prefix` a running sum: the slot totals accumulate over `lo..=hi`, and the tail past `hi`
/// receives the full total. `slots` yields one slot per delta, in delta order.
fn scatter_and_sweep<const LINEAR: bool>(
    prefix: &mut [PrefixStats],
    (lo, hi): (usize, usize),
    slots: impl Iterator<Item = usize>,
    deltas: &[PrefixStats],
    buckets: &mut Vec<PrefixStats>,
) {
    buckets.clear();
    buckets.resize(hi - lo + 1, PrefixStats::ZERO);
    for (slot, &delta) in slots.zip(deltas) {
        add_statistics::<LINEAR>(&mut buckets[slot - lo], delta);
    }
    let mut accumulated = PrefixStats::ZERO;
    for (value, &bucket) in prefix[lo..=hi].iter_mut().zip(buckets.iter()) {
        add_statistics::<LINEAR>(&mut accumulated, bucket);
        add_statistics::<LINEAR>(value, accumulated);
    }
    if let Some(tail) = prefix.get_mut((hi + 1)..) {
        for value in tail {
            add_statistics::<LINEAR>(value, accumulated);
        }
    }
}
