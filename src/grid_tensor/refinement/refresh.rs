//! Full rebuilds of the row caches and prefix sums after a residual or consensus change.

use super::RefinementStrategy;
use crate::grid_tensor::state::{AffectedRange, FittingState, PrecomputedStatistics, PrefixStats};

/// Binned columns scattered per row pass of the linear rebin.
const LINEAR_TILE_COLUMNS: usize = 4;

impl RefinementStrategy {
    /// Change only the peer reference while retaining current all-row data statistics.
    /// Binned, merge-free fits update anchor contributions in O(p(m + bins)) work.
    pub(crate) fn refresh_consensus_reference<'a>(
        &self,
        mut state: FittingState<'a>,
        reference: crate::grid_tensor::state::BranchConsensus,
    ) -> FittingState<'a> {
        use crate::grid_tensor::state::same_consensus_support;
        let sparse = !state.linear_statistics_stale
            && reference.anchors.is_some()
            && !state.split_strategy_state.merge_enabled
            && state
                .precomputed_statistics
                .bin_edges
                .iter()
                .all(|edges| !edges.is_empty())
            && state.precomputed_statistics.consensus_weight == reference.weight
            && same_consensus_support(
                &state.precomputed_statistics.consensus_anchors,
                &reference.anchors,
            )
            && state.branch_consensus.as_ref().is_some_and(|old| {
                old.weight == reference.weight
                    && same_consensus_support(&old.anchors, &reference.anchors)
            });
        if !sparse {
            state.branch_consensus = Some(reference);
            return self.refresh_statistics(state);
        }
        let anchors = reference.anchors.as_ref().unwrap().clone();
        state.branch_consensus = Some(reference);
        let mut deltas = Vec::with_capacity(anchors.rows.len());
        for &row in &anchors.rows {
            // Re-evaluate these row totals to bound drift from repeated reference updates.
            let point = state.point_statistics(row, self.weight(state.residuals[row]));
            let statistics = &mut state.precomputed_statistics;
            deltas.push((
                point.t1 - statistics.c_t1[row],
                point.t2 - statistics.c_t2[row],
            ));
            statistics.c_t1[row] = point.t1;
            statistics.c_t2[row] = point.t2;
        }
        let mut bins = Vec::new();
        for col in 0..state.p {
            let statistics = &mut state.precomputed_statistics;
            let prefix = &mut statistics.binned_prefix_sums[col];
            bins.clear();
            bins.resize(prefix.len(), (0.0, 0.0));
            for (&row, &(delta_t1, delta_t2)) in anchors.rows.iter().zip(&deltas) {
                let bin = statistics.bin_indices[col][row] as usize;
                bins[bin].0 += delta_t1;
                bins[bin].1 += delta_t2;
            }
            let (mut t1, mut t2) = (0.0, 0.0);
            for (slot, &(delta_t1, delta_t2)) in prefix.iter_mut().zip(&bins) {
                t1 += delta_t1;
                t2 += delta_t2;
                slot.t1 += t1;
                slot.t2 += t2;
            }
        }
        self.refresh_all_scores(state)
    }

    /// Recompute every column's interval statistics and every split, resplit and merge score.
    fn refresh_all_scores<'a>(&self, mut state: FittingState<'a>) -> FittingState<'a> {
        for col in 0..state.p {
            let last_interval = state.boundaries[col].len();
            Self::update_interval_stats_for_col_range(&mut state, col, 0, last_interval);
        }
        let ranges: Vec<AffectedRange> = (0..state.p)
            .map(|col| AffectedRange {
                col,
                point_range: (0, state.n - 1),
                interval_range: (0, 0),
            })
            .collect();
        self.refresh_error_reduction_caches_for_affected(state, &ranges)
    }

    /// Populate linear terms from contiguous full-row references with retained L2 curvature.
    fn refresh_full_row_linear_slices(state: &mut FittingState<'_>) -> bool {
        let Some(reference) = state.branch_consensus.as_ref() else {
            return false;
        };
        let n = state.n;
        if reference.anchors.is_some() || state.residuals.len() < n {
            return false;
        }
        let (Some(plus), Some(minus), Some(residual), Some(target_plus), Some(target_minus)) = (
            state.f_plus.as_slice(),
            state.f_minus.as_slice(),
            state.r_tilde.as_slice(),
            reference.plus.as_slice(),
            reference.minus.as_slice(),
        ) else {
            return false;
        };
        let statistics = &mut state.precomputed_statistics;
        if [
            plus.len(),
            minus.len(),
            residual.len(),
            target_plus.len(),
            target_minus.len(),
            statistics.c_t1.len(),
            statistics.c_t2.len(),
        ]
        .into_iter()
        .any(|len| len < n)
        {
            return false;
        }
        let (plus, minus, residual, target_plus, target_minus) = (
            &plus[..n],
            &minus[..n],
            &residual[..n],
            &target_plus[..n],
            &target_minus[..n],
        );
        let (c_t1, c_t2) = (&mut statistics.c_t1[..n], &mut statistics.c_t2[..n]);
        let rho = reference.weight;
        for row in 0..n {
            let (fp, fm, rt) = (plus[row], minus[row], residual[row]);
            let mut t1 = 1.0 * rt * fp;
            let mut t2 = 1.0 * rt * fm;
            t1 += rho * (target_plus[row] - fp) * fp;
            t2 += rho * (fm - target_minus[row]) * fm;
            c_t1[row] = t1;
            c_t2[row] = t2;
        }
        true
    }

    /// Scatter the linear row terms of a tile of binned columns in one row pass.
    ///
    /// Every bucket accumulates its rows in ascending row order from `+0.0`; columns are
    /// independent accumulators.
    fn scatter_linear_tile(
        maps: &[&[u16]],
        t1: &[f64],
        t2: &[f64],
        scratch: &mut [[f64; 2]],
        widths: &[usize],
    ) {
        match (maps, widths) {
            ([m0, m1, m2, m3], &[w0, w1, w2, _]) => {
                let (b0, rest) = scratch.split_at_mut(w0);
                let (b1, rest) = rest.split_at_mut(w1);
                let (b2, b3) = rest.split_at_mut(w2);
                for (((((&x0, &x1), &x2), &x3), &v1), &v2) in
                    m0.iter().zip(*m1).zip(*m2).zip(*m3).zip(t1).zip(t2)
                {
                    let s0 = &mut b0[x0 as usize];
                    s0[0] += v1;
                    s0[1] += v2;
                    let s1 = &mut b1[x1 as usize];
                    s1[0] += v1;
                    s1[1] += v2;
                    let s2 = &mut b2[x2 as usize];
                    s2[0] += v1;
                    s2[1] += v2;
                    let s3 = &mut b3[x3 as usize];
                    s3[0] += v1;
                    s3[1] += v2;
                }
            }
            ([m0, m1, m2], &[w0, w1, _]) => {
                let (b0, rest) = scratch.split_at_mut(w0);
                let (b1, b2) = rest.split_at_mut(w1);
                for ((((&x0, &x1), &x2), &v1), &v2) in m0.iter().zip(*m1).zip(*m2).zip(t1).zip(t2) {
                    let s0 = &mut b0[x0 as usize];
                    s0[0] += v1;
                    s0[1] += v2;
                    let s1 = &mut b1[x1 as usize];
                    s1[0] += v1;
                    s1[1] += v2;
                    let s2 = &mut b2[x2 as usize];
                    s2[0] += v1;
                    s2[1] += v2;
                }
            }
            ([m0, m1], &[w0, _]) => {
                let (b0, b1) = scratch.split_at_mut(w0);
                for (((&x0, &x1), &v1), &v2) in m0.iter().zip(*m1).zip(t1).zip(t2) {
                    let s0 = &mut b0[x0 as usize];
                    s0[0] += v1;
                    s0[1] += v2;
                    let s1 = &mut b1[x1 as usize];
                    s1[0] += v1;
                    s1[1] += v2;
                }
            }
            ([m0], &[_]) => {
                let b0 = scratch;
                for ((&x0, &v1), &v2) in m0.iter().zip(t1).zip(t2) {
                    let s0 = &mut b0[x0 as usize];
                    s0[0] += v1;
                    s0[1] += v2;
                }
            }
            _ => unreachable!("tiles hold one to four columns"),
        }
    }

    /// Rebuild the binned linear prefix sums of every binned column, `LINEAR_TILE_COLUMNS`
    /// columns per row pass over the first `n` rows of the linear caches.
    fn rebin_linear_tiles(statistics: &mut PrecomputedStatistics, n: usize) {
        let PrecomputedStatistics {
            c_t1,
            c_t2,
            bin_indices,
            bin_edges,
            binned_prefix_sums,
            ..
        } = statistics;
        let (t1, t2) = (&c_t1[..n], &c_t2[..n]);
        let binned: Vec<usize> = (0..bin_edges.len())
            .filter(|&col| !bin_edges[col].is_empty())
            .collect();
        let mut scratch: Vec<[f64; 2]> = Vec::new();
        for tile in binned.chunks(LINEAR_TILE_COLUMNS) {
            let widths: Vec<usize> = tile
                .iter()
                .map(|&col| binned_prefix_sums[col].len())
                .collect();
            scratch.clear();
            scratch.resize(widths.iter().sum(), [0.0; 2]);
            let maps: Vec<&[u16]> = tile.iter().map(|&col| &bin_indices[col][..n]).collect();
            Self::scatter_linear_tile(&maps, t1, t2, &mut scratch, &widths);
            let mut offset = 0;
            for (&col, &width) in tile.iter().zip(&widths) {
                let bucket = &scratch[offset..offset + width];
                offset += width;
                let (mut t1, mut t2) = (0.0, 0.0);
                for (slot, b) in binned_prefix_sums[col].iter_mut().zip(bucket) {
                    t1 += b[0];
                    t2 += b[1];
                    slot.t1 = t1;
                    slot.t2 = t2;
                }
            }
        }
    }

    /// Rebuild every statistic that depends on the residuals or the branch consensus.
    ///
    /// The partition and factors are kept; per-point sufficient statistics, prefix sums,
    /// interval statistics and all split, resplit and merge scores are recomputed.
    pub(crate) fn refresh_statistics<'a>(&self, mut state: FittingState<'a>) -> FittingState<'a> {
        let n = state.n;
        // L2 weights are constant, so while the consensus weight is unchanged only the
        // reference-dependent t1/t2 sums move.
        let consensus_weight = state.branch_consensus.as_ref().map_or(0.0, |c| c.weight);
        let consensus_anchors = state
            .branch_consensus
            .as_ref()
            .and_then(|c| c.anchors.clone());
        let weights_fixed = matches!(self, RefinementStrategy::L2Refinement { .. })
            && state.precomputed_statistics.consensus_weight == consensus_weight
            && crate::grid_tensor::state::same_consensus_support(
                &state.precomputed_statistics.consensus_anchors,
                &consensus_anchors,
            );
        if weights_fixed {
            if !Self::refresh_full_row_linear_slices(&mut state) {
                for i in 0..n {
                    let point = state.point_statistics(i, self.weight(state.residuals[i]));
                    state.precomputed_statistics.c_t1[i] = point.t1;
                    state.precomputed_statistics.c_t2[i] = point.t2;
                }
            }
        } else {
            for i in 0..n {
                let point = state.point_statistics(i, self.weight(state.residuals[i]));
                let statistics = &mut state.precomputed_statistics;
                statistics.curvature[i].s11 = point.s11;
                statistics.curvature[i].s22 = point.s22;
                statistics.curvature[i].s12 = point.s12;
                statistics.c_t1[i] = point.t1;
                statistics.c_t2[i] = point.t2;
            }
        }
        let statistics = &mut state.precomputed_statistics;
        statistics.consensus_weight = consensus_weight;
        statistics.consensus_anchors = consensus_anchors;
        if weights_fixed {
            debug_assert!(
                statistics.c_t1.len() >= n
                    && statistics.c_t2.len() >= n
                    && (0..state.p).all(|col| {
                        statistics.bin_edges[col].is_empty()
                            || (statistics.bin_indices[col].len() == n
                                && statistics.binned_prefix_sums[col].len()
                                    == statistics.bin_edges[col].len() + 1)
                    }),
                "binned columns map every row and hold one cumulant per bin"
            );
            Self::rebin_linear_tiles(statistics, n);
        }
        for col in 0..state.p {
            if statistics.bin_edges[col].is_empty() {
                let prefix = &mut statistics.prefix_sums[col];
                let mut acc = PrefixStats::ZERO;
                for (slot, &i) in prefix.iter_mut().zip(&statistics.sorted_indices[col]) {
                    if !weights_fixed {
                        acc.s11 += statistics.curvature[i].s11;
                        acc.s22 += statistics.curvature[i].s22;
                        acc.s12 += statistics.curvature[i].s12;
                        slot.s11 = acc.s11;
                        slot.s22 = acc.s22;
                        slot.s12 = acc.s12;
                    }
                    acc.t1 += statistics.c_t1[i];
                    acc.t2 += statistics.c_t2[i];
                    slot.t1 = acc.t1;
                    slot.t2 = acc.t2;
                }
            } else if !weights_fixed {
                // With fixed weights the tiled pass has rebuilt the binned columns.
                let binned = &mut statistics.binned_prefix_sums[col];
                binned.fill(PrefixStats::ZERO);
                for (i, &bin) in statistics.bin_indices[col].iter().enumerate() {
                    binned[bin as usize] += PrefixStats {
                        s11: statistics.curvature[i].s11,
                        s22: statistics.curvature[i].s22,
                        s12: statistics.curvature[i].s12,
                        t1: statistics.c_t1[i],
                        t2: statistics.c_t2[i],
                    };
                }
                let mut acc = PrefixStats::ZERO;
                for bin in binned.iter_mut() {
                    acc += *bin;
                    *bin = acc;
                }
            }
        }

        state.linear_statistics_stale = false;
        self.refresh_all_scores(state)
    }
}
