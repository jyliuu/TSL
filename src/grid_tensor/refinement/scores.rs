//! Interval statistics and the score-cache refresh over the ranges an action touched.

use super::gain::SplitEvaluator;
use super::prefix::{binned_prefix_stats_range, prefix_stats_range};
use super::RefinementStrategy;
use crate::grid_tensor::state::{AffectedRange, FittingState};

impl RefinementStrategy {
    pub fn refresh_error_reduction_caches_for_affected<'a>(
        &self,
        mut state: FittingState<'a>,
        affected_ranges: &[AffectedRange],
    ) -> FittingState<'a> {
        state.require_current_statistics();
        let evaluator = SplitEvaluator::new(self, state.is_stage1_positive_only());
        if state.split_scores_stale && (!state.sampled_split_scoring || evaluator.is_stage1()) {
            Self::refresh_all_split_scores_with(&mut state, &evaluator);
        }
        for affected_range in affected_ranges {
            let c = affected_range.col;
            let (lo, hi) = affected_range.point_range;

            Self::update_error_reductions_split_for_col_range(&mut state, &evaluator, c, lo, hi);

            // `interval_range` indexes allowed intervals; the boundary caches are indexed
            // from the point range instead.
            if state.boundaries[c].is_empty() {
                continue;
            }
            let (blo, bhi) = state.compute_boundary_index_range(c, lo, hi);
            Self::update_interval_stats_for_col_range(&mut state, c, blo, bhi);

            // Intervals [blo, bhi] border boundaries [blo - 1, bhi], clamped to valid boundaries.
            let boundary_lo = blo.saturating_sub(1);
            let boundary_hi = bhi.min(state.boundaries[c].len() - 1);
            Self::update_error_reductions_resplit_for_col_range(
                &mut state,
                &evaluator,
                c,
                boundary_lo,
                boundary_hi,
            );
            if state.split_strategy_state.merge_enabled {
                self.update_error_reductions_merge_for_col_range(
                    &mut state,
                    &evaluator,
                    c,
                    boundary_lo,
                    boundary_hi,
                );
            }
        }

        state
    }

    /// Recompute interval stats for affected intervals using prefix sums
    pub(super) fn update_interval_stats_for_col_range(
        state: &mut FittingState,
        col: usize,
        blo: usize,
        bhi: usize,
    ) {
        let boundaries = &state.boundaries[col];
        let n = state.n;
        let bin_edges = &state.precomputed_statistics.bin_edges[col];
        let is_binned = !bin_edges.is_empty();

        // Update stats for intervals [blo..=bhi+1] since a boundary affects two adjacent intervals
        let start_interval = blo;
        let end_interval = (bhi + 2).min(state.precomputed_statistics.interval_stats[col].len());

        for interval_idx in start_interval..end_interval {
            let start = if interval_idx == 0 {
                0
            } else {
                boundaries[interval_idx - 1]
            };
            let end = boundaries.get(interval_idx).copied().unwrap_or(n);
            let interval_n = end - start;

            let stats = if is_binned {
                // Binned: start and end are always bin-aligned (splits only
                // happen at bin edges in binned mode). Read directly from
                // length-B cumulants.
                let b = &state.precomputed_statistics.binned_prefix_sums[col];
                binned_prefix_stats_range(b, bin_edges, start, end)
            } else {
                let p = &state.precomputed_statistics.prefix_sums[col];
                prefix_stats_range(p, start, end)
            };

            state.precomputed_statistics.interval_stats[col][interval_idx] =
                crate::grid_tensor::state::IntervalStats {
                    sum_s11: stats.s11,
                    sum_s22: stats.s22,
                    sum_s12: stats.s12,
                    sum_t1: stats.t1,
                    sum_t2: stats.t2,
                    n: interval_n,
                };
        }
    }
}
