//! Interval statistics and the score-cache refresh over the ranges an action touched.

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
        if state.split_scores_stale
            && (!state.sampled_split_scoring || state.is_stage1_positive_only())
        {
            self.refresh_all_split_scores(&mut state);
        }
        for affected_range in affected_ranges {
            let c = affected_range.col;
            let (lo, hi) = affected_range.point_range;

            // Update split error reductions using point range
            self.update_error_reductions_split_for_col_range(&mut state, c, lo, hi);

            // Update interval stats and boundary caches using boundary interval range
            // Note: interval_range in AffectedRange is for allowed_intervals,
            // but we need boundary indices for cache refreshing, so compute from point range
            let n_splits = state.boundaries[c].len();
            if n_splits > 0 {
                // Compute boundary interval indices from point range
                let (blo, bhi) = state.compute_boundary_index_range(c, lo, hi);

                // Update interval stats for affected intervals (O(1) lookup for resplit/merge)
                Self::update_interval_stats_for_col_range(&mut state, c, blo, bhi);

                // Convert interval range to boundary range
                // Intervals [blo, bhi] affect boundaries [blo-1, bhi] (clamped to valid boundaries)
                let n_boundaries = state.boundaries[c].len();
                if n_boundaries > 0 {
                    let boundary_lo = blo.saturating_sub(1);
                    let boundary_hi = bhi.min(n_boundaries - 1);

                    self.update_error_reductions_resplit_for_col_range(
                        &mut state,
                        c,
                        boundary_lo,
                        boundary_hi,
                    );
                    // Only update merge error reductions if merge is enabled
                    if state.split_strategy_state.merge_enabled {
                        self.update_error_reductions_merge_for_col_range(
                            &mut state,
                            c,
                            boundary_lo,
                            boundary_hi,
                        );
                    }
                }
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
        let n = state.x.nrows();
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
