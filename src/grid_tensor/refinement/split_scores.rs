//! Split candidate scores and updates for every position of a column's intervals.

use std::iter::once;

use super::gain::{solver_stats, SplitEvaluator};
use super::prefix::binned_prefix_stats_at;
use super::RefinementStrategy;
use crate::grid_tensor::state::{FittingState, PrefixStats};

impl RefinementStrategy {
    pub(super) fn update_error_reductions_split_for_all_cols<'a>(
        &self,
        mut state: FittingState<'a>,
    ) -> FittingState<'a> {
        let n = state.x.nrows();
        for col in 0..state.boundaries.len() {
            self.update_error_reductions_split_for_col_range(&mut state, col, 0, n);
        }
        state
    }

    pub(super) fn update_error_reductions_split_for_col_range(
        &self,
        state: &mut FittingState,
        col: usize,
        lo: usize,
        hi: usize,
    ) {
        let n = state.x.nrows();
        let is_stage1 = state.is_stage1_positive_only();
        let mut start = 0usize;
        for &b in state.boundaries[col].iter().chain(once(&n)) {
            // Computes the error reductions for any interval that intersects with the updated points range

            let end = b;
            if end > lo && state.sampled_split_scoring && !is_stage1 {
                // solve_two_tensor returns finite gains, so summing the two sides cannot
                // create NaN eligibility masks. Positive-only scoring uses the eager path.
                // Preserve every historical mask, including endpoints of past intervals.
                if end.saturating_sub(start) >= 2 {
                    for pos in [start, end - 1] {
                        state.precomputed_statistics.error_reductions_split[col][pos] = f64::NAN;
                        state.precomputed_statistics.update_pairs_split_left[col][pos] =
                            (f64::NAN, f64::NAN);
                        state.precomputed_statistics.update_pairs_split_right[col][pos] =
                            (f64::NAN, f64::NAN);
                        state.precomputed_statistics.error_reductions_split_pairs[col][pos] =
                            (f64::NAN, f64::NAN);
                    }
                }
                state.split_scores_stale = true;
            } else if end > lo {
                self.update_error_reductions_split_single_interval(
                    &state.precomputed_statistics.prefix_sums[col],
                    &state.precomputed_statistics.binned_prefix_sums[col],
                    &mut state.precomputed_statistics.update_pairs_split_left[col],
                    &mut state.precomputed_statistics.update_pairs_split_right[col],
                    &mut state.precomputed_statistics.error_reductions_split[col],
                    &mut state.precomputed_statistics.error_reductions_split_pairs[col],
                    &state.precomputed_statistics.bin_edges[col],
                    (start, end),
                    is_stage1,
                );
            }
            start = end;
            if start > hi {
                break;
            }
        }
    }

    /// Rebuild split scores before an eager consumer or scoring-mode transition.
    pub(crate) fn refresh_all_split_scores(&self, state: &mut FittingState) {
        state.require_current_statistics();
        let sampled = state.sampled_split_scoring;
        state.sampled_split_scoring = false;
        for col in 0..state.p {
            self.update_error_reductions_split_for_col_range(state, col, 0, state.n);
        }
        state.sampled_split_scoring = sampled;
        state.split_scores_stale = false;
    }

    /// Evaluate one sampled coordinate against its current structural interval.
    pub(crate) fn refresh_sampled_split(&self, state: &mut FittingState, col: usize, pos: usize) {
        state.require_current_statistics();
        let statistics = &state.precomputed_statistics;
        let Some(gain) = statistics.error_reductions_split[col].get(pos) else {
            return;
        };
        if gain.is_nan() {
            return;
        }
        let interval = state.boundaries[col].partition_point(|&boundary| boundary <= pos);
        let (start, end) = state.interval_range(col, interval);
        if pos <= start || pos >= end.saturating_sub(1) {
            return;
        }
        let bin_edges = &statistics.bin_edges[col];
        let (base, left, total) = if bin_edges.is_empty() {
            let ps = &statistics.prefix_sums[col];
            (
                if start == 0 {
                    PrefixStats::ZERO
                } else {
                    ps[start - 1]
                },
                ps[pos - 1],
                ps[end - 1],
            )
        } else {
            let Ok(bin) = bin_edges.binary_search(&pos) else {
                return;
            };
            let bs = &statistics.binned_prefix_sums[col];
            (
                if start == 0 {
                    PrefixStats::ZERO
                } else {
                    binned_prefix_stats_at(bs, bin_edges, start - 1)
                },
                bs[bin],
                binned_prefix_stats_at(bs, bin_edges, end - 1),
            )
        };
        let (left, right, gains) = SplitEvaluator::new(self, state.is_stage1_positive_only())
            .evaluate(solver_stats(left, base), solver_stats(total, base));
        let statistics = &mut state.precomputed_statistics;
        statistics.update_pairs_split_left[col][pos] = left;
        statistics.update_pairs_split_right[col][pos] = right;
        statistics.error_reductions_split_pairs[col][pos] = gains;
        statistics.error_reductions_split[col][pos] = gains.0 + gains.1;
    }

    #[allow(clippy::too_many_arguments)]
    fn update_error_reductions_split_single_interval(
        &self,
        ps: &[PrefixStats],
        bs: &[PrefixStats],
        update_left: &mut [(f64, f64)],
        update_right: &mut [(f64, f64)],
        error_reductions: &mut [f64],
        error_reductions_pairs: &mut [(f64, f64)],
        bin_edges: &[usize],
        spanned_interval: (usize, usize),
        is_stage1: bool,
    ) {
        let (start, end) = spanned_interval;
        let len = end.saturating_sub(start);
        if len < 2 {
            return;
        }

        // Disallow splitting at the first position.
        update_left[start] = (f64::NAN, f64::NAN);
        update_right[start] = (f64::NAN, f64::NAN);
        error_reductions[start] = f64::NAN;
        error_reductions_pairs[start] = (f64::NAN, f64::NAN);

        let evaluator = SplitEvaluator::new(self, is_stage1);

        // Evaluate a single candidate position given left-side sufficient stats.
        // Closed over the output slices + total stats. Shared between the
        // exact and binned iteration paths.
        let eval_pos = |pos: usize,
                        s11_l: f64,
                        s22_l: f64,
                        s12_l: f64,
                        t1_l: f64,
                        t2_l: f64,
                        total_s11: f64,
                        total_s22: f64,
                        total_s12: f64,
                        total_t1: f64,
                        total_t2: f64,
                        update_left: &mut [(f64, f64)],
                        update_right: &mut [(f64, f64)],
                        error_reductions: &mut [f64],
                        error_reductions_pairs: &mut [(f64, f64)]| {
            if error_reductions[pos].is_nan() {
                update_left[pos] = (f64::NAN, f64::NAN);
                update_right[pos] = (f64::NAN, f64::NAN);
                error_reductions_pairs[pos] = (f64::NAN, f64::NAN);
                return;
            }

            let (left, right, gains) = evaluator.evaluate(
                PrefixStats {
                    s11: s11_l,
                    s22: s22_l,
                    s12: s12_l,
                    t1: t1_l,
                    t2: t2_l,
                },
                PrefixStats {
                    s11: total_s11,
                    s22: total_s22,
                    s12: total_s12,
                    t1: total_t1,
                    t2: total_t2,
                },
            );
            update_left[pos] = left;
            update_right[pos] = right;
            error_reductions[pos] = gains.0 + gains.1;
            error_reductions_pairs[pos] = gains;
        };

        if bin_edges.is_empty() {
            // Exact path: read length-n prefix sums at every candidate position.
            let base = if start == 0 {
                PrefixStats::ZERO
            } else {
                ps[start - 1]
            };

            let total_s11 = ps[end - 1].s11 - base.s11;
            let total_s22 = ps[end - 1].s22 - base.s22;
            let total_s12 = -(ps[end - 1].s12 - base.s12);
            let total_t1 = ps[end - 1].t1 - base.t1;
            let total_t2 = -(ps[end - 1].t2 - base.t2);

            for pos in (start + 1)..end.saturating_sub(1) {
                let pos_end = pos - 1;
                let s11_l = ps[pos_end].s11 - base.s11;
                let s22_l = ps[pos_end].s22 - base.s22;
                let s12_l = -(ps[pos_end].s12 - base.s12);
                let t1_l = ps[pos_end].t1 - base.t1;
                let t2_l = -(ps[pos_end].t2 - base.t2);
                eval_pos(
                    pos,
                    s11_l,
                    s22_l,
                    s12_l,
                    t1_l,
                    t2_l,
                    total_s11,
                    total_s22,
                    total_s12,
                    total_t1,
                    total_t2,
                    update_left,
                    update_right,
                    error_reductions,
                    error_reductions_pairs,
                );
            }
        } else {
            // Binned path: read length-B bin-cumulative prefix sums at bin
            // boundaries. Splits land at `bin_edges[k]` so `pos_end = pos - 1`
            // is the last position of bin k → cumulant at bin index k.
            let base = if start == 0 {
                PrefixStats::ZERO
            } else {
                binned_prefix_stats_at(bs, bin_edges, start - 1)
            };
            let end_v = binned_prefix_stats_at(bs, bin_edges, end - 1);

            let total_s11 = end_v.s11 - base.s11;
            let total_s22 = end_v.s22 - base.s22;
            let total_s12 = -(end_v.s12 - base.s12);
            let total_t1 = end_v.t1 - base.t1;
            let total_t2 = -(end_v.t2 - base.t2);

            let lo_idx = bin_edges.partition_point(|&e| e <= start);
            let hi_idx = bin_edges.partition_point(|&e| e < end.saturating_sub(1));
            for k in lo_idx..hi_idx {
                let pos = bin_edges[k];
                let s11_l = bs[k].s11 - base.s11;
                let s22_l = bs[k].s22 - base.s22;
                let s12_l = -(bs[k].s12 - base.s12);
                let t1_l = bs[k].t1 - base.t1;
                let t2_l = -(bs[k].t2 - base.t2);
                eval_pos(
                    pos,
                    s11_l,
                    s22_l,
                    s12_l,
                    t1_l,
                    t2_l,
                    total_s11,
                    total_s22,
                    total_s12,
                    total_t1,
                    total_t2,
                    update_left,
                    update_right,
                    error_reductions,
                    error_reductions_pairs,
                );
            }
        }

        // Disallow splitting at the last position.
        let last = end - 1;
        update_left[last] = (f64::NAN, f64::NAN);
        update_right[last] = (f64::NAN, f64::NAN);
        error_reductions[last] = f64::NAN;
        error_reductions_pairs[last] = (f64::NAN, f64::NAN);
    }
}
