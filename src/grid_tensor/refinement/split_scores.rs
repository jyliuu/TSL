//! Split candidate scores and updates for every position of a column's intervals.

use std::iter::once;

use super::gain::{solver_stats, SplitEvaluator};
use super::prefix::binned_prefix_stats_at;
use super::RefinementStrategy;
use crate::grid_tensor::state::{FittingState, PrecomputedStatistics, PrefixStats};

/// A column's prefix statistics: exact (one entry per sorted position) or bin-cumulative.
struct ColumnPrefix<'s> {
    exact: &'s [PrefixStats],
    binned: &'s [PrefixStats],
    bin_edges: &'s [usize],
}

impl ColumnPrefix<'_> {
    /// Prefix statistics before sorted position `start` and through `end - 1`.
    fn interval_bounds(&self, start: usize, end: usize) -> (PrefixStats, PrefixStats) {
        if self.bin_edges.is_empty() {
            let base = if start == 0 {
                PrefixStats::ZERO
            } else {
                self.exact[start - 1]
            };
            (base, self.exact[end - 1])
        } else {
            let base = if start == 0 {
                PrefixStats::ZERO
            } else {
                binned_prefix_stats_at(self.binned, self.bin_edges, start - 1)
            };
            (
                base,
                binned_prefix_stats_at(self.binned, self.bin_edges, end - 1),
            )
        }
    }
}

/// The split-score caches of one column, indexed by sorted position.
pub(super) struct SplitCaches<'s> {
    update_left: &'s mut [(f64, f64)],
    update_right: &'s mut [(f64, f64)],
    gains: &'s mut [f64],
    gain_pairs: &'s mut [(f64, f64)],
}

impl<'s> SplitCaches<'s> {
    pub(super) fn of(statistics: &'s mut PrecomputedStatistics, col: usize) -> Self {
        split_column(statistics, col).1
    }

    /// Mark `pos` as an ineligible split position.
    pub(super) fn mask(&mut self, pos: usize) {
        self.update_left[pos] = (f64::NAN, f64::NAN);
        self.update_right[pos] = (f64::NAN, f64::NAN);
        self.gains[pos] = f64::NAN;
        self.gain_pairs[pos] = (f64::NAN, f64::NAN);
    }

    fn store(&mut self, pos: usize, (left, right, gains): ((f64, f64), (f64, f64), (f64, f64))) {
        self.update_left[pos] = left;
        self.update_right[pos] = right;
        self.gain_pairs[pos] = gains;
        self.gains[pos] = gains.0 + gains.1;
    }
}

/// Borrow a column's prefix statistics alongside its split-score caches.
fn split_column(
    statistics: &mut PrecomputedStatistics,
    col: usize,
) -> (ColumnPrefix<'_>, SplitCaches<'_>) {
    let PrecomputedStatistics {
        prefix_sums,
        binned_prefix_sums,
        bin_edges,
        update_pairs_split_left,
        update_pairs_split_right,
        error_reductions_split,
        error_reductions_split_pairs,
        ..
    } = statistics;
    (
        ColumnPrefix {
            exact: &prefix_sums[col],
            binned: &binned_prefix_sums[col],
            bin_edges: &bin_edges[col],
        },
        SplitCaches {
            update_left: &mut update_pairs_split_left[col],
            update_right: &mut update_pairs_split_right[col],
            gains: &mut error_reductions_split[col],
            gain_pairs: &mut error_reductions_split_pairs[col],
        },
    )
}

impl RefinementStrategy {
    pub(super) fn update_error_reductions_split_for_all_cols<'a>(
        &self,
        mut state: FittingState<'a>,
    ) -> FittingState<'a> {
        let evaluator = SplitEvaluator::new(self, state.is_stage1_positive_only());
        let (n, sampled) = (state.n, state.sampled_split_scoring);
        for col in 0..state.p {
            Self::update_error_reductions_split_for_col_range(
                &mut state, &evaluator, col, 0, n, sampled,
            );
        }
        state
    }

    /// Rescore every interval that intersects sorted positions `lo..=hi` of `col`; with
    /// `sampled` scoring, mask the intervals' endpoints for on-demand rescoring instead.
    pub(super) fn update_error_reductions_split_for_col_range(
        state: &mut FittingState,
        evaluator: &SplitEvaluator,
        col: usize,
        lo: usize,
        hi: usize,
        sampled: bool,
    ) {
        let n = state.n;
        let mut start = 0usize;
        for &end in state.boundaries[col].iter().chain(once(&n)) {
            if end > lo && sampled && !evaluator.is_stage1() {
                // Sampled scoring rescores positions on demand. solve_two_tensor returns
                // finite gains, so summing the two sides never creates a NaN mask, and every
                // mask persists, including at endpoints of intervals a merge has joined.
                // Positive-only scoring uses the eager path.
                if end.saturating_sub(start) >= 2 {
                    let mut caches = SplitCaches::of(&mut state.precomputed_statistics, col);
                    caches.mask(start);
                    caches.mask(end - 1);
                }
                state.split_scores_stale = true;
            } else if end > lo {
                let (prefix, mut caches) = split_column(&mut state.precomputed_statistics, col);
                score_interval(&prefix, &mut caches, evaluator, start, end);
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
        let evaluator = SplitEvaluator::new(self, state.is_stage1_positive_only());
        Self::refresh_all_split_scores_with(state, &evaluator);
    }

    pub(super) fn refresh_all_split_scores_with(
        state: &mut FittingState,
        evaluator: &SplitEvaluator,
    ) {
        for col in 0..state.p {
            Self::update_error_reductions_split_for_col_range(
                state, evaluator, col, 0, state.n, false,
            );
        }
        state.split_scores_stale = false;
    }

    /// Evaluate one sampled coordinate against its current structural interval.
    pub(crate) fn refresh_sampled_split(&self, state: &mut FittingState, col: usize, pos: usize) {
        state.require_current_statistics();
        let Some(gain) = state.precomputed_statistics.error_reductions_split[col].get(pos) else {
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
        let evaluator = SplitEvaluator::new(self, state.is_stage1_positive_only());
        let (prefix, mut caches) = split_column(&mut state.precomputed_statistics, col);
        let left = if prefix.bin_edges.is_empty() {
            prefix.exact[pos - 1]
        } else {
            let Ok(bin) = prefix.bin_edges.binary_search(&pos) else {
                return;
            };
            prefix.binned[bin]
        };
        let (base, total) = prefix.interval_bounds(start, end);
        caches.store(
            pos,
            evaluator.evaluate(solver_stats(left, base), solver_stats(total, base)),
        );
    }
}

/// Score every candidate position strictly inside the interval `[start, end)` and mask
/// both endpoints. Already-masked positions stay masked.
fn score_interval(
    prefix: &ColumnPrefix,
    caches: &mut SplitCaches,
    evaluator: &SplitEvaluator,
    start: usize,
    end: usize,
) {
    if end.saturating_sub(start) < 2 {
        return;
    }
    caches.mask(start);

    let (base, end_value) = prefix.interval_bounds(start, end);
    let total = solver_stats(end_value, base);
    let mut score = |pos: usize, prefix_at_left_end: PrefixStats| {
        if caches.gains[pos].is_nan() {
            caches.update_left[pos] = (f64::NAN, f64::NAN);
            caches.update_right[pos] = (f64::NAN, f64::NAN);
            caches.gain_pairs[pos] = (f64::NAN, f64::NAN);
        } else {
            caches.store(
                pos,
                evaluator.evaluate(solver_stats(prefix_at_left_end, base), total),
            );
        }
    };
    if prefix.bin_edges.is_empty() {
        for pos in (start + 1)..end.saturating_sub(1) {
            score(pos, prefix.exact[pos - 1]);
        }
    } else {
        // Splits land on bin edges, and the bin before edge `bin_edges[k]` ends at its
        // cumulant `binned[k]`.
        let lo_bin = prefix.bin_edges.partition_point(|&e| e <= start);
        let hi_bin = prefix
            .bin_edges
            .partition_point(|&e| e < end.saturating_sub(1));
        for k in lo_bin..hi_bin {
            score(prefix.bin_edges[k], prefix.binned[k]);
        }
    }

    caches.mask(end - 1);
}
