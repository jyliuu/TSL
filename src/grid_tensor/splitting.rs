//! Split strategies: allowed-interval setup, unified action selection and post-split bookkeeping.

use std::cmp::Ordering;

use rand::Rng;

use crate::grid_tensor::{
    action::FittingAction, refinement::RefinementStrategy, state::FittingState,
};

mod boundary_candidates;
mod candidates;
mod sampling;
#[cfg(test)]
mod tests;

pub use candidates::{
    Interval, MergeCandidate, ResplitCandidate, SplitCandidate, SplitStrategyState, SplitType,
};
use sampling::{
    best_scored_random_position, initialize_intervals, sample_best_split, sample_random_split,
    sample_random_split_positions, sample_top_k_split,
};

const ACTION_SCORE_RELATIVE_TOLERANCE: f64 = 1e-12;

/// Enum dispatch for all split strategy types (matching actual implementation)
#[derive(Debug, Clone)]
pub enum SplitStrategy {
    Random {
        split_try: usize,
        colsample_bytree: f64,
        min_interval_samples: usize,
        /// Cost-complexity strength for every structural action.
        complexity_penalty: f64,
        min_split_loss: f64,
    },
    Best {
        min_interval_samples: usize,
        complexity_penalty: f64,
        min_split_loss: f64,
    },
    TopK {
        top_k: usize,
        must_fill_all_k: bool,
        min_interval_samples: usize,
        complexity_penalty: f64,
        min_split_loss: f64,
    },
}

impl SplitStrategy {
    pub fn initialize<'a>(&self, mut state: FittingState<'a>) -> FittingState<'a> {
        if !state.precomputed_statistics.initialized {
            panic!("Statistics are not initialized yet!")
        }
        let min_interval_samples = self.min_interval_samples();
        let error_reductions = &state.precomputed_statistics.error_reductions_split;

        let (allowed_intervals, total_positions) =
            initialize_intervals(error_reductions, min_interval_samples);

        state.split_strategy_state.allowed_intervals = allowed_intervals;
        state.split_strategy_state.total_positions = total_positions;
        let complexity_penalty = self.complexity_penalty();
        state.split_strategy_state.merge_enabled =
            complexity_penalty.is_finite() && complexity_penalty > 0.0;
        state.split_strategy_state.boundary_penalty = if state.split_strategy_state.merge_enabled
            && state.n > 0
            && state.current_error.is_finite()
            && state.current_error >= 0.0
        {
            let loss_scale = state.current_error / state.n as f64;
            let degrees_of_freedom = if state.is_stage1_positive_only() {
                1.0
            } else {
                2.0
            };
            complexity_penalty * loss_scale * degrees_of_freedom * (state.n as f64).max(2.0).ln()
        } else {
            0.0
        };

        state
    }

    /// Propose the best sampled action under the unified cost-complexity objective.
    pub fn propose_next_action<R: Rng + ?Sized>(
        &self,
        state: &FittingState,
        rng: &mut R,
    ) -> Option<FittingAction> {
        self.propose_next_action_with_split_budget(state, rng, true)
    }

    /// Propose one action, optionally excluding splits when the fineness budget is full.
    pub fn propose_next_action_with_split_budget<R: Rng + ?Sized>(
        &self,
        state: &FittingState,
        rng: &mut R,
        allow_split: bool,
    ) -> Option<FittingAction> {
        state.require_current_statistics();
        let best_split_candidate = allow_split
            .then(|| self.propose_best_split(state, rng))
            .flatten();
        self.action_from_candidates(state, best_split_candidate)
    }

    pub(crate) fn propose_next_action_with_sampled_scores<R: Rng + ?Sized>(
        &self,
        state: &mut FittingState,
        refinement: &RefinementStrategy,
        rng: &mut R,
        allow_split: bool,
    ) -> Option<FittingAction> {
        state.require_current_statistics();
        let Self::Random {
            colsample_bytree,
            split_try,
            ..
        } = self
        else {
            if state.split_scores_stale {
                refinement.refresh_all_split_scores(state);
            }
            return self.propose_next_action_with_split_budget(state, rng, allow_split);
        };
        if !state.sampled_split_scoring
            || !state.split_scores_stale
            || state.is_stage1_positive_only()
        {
            if state.split_scores_stale {
                refinement.refresh_all_split_scores(state);
            }
            return self.propose_next_action_with_split_budget(state, rng, allow_split);
        }
        let best_split = if allow_split {
            let positions = sample_random_split_positions(
                &state.split_strategy_state.allowed_intervals,
                &state.split_strategy_state.total_positions,
                *colsample_bytree,
                *split_try,
                rng,
                state.p,
            );
            best_scored_random_position(positions, |col, index| {
                refinement.refresh_sampled_split(state, col, index);
                state.precomputed_statistics.error_reductions_split[col]
                    .get(index)
                    .copied()
            })
            .map(|(col, index, interval_idx, gain)| SplitCandidate {
                col,
                index,
                allowed_interval_idx: interval_idx,
                error_reduction: gain,
                update_left: state.precomputed_statistics.update_pairs_split_left[col][index],
                update_right: state.precomputed_statistics.update_pairs_split_right[col][index],
            })
        } else {
            None
        };
        self.action_from_candidates(state, best_split)
    }

    fn action_from_candidates(
        &self,
        state: &FittingState,
        best_split_candidate: Option<SplitCandidate>,
    ) -> Option<FittingAction> {
        let best_resplit_candidate = self.propose_best_resplit(state);
        let best_merge_candidate = self.propose_best_merge(state);

        let split_action = best_split_candidate.map(|candidate| {
            (
                candidate.error_reduction - state.split_strategy_state.boundary_penalty,
                FittingAction::ApplySplit { split: candidate },
            )
        });

        let resplit_action = best_resplit_candidate.map(|candidate| {
            (
                candidate.error_reduction,
                FittingAction::ApplyResplit { resplit: candidate },
            )
        });

        let merge_action = best_merge_candidate.map(|candidate| {
            (
                candidate.error_reduction + state.split_strategy_state.boundary_penalty,
                FittingAction::ApplyMerge { merge: candidate },
            )
        });

        let tolerance = ACTION_SCORE_RELATIVE_TOLERANCE * state.current_error.abs().max(1.0);
        let minimum_gain = self.min_split_loss().max(tolerance);

        [split_action, resplit_action, merge_action]
            .into_iter()
            .flatten()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
            .filter(|(score, _)| score.is_finite() && *score > minimum_gain)
            .map(|(_, action)| action)
    }

    /// Propose the highest-scoring merge under the same action threshold.
    pub fn propose_next_merge(&self, state: &FittingState) -> Option<FittingAction> {
        state.require_current_statistics();
        let merge = self.propose_best_merge(state)?;
        let score = merge.error_reduction + state.split_strategy_state.boundary_penalty;
        let tolerance = self
            .min_split_loss()
            .max(ACTION_SCORE_RELATIVE_TOLERANCE * state.current_error.abs().max(1.0));
        (score.is_finite() && score > tolerance).then_some(FittingAction::ApplyMerge { merge })
    }

    /// Forbids splits around the given position by updating only the specific interval that contained the split
    pub fn forbid_around_split<'a>(
        &self,
        mut state: FittingState<'a>,
        split_candidate: &SplitCandidate,
    ) -> FittingState<'a> {
        let margin = self.min_interval_samples();
        let position = split_candidate.index;
        let interval_idx = split_candidate.allowed_interval_idx;
        let col = split_candidate.col;
        let allowed_intervals = &mut state.split_strategy_state.allowed_intervals;
        let total_positions = &mut state.split_strategy_state.total_positions;
        let forbidden_start = position.saturating_sub(margin - 1);
        let forbidden_end = position + margin;
        let forbidden_interval = Interval::new(forbidden_start, forbidden_end - forbidden_start);

        // Only update the specific interval that contained the split
        let original_interval = &allowed_intervals[col][interval_idx];
        let new_intervals = original_interval.subtract(&forbidden_interval);

        // Remove empty intervals
        let filtered_intervals: Vec<Interval> = new_intervals
            .into_iter()
            .filter(|interval| interval.length > 0)
            .collect();

        // Calculate the change in total positions
        let original_length = original_interval.length;
        let new_total: usize = filtered_intervals
            .iter()
            .map(|interval| interval.length)
            .sum();
        let change = new_total as i32 - original_length as i32;

        // Replace the original interval with the new ones
        if filtered_intervals.is_empty() {
            // If no intervals remain, remove this interval entirely
            allowed_intervals[col].remove(interval_idx);
        } else {
            // Replace the original interval with the new ones
            allowed_intervals[col].remove(interval_idx);
            allowed_intervals[col].extend(filtered_intervals);

            // Re-sort to maintain the sorted order
            allowed_intervals[col].sort_by_key(|interval| interval.start);
        }

        // Update the cached total positions
        total_positions[col] = (total_positions[col] as i32 + change) as usize;

        state
    }

    pub fn min_interval_samples(&self) -> usize {
        match self {
            SplitStrategy::Random {
                min_interval_samples,
                ..
            } => *min_interval_samples,
            SplitStrategy::Best {
                min_interval_samples,
                ..
            } => *min_interval_samples,
            SplitStrategy::TopK {
                min_interval_samples,
                ..
            } => *min_interval_samples,
        }
    }

    /// Return the cost-complexity strength used by every structural action.
    ///
    /// Positive finite values assign a fixed-scale cost to each boundary. Splits
    /// pay that cost, resplits leave it unchanged, and merges recover it.
    pub fn complexity_penalty(&self) -> f64 {
        match self {
            SplitStrategy::Random {
                complexity_penalty, ..
            } => *complexity_penalty,
            SplitStrategy::Best {
                complexity_penalty, ..
            } => *complexity_penalty,
            SplitStrategy::TopK {
                complexity_penalty, ..
            } => *complexity_penalty,
        }
    }

    pub fn min_split_loss(&self) -> f64 {
        match self {
            SplitStrategy::Random { min_split_loss, .. } => *min_split_loss,
            SplitStrategy::Best { min_split_loss, .. } => *min_split_loss,
            SplitStrategy::TopK { min_split_loss, .. } => *min_split_loss,
        }
    }

    fn propose_best_split<R: Rng + ?Sized>(
        &self,
        state: &FittingState,
        rng: &mut R,
    ) -> Option<SplitCandidate> {
        let best_split_info = match self {
            SplitStrategy::Random {
                colsample_bytree,
                split_try,
                ..
            } => {
                // Random strategy implementation - uses state.strategy_state for allowed intervals
                sample_random_split(
                    &state.split_strategy_state.allowed_intervals,
                    &state.split_strategy_state.total_positions,
                    *colsample_bytree,
                    *split_try,
                    rng,
                    &state.precomputed_statistics.error_reductions_split,
                )
            }
            SplitStrategy::Best { .. } => {
                // Best strategy implementation - uses state.working_buffers for error reductions
                sample_best_split(
                    &state.split_strategy_state.allowed_intervals,
                    &state.precomputed_statistics.error_reductions_split,
                )
            }
            SplitStrategy::TopK {
                top_k,
                must_fill_all_k,
                ..
            } => {
                // Top-k strategy implementation
                sample_top_k_split(
                    &state.split_strategy_state.allowed_intervals,
                    *top_k,
                    *must_fill_all_k,
                    rng,
                    &state.precomputed_statistics.error_reductions_split,
                )
            }
        };

        if let Some((col, index, interval_idx, err_reduction)) = best_split_info {
            let update_left = state.precomputed_statistics.update_pairs_split_left[col][index];
            let update_right = state.precomputed_statistics.update_pairs_split_right[col][index];
            Some(SplitCandidate {
                col,
                index,
                error_reduction: err_reduction,
                allowed_interval_idx: interval_idx,
                update_left,
                update_right,
            })
        } else {
            None
        }
    }
}
