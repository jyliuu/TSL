//! Split strategies: allowed-interval setup, unified action selection and post-split bookkeeping.

use std::cmp::Ordering;

use rand::Rng;

use crate::grid_tensor::{
    action::FittingAction, refinement::RefinementStrategy, state::FittingState,
};

mod boundary_candidates;
mod candidates;
mod sampling;

pub use candidates::{
    Interval, MergeCandidate, ResplitCandidate, SplitCandidate, SplitStrategyState,
};
use sampling::{
    best_scored_position, initialize_intervals, sample_best_split, sample_random_split,
    sample_random_split_positions, sample_top_k_split,
};

const ACTION_SCORE_RELATIVE_TOLERANCE: f64 = 1e-12;

/// How split positions are proposed (sampled at random, the best one, or one of the top k),
/// and the thresholds every structural action must clear.
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
        let sampled = match self {
            Self::Random {
                colsample_bytree,
                split_try,
                ..
            } if state.sampled_split_scoring
                && state.split_scores_stale
                && !state.is_stage1_positive_only() =>
            {
                Some((*colsample_bytree, *split_try))
            }
            _ => None,
        };
        let Some((colsample_bytree, split_try)) = sampled else {
            if state.split_scores_stale {
                refinement.refresh_all_split_scores(state);
            }
            return self.propose_next_action_with_split_budget(state, rng, allow_split);
        };
        let best_split = if allow_split {
            let positions = sample_random_split_positions(
                &state.split_strategy_state.allowed_intervals,
                &state.split_strategy_state.total_positions,
                colsample_bytree,
                split_try,
                rng,
                state.p,
            );
            best_scored_position(positions, |col, index| {
                refinement.refresh_sampled_split(state, col, index);
                state.precomputed_statistics.error_reductions_split[col]
                    .get(index)
                    .copied()
            })
            .map(|scored| split_candidate(state, scored))
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
                merge_score(state, &candidate),
                FittingAction::ApplyMerge { merge: candidate },
            )
        });

        let minimum_gain = self.minimum_action_gain(state);

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
        let score = merge_score(state, &merge);
        let minimum_gain = self.minimum_action_gain(state);
        (score.is_finite() && score > minimum_gain).then_some(FittingAction::ApplyMerge { merge })
    }

    /// Score an action must exceed: `min_split_loss`, and at least a tolerance relative to
    /// the current error.
    fn minimum_action_gain(&self, state: &FittingState) -> f64 {
        let tolerance = ACTION_SCORE_RELATIVE_TOLERANCE * state.current_error.abs().max(1.0);
        self.min_split_loss().max(tolerance)
    }

    /// Forbids splits within `min_interval_samples` of the split position by cutting that
    /// neighbourhood out of the allowed interval that contained the split.
    pub fn forbid_around_split<'a>(
        &self,
        mut state: FittingState<'a>,
        split_candidate: &SplitCandidate,
    ) -> FittingState<'a> {
        let margin = self.min_interval_samples();
        let position = split_candidate.index;
        let interval_idx = split_candidate.allowed_interval_idx;
        let col = split_candidate.col;
        let forbidden_start = position.saturating_sub(margin - 1);
        let forbidden_end = position + margin;
        let forbidden_interval = Interval::new(forbidden_start, forbidden_end - forbidden_start);

        let strategy_state = &mut state.split_strategy_state;
        let original_interval = strategy_state.allowed_intervals[col][interval_idx];
        let remaining: Vec<Interval> = original_interval
            .subtract(&forbidden_interval)
            .into_iter()
            .filter(|interval| interval.length > 0)
            .collect();
        let remaining_positions: usize = remaining.iter().map(|interval| interval.length).sum();

        // The remaining pieces lie inside the original interval in ascending order, so the
        // column's disjoint allowed intervals stay sorted by start.
        strategy_state.allowed_intervals[col].splice(interval_idx..=interval_idx, remaining);
        strategy_state.total_positions[col] =
            strategy_state.total_positions[col] + remaining_positions - original_interval.length;

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
            } => sample_random_split(
                &state.split_strategy_state.allowed_intervals,
                &state.split_strategy_state.total_positions,
                *colsample_bytree,
                *split_try,
                rng,
                &state.precomputed_statistics.error_reductions_split,
            ),
            SplitStrategy::Best { .. } => sample_best_split(
                &state.split_strategy_state.allowed_intervals,
                &state.precomputed_statistics.error_reductions_split,
            ),
            SplitStrategy::TopK {
                top_k,
                must_fill_all_k,
                ..
            } => sample_top_k_split(
                &state.split_strategy_state.allowed_intervals,
                *top_k,
                *must_fill_all_k,
                rng,
                &state.precomputed_statistics.error_reductions_split,
            ),
        };
        best_split_info.map(|scored| split_candidate(state, scored))
    }
}

/// Split candidate at a scored position `(col, index, allowed_interval_idx, error_reduction)`,
/// with the cached updates of its two sides.
fn split_candidate(
    state: &FittingState,
    (col, index, allowed_interval_idx, error_reduction): (usize, usize, usize, f64),
) -> SplitCandidate {
    SplitCandidate {
        col,
        index,
        error_reduction,
        allowed_interval_idx,
        update_left: state.precomputed_statistics.update_pairs_split_left[col][index],
        update_right: state.precomputed_statistics.update_pairs_split_right[col][index],
    }
}

/// A merge recovers the cost of the boundary it removes.
fn merge_score(state: &FittingState, merge: &MergeCandidate) -> f64 {
    merge.error_reduction + state.split_strategy_state.boundary_penalty
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use ndarray::{Array1, Array2};
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::grid_tensor::state::MAX_CONSECUTIVE_RESPLIT;

    const NAN: f64 = f64::NAN;

    /// A 20-row state with error 20 (loss scale 1) and the given split scores per column.
    fn scored_state(split_gains: Vec<Vec<f64>>) -> FittingState<'static> {
        let x: &'static Array2<f64> = Box::leak(Box::new(Array2::zeros((20, split_gains.len()))));
        let y: &'static Array1<f64> = Box::leak(Box::new(Array1::zeros(20)));
        let mut state = FittingState::new(x.view(), y.view());
        state.current_error = 20.0;
        state.precomputed_statistics.error_reductions_split = split_gains;
        state.precomputed_statistics.initialized = true;
        state
    }

    /// Adds one merge candidate at boundary 10 of column 0 with valid merged factors.
    fn with_merge(mut state: FittingState<'static>, gain: f64) -> FittingState<'static> {
        state.boundaries[0] = vec![10];
        state.precomputed_statistics.error_reductions_merge[0] = vec![gain];
        state.precomputed_statistics.update_pairs_merge[0] = vec![(1.0, 1.0)];
        state
    }

    fn best(min_interval_samples: usize, complexity_penalty: f64) -> SplitStrategy {
        SplitStrategy::Best {
            min_interval_samples,
            complexity_penalty,
            min_split_loss: 0.0,
        }
    }

    fn propose(strategy: &SplitStrategy, state: &FittingState) -> Option<FittingAction> {
        strategy.propose_next_action(state, &mut StdRng::seed_from_u64(42))
    }

    fn proposed_split(strategy: &SplitStrategy, state: &FittingState) -> SplitCandidate {
        match propose(strategy, state) {
            Some(FittingAction::ApplySplit { split }) => split,
            action => panic!("expected a split, got {action:?}"),
        }
    }

    /// A positive finite complexity penalty enables merges and fixes the boundary cost at
    /// `penalty · (initial error / n) · dof · ln n`, with one degree of freedom per boundary
    /// in the positive-only first stage and two once the negative branch is active.
    #[test]
    fn boundary_cost_is_fixed_at_initialization_for_positive_finite_penalties() {
        for penalty in [0.0, -1.0, NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let strategy = best(1, penalty);
            let state = with_merge(
                strategy.initialize(scored_state(vec![vec![NAN; 20]])),
                100.0,
            );
            assert!(!state.split_strategy_state.merge_enabled);
            assert_eq!(state.split_strategy_state.boundary_penalty, 0.0);
            assert!(strategy.propose_next_merge(&state).is_none());
        }

        let ln_n = 20.0_f64.ln();
        let strategy = best(1, 1.0);
        let mut state = with_merge(strategy.initialize(scored_state(vec![vec![NAN; 20]])), -2.0);
        assert!(state.split_strategy_state.merge_enabled);
        assert!((state.split_strategy_state.boundary_penalty - ln_n).abs() < 1e-12);

        // At the initial loss scale the merge scores −2 + ln 20 > 0 for any later error.
        state.current_error = 2.0;
        assert!(strategy.propose_next_merge(&state).is_some());

        let mut signed = scored_state(vec![vec![NAN; 20]]);
        signed.lambda_minus = 1.0;
        let signed = strategy.initialize(signed);
        assert!((signed.split_strategy_state.boundary_penalty - 2.0 * ln_n).abs() < 1e-12);
    }

    /// A split pays the boundary cost and a merge recovers it, so the selector ranks a split
    /// against its inverse merge by their penalized scores.
    #[test]
    fn split_pays_and_merge_recovers_the_boundary_cost() {
        let strategy = best(1, 1.0);
        let mut split_gains = vec![NAN; 20];
        split_gains[5] = 4.0;
        let mut state = with_merge(strategy.initialize(scored_state(vec![split_gains])), -4.0);

        // With ln 20 ≈ 3, the split scores 4 − 3 = 1 and the merge −4 + 3 = −1.
        assert!(matches!(
            propose(&strategy, &state),
            Some(FittingAction::ApplySplit { .. })
        ));

        // The split scores 2 − 3 = −1 and the merge −2 + 3 = 1.
        state.precomputed_statistics.error_reductions_split[0][5] = 2.0;
        state.precomputed_statistics.error_reductions_merge[0][0] = -2.0;
        assert!(matches!(
            propose(&strategy, &state),
            Some(FittingAction::ApplyMerge { .. })
        ));
    }

    /// Merges with non-finite or non-positive merged factors are skipped, and the best valid
    /// merge is proposed only when its penalized score is positive.
    #[test]
    fn merge_proposals_skip_invalid_factors_and_non_improving_scores() {
        let strategy = best(1, 1.0);
        let mut state = strategy.initialize(scored_state(vec![vec![NAN; 20]]));
        state.boundaries[0] = vec![1, 10, 15, 17];
        state.precomputed_statistics.error_reductions_merge[0] = vec![-5.0, 1.0, 100.0, 50.0];
        state.precomputed_statistics.update_pairs_merge[0] =
            vec![(2.0, 2.0), (4.0, 4.0), (NAN, 7.0), (0.0, 3.0)];

        let Some(FittingAction::ApplyMerge { merge }) = strategy.propose_next_merge(&state) else {
            panic!("expected a merge");
        };
        assert_eq!((merge.col, merge.interval_idx, merge.index), (0, 1, 10));
        assert_eq!(merge.error_reduction, 1.0);

        // With the second merge at −6, the best valid merge scores −5 + ln 20 < 0.
        state.precomputed_statistics.error_reductions_merge[0][1] = -6.0;
        assert!(strategy.propose_next_merge(&state).is_none());
    }

    /// The distinct split positions TopK proposes over repeated draws.
    fn top_k_draws(split_gains: Vec<f64>, top_k: usize, must_fill_all_k: bool) -> BTreeSet<usize> {
        let strategy = SplitStrategy::TopK {
            top_k,
            must_fill_all_k,
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss: 0.0,
        };
        let state = strategy.initialize(scored_state(vec![split_gains]));
        let mut rng = StdRng::seed_from_u64(42);
        (0..64)
            .filter_map(|_| strategy.propose_next_action(&state, &mut rng))
            .map(|action| match action {
                FittingAction::ApplySplit { split } => split.index,
                action => panic!("expected a split, got {action:?}"),
            })
            .collect()
    }

    #[test]
    fn top_k_draws_among_the_k_best_scored_positions() {
        // Unscored positions are skipped; the two best are 5 at index 2 and 4 at index 4.
        assert_eq!(
            top_k_draws(vec![NAN, 1.0, 5.0, NAN, 4.0, 3.0], 2, false),
            BTreeSet::from([2, 4])
        );

        // With fewer than k candidates, all of them are drawn unless all k must be filled.
        let three = vec![NAN, 1.0, 2.0, 3.0, NAN];
        assert_eq!(
            top_k_draws(three.clone(), 5, false),
            BTreeSet::from([1, 2, 3])
        );
        assert_eq!(
            top_k_draws(three.clone(), 3, true),
            BTreeSet::from([1, 2, 3])
        );
        assert!(top_k_draws(three, 5, true).is_empty());
        assert!(top_k_draws(vec![NAN; 3], 2, false).is_empty());
    }

    #[test]
    fn random_samples_at_most_split_try_allowed_positions_per_column() {
        // Unscored positions and the `min_interval_samples` margin at either end are excluded.
        let gains = vec![
            vec![NAN, 1.0, 2.0, 3.0, 4.0, 5.0, NAN, NAN, 8.0, 9.0, 10.0],
            vec![NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ];
        let (allowed, totals) = initialize_intervals(&gains, 2);
        assert_eq!(
            allowed,
            vec![
                vec![Interval::new(2, 4), Interval::new(8, 2)],
                vec![Interval::new(2, 7)]
            ]
        );
        assert_eq!(totals, vec![6, 7]);

        let mut rng = StdRng::seed_from_u64(42);
        for split_try in [1, 3, 10] {
            let positions =
                sample_random_split_positions(&allowed, &totals, 1.0, split_try, &mut rng, 2);
            assert!(!positions.is_empty());
            for (col, intervals) in allowed.iter().enumerate() {
                let sampled: Vec<usize> = positions
                    .iter()
                    .filter(|&&(c, _, _)| c == col)
                    .map(|&(_, index, interval_idx)| {
                        assert!(intervals[interval_idx].contains(index));
                        index
                    })
                    .collect();
                assert!(sampled.len() <= split_try);
                assert_eq!(sampled.iter().collect::<BTreeSet<_>>().len(), sampled.len());
            }
        }
    }

    #[test]
    fn random_proposes_the_best_sampled_split_above_min_split_loss() {
        let random = |min_split_loss| SplitStrategy::Random {
            split_try: 10,
            colsample_bytree: 1.0,
            min_interval_samples: 1,
            complexity_penalty: 0.0,
            min_split_loss,
        };
        let gains = vec![vec![NAN, 0.1, 0.2, NAN, 0.4, 0.5, NAN, 0.3]];

        let strategy = random(0.0);
        let state = strategy.initialize(scored_state(gains.clone()));
        assert_eq!(proposed_split(&strategy, &state).index, 5);

        let strategy = random(0.5);
        let state = strategy.initialize(scored_state(gains));
        assert!(propose(&strategy, &state).is_none());
    }

    #[test]
    fn best_proposes_the_highest_scored_split_outside_forbidden_regions() {
        let strategy = best(2, 0.0);
        let gains = vec![NAN, NAN, 1.0, 4.0, 5.0, 4.0, NAN, 3.0, 2.0, NAN];
        let mut state = strategy.initialize(scored_state(vec![gains]));

        let split = proposed_split(&strategy, &state);
        assert_eq!((split.col, split.index, split.error_reduction), (0, 4, 5.0));

        // A margin of 2 forbids positions 3..6, including both runner-up neighbours.
        state = strategy.forbid_around_split(state, &split);
        assert_eq!(state.split_strategy_state.total_positions, vec![3]);
        let split = proposed_split(&strategy, &state);
        assert_eq!((split.index, split.error_reduction), (7, 3.0));
    }

    #[test]
    fn best_prefers_a_better_resplit_until_the_consecutive_resplit_limit() {
        let strategy = best(1, 0.0);
        let gains = vec![NAN, 3.0, 3.0, NAN, 3.0, 3.0, NAN, 3.0];
        let mut state = strategy.initialize(scored_state(vec![gains]));
        state.boundaries[0] = vec![3];
        let statistics = &mut state.precomputed_statistics;
        statistics.error_reductions_resplit[0] = vec![20.0];
        statistics.update_pairs_resplit_left[0] = vec![(0.0, 0.0)];
        statistics.update_pairs_resplit_right[0] = vec![(0.0, 0.0)];

        assert!(matches!(
            propose(&strategy, &state),
            Some(FittingAction::ApplyResplit { .. })
        ));

        state.loop_state.consecutive_resplits = MAX_CONSECUTIVE_RESPLIT;
        assert_eq!(proposed_split(&strategy, &state).error_reduction, 3.0);
    }
}
