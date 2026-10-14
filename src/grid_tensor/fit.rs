//! The grid-tensor fitting loop: prepare the state, apply proposed structural actions, and
//! identify the fitted grid.

use ndarray::ArrayView2;
use rand::Rng;

use crate::{
    grid_tensor::{
        identification::l2_identify,
        reducer::{fitting_reducer_with_candidate_refresh, CandidateRefresh},
        refinement::RefinementStrategy,
        splitting::SplitStrategy,
        state::FittingState,
        GridTensor, GridTensorParams,
    },
    FitResult,
};

const MAX_ITERATIONS_MULTIPLIER: usize = 3;

/// Fit a tree grid model using the action-based reducer pattern
///
/// The fitting loop follows a clean action-based architecture:
/// 1. Strategy proposes an action (split/resplit/merge)
/// 2. Reducer applies the action to update state
///
/// This separation of concerns makes the code easier to understand and test.
pub fn fit<R: Rng + ?Sized>(
    x: ndarray::ArrayView2<f64>,
    y: ndarray::ArrayView1<f64>,
    hyperparameters: &GridTensorParams,
    rng: &mut R,
) -> (FitResult, GridTensor) {
    let state = initialize_state(x.view(), y.view(), hyperparameters);
    fit_initialized(state, hyperparameters, rng)
}

/// Prepare the deterministic statistics and candidate caches for one stage.
pub(crate) fn initialize_state<'a>(
    x: ndarray::ArrayView2<'a, f64>,
    y: ndarray::ArrayView1<'a, f64>,
    hyperparameters: &GridTensorParams,
) -> FittingState<'a> {
    initialize_state_with_input(x, y, hyperparameters, None)
}

pub(crate) fn initialize_state_with_input<'a>(
    x: ndarray::ArrayView2<'a, f64>,
    y: ndarray::ArrayView1<'a, f64>,
    hyperparameters: &GridTensorParams,
    prepared: Option<&super::prepared_input::PreparedInput<'_>>,
) -> FittingState<'a> {
    initialize_weighted_state(x, y, None, hyperparameters, prepared)
}

/// Prepare a stage whose loss weighs row `i` by `sample_weight[i]`.
pub(crate) fn initialize_weighted_state<'a>(
    x: ndarray::ArrayView2<'a, f64>,
    y: ndarray::ArrayView1<'a, f64>,
    sample_weight: Option<ndarray::ArrayView1<'a, f64>>,
    hyperparameters: &GridTensorParams,
    prepared: Option<&super::prepared_input::PreparedInput<'_>>,
) -> FittingState<'a> {
    if let Some(prepared) = prepared {
        prepared.assert_matches(x, hyperparameters.max_bins);
    }
    let refinement_strategy = hyperparameters
        .refinement_strategy_params
        .get_refinement_strategy();

    let split_strategy = hyperparameters.split_strategy_params.get_split_strategy();

    let mut state = FittingState::new(x, y);
    state.sample_weight = sample_weight;

    // Initialize logging state if logging is enabled
    #[cfg(feature = "evo-logging")]
    {
        use crate::grid_tensor::logging_helpers::LoggingState;
        use crate::logging::is_logging_enabled;
        if is_logging_enabled() {
            state.logging_state = Some(LoggingState::new());
        }
    }

    let state =
        refinement_strategy.initialize_with_input(state, prepared, hyperparameters.max_bins);
    let mut state = state;
    state.current_error =
        refinement_strategy.weighted_squared_error(&state.residuals, state.sample_weight);
    split_strategy.initialize(state)
}

/// Fit from a state initialized with the same data and hyperparameters.
pub(crate) fn fit_initialized<R: Rng + ?Sized>(
    state: FittingState<'_>,
    hyperparameters: &GridTensorParams,
    rng: &mut R,
) -> (FitResult, GridTensor) {
    finish_fit(fit_structure(state, hyperparameters, rng))
}

/// Apply structural actions until the grid is finished or its action budget is spent.
pub(crate) fn fit_structure<'a, R: Rng + ?Sized>(
    mut state: FittingState<'a>,
    hyperparameters: &GridTensorParams,
    rng: &mut R,
) -> FittingState<'a> {
    let refinement_strategy = hyperparameters
        .refinement_strategy_params
        .get_refinement_strategy();
    let split_strategy = hyperparameters.split_strategy_params.get_split_strategy();

    for iter in 0..max_iterations(hyperparameters.n_iter) {
        let advanced;
        (state, advanced) = fit_step(
            state,
            iter,
            hyperparameters.n_iter,
            &refinement_strategy,
            &split_strategy,
            rng,
        );
        if !advanced {
            break;
        }
    }
    state
}

/// Upper bound on structural actions for a grid with fineness budget `n_iter`.
pub(crate) fn max_iterations(n_iter: usize) -> usize {
    n_iter * MAX_ITERATIONS_MULTIPLIER
}

/// Propose and apply one structural action.
///
/// Returns `false` once the grid is finished: the split budget is exhausted with merges
/// disabled, or no objective-improving action remains.
pub(crate) fn fit_step<'a, R: Rng + ?Sized>(
    state: FittingState<'a>,
    iter: usize,
    n_iter: usize,
    refinement_strategy: &RefinementStrategy,
    split_strategy: &SplitStrategy,
    rng: &mut R,
) -> (FittingState<'a>, bool) {
    fit_step_with_candidate_refresh(
        state,
        iter,
        n_iter,
        refinement_strategy,
        split_strategy,
        rng,
        CandidateRefresh::Eager,
    )
}

/// Take one action, retaining row and prefix statistics under either refresh policy.
/// Deferred candidate scores require a full statistics refresh before the next proposal.
pub(crate) fn fit_step_with_candidate_refresh<'a, R: Rng + ?Sized>(
    mut state: FittingState<'a>,
    iter: usize,
    n_iter: usize,
    refinement_strategy: &RefinementStrategy,
    split_strategy: &SplitStrategy,
    rng: &mut R,
    candidate_refresh: CandidateRefresh,
) -> (FittingState<'a>, bool) {
    state.iteration = iter;

    if state.structure_is_final(n_iter) {
        return (state, false);
    }

    // The fineness budget excludes new splits but leaves resplits and merges
    // available under the same objective. A merge frees capacity for a later split.
    let allow_split = state.loop_state.fineness < n_iter;
    let action = if state.sampled_split_scoring || state.split_scores_stale {
        split_strategy.propose_next_action_with_sampled_scores(
            &mut state,
            refinement_strategy,
            rng,
            allow_split,
        )
    } else {
        split_strategy.propose_next_action_with_split_budget(&state, rng, allow_split)
    };
    let Some(action) = action else {
        log::debug!("No objective-improving structural action remains, stopping");
        return (state, false);
    };

    // Apply action through reducer (which calls logging reducer internally)
    state = fitting_reducer_with_candidate_refresh(
        state,
        action,
        refinement_strategy,
        split_strategy,
        candidate_refresh,
    );

    // Log the f+/f- statistics and component states after every action; both reducers
    // pass a `None` logging state through unchanged.
    #[cfg(feature = "evo-logging")]
    use crate::logging::reducer::{log_component_reducer, log_f_component_reducer};

    #[cfg(feature = "evo-logging")]
    {
        state.logging_state = log_f_component_reducer(state.logging_state.take(), &state, iter);

        state.logging_state = log_component_reducer(state.logging_state.take(), &state, iter);
    }
    (state, true)
}

/// Identify the fitted grid and log its final component states.
pub(crate) fn finish_grid(mut state: FittingState<'_>) -> GridTensor {
    log::debug!(
        "Split/Resplit/Merge counts: {}/{}/{}",
        state.loop_state.split_count,
        state.loop_state.resplit_count,
        state.loop_state.merge_count
    );

    // Extract logging state before conversion (for final state logging)
    #[cfg(feature = "evo-logging")]
    let logging_state = state.logging_state.take();
    #[cfg(feature = "evo-logging")]
    let final_iteration = state.iteration;

    let grid_tensor = convert_state_to_grid(state);

    // Log the final component states after identification, so they match the returned grid
    // (including λ_±); a `None` logging state is a no-op.
    #[cfg(feature = "evo-logging")]
    {
        use crate::grid_tensor::logging_helpers::log_final_component_states;
        log_final_component_states(logging_state, &grid_tensor, final_iteration);
    }
    grid_tensor
}

/// Identify the fitted grid and evaluate it on the training rows.
pub(crate) fn finish_fit(state: FittingState<'_>) -> (FitResult, GridTensor) {
    let (x, labels, sample_weight) = (state.x, state.labels, state.sample_weight);
    let grid_tensor = finish_grid(state);
    // `FitResult` is consistent with the returned `GridTensor`: after identification the
    // returned y_hat/residuals/err equal `grid_tensor.predict(x)` and its residuals.
    let y_hat = grid_tensor.predict(x.view());
    let residuals = &labels - &y_hat;
    let err = crate::weighted_mean_squared_error(residuals.view(), sample_weight);
    (
        FitResult {
            err,
            residuals,
            y_hat,
        },
        grid_tensor,
    )
}

/// Compute real intervals from state boundaries and sorted indices
fn compute_real_intervals(
    x: ArrayView2<f64>,
    boundaries: &[Vec<usize>],
    sorted_indices: &[Vec<usize>],
) -> Vec<Vec<(f64, f64)>> {
    boundaries
        .iter()
        .enumerate()
        .map(|(col, bounds)| {
            let n_points = sorted_indices[col].len();
            let mut v = Vec::with_capacity(bounds.len() + 1);
            let mut prev = 0usize;
            for &b in bounds {
                let left = if prev == 0 {
                    f64::NEG_INFINITY
                } else {
                    x[[sorted_indices[col][prev], col]]
                };
                let right = if b == n_points {
                    f64::INFINITY
                } else {
                    x[[sorted_indices[col][b], col]]
                };
                v.push((left, right));
                prev = b;
            }
            let left = if prev == 0 {
                f64::NEG_INFINITY
            } else {
                x[[sorted_indices[col][prev], col]]
            };
            let right = f64::INFINITY;
            v.push((left, right));
            v
        })
        .collect()
}

/// Compute splits from state boundaries and sorted indices
fn compute_splits(
    x: ArrayView2<f64>,
    boundaries: &[Vec<usize>],
    sorted_indices: &[Vec<usize>],
) -> Vec<Vec<f64>> {
    boundaries
        .iter()
        .enumerate()
        .map(|(col, bounds)| {
            bounds
                .iter()
                .map(|&pos| x[[sorted_indices[col][pos], col]])
                .collect()
        })
        .collect()
}

/// Compute observation counts from state boundaries
fn compute_observation_counts(
    boundaries: &[Vec<usize>],
    sorted_indices: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    boundaries
        .iter()
        .enumerate()
        .map(|(col, bounds)| {
            let n_points = sorted_indices[col].len();
            let mut out = Vec::with_capacity(bounds.len() + 1);
            let mut prev = 0usize;
            for &b in bounds {
                out.push(b - prev);
                prev = b;
            }
            out.push(n_points - prev);
            out
        })
        .collect()
}

/// Convert FittingState into the final output types
fn convert_state_to_grid(state: FittingState) -> GridTensor {
    let x = state.x;
    let boundaries = state.boundaries;
    let precomputed_statistics = state.precomputed_statistics;

    let mut backbone_values = state.backbone_values;
    let mut tilt_values = state.tilt_values;
    let mut lambda_plus = state.lambda_plus;
    let mut lambda_minus = state.lambda_minus;

    // Compute components needed for output regardless of identification/reproject.
    let real_intervals =
        compute_real_intervals(x, &boundaries, &precomputed_statistics.sorted_indices);
    let splits = compute_splits(x, &boundaries, &precomputed_statistics.sorted_indices);
    let observation_counts =
        compute_observation_counts(&boundaries, &precomputed_statistics.sorted_indices);

    // Apply two-tensor identification/normalization (prediction-preserving).
    l2_identify(
        &mut backbone_values,
        &mut tilt_values,
        &observation_counts,
        &mut lambda_plus,
        &mut lambda_minus,
    );

    GridTensor::new_two_tensor(
        splits,
        observation_counts,
        real_intervals,
        backbone_values,
        tilt_values,
        lambda_plus,
        lambda_minus,
    )
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::grid_tensor::{
        action::FittingAction,
        axis_factors,
        params::{RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder},
        reducer::fitting_reducer,
        splitting::{MergeCandidate, ResplitCandidate, SplitCandidate},
        two_tensor_solver::{log_coordinate_penalty, MAX_BACKBONE, MAX_TILT, MIN_BACKBONE},
        GridTensorParamsBuilder,
    };

    /// Two features and a target with an interaction, repeated feature values and outliers
    /// beyond the unit Huber threshold. The target changes sign unless `offset` lifts it.
    fn data(offset: f64) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((24, 2), |(i, j)| {
            ((i * (5 + 2 * j) + 3 * j) % 17) as f64 / 4.0 - 2.0
        });
        let y = Array1::from_shape_fn(24, |i| {
            let outlier = if i % 7 == 3 { 4.0 } else { 0.0 };
            x[[i, 0]] * x[[i, 1]].cos() + outlier + offset
        });
        (x, y)
    }

    /// Upper-right, lower-right and lower-left quadrants with targets {2, 4}, {2, 4} and
    /// {-12, -6}; the upper-left quadrant is empty.
    fn quadrant_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 2.0, 1.5, 3.0, 2.0, 4.0, 2.5, // upper right
                1.1, -1.0, 2.1, -1.5, 3.1, -2.0, 4.1, -2.5, // lower right
                -1.0, -1.0, -2.0, -1.5, -3.0, -2.0, -4.0, -2.5, // lower left
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0, -12.0, -12.0, -6.0, -6.0,
        ]);
        (x, y)
    }

    fn params(
        refinement: RefinementStrategyParamsBuilder,
        complexity_penalty: f64,
    ) -> GridTensorParams {
        GridTensorParamsBuilder::new()
            .n_iter(8)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .best_split()
                    .complexity_penalty(complexity_penalty)
                    .build(),
            )
            .refinement_strategy(refinement.build())
            .build()
    }

    /// The initialized state followed by the state after each action of the fit.
    fn action_states<'a>(
        x: &'a Array2<f64>,
        y: &'a Array1<f64>,
        sample_weight: Option<&'a Array1<f64>>,
        params: &GridTensorParams,
    ) -> Vec<FittingState<'a>> {
        let refinement = params.refinement_strategy_params.get_refinement_strategy();
        let split = params.split_strategy_params.get_split_strategy();
        let mut rng = StdRng::seed_from_u64(0);
        let mut states = vec![initialize_weighted_state(
            x.view(),
            y.view(),
            sample_weight.map(|weights| weights.view()),
            params,
            None,
        )];
        for iter in 0..max_iterations(params.n_iter) {
            let state = states.last().unwrap().clone();
            let (state, advanced) =
                fit_step(state, iter, params.n_iter, &refinement, &split, &mut rng);
            if !advanced {
                break;
            }
            states.push(state);
        }
        states
    }

    /// Every finite cached candidate of `state`: the splits at allowed positions and the
    /// resplit and merge of every boundary.
    fn cached_actions(state: &FittingState) -> Vec<FittingAction> {
        let statistics = &state.precomputed_statistics;
        let mut actions = Vec::new();
        for col in 0..state.p {
            let allowed = &state.split_strategy_state.allowed_intervals[col];
            for (allowed_interval_idx, interval) in allowed.iter().enumerate() {
                for index in interval.start..interval.end() {
                    let split = SplitCandidate {
                        col,
                        error_reduction: statistics.error_reductions_split[col][index],
                        allowed_interval_idx,
                        index,
                        update_left: statistics.update_pairs_split_left[col][index],
                        update_right: statistics.update_pairs_split_right[col][index],
                    };
                    actions.push(FittingAction::ApplySplit { split });
                }
            }
            for (interval_idx, &index) in state.boundaries[col].iter().enumerate() {
                let resplit = ResplitCandidate {
                    col,
                    error_reduction: statistics.error_reductions_resplit[col][interval_idx],
                    interval_idx,
                    index,
                    update_left: statistics.update_pairs_resplit_left[col][interval_idx],
                    update_right: statistics.update_pairs_resplit_right[col][interval_idx],
                };
                let merge = MergeCandidate {
                    col,
                    error_reduction: statistics.error_reductions_merge[col][interval_idx],
                    interval_idx,
                    index,
                };
                actions.push(FittingAction::ApplyResplit { resplit });
                actions.push(FittingAction::ApplyMerge { merge });
            }
        }
        actions.retain(|action| {
            match action {
                FittingAction::ApplySplit { split } => split.error_reduction,
                FittingAction::ApplyResplit { resplit } => resplit.error_reduction,
                FittingAction::ApplyMerge { merge } => merge.error_reduction,
            }
            .is_finite()
        });
        actions
    }

    fn factors(state: &FittingState, col: usize, interval_idx: usize) -> (f64, f64) {
        axis_factors(
            state.backbone_values[col][interval_idx],
            state.tilt_values[col][interval_idx],
        )
    }

    /// Apply `action` to a copy of `state` and check its cached score: the decrease of the
    /// loss with row weights frozen at the current residuals, less the log-coordinate penalty
    /// of the multipliers that move each resulting interval off its reference factors. A
    /// split's sides start from the parent's factors, a resplit's from their own, and a
    /// merged interval from the count-weighted geometric mean of the two it joins.
    fn assert_score_is_the_applied_decrease(
        state: &FittingState,
        action: FittingAction,
        refinement: &RefinementStrategy,
        split_strategy: &SplitStrategy,
    ) {
        let (col, first, score, references) = match &action {
            FittingAction::ApplySplit { split } => {
                let parent_idx = state.boundaries[split.col].partition_point(|&b| b <= split.index);
                let parent = factors(state, split.col, parent_idx);
                (
                    split.col,
                    parent_idx,
                    split.error_reduction,
                    vec![parent, parent],
                )
            }
            FittingAction::ApplyResplit { resplit } => {
                let (col, k) = (resplit.col, resplit.interval_idx);
                let sides = vec![factors(state, col, k), factors(state, col, k + 1)];
                (col, k, resplit.error_reduction, sides)
            }
            FittingAction::ApplyMerge { merge } => {
                let (col, k) = (merge.col, merge.interval_idx);
                let (start, index, end) = state.interval_range_left_and_right(col, k);
                let share = (index - start) as f64 / (end - start) as f64;
                let mean =
                    |left: f64, right: f64| (share * left.ln() + (1.0 - share) * right.ln()).exp();
                let (left, right) = (factors(state, col, k), factors(state, col, k + 1));
                let reference = (mean(left.0, right.0), mean(left.1, right.1));
                (col, k, merge.error_reduction, vec![reference])
            }
        };
        let after = fitting_reducer(state.clone(), action.clone(), refinement, split_strategy);

        let penalty: f64 = references
            .iter()
            .enumerate()
            .map(|(side, reference)| {
                let moved = factors(&after, col, first + side);
                log_coordinate_penalty(
                    moved.0 / reference.0,
                    moved.1 / reference.1,
                    refinement.alpha(),
                    refinement.tilt_tau(),
                    refinement.tilt_rho(),
                )
            })
            .sum();
        let decrease: f64 = state
            .residuals
            .iter()
            .zip(&after.residuals)
            .enumerate()
            .map(|(row, (&before, &now))| {
                state.sample_weight(row) * refinement.weight(before) * (before * before - now * now)
            })
            .sum();
        let tolerance = 1e-9 * state.current_error.max(1.0);
        assert!(
            (decrease - penalty - score).abs() <= tolerance,
            "{action:?}: score {score}, decrease {decrease}, penalty {penalty}"
        );
        assert!(
            (after.current_error
                - refinement.weighted_squared_error(&after.residuals, after.sample_weight))
            .abs()
                <= tolerance
        );
    }

    /// The returned grid reproduces the incrementally maintained predictions of the fitted
    /// state, and the `FitResult` holds that grid's predictions, residuals and mean squared
    /// error on the training rows.
    #[test]
    fn fit_result_and_grid_reproduce_the_fitted_predictions() {
        let (x, y) = data(0.0);
        let params = GridTensorParamsBuilder::new().n_iter(10).build();
        let state = initialize_state(x.view(), y.view(), &params);
        let state = fit_structure(state, &params, &mut StdRng::seed_from_u64(42));
        assert!(state
            .boundaries
            .iter()
            .any(|boundaries| !boundaries.is_empty()));
        let fitted = state.f.clone();

        let (result, grid) = finish_fit(state);
        let predictions = grid.predict(x.view());
        for row in 0..x.nrows() {
            assert!((predictions[row] - fitted[row]).abs() <= 1e-10 * fitted[row].abs().max(1.0));
            assert_eq!(result.y_hat[row], predictions[row]);
            assert_eq!(result.residuals[row], y[row] - predictions[row]);
        }
        let mse = result.residuals.mapv(|r| r * r).mean().unwrap();
        assert!((result.err - mse).abs() <= 1e-13);
    }

    /// Without penalties, best actions recover the quadrant targets exactly, with one boundary
    /// between the left and right halves and one at each change of target level along the
    /// second feature.
    #[test]
    fn unpenalized_fit_recovers_the_quadrant_targets() {
        let (x, y) = quadrant_data();
        let refinement = RefinementStrategyParamsBuilder::new()
            .l2()
            .alpha(0.0)
            .tilt_tau(0.0);
        let (result, grid) = fit(
            x.view(),
            y.view(),
            &params(refinement, 0.0),
            &mut StdRng::seed_from_u64(0),
        );
        assert_eq!(grid.splits, vec![vec![1.0], vec![-1.5, 2.0]]);
        for (prediction, target) in result.y_hat.iter().zip(&y) {
            assert!(
                (prediction - target).abs() <= 1e-9,
                "{prediction} vs {target}"
            );
        }
    }

    /// Every cached split, resplit and merge score equals the penalized loss decrease of
    /// applying it, for L2 and Huber objectives, every penalty, a multiplier cap, sample
    /// weights, and both two-tensor and stage-1 (positive target) states along a fit.
    #[test]
    fn action_scores_equal_the_penalized_loss_decrease_when_applied() {
        let refinements = [
            RefinementStrategyParamsBuilder::new().l2(),
            RefinementStrategyParamsBuilder::new()
                .l2()
                .alpha(0.1)
                .tilt_tau(0.05)
                .tilt_rho(0.02),
            RefinementStrategyParamsBuilder::new()
                .l2()
                .alpha(0.01)
                .update_clamp(1.0),
            RefinementStrategyParamsBuilder::new()
                .huber()
                .alpha(0.01)
                .tilt_tau(0.01)
                .tilt_rho(0.01),
        ];
        let mut checked = [0usize; 3];
        let weights = Array1::from_shape_fn(24, |i| ((i * 7) % 5) as f64 / 2.0);
        for refinement in refinements {
            for (offset, weighted) in [(0.0, false), (3.0, false), (0.0, true), (3.0, true)] {
                let (x, y) = data(offset);
                let params = params(refinement.clone(), 1e-3);
                let strategy = params.refinement_strategy_params.get_refinement_strategy();
                let split = params.split_strategy_params.get_split_strategy();
                for state in action_states(&x, &y, weighted.then_some(&weights), &params) {
                    for action in cached_actions(&state) {
                        checked[match action {
                            FittingAction::ApplySplit { .. } => 0,
                            FittingAction::ApplyResplit { .. } => 1,
                            FittingAction::ApplyMerge { .. } => 2,
                        }] += 1;
                        assert_score_is_the_applied_decrease(&state, action, &strategy, &split);
                    }
                }
            }
        }
        assert!(checked.iter().all(|&count| count > 0), "{checked:?}");
    }

    /// Without penalties, a split whose multipliers stay inside the bounds applies the optimal
    /// update of each side, so refitting the two sides of the new boundary gains nothing.
    #[test]
    fn unclamped_unpenalized_split_leaves_no_resplit_gain() {
        let refinement = RefinementStrategyParamsBuilder::new()
            .l2()
            .alpha(0.0)
            .tilt_tau(0.0);
        let params = params(refinement, 0.0);
        let strategy = params.refinement_strategy_params.get_refinement_strategy();
        let split_strategy = params.split_strategy_params.get_split_strategy();
        let (v_min, v_max) = strategy.multiplier_bounds();
        let inside = |(u_plus, u_minus): (f64, f64)| {
            [u_plus, u_minus]
                .iter()
                .all(|u| 1.0 + u > v_min + 1e-9 && 1.0 + u < v_max - 1e-9)
        };
        let mut checked = 0;
        for offset in [0.0, 3.0] {
            let (x, y) = data(offset);
            for state in action_states(&x, &y, None, &params) {
                for action in cached_actions(&state) {
                    let FittingAction::ApplySplit { split } = &action else {
                        continue;
                    };
                    if !(inside(split.update_left) && inside(split.update_right)) {
                        continue;
                    }
                    let (col, index) = (split.col, split.index);
                    let after = fitting_reducer(state.clone(), action, &strategy, &split_strategy);
                    let boundary = after.boundaries[col].binary_search(&index).unwrap();
                    let gain = after.precomputed_statistics.error_reductions_resplit[col][boundary];
                    assert!(
                        gain.abs() <= 1e-9 * state.current_error,
                        "resplit gain {gain}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0);
    }

    /// λ₊, λ₋ and every backbone stay positive and both branches stay finite and nonnegative
    /// after every action of a mixed-sign fit, and the identified grid keeps them nonnegative.
    #[test]
    fn lambdas_backbones_and_branches_stay_positive() {
        let (x, y) = data(0.0);
        let refinement = RefinementStrategyParamsBuilder::new().alpha(0.01);
        for refinement in [refinement.clone().l2(), refinement.huber()] {
            let states = action_states(&x, &y, None, &params(refinement, 1e-3));
            for state in &states {
                assert!(state.lambda_plus > 0.0 && state.lambda_minus > 0.0);
                assert!(state.backbone_values.iter().flatten().all(|&b| b > 0.0));
                let mut branches = state.f_plus.iter().chain(&state.f_minus);
                assert!(branches.all(|&f| f.is_finite() && f >= 0.0));
            }
            let grid = finish_grid(states.into_iter().last().unwrap());
            assert!(grid.lambda_plus >= 0.0 && grid.lambda_minus >= 0.0);
            assert!(grid.backbone_values.iter().flatten().all(|&b| b >= 0.0));
        }
    }

    /// Every split and resplit moves the branches of its rows by multipliers inside the
    /// refinement's bounds, which the quadrant targets reach, and every backbone and tilt
    /// stays inside the grid's parameter box.
    #[test]
    fn applied_multipliers_and_parameters_stay_within_their_bounds() {
        let (x, y) = quadrant_data();
        for update_clamp in [f64::INFINITY, 2.0, 0.5] {
            let params = params(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .update_clamp(update_clamp),
                0.0,
            );
            let strategy = params.refinement_strategy_params.get_refinement_strategy();
            let (v_min, v_max) = strategy.multiplier_bounds();
            let mut binding = false;
            for pair in action_states(&x, &y, None, &params).windows(2) {
                let [before, after] = [&pair[0], &pair[1]];
                for row in 0..x.nrows() {
                    for v in [
                        after.f_plus[row] / before.f_plus[row],
                        after.f_minus[row] / before.f_minus[row],
                    ] {
                        assert!(v >= v_min * (1.0 - 1e-12) && v <= v_max * (1.0 + 1e-12));
                        binding |= (v / v_min - 1.0).abs() < 1e-9 || (v / v_max - 1.0).abs() < 1e-9;
                    }
                }
                let mut backbones = after.backbone_values.iter().flatten();
                assert!(backbones.all(|b| (MIN_BACKBONE..=MAX_BACKBONE).contains(b)));
                let mut tilts = after.tilt_values.iter().flatten();
                assert!(tilts.all(|d| d.abs() <= MAX_TILT));
            }
            assert!(binding, "update_clamp {update_clamp} never binds");
        }
    }
}
