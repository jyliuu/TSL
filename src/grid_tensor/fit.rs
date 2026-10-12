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
    if let Some(prepared) = prepared {
        prepared.assert_matches(x, hyperparameters.max_bins);
    }
    let refinement_strategy = hyperparameters
        .refinement_strategy_params
        .get_refinement_strategy();

    let split_strategy = hyperparameters.split_strategy_params.get_split_strategy();

    // Initialize state
    let mut state = FittingState::new(x, y);

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
    state.current_error = refinement_strategy.weighted_squared_error(&state.residuals);
    split_strategy.initialize(state)
}

/// Fit from a state initialized with the same data and hyperparameters.
pub(crate) fn fit_initialized<R: Rng + ?Sized>(
    mut state: FittingState<'_>,
    hyperparameters: &GridTensorParams,
    rng: &mut R,
) -> (FitResult, GridTensor) {
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
    finish_fit(state)
}

/// Upper bound on structural actions for a grid with fineness budget `n_iter`.
pub(crate) fn max_iterations(n_iter: usize) -> usize {
    n_iter * MAX_ITERATIONS_MULTIPLIER
}

/// Propose and apply one structural action.
///
/// Returns `false` once the grid is finished: the split budget is exhausted with merges
/// disabled, no objective-improving action remains, or the state is marked terminated.
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
    if state.terminated {
        return (state, false);
    }

    // Log component states and f+/f- statistics every iteration
    // Use reducer pattern - always call it, even if logging_state is None (no-op)
    // Feature flag is encapsulated inside the reducers
    #[cfg(feature = "evo-logging")]
    use crate::logging::reducer::{log_component_reducer, log_f_component_reducer};

    // Log f+/f- statistics (reducer computes stats internally from state)
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

    // Add component state logging for final state AFTER identification has been applied
    // This ensures we save the identified components (including λ_±), not pre-identification ones.
    // Feature flag is encapsulated inside the logging helper function
    // Always call it - it handles None internally
    #[cfg(feature = "evo-logging")]
    {
        use crate::grid_tensor::logging_helpers::log_final_component_states;
        log_final_component_states(logging_state, &grid_tensor, final_iteration);
    }
    grid_tensor
}

/// Identify the fitted grid and evaluate it on the training rows.
pub(crate) fn finish_fit(state: FittingState<'_>) -> (FitResult, GridTensor) {
    let (x, labels) = (state.x, state.labels);
    let grid_tensor = finish_grid(state);
    // `FitResult` is consistent with the returned `GridTensor`: after identification the
    // returned y_hat/residuals/err equal `grid_tensor.predict(x)` and its residuals.
    let y_hat = grid_tensor.predict(x.view());
    let residuals = (labels.to_owned() - y_hat.view()).to_owned();
    let err = residuals.pow2().mean().unwrap();
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

    // grid_values is no longer stored - computed on-demand via get_mean_factor()

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
mod tests;

#[cfg(test)]
mod two_tensor_invariant_tests;
