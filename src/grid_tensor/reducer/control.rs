//! Loop-control counters per structural action, and the terminate, composite and scaling actions.

use super::fitting_reducer;
use crate::grid_tensor::{
    action::FittingAction,
    refinement::RefinementStrategy,
    splitting::{MergeCandidate, ResplitCandidate, SplitCandidate, SplitStrategy},
    state::FittingState,
};

pub(super) fn reduce_loop_state_for_split<'a>(
    mut state: FittingState<'a>,
    _split_candidate: &SplitCandidate,
) -> FittingState<'a> {
    state.loop_state.split_count += 1;
    state.loop_state.fineness += 1;
    state.loop_state.consecutive_resplits = 0;
    state
}

pub(super) fn reduce_loop_state_for_resplit<'a>(
    mut state: FittingState<'a>,
    _resplit_candidate: &ResplitCandidate,
) -> FittingState<'a> {
    state.loop_state.resplit_count += 1;
    state.loop_state.consecutive_resplits += 1;
    state
}

pub(super) fn reduce_loop_state_for_merge<'a>(
    mut state: FittingState<'a>,
    _merge_candidate: &MergeCandidate,
) -> FittingState<'a> {
    state.loop_state.merge_count += 1;
    state.loop_state.fineness = state.loop_state.fineness.saturating_sub(1);
    state.loop_state.consecutive_resplits = 0;
    state
}

/// Terminate the fitting process
pub(super) fn reduce_terminate(mut state: FittingState<'_>, reason: String) -> FittingState<'_> {
    // For now, just return the state
    // In the future, we could set a termination flag
    // The reason is consumed but not currently used
    log::debug!("Terminating; reason='{}'", reason);
    state.terminated = true;
    state
}

/// Apply multiple actions in sequence
pub(super) fn reduce_composite<'a>(
    mut state: FittingState<'a>,
    actions: Vec<FittingAction>,
    refinement_strategy: &RefinementStrategy,
    split_strategy: &SplitStrategy,
) -> FittingState<'a> {
    for action in actions {
        state = fitting_reducer(state, action, refinement_strategy, split_strategy);
    }
    state
}

/// Multiply the model's global scaling by a factor
pub(super) fn reduce_apply_scaling(mut state: FittingState<'_>, factor: f64) -> FittingState<'_> {
    log::debug!(
        "Applying scaling factor: {} (old scaling: {})",
        factor,
        state.scaling
    );
    state.scaling *= factor;
    state.y_hat *= factor;
    state.residuals = &state.labels - &state.y_hat;
    state
}
