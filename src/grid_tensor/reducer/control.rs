//! Loop-control counters per structural action.

use crate::grid_tensor::{
    splitting::{MergeCandidate, ResplitCandidate, SplitCandidate},
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
