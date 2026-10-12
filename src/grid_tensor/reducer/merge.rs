//! Merge: removes a boundary and moves the rows of both intervals onto the merged factors.

use super::{
    adjacent_factors,
    caches::{refresh_action_statistics, update_prediction_caches},
    StatisticsPlan,
};
use crate::grid_tensor::{
    axis_factors,
    refinement::RefinementStrategy,
    splitting::MergeCandidate,
    state::FittingState,
    two_tensor_solver::{
        convert_multipliers_to_backbone_tilt, MAX_BACKBONE, MAX_TILT, MIN_BACKBONE,
    },
};

/// Merge intervals `merge.interval_idx` and `merge.interval_idx + 1` on `merge.col` into one
/// interval with the cached optimal axis factors.
pub(super) fn apply<'a>(
    mut state: FittingState<'a>,
    merge: MergeCandidate,
    refinement: &RefinementStrategy,
    plan: StatisticsPlan,
) -> FittingState<'a> {
    let col = merge.col;
    let interval_idx = merge.interval_idx;
    let (start, index, end) = state.interval_range_left_and_right(col, interval_idx);
    let [left, right] = adjacent_factors(&state, col, interval_idx);
    let (a_plus, a_minus) = state.precomputed_statistics.update_pairs_merge[col][interval_idx];
    let (backbone, tilt) = convert_multipliers_to_backbone_tilt(a_plus, a_minus);

    remove_boundary(&mut state, col, interval_idx, backbone, tilt);
    let merged = axis_factors(
        state.backbone_values[col][interval_idx],
        state.tilt_values[col][interval_idx],
    );
    // Each row divides out its own side's factor before taking the merged one.
    update_prediction_caches(&mut state, col, start, index, |_, plus, minus| {
        (plus / left.0 * merged.0, minus / left.1 * merged.1)
    });
    update_prediction_caches(&mut state, col, index, end, |_, plus, minus| {
        (plus / right.0 * merged.0, minus / right.1 * merged.1)
    });

    state = refresh_action_statistics(state, col, (start, end), plan, refinement);
    state.current_error = refinement.weighted_squared_error(&state.residuals, state.sample_weight);
    state.loop_state.record_merge();
    state
}

/// Remove boundary `interval_idx` on `col`; the merged interval takes the absolute parameters
/// `(backbone, tilt)`, clamped to the grid's bounds.
fn remove_boundary(
    state: &mut FittingState<'_>,
    col: usize,
    interval_idx: usize,
    backbone: f64,
    tilt: f64,
) {
    state.boundaries[col].remove(interval_idx);

    state.backbone_values[col][interval_idx] = backbone.clamp(MIN_BACKBONE, MAX_BACKBONE);
    state.tilt_values[col][interval_idx] = tilt.clamp(-MAX_TILT, MAX_TILT);
    state.backbone_values[col].remove(interval_idx + 1);
    state.tilt_values[col].remove(interval_idx + 1);

    let statistics = &mut state.precomputed_statistics;
    statistics.error_reductions_resplit[col].remove(interval_idx);
    statistics.update_pairs_resplit_left[col].remove(interval_idx);
    statistics.update_pairs_resplit_right[col].remove(interval_idx);
    statistics.error_reductions_resplit_pairs[col].remove(interval_idx);
    statistics.error_reductions_merge[col].remove(interval_idx);
    statistics.update_pairs_merge[col].remove(interval_idx);
    statistics.interval_stats[col].remove(interval_idx + 1);
}
