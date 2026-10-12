//! Merge: removes a boundary and reassigns its rows from the axis partial products.

use super::{
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
    let (start, _, end) = state.interval_range_left_and_right(col, interval_idx);
    let (partial_plus, partial_minus) = compute_partial_products_for_axis(col, &state);
    let (a_plus, a_minus) = state.precomputed_statistics.update_pairs_merge[col][interval_idx];
    let (backbone, tilt) = convert_multipliers_to_backbone_tilt(a_plus, a_minus);

    remove_boundary(&mut state, col, interval_idx, backbone, tilt);
    let (g_plus, g_minus) = axis_factors(
        state.backbone_values[col][interval_idx],
        state.tilt_values[col][interval_idx],
    );
    update_prediction_caches(&mut state, col, start, end, |row, _, _| {
        (partial_plus[row] * g_plus, partial_minus[row] * g_minus)
    });

    state = refresh_action_statistics(state, col, (start, end), plan, refinement);
    state.current_error = refinement.weighted_squared_error(&state.residuals);
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

/// Compute partial products g_{\pm}^{(-j)} for all points by dividing out axis j factor.
///
/// Uses the efficient approach: g_{+,i}^{(-j)} = f_{+,i} / a_{+,j}^{k_j(i)}
/// where a_{+,j}^{k_j(i)} = b_{j,i} * exp(d_{j,i}) and k_j(i) is the number of boundaries
/// at or below the row's rank on axis j.
///
/// # Arguments
/// * `col` - The axis index j
/// * `state` - The current fitting state
///
/// # Returns
/// `(g_plus, g_minus)` vectors of length n, where:
/// - `g_plus[i] = f_{+,i} / (b_{j,i} * exp(d_{j,i}))`
/// - `g_minus[i] = f_{-,i} / (b_{j,i} * exp(-d_{j,i}))`
pub(crate) fn compute_partial_products_for_axis(
    col: usize,
    state: &FittingState,
) -> (Vec<f64>, Vec<f64>) {
    (0..state.n)
        .map(|i| {
            let interval_idx = state.interval_index(col, i);
            let (a_plus_j, a_minus_j) = axis_factors(
                state.backbone_values[col][interval_idx],
                state.tilt_values[col][interval_idx],
            );
            (state.f_plus[i] / a_plus_j, state.f_minus[i] / a_minus_j)
        })
        .unzip()
}
