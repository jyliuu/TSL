//! Merge reducer: removes a boundary and reassigns rows from the axis partial products.

use super::caches::update_prediction_caches;
use crate::grid_tensor::state::FittingState;

/// Merge=OptimalMerge: apply optimal merged parameters and remove boundary
///
/// # Arguments
/// * `b_merged_optimal` - Absolute optimal backbone value for merged interval
/// * `d_merged_optimal` - Absolute optimal tilt value for merged interval
pub(super) fn reduce_grid_merge_optimal<'a>(
    mut state: FittingState<'a>,
    col: usize,
    interval_idx: usize,
    b_merged_optimal: f64,
    d_merged_optimal: f64,
) -> FittingState<'a> {
    // Remove the boundary inserted by the split
    state.boundaries[col].remove(interval_idx);

    // Apply merged params: set to absolute optimal values (not relative to left child)
    const MIN_BACKBONE: f64 = 1e-10;
    const MAX_BACKBONE: f64 = 1e10;
    const MAX_TILT: f64 = 10.0;
    state.backbone_values[col][interval_idx] = b_merged_optimal.clamp(MIN_BACKBONE, MAX_BACKBONE);
    state.tilt_values[col][interval_idx] = d_merged_optimal.clamp(-MAX_TILT, MAX_TILT);

    // Remove right child interval
    state.backbone_values[col].remove(interval_idx + 1);
    state.tilt_values[col].remove(interval_idx + 1);

    // Remove resplit/merge statistics aligned with the removed boundary
    state.precomputed_statistics.error_reductions_resplit[col].remove(interval_idx);
    state.precomputed_statistics.update_pairs_resplit_left[col].remove(interval_idx);
    state.precomputed_statistics.update_pairs_resplit_right[col].remove(interval_idx);
    state.precomputed_statistics.error_reductions_resplit_pairs[col].remove(interval_idx);
    state.precomputed_statistics.error_reductions_merge[col].remove(interval_idx);
    state.precomputed_statistics.update_pairs_merge[col].remove(interval_idx);

    // Remove right interval stats
    state.precomputed_statistics.interval_stats[col].remove(interval_idx + 1);

    state
}

/// Assign predictions from pre-merge partial products and the stored merged axis factors.
pub(super) fn reduce_grid_predictions_and_residuals_merge_from_partial_products<'a>(
    mut state: FittingState<'a>,
    col: usize,
    original_affected_interval: &(usize, usize),
    partial_plus: &[f64],
    partial_minus: &[f64],
    g_plus_new: f64,
    g_minus_new: f64,
) -> FittingState<'a> {
    let &(start, end) = original_affected_interval;
    update_prediction_caches(&mut state, col, start, end, |row, _, _| {
        (
            partial_plus[row] * g_plus_new,
            partial_minus[row] * g_minus_new,
        )
    });

    state
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
    let n = state.n;
    let mut g_plus = Vec::with_capacity(n);
    let mut g_minus = Vec::with_capacity(n);
    let boundaries = &state.boundaries[col];
    let ranks = &state.precomputed_statistics.sort_order[col];

    for i in 0..n {
        let interval_idx = boundaries.partition_point(|&b| b <= ranks[i]);

        let b_j = state.backbone_values[col][interval_idx];
        let d_j = state.tilt_values[col][interval_idx];

        // Compute axis factors: a_{+,j}^{k_j(i)} = b_j * exp(d_j), a_{-,j}^{k_j(i)} = b_j * exp(-d_j)
        let a_plus_j = b_j * d_j.exp();
        let a_minus_j = b_j * (-d_j).exp();

        g_plus.push(state.f_plus[i] / a_plus_j);
        g_minus.push(state.f_minus[i] / a_minus_j);
    }

    (g_plus, g_minus)
}
