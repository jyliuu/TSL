//! Resplit reducer: updates the two intervals around a boundary and rescales their rows.

use super::caches::update_prediction_caches;
use crate::grid_tensor::state::FittingState;

/// Update backbone and tilt values for a resplit using two-tensor (v_b, Δd) conversion
pub(super) fn reduce_grid_resplit_two_tensor(
    mut state: FittingState<'_>,
    v_b_l: f64,
    delta_d_l: f64,
    v_b_r: f64,
    delta_d_r: f64,
    col: usize,
    interval_idx: usize,
) -> FittingState<'_> {
    // Get parent parameters (for resplit, we update existing intervals)
    let b_p_left = state.backbone_values[col][interval_idx];
    let d_p_left = state.tilt_values[col][interval_idx];
    let b_p_right = state.backbone_values[col][interval_idx + 1];
    let d_p_right = state.tilt_values[col][interval_idx + 1];

    // Update left interval: b_L = b_p * v_b_L, d_L = d_p + Δd_L
    const MIN_BACKBONE: f64 = 1e-10;
    const MAX_BACKBONE: f64 = 1e10;
    const MAX_TILT: f64 = 10.0;
    state.backbone_values[col][interval_idx] = (b_p_left * v_b_l).clamp(MIN_BACKBONE, MAX_BACKBONE);
    state.tilt_values[col][interval_idx] = (d_p_left + delta_d_l).clamp(-MAX_TILT, MAX_TILT);

    // Update right interval: b_R = b_p * v_b_R, d_R = d_p + Δd_R
    state.backbone_values[col][interval_idx + 1] =
        (b_p_right * v_b_r).clamp(MIN_BACKBONE, MAX_BACKBONE);
    state.tilt_values[col][interval_idx + 1] = (d_p_right + delta_d_r).clamp(-MAX_TILT, MAX_TILT);

    // Stage 1 invariants: I21 (debug builds only)
    #[cfg(debug_assertions)]
    {
        if state.is_stage1_positive_only() {
            const EPS: f64 = 1e-10;
            // I21: all d_j = 0
            for (j, d_vec) in state.tilt_values.iter().enumerate() {
                for (k, &d) in d_vec.iter().enumerate() {
                    assert!(
                        d.abs() < EPS,
                        "I21 violation: tilt[{j}][{k}] should be 0 in Stage 1 mode, got {}",
                        d
                    );
                }
            }
        }
    }

    state
}

/// Update per-point caches for a resplit using explicit old/new axis factors per side.
pub(super) fn reduce_grid_predictions_and_residuals_resplit_two_tensor_factors<'a>(
    mut state: FittingState<'a>,
    col: usize,
    index: usize,
    original_affected_interval: &(usize, usize),
    g_plus_old_l: f64,
    g_minus_old_l: f64,
    g_plus_old_r: f64,
    g_minus_old_r: f64,
    g_plus_new_l: f64,
    g_minus_new_l: f64,
    g_plus_new_r: f64,
    g_minus_new_r: f64,
) -> FittingState<'a> {
    let &(start, end) = original_affected_interval;

    let ratio_plus_left = g_plus_new_l / g_plus_old_l;
    let ratio_minus_left = g_minus_new_l / g_minus_old_l;
    let ratio_plus_right = g_plus_new_r / g_plus_old_r;
    let ratio_minus_right = g_minus_new_r / g_minus_old_r;

    update_prediction_caches(&mut state, col, start, index, |_, plus, minus| {
        (plus * ratio_plus_left, minus * ratio_minus_left)
    });
    update_prediction_caches(&mut state, col, index, end, |_, plus, minus| {
        (plus * ratio_plus_right, minus * ratio_minus_right)
    });

    state
}
