//! Split reducer: inserts a boundary, rescales the affected rows, and forbids nearby splits.

use super::caches::update_prediction_caches;
use crate::grid_tensor::{
    splitting::{SplitCandidate, SplitStrategy},
    state::FittingState,
};

/// Update tree structure: boundaries and grid_values
pub(super) fn reduce_grid_split_two_tensor(
    mut state: FittingState<'_>,
    v_b_l: f64,
    delta_d_l: f64,
    v_b_r: f64,
    delta_d_r: f64,
    col: usize,
    index: usize,
) -> FittingState<'_> {
    let insert_at = state.boundaries[col].partition_point(|&b| b <= index);

    // Insert new boundary at the correct position
    state.boundaries[col].insert(insert_at, index);

    // Parent parameters
    let b_p = state.backbone_values[col][insert_at];
    let d_p = state.tilt_values[col][insert_at];

    // Child params: b_child = b_p * v_b, d_child = d_p + Δd
    // Use same clamping values as resplit for consistency
    const MIN_BACKBONE: f64 = 1e-10;
    const MAX_BACKBONE: f64 = 1e10;
    const MAX_TILT: f64 = 10.0;

    state.backbone_values[col][insert_at] = (b_p * v_b_l).clamp(MIN_BACKBONE, MAX_BACKBONE);
    let new_tilt_l = (d_p + delta_d_l).clamp(-MAX_TILT, MAX_TILT);
    state.tilt_values[col][insert_at] = new_tilt_l;

    state.backbone_values[col].insert(
        insert_at + 1,
        (b_p * v_b_r).clamp(MIN_BACKBONE, MAX_BACKBONE),
    );
    state.tilt_values[col].insert(insert_at + 1, (d_p + delta_d_r).clamp(-MAX_TILT, MAX_TILT));

    // Stage 1 invariants: I20-I27 (debug builds only)
    #[cfg(debug_assertions)]
    {
        if state.is_stage1_positive_only() {
            const EPS: f64 = 1e-10;
            // I20: λ_- = 0
            assert!(
                state.lambda_minus.abs() < EPS,
                "I20 violation: lambda_minus should be 0 in Stage 1 mode, got {}",
                state.lambda_minus
            );
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
            // I22: f_- = 0
            for (i, &fm) in state.f_minus.iter().enumerate() {
                assert!(
                    fm.abs() < EPS,
                    "I22 violation: f_minus[{}] should be 0 in Stage 1 mode, got {}",
                    i,
                    fm
                );
            }
            // I23: f >= 0
            for (i, &f) in state.f.iter().enumerate() {
                assert!(
                    f >= -EPS,
                    "I23 violation: f[{}] should be >= 0 in Stage 1 mode, got {}",
                    i,
                    f
                );
            }
        }
    }

    // Insert placeholders for resplit/merge statistics aligned with the new boundary
    state.precomputed_statistics.error_reductions_resplit[col].insert(insert_at, f64::NAN);
    state.precomputed_statistics.update_pairs_resplit_left[col]
        .insert(insert_at, (f64::NAN, f64::NAN));
    state.precomputed_statistics.update_pairs_resplit_right[col]
        .insert(insert_at, (f64::NAN, f64::NAN));
    state.precomputed_statistics.error_reductions_resplit_pairs[col]
        .insert(insert_at, (f64::NAN, f64::NAN));
    state.precomputed_statistics.error_reductions_merge[col].insert(insert_at, f64::NAN);
    state.precomputed_statistics.update_pairs_merge[col].insert(insert_at, (f64::NAN, f64::NAN));

    // Split interval stats: insert placeholder for the new right interval
    // (actual values will be computed in update_statistics)
    state.precomputed_statistics.interval_stats[col].insert(
        insert_at + 1,
        crate::grid_tensor::state::IntervalStats::default(),
    );

    state
}

/// Update per-point caches (f_±, f, y_hat, residuals, r_tilde) for a split using explicit axis factors.
///
/// Here:
/// - `g_plus_old/g_minus_old` are the parent axis factors on this column before the split
/// - `g_plus_new_* / g_minus_new_*` are the child axis factors for left/right after the split
pub(super) fn reduce_grid_predictions_and_residuals_two_tensor_factors<'a>(
    mut state: FittingState<'a>,
    col: usize,
    index: usize,
    original_affected_interval: &(usize, usize),
    g_plus_old: f64,
    g_minus_old: f64,
    g_plus_new_l: f64,
    g_minus_new_l: f64,
    g_plus_new_r: f64,
    g_minus_new_r: f64,
) -> FittingState<'a> {
    let &(start, end) = original_affected_interval;

    let ratio_plus_left = g_plus_new_l / g_plus_old;
    let ratio_minus_left = g_minus_new_l / g_minus_old;
    let ratio_plus_right = g_plus_new_r / g_plus_old;
    let ratio_minus_right = g_minus_new_r / g_minus_old;

    update_prediction_caches(&mut state, col, start, index, |_, plus, minus| {
        (plus * ratio_plus_left, minus * ratio_minus_left)
    });
    update_prediction_caches(&mut state, col, index, end, |_, plus, minus| {
        (plus * ratio_plus_right, minus * ratio_minus_right)
    });

    state
}

/// Forbid splits around the given position (ported from SplitStrategy::forbid_around_split)
pub(super) fn reduce_forbid_around_split<'a>(
    state: FittingState<'a>,
    split_candidate: &SplitCandidate,
    split_strategy: &SplitStrategy,
) -> FittingState<'a> {
    split_strategy.forbid_around_split(state, split_candidate)
}
