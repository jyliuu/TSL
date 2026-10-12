//! Resplit: refits the two intervals around an existing boundary and rescales their rows.

use super::{
    adjacent_factors,
    caches::{refresh_action_statistics, rescale_branches},
    scaled_parameters, side_multipliers, StatisticsPlan,
};
use crate::grid_tensor::{
    refinement::RefinementStrategy, splitting::ResplitCandidate, state::FittingState,
};

/// Refit intervals `resplit.interval_idx` and `resplit.interval_idx + 1` on `resplit.col`.
pub(super) fn apply<'a>(
    mut state: FittingState<'a>,
    resplit: ResplitCandidate,
    refinement: &RefinementStrategy,
    plan: StatisticsPlan,
) -> FittingState<'a> {
    let col = resplit.col;
    let interval_idx = resplit.interval_idx;
    let (start, _, end) = state.interval_range_left_and_right(col, interval_idx);
    let multipliers = side_multipliers(
        &state,
        refinement,
        [resplit.update_left, resplit.update_right],
    );
    let old_factors = adjacent_factors(&state, col, interval_idx);

    scale_sides(&mut state, col, interval_idx, multipliers);
    let new_factors = adjacent_factors(&state, col, interval_idx);
    rescale_branches(
        &mut state,
        col,
        (start, resplit.index, end),
        old_factors,
        new_factors,
    );

    state = refresh_action_statistics(state, col, (start, end), plan, refinement);
    state.current_error = refinement.weighted_squared_error(&state.residuals);
    state.loop_state.record_resplit();
    state
}

/// Scale the parameters of intervals `interval_idx` and `interval_idx + 1` on `col` by their
/// `(v_b, Δd)`.
fn scale_sides(
    state: &mut FittingState<'_>,
    col: usize,
    interval_idx: usize,
    multipliers: [(f64, f64); 2],
) {
    for (k, multiplier) in (interval_idx..).zip(multipliers) {
        let (backbone, tilt) = scaled_parameters(
            state.backbone_values[col][k],
            state.tilt_values[col][k],
            multiplier,
        );
        state.backbone_values[col][k] = backbone;
        state.tilt_values[col][k] = tilt;
    }

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
}
