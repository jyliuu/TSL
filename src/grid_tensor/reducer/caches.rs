//! Refreshes the row prediction caches and candidate statistics a structural action invalidates.

use super::CandidateRefresh;
use crate::grid_tensor::{
    refinement::RefinementStrategy,
    state::{AffectedRange, FittingState},
};

/// Scatter branch predictions and their dependent caches in sorted-row order.
pub(super) fn update_prediction_caches(
    state: &mut FittingState<'_>,
    col: usize,
    start: usize,
    end: usize,
    update_branches: impl Fn(usize, f64, f64) -> (f64, f64),
) {
    let rows = &state.precomputed_statistics.sorted_indices[col][start..end];
    if let (
        Some(f_plus),
        Some(f_minus),
        Some(f),
        Some(y_hat),
        Some(residuals),
        Some(r_tilde),
        Some(labels),
    ) = (
        state.f_plus.as_slice_mut(),
        state.f_minus.as_slice_mut(),
        state.f.as_slice_mut(),
        state.y_hat.as_slice_mut(),
        state.residuals.as_slice_mut(),
        state.r_tilde.as_slice_mut(),
        state.labels.as_slice(),
    ) {
        for &row in rows {
            let (plus, minus) = update_branches(row, f_plus[row], f_minus[row]);
            let prediction = plus - minus;
            let residual = labels[row] - prediction;
            f_plus[row] = plus;
            f_minus[row] = minus;
            f[row] = prediction;
            y_hat[row] = prediction;
            residuals[row] = residual;
            r_tilde[row] = residual;
        }
    } else {
        // Views and externally supplied state arrays may have non-unit strides.
        for &row in rows {
            let (plus, minus) = update_branches(row, state.f_plus[row], state.f_minus[row]);
            let prediction = plus - minus;
            let residual = state.labels[row] - prediction;
            state.f_plus[row] = plus;
            state.f_minus[row] = minus;
            state.f[row] = prediction;
            state.y_hat[row] = prediction;
            state.residuals[row] = residual;
            state.r_tilde[row] = residual;
        }
    }
}

/// Wrapper: update statistics and refresh ER caches for affected ranges
pub(super) fn reduce_grid_statistics_and_refresh_caches<'a>(
    mut state: FittingState<'a>,
    col: usize,
    original_affected_interval: &(usize, usize),
    affected_ranges: Option<&[AffectedRange]>,
    refinement: &RefinementStrategy,
    candidate_refresh: CandidateRefresh,
) -> FittingState<'a> {
    let &(start, end) = original_affected_interval;
    let Some(affected_ranges) = affected_ranges else {
        assert_eq!(candidate_refresh, CandidateRefresh::Deferred);
        return refinement.update_binned_curvature_statistics(state, col, start, end);
    };
    // Convert AffectedRange to point ranges for update_statistics
    // update_statistics expects a Vec indexed by column number, so we need to build
    // a full vector with empty ranges (1, 0) for unaffected columns
    let mut affected_points_range = vec![(1usize, 0usize); state.p];
    for affected_range in affected_ranges {
        affected_points_range[affected_range.col] = affected_range.point_range;
    }
    state = if candidate_refresh == CandidateRefresh::Deferred
        && matches!(refinement, RefinementStrategy::L2Refinement { .. })
        && state
            .branch_consensus
            .as_ref()
            .is_some_and(|reference| reference.anchors.is_none())
    {
        refinement.update_curvature_statistics(state, col, start, end, &affected_points_range)
    } else {
        refinement.update_statistics(state, col, start, end, &affected_points_range)
    };
    if candidate_refresh == CandidateRefresh::Eager {
        state = refinement.refresh_error_reduction_caches_for_affected(state, affected_ranges);
    }
    state
}
