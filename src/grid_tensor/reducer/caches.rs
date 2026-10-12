//! Refreshes the row prediction caches and candidate statistics a structural action invalidates.

use super::StatisticsPlan;
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

/// Rescale the branch predictions of the sorted rows `start..end` on `col` by the new-to-old
/// ratio of their side's axis factors: rows before `index` are the left side.
pub(super) fn rescale_branches(
    state: &mut FittingState<'_>,
    col: usize,
    (start, index, end): (usize, usize, usize),
    old_factors: [(f64, f64); 2],
    new_factors: [(f64, f64); 2],
) {
    let [left, right] = [0, 1].map(|side| {
        (
            new_factors[side].0 / old_factors[side].0,
            new_factors[side].1 / old_factors[side].1,
        )
    });
    update_prediction_caches(state, col, start, index, |_, plus, minus| {
        (plus * left.0, minus * left.1)
    });
    update_prediction_caches(state, col, index, end, |_, plus, minus| {
        (plus * right.0, minus * right.1)
    });
}

/// Update the statistics of the sorted rows `start..end` on `col` as `plan` directs.
pub(super) fn refresh_action_statistics<'a>(
    state: FittingState<'a>,
    col: usize,
    (start, end): (usize, usize),
    plan: StatisticsPlan,
    refinement: &RefinementStrategy,
) -> FittingState<'a> {
    match plan {
        StatisticsPlan::BinnedCurvature => {
            refinement.update_binned_curvature_statistics(state, col, start, end)
        }
        StatisticsPlan::Curvature => {
            let (_, points) = affected_ranges(&state, col, (start, end));
            refinement.update_curvature_statistics(state, col, start, end, &points)
        }
        StatisticsPlan::Deferred => {
            let (_, points) = affected_ranges(&state, col, (start, end));
            refinement.update_statistics(state, col, start, end, &points)
        }
        StatisticsPlan::Eager => {
            let (ranges, points) = affected_ranges(&state, col, (start, end));
            let state = refinement.update_statistics(state, col, start, end, &points);
            refinement.refresh_error_reduction_caches_for_affected(state, &ranges)
        }
    }
}

/// The affected ranges of every column, and their point ranges indexed by column with the
/// empty range `(1, 0)` for columns the action leaves untouched.
fn affected_ranges(
    state: &FittingState,
    col: usize,
    interval: (usize, usize),
) -> (Vec<AffectedRange>, Vec<(usize, usize)>) {
    let ranges = state.get_affected_ranges(col, interval);
    let mut points = vec![(1usize, 0usize); state.p];
    for range in &ranges {
        points[range.col] = range.point_range;
    }
    (ranges, points)
}
