//! Split: inserts a boundary inside one interval, rescales its rows and forbids nearby splits.

use super::{
    adjacent_factors,
    caches::{refresh_action_statistics, rescale_branches},
    scaled_parameters, side_multipliers, StatisticsPlan,
};
use crate::grid_tensor::{
    axis_factors,
    refinement::RefinementStrategy,
    splitting::{SplitCandidate, SplitStrategy},
    state::{FittingState, IntervalStats},
};

/// Split the interval holding sorted position `split.index` on `split.col` at that position.
pub(super) fn apply<'a>(
    mut state: FittingState<'a>,
    split: SplitCandidate,
    refinement: &RefinementStrategy,
    split_strategy: &SplitStrategy,
    plan: StatisticsPlan,
) -> FittingState<'a> {
    let col = split.col;
    let interval_idx = state.boundaries[col].partition_point(|&b| b <= split.index);
    let (start, end) = state.interval_range(col, interval_idx);
    let multipliers = side_multipliers(&state, refinement, [split.update_left, split.update_right]);
    let parent = axis_factors(
        state.backbone_values[col][interval_idx],
        state.tilt_values[col][interval_idx],
    );

    insert_boundary(&mut state, col, interval_idx, split.index, multipliers);
    let children = adjacent_factors(&state, col, interval_idx);
    rescale_branches(
        &mut state,
        col,
        (start, split.index, end),
        [parent, parent],
        children,
    );

    state = refresh_action_statistics(state, col, (start, end), plan, refinement);
    state = split_strategy.forbid_around_split(state, &split);
    state.current_error = refinement.weighted_squared_error(&state.residuals, state.sample_weight);
    state.loop_state.record_split();
    state
}

/// Insert boundary `index` as boundary `interval_idx` on `col`.
///
/// Each child takes the parent's parameters scaled by its `(v_b, Δd)`, and the new boundary's
/// resplit and merge statistics start as NaN placeholders.
fn insert_boundary(
    state: &mut FittingState<'_>,
    col: usize,
    interval_idx: usize,
    index: usize,
    [left, right]: [(f64, f64); 2],
) {
    state.boundaries[col].insert(interval_idx, index);

    let b_p = state.backbone_values[col][interval_idx];
    let d_p = state.tilt_values[col][interval_idx];
    let (b_l, d_l) = scaled_parameters(b_p, d_p, left);
    let (b_r, d_r) = scaled_parameters(b_p, d_p, right);
    state.backbone_values[col][interval_idx] = b_l;
    state.tilt_values[col][interval_idx] = d_l;
    state.backbone_values[col].insert(interval_idx + 1, b_r);
    state.tilt_values[col].insert(interval_idx + 1, d_r);

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

    let statistics = &mut state.precomputed_statistics;
    statistics.error_reductions_resplit[col].insert(interval_idx, f64::NAN);
    statistics.update_pairs_resplit_left[col].insert(interval_idx, (f64::NAN, f64::NAN));
    statistics.update_pairs_resplit_right[col].insert(interval_idx, (f64::NAN, f64::NAN));
    statistics.error_reductions_resplit_pairs[col].insert(interval_idx, (f64::NAN, f64::NAN));
    statistics.error_reductions_merge[col].insert(interval_idx, f64::NAN);
    statistics.update_pairs_merge[col].insert(interval_idx, (f64::NAN, f64::NAN));
    // Placeholder for the right child's interval statistics.
    statistics.interval_stats[col].insert(interval_idx + 1, IntervalStats::default());
}
