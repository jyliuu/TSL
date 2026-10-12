//! Applies a `FittingAction` to the `FittingState` and dispatches each structural action.

mod caches;
mod control;
mod merge;
mod resplit;
mod split;

pub(crate) use merge::compute_partial_products_for_axis;

#[cfg(feature = "evo-logging")]
use crate::grid_tensor::logging_helpers::create_simple_split_event;
use crate::grid_tensor::{
    action::FittingAction, refinement::RefinementStrategy, splitting::SplitStrategy,
    state::FittingState,
};
#[cfg(feature = "evo-logging")]
use crate::logging::reducer::log_split_reducer;

use caches::reduce_grid_statistics_and_refresh_caches;
use control::{
    reduce_apply_scaling, reduce_composite, reduce_loop_state_for_merge,
    reduce_loop_state_for_resplit, reduce_loop_state_for_split, reduce_terminate,
};
use merge::{
    reduce_grid_merge_optimal, reduce_grid_predictions_and_residuals_merge_from_partial_products,
};
use resplit::{
    reduce_grid_predictions_and_residuals_resplit_two_tensor_factors,
    reduce_grid_resplit_two_tensor,
};
use split::{
    reduce_forbid_around_split, reduce_grid_predictions_and_residuals_two_tensor_factors,
    reduce_grid_split_two_tensor,
};

/// Whether candidate scores are needed before the next full statistics refresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CandidateRefresh {
    Eager,
    /// Requires `refresh_statistics` before another action is proposed.
    Deferred,
}

/// Central reducer function that consumes state and action
/// Returns the new state (using move semantics, no cloning)
pub fn fitting_reducer<'a>(
    state: FittingState<'a>,
    action: FittingAction,
    refinement_strategy: &RefinementStrategy,
    split_strategy: &SplitStrategy,
) -> FittingState<'a> {
    fitting_reducer_with_candidate_refresh(
        state,
        action,
        refinement_strategy,
        split_strategy,
        CandidateRefresh::Eager,
    )
}

pub(crate) fn fitting_reducer_with_candidate_refresh<'a>(
    mut state: FittingState<'a>,
    action: FittingAction,
    refinement_strategy: &RefinementStrategy,
    split_strategy: &SplitStrategy,
    candidate_refresh: CandidateRefresh,
) -> FittingState<'a> {
    state.require_current_statistics();
    let sampled_split_scoring = state.sampled_split_scoring;
    let eager = candidate_refresh == CandidateRefresh::Eager
        || matches!(&action, FittingAction::Composite { .. });
    if eager {
        // Untouched intervals retain scores computed before prefix-tail shifts.
        // Materialize pending scores before an action that permits another local proposal.
        if state.split_scores_stale {
            refinement_strategy.refresh_all_split_scores(&mut state);
        }
        state.sampled_split_scoring = false;
    }
    let mut state = match action {
        FittingAction::ApplyScaling { factor } => reduce_apply_scaling(state, factor),
        FittingAction::Terminate { reason } => reduce_terminate(state, reason),
        FittingAction::Composite { actions } => {
            // Consecutive subactions can consume the candidate caches.
            reduce_composite(state, actions, refinement_strategy, split_strategy)
        }
        FittingAction::ApplyMerge { .. }
        | FittingAction::ApplyResplit { .. }
        | FittingAction::ApplySplit { .. } => reduce_apply_transformation(
            state,
            action,
            refinement_strategy,
            split_strategy,
            candidate_refresh,
        ),
    };
    state.sampled_split_scoring = sampled_split_scoring;
    state
}

/// Apply a split/resplit/merge with all associated side effects
/// This combines: update_tree + forbid_around_split + update_statistics
fn reduce_apply_transformation<'a>(
    mut state: FittingState<'a>,
    transformation: FittingAction,
    refinement_strategy: &RefinementStrategy,
    split_strategy: &SplitStrategy,
    candidate_refresh: CandidateRefresh,
) -> FittingState<'a> {
    #[cfg(feature = "evo-logging")]
    let logging_event = state
        .logging_state
        .as_ref()
        .map(|_| (transformation.clone(), state.current_error));
    let fused_binned_curvature = candidate_refresh == CandidateRefresh::Deferred
        && matches!(refinement_strategy, RefinementStrategy::L2Refinement { .. })
        && state
            .branch_consensus
            .as_ref()
            .is_some_and(|reference| reference.anchors.is_none())
        && state
            .precomputed_statistics
            .bin_edges
            .iter()
            .all(|edges| !edges.is_empty());

    // Step 1: Update tree structure (insert/remove/update boundaries and grid_values)

    state = match transformation {
        FittingAction::ApplySplit {
            split: split_candidate,
        } => {
            let idx = split_candidate.index;
            let col = split_candidate.col;
            let interval_idx = state.boundaries[col].partition_point(|&b| b <= idx);
            let original_affected_interval = state.interval_range(col, interval_idx);
            let (u_plus_l, u_minus_l) = split_candidate.update_left;
            let (u_plus_r, u_minus_r) = split_candidate.update_right;

            let (v_min, v_max) = refinement_strategy.multiplier_bounds();

            let is_stage1 = state.is_stage1_positive_only();
            let (v_b_l, delta_d_l, v_b_r, delta_d_r) = if is_stage1 {
                // Stage 1 positive-only: v_b = 1 + u directly, Δd = 0
                let v_b_l = (1.0 + u_plus_l).clamp(v_min, v_max);
                let v_b_r = (1.0 + u_plus_r).clamp(v_min, v_max);
                (v_b_l, 0.0, v_b_r, 0.0)
            } else {
                // Full two-tensor: convert (v_+, v_-) to (v_b, Δd)
                let v_plus_l = (1.0 + u_plus_l).clamp(v_min, v_max);
                let v_minus_l = (1.0 + u_minus_l).clamp(v_min, v_max);
                let v_plus_r = (1.0 + u_plus_r).clamp(v_min, v_max);
                let v_minus_r = (1.0 + u_minus_r).clamp(v_min, v_max);

                let (v_b_l, delta_d_l) =
                    crate::grid_tensor::two_tensor_solver::convert_multipliers_to_backbone_tilt(
                        v_plus_l, v_minus_l,
                    );
                let (v_b_r, delta_d_r) =
                    crate::grid_tensor::two_tensor_solver::convert_multipliers_to_backbone_tilt(
                        v_plus_r, v_minus_r,
                    );
                (v_b_l, delta_d_l, v_b_r, delta_d_r)
            };

            // Capture parent (old) axis factors before mutating parameters.
            let insert_at = state.boundaries[col].partition_point(|&b| b <= idx);
            let b_p = state.backbone_values[col][insert_at];
            let d_p = state.tilt_values[col][insert_at];
            let g_plus_old = b_p * d_p.exp();
            let g_minus_old = b_p * (-d_p).exp();

            state =
                reduce_grid_split_two_tensor(state, v_b_l, delta_d_l, v_b_r, delta_d_r, col, idx);
            let affected_ranges = (!fused_binned_curvature)
                .then(|| state.get_affected_ranges(col, original_affected_interval));

            // Compute new axis factors after split using the *clamped* child parameters stored in state.
            let b_l = state.backbone_values[col][insert_at];
            let d_l = state.tilt_values[col][insert_at];
            let g_plus_new_l = b_l * d_l.exp();
            let g_minus_new_l = b_l * (-d_l).exp();
            let b_r = state.backbone_values[col][insert_at + 1];
            let d_r = state.tilt_values[col][insert_at + 1];
            let g_plus_new_r = b_r * d_r.exp();
            let g_minus_new_r = b_r * (-d_r).exp();

            state = reduce_grid_predictions_and_residuals_two_tensor_factors(
                state,
                col,
                idx,
                &original_affected_interval,
                g_plus_old,
                g_minus_old,
                g_plus_new_l,
                g_minus_new_l,
                g_plus_new_r,
                g_minus_new_r,
            );

            state = reduce_grid_statistics_and_refresh_caches(
                state,
                col,
                &original_affected_interval,
                affected_ranges.as_deref(),
                refinement_strategy,
                candidate_refresh,
            );

            state = reduce_forbid_around_split(state, &split_candidate, split_strategy);
            state.current_error = refinement_strategy.weighted_squared_error(&state.residuals);
            state = reduce_loop_state_for_split(state, &split_candidate);
            state
        }
        FittingAction::ApplyMerge {
            merge: merge_candidate,
        } => {
            let col = merge_candidate.col;
            let interval_idx = merge_candidate.interval_idx;
            let interval_left_and_right = state.interval_range_left_and_right(col, interval_idx);
            let original_affected_interval = (interval_left_and_right.0, interval_left_and_right.2);
            let (partial_plus, partial_minus) = compute_partial_products_for_axis(col, &state);
            let (a_plus_merged, a_minus_merged) =
                state.precomputed_statistics.update_pairs_merge[col][interval_idx];
            let b_merged_optimal = (a_plus_merged * a_minus_merged).sqrt();
            let d_merged_optimal = 0.5 * (a_plus_merged / a_minus_merged).ln();

            state = reduce_grid_merge_optimal(
                state,
                col,
                interval_idx,
                b_merged_optimal,
                d_merged_optimal,
            );
            let affected_ranges = (!fused_binned_curvature)
                .then(|| state.get_affected_ranges(col, original_affected_interval));

            let b_merged = state.backbone_values[col][interval_idx];
            let d_merged = state.tilt_values[col][interval_idx];
            let g_plus_new = b_merged * d_merged.exp();
            let g_minus_new = b_merged * (-d_merged).exp();

            state = reduce_grid_predictions_and_residuals_merge_from_partial_products(
                state,
                col,
                &original_affected_interval,
                &partial_plus,
                &partial_minus,
                g_plus_new,
                g_minus_new,
            );

            state = reduce_grid_statistics_and_refresh_caches(
                state,
                col,
                &original_affected_interval,
                affected_ranges.as_deref(),
                refinement_strategy,
                candidate_refresh,
            );

            state.current_error = refinement_strategy.weighted_squared_error(&state.residuals);

            state = reduce_loop_state_for_merge(state, &merge_candidate);
            state
        }
        FittingAction::ApplyResplit {
            resplit: resplit_candidate,
        } => {
            // Resplit doesn't change boundaries; no map updates needed
            let col = resplit_candidate.col;
            let interval_idx = resplit_candidate.interval_idx;
            let index = resplit_candidate.index;
            let interval_left_and_right = state.interval_range_left_and_right(col, interval_idx);
            let original_affected_interval = (interval_left_and_right.0, interval_left_and_right.2);
            // Convert two-tensor updates to (v_b, delta_d) for each side
            let (u_plus_l, u_minus_l) = resplit_candidate.update_left;
            let (u_plus_r, u_minus_r) = resplit_candidate.update_right;
            let (v_min, v_max) = refinement_strategy.multiplier_bounds();

            let is_stage1 = state.is_stage1_positive_only();
            let (v_b_l, delta_d_l, v_b_r, delta_d_r) = if is_stage1 {
                // Stage 1 positive-only: v_b = 1 + u directly, Δd = 0
                let v_b_l = (1.0 + u_plus_l).clamp(v_min, v_max);
                let v_b_r = (1.0 + u_plus_r).clamp(v_min, v_max);
                (v_b_l, 0.0, v_b_r, 0.0)
            } else {
                // Full two-tensor: convert (v_+, v_-) to (v_b, Δd)
                let v_plus_l = (1.0 + u_plus_l).clamp(v_min, v_max);
                let v_minus_l = (1.0 + u_minus_l).clamp(v_min, v_max);
                let v_plus_r = (1.0 + u_plus_r).clamp(v_min, v_max);
                let v_minus_r = (1.0 + u_minus_r).clamp(v_min, v_max);

                let (v_b_l, delta_d_l) =
                    crate::grid_tensor::two_tensor_solver::convert_multipliers_to_backbone_tilt(
                        v_plus_l, v_minus_l,
                    );
                let (v_b_r, delta_d_r) =
                    crate::grid_tensor::two_tensor_solver::convert_multipliers_to_backbone_tilt(
                        v_plus_r, v_minus_r,
                    );
                (v_b_l, delta_d_l, v_b_r, delta_d_r)
            };

            // Capture old axis factors for both intervals before updating.
            let b_left_old = state.backbone_values[col][interval_idx];
            let d_left_old = state.tilt_values[col][interval_idx];
            let g_plus_old_l = b_left_old * d_left_old.exp();
            let g_minus_old_l = b_left_old * (-d_left_old).exp();
            let b_right_old = state.backbone_values[col][interval_idx + 1];
            let d_right_old = state.tilt_values[col][interval_idx + 1];
            let g_plus_old_r = b_right_old * d_right_old.exp();
            let g_minus_old_r = b_right_old * (-d_right_old).exp();

            state = reduce_grid_resplit_two_tensor(
                state,
                v_b_l,
                delta_d_l,
                v_b_r,
                delta_d_r,
                col,
                interval_idx,
            );
            let affected_ranges = (!fused_binned_curvature)
                .then(|| state.get_affected_ranges(col, original_affected_interval));

            // Compute new factors (after param update) for both intervals using stored (clamped) params.
            let b_left_new = state.backbone_values[col][interval_idx];
            let d_left_new = state.tilt_values[col][interval_idx];
            let g_plus_new_l = b_left_new * d_left_new.exp();
            let g_minus_new_l = b_left_new * (-d_left_new).exp();
            let b_right_new = state.backbone_values[col][interval_idx + 1];
            let d_right_new = state.tilt_values[col][interval_idx + 1];
            let g_plus_new_r = b_right_new * d_right_new.exp();
            let g_minus_new_r = b_right_new * (-d_right_new).exp();

            state = reduce_grid_predictions_and_residuals_resplit_two_tensor_factors(
                state,
                col,
                index,
                &original_affected_interval,
                g_plus_old_l,
                g_minus_old_l,
                g_plus_old_r,
                g_minus_old_r,
                g_plus_new_l,
                g_minus_new_l,
                g_plus_new_r,
                g_minus_new_r,
            );

            state = reduce_grid_statistics_and_refresh_caches(
                state,
                col,
                &original_affected_interval,
                affected_ranges.as_deref(),
                refinement_strategy,
                candidate_refresh,
            );

            state.current_error = refinement_strategy.weighted_squared_error(&state.residuals);
            state = reduce_loop_state_for_resplit(state, &resplit_candidate);
            state
        }
        _ => unreachable!(),
    };

    #[cfg(feature = "evo-logging")]
    if let Some((action_for_logging, old_error)) = logging_event {
        let split_event = create_simple_split_event(&state, &action_for_logging, old_error);
        state.logging_state = log_split_reducer(state.logging_state.take(), split_event);
    }

    state
}
