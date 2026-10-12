//! Applies a `FittingAction` to the `FittingState` and dispatches each structural action.

mod caches;
mod merge;
mod resplit;
mod split;

#[cfg(feature = "evo-logging")]
use crate::grid_tensor::logging_helpers::create_simple_split_event;
use crate::grid_tensor::{
    action::FittingAction,
    axis_factors,
    refinement::RefinementStrategy,
    splitting::SplitStrategy,
    state::FittingState,
    two_tensor_solver::{
        convert_multipliers_to_backbone_tilt, MAX_BACKBONE, MAX_TILT, MIN_BACKBONE,
    },
};
#[cfg(feature = "evo-logging")]
use crate::logging::reducer::log_split_reducer;

/// Whether candidate scores are needed before the next full statistics refresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CandidateRefresh {
    Eager,
    /// Requires `refresh_statistics` before another action is proposed.
    Deferred,
}

/// How a structural action updates the statistics of the rows it moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StatisticsPlan {
    /// Row and prefix statistics, then the candidate scores of the affected ranges.
    Eager,
    /// Row and prefix statistics; candidate scores wait for the next full refresh.
    Deferred,
    /// Curvature statistics only, over each column's affected range.
    Curvature,
    /// Curvature statistics only, scattered into the binned prefix sums of every column.
    BinnedCurvature,
}

impl StatisticsPlan {
    /// Deferred L2 fits pulled toward an unanchored consensus rebuild their linear terms on the
    /// next full refresh, so an action only keeps their curvature current.
    fn new(
        state: &FittingState,
        refinement: &RefinementStrategy,
        candidate_refresh: CandidateRefresh,
    ) -> Self {
        if candidate_refresh == CandidateRefresh::Eager {
            return Self::Eager;
        }
        let curvature_only = matches!(refinement, RefinementStrategy::L2Refinement { .. })
            && state
                .branch_consensus
                .as_ref()
                .is_some_and(|reference| reference.anchors.is_none());
        if !curvature_only {
            Self::Deferred
        } else if state
            .precomputed_statistics
            .bin_edges
            .iter()
            .all(|edges| !edges.is_empty())
        {
            Self::BinnedCurvature
        } else {
            Self::Curvature
        }
    }
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
    // Untouched intervals retain scores computed before prefix-tail shifts, so an eager
    // action materializes pending scores before it permits another local proposal.
    if candidate_refresh == CandidateRefresh::Eager && state.split_scores_stale {
        refinement_strategy.refresh_all_split_scores(&mut state);
    }

    #[cfg(feature = "evo-logging")]
    let logging_event = state
        .logging_state
        .as_ref()
        .map(|_| (action.clone(), state.current_error));
    let plan = StatisticsPlan::new(&state, refinement_strategy, candidate_refresh);

    let mut state = match action {
        FittingAction::ApplySplit { split } => {
            split::apply(state, split, refinement_strategy, split_strategy, plan)
        }
        FittingAction::ApplyResplit { resplit } => {
            resplit::apply(state, resplit, refinement_strategy, plan)
        }
        FittingAction::ApplyMerge { merge } => {
            merge::apply(state, merge, refinement_strategy, plan)
        }
    };

    #[cfg(feature = "evo-logging")]
    if let Some((action_for_logging, old_error)) = logging_event {
        let split_event = create_simple_split_event(&state, &action_for_logging, old_error);
        state.logging_state = log_split_reducer(state.logging_state.take(), split_event);
    }

    state
}

/// Axis factors `(b·e^d, b·e^{−d})` of intervals `interval_idx` and `interval_idx + 1` on `col`.
fn adjacent_factors(state: &FittingState, col: usize, interval_idx: usize) -> [(f64, f64); 2] {
    [interval_idx, interval_idx + 1]
        .map(|k| axis_factors(state.backbone_values[col][k], state.tilt_values[col][k]))
}

/// Backbone multiplier and tilt shift `(v_b, Δd)` of each side of a split or resplit.
///
/// A side's update `u_±` becomes the multiplier `v_± = 1 + u_±`, clamped to the refinement's
/// bounds. Stage 1 (positive-only) grids take `v_b = v_+` and keep a zero tilt.
fn side_multipliers(
    state: &FittingState,
    refinement: &RefinementStrategy,
    updates: [(f64, f64); 2],
) -> [(f64, f64); 2] {
    let (v_min, v_max) = refinement.multiplier_bounds();
    let stage1 = state.is_stage1_positive_only();
    updates.map(|(u_plus, u_minus)| {
        let v_plus = (1.0 + u_plus).clamp(v_min, v_max);
        if stage1 {
            (v_plus, 0.0)
        } else {
            convert_multipliers_to_backbone_tilt(v_plus, (1.0 + u_minus).clamp(v_min, v_max))
        }
    })
}

/// Parameters `(b·v_b, d + Δd)` clamped to the grid's backbone and tilt bounds.
fn scaled_parameters(backbone: f64, tilt: f64, (v_b, delta_d): (f64, f64)) -> (f64, f64) {
    (
        (backbone * v_b).clamp(MIN_BACKBONE, MAX_BACKBONE),
        (tilt + delta_d).clamp(-MAX_TILT, MAX_TILT),
    )
}
