//! Merge and resplit candidate scans over the existing grid boundaries.

use crate::grid_tensor::{
    action::FittingAction,
    state::{FittingState, MAX_CONSECUTIVE_RESPLIT},
};

use super::{MergeCandidate, ResplitCandidate, SplitStrategy};

impl SplitStrategy {
    pub(super) fn propose_best_merge(&self, state: &FittingState) -> Option<MergeCandidate> {
        if !state.split_strategy_state.merge_enabled {
            return None;
        }

        let mut best: Option<MergeCandidate> = None;
        for (col, ((raw_gains, merged_factors), boundaries)) in state
            .precomputed_statistics
            .error_reductions_merge
            .iter()
            .zip(state.precomputed_statistics.update_pairs_merge.iter())
            .zip(state.boundaries.iter())
            .enumerate()
        {
            for (interval_idx, ((&error_reduction, &factors), &index)) in raw_gains
                .iter()
                .zip(merged_factors.iter())
                .zip(boundaries.iter())
                .enumerate()
            {
                if !error_reduction.is_finite()
                    || !factors.0.is_finite()
                    || !factors.1.is_finite()
                    || factors.0 <= 0.0
                    || factors.1 <= 0.0
                {
                    continue;
                }

                if best
                    .as_ref()
                    .is_none_or(|candidate| error_reduction > candidate.error_reduction)
                {
                    best = Some(MergeCandidate {
                        col,
                        error_reduction,
                        interval_idx,
                        index,
                    });
                }
            }
        }
        best
    }

    pub(super) fn propose_best_resplit(&self, state: &FittingState) -> Option<ResplitCandidate> {
        if !state.split_strategy_state.resplit_enabled {
            return None;
        }
        // Prevent too many consecutive resplits to avoid infinite loops
        if state.loop_state.consecutive_resplits >= MAX_CONSECUTIVE_RESPLIT {
            return None;
        }
        let mut best_resplit_candidate: Option<ResplitCandidate> = None;
        for (col_idx, (err_reductions, boundaries)) in state
            .precomputed_statistics
            .error_reductions_resplit
            .iter()
            .zip(state.boundaries.iter())
            .enumerate()
        {
            for (index, (&err_reduction, &data_index)) in
                err_reductions.iter().zip(boundaries.iter()).enumerate()
            {
                // Skip NaN values to prevent panic in partial_cmp
                if err_reduction.is_nan() {
                    continue;
                }
                if let Some(last_transformation) =
                    state.split_strategy_state.last_transformation.as_ref()
                {
                    match last_transformation {
                        FittingAction::ApplySplit { split } => {
                            if col_idx == split.col && data_index == split.index {
                                continue;
                            }
                        }
                        FittingAction::ApplyResplit { resplit } => {
                            if col_idx == resplit.col && data_index == resplit.index {
                                continue;
                            }
                        }
                        _ => {}
                    }
                }
                if best_resplit_candidate
                    .as_ref()
                    .is_none_or(|sc| err_reduction > sc.error_reduction)
                {
                    let update_left =
                        state.precomputed_statistics.update_pairs_resplit_left[col_idx][index];
                    let update_right =
                        state.precomputed_statistics.update_pairs_resplit_right[col_idx][index];
                    let candidate = ResplitCandidate {
                        col: col_idx,
                        index: data_index,
                        error_reduction: err_reduction,
                        interval_idx: index,
                        update_left,
                        update_right,
                    };
                    best_resplit_candidate = Some(candidate);
                }
            }
        }
        best_resplit_candidate
    }
}
