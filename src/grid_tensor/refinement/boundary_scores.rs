//! Resplit and merge scores for the boundaries between adjacent intervals.

use super::gain::{
    canonical_merge_factors, huber_weighting, interval_solver_stats, SplitEvaluator,
};
use super::RefinementStrategy;
use crate::grid_tensor::axis_factors;
use crate::grid_tensor::state::{BranchConsensus, FittingState};
use crate::grid_tensor::two_tensor_solver::{data_loss_gain, log_coordinate_penalty};

impl RefinementStrategy {
    /// Rescore boundaries `lo_boundary_idx..=hi_boundary_idx` of `col` by refitting the two
    /// intervals around each one.
    pub(super) fn update_error_reductions_resplit_for_col_range(
        state: &mut FittingState,
        evaluator: &SplitEvaluator,
        col: usize,
        lo_boundary_idx: usize,
        hi_boundary_idx: usize,
    ) {
        let n_boundaries = state.boundaries[col].len();
        if n_boundaries == 0 {
            return;
        }
        let statistics = &mut state.precomputed_statistics;
        debug_assert!(
            [
                statistics.error_reductions_resplit[col].len(),
                statistics.update_pairs_resplit_left[col].len(),
                statistics.update_pairs_resplit_right[col].len(),
                statistics.error_reductions_resplit_pairs[col].len(),
            ]
            .iter()
            .all(|&len| len == n_boundaries),
            "resplit caches hold one entry per boundary"
        );

        for i in lo_boundary_idx..=hi_boundary_idx {
            let intervals = &statistics.interval_stats[col];
            let (left, gain_l) = evaluator.evaluate_side(interval_solver_stats(&intervals[i]));
            let (right, gain_r) = evaluator.evaluate_side(interval_solver_stats(&intervals[i + 1]));
            statistics.error_reductions_resplit[col][i] = gain_l + gain_r;
            statistics.update_pairs_resplit_left[col][i] = left;
            statistics.update_pairs_resplit_right[col][i] = right;
            statistics.error_reductions_resplit_pairs[col][i] = (gain_l, gain_r);
        }
    }

    pub(super) fn update_error_reductions_merge_for_col_range(
        &self,
        state: &mut FittingState,
        evaluator: &SplitEvaluator,
        col: usize,
        lo_boundary_idx: usize,
        hi_boundary_idx: usize,
    ) {
        if state.boundaries[col].is_empty() {
            return;
        }

        let sorted_indices = &state.precomputed_statistics.sorted_indices[col];

        for i in lo_boundary_idx..=hi_boundary_idx {
            let (start, index, end) = state.interval_range_left_and_right(col, i);
            let left_count = index - start;
            let right_count = end - index;
            let total_count = (left_count + right_count) as f64;

            let interval_factors = |interval_idx: usize| {
                axis_factors(
                    state.backbone_values[col][interval_idx],
                    state.tilt_values[col][interval_idx],
                )
            };
            let left_factors = interval_factors(i);
            let right_factors = interval_factors(i + 1);
            let log_reference = |left: f64, right: f64| {
                ((left_count as f64 * left.ln() + right_count as f64 * right.ln()) / total_count)
                    .exp()
            };
            let reference = (
                log_reference(left_factors.0, right_factors.0),
                log_reference(left_factors.1, right_factors.1),
            );

            let segments = [
                (&sorted_indices[start..index], left_factors),
                (&sorted_indices[index..end], right_factors),
            ];
            let MergeSums {
                s11,
                s22,
                s12,
                t1,
                t2,
                current_loss,
                reference_loss,
            } = self.merge_sums(state, &segments, reference);

            let merged_factors = if evaluator.is_stage1() {
                let multiplier = evaluator.backbone_multiplier(t1, s11);
                canonical_merge_factors(reference.0 * multiplier, reference.0 * multiplier, true)
            } else {
                let (u_plus, u_minus, _) = evaluator.solve(s11, s22, s12, t1, t2);
                let v_plus = 1.0 + u_plus;
                let v_minus = 1.0 + u_minus;
                canonical_merge_factors(reference.0 * v_plus, reference.1 * v_minus, false)
            };
            let parameter_penalty = log_coordinate_penalty(
                merged_factors.0 / reference.0,
                merged_factors.1 / reference.1,
                self.alpha(),
                self.tilt_tau(),
                self.tilt_rho(),
            );
            let u_plus = merged_factors.0 / reference.0 - 1.0;
            let u_minus = merged_factors.1 / reference.1 - 1.0;
            // The reference-centered quadratic gives the merged loss without a second row scan.
            let reference_to_merged_gain = data_loss_gain(u_plus, u_minus, s11, s22, s12, t1, t2);
            let merged_loss = reference_loss - reference_to_merged_gain;

            let merge_gain = current_loss - merged_loss - parameter_penalty;
            if merge_gain.is_finite()
                && merged_factors.0.is_finite()
                && merged_factors.1.is_finite()
            {
                state.precomputed_statistics.error_reductions_merge[col][i] = merge_gain;
                state.precomputed_statistics.update_pairs_merge[col][i] = merged_factors;
            } else {
                state.precomputed_statistics.error_reductions_merge[col][i] = f64::NAN;
                state.precomputed_statistics.update_pairs_merge[col][i] = (f64::NAN, f64::NAN);
            }
        }
    }

    /// [`MergeSums::accumulate`] with the loss weights and consensus targets resolved once
    /// per boundary rather than per row.
    fn merge_sums(
        &self,
        state: &FittingState,
        segments: &MergeSegments,
        reference: (f64, f64),
    ) -> MergeSums {
        let consensus = state.branch_consensus.as_ref();
        let (Some(f_plus), Some(f_minus), Some(residuals), Some(labels), None) = (
            state.f_plus.as_slice(),
            state.f_minus.as_slice(),
            state.residuals.as_slice(),
            state.labels.as_slice(),
            state.sample_weight,
        ) else {
            return MergeSums::accumulate(
                segments,
                reference,
                |row| {
                    (
                        state.f_plus[row],
                        state.f_minus[row],
                        state.residuals[row],
                        state.labels[row],
                    )
                },
                |row, residual| state.sample_weight(row) * self.weight(residual),
                |row| {
                    consensus.and_then(|consensus| consensus.target(row)).map(
                        |(rho, target_plus, target_minus)| {
                            (rho * state.sample_weight(row), target_plus, target_minus)
                        },
                    )
                },
            );
        };
        let values = |row: usize| (f_plus[row], f_minus[row], residuals[row], labels[row]);
        match *self {
            Self::L2Refinement { .. } => MergeSums::accumulate_with_consensus(
                segments,
                reference,
                values,
                |_, _| 1.0,
                consensus,
            ),
            Self::HuberRefinement { c, .. } => MergeSums::accumulate_with_consensus(
                segments,
                reference,
                values,
                |_, residual| huber_weighting(residual, c),
                consensus,
            ),
        }
    }
}

/// Rows of the two intervals around a boundary, each with its interval's axis factors.
type MergeSegments<'r> = [(&'r [usize], (f64, f64)); 2];

/// Row sums of the reference-centred merge quadratic and of the current and reference
/// losses over the rows of two adjacent intervals.
#[derive(Default)]
struct MergeSums {
    s11: f64,
    s22: f64,
    s12: f64,
    t1: f64,
    t2: f64,
    current_loss: f64,
    reference_loss: f64,
}

impl MergeSums {
    /// [`Self::accumulate`] with the consensus targets read from contiguous full-row
    /// references when there are no anchors.
    #[inline(always)]
    fn accumulate_with_consensus(
        segments: &MergeSegments,
        reference: (f64, f64),
        values: impl Fn(usize) -> (f64, f64, f64, f64),
        weight: impl Fn(usize, f64) -> f64,
        consensus: Option<&BranchConsensus>,
    ) -> Self {
        let Some(consensus) = consensus else {
            return Self::accumulate(segments, reference, values, weight, |_| None);
        };
        match (
            &consensus.anchors,
            consensus.plus.as_slice(),
            consensus.minus.as_slice(),
        ) {
            (None, Some(plus), Some(minus)) => {
                let rho = consensus.weight;
                Self::accumulate(segments, reference, values, weight, |row| {
                    Some((rho, plus[row], minus[row]))
                })
            }
            _ => Self::accumulate(segments, reference, values, weight, |row| {
                consensus.target(row)
            }),
        }
    }

    /// Sum both intervals' rows in order. `values(row)` is `(f₊, f₋, residual, label)`,
    /// `weight(row, residual)` is the row's loss weight and `target(row)` gives the row's
    /// consensus weight and branch targets.
    #[inline(always)]
    fn accumulate(
        segments: &MergeSegments,
        reference: (f64, f64),
        values: impl Fn(usize) -> (f64, f64, f64, f64),
        weight: impl Fn(usize, f64) -> f64,
        target: impl Fn(usize) -> Option<(f64, f64, f64)>,
    ) -> Self {
        let mut sums = Self::default();
        for &(rows, factors) in segments {
            for &row in rows {
                let (f_plus, f_minus, residual, label) = values(row);
                let weight = weight(row, residual);
                let partial_plus = f_plus / factors.0;
                let partial_minus = f_minus / factors.1;
                let phi_plus = partial_plus * reference.0;
                let phi_minus = -partial_minus * reference.1;
                let reference_residual = label - phi_plus - phi_minus;
                sums.s11 += weight * phi_plus * phi_plus;
                sums.s22 += weight * phi_minus * phi_minus;
                sums.s12 += weight * phi_plus * phi_minus;
                sums.t1 += weight * reference_residual * phi_plus;
                sums.t2 += weight * reference_residual * phi_minus;
                sums.current_loss += weight * residual * residual;
                sums.reference_loss += weight * reference_residual * reference_residual;
                if let Some((rho, target_plus, target_minus)) = target(row) {
                    sums.s11 += rho * phi_plus * phi_plus;
                    sums.s22 += rho * phi_minus * phi_minus;
                    sums.t1 += rho * (target_plus - phi_plus) * phi_plus;
                    sums.t2 += rho * (-target_minus - phi_minus) * phi_minus;
                    sums.current_loss +=
                        rho * ((f_plus - target_plus).powi(2) + (f_minus - target_minus).powi(2));
                    sums.reference_loss += rho
                        * ((phi_plus - target_plus).powi(2) + (phi_minus + target_minus).powi(2));
                }
            }
        }
        sums
    }
}
