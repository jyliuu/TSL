//! Resplit and merge scores for the boundaries between adjacent intervals.

use super::gain::{canonical_merge_factors, interval_solver_stats, SplitEvaluator};
use super::RefinementStrategy;
use crate::grid_tensor::axis_factors;
use crate::grid_tensor::state::FittingState;
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

            let mut s11 = 0.0;
            let mut s22 = 0.0;
            let mut s12 = 0.0;
            let mut t1 = 0.0;
            let mut t2 = 0.0;
            let mut current_loss = 0.0;
            let mut reference_loss = 0.0;

            for (range, factors) in [(start..index, left_factors), (index..end, right_factors)] {
                for &row in &sorted_indices[range] {
                    let weight = self.weight(state.residuals[row]);
                    let partial_plus = state.f_plus[row] / factors.0;
                    let partial_minus = state.f_minus[row] / factors.1;
                    let phi_plus = partial_plus * reference.0;
                    let phi_minus = -partial_minus * reference.1;
                    let reference_residual = state.labels[row] - phi_plus - phi_minus;
                    s11 += weight * phi_plus * phi_plus;
                    s22 += weight * phi_minus * phi_minus;
                    s12 += weight * phi_plus * phi_minus;
                    t1 += weight * reference_residual * phi_plus;
                    t2 += weight * reference_residual * phi_minus;
                    current_loss += weight * state.residuals[row] * state.residuals[row];
                    reference_loss += weight * reference_residual * reference_residual;
                    if let Some((rho, target_plus, target_minus)) = state
                        .branch_consensus
                        .as_ref()
                        .and_then(|consensus| consensus.target(row))
                    {
                        s11 += rho * phi_plus * phi_plus;
                        s22 += rho * phi_minus * phi_minus;
                        t1 += rho * (target_plus - phi_plus) * phi_plus;
                        t2 += rho * (-target_minus - phi_minus) * phi_minus;
                        current_loss += rho
                            * ((state.f_plus[row] - target_plus).powi(2)
                                + (state.f_minus[row] - target_minus).powi(2));
                        reference_loss += rho
                            * ((phi_plus - target_plus).powi(2)
                                + (phi_minus + target_minus).powi(2));
                    }
                }
            }

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
}
