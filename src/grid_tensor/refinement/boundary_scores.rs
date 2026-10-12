//! Resplit and merge scores for the boundaries between adjacent intervals.

use super::gain::{backbone_regularized_gain, canonical_merge_factors, l2_update_unanchored};
use super::RefinementStrategy;
use crate::grid_tensor::state::FittingState;
use crate::grid_tensor::two_tensor_solver::{
    data_loss_gain, log_coordinate_penalty, solve_two_tensor,
};

impl RefinementStrategy {
    pub(super) fn update_error_reductions_resplit_for_col_range(
        &self,
        state: &mut FittingState,
        col: usize,
        lo_boundary_idx: usize,
        hi_boundary_idx: usize,
    ) {
        let boundary_pos = &state.boundaries[col];
        if boundary_pos.is_empty() {
            return;
        }

        // Ensure all resplit caches are sized to the current number of boundaries
        let target_len = boundary_pos.len();
        if state.precomputed_statistics.error_reductions_resplit[col].len() < target_len {
            state.precomputed_statistics.error_reductions_resplit[col].resize(target_len, f64::NAN);
        }
        if state.precomputed_statistics.update_pairs_resplit_left[col].len() < target_len {
            state.precomputed_statistics.update_pairs_resplit_left[col]
                .resize(target_len, (f64::NAN, f64::NAN));
        }
        if state.precomputed_statistics.update_pairs_resplit_right[col].len() < target_len {
            state.precomputed_statistics.update_pairs_resplit_right[col]
                .resize(target_len, (f64::NAN, f64::NAN));
        }
        if state.precomputed_statistics.error_reductions_resplit_pairs[col].len() < target_len {
            state.precomputed_statistics.error_reductions_resplit_pairs[col]
                .resize(target_len, (f64::NAN, f64::NAN));
        }

        let is_stage1 = state.is_stage1_positive_only();
        let alpha = self.alpha();
        let (v_min, v_max) = self.multiplier_bounds();

        for i in lo_boundary_idx..=hi_boundary_idx {
            let left_stats = &state.precomputed_statistics.interval_stats[col][i];
            let right_stats = &state.precomputed_statistics.interval_stats[col][i + 1];

            if is_stage1 {
                // Stage 1 positive-only: use the 1D local log-backbone proposal.
                // H^L = S_{11}^L, g^L = t_1^L
                let h_l = left_stats.sum_s11;
                let g_l = left_stats.sum_t1;
                let u_l = l2_update_unanchored(alpha, g_l, h_l);
                let v_b_l = (1.0 + u_l).clamp(v_min, v_max);
                let u_l_clamped = v_b_l - 1.0;
                let gain_l = backbone_regularized_gain(u_l_clamped, g_l, h_l, alpha);

                // H^R = S_{11}^R, g^R = t_1^R
                let h_r = right_stats.sum_s11;
                let g_r = right_stats.sum_t1;
                let u_r = l2_update_unanchored(alpha, g_r, h_r);
                let v_b_r = (1.0 + u_r).clamp(v_min, v_max);
                let u_r_clamped = v_b_r - 1.0;
                let gain_r = backbone_regularized_gain(u_r_clamped, g_r, h_r, alpha);

                // Store updates: (u_plus, u_minus) = (u, 0) for Stage 1
                state.precomputed_statistics.error_reductions_resplit[col][i] = gain_l + gain_r;
                state.precomputed_statistics.update_pairs_resplit_left[col][i] = (u_l_clamped, 0.0);
                state.precomputed_statistics.update_pairs_resplit_right[col][i] =
                    (u_r_clamped, 0.0);
                state.precomputed_statistics.error_reductions_resplit_pairs[col][i] =
                    (gain_l, gain_r);
            } else {
                // Full two-tensor: use 2×2 solver
                let tau = self.tilt_tau();
                let rho = self.tilt_rho();

                // Solve for left side
                let (u_plus_l, u_minus_l, gain_l) = solve_two_tensor(
                    left_stats.sum_s11,
                    left_stats.sum_s22,
                    -left_stats.sum_s12,
                    left_stats.sum_t1,
                    -left_stats.sum_t2,
                    alpha,
                    tau,
                    rho,
                    v_min,
                    v_max,
                );
                // Solve for right side
                let (u_plus_r, u_minus_r, gain_r) = solve_two_tensor(
                    right_stats.sum_s11,
                    right_stats.sum_s22,
                    -right_stats.sum_s12,
                    right_stats.sum_t1,
                    -right_stats.sum_t2,
                    alpha,
                    tau,
                    rho,
                    v_min,
                    v_max,
                );
                // Store updates as (u_plus, u_minus) pairs for each side
                // Note: We'll convert to (v_b, delta_d) when applying the split
                state.precomputed_statistics.error_reductions_resplit[col][i] = gain_l + gain_r;
                state.precomputed_statistics.update_pairs_resplit_left[col][i] =
                    (u_plus_l, u_minus_l);
                state.precomputed_statistics.update_pairs_resplit_right[col][i] =
                    (u_plus_r, u_minus_r);
                state.precomputed_statistics.error_reductions_resplit_pairs[col][i] =
                    (gain_l, gain_r);
            }
        }
    }

    pub(super) fn update_error_reductions_merge_for_col_range(
        &self,
        state: &mut FittingState,
        col: usize,
        lo_boundary_idx: usize,
        hi_boundary_idx: usize,
    ) {
        if state.boundaries[col].is_empty() {
            return;
        }

        let is_stage1 = state.is_stage1_positive_only();
        let (v_min, v_max) = self.multiplier_bounds();
        let sorted_indices = &state.precomputed_statistics.sorted_indices[col];

        for i in lo_boundary_idx..=hi_boundary_idx {
            let (start, index, end) = state.interval_range_left_and_right(col, i);
            let left_count = index - start;
            let right_count = end - index;
            let total_count = (left_count + right_count) as f64;

            let axis_factors = |interval_idx: usize| {
                let backbone = state.backbone_values[col][interval_idx];
                let tilt = state.tilt_values[col][interval_idx];
                (backbone * tilt.exp(), backbone * (-tilt).exp())
            };
            let left_factors = axis_factors(i);
            let right_factors = axis_factors(i + 1);
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

            let merged_factors = if is_stage1 {
                let u = l2_update_unanchored(self.alpha(), t1, s11);
                let multiplier = (1.0 + u).clamp(v_min, v_max);
                canonical_merge_factors(reference.0 * multiplier, reference.0 * multiplier, true)
            } else {
                let (u_plus, u_minus, _) = solve_two_tensor(
                    s11,
                    s22,
                    s12,
                    t1,
                    t2,
                    self.alpha(),
                    self.tilt_tau(),
                    self.tilt_rho(),
                    v_min,
                    v_max,
                );
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
