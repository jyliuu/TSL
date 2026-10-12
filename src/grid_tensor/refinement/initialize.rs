//! Initial lambdas, per-point caches, sorted prefix sums and split scores of a fit.

use super::RefinementStrategy;
use crate::grid_tensor::state::{CurvatureCache, FittingState, PrefixStats};

impl RefinementStrategy {
    pub fn initialize<'a>(&self, state: FittingState<'a>) -> FittingState<'a> {
        self.initialize_with_input(state, None, None)
    }

    pub(crate) fn initialize_with_input<'a>(
        &self,
        mut state: FittingState<'a>,
        prepared: Option<&crate::grid_tensor::prepared_input::PreparedInput<'_>>,
        max_bins: Option<u16>,
    ) -> FittingState<'a> {
        self.multiplier_bounds();
        // Check if Stage 1 positive-only mode: all residuals are nonnegative
        let is_stage1 = state.residuals.iter().all(|&r| r >= 0.0);

        // Initialize two-tensor lambdas from current outer residuals (labels - stage prediction).
        const EPS_LAMBDA: f64 = 1e-10;

        if is_stage1 {
            // Stage 1 positive-only initialization
            // Set λ_+ = 1 (canonical starting point) or scale-matched value
            // Set λ_- = 0 (fixed, no negative component)
            state.lambda_plus = 1.0;
            state.lambda_minus = 0.0;

            // Ensure all tilt values are zero (already initialized in FittingState::new)
            // But explicitly set them to be safe
            for d_vec in state.tilt_values.iter_mut() {
                for d in d_vec.iter_mut() {
                    *d = 0.0;
                }
            }
        } else {
            // Full two-tensor initialization
            let mut sum_w = 0.0;
            let mut sum_pos = 0.0;
            let mut sum_neg = 0.0;
            for i in 0..state.n {
                let w = self.weight(state.residuals[i]);
                let r = state.residuals[i];
                sum_w += w;
                if r > 0.0 {
                    sum_pos += w * r;
                } else {
                    sum_neg += w * (-r);
                }
            }
            let denom = sum_w.max(EPS_LAMBDA);
            state.lambda_plus = (sum_pos / denom).max(EPS_LAMBDA);
            state.lambda_minus = (sum_neg / denom).max(EPS_LAMBDA);
        }

        // Initialize per-point caches from (b=1, d=0) and lambdas.
        // In Stage 1: f_+ = λ_+ * prod_j b_j, f_- = 0, f = f_+
        // In full mode: f_+ = λ_+ * prod_j b_j * exp(d_j), f_- = λ_- * prod_j b_j * exp(-d_j), f = f_+ - f_-
        state.f_plus.fill(state.lambda_plus);
        state.f_minus.fill(state.lambda_minus);
        state.f.assign(&(&state.f_plus - &state.f_minus));
        state.y_hat.assign(&state.f);
        state
            .residuals
            .assign(&(state.labels.to_owned() - state.y_hat.view()));
        state.r_tilde.assign(&state.residuals);
        state.current_error = state.residuals.iter().map(|r| r * r).sum();

        for col in 0..state.p {
            let indices = if let Some(prepared) = prepared {
                state.precomputed_statistics.sort_order[col].clone_from(&prepared.sort_order[col]);
                prepared.sorted_indices[col].clone()
            } else {
                let mut indices = (0..state.n).collect::<Vec<_>>();
                indices
                    .sort_by(|&a, &b| state.x[[a, col]].partial_cmp(&state.x[[b, col]]).unwrap());
                for (pos, &i) in indices.iter().enumerate() {
                    state.precomputed_statistics.sort_order[col][i] = pos;
                }
                indices
            };

            // Compute per-point two-tensor statistics and build prefix sums
            // For two-tensor, we need 5 sufficient statistics:
            //   curvature[i].s11 = w_i * f_plus[i]²
            //   curvature[i].s22 = w_i * f_minus[i]²
            //   curvature[i].s12 = w_i * f_plus[i] * f_minus[i]
            //   c_t1[i] = w_i * r_tilde[i] * f_plus[i]
            //   c_t2[i] = w_i * r_tilde[i] * f_minus[i]
            // We only need to populate per-point contributions once (col == 0), since they're point-specific
            let mut acc = PrefixStats::ZERO;
            let mut prefix: Vec<PrefixStats> = Vec::with_capacity(indices.len());

            for &i in indices.iter() {
                let fp = state.f_plus[i];
                let fm = state.f_minus[i];
                let rt = state.r_tilde[i];
                let w = self.weight(state.residuals[i]);

                // In Stage 1 mode: s22 = 0, s12 = 0, c_t2 = 0
                // Reuse existing caches but set unused ones to zero
                let c_s11_val = w * fp * fp;
                let c_s22_val = if is_stage1 { 0.0 } else { w * fm * fm };
                let c_s12_val = if is_stage1 { 0.0 } else { w * fp * fm };
                let c_t1_val = w * rt * fp;
                let c_t2_val = if is_stage1 { 0.0 } else { w * rt * fm };

                // Store per-point values (only on first column to avoid redundant writes)
                if col == 0 {
                    state.precomputed_statistics.curvature[i] = CurvatureCache {
                        s11: c_s11_val,
                        s22: c_s22_val,
                        s12: c_s12_val,
                    };
                    state.precomputed_statistics.c_t1[i] = c_t1_val;
                    state.precomputed_statistics.c_t2[i] = c_t2_val;
                }

                acc.s11 += c_s11_val;
                acc.s22 += c_s22_val;
                acc.s12 += c_s12_val;
                acc.t1 += c_t1_val;
                acc.t2 += c_t2_val;
                prefix.push(acc);
            }

            // Store prefix sums and sorted indices BEFORE computing error reductions
            state.precomputed_statistics.prefix_sums[col] = prefix;
            state.precomputed_statistics.sorted_indices[col] = indices;

            // Initialize interval stats: one interval covering all points
            // Note: S12 and t2 have negative signs in the sufficient stats definition
            state.precomputed_statistics.interval_stats[col] =
                vec![crate::grid_tensor::state::IntervalStats {
                    sum_s11: acc.s11,
                    sum_s22: acc.s22,
                    sum_s12: acc.s12, // Will be negated when used: S12 = -sum(curvature[i].s12)
                    sum_t1: acc.t1,
                    sum_t2: acc.t2, // Will be negated when used: t2 = -sum(c_t2)
                    n: state.n,
                }];

            // Mark duplicate values as NaN in error_reductions
            state.precomputed_statistics.sorted_indices[col]
                .iter()
                .enumerate()
                .scan(None, |last_value: &mut Option<f64>, (i, &idx)| {
                    let current_value = state.x[[idx, col]];
                    let is_duplicate = last_value
                        .map(|val| (current_value - val).abs() < f64::EPSILON)
                        .unwrap_or(false);
                    *last_value = Some(current_value);
                    Some((i, is_duplicate))
                })
                .filter(|&(_i, is_duplicate)| is_duplicate)
                .for_each(|(i, _)| {
                    state.precomputed_statistics.error_reductions_split[col][i] = f64::NAN;
                    state.precomputed_statistics.update_pairs_split_left[col][i] =
                        (f64::NAN, f64::NAN);
                    state.precomputed_statistics.update_pairs_split_right[col][i] =
                        (f64::NAN, f64::NAN);
                    state.precomputed_statistics.error_reductions_split_pairs[col][i] =
                        (f64::NAN, f64::NAN);
                });
        }

        state = crate::grid_tensor::histogram_binning::apply_histogram_binning_mask_with_input(
            state, max_bins, prepared,
        );
        state = self.update_error_reductions_split_for_all_cols(state);

        state.precomputed_statistics.initialized = true;
        state
    }
}
