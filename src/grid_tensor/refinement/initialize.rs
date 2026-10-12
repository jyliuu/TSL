//! Initial lambdas, per-point caches, sorted prefix sums and split scores of a fit.

use super::split_scores::SplitCaches;
use super::RefinementStrategy;
use crate::grid_tensor::prepared_input::sort_column;
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
        self.assert_valid_update_clamp();
        // Check if Stage 1 positive-only mode: all residuals are nonnegative
        let is_stage1 = state.residuals.iter().all(|&r| r >= 0.0);

        // Initialize two-tensor lambdas from current outer residuals (labels - stage prediction).
        const EPS_LAMBDA: f64 = 1e-10;

        if is_stage1 {
            // Stage 1 positive-only initialization: λ_+ = 1 (canonical starting point),
            // λ_- = 0 (no negative component) and zero tilts.
            state.lambda_plus = 1.0;
            state.lambda_minus = 0.0;
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

        // Per-point two-tensor statistics:
        //   curvature[i].s11 = w_i * f_plus[i]²
        //   curvature[i].s22 = w_i * f_minus[i]²
        //   curvature[i].s12 = w_i * f_plus[i] * f_minus[i]
        //   c_t1[i] = w_i * r_tilde[i] * f_plus[i]
        //   c_t2[i] = w_i * r_tilde[i] * f_minus[i]
        // Stage 1 has no f_- terms, so s22, s12 and c_t2 are zero.
        for i in 0..state.n {
            let fp = state.f_plus[i];
            let fm = state.f_minus[i];
            let rt = state.r_tilde[i];
            let w = self.weight(state.residuals[i]);
            state.precomputed_statistics.curvature[i] = CurvatureCache {
                s11: w * fp * fp,
                s22: if is_stage1 { 0.0 } else { w * fm * fm },
                s12: if is_stage1 { 0.0 } else { w * fp * fm },
            };
            state.precomputed_statistics.c_t1[i] = w * rt * fp;
            state.precomputed_statistics.c_t2[i] = if is_stage1 { 0.0 } else { w * rt * fm };
        }

        for col in 0..state.p {
            let indices = if let Some(prepared) = prepared {
                state.precomputed_statistics.sort_order[col].clone_from(&prepared.sort_order[col]);
                prepared.sorted_indices[col].clone()
            } else {
                let (indices, ranks) = sort_column(state.x, col);
                state.precomputed_statistics.sort_order[col] = ranks;
                indices
            };

            // Prefix sums of the per-point statistics in sorted order.
            let statistics = &state.precomputed_statistics;
            let mut acc = PrefixStats::ZERO;
            let prefix: Vec<PrefixStats> = indices
                .iter()
                .map(|&i| {
                    let curvature = statistics.curvature[i];
                    acc.s11 += curvature.s11;
                    acc.s22 += curvature.s22;
                    acc.s12 += curvature.s12;
                    acc.t1 += statistics.c_t1[i];
                    acc.t2 += statistics.c_t2[i];
                    acc
                })
                .collect();

            // A position whose value repeats the previous one cannot split.
            let x = state.x;
            let mut caches = SplitCaches::of(&mut state.precomputed_statistics, col);
            for (pos, pair) in indices.windows(2).enumerate() {
                if (x[[pair[1], col]] - x[[pair[0], col]]).abs() < f64::EPSILON {
                    caches.mask(pos + 1);
                }
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
        }

        state = crate::grid_tensor::histogram_binning::apply_histogram_binning_mask_with_input(
            state, max_bins, prepared,
        );
        state = self.update_error_reductions_split_for_all_cols(state);

        state.precomputed_statistics.initialized = true;
        state
    }
}
