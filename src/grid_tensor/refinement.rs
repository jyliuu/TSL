//! Refinement module for updating tree statistics after splits
//!
//! This module contains the refinement strategies that update precomputed statistics
//! after each split/resplit/merge operation. The actual tree updates are handled by
//! the reducer in `grid/reducer.rs`.

use std::iter::once;

use crate::grid_tensor::state::{
    AffectedRange, CurvatureCache, FittingState, PrecomputedStatistics, PrefixStats,
};
use crate::grid_tensor::two_tensor_solver::{
    data_loss_gain, log_coordinate_penalty, solve_two_tensor, DEFAULT_V_MAX, DEFAULT_V_MIN,
};

pub enum RefinementStrategy {
    L2Refinement {
        /// L2 penalty on the log-backbone update.
        alpha: f64,
        /// L2 penalty on the tilt update.
        tilt_tau: f64,
        /// L1 penalty on the tilt update.
        tilt_rho: f64,
        /// Prior sample size for parent anchoring (tau_0).
        /// Interpreted as "how many samples worth of confidence in the parent".
        prior_sample_size: f64,
        update_clamp: f64,
    },
    HuberRefinement {
        alpha: f64,
        c: f64,
        tilt_tau: f64,
        tilt_rho: f64,
        prior_sample_size: f64,
        update_clamp: f64,
    },
}

#[inline]
fn add_statistics<const LINEAR: bool>(target: &mut PrefixStats, value: PrefixStats) {
    target.s11 += value.s11;
    target.s22 += value.s22;
    target.s12 += value.s12;
    if LINEAR {
        target.t1 += value.t1;
        target.t2 += value.t2;
    }
}

struct SplitEvaluator {
    alpha: f64,
    tau: f64,
    rho: f64,
    v_min: f64,
    v_max: f64,
    is_stage1: bool,
}

impl SplitEvaluator {
    fn new(strategy: &RefinementStrategy, is_stage1: bool) -> Self {
        let (v_min, v_max) = strategy.multiplier_bounds();
        Self {
            alpha: strategy.alpha(),
            tau: strategy.tilt_tau(),
            rho: strategy.tilt_rho(),
            v_min,
            v_max,
            is_stage1,
        }
    }

    fn evaluate(
        &self,
        left: PrefixStats,
        total: PrefixStats,
    ) -> ((f64, f64), (f64, f64), (f64, f64)) {
        let Self {
            alpha,
            tau,
            rho,
            v_min,
            v_max,
            is_stage1,
        } = *self;
        if is_stage1 {
            let h_l = left.s11;
            let g_l = left.t1;
            let h_r = total.s11 - h_l;
            let g_r = total.t1 - g_l;
            let (u_l, _, _) = l2_update_unanchored(alpha, g_l, h_l);
            let v_b_l = (1.0 + u_l).clamp(v_min, v_max);
            let u_l_clamped = v_b_l - 1.0;
            let gain_l = backbone_regularized_gain(u_l_clamped, g_l, h_l, alpha);
            let (u_r, _, _) = l2_update_unanchored(alpha, g_r, h_r);
            let v_b_r = (1.0 + u_r).clamp(v_min, v_max);
            let u_r_clamped = v_b_r - 1.0;
            let gain_r = backbone_regularized_gain(u_r_clamped, g_r, h_r, alpha);
            ((u_l_clamped, 0.0), (u_r_clamped, 0.0), (gain_l, gain_r))
        } else {
            let s11_r = total.s11 - left.s11;
            let s22_r = total.s22 - left.s22;
            let s12_r = total.s12 - left.s12;
            let t1_r = total.t1 - left.t1;
            let t2_r = total.t2 - left.t2;
            let (u_plus_l, u_minus_l, gain_l) = solve_two_tensor(
                left.s11, left.s22, left.s12, left.t1, left.t2, alpha, tau, rho, v_min, v_max,
            );
            let (u_plus_r, u_minus_r, gain_r) = solve_two_tensor(
                s11_r, s22_r, s12_r, t1_r, t2_r, alpha, tau, rho, v_min, v_max,
            );
            (
                (u_plus_l, u_minus_l),
                (u_plus_r, u_minus_r),
                (gain_l, gain_r),
            )
        }
    }
}

fn solver_stats(value: PrefixStats, base: PrefixStats) -> PrefixStats {
    PrefixStats {
        s11: value.s11 - base.s11,
        s22: value.s22 - base.s22,
        s12: -(value.s12 - base.s12),
        t1: value.t1 - base.t1,
        t2: -(value.t2 - base.t2),
    }
}

impl RefinementStrategy {
    /// Multiplier limits combine the numerical box with the configured log-step cap.
    #[inline]
    pub(crate) fn multiplier_bounds(&self) -> (f64, f64) {
        let clamp = self.update_clamp();
        assert!(
            clamp >= 0.0,
            "update_clamp must be nonnegative or positive infinity"
        );
        if clamp >= DEFAULT_V_MAX.ln() {
            (DEFAULT_V_MIN, DEFAULT_V_MAX)
        } else {
            (
                (-clamp).exp().max(DEFAULT_V_MIN),
                clamp.exp().min(DEFAULT_V_MAX),
            )
        }
    }

    #[inline]
    pub fn tilt_tau(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { tilt_tau, .. }
            | RefinementStrategy::HuberRefinement { tilt_tau, .. } => *tilt_tau,
        }
    }

    #[inline]
    pub fn tilt_rho(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { tilt_rho, .. }
            | RefinementStrategy::HuberRefinement { tilt_rho, .. } => *tilt_rho,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huber_error_uses_the_current_irls_quadratic_surrogate() {
        let strategy = RefinementStrategy::HuberRefinement {
            alpha: 0.0,
            c: 1.0,
            tilt_tau: 0.0,
            tilt_rho: 0.0,
            prior_sample_size: 0.0,
            update_clamp: f64::INFINITY,
        };
        let residuals = ndarray::Array1::from_vec(vec![0.5, 2.0, -4.0]);

        assert!((strategy.weighted_squared_error(&residuals) - 6.25).abs() < 1e-12);
    }
}

#[inline]
pub fn prefix_range(prefix: &[f64], start: usize, end: usize) -> f64 {
    if start == 0 {
        prefix[end - 1]
    } else {
        prefix[end - 1] - prefix[start - 1]
    }
}

/// Subtract two `PrefixStats` componentwise.
#[inline]
fn sub_stats(a: PrefixStats, b: PrefixStats) -> PrefixStats {
    PrefixStats {
        s11: a.s11 - b.s11,
        s22: a.s22 - b.s22,
        s12: a.s12 - b.s12,
        t1: a.t1 - b.t1,
        t2: a.t2 - b.t2,
    }
}

/// Sum the per-stat prefix sums over the half-open sorted range `[start, end)`.
#[inline]
pub fn prefix_stats_range(prefix: &[PrefixStats], start: usize, end: usize) -> PrefixStats {
    if start == 0 {
        prefix[end - 1]
    } else {
        sub_stats(prefix[end - 1], prefix[start - 1])
    }
}

/// Read a bin-cumulative prefix sum at sorted position `pos`. `pos` must be
/// the last position of some bin (i.e., `pos = bin_edges[k] - 1` for some `k`
/// or `pos = n - 1`).
#[inline]
pub fn binned_prefix_stats_at(
    binned: &[PrefixStats],
    bin_edges: &[usize],
    pos: usize,
) -> PrefixStats {
    let k = bin_edges.partition_point(|&e| e <= pos);
    binned[k]
}

/// Bin-cumulative equivalent of `prefix_stats_range`: sum over sorted
/// positions `[start, end)`. Both endpoints must be bin-aligned.
#[inline]
pub fn binned_prefix_stats_range(
    binned: &[PrefixStats],
    bin_edges: &[usize],
    start: usize,
    end: usize,
) -> PrefixStats {
    let end_v = binned_prefix_stats_at(binned, bin_edges, end - 1);
    if start == 0 {
        end_v
    } else {
        sub_stats(end_v, binned_prefix_stats_at(binned, bin_edges, start - 1))
    }
}

#[inline]
pub fn l2_update_unanchored(alpha: f64, s_rb: f64, s_bb: f64) -> (f64, f64, f64) {
    let numerator = s_rb;
    let denominator = s_bb + alpha;
    let u = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };
    (u, denominator, numerator)
}

#[inline]
pub fn l2_gain_raw(update: f64, s_rb: f64, s_bb: f64) -> f64 {
    2.0 * update * s_rb - s_bb * update * update
}

#[inline]
fn backbone_regularized_gain(update: f64, s_rb: f64, s_bb: f64, alpha: f64) -> f64 {
    let multiplier = 1.0 + update;
    l2_gain_raw(update, s_rb, s_bb)
        - log_coordinate_penalty(multiplier, multiplier, alpha, 0.0, 0.0)
}

#[inline]
pub fn huber_weighting(res: f64, c: f64) -> f64 {
    if res.abs() <= c {
        1.0
    } else {
        c / res.abs()
    }
}

const MIN_BACKBONE: f64 = 1e-10;
const MAX_BACKBONE: f64 = 1e10;
const MAX_TILT: f64 = 10.0;

fn canonical_merge_factors(a_plus: f64, a_minus: f64, is_stage1: bool) -> (f64, f64) {
    if is_stage1 {
        let backbone = a_plus.clamp(MIN_BACKBONE, MAX_BACKBONE);
        return (backbone, backbone);
    }

    let min_axis_factor = MIN_BACKBONE * (-MAX_TILT).exp();
    let max_axis_factor = MAX_BACKBONE * MAX_TILT.exp();
    let log_plus = a_plus.clamp(min_axis_factor, max_axis_factor).ln();
    let log_minus = a_minus.clamp(min_axis_factor, max_axis_factor).ln();
    let log_backbone = (0.5 * (log_plus + log_minus)).clamp(MIN_BACKBONE.ln(), MAX_BACKBONE.ln());
    let tilt = (0.5 * (log_plus - log_minus)).clamp(-MAX_TILT, MAX_TILT);
    let backbone = log_backbone.exp();
    (backbone * tilt.exp(), backbone * (-tilt).exp())
}

/// Binned columns scattered per row pass of the linear rebin.
const LINEAR_TILE_COLUMNS: usize = 4;

impl RefinementStrategy {
    pub fn initialize<'a>(&self, state: FittingState<'a>) -> FittingState<'a> {
        self.initialize_with_input(state, None, None)
    }

    pub(crate) fn initialize_with_input<'a>(
        &self,
        mut state: FittingState<'a>,
        prepared: Option<&super::prepared_input::PreparedInput<'_>>,
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

        state = super::histogram_binning::apply_histogram_binning_mask_with_input(
            state, max_bins, prepared,
        );
        state = self.update_error_reductions_split_for_all_cols(state);

        state.precomputed_statistics.initialized = true;
        state
    }

    pub fn update_statistics<'a>(
        &self,
        state: FittingState<'a>,
        col: usize,
        start: usize,
        end: usize,
        affected_points_range: &[(usize, usize)],
    ) -> FittingState<'a> {
        self.update_statistics_impl::<true>(state, col, start, end, affected_points_range)
    }

    /// Retain exact incremental curvature before a full-row reference rebuild.
    pub(crate) fn update_curvature_statistics<'a>(
        &self,
        state: FittingState<'a>,
        col: usize,
        start: usize,
        end: usize,
        affected_points_range: &[(usize, usize)],
    ) -> FittingState<'a> {
        assert!(matches!(self, Self::L2Refinement { .. }));
        assert!(state
            .branch_consensus
            .as_ref()
            .is_some_and(|reference| reference.anchors.is_none()));
        self.update_statistics_impl::<false>(state, col, start, end, affected_points_range)
    }

    /// Scatter curvature deltas while locating the touched bins in the same row traversal.
    pub(crate) fn update_binned_curvature_statistics<'a>(
        &self,
        mut state: FittingState<'a>,
        col: usize,
        start: usize,
        end: usize,
    ) -> FittingState<'a> {
        state.require_current_statistics();
        assert!(matches!(self, Self::L2Refinement { .. }));
        assert!(state
            .branch_consensus
            .as_ref()
            .is_some_and(|reference| reference.anchors.is_none()));
        assert!(state
            .precomputed_statistics
            .bin_edges
            .iter()
            .all(|edges| !edges.is_empty()));
        let updated_points = &state.precomputed_statistics.sorted_indices[col][start..end];
        if updated_points.is_empty() {
            return state;
        }

        let mut deltas = self.collect_binned_curvature_deltas(&mut state, col, start, end);
        let n = state.n;
        if let Some(streams) = state.precomputed_statistics.bin_ranks.get(col).cloned() {
            // Each feature's bins over the affected ranks form one contiguous stream. Features
            // are scattered in pairs, and every histogram receives its additions in
            // affected-row order.
            assert_eq!(streams.len(), state.p * n);
            let prefixes = &mut state.precomputed_statistics.binned_prefix_sums;
            let mut buckets = Vec::new();
            let mut feature = 0;
            while feature < state.p {
                let paired = feature + 1 < state.p;
                let first_count = prefixes[feature].len();
                let second_count = if paired {
                    prefixes[feature + 1].len()
                } else {
                    0
                };
                buckets.clear();
                buckets.resize(first_count + second_count, [0.0; 3]);
                let (first_buckets, second_buckets) = buckets.split_at_mut(first_count);
                let first_stream = &streams[feature * n + start..feature * n + end];
                let (mut first_min, mut first_max) = (usize::MAX, 0);
                let (mut second_min, mut second_max) = (usize::MAX, 0);
                if paired {
                    let second_stream =
                        &streams[(feature + 1) * n + start..(feature + 1) * n + end];
                    for ((&first_bin, &second_bin), delta) in
                        first_stream.iter().zip(second_stream).zip(&deltas)
                    {
                        let first_bin = first_bin as usize;
                        let second_bin = second_bin as usize;
                        first_min = first_min.min(first_bin);
                        first_max = first_max.max(first_bin);
                        second_min = second_min.min(second_bin);
                        second_max = second_max.max(second_bin);
                        let first_slot = &mut first_buckets[first_bin];
                        first_slot[0] += delta[0];
                        first_slot[1] += delta[1];
                        first_slot[2] += delta[2];
                        let second_slot = &mut second_buckets[second_bin];
                        second_slot[0] += delta[0];
                        second_slot[1] += delta[1];
                        second_slot[2] += delta[2];
                    }
                } else {
                    for (&bin, delta) in first_stream.iter().zip(&deltas) {
                        let bin = bin as usize;
                        // Every touched row contributes to the range, including zero deltas.
                        first_min = first_min.min(bin);
                        first_max = first_max.max(bin);
                        let slot = &mut first_buckets[bin];
                        slot[0] += delta[0];
                        slot[1] += delta[1];
                        slot[2] += delta[2];
                    }
                }
                // Bin membership is monotone in sorted rank. These bounds equal the bins
                // of the minimum and maximum affected ranks, without a separate rank scan.
                let histograms = [
                    (feature, &*first_buckets, first_min, first_max),
                    (feature + 1, &*second_buckets, second_min, second_max),
                ];
                for &(feature, buckets, first, last) in histograms.iter().take(1 + paired as usize)
                {
                    let prefix = &mut prefixes[feature];
                    let mut accumulated = [0.0; 3];
                    for bin in first..=last {
                        accumulated[0] += buckets[bin][0];
                        accumulated[1] += buckets[bin][1];
                        accumulated[2] += buckets[bin][2];
                        prefix[bin].s11 += accumulated[0];
                        prefix[bin].s22 += accumulated[1];
                        prefix[bin].s12 += accumulated[2];
                    }
                    for slot in &mut prefix[(last + 1)..] {
                        slot.s11 += accumulated[0];
                        slot.s22 += accumulated[1];
                        slot.s12 += accumulated[2];
                    }
                }
                feature += 1 + paired as usize;
            }
            deltas.clear();
            state.precomputed_statistics.curvature_deltas = deltas;
            state.linear_statistics_stale = true;
            return state;
        }
        let updated_points = &state.precomputed_statistics.sorted_indices[col][start..end];

        let mut buckets = Vec::new();
        let mut features = (0..state.p).filter(|&feature| feature != col);
        let mut unpaired = None;
        while let Some(first_feature) = features.next() {
            let Some(second_feature) = features.next() else {
                unpaired = Some(first_feature);
                break;
            };
            let first_bins = &state.precomputed_statistics.bin_indices[first_feature];
            let second_bins = &state.precomputed_statistics.bin_indices[second_feature];
            let first_count = state.precomputed_statistics.binned_prefix_sums[first_feature].len();
            let second_count = state.precomputed_statistics.binned_prefix_sums[second_feature].len();
            buckets.clear();
            buckets.resize(first_count + second_count, [0.0; 3]);
            let (first_buckets, second_buckets) = buckets.split_at_mut(first_count);
            let (mut first_min, mut first_max) = (usize::MAX, 0);
            let (mut second_min, mut second_max) = (usize::MAX, 0);
            // Each histogram receives additions in affected-row order.
            for (&row, delta) in updated_points.iter().zip(&deltas) {
                let first_bin = first_bins[row] as usize;
                let second_bin = second_bins[row] as usize;
                first_min = first_min.min(first_bin);
                first_max = first_max.max(first_bin);
                second_min = second_min.min(second_bin);
                second_max = second_max.max(second_bin);
                let first_slot = &mut first_buckets[first_bin];
                first_slot[0] += delta[0];
                first_slot[1] += delta[1];
                first_slot[2] += delta[2];
                let second_slot = &mut second_buckets[second_bin];
                second_slot[0] += delta[0];
                second_slot[1] += delta[1];
                second_slot[2] += delta[2];
            }
            for (feature, buckets, first, last) in [
                (first_feature, &*first_buckets, first_min, first_max),
                (second_feature, &*second_buckets, second_min, second_max),
            ] {
                let prefix = &mut state.precomputed_statistics.binned_prefix_sums[feature];
                let mut accumulated = [0.0; 3];
                for bin in first..=last {
                    accumulated[0] += buckets[bin][0];
                    accumulated[1] += buckets[bin][1];
                    accumulated[2] += buckets[bin][2];
                    prefix[bin].s11 += accumulated[0];
                    prefix[bin].s22 += accumulated[1];
                    prefix[bin].s12 += accumulated[2];
                }
                for slot in &mut prefix[(last + 1)..] {
                    slot.s11 += accumulated[0];
                    slot.s22 += accumulated[1];
                    slot.s12 += accumulated[2];
                }
            }
        }
        for feature in unpaired.into_iter().chain(once(col)) {
            let bins = &state.precomputed_statistics.bin_indices[feature];
            let bin_count = state.precomputed_statistics.binned_prefix_sums[feature].len();
            buckets.clear();
            buckets.resize(bin_count, [0.0; 3]);
            let (mut first, mut last) = (usize::MAX, 0);
            for (&row, delta) in updated_points.iter().zip(&deltas) {
                let bin = bins[row] as usize;
                // Every touched row contributes to the range, including zero deltas.
                first = first.min(bin);
                last = last.max(bin);
                let slot = &mut buckets[bin];
                slot[0] += delta[0];
                slot[1] += delta[1];
                slot[2] += delta[2];
            }

            // Bin membership is monotone in sorted rank. These bounds equal the bins
            // of the minimum and maximum affected ranks, without a separate rank scan.
            let prefix = &mut state.precomputed_statistics.binned_prefix_sums[feature];
            let mut accumulated = [0.0; 3];
            for bin in first..=last {
                accumulated[0] += buckets[bin][0];
                accumulated[1] += buckets[bin][1];
                accumulated[2] += buckets[bin][2];
                prefix[bin].s11 += accumulated[0];
                prefix[bin].s22 += accumulated[1];
                prefix[bin].s12 += accumulated[2];
            }
            for slot in &mut prefix[(last + 1)..] {
                slot.s11 += accumulated[0];
                slot.s22 += accumulated[1];
                slot.s12 += accumulated[2];
            }
        }
        deltas.clear();
        state.precomputed_statistics.curvature_deltas = deltas;
        state.linear_statistics_stale = true;
        state
    }

    /// Collect full-row L2 curvature deltas in affected-row order.
    fn collect_binned_curvature_deltas(
        &self,
        state: &mut FittingState<'_>,
        col: usize,
        start: usize,
        end: usize,
    ) -> Vec<[f64; 3]> {
        let updated_points = &state.precomputed_statistics.sorted_indices[col][start..end];
        // The scratch keeps its capacity across actions; it is returned empty afterwards.
        let mut deltas = std::mem::take(&mut state.precomputed_statistics.curvature_deltas);
        deltas.clear();
        deltas.reserve(updated_points.len());
        let n = state.n;
        if let (Some(f_plus), Some(f_minus), Some(r_tilde), Some(residuals), Some(reference)) = (
            state.f_plus.as_slice(),
            state.f_minus.as_slice(),
            state.r_tilde.as_slice(),
            state.residuals.as_slice(),
            state.branch_consensus.as_ref(),
        ) {
            if let (Some(target_plus), Some(target_minus)) =
                (reference.plus.as_slice(), reference.minus.as_slice())
            {
                if matches!(self, Self::L2Refinement { .. })
                    && reference.anchors.is_none()
                    && [
                        f_plus.len(),
                        f_minus.len(),
                        r_tilde.len(),
                        residuals.len(),
                        target_plus.len(),
                        target_minus.len(),
                        state.precomputed_statistics.curvature.len(),
                    ]
                    .into_iter()
                    .all(|length| length == n)
                {
                    let rho = reference.weight;
                    let curvature = &mut state.precomputed_statistics.curvature[..n];
                    for &row in updated_points {
                        if row >= n {
                            // The first failing row uses the residual array's indexing behavior.
                            let _ = state.residuals[row];
                            unreachable!();
                        }
                        let fp = f_plus[row];
                        let fm = f_minus[row];
                        let mut s11 = 1.0 * fp * fp;
                        let mut s22 = 1.0 * fm * fm;
                        let s12 = 1.0 * fp * fm;
                        s11 += rho * fp * fp;
                        s22 += rho * fm * fm;
                        let cache = &mut curvature[row];
                        deltas.push([s11 - cache.s11, s22 - cache.s22, s12 - cache.s12]);
                        *cache = CurvatureCache { s11, s22, s12 };
                    }
                    return deltas;
                }
            }
        }
        for &row in updated_points {
            let point = state.point_curvature_statistics(row, self.weight(state.residuals[row]));
            deltas.push([
                point.s11 - state.precomputed_statistics.curvature[row].s11,
                point.s22 - state.precomputed_statistics.curvature[row].s22,
                point.s12 - state.precomputed_statistics.curvature[row].s12,
            ]);
            // Row statistics depend on predictions and the reference, not another row's cache.
            state.precomputed_statistics.curvature[row].s11 = point.s11;
            state.precomputed_statistics.curvature[row].s22 = point.s22;
            state.precomputed_statistics.curvature[row].s12 = point.s12;
        }
        deltas
    }

    fn update_statistics_impl<'a, const LINEAR: bool>(
        &self,
        mut state: FittingState<'a>,
        col: usize,
        start: usize,
        end: usize,
        affected_points_range: &[(usize, usize)],
    ) -> FittingState<'a> {
        state.require_current_statistics();
        let updated_points = &state.precomputed_statistics.sorted_indices[col][start..end];
        if updated_points.is_empty() {
            return state;
        }

        // 1) Compute the retained per-point statistics and their deltas vs stored c_*.
        // Precomputing deltas avoids re-loading old c_* values for every affected column.
        let residuals = &state.residuals;

        let (c_s11_new, c_s22_new, c_s12_new, c_t1_new, c_t2_new, d_s11, d_s22, d_s12, d_t1, d_t2) = {
            let old = &state.precomputed_statistics.curvature;
            let old_t1 = &state.precomputed_statistics.c_t1;
            let old_t2 = &state.precomputed_statistics.c_t2;

            let mut c_s11_new = Vec::with_capacity(updated_points.len());
            let mut c_s22_new = Vec::with_capacity(updated_points.len());
            let mut c_s12_new = Vec::with_capacity(updated_points.len());
            let mut c_t1_new = Vec::with_capacity(if LINEAR { updated_points.len() } else { 0 });
            let mut c_t2_new = Vec::with_capacity(if LINEAR { updated_points.len() } else { 0 });
            let mut d_s11 = Vec::with_capacity(updated_points.len());
            let mut d_s22 = Vec::with_capacity(updated_points.len());
            let mut d_s12 = Vec::with_capacity(updated_points.len());
            let mut d_t1 = Vec::with_capacity(if LINEAR { updated_points.len() } else { 0 });
            let mut d_t2 = Vec::with_capacity(if LINEAR { updated_points.len() } else { 0 });

            for &pt in updated_points.iter() {
                let PrefixStats {
                    s11: new_s11,
                    s22: new_s22,
                    s12: new_s12,
                    t1: new_t1,
                    t2: new_t2,
                } = if LINEAR {
                    state.point_statistics(pt, self.weight(residuals[pt]))
                } else {
                    state.point_curvature_statistics(pt, self.weight(residuals[pt]))
                };

                c_s11_new.push(new_s11);
                c_s22_new.push(new_s22);
                c_s12_new.push(new_s12);
                if LINEAR {
                    c_t1_new.push(new_t1);
                    c_t2_new.push(new_t2);
                }

                d_s11.push(new_s11 - old[pt].s11);
                d_s22.push(new_s22 - old[pt].s22);
                d_s12.push(new_s12 - old[pt].s12);
                if LINEAR {
                    d_t1.push(new_t1 - old_t1[pt]);
                    d_t2.push(new_t2 - old_t2[pt]);
                }
            }

            (
                c_s11_new, c_s22_new, c_s12_new, c_t1_new, c_t2_new, d_s11, d_s22, d_s12, d_t1,
                d_t2,
            )
        };

        // 2) For other (non-split) columns, apply delta updates.
        //    Binned columns: scatter deltas into B-length bin buckets and
        //    sweep cumulants over the affected bin range (~B work instead of
        //    ~n work — the main optimization).
        //    Exact columns: original O(span) scatter + sweep over length-n
        //    prefix sums (parity path).
        //
        //    The bucket scatter groups the retained deltas at offset `off`
        //    into one `PrefixStats` per bucket slot so the prefix-sum
        //    sweep touches a single 40-byte struct per index instead of
        //    five independent `Vec<f64>`. Order of additions is unchanged.
        let mut bucket_d: Vec<PrefixStats> = Vec::new();
        for (c, &(min_pos, max_pos)) in affected_points_range
            .iter()
            .enumerate()
            .filter(|&(c, _)| c != col)
        {
            if min_pos > max_pos {
                continue;
            }
            let is_binned = !state.precomputed_statistics.bin_edges[c].is_empty();

            if is_binned {
                // ---- Binned path ----
                let bin_edges = &state.precomputed_statistics.bin_edges[c];
                let min_bin = bin_edges.partition_point(|&e| e <= min_pos);
                let max_bin = bin_edges.partition_point(|&e| e <= max_pos);
                let bin_span = max_bin - min_bin + 1;

                bucket_d.clear();
                bucket_d.resize(bin_span, PrefixStats::ZERO);

                let bin_indices = &state.precomputed_statistics.bin_indices[c];
                for (j, &pt) in updated_points.iter().enumerate() {
                    let bin = bin_indices[pt] as usize;
                    let off = bin - min_bin;
                    let slot = &mut bucket_d[off];
                    slot.s11 += d_s11[j];
                    slot.s22 += d_s22[j];
                    slot.s12 += d_s12[j];
                    if LINEAR {
                        slot.t1 += d_t1[j];
                        slot.t2 += d_t2[j];
                    }
                }

                let b = &mut state.precomputed_statistics.binned_prefix_sums[c];
                let mut acc_d = PrefixStats::ZERO;
                for off in 0..bin_span {
                    add_statistics::<LINEAR>(&mut acc_d, bucket_d[off]);
                    add_statistics::<LINEAR>(&mut b[min_bin + off], acc_d);
                }
                if let Some(tail) = b.get_mut((max_bin + 1)..) {
                    for v in tail {
                        add_statistics::<LINEAR>(v, acc_d);
                    }
                }
            } else {
                // ---- Exact path (length-n prefix sums) ----
                let span = max_pos - min_pos + 1;
                bucket_d.clear();
                bucket_d.resize(span, PrefixStats::ZERO);

                let sort_order = &state.precomputed_statistics.sort_order[c];
                let prefix_sums = &mut state.precomputed_statistics.prefix_sums[c];

                for (j, &pt) in updated_points.iter().enumerate() {
                    let pos_full = sort_order[pt];
                    let off = pos_full - min_pos;
                    let slot = &mut bucket_d[off];
                    slot.s11 += d_s11[j];
                    slot.s22 += d_s22[j];
                    slot.s12 += d_s12[j];
                    if LINEAR {
                        slot.t1 += d_t1[j];
                        slot.t2 += d_t2[j];
                    }
                }

                let mut acc_d = PrefixStats::ZERO;
                let range = &mut prefix_sums[min_pos..=max_pos];
                for off in 0..span {
                    add_statistics::<LINEAR>(&mut acc_d, bucket_d[off]);
                    add_statistics::<LINEAR>(&mut range[off], acc_d);
                }

                if let Some(tail) = prefix_sums.get_mut((max_pos + 1)..) {
                    for v in tail {
                        add_statistics::<LINEAR>(v, acc_d);
                    }
                }
            }
        }

        // 3) Split column: same path-split as non-split columns, but the
        //    affected range is the full interval [start..end).
        //    Binned path uses the same bin-indexed delta scatter as above.
        //    Exact path keeps the original rebuild-from-new-values logic for
        //    parity with the prior implementation.
        let split_is_binned = !state.precomputed_statistics.bin_edges[col].is_empty();

        if split_is_binned {
            // Binned split column: bin-indexed delta scatter, identical to
            // the non-split binned loop above but using (start, end-1).
            let min_pos = start;
            let max_pos = end - 1;

            let bin_edges = &state.precomputed_statistics.bin_edges[col];
            let min_bin = bin_edges.partition_point(|&e| e <= min_pos);
            let max_bin = bin_edges.partition_point(|&e| e <= max_pos);
            let bin_span = max_bin - min_bin + 1;

            bucket_d.clear();
            bucket_d.resize(bin_span, PrefixStats::ZERO);

            let bin_indices = &state.precomputed_statistics.bin_indices[col];
            for (j, &pt) in updated_points.iter().enumerate() {
                let bin = bin_indices[pt] as usize;
                let off = bin - min_bin;
                let slot = &mut bucket_d[off];
                slot.s11 += d_s11[j];
                slot.s22 += d_s22[j];
                slot.s12 += d_s12[j];
                if LINEAR {
                    slot.t1 += d_t1[j];
                    slot.t2 += d_t2[j];
                }
            }

            let b = &mut state.precomputed_statistics.binned_prefix_sums[col];
            let mut acc_d = PrefixStats::ZERO;
            for off in 0..bin_span {
                add_statistics::<LINEAR>(&mut acc_d, bucket_d[off]);
                add_statistics::<LINEAR>(&mut b[min_bin + off], acc_d);
            }
            if let Some(tail) = b.get_mut((max_bin + 1)..) {
                for v in tail {
                    add_statistics::<LINEAR>(v, acc_d);
                }
            }
        } else {
            let prefix_sums = &mut state.precomputed_statistics.prefix_sums[col];

            // Exact split column: rebuild the interval [start..end) and shift the tail.
            // Each per-stat prefix is rebuilt by accumulating the matching
            // c_*_new vector — same order of additions as before.
            let min_pos = start;
            let max_pos = end - 1;
            let prev = if min_pos > 0 {
                prefix_sums[min_pos - 1]
            } else {
                PrefixStats::ZERO
            };
            let old_last = prefix_sums[max_pos];

            let mut acc = prev;
            for (k, v) in c_s11_new.iter().enumerate() {
                acc.s11 += *v;
                prefix_sums[min_pos + k].s11 = acc.s11;
            }
            acc.s22 = prev.s22;
            for (k, v) in c_s22_new.iter().enumerate() {
                acc.s22 += *v;
                prefix_sums[min_pos + k].s22 = acc.s22;
            }
            acc.s12 = prev.s12;
            for (k, v) in c_s12_new.iter().enumerate() {
                acc.s12 += *v;
                prefix_sums[min_pos + k].s12 = acc.s12;
            }
            acc.t1 = prev.t1;
            for (k, v) in c_t1_new.iter().enumerate() {
                acc.t1 += *v;
                prefix_sums[min_pos + k].t1 = acc.t1;
            }
            acc.t2 = prev.t2;
            for (k, v) in c_t2_new.iter().enumerate() {
                acc.t2 += *v;
                prefix_sums[min_pos + k].t2 = acc.t2;
            }

            let diff = if LINEAR {
                sub_stats(prefix_sums[max_pos], old_last)
            } else {
                PrefixStats {
                    s11: prefix_sums[max_pos].s11 - old_last.s11,
                    s22: prefix_sums[max_pos].s22 - old_last.s22,
                    s12: prefix_sums[max_pos].s12 - old_last.s12,
                    t1: 0.0,
                    t2: 0.0,
                }
            };
            if let Some(tail) = prefix_sums.get_mut((max_pos + 1)..) {
                for v in tail {
                    add_statistics::<LINEAR>(v, diff);
                }
            }
        }

        // 4) Store the per-point curvature and linear caches
        {
            let curvature = &mut state.precomputed_statistics.curvature;
            let c_t1 = &mut state.precomputed_statistics.c_t1;
            let c_t2 = &mut state.precomputed_statistics.c_t2;
            for (j, &pt) in updated_points.iter().enumerate() {
                curvature[pt].s11 = c_s11_new[j];
                curvature[pt].s22 = c_s22_new[j];
                curvature[pt].s12 = c_s12_new[j];
                if LINEAR {
                    c_t1[pt] = c_t1_new[j];
                    c_t2[pt] = c_t2_new[j];
                }
            }
        }

        state.linear_statistics_stale = !LINEAR;
        state
    }

    /// Change only the peer reference while retaining current all-row data statistics.
    /// Binned, merge-free fits update anchor contributions in O(p(m + bins)) work.
    pub(crate) fn refresh_consensus_reference<'a>(
        &self,
        mut state: FittingState<'a>,
        reference: crate::grid_tensor::state::BranchConsensus,
    ) -> FittingState<'a> {
        use crate::grid_tensor::state::same_consensus_support;
        let sparse = !state.linear_statistics_stale
            && reference.anchors.is_some()
            && !state.split_strategy_state.merge_enabled
            && state
                .precomputed_statistics
                .bin_edges
                .iter()
                .all(|edges| !edges.is_empty())
            && state.precomputed_statistics.consensus_weight == reference.weight
            && same_consensus_support(
                &state.precomputed_statistics.consensus_anchors,
                &reference.anchors,
            )
            && state.branch_consensus.as_ref().is_some_and(|old| {
                old.weight == reference.weight
                    && same_consensus_support(&old.anchors, &reference.anchors)
            });
        if !sparse {
            state.branch_consensus = Some(reference);
            return self.refresh_statistics(state);
        }
        let anchors = reference.anchors.as_ref().unwrap().clone();
        state.branch_consensus = Some(reference);
        let mut deltas = Vec::with_capacity(anchors.rows.len());
        for &row in &anchors.rows {
            // Re-evaluate these row totals to bound drift from repeated reference updates.
            let point = state.point_statistics(row, self.weight(state.residuals[row]));
            let statistics = &mut state.precomputed_statistics;
            deltas.push((
                point.t1 - statistics.c_t1[row],
                point.t2 - statistics.c_t2[row],
            ));
            statistics.c_t1[row] = point.t1;
            statistics.c_t2[row] = point.t2;
        }
        let mut bins = Vec::new();
        for col in 0..state.p {
            let statistics = &mut state.precomputed_statistics;
            let prefix = &mut statistics.binned_prefix_sums[col];
            bins.clear();
            bins.resize(prefix.len(), (0.0, 0.0));
            for (&row, &(delta_t1, delta_t2)) in anchors.rows.iter().zip(&deltas) {
                let bin = statistics.bin_indices[col][row] as usize;
                bins[bin].0 += delta_t1;
                bins[bin].1 += delta_t2;
            }
            let (mut t1, mut t2) = (0.0, 0.0);
            for (slot, &(delta_t1, delta_t2)) in prefix.iter_mut().zip(&bins) {
                t1 += delta_t1;
                t2 += delta_t2;
                slot.t1 += t1;
                slot.t2 += t2;
            }
        }
        for col in 0..state.p {
            let last_interval = state.boundaries[col].len();
            Self::update_interval_stats_for_col_range(&mut state, col, 0, last_interval);
        }
        let ranges: Vec<AffectedRange> = (0..state.p)
            .map(|col| AffectedRange {
                col,
                point_range: (0, state.n - 1),
                interval_range: (0, 0),
            })
            .collect();
        self.refresh_error_reduction_caches_for_affected(state, &ranges)
    }

    /// Populate linear terms from contiguous full-row references with retained L2 curvature.
    fn refresh_full_row_linear_slices(state: &mut FittingState<'_>) -> bool {
        let Some(reference) = state.branch_consensus.as_ref() else {
            return false;
        };
        let n = state.n;
        if reference.anchors.is_some() || state.residuals.len() < n {
            return false;
        }
        let (Some(plus), Some(minus), Some(residual), Some(target_plus), Some(target_minus)) = (
            state.f_plus.as_slice(),
            state.f_minus.as_slice(),
            state.r_tilde.as_slice(),
            reference.plus.as_slice(),
            reference.minus.as_slice(),
        ) else {
            return false;
        };
        let statistics = &mut state.precomputed_statistics;
        if [
            plus.len(),
            minus.len(),
            residual.len(),
            target_plus.len(),
            target_minus.len(),
            statistics.c_t1.len(),
            statistics.c_t2.len(),
        ]
        .into_iter()
        .any(|len| len < n)
        {
            return false;
        }
        let (plus, minus, residual, target_plus, target_minus) = (
            &plus[..n],
            &minus[..n],
            &residual[..n],
            &target_plus[..n],
            &target_minus[..n],
        );
        let (c_t1, c_t2) = (&mut statistics.c_t1[..n], &mut statistics.c_t2[..n]);
        let rho = reference.weight;
        for row in 0..n {
            let (fp, fm, rt) = (plus[row], minus[row], residual[row]);
            let mut t1 = 1.0 * rt * fp;
            let mut t2 = 1.0 * rt * fm;
            t1 += rho * (target_plus[row] - fp) * fp;
            t2 += rho * (fm - target_minus[row]) * fm;
            c_t1[row] = t1;
            c_t2[row] = t2;
        }
        true
    }

    /// Scatter the linear row terms of a tile of binned columns in one row pass.
    ///
    /// Every bucket receives its rows in ascending row order from `+0.0`, as the
    /// per-column loop does; columns are independent accumulators.
    fn scatter_linear_tile(
        maps: &[&[u16]],
        t1: &[f64],
        t2: &[f64],
        scratch: &mut [[f64; 2]],
        widths: &[usize],
    ) {
        match (maps, widths) {
            ([m0, m1, m2, m3], &[w0, w1, w2, _]) => {
                let (b0, rest) = scratch.split_at_mut(w0);
                let (b1, rest) = rest.split_at_mut(w1);
                let (b2, b3) = rest.split_at_mut(w2);
                for (((((&x0, &x1), &x2), &x3), &v1), &v2) in
                    m0.iter().zip(*m1).zip(*m2).zip(*m3).zip(t1).zip(t2)
                {
                    let s0 = &mut b0[x0 as usize];
                    s0[0] += v1;
                    s0[1] += v2;
                    let s1 = &mut b1[x1 as usize];
                    s1[0] += v1;
                    s1[1] += v2;
                    let s2 = &mut b2[x2 as usize];
                    s2[0] += v1;
                    s2[1] += v2;
                    let s3 = &mut b3[x3 as usize];
                    s3[0] += v1;
                    s3[1] += v2;
                }
            }
            ([m0, m1, m2], &[w0, w1, _]) => {
                let (b0, rest) = scratch.split_at_mut(w0);
                let (b1, b2) = rest.split_at_mut(w1);
                for ((((&x0, &x1), &x2), &v1), &v2) in m0.iter().zip(*m1).zip(*m2).zip(t1).zip(t2) {
                    let s0 = &mut b0[x0 as usize];
                    s0[0] += v1;
                    s0[1] += v2;
                    let s1 = &mut b1[x1 as usize];
                    s1[0] += v1;
                    s1[1] += v2;
                    let s2 = &mut b2[x2 as usize];
                    s2[0] += v1;
                    s2[1] += v2;
                }
            }
            ([m0, m1], &[w0, _]) => {
                let (b0, b1) = scratch.split_at_mut(w0);
                for (((&x0, &x1), &v1), &v2) in m0.iter().zip(*m1).zip(t1).zip(t2) {
                    let s0 = &mut b0[x0 as usize];
                    s0[0] += v1;
                    s0[1] += v2;
                    let s1 = &mut b1[x1 as usize];
                    s1[0] += v1;
                    s1[1] += v2;
                }
            }
            ([m0], &[_]) => {
                let b0 = scratch;
                for ((&x0, &v1), &v2) in m0.iter().zip(t1).zip(t2) {
                    let s0 = &mut b0[x0 as usize];
                    s0[0] += v1;
                    s0[1] += v2;
                }
            }
            _ => unreachable!("tiles hold one to four columns"),
        }
    }

    /// Rebuild the binned linear prefix sums of every binned column, `LINEAR_TILE_COLUMNS`
    /// columns per row pass over the first `n` rows of the linear caches.
    fn rebin_linear_tiles(statistics: &mut PrecomputedStatistics, n: usize) {
        let PrecomputedStatistics {
            c_t1,
            c_t2,
            bin_indices,
            bin_edges,
            binned_prefix_sums,
            ..
        } = statistics;
        let (t1, t2) = (&c_t1[..n], &c_t2[..n]);
        let binned: Vec<usize> = (0..bin_edges.len())
            .filter(|&col| !bin_edges[col].is_empty())
            .collect();
        let mut scratch: Vec<[f64; 2]> = Vec::new();
        for tile in binned.chunks(LINEAR_TILE_COLUMNS) {
            let widths: Vec<usize> = tile
                .iter()
                .map(|&col| binned_prefix_sums[col].len())
                .collect();
            scratch.clear();
            scratch.resize(widths.iter().sum(), [0.0; 2]);
            let maps: Vec<&[u16]> = tile.iter().map(|&col| &bin_indices[col][..n]).collect();
            Self::scatter_linear_tile(&maps, t1, t2, &mut scratch, &widths);
            let mut offset = 0;
            for (&col, &width) in tile.iter().zip(&widths) {
                let bucket = &scratch[offset..offset + width];
                offset += width;
                let (mut t1, mut t2) = (0.0, 0.0);
                for (slot, b) in binned_prefix_sums[col].iter_mut().zip(bucket) {
                    t1 += b[0];
                    t2 += b[1];
                    slot.t1 = t1;
                    slot.t2 = t2;
                }
            }
        }
    }

    /// Rebuild every statistic that depends on the residuals or the branch consensus.
    ///
    /// The partition and factors are kept; per-point sufficient statistics, prefix sums,
    /// interval statistics and all split, resplit and merge scores are recomputed.
    pub(crate) fn refresh_statistics<'a>(&self, mut state: FittingState<'a>) -> FittingState<'a> {
        let n = state.n;
        // L2 weights are constant, so while the consensus weight is unchanged only the
        // reference-dependent t1/t2 sums move.
        let consensus_weight = state.branch_consensus.as_ref().map_or(0.0, |c| c.weight);
        let consensus_anchors = state
            .branch_consensus
            .as_ref()
            .and_then(|c| c.anchors.clone());
        let weights_fixed = matches!(self, RefinementStrategy::L2Refinement { .. })
            && state.precomputed_statistics.consensus_weight == consensus_weight
            && crate::grid_tensor::state::same_consensus_support(
                &state.precomputed_statistics.consensus_anchors,
                &consensus_anchors,
            );
        if weights_fixed {
            if !Self::refresh_full_row_linear_slices(&mut state) {
                for i in 0..n {
                    let point = state.point_statistics(i, self.weight(state.residuals[i]));
                    state.precomputed_statistics.c_t1[i] = point.t1;
                    state.precomputed_statistics.c_t2[i] = point.t2;
                }
            }
        } else {
            for i in 0..n {
                let point = state.point_statistics(i, self.weight(state.residuals[i]));
                let statistics = &mut state.precomputed_statistics;
                statistics.curvature[i].s11 = point.s11;
                statistics.curvature[i].s22 = point.s22;
                statistics.curvature[i].s12 = point.s12;
                statistics.c_t1[i] = point.t1;
                statistics.c_t2[i] = point.t2;
            }
        }
        let statistics = &mut state.precomputed_statistics;
        statistics.consensus_weight = consensus_weight;
        statistics.consensus_anchors = consensus_anchors;
        let tiled = weights_fixed
            && statistics.c_t1.len() >= n
            && statistics.c_t2.len() >= n
            && (0..state.p).all(|col| {
                statistics.bin_edges[col].is_empty()
                    || (statistics.bin_indices[col].len() == n
                        && statistics.binned_prefix_sums[col].len()
                            == statistics.bin_edges[col].len() + 1)
            });
        if tiled {
            Self::rebin_linear_tiles(statistics, n);
        }
        let mut bins_t1 = Vec::new();
        let mut bins_t2 = Vec::new();
        for col in 0..state.p {
            if statistics.bin_edges[col].is_empty() {
                let prefix = &mut statistics.prefix_sums[col];
                let mut acc = PrefixStats::ZERO;
                for (slot, &i) in prefix.iter_mut().zip(&statistics.sorted_indices[col]) {
                    if !weights_fixed {
                        acc.s11 += statistics.curvature[i].s11;
                        acc.s22 += statistics.curvature[i].s22;
                        acc.s12 += statistics.curvature[i].s12;
                        slot.s11 = acc.s11;
                        slot.s22 = acc.s22;
                        slot.s12 = acc.s12;
                    }
                    acc.t1 += statistics.c_t1[i];
                    acc.t2 += statistics.c_t2[i];
                    slot.t1 = acc.t1;
                    slot.t2 = acc.t2;
                }
            } else if tiled {
                // Rebuilt by the tiled pass.
            } else if weights_fixed {
                let binned = &mut statistics.binned_prefix_sums[col];
                bins_t1.clear();
                bins_t1.resize(binned.len(), 0.0);
                bins_t2.clear();
                bins_t2.resize(binned.len(), 0.0);
                let row_bins = &statistics.bin_indices[col];
                match (
                    statistics.c_t1.get(..row_bins.len()),
                    statistics.c_t2.get(..row_bins.len()),
                ) {
                    (Some(t1), Some(t2)) => {
                        for ((&bin, &value1), &value2) in row_bins.iter().zip(t1).zip(t2) {
                            bins_t1[bin as usize] += value1;
                            bins_t2[bin as usize] += value2;
                        }
                    }
                    _ => {
                        for (i, &bin) in statistics.bin_indices[col].iter().enumerate() {
                            bins_t1[bin as usize] += statistics.c_t1[i];
                            bins_t2[bin as usize] += statistics.c_t2[i];
                        }
                    }
                }
                let (mut t1, mut t2) = (0.0, 0.0);
                for (slot, (b1, b2)) in binned.iter_mut().zip(bins_t1.iter().zip(&bins_t2)) {
                    t1 += b1;
                    t2 += b2;
                    slot.t1 = t1;
                    slot.t2 = t2;
                }
            } else {
                let binned = &mut statistics.binned_prefix_sums[col];
                binned.fill(PrefixStats::ZERO);
                for (i, &bin) in statistics.bin_indices[col].iter().enumerate() {
                    binned[bin as usize] += PrefixStats {
                        s11: statistics.curvature[i].s11,
                        s22: statistics.curvature[i].s22,
                        s12: statistics.curvature[i].s12,
                        t1: statistics.c_t1[i],
                        t2: statistics.c_t2[i],
                    };
                }
                let mut acc = PrefixStats::ZERO;
                for bin in binned.iter_mut() {
                    acc += *bin;
                    *bin = acc;
                }
            }
        }

        state.linear_statistics_stale = false;
        for col in 0..state.p {
            let last_interval = state.boundaries[col].len();
            Self::update_interval_stats_for_col_range(&mut state, col, 0, last_interval);
        }
        let ranges: Vec<AffectedRange> = (0..state.p)
            .map(|col| AffectedRange {
                col,
                point_range: (0, n - 1),
                interval_range: (0, 0),
            })
            .collect();
        self.refresh_error_reduction_caches_for_affected(state, &ranges)
    }

    pub fn refresh_error_reduction_caches_for_affected<'a>(
        &self,
        mut state: FittingState<'a>,
        affected_ranges: &[AffectedRange],
    ) -> FittingState<'a> {
        state.require_current_statistics();
        if state.split_scores_stale
            && (!state.sampled_split_scoring || state.is_stage1_positive_only())
        {
            self.refresh_all_split_scores(&mut state);
        }
        for affected_range in affected_ranges {
            let c = affected_range.col;
            let (lo, hi) = affected_range.point_range;

            // Update split error reductions using point range
            self.update_error_reductions_split_for_col_range(&mut state, c, lo, hi);

            // Update interval stats and boundary caches using boundary interval range
            // Note: interval_range in AffectedRange is for allowed_intervals,
            // but we need boundary indices for cache refreshing, so compute from point range
            let n_splits = state.boundaries[c].len();
            if n_splits > 0 {
                // Compute boundary interval indices from point range
                let (blo, bhi) = state.compute_boundary_index_range(c, lo, hi);

                // Update interval stats for affected intervals (O(1) lookup for resplit/merge)
                Self::update_interval_stats_for_col_range(&mut state, c, blo, bhi);

                // Convert interval range to boundary range
                // Intervals [blo, bhi] affect boundaries [blo-1, bhi] (clamped to valid boundaries)
                let n_boundaries = state.boundaries[c].len();
                if n_boundaries > 0 {
                    let boundary_lo = blo.saturating_sub(1);
                    let boundary_hi = bhi.min(n_boundaries - 1);

                    self.update_error_reductions_resplit_for_col_range(
                        &mut state,
                        c,
                        boundary_lo,
                        boundary_hi,
                    );
                    // Only update merge error reductions if merge is enabled
                    if state.split_strategy_state.merge_enabled {
                        self.update_error_reductions_merge_for_col_range(
                            &mut state,
                            c,
                            boundary_lo,
                            boundary_hi,
                        );
                    }
                }
            }
        }

        state
    }

    /// Recompute interval stats for affected intervals using prefix sums
    fn update_interval_stats_for_col_range(
        state: &mut FittingState,
        col: usize,
        blo: usize,
        bhi: usize,
    ) {
        let boundaries = &state.boundaries[col];
        let n = state.x.nrows();
        let bin_edges = &state.precomputed_statistics.bin_edges[col];
        let is_binned = !bin_edges.is_empty();

        // Update stats for intervals [blo..=bhi+1] since a boundary affects two adjacent intervals
        let start_interval = blo;
        let end_interval = (bhi + 2).min(state.precomputed_statistics.interval_stats[col].len());

        for interval_idx in start_interval..end_interval {
            let start = if interval_idx == 0 {
                0
            } else {
                boundaries[interval_idx - 1]
            };
            let end = boundaries.get(interval_idx).copied().unwrap_or(n);
            let interval_n = end - start;

            let stats = if is_binned {
                // Binned: start and end are always bin-aligned (splits only
                // happen at bin edges in binned mode). Read directly from
                // length-B cumulants.
                let b = &state.precomputed_statistics.binned_prefix_sums[col];
                binned_prefix_stats_range(b, bin_edges, start, end)
            } else {
                let p = &state.precomputed_statistics.prefix_sums[col];
                prefix_stats_range(p, start, end)
            };

            state.precomputed_statistics.interval_stats[col][interval_idx] =
                crate::grid_tensor::state::IntervalStats {
                    sum_s11: stats.s11,
                    sum_s22: stats.s22,
                    sum_s12: stats.s12,
                    sum_t1: stats.t1,
                    sum_t2: stats.t2,
                    n: interval_n,
                };
        }
    }

    fn update_error_reductions_split_for_all_cols<'a>(
        &self,
        mut state: FittingState<'a>,
    ) -> FittingState<'a> {
        let n = state.x.nrows();
        for col in 0..state.boundaries.len() {
            self.update_error_reductions_split_for_col_range(&mut state, col, 0, n);
        }
        state
    }

    fn update_error_reductions_split_for_col_range(
        &self,
        state: &mut FittingState,
        col: usize,
        lo: usize,
        hi: usize,
    ) {
        let n = state.x.nrows();
        let is_stage1 = state.is_stage1_positive_only();
        let mut start = 0usize;
        for &b in state.boundaries[col].iter().chain(once(&n)) {
            // Computes the error reductions for any interval that intersects with the updated points range

            let end = b;
            if end > lo && state.sampled_split_scoring && !is_stage1 {
                // solve_two_tensor returns finite gains, so summing the two sides cannot
                // create NaN eligibility masks. Positive-only scoring uses the eager path.
                // Preserve every historical mask, including endpoints of past intervals.
                if end.saturating_sub(start) >= 2 {
                    for pos in [start, end - 1] {
                        state.precomputed_statistics.error_reductions_split[col][pos] = f64::NAN;
                        state.precomputed_statistics.update_pairs_split_left[col][pos] =
                            (f64::NAN, f64::NAN);
                        state.precomputed_statistics.update_pairs_split_right[col][pos] =
                            (f64::NAN, f64::NAN);
                        state.precomputed_statistics.error_reductions_split_pairs[col][pos] =
                            (f64::NAN, f64::NAN);
                    }
                }
                state.split_scores_stale = true;
            } else if end > lo {
                self.update_error_reductions_split_single_interval(
                    &state.precomputed_statistics.prefix_sums[col],
                    &state.precomputed_statistics.binned_prefix_sums[col],
                    &mut state.precomputed_statistics.update_pairs_split_left[col],
                    &mut state.precomputed_statistics.update_pairs_split_right[col],
                    &mut state.precomputed_statistics.error_reductions_split[col],
                    &mut state.precomputed_statistics.error_reductions_split_pairs[col],
                    &state.precomputed_statistics.bin_edges[col],
                    (start, end),
                    is_stage1,
                );
            }
            start = end;
            if start > hi {
                break;
            }
        }
    }

    /// Rebuild split scores before an eager consumer or scoring-mode transition.
    pub(crate) fn refresh_all_split_scores(&self, state: &mut FittingState) {
        state.require_current_statistics();
        let sampled = state.sampled_split_scoring;
        state.sampled_split_scoring = false;
        for col in 0..state.p {
            self.update_error_reductions_split_for_col_range(state, col, 0, state.n);
        }
        state.sampled_split_scoring = sampled;
        state.split_scores_stale = false;
    }

    /// Evaluate one sampled coordinate against its current structural interval.
    pub(crate) fn refresh_sampled_split(&self, state: &mut FittingState, col: usize, pos: usize) {
        state.require_current_statistics();
        let statistics = &state.precomputed_statistics;
        let Some(gain) = statistics.error_reductions_split[col].get(pos) else {
            return;
        };
        if gain.is_nan() {
            return;
        }
        let interval = state.boundaries[col].partition_point(|&boundary| boundary <= pos);
        let (start, end) = state.interval_range(col, interval);
        if pos <= start || pos >= end.saturating_sub(1) {
            return;
        }
        let bin_edges = &statistics.bin_edges[col];
        let (base, left, total) = if bin_edges.is_empty() {
            let ps = &statistics.prefix_sums[col];
            (
                if start == 0 {
                    PrefixStats::ZERO
                } else {
                    ps[start - 1]
                },
                ps[pos - 1],
                ps[end - 1],
            )
        } else {
            let Ok(bin) = bin_edges.binary_search(&pos) else {
                return;
            };
            let bs = &statistics.binned_prefix_sums[col];
            (
                if start == 0 {
                    PrefixStats::ZERO
                } else {
                    binned_prefix_stats_at(bs, bin_edges, start - 1)
                },
                bs[bin],
                binned_prefix_stats_at(bs, bin_edges, end - 1),
            )
        };
        let (left, right, gains) = SplitEvaluator::new(self, state.is_stage1_positive_only())
            .evaluate(solver_stats(left, base), solver_stats(total, base));
        let statistics = &mut state.precomputed_statistics;
        statistics.update_pairs_split_left[col][pos] = left;
        statistics.update_pairs_split_right[col][pos] = right;
        statistics.error_reductions_split_pairs[col][pos] = gains;
        statistics.error_reductions_split[col][pos] = gains.0 + gains.1;
    }

    fn update_error_reductions_resplit_for_col_range(
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
            // Use interval stats directly (O(1) lookup instead of prefix_range computation)
            let left_stats = &state.precomputed_statistics.interval_stats[col][i];
            let right_stats = &state.precomputed_statistics.interval_stats[col][i + 1];

            if is_stage1 {
                // Stage 1 positive-only: use the 1D local log-backbone proposal.
                // H^L = S_{11}^L, g^L = t_1^L
                let h_l = left_stats.sum_s11;
                let g_l = left_stats.sum_t1;
                let (u_l, _denom_l, _num_l) = l2_update_unanchored(alpha, g_l, h_l);
                let v_b_l = (1.0 + u_l).clamp(v_min, v_max);
                let u_l_clamped = v_b_l - 1.0;
                let gain_l = backbone_regularized_gain(u_l_clamped, g_l, h_l, alpha);

                // H^R = S_{11}^R, g^R = t_1^R
                let h_r = right_stats.sum_s11;
                let g_r = right_stats.sum_t1;
                let (u_r, _denom_r, _num_r) = l2_update_unanchored(alpha, g_r, h_r);
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

    fn update_error_reductions_merge_for_col_range(
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
                let (u, _, _) = l2_update_unanchored(self.alpha(), t1, s11);
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

    #[allow(clippy::too_many_arguments)]
    fn update_error_reductions_split_single_interval(
        &self,
        ps: &[PrefixStats],
        bs: &[PrefixStats],
        update_left: &mut [(f64, f64)],
        update_right: &mut [(f64, f64)],
        error_reductions: &mut [f64],
        error_reductions_pairs: &mut [(f64, f64)],
        bin_edges: &[usize],
        spanned_interval: (usize, usize),
        is_stage1: bool,
    ) {
        let (start, end) = spanned_interval;
        let len = end.saturating_sub(start);
        if len < 2 {
            return;
        }

        // Disallow splitting at the first position.
        update_left[start] = (f64::NAN, f64::NAN);
        update_right[start] = (f64::NAN, f64::NAN);
        error_reductions[start] = f64::NAN;
        error_reductions_pairs[start] = (f64::NAN, f64::NAN);

        let evaluator = SplitEvaluator::new(self, is_stage1);

        // Evaluate a single candidate position given left-side sufficient stats.
        // Closed over the output slices + total stats. Shared between the
        // exact and binned iteration paths.
        let eval_pos = |pos: usize,
                        s11_l: f64,
                        s22_l: f64,
                        s12_l: f64,
                        t1_l: f64,
                        t2_l: f64,
                        total_s11: f64,
                        total_s22: f64,
                        total_s12: f64,
                        total_t1: f64,
                        total_t2: f64,
                        update_left: &mut [(f64, f64)],
                        update_right: &mut [(f64, f64)],
                        error_reductions: &mut [f64],
                        error_reductions_pairs: &mut [(f64, f64)]| {
            if error_reductions[pos].is_nan() {
                update_left[pos] = (f64::NAN, f64::NAN);
                update_right[pos] = (f64::NAN, f64::NAN);
                error_reductions_pairs[pos] = (f64::NAN, f64::NAN);
                return;
            }

            let (left, right, gains) = evaluator.evaluate(
                PrefixStats {
                    s11: s11_l,
                    s22: s22_l,
                    s12: s12_l,
                    t1: t1_l,
                    t2: t2_l,
                },
                PrefixStats {
                    s11: total_s11,
                    s22: total_s22,
                    s12: total_s12,
                    t1: total_t1,
                    t2: total_t2,
                },
            );
            update_left[pos] = left;
            update_right[pos] = right;
            error_reductions[pos] = gains.0 + gains.1;
            error_reductions_pairs[pos] = gains;
        };

        if bin_edges.is_empty() {
            // Exact path: read length-n prefix sums at every candidate position.
            let base = if start == 0 {
                PrefixStats::ZERO
            } else {
                ps[start - 1]
            };

            let total_s11 = ps[end - 1].s11 - base.s11;
            let total_s22 = ps[end - 1].s22 - base.s22;
            let total_s12 = -(ps[end - 1].s12 - base.s12);
            let total_t1 = ps[end - 1].t1 - base.t1;
            let total_t2 = -(ps[end - 1].t2 - base.t2);

            for pos in (start + 1)..end.saturating_sub(1) {
                let pos_end = pos - 1;
                let s11_l = ps[pos_end].s11 - base.s11;
                let s22_l = ps[pos_end].s22 - base.s22;
                let s12_l = -(ps[pos_end].s12 - base.s12);
                let t1_l = ps[pos_end].t1 - base.t1;
                let t2_l = -(ps[pos_end].t2 - base.t2);
                eval_pos(
                    pos,
                    s11_l,
                    s22_l,
                    s12_l,
                    t1_l,
                    t2_l,
                    total_s11,
                    total_s22,
                    total_s12,
                    total_t1,
                    total_t2,
                    update_left,
                    update_right,
                    error_reductions,
                    error_reductions_pairs,
                );
            }
        } else {
            // Binned path: read length-B bin-cumulative prefix sums at bin
            // boundaries. Splits land at `bin_edges[k]` so `pos_end = pos - 1`
            // is the last position of bin k → cumulant at bin index k.
            let base = if start == 0 {
                PrefixStats::ZERO
            } else {
                binned_prefix_stats_at(bs, bin_edges, start - 1)
            };
            let end_v = binned_prefix_stats_at(bs, bin_edges, end - 1);

            let total_s11 = end_v.s11 - base.s11;
            let total_s22 = end_v.s22 - base.s22;
            let total_s12 = -(end_v.s12 - base.s12);
            let total_t1 = end_v.t1 - base.t1;
            let total_t2 = -(end_v.t2 - base.t2);

            let lo_idx = bin_edges.partition_point(|&e| e <= start);
            let hi_idx = bin_edges.partition_point(|&e| e < end.saturating_sub(1));
            for k in lo_idx..hi_idx {
                let pos = bin_edges[k];
                let s11_l = bs[k].s11 - base.s11;
                let s22_l = bs[k].s22 - base.s22;
                let s12_l = -(bs[k].s12 - base.s12);
                let t1_l = bs[k].t1 - base.t1;
                let t2_l = -(bs[k].t2 - base.t2);
                eval_pos(
                    pos,
                    s11_l,
                    s22_l,
                    s12_l,
                    t1_l,
                    t2_l,
                    total_s11,
                    total_s22,
                    total_s12,
                    total_t1,
                    total_t2,
                    update_left,
                    update_right,
                    error_reductions,
                    error_reductions_pairs,
                );
            }
        }

        // Disallow splitting at the last position.
        let last = end - 1;
        update_left[last] = (f64::NAN, f64::NAN);
        update_right[last] = (f64::NAN, f64::NAN);
        error_reductions[last] = f64::NAN;
        error_reductions_pairs[last] = (f64::NAN, f64::NAN);
    }

    pub fn alpha(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { alpha, .. } => *alpha,
            RefinementStrategy::HuberRefinement { alpha, .. } => *alpha,
        }
    }

    pub fn weight(&self, res: f64) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { .. } => 1.0,
            RefinementStrategy::HuberRefinement { c, .. } => huber_weighting(res, *c),
        }
    }

    /// Current IRLS quadratic surrogate used to compare every structural action
    /// and its stopping tolerance on one loss scale.
    pub(crate) fn weighted_squared_error(&self, residuals: &ndarray::Array1<f64>) -> f64 {
        residuals
            .iter()
            .map(|&residual| self.weight(residual) * residual * residual)
            .sum()
    }

    /// Returns the prior sample size (tau_0) for parent anchoring.
    ///
    /// This represents "how many samples worth of confidence we have that
    /// children should equal their parent". With tau_0 = 30, a child interval
    /// with 10 samples will be heavily shrunk toward the parent, while a child
    /// with 100 samples will mostly trust its own data.
    pub fn prior_sample_size(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement {
                prior_sample_size, ..
            } => *prior_sample_size,
            RefinementStrategy::HuberRefinement {
                prior_sample_size, ..
            } => *prior_sample_size,
        }
    }

    pub fn update_clamp(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { update_clamp, .. } => *update_clamp,
            RefinementStrategy::HuberRefinement { update_clamp, .. } => *update_clamp,
        }
    }

    /// Compute the gain for setting the multiplier to zero (u = -1).
    pub fn gain_for_zero_multiplier(&self, s_rb: f64, s_bb: f64, _n: f64) -> f64 {
        // When forcing multiplier to 0, the additive update is u = 0 - 1 = -1.
        // gain = 2u S_rb - u^2 S_bb = 2(-1)S_rb - (-1)^2 S_bb = -2 S_rb - S_bb.
        let u = -1.0;
        l2_gain_raw(u, s_rb, s_bb)
    }
}

#[cfg(test)]
mod refresh_tests {
    use ndarray::{Array1, Array2};
    use rand::{rngs::StdRng, SeedableRng};

    use crate::grid_tensor::{
        fit::{fit_step, initialize_state_with_input},
        params::{
            GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
        },
        state::FittingState,
    };

    fn fitted_state<'a>(
        x: &'a Array2<f64>,
        y: &'a Array1<f64>,
        max_bins: Option<u16>,
        huber: bool,
        complexity_penalty: f64,
    ) -> (FittingState<'a>, super::RefinementStrategy) {
        let refinement = RefinementStrategyParamsBuilder::new()
            .alpha(0.01)
            .tilt_tau(0.01);
        let refinement = if huber {
            refinement.huber()
        } else {
            refinement.l2()
        }
        .build();
        let params = GridTensorParamsBuilder::new()
            .n_iter(12)
            .max_bins(max_bins)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .complexity_penalty(complexity_penalty)
                    .build(),
            )
            .refinement_strategy(refinement)
            .build();
        let strategy = params.refinement_strategy_params.get_refinement_strategy();
        let split = params.split_strategy_params.get_split_strategy();
        let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
        let mut rng = StdRng::seed_from_u64(7);
        for iter in 0..10 {
            let advanced;
            (state, advanced) = fit_step(state, iter, params.n_iter, &strategy, &split, &mut rng);
            if !advanced {
                break;
            }
        }
        (state, strategy)
    }

    fn assert_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
        for (a, e) in actual.iter().zip(expected) {
            let scale = e
                .iter()
                .filter(|v| v.is_finite())
                .fold(1e-300_f64, |m, v| m.max(v.abs()));
            for (a, e) in a.iter().zip(e) {
                assert_eq!(a.is_nan(), e.is_nan());
                // Incremental sums accumulate in action order and a rebuild in row order, so
                // scores agree to rounding, not bitwise.
                if e.is_finite() {
                    assert!((a - e).abs() <= 1e-8 * scale, "{a} vs {e}");
                }
            }
        }
    }

    fn assert_caches_close(actual: &FittingState, expected: &FittingState) {
        let (a, e) = (
            &actual.precomputed_statistics,
            &expected.precomputed_statistics,
        );
        assert_close(&a.error_reductions_split, &e.error_reductions_split);
        assert_close(&a.error_reductions_resplit, &e.error_reductions_resplit);
        assert_close(&a.error_reductions_merge, &e.error_reductions_merge);
        // Interval statistics are read only on columns that have boundaries.
        let sums = |s: &FittingState| -> Vec<Vec<f64>> {
            s.precomputed_statistics
                .interval_stats
                .iter()
                .zip(&expected.boundaries)
                .filter(|(_, boundaries)| !boundaries.is_empty())
                .map(|(col, _)| {
                    col.iter()
                        .flat_map(|i| [i.sum_s11, i.sum_t1, i.sum_t2])
                        .collect()
                })
                .collect()
        };
        assert_close(&sums(actual), &sums(expected));
        assert!(
            (actual.current_error - expected.current_error).abs() <= 1e-9 * expected.current_error
        );
    }

    fn data(signed: bool) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((240, 3), |(i, j)| ((i * (2 * j + 3) + j) % 61) as f64 / 7.0);
        let offset = if signed { 0.0 } else { 6.0 };
        let y = Array1::from_shape_fn(240, |i| {
            (x[[i, 0]]).sin() * x[[i, 1]] - 0.3 * x[[i, 2]] + offset
        });
        let other = Array1::from_shape_fn(240, |i| x[[i, 2]].cos() + 0.2 * x[[i, 0]] + offset);
        (x, y, other)
    }

    /// The row caches, binned prefix sums and every split, resplit and merge score are
    /// maintained incrementally after each action; a full rebuild must reproduce them.
    #[test]
    fn refresh_reproduces_incremental_caches() {
        let mut merged = false;
        for signed in [false, true] {
            for max_bins in [None, Some(16)] {
                for huber in [false, true] {
                    for complexity_penalty in [0.0, 0.5] {
                        let (x, y, _) = data(signed);
                        let (state, strategy) =
                            fitted_state(&x, &y, max_bins, huber, complexity_penalty);
                        assert!(!state.boundaries.iter().all(Vec::is_empty));
                        merged |= state.loop_state.merge_count > 0;
                        let rebuilt = strategy.refresh_statistics(state.clone());
                        assert_caches_close(&rebuilt, &state);
                    }
                }
            }
        }
        assert!(merged, "fixture must exercise merges");
    }

    fn branch_objective(state: &FittingState) -> f64 {
        let consensus = state.branch_consensus.as_ref().unwrap();
        (0..state.n)
            .map(|i| {
                (state.labels[i] - state.f[i]).powi(2)
                    + consensus.target(i).map_or(0.0, |(weight, plus, minus)| {
                        weight
                            * ((state.f_plus[i] - plus).powi(2)
                                + (state.f_minus[i] - minus).powi(2))
                    })
            })
            .sum()
    }

    #[test]
    fn branch_consensus_split_gains_match_the_penalized_objective() {
        use crate::grid_tensor::{params::SplitStrategyParamsBuilder, state::BranchConsensus};
        let (x, y, other) = data(true);
        for max_bins in [None, Some(16)] {
            let refinement = RefinementStrategyParamsBuilder::new()
                .alpha(0.0)
                .tilt_tau(0.0)
                .l2()
                .build();
            let params = GridTensorParamsBuilder::new()
                .n_iter(6)
                .max_bins(max_bins)
                .split_strategy(SplitStrategyParamsBuilder::new().best_split().build())
                .refinement_strategy(refinement)
                .build();
            let strategy = params.refinement_strategy_params.get_refinement_strategy();
            let split = params.split_strategy_params.get_split_strategy();
            let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
            let mut rng = StdRng::seed_from_u64(1);
            for iter in 0..4 {
                state.branch_consensus = Some(BranchConsensus {
                    anchors: None,
                    weight: 0.7,
                    plus: other.mapv(|v| 1.5 + 0.2 * v.abs()),
                    minus: other.mapv(|v| 0.8 + 0.1 * v * v),
                });
                state = strategy.refresh_statistics(state);
                let predicted = state
                    .precomputed_statistics
                    .error_reductions_split
                    .iter()
                    .flatten()
                    .chain(
                        state
                            .precomputed_statistics
                            .error_reductions_resplit
                            .iter()
                            .flatten(),
                    )
                    .filter(|v| v.is_finite())
                    .fold(f64::NEG_INFINITY, |m, &v| m.max(v));
                let before = branch_objective(&state);
                let advanced;
                (state, advanced) =
                    fit_step(state, iter, params.n_iter, &strategy, &split, &mut rng);
                assert!(advanced);
                let realized = before - branch_objective(&state);
                assert!(
                    (realized - predicted).abs() <= 1e-8 * before,
                    "predicted {predicted}, realized {realized}"
                );
            }
        }
    }

    #[test]
    fn incremental_updates_track_branch_consensus_statistics() {
        use crate::grid_tensor::state::BranchConsensus;
        for max_bins in [None, Some(16)] {
            for huber in [false, true] {
                let (x, y, other) = data(true);
                let refinement = RefinementStrategyParamsBuilder::new()
                    .alpha(0.01)
                    .tilt_tau(0.01);
                let refinement = if huber {
                    refinement.huber()
                } else {
                    refinement.l2()
                }
                .build();
                let params = GridTensorParamsBuilder::new()
                    .n_iter(12)
                    .max_bins(max_bins)
                    .refinement_strategy(refinement)
                    .build();
                let strategy = params.refinement_strategy_params.get_refinement_strategy();
                let split = params.split_strategy_params.get_split_strategy();
                let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
                state.branch_consensus = Some(BranchConsensus {
                    anchors: None,
                    weight: 2.0,
                    plus: other.mapv(|v| 1.0 + 0.3 * v.abs()),
                    minus: other.mapv(|v| 0.5 + 0.05 * v * v),
                });
                state = strategy.refresh_statistics(state);
                let mut rng = StdRng::seed_from_u64(5);
                for iter in 0..10 {
                    let advanced;
                    (state, advanced) =
                        fit_step(state, iter, params.n_iter, &strategy, &split, &mut rng);
                    if !advanced {
                        break;
                    }
                }
                let rebuilt = strategy.refresh_statistics(state.clone());
                assert_caches_close(&rebuilt, &state);
            }
        }
    }

    #[test]
    fn clearing_branch_consensus_restores_the_plain_statistics() {
        use crate::grid_tensor::state::BranchConsensus;
        for max_bins in [None, Some(16)] {
            let (x, y, other) = data(true);
            let (state, strategy) = fitted_state(&x, &y, max_bins, false, 0.0);
            let mut pulled = state.clone();
            pulled.branch_consensus = Some(BranchConsensus {
                anchors: None,
                weight: 1.5,
                plus: other.mapv(|v| 1.0 + v.abs()),
                minus: other.mapv(|v| 0.5 + 0.1 * v * v),
            });
            let mut pulled = strategy.refresh_statistics(pulled);
            let moved = |a: &FittingState, b: &FittingState| {
                a.precomputed_statistics
                    .error_reductions_split
                    .iter()
                    .flatten()
                    .zip(
                        b.precomputed_statistics
                            .error_reductions_split
                            .iter()
                            .flatten(),
                    )
                    .any(|(u, v)| u.is_finite() && (u - v).abs() > 1e-6 * v.abs().max(1.0))
            };
            assert!(moved(&pulled, &state));
            pulled.branch_consensus = None;
            let restored = strategy.refresh_statistics(pulled);
            assert_caches_close(&restored, &state);
        }
    }
}
