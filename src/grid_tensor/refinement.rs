//! Refinement strategies: the L2 and Huber local objectives and the sufficient statistics
//! that score every structural action.
//!
//! A [`RefinementStrategy`] maintains the precomputed statistics of a fitting state: the
//! per-row caches, the exact and bin-cumulative prefix sums, the interval statistics and
//! the split, resplit and merge score caches. It builds them when a fit starts
//! (`initialize`), applies per-action deltas (`update_statistics` and its curvature-only
//! variants), rebuilds them after a residual or consensus change (`refresh_statistics`,
//! `refresh_consensus_reference`) and rescores the ranges an action touched
//! (`refresh_error_reduction_caches_for_affected`). The structural updates themselves are
//! applied by [`crate::grid_tensor::reducer`].

mod binned_curvature;
mod boundary_scores;
mod gain;
mod incremental;
mod initialize;
mod prefix;
mod refresh;
mod scores;
mod split_scores;

pub use gain::{huber_weighting, l2_gain_raw, l2_update_unanchored};
pub use prefix::{binned_prefix_stats_at, binned_prefix_stats_range, prefix_stats_range};

use crate::grid_tensor::two_tensor_solver::{DEFAULT_V_MAX, DEFAULT_V_MIN};

pub enum RefinementStrategy {
    L2Refinement {
        /// L2 penalty on the log-backbone update.
        alpha: f64,
        /// L2 penalty on the tilt update.
        tilt_tau: f64,
        /// L1 penalty on the tilt update.
        tilt_rho: f64,
        update_clamp: f64,
    },
    HuberRefinement {
        alpha: f64,
        c: f64,
        tilt_tau: f64,
        tilt_rho: f64,
        update_clamp: f64,
    },
}

impl RefinementStrategy {
    /// Panics unless `update_clamp` is nonnegative or positive infinity.
    pub(crate) fn assert_valid_update_clamp(&self) {
        assert!(
            self.update_clamp() >= 0.0,
            "update_clamp must be nonnegative or positive infinity"
        );
    }

    /// Multiplier limits combine the numerical box with the configured log-step cap.
    #[inline]
    pub(crate) fn multiplier_bounds(&self) -> (f64, f64) {
        self.assert_valid_update_clamp();
        let clamp = self.update_clamp();
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

    pub fn alpha(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { alpha, .. } => *alpha,
            RefinementStrategy::HuberRefinement { alpha, .. } => *alpha,
        }
    }

    #[inline]
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

    pub fn update_clamp(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { update_clamp, .. } => *update_clamp,
            RefinementStrategy::HuberRefinement { update_clamp, .. } => *update_clamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use rand::{rngs::StdRng, SeedableRng};

    use crate::grid_tensor::{
        fit::{fit_step, initialize_state_with_input},
        params::{
            GridTensorParams, GridTensorParamsBuilder, RefinementStrategyParamsBuilder,
            SplitStrategyParamsBuilder,
        },
        state::{BranchConsensus, FittingState},
    };

    use super::*;

    #[test]
    fn huber_error_uses_the_current_irls_quadratic_surrogate() {
        let strategy = RefinementStrategy::HuberRefinement {
            alpha: 0.0,
            c: 1.0,
            tilt_tau: 0.0,
            tilt_rho: 0.0,
            update_clamp: f64::INFINITY,
        };
        let residuals = ndarray::Array1::from_vec(vec![0.5, 2.0, -4.0]);

        assert!((strategy.weighted_squared_error(&residuals) - 6.25).abs() < 1e-12);
    }

    fn params(max_bins: Option<u16>, huber: bool, complexity_penalty: f64) -> GridTensorParams {
        let refinement = RefinementStrategyParamsBuilder::new()
            .alpha(0.01)
            .tilt_tau(0.01);
        let refinement = if huber {
            refinement.huber()
        } else {
            refinement.l2()
        }
        .build();
        GridTensorParamsBuilder::new()
            .n_iter(12)
            .max_bins(max_bins)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .complexity_penalty(complexity_penalty)
                    .build(),
            )
            .refinement_strategy(refinement)
            .build()
    }

    /// Up to ten fitting steps from `state`.
    fn fit<'a>(
        mut state: FittingState<'a>,
        params: &GridTensorParams,
        seed: u64,
    ) -> FittingState<'a> {
        let strategy = params.refinement_strategy_params.get_refinement_strategy();
        let split = params.split_strategy_params.get_split_strategy();
        let mut rng = StdRng::seed_from_u64(seed);
        for iter in 0..10 {
            let advanced;
            (state, advanced) = fit_step(state, iter, params.n_iter, &strategy, &split, &mut rng);
            if !advanced {
                break;
            }
        }
        state
    }

    fn fitted_state<'a>(
        x: &'a Array2<f64>,
        y: &'a Array1<f64>,
        max_bins: Option<u16>,
        huber: bool,
        complexity_penalty: f64,
    ) -> (FittingState<'a>, RefinementStrategy) {
        let params = params(max_bins, huber, complexity_penalty);
        let state = initialize_state_with_input(x.view(), y.view(), &params, None);
        (
            fit(state, &params, 7),
            params.refinement_strategy_params.get_refinement_strategy(),
        )
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

    /// Positive reference branches built from `other`, with per-row weight `weight`.
    fn consensus(other: &Array1<f64>, weight: f64) -> BranchConsensus {
        BranchConsensus {
            anchors: None,
            weight,
            plus: other.mapv(|v| 1.0 + 0.3 * v.abs()),
            minus: other.mapv(|v| 0.5 + 0.05 * v * v),
        }
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
                state.branch_consensus = Some(consensus(&other, 0.7));
                state = strategy.refresh_statistics(state);
                let statistics = &state.precomputed_statistics;
                let predicted = statistics
                    .error_reductions_split
                    .iter()
                    .chain(&statistics.error_reductions_resplit)
                    .flatten()
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

    /// Under a consensus reference the incrementally maintained statistics match a rebuild,
    /// and clearing the reference restores the statistics of the plain fit.
    #[test]
    fn branch_consensus_statistics_track_the_reference_and_clear_with_it() {
        let (x, y, other) = data(true);
        let split_scores =
            |s: &FittingState| s.precomputed_statistics.error_reductions_split.concat();
        for max_bins in [None, Some(16)] {
            for huber in [false, true] {
                let params = params(max_bins, huber, 0.0);
                let strategy = params.refinement_strategy_params.get_refinement_strategy();
                let mut pulled = initialize_state_with_input(x.view(), y.view(), &params, None);
                pulled.branch_consensus = Some(consensus(&other, 2.0));
                let pulled = fit(strategy.refresh_statistics(pulled), &params, 5);
                assert_caches_close(&strategy.refresh_statistics(pulled.clone()), &pulled);

                let (plain, _) = fitted_state(&x, &y, max_bins, huber, 0.0);
                let mut cleared = plain.clone();
                cleared.branch_consensus = Some(consensus(&other, 1.5));
                cleared = strategy.refresh_statistics(cleared);
                assert!(split_scores(&cleared)
                    .iter()
                    .zip(split_scores(&plain))
                    .any(|(u, v)| u.is_finite() && (u - v).abs() > 1e-6 * v.abs().max(1.0)));
                cleared.branch_consensus = None;
                assert_caches_close(&strategy.refresh_statistics(cleared), &plain);
            }
        }
    }
}
