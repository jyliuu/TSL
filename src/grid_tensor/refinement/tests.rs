//! Tests of the Huber loss surrogate, full cache rebuilds and branch-consensus statistics.

use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, SeedableRng};

use crate::grid_tensor::{
    fit::{fit_step, initialize_state_with_input},
    params::{
        GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
    },
    state::FittingState,
};

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
    assert!((actual.current_error - expected.current_error).abs() <= 1e-9 * expected.current_error);
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
                    weight * ((state.f_plus[i] - plus).powi(2) + (state.f_minus[i] - minus).powi(2))
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
            (state, advanced) = fit_step(state, iter, params.n_iter, &strategy, &split, &mut rng);
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
