use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, SeedableRng};
use tsl::grid_tensor::{
    fit,
    params::{
        GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
    },
    refinement::RefinementStrategy,
    state::FittingState,
};

fn default_model_bytes(huber: bool, positive: bool, bins: Option<u16>, clamp: f64) -> Vec<u8> {
    let x = Array2::from_shape_fn((48, 2), |(row, axis)| {
        ((row * (7 + axis * 4) + axis * 3) % 53) as f64 / 5.0
    });
    let y = Array1::from_shape_fn(48, |row| {
        if positive {
            0.01 + (0.3 * x[[row, 0]]).exp() * (1.0 + x[[row, 1]])
        } else {
            4.0 * (0.7 * x[[row, 0]]).sin() - 0.8 * x[[row, 1]]
        }
    });
    let refinement = RefinementStrategyParamsBuilder::new()
        .alpha(0.1)
        .tilt_tau(0.03)
        .tilt_rho(0.01)
        .update_clamp(clamp);
    let refinement = if huber {
        refinement.huber()
    } else {
        refinement.l2()
    };
    let params = GridTensorParamsBuilder::new()
        .n_iter(5)
        .max_bins(bins)
        .split_strategy(
            SplitStrategyParamsBuilder::new()
                .random_split()
                .split_try(9)
                .colsample_bytree(1.0)
                .min_interval_samples(2)
                .build(),
        )
        .refinement_strategy(refinement.build())
        .build();
    let (_, model) = fit(x.view(), y.view(), &params, &mut StdRng::seed_from_u64(314));
    bincode::serialize(&model).unwrap()
}

#[test]
fn infinite_cap_matches_numerical_bound_caps() {
    for huber in [false, true] {
        for positive in [false, true] {
            for bins in [None, Some(16)] {
                let bytes = default_model_bytes(huber, positive, bins, f64::INFINITY);
                for cap in [20.0_f64.ln(), 10.0] {
                    assert_eq!(bytes, default_model_bytes(huber, positive, bins, cap));
                }
            }
        }
    }
}

fn refinement(huber: bool, clamp: f64) -> RefinementStrategy {
    if huber {
        RefinementStrategy::HuberRefinement {
            alpha: 0.1,
            c: 1.345,
            tilt_tau: 0.03,
            tilt_rho: 0.01,
            prior_sample_size: 0.0,
            update_clamp: clamp,
        }
    } else {
        RefinementStrategy::L2Refinement {
            alpha: 0.1,
            tilt_tau: 0.03,
            tilt_rho: 0.01,
            prior_sample_size: 0.0,
            update_clamp: clamp,
        }
    }
}

fn fixture(positive: bool) -> (Array2<f64>, Array1<f64>, usize) {
    let x = Array2::from_shape_fn((6, 1), |(row, _)| row as f64);
    let (targets, index) = if positive {
        (vec![0.01, 0.01, 80.0, 80.0, 80.0, 80.0], 2)
    } else {
        (vec![-100.0, -100.0, -100.0, 100.0, 100.0, 100.0], 3)
    };
    (x, Array1::from_vec(targets), index)
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-9 * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

fn check_multipliers(v: (f64, f64), cap: f64) -> bool {
    let lo = (-cap).exp();
    let hi = cap.exp();
    for value in [v.0, v.1] {
        assert!(
            value >= lo - 1e-12 && value <= hi + 1e-12,
            "multiplier {value} outside [{lo}, {hi}]"
        );
    }
    (v.0 - lo).abs() < 1e-10
        || (v.0 - hi).abs() < 1e-10
        || (v.1 - lo).abs() < 1e-10
        || (v.1 - hi).abs() < 1e-10
}

#[test]
fn negative_and_nan_caps_fail_even_without_split_candidates() {
    let x = Array2::zeros((1, 1));
    let y = Array1::ones(1);
    for huber in [false, true] {
        for cap in [-1.0, f64::NEG_INFINITY, f64::NAN] {
            let error = std::panic::catch_unwind(|| {
                refinement(huber, cap).initialize(FittingState::new(x.view(), y.view()));
            })
            .expect_err("invalid update_clamp must fail");
            let message = error
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied())
                .unwrap();
            assert_eq!(
                message,
                "update_clamp must be nonnegative or positive infinity"
            );
        }
    }
}

#[test]
fn one_step_fit_obeys_finite_caps_with_exact_and_histogram_candidates() {
    for huber in [false, true] {
        for positive in [false, true] {
            for bins in [None, Some(3)] {
                for cap in [0.0, 2.0_f64.ln()] {
                    let (x, y, _) = fixture(positive);
                    let initial =
                        refinement(huber, cap).initialize(FittingState::new(x.view(), y.view()));
                    let builder = RefinementStrategyParamsBuilder::new()
                        .alpha(0.1)
                        .tilt_tau(0.03)
                        .tilt_rho(0.01)
                        .update_clamp(cap);
                    let builder = if huber { builder.huber() } else { builder.l2() };
                    let params = GridTensorParamsBuilder::new()
                        .n_iter(1)
                        .max_bins(bins)
                        .split_strategy(
                            SplitStrategyParamsBuilder::new()
                                .best_split()
                                .min_interval_samples(1)
                                .build(),
                        )
                        .refinement_strategy(builder.build())
                        .build();
                    let (_, grid) =
                        fit(x.view(), y.view(), &params, &mut StdRng::seed_from_u64(314));
                    let mut binding = false;
                    for row in 0..x.nrows() {
                        let interval = grid.splits[0].partition_point(|&cut| cut <= x[[row, 0]]);
                        let b = grid.backbone_values[0][interval];
                        let d = grid.tilt_values[0][interval];
                        let plus = grid.lambda_plus * b * d.exp();
                        let minus = grid.lambda_minus * b * (-d).exp();
                        let v_plus = plus / initial.lambda_plus;
                        let v_minus = if positive {
                            v_plus
                        } else {
                            minus / initial.lambda_minus
                        };
                        binding |= check_multipliers((v_plus, v_minus), cap);
                        if cap == 0.0 {
                            close(plus - minus, initial.f[row]);
                        }
                    }
                    assert!(binding);
                }
            }
        }
    }
}
