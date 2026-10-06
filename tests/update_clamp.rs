use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, SeedableRng};
use tsl::grid_tensor::{
    action::FittingAction,
    fit,
    params::{
        GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
    },
    reducer::fitting_reducer,
    refinement::RefinementStrategy,
    splitting::{MergeCandidate, ResplitCandidate, SplitCandidate, SplitStrategy},
    state::{AffectedRange, FittingState},
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

fn splitting() -> SplitStrategy {
    SplitStrategy::Best {
        min_interval_samples: 1,
        complexity_penalty: 1.0,
        min_split_loss: 0.0,
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

fn factors(state: &FittingState<'_>, interval: usize) -> (f64, f64) {
    let b = state.backbone_values[0][interval];
    let d = state.tilt_values[0][interval];
    (b * d.exp(), b * (-d).exp())
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-9 * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

fn movement_penalty(v: (f64, f64), strategy: &RefinementStrategy) -> f64 {
    let beta = 0.5 * (v.0.ln() + v.1.ln());
    let delta = 0.5 * (v.0.ln() - v.1.ln());
    strategy.alpha() * beta.powi(2)
        + strategy.tilt_tau() * delta.powi(2)
        + strategy.tilt_rho() * delta.abs()
}

fn actual_gain(
    old: &FittingState<'_>,
    new: &FittingState<'_>,
    strategy: &RefinementStrategy,
) -> f64 {
    // Huber action scores use the weights at proposal time.
    old.residuals
        .iter()
        .zip(new.residuals.iter())
        .map(|(&before, &after)| strategy.weight(before) * (before * before - after * after))
        .sum()
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

fn split_action(state: &FittingState<'_>, index: usize) -> FittingAction {
    let stats = &state.precomputed_statistics;
    FittingAction::ApplySplit {
        split: SplitCandidate {
            col: 0,
            error_reduction: stats.error_reductions_split[0][index],
            allowed_interval_idx: 0,
            index,
            update_left: stats.update_pairs_split_left[0][index],
            update_right: stats.update_pairs_split_right[0][index],
        },
    }
}

fn resplit_action(state: &FittingState<'_>, index: usize) -> FittingAction {
    let stats = &state.precomputed_statistics;
    FittingAction::ApplyResplit {
        resplit: ResplitCandidate {
            col: 0,
            error_reduction: stats.error_reductions_resplit[0][0],
            interval_idx: 0,
            index,
            update_left: stats.update_pairs_resplit_left[0][0],
            update_right: stats.update_pairs_resplit_right[0][0],
        },
    }
}

#[test]
fn split_and_resplit_caps_bind_and_scores_match_applied_predictions() {
    for huber in [false, true] {
        for positive in [false, true] {
            for cap in [0.0, 2.0_f64.ln(), 1e-8] {
                let strategy = refinement(huber, cap);
                let splitter = splitting();
                let (x, y, index) = fixture(positive);
                let mut state =
                    splitter.initialize(strategy.initialize(FittingState::new(x.view(), y.view())));
                for resplit in [false, true] {
                    let action = if resplit {
                        resplit_action(&state, index)
                    } else {
                        split_action(&state, index)
                    };
                    let (gain, updates) = match &action {
                        FittingAction::ApplySplit { split } => (
                            split.error_reduction,
                            [split.update_left, split.update_right],
                        ),
                        FittingAction::ApplyResplit { resplit } => (
                            resplit.error_reduction,
                            [resplit.update_left, resplit.update_right],
                        ),
                        _ => unreachable!(),
                    };
                    assert!(gain.is_finite());
                    let old = state.clone();
                    state = fitting_reducer(state, action, &strategy, &splitter);
                    let mut penalty = 0.0;
                    let mut binding = false;
                    for (side, update) in updates.into_iter().enumerate() {
                        let before = factors(&old, if resplit { side } else { 0 });
                        let after = factors(&state, side);
                        let v = (after.0 / before.0, after.1 / before.1);
                        close(v.0, 1.0 + update.0);
                        close(v.1, 1.0 + if positive { update.0 } else { update.1 });
                        binding |= check_multipliers(v, cap);
                        penalty += movement_penalty(v, &strategy);
                    }
                    close(actual_gain(&old, &state, &strategy) - penalty, gain);
                    if !resplit {
                        assert!(binding, "fixture must exercise the clamp");
                    }
                    if cap == 0.0 {
                        assert_eq!(state.f, old.f);
                        assert_eq!(gain, 0.0);
                    }
                }
            }
        }
    }
}

#[test]
fn merge_caps_reference_multipliers_and_scores_the_applied_factors() {
    for huber in [false, true] {
        for positive in [false, true] {
            for cap in [0.0, 2.0_f64.ln()] {
                let unbounded = refinement(huber, f64::INFINITY);
                let strategy = refinement(huber, cap);
                let splitter = splitting();
                let (x, y, index) = fixture(positive);
                let state = splitter
                    .initialize(unbounded.initialize(FittingState::new(x.view(), y.view())));
                // Equal child factors give an exact reference oracle for the merge.
                let update = if positive { (0.0, 0.0) } else { (19.0, -0.95) };
                let action = FittingAction::ApplySplit {
                    split: SplitCandidate {
                        col: 0,
                        error_reduction: 0.0,
                        allowed_interval_idx: 0,
                        index,
                        update_left: update,
                        update_right: update,
                    },
                };
                let state = fitting_reducer(state, action, &unbounded, &splitter);
                let state = strategy.refresh_error_reduction_caches_for_affected(
                    state,
                    &[AffectedRange {
                        col: 0,
                        point_range: (0, 5),
                        interval_range: (0, 1),
                    }],
                );
                let reference = factors(&state, 0);
                let merged = state.precomputed_statistics.update_pairs_merge[0][0];
                let gain = state.precomputed_statistics.error_reductions_merge[0][0];
                assert!(gain.is_finite());
                let action = FittingAction::ApplyMerge {
                    merge: MergeCandidate {
                        col: 0,
                        error_reduction: gain,
                        interval_idx: 0,
                        index,
                    },
                };
                let old = state.clone();
                let state = fitting_reducer(state, action, &strategy, &splitter);
                let actual = factors(&state, 0);
                close(actual.0, merged.0);
                close(actual.1, merged.1);
                let v = (actual.0 / reference.0, actual.1 / reference.1);
                assert!(
                    check_multipliers(v, cap),
                    "merge fixture must exercise the clamp: huber={huber}, positive={positive}, cap={cap}, multipliers={v:?}"
                );
                close(
                    actual_gain(&old, &state, &strategy) - movement_penalty(v, &strategy),
                    gain,
                );
                if cap == 0.0 {
                    for (&new, &old) in state.f.iter().zip(old.f.iter()) {
                        close(new, old);
                    }
                    close(gain, 0.0);
                }
            }
        }
    }
}

#[test]
fn reducer_caps_oversized_split_and_resplit_actions() {
    for huber in [false, true] {
        for positive in [false, true] {
            for cap in [0.0, 2.0_f64.ln()] {
                let strategy = refinement(huber, cap);
                let splitter = splitting();
                let (x, y, index) = fixture(positive);
                let mut state =
                    splitter.initialize(strategy.initialize(FittingState::new(x.view(), y.view())));
                for resplit in [false, true] {
                    let action = if resplit {
                        FittingAction::ApplyResplit {
                            resplit: ResplitCandidate {
                                col: 0,
                                error_reduction: 0.0,
                                interval_idx: 0,
                                index,
                                update_left: (100.0, -0.999),
                                update_right: (-0.999, 100.0),
                            },
                        }
                    } else {
                        FittingAction::ApplySplit {
                            split: SplitCandidate {
                                col: 0,
                                error_reduction: 0.0,
                                allowed_interval_idx: 0,
                                index,
                                update_left: (100.0, -0.999),
                                update_right: (-0.999, 100.0),
                            },
                        }
                    };
                    let old = state.clone();
                    state = fitting_reducer(state, action, &strategy, &splitter);
                    for side in 0..2 {
                        let before = factors(&old, if resplit { side } else { 0 });
                        let after = factors(&state, side);
                        let v = (after.0 / before.0, after.1 / before.1);
                        assert!(check_multipliers(v, cap));
                        close(v.0, if side == 0 { cap.exp() } else { (-cap).exp() });
                        close(v.1, if positive { v.0 } else { 1.0 / v.0 });
                    }
                }
            }
        }
    }
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

#[test]
fn zero_cap_merge_uses_geometric_reference_for_unequal_intervals() {
    for huber in [false, true] {
        for positive in [false, true] {
            let unbounded = refinement(huber, f64::INFINITY);
            let capped = refinement(huber, 0.0);
            let splitter = splitting();
            let (x, y, index) = fixture(positive);
            let state =
                splitter.initialize(unbounded.initialize(FittingState::new(x.view(), y.view())));
            let split = FittingAction::ApplySplit {
                split: SplitCandidate {
                    col: 0,
                    error_reduction: 0.0,
                    allowed_interval_idx: 0,
                    index,
                    update_left: (-0.5, 1.0),
                    update_right: (1.0, -0.5),
                },
            };
            let state = fitting_reducer(state, split, &unbounded, &splitter);
            let state = capped.refresh_error_reduction_caches_for_affected(
                state,
                &[AffectedRange {
                    col: 0,
                    point_range: (0, 5),
                    interval_range: (0, 1),
                }],
            );
            let left = factors(&state, 0);
            let right = factors(&state, 1);
            let fraction = index as f64 / x.nrows() as f64;
            let reference = (
                (fraction * left.0.ln() + (1.0 - fraction) * right.0.ln()).exp(),
                (fraction * left.1.ln() + (1.0 - fraction) * right.1.ln()).exp(),
            );
            let gain = state.precomputed_statistics.error_reductions_merge[0][0];
            let merge = FittingAction::ApplyMerge {
                merge: MergeCandidate {
                    col: 0,
                    error_reduction: gain,
                    interval_idx: 0,
                    index,
                },
            };
            let old = state.clone();
            let state = fitting_reducer(state, merge, &capped, &splitter);
            close(factors(&state, 0).0, reference.0);
            close(factors(&state, 0).1, reference.1);
            close(actual_gain(&old, &state, &capped), gain);
            assert!(old
                .f
                .iter()
                .zip(state.f.iter())
                .any(|(a, b)| (a - b).abs() > 1e-3));
        }
    }
}
