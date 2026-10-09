use std::sync::Arc;

use ndarray::{s, Array1, Array2};
use rand::{rngs::StdRng, SeedableRng};

use super::{
    cycle5_refresh_tests::{assert_statistics_bits, two_pass_refresh},
    RefinementStrategy,
};
use crate::grid_tensor::{
    fit::{fit_step, initialize_state_with_input},
    params::{GridTensorParamsBuilder, RefinementStrategyParamsBuilder},
    state::{BranchConsensus, ConsensusAnchors, FittingState},
};

fn data(positive: bool, mixed: bool) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((67, 3), |(row, col)| {
        if mixed && col == 2 {
            1.0
        } else {
            ((row * [1, 29, 43][col] + 3 * col) % 67) as f64 / 11.0
        }
    });
    let y = Array1::from_shape_fn(67, |row| {
        let value = x[(row, 0)].sin() * (1.0 + x[(row, 1)]) - 0.4 * x[(row, 2)];
        if positive {
            value.abs() + 1.0
        } else {
            value
        }
    });
    (x, y)
}

fn fitted<'a>(
    x: &'a Array2<f64>,
    y: &'a Array1<f64>,
    bins: Option<u16>,
    huber: bool,
) -> (FittingState<'a>, RefinementStrategy) {
    let refinement = RefinementStrategyParamsBuilder::new()
        .alpha(0.01)
        .tilt_tau(0.01);
    let params = GridTensorParamsBuilder::new()
        .n_iter(8)
        .max_bins(bins)
        .refinement_strategy(
            if huber {
                refinement.huber()
            } else {
                refinement.l2()
            }
            .build(),
        )
        .build();
    let strategy = params.refinement_strategy_params.get_refinement_strategy();
    let split = params.split_strategy_params.get_split_strategy();
    let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
    let mut rng = StdRng::seed_from_u64(29);
    for iteration in 0..3 {
        let advanced;
        (state, advanced) = fit_step(state, iteration, params.n_iter, &strategy, &split, &mut rng);
        if !advanced {
            break;
        }
    }
    assert!(state.boundaries.iter().any(|column| !column.is_empty()));
    (state, strategy)
}

fn reference(n: usize, rho: f64) -> BranchConsensus {
    BranchConsensus {
        weight: rho,
        plus: Array1::from_shape_fn(n, |row| 1.3 + 0.2 * row as f64 / n as f64),
        minus: Array1::from_shape_fn(n, |row| 0.4 + 0.1 * (row as f64).cos()),
        anchors: None,
    }
}

fn prepare_reference<'a>(
    mut state: FittingState<'a>,
    strategy: &RefinementStrategy,
    rho: f64,
) -> FittingState<'a> {
    state.branch_consensus = Some(reference(state.n, rho));
    two_pass_refresh(strategy, state)
}

fn assert_slice_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{actual:?} != {expected:?}"
        );
    }
}

fn assert_refresh(state: FittingState<'_>, strategy: &RefinementStrategy) {
    let expected = two_pass_refresh(strategy, state.clone());
    let actual = strategy.refresh_statistics(state);
    assert_statistics_bits(&actual, &expected);
    assert!(!actual.linear_statistics_stale);
}

#[test]
fn cycle7_refresh_eligible_rows_match_oracle_with_stale_and_signed_inputs() {
    for (bins, mixed) in [(None, false), (Some(8), false), (Some(8), true)] {
        for positive in [false, true] {
            let (x, y) = data(positive, mixed);
            let (initial, strategy) = fitted(&x, &y, bins, false);
            if positive {
                assert_eq!(initial.lambda_minus.to_bits(), 0.0_f64.to_bits());
            }
            for rho in [0.0, -0.0, 0.3] {
                for scenario in 0..4 {
                    let mut state = prepare_reference(initial.clone(), &strategy, rho);
                    for row in 0..state.n {
                        match scenario {
                            1 => {
                                state.f_plus[row] = if row % 2 == 0 { -0.0 } else { 0.0 };
                                state.f_minus[row] = if row % 3 == 0 { -0.0 } else { 0.0 };
                                state.r_tilde[row] = if row % 5 == 0 { -0.0 } else { 0.0 };
                            }
                            2 => {
                                let magnitude =
                                    [f64::from_bits(1), 1e-120, 1e120, 1e-40, 1e40][row % 5];
                                state.f_plus[row] = magnitude;
                                state.f_minus[row] = if positive { 0.0 } else { magnitude * 0.5 };
                                state.r_tilde[row] = row as f64 - 33.0;
                            }
                            3 => {
                                let target = state.branch_consensus.as_mut().unwrap();
                                target.plus[row] = state.f_plus[row] * (1.0 + f64::EPSILON);
                                target.minus[row] = state.f_minus[row] * (1.0 - f64::EPSILON);
                            }
                            _ => {}
                        }
                        // L2 linear terms use r_tilde even when the outer residual cache differs.
                        state.residuals[row] = state.r_tilde[row] + 17.0;
                    }
                    state.precomputed_statistics.c_s11[0] = -0.0;
                    state.precomputed_statistics.c_s12[0] = -0.0;
                    state.precomputed_statistics.c_t1.fill(f64::NAN);
                    state.precomputed_statistics.c_t2.fill(f64::NAN);
                    state.linear_statistics_stale = true;
                    state.sampled_split_scoring = true;
                    state.split_scores_stale = true;
                    let mut direct = state.clone();
                    assert!(RefinementStrategy::refresh_full_row_linear_slices(
                        &mut direct
                    ));
                    assert!(direct.linear_statistics_stale);
                    let expected = two_pass_refresh(&strategy, state.clone());
                    assert_slice_bits(
                        &direct.precomputed_statistics.c_t1,
                        &expected.precomputed_statistics.c_t1,
                    );
                    assert_slice_bits(
                        &direct.precomputed_statistics.c_t2,
                        &expected.precomputed_statistics.c_t2,
                    );
                    assert_slice_bits(
                        &direct.precomputed_statistics.c_s11,
                        &state.precomputed_statistics.c_s11,
                    );
                    assert_slice_bits(
                        &direct.precomputed_statistics.c_s22,
                        &state.precomputed_statistics.c_s22,
                    );
                    assert_slice_bits(
                        &direct.precomputed_statistics.c_s12,
                        &state.precomputed_statistics.c_s12,
                    );
                    assert_refresh(state, &strategy);
                }
            }
        }
    }
}

fn input<'s>(state: &'s mut FittingState<'_>, index: usize) -> &'s mut Array1<f64> {
    match index {
        0 => &mut state.f_plus,
        1 => &mut state.f_minus,
        2 => &mut state.r_tilde,
        3 => &mut state.branch_consensus.as_mut().unwrap().plus,
        4 => &mut state.branch_consensus.as_mut().unwrap().minus,
        5 => &mut state.residuals,
        _ => unreachable!(),
    }
}

fn with_stride(values: &Array1<f64>, reverse: bool) -> Array1<f64> {
    if reverse {
        Array1::from_iter(values.iter().rev().copied()).slice_move(s![..;-1])
    } else {
        Array1::from_shape_fn(values.len() * 2, |row| {
            if row % 2 == 0 {
                values[row / 2]
            } else {
                123.0
            }
        })
        .slice_move(s![..;2])
    }
}

#[test]
fn cycle7_refresh_each_strided_input_matches_oracle() {
    let (x, y) = data(false, false);
    let (initial, strategy) = fitted(&x, &y, Some(8), false);
    let initial = prepare_reference(initial, &strategy, 0.3);
    for index in 0..6 {
        for reverse in [false, true] {
            let mut state = initial.clone();
            let values = input(&mut state, index);
            *values = with_stride(values, reverse);
            assert!(values.as_slice().is_none());
            let mut probe = state.clone();
            assert_eq!(
                RefinementStrategy::refresh_full_row_linear_slices(&mut probe),
                index == 5
            );
            assert_refresh(state, &strategy);
        }
    }
}

#[test]
fn cycle7_refresh_buffer_lengths_preserve_tails_and_bounds_failures() {
    let (x, y) = data(false, false);
    let (initial, strategy) = fitted(&x, &y, Some(8), false);
    let initial = prepare_reference(initial, &strategy, 0.3);
    for index in 0..8 {
        let mut longer = initial.clone();
        if index < 6 {
            let values = input(&mut longer, index);
            *values = Array1::from_iter(values.iter().copied().chain([123.0, -0.0, 456.0]));
        } else {
            let values = if index == 6 {
                &mut longer.precomputed_statistics.c_t1
            } else {
                &mut longer.precomputed_statistics.c_t2
            };
            values.extend([123.0, -0.0, 456.0]);
        }
        let mut probe = longer.clone();
        assert!(RefinementStrategy::refresh_full_row_linear_slices(
            &mut probe
        ));
        if index >= 6 {
            let values = if index == 6 {
                &probe.precomputed_statistics.c_t1
            } else {
                &probe.precomputed_statistics.c_t2
            };
            assert_slice_bits(&values[probe.n..], &[123.0, -0.0, 456.0]);
        }
        assert_refresh(longer, &strategy);

        let mut shorter = initial.clone();
        let end = shorter.n - 1;
        if index < 6 {
            let values = input(&mut shorter, index);
            *values = values.clone().slice_move(s![..end]);
        } else if index == 6 {
            shorter.precomputed_statistics.c_t1.truncate(end);
        } else {
            shorter.precomputed_statistics.c_t2.truncate(end);
        }
        let mut probe = shorter.clone();
        assert!(!RefinementStrategy::refresh_full_row_linear_slices(
            &mut probe
        ));
        let expected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            two_pass_refresh(&strategy, shorter.clone())
        }));
        let actual = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            strategy.refresh_statistics(shorter)
        }));
        assert!(expected.is_err());
        assert!(actual.is_err());
    }
}

#[test]
fn cycle7_refresh_weight_and_support_fallbacks_match_oracle() {
    let (x, y) = data(false, false);
    for context in 0..6 {
        let (mut state, strategy) = fitted(&x, &y, Some(8), context == 1);
        if context != 0 {
            state = prepare_reference(state, &strategy, 0.3);
        }
        match context {
            2 => state.branch_consensus.as_mut().unwrap().weight = 0.7,
            3 | 4 => {
                let anchors = Arc::new(ConsensusAnchors::new(
                    state.n,
                    (0..state.n).step_by(5).collect(),
                ));
                let mut target = reference(anchors.rows.len(), 0.3);
                target.anchors = Some(anchors);
                state.branch_consensus = Some(target);
                if context == 4 {
                    state = two_pass_refresh(&strategy, state);
                }
            }
            5 => state.branch_consensus = None,
            _ => {}
        }
        state.linear_statistics_stale = true;
        state.precomputed_statistics.c_t1.fill(f64::NAN);
        state.precomputed_statistics.c_t2.fill(f64::NAN);
        assert_refresh(state, &strategy);
    }
}
