use std::sync::Arc;

use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, SeedableRng};

use super::{
    cycle5_refresh_tests::{assert_statistics_bits, two_pass_refresh},
    RefinementStrategy,
};
use crate::grid_tensor::{
    fit::{fit_step, initialize_state_with_input},
    params::{
        GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
    },
    state::{BranchConsensus, FittingState, PrefixStats},
};

fn fixture(mixed: bool) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((97, 4), |(row, col)| match col {
        0 => ((row * 29) % 2) as f64,
        1 => ((row * 43) % 97) as f64 / 17.0,
        2 if mixed => 2.0,
        2 => ((row * 71) % 11) as f64,
        _ => ((row * 89) % 5) as f64,
    });
    let y = Array1::from_shape_fn(97, |row| {
        x[(row, 1)].sin() * (1.0 + x[(row, 2)]) - 0.3 * x[(row, 3)] - 1.4 * x[(row, 0)]
    });
    (x, y)
}

fn prepared<'a>(
    x: &'a Array2<f64>,
    y: &'a Array1<f64>,
    bins: Option<u16>,
) -> (FittingState<'a>, RefinementStrategy) {
    let params = GridTensorParamsBuilder::new()
        .n_iter(8)
        .max_bins(bins)
        .refinement_strategy(
            RefinementStrategyParamsBuilder::new()
                .l2()
                .alpha(0.01)
                .tilt_tau(0.01)
                .build(),
        )
        .split_strategy(
            SplitStrategyParamsBuilder::new()
                .random_split()
                .split_try(10)
                .min_interval_samples(1)
                .complexity_penalty(0.01)
                .build(),
        )
        .build();
    let refinement = params.refinement_strategy_params.get_refinement_strategy();
    let split = params.split_strategy_params.get_split_strategy();
    let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
    let mut rng = StdRng::seed_from_u64(29);
    for iteration in 0..3 {
        let advanced;
        (state, advanced) = fit_step(
            state,
            iteration,
            params.n_iter,
            &refinement,
            &split,
            &mut rng,
        );
        if !advanced {
            break;
        }
    }
    assert!(state.boundaries.iter().any(|column| !column.is_empty()));
    state.branch_consensus = Some(BranchConsensus {
        weight: 0.3,
        plus: state.f_plus.clone(),
        minus: state.f_minus.clone(),
        anchors: None,
    });
    (two_pass_refresh(&refinement, state), refinement)
}

#[test]
fn cycle11_linear_bin_slices_preserve_numeric_and_storage_order() {
    for (bins, mixed) in [(Some(16), false), (Some(16), true), (None, false)] {
        let (x, y) = fixture(mixed);
        let (initial, refinement) = prepared(&x, &y, bins);
        let bucket_counts: Vec<_> = initial
            .precomputed_statistics
            .binned_prefix_sums
            .iter()
            .map(Vec::len)
            .collect();
        assert_eq!(
            bucket_counts,
            if bins.is_none() {
                vec![0; 4]
            } else if mixed {
                vec![2, 16, 0, 5]
            } else {
                vec![2, 16, 11, 5]
            }
        );
        for scenario in 0..4 {
            let mut state = initial.clone();
            for row in 0..state.n {
                let (plus, minus, residual) = match scenario {
                    0 => (
                        1.0 + row as f64 / 100.0,
                        0.25 + (row % 7) as f64 / 50.0,
                        if row % 2 == 0 {
                            row as f64 * 0.25
                        } else {
                            -(row as f64) * 0.5
                        },
                    ),
                    1 => (
                        1.0,
                        [0.5, 1.0, 2.0, 4.0][row % 4],
                        [1e16, 1.0, -1e16, -1.0][row % 4],
                    ),
                    2 => (
                        if row % 2 == 0 { -0.0 } else { 0.0 },
                        if row % 3 == 0 { -0.0 } else { 0.0 },
                        if row % 5 == 0 { -0.0 } else { 0.0 },
                    ),
                    _ => {
                        let magnitude = [f64::from_bits(1), 1e-140, 1e140, 1e-70, 1e70][row % 5];
                        (magnitude, magnitude * 0.5, row as f64 - 48.0)
                    }
                };
                state.f_plus[row] = plus;
                state.f_minus[row] = minus;
                state.r_tilde[row] = residual;
                state.residuals[row] = residual;
                let reference = state.branch_consensus.as_mut().unwrap();
                reference.plus[row] = plus;
                reference.minus[row] = minus;
            }
            state.precomputed_statistics.curvature[0].s11 = -0.0;
            state.precomputed_statistics.curvature[0].s12 = -0.0;
            state.precomputed_statistics.c_t1.fill(f64::NAN);
            state.precomputed_statistics.c_t2.fill(f64::NAN);
            state.linear_statistics_stale = true;
            state.sampled_split_scoring = true;
            state.split_scores_stale = true;
            let expected = two_pass_refresh(&refinement, state.clone());
            let actual = refinement.refresh_statistics(state);
            assert_statistics_bits(&actual, &expected);
            assert!(!actual.linear_statistics_stale);
        }
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        payload
            .downcast_ref::<&str>()
            .copied()
            .expect("string panic payload")
    }
}

#[test]
fn cycle11_linear_bin_slices_preserve_map_tails_and_panic_order() {
    let (x, y) = fixture(false);
    let (initial, refinement) = prepared(&x, &y, Some(16));
    for scenario in 0..17 {
        let mut state = initial.clone();
        let n = state.n;
        let statistics = &mut state.precomputed_statistics;
        let mut map = statistics.bin_indices[0].to_vec();
        let invalid_bin = (statistics.binned_prefix_sums[0].len() + 3) as u16;
        match scenario {
            0 => {
                map.pop();
            }
            1 => map.clear(),
            2 => {
                map.push(0);
                statistics.c_t1.push(3.25);
                statistics.c_t2.push(-7.5);
            }
            3 => {
                statistics.c_t1.extend([123.0, -0.0]);
                statistics.c_t2.extend([-456.0, 0.0]);
            }
            4 => statistics.binned_prefix_sums[0].extend(
                [PrefixStats {
                    s11: -0.0,
                    s22: 17.0,
                    s12: -31.0,
                    t1: f64::NAN,
                    t2: f64::NAN,
                }; 2],
            ),
            5 => {
                let last = statistics.binned_prefix_sums[1].len() - 1;
                let mut sparse = vec![0; n];
                sparse[n - 1] = last as u16;
                statistics.bin_indices[1] = Arc::from(sparse);
            }
            6 => map.push(invalid_bin),
            7 => {
                map.push(invalid_bin);
                statistics.c_t1.push(3.25);
            }
            8 => {
                map.push(0);
                statistics.c_t1.push(3.25);
            }
            9 => statistics.binned_prefix_sums[0].truncate(1),
            10 => {
                map.clear();
                statistics.binned_prefix_sums[0].clear();
            }
            11 => statistics.c_t1.truncate(n - 1),
            12 => statistics.c_t2.truncate(n - 1),
            13 => {
                map.push(0);
                statistics.c_t1.extend([3.25, -0.0]);
                statistics.c_t2.extend([-7.5, 0.0, 123.0]);
            }
            14 => {
                map.push(0);
                statistics.c_t1.extend([3.25, -0.0, 123.0]);
                statistics.c_t2.extend([-7.5, 0.0]);
            }
            15 => {
                map[0] = invalid_bin;
                map.push(0);
                statistics.c_t2.push(-7.5);
            }
            _ => {
                map[0] = invalid_bin;
                map.push(0);
                statistics.c_t1.push(3.25);
            }
        }
        statistics.bin_indices[0] = Arc::from(map);
        let expected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            two_pass_refresh(&refinement, state.clone())
        }));
        let actual = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            refinement.refresh_statistics(state)
        }));
        let succeeds = scenario < 6 || scenario == 13 || scenario == 14;
        assert_eq!(expected.is_ok(), succeeds, "oracle scenario {scenario}");
        assert_eq!(actual.is_ok(), succeeds, "sliced scenario {scenario}");
        match (actual, expected) {
            (Ok(actual), Ok(expected)) => assert_statistics_bits(&actual, &expected),
            (Err(actual), Err(expected)) => {
                assert_eq!(
                    panic_message(actual.as_ref()),
                    panic_message(expected.as_ref()),
                    "scenario {scenario}"
                );
                if scenario == 6 || scenario == 8 {
                    assert_eq!(
                        panic_message(expected.as_ref()),
                        format!("index out of bounds: the len is {n} but the index is {n}")
                    );
                } else if scenario == 7 || scenario == 15 || scenario == 16 {
                    assert_eq!(
                        panic_message(expected.as_ref()),
                        format!("index out of bounds: the len is 2 but the index is {invalid_bin}")
                    );
                }
            }
            _ => unreachable!(),
        }
    }
}
