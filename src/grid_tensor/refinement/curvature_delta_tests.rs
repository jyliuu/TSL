use ndarray::{s, Array1, Array2};

use super::{streamed_refresh_tests::assert_statistics_bits, RefinementStrategy};
use crate::grid_tensor::{
    fit::initialize_state_with_input,
    params::GridTensorParamsBuilder,
    state::{BranchConsensus, FittingState},
};

// A row-at-a-time collector is the arithmetic and partial-mutation oracle.
fn row_at_a_time_deltas(
    refinement: &RefinementStrategy,
    state: &mut FittingState<'_>,
    col: usize,
    start: usize,
    end: usize,
) -> Vec<[f64; 3]> {
    let updated_points = &state.precomputed_statistics.sorted_indices[col][start..end];
    let mut deltas = Vec::with_capacity(updated_points.len());
    for &row in updated_points {
        let point = state.point_curvature_statistics(row, refinement.weight(state.residuals[row]));
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

fn fixture() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((31, 3), |(row, col)| {
        ((row * [7, 11, 19][col]) % 31) as f64 / 7.0
    });
    let y = Array1::from_shape_fn(31, |row| x[(row, 0)].sin() - 0.3 * x[(row, 1)]);
    (x, y)
}

fn initial<'a>(
    x: &'a Array2<f64>,
    y: &'a Array1<f64>,
    rho: f64,
) -> (FittingState<'a>, RefinementStrategy) {
    let params = GridTensorParamsBuilder::new().max_bins(Some(8)).build();
    let refinement = params.refinement_strategy_params.get_refinement_strategy();
    let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
    state.branch_consensus = Some(BranchConsensus {
        weight: rho,
        plus: Array1::from_shape_fn(state.n, |row| 1.2 + row as f64 / 17.0),
        minus: Array1::from_shape_fn(state.n, |row| 0.4 + row as f64 / 29.0),
        anchors: None,
    });
    for row in 0..state.n {
        state.f_plus[row] = 1.25 + row as f64 / 13.0;
        state.f_minus[row] = 0.25 + row as f64 / 29.0;
        state.residuals[row] = row as f64 - 15.0;
        state.r_tilde[row] = row as f64 / 7.0 - 3.0;
    }
    (state, refinement)
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

fn assert_state(actual: &FittingState<'_>, expected: &FittingState<'_>) {
    assert_statistics_bits(actual, expected);
    for (actual, expected) in [
        (&actual.f_plus, &expected.f_plus),
        (&actual.f_minus, &expected.f_minus),
        (&actual.f, &expected.f),
        (&actual.y_hat, &expected.y_hat),
        (&actual.r_tilde, &expected.r_tilde),
        (&actual.residuals, &expected.residuals),
    ] {
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual.strides(), expected.strides());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

fn compare(
    mut actual: FittingState<'_>,
    refinement: &RefinementStrategy,
    start: usize,
    end: usize,
    panics: bool,
) -> Option<Vec<[f64; 3]>> {
    let mut expected = actual.clone();
    let expected_deltas = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        row_at_a_time_deltas(refinement, &mut expected, 0, start, end)
    }));
    let actual_deltas = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        refinement.collect_binned_curvature_deltas(&mut actual, 0, start, end)
    }));
    assert_eq!(expected_deltas.is_err(), panics);
    assert_eq!(actual_deltas.is_err(), panics);
    assert_state(&actual, &expected);
    match (actual_deltas, expected_deltas) {
        (Ok(actual), Ok(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                for (&actual, &expected) in actual.iter().zip(expected) {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
            Some(actual)
        }
        (Err(actual), Err(expected)) => {
            assert_eq!(
                panic_message(actual.as_ref()),
                panic_message(expected.as_ref())
            );
            None
        }
        _ => unreachable!(),
    }
}

#[test]
fn curvature_deltas_preserve_arithmetic_and_row_order() {
    let (x, y) = fixture();
    for rho in [0.0, -0.0, 0.3] {
        let (initial, refinement) = initial(&x, &y, rho);
        for scenario in 0..4 {
            let mut state = initial.clone();
            for row in 0..state.n {
                match scenario {
                    0 => {}
                    1 => {
                        state.f_plus[row] = if row % 2 == 0 { -0.0 } else { 0.0 };
                        state.f_minus[row] = if row % 3 == 0 { -0.0 } else { 0.0 };
                        state.precomputed_statistics.curvature[row].s11 = -0.0;
                        state.precomputed_statistics.curvature[row].s22 = 0.0;
                        state.precomputed_statistics.curvature[row].s12 = -0.0;
                    }
                    2 => {
                        let value = [f64::from_bits(1), 1e-140, 1e140, 1e-70, 1e70][row % 5];
                        state.f_plus[row] = value;
                        state.f_minus[row] = value * 0.5;
                        state.precomputed_statistics.curvature[row].s11 = -1e280;
                        state.precomputed_statistics.curvature[row].s22 = 1e280;
                        state.precomputed_statistics.curvature[row].s12 = -1e280;
                    }
                    _ => state.f_minus[row] = 0.0,
                }
            }
            let (start, end) = match scenario {
                0 => (0, state.n),
                1 => (7, 8),
                2 => (3, state.n - 2),
                _ => {
                    state.lambda_minus = 0.0;
                    assert!(state.is_stage1_positive_only());
                    state.precomputed_statistics.sorted_indices[0] = vec![2, 7, 2, 7, 9];
                    (0, 5)
                }
            };
            let deltas = compare(state, &refinement, start, end, false).unwrap();
            if scenario == 3 {
                for delta in &deltas[2..4] {
                    for value in delta {
                        assert_eq!(value.to_bits(), 0.0_f64.to_bits());
                    }
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
        3 => &mut state.residuals,
        4 => &mut state.branch_consensus.as_mut().unwrap().plus,
        5 => &mut state.branch_consensus.as_mut().unwrap().minus,
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
fn curvature_deltas_preserve_each_shape_and_stride_fallback() {
    let (x, y) = fixture();
    let (initial, refinement) = initial(&x, &y, 0.3);
    for index in 0..6 {
        for variant in 0..4 {
            let mut state = initial.clone();
            let n = state.n;
            let values = input(&mut state, index);
            *values = match variant {
                0 => Array1::from_iter(values.iter().take(n - 1).copied()),
                1 => Array1::from_iter(values.iter().copied().chain([123.0])),
                2 => with_stride(values, false),
                _ => with_stride(values, true),
            };
            if variant >= 2 {
                assert!(values.as_slice().is_none());
                assert!(input(&mut state.clone(), index).as_slice().is_none());
            }
            compare(state, &refinement, 0, n, variant == 0);
        }
    }
    // One vector holds all three curvature statistics; one shape sweep covers it.
    for shorter in [false, true] {
        let mut state = initial.clone();
        let n = state.n;
        let cache = &mut state.precomputed_statistics.curvature;
        if shorter {
            cache.truncate(n - 1);
        } else {
            cache.push(crate::grid_tensor::state::CurvatureCache {
                s11: -0.0,
                s22: -0.0,
                s12: -0.0,
            });
        }
        compare(state, &refinement, 0, n, shorter);
    }
}

#[test]
fn curvature_deltas_preserve_empty_and_invalid_row_failures() {
    let (x, y) = fixture();
    let (initial, refinement) = initial(&x, &y, 0.3);
    for scenario in 0..6 {
        let mut state = initial.clone();
        let n = state.n;
        let end = match scenario {
            0 => {
                state.f_plus = Array1::zeros(0);
                state.residuals = Array1::zeros(0);
                state.precomputed_statistics.curvature.clear();
                state.precomputed_statistics.sorted_indices[0] = vec![usize::MAX];
                let expected = state.clone();
                let actual = refinement.update_binned_curvature_statistics(state.clone(), 0, 0, 0);
                assert_state(&actual, &expected);
                0
            }
            1 => {
                state.precomputed_statistics.sorted_indices[0] = vec![n, 1];
                2
            }
            2 => {
                state.precomputed_statistics.sorted_indices[0] = vec![3, 1, n, 2];
                4
            }
            3 => {
                state.precomputed_statistics.sorted_indices[0] = vec![usize::MAX];
                1
            }
            4 => {
                state.r_tilde = Array1::from_iter(state.r_tilde.iter().take(n - 2).copied());
                let reference = state.branch_consensus.as_mut().unwrap();
                reference.plus = Array1::from_iter(reference.plus.iter().take(n - 1).copied());
                state.precomputed_statistics.sorted_indices[0] = vec![0, n - 1, 3];
                3
            }
            _ => {
                let reference = state.branch_consensus.as_mut().unwrap();
                reference.plus = Array1::from_iter(reference.plus.iter().take(n - 1).copied());
                state.precomputed_statistics.curvature.truncate(n - 2);
                state.precomputed_statistics.sorted_indices[0] = vec![0, n - 1, 3];
                3
            }
        };
        compare(state, &refinement, 0, end, scenario != 0);
    }
}
