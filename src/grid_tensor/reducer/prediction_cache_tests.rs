use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

use super::{
    reduce_grid_predictions_and_residuals_merge_from_partial_products,
    reduce_grid_predictions_and_residuals_resplit_two_tensor_factors,
    reduce_grid_predictions_and_residuals_two_tensor_factors,
};
use crate::grid_tensor::state::FittingState;

fn fixture<'a>(
    x: ArrayView2<'a, f64>,
    labels: ArrayView1<'a, f64>,
    strided_cache: Option<usize>,
) -> FittingState<'a> {
    let mut state = FittingState::new(x, labels);
    state.precomputed_statistics.sorted_indices[0] = vec![5, 2, 7, 0, 6, 1, 4, 3];
    state.f_plus = Array1::from_vec(vec![
        0.0,
        -0.0,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1.0 + f64::EPSILON,
        1e150,
        1e-150,
        3.5,
    ]);
    state.f_minus = Array1::from_vec(vec![
        -0.0,
        0.0,
        f64::from_bits(2),
        f64::MIN_POSITIVE,
        1.0,
        5e149,
        2e-150,
        7.0,
    ]);
    for (index, cache) in [
        &mut state.f_plus,
        &mut state.f_minus,
        &mut state.f,
        &mut state.y_hat,
        &mut state.residuals,
        &mut state.r_tilde,
    ]
    .into_iter()
    .enumerate()
    {
        if index >= 2 {
            *cache = Array1::from_shape_fn(8, |row| 10.0 * index as f64 + row as f64);
        }
        if strided_cache == Some(index) {
            *cache = Array1::from_shape_fn(16, |row| {
                if row % 2 == 0 {
                    cache[row / 2]
                } else {
                    -1234.0
                }
            })
            .slice_move(s![..;2]);
            assert!(cache.as_slice().is_none());
        }
    }
    state
}

fn assert_cache_bits(actual: &FittingState<'_>, expected: &FittingState<'_>) {
    for (actual, expected) in [
        (&actual.f_plus, &expected.f_plus),
        (&actual.f_minus, &expected.f_minus),
        (&actual.f, &expected.f),
        (&actual.y_hat, &expected.y_hat),
        (&actual.residuals, &expected.residuals),
        (&actual.r_tilde, &expected.r_tilde),
    ] {
        for (row, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "row {row}");
        }
    }
    assert_eq!(
        actual.current_error.to_bits(),
        expected.current_error.to_bits()
    );
    assert_eq!(actual.interval_id, expected.interval_id);
    assert_eq!(actual.boundaries, expected.boundaries);
}

// The direct ndarray oracle retains each cache read and write of the row recurrence.
fn direct_scatter(
    state: &mut FittingState<'_>,
    start: usize,
    end: usize,
    plus: f64,
    minus: f64,
    partial: Option<(&[f64], &[f64])>,
) {
    for offset in start..end {
        let row = state.precomputed_statistics.sorted_indices[0][offset];
        if let Some((partial_plus, partial_minus)) = partial {
            state.f_plus[row] = partial_plus[row] * plus;
            state.f_minus[row] = partial_minus[row] * minus;
        } else {
            state.f_plus[row] *= plus;
            state.f_minus[row] *= minus;
        }
        state.f[row] = state.f_plus[row] - state.f_minus[row];
        state.y_hat[row] = state.f[row];
        state.residuals[row] = state.labels[row] - state.y_hat[row];
        state.r_tilde[row] = state.residuals[row];
    }
}

fn compare_updates<'a>(x: ArrayView2<'a, f64>, labels: ArrayView1<'a, f64>, cache: Option<usize>) {
    for (start, split, end) in [(0, 0, 0), (0, 0, 8), (0, 8, 8), (1, 4, 7)] {
        for action in 0..3 {
            let state = fixture(x, labels, cache);
            let mut expected = state.clone();
            let actual = match action {
                0 => {
                    direct_scatter(&mut expected, start, split, 0.625 / 1.25, 1.5 / 0.75, None);
                    direct_scatter(&mut expected, split, end, 2.5 / 1.25, 0.375 / 0.75, None);
                    reduce_grid_predictions_and_residuals_two_tensor_factors(
                        state,
                        0,
                        split,
                        &(start, end),
                        1.25,
                        0.75,
                        0.625,
                        1.5,
                        2.5,
                        0.375,
                    )
                }
                1 => {
                    direct_scatter(&mut expected, start, split, 0.625 / 1.25, 1.5 / 0.75, None);
                    direct_scatter(&mut expected, split, end, 3.125 / 2.5, 0.625 / 1.25, None);
                    reduce_grid_predictions_and_residuals_resplit_two_tensor_factors(
                        state,
                        0,
                        split,
                        &(start, end),
                        1.25,
                        0.75,
                        2.5,
                        1.25,
                        0.625,
                        1.5,
                        3.125,
                        0.625,
                    )
                }
                _ => {
                    let partial_plus = [3.0, -0.0, f64::from_bits(1), 1e150, 0.0, 1.0, 1e-150, 7.0];
                    let partial_minus = [1.0, 0.0, 0.0, 5e149, -0.0, 2.0, 2e-150, 3.5];
                    direct_scatter(
                        &mut expected,
                        start,
                        end,
                        0.5,
                        1.25,
                        Some((&partial_plus, &partial_minus)),
                    );
                    reduce_grid_predictions_and_residuals_merge_from_partial_products(
                        state,
                        0,
                        &(start, end),
                        &partial_plus,
                        &partial_minus,
                        0.5,
                        1.25,
                    )
                }
            };
            assert_cache_bits(&actual, &expected);
        }
    }
}

#[test]
fn prediction_cache_scatter_preserves_signed_zero_and_extreme_finite_bits() {
    let x = Array2::from_shape_fn((8, 1), |(row, _)| row as f64);
    let labels = Array1::from_vec(vec![
        -0.0,
        0.0,
        f64::from_bits(1),
        1e150,
        3.0,
        -7.0,
        1e-150,
        2.0,
    ]);
    compare_updates(x.view(), labels.view(), None);
}

#[test]
fn prediction_cache_scatter_preserves_strided_labels_and_state_arrays() {
    let x = Array2::from_shape_fn((8, 1), |(row, _)| row as f64);
    let labels =
        Array1::from_shape_fn(16, |row| if row % 2 == 0 { row as f64 - 4.0 } else { 99.0 });
    let strided = labels.slice(s![..;2]);
    let reversed = strided.slice(s![..;-1]);
    assert!(strided.as_slice().is_none());
    assert!(reversed.as_slice().is_none());
    let contiguous = strided.to_owned();
    for labels in [contiguous.view(), strided, reversed] {
        compare_updates(x.view(), labels, None);
        for cache in 0..6 {
            compare_updates(x.view(), labels, Some(cache));
        }
    }
}
