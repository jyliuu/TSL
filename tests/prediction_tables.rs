use ndarray::{s, Array1, Array2, ArrayView2};
use tsl::grid_tensor::GridTensor;
use tsl::stage_predictor::{extract_two_tensor_predictions_unscaled, Aggregation, StagePredictor};

fn grid_with_factors(backbone: Vec<Vec<f64>>, tilt: Vec<Vec<f64>>) -> GridTensor {
    let splits: Vec<Vec<f64>> = backbone
        .iter()
        .map(|axis| (1..axis.len()).map(|bin| bin as f64 - 1.5).collect())
        .collect();
    let intervals = splits
        .iter()
        .map(|axis| {
            std::iter::once(f64::NEG_INFINITY)
                .chain(axis.iter().copied())
                .zip(axis.iter().copied().chain(std::iter::once(f64::INFINITY)))
                .collect()
        })
        .collect();
    let counts = backbone.iter().map(|axis| vec![1; axis.len()]).collect();
    GridTensor::new_two_tensor(splits, counts, intervals, backbone, tilt, 1.375, 0.625)
}

fn reference_components(grid: &GridTensor, x: ArrayView2<f64>) -> (Array1<f64>, Array1<f64>) {
    let mut plus = Array1::zeros(x.nrows());
    let mut minus = Array1::zeros(x.nrows());
    for row in 0..x.nrows() {
        let mut fp = grid.lambda_plus;
        let mut fm = grid.lambda_minus;
        for axis in 0..x.ncols() {
            let bin = grid.splits[axis].partition_point(|&split| split <= x[[row, axis]]);
            let bin = bin.min(grid.backbone_values[axis].len() - 1);
            let b = grid.backbone_values[axis][bin];
            let d = grid.tilt_values[axis][bin];
            fp *= b * d.exp();
            fm *= b * (-d).exp();
        }
        plus[row] = fp;
        minus[row] = fm;
    }
    (plus, minus)
}

fn assert_identical(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{actual} != {expected}"
        );
    }
}

fn assert_predictions(grid: &GridTensor, x: ArrayView2<f64>) {
    let (expected_plus, expected_minus) = reference_components(grid, x);
    let (plus, minus) = extract_two_tensor_predictions_unscaled(grid, x);
    let predicted = grid.predict(x);
    let mut stage = StagePredictor::new_exact(vec![grid.clone()], Aggregation::Combined);
    stage.scaling_plus = Some(0.75);
    stage.scaling_minus = Some(-0.5);
    let stage_predictions = stage.predict(x);
    for row in 0..x.nrows() {
        assert_identical(plus[row], expected_plus[row]);
        assert_identical(minus[row], expected_minus[row]);
        let expected = expected_plus[row] - expected_minus[row];
        assert_identical(predicted[row], expected);
        assert_identical(grid.predict_single(&x.row(row).to_vec()), expected);
        assert_identical(
            stage_predictions[row],
            expected_plus[row] * 0.75 + (-expected_minus[row]) * -0.5,
        );
    }
}

#[test]
fn batch_predictions_match_scalar_arithmetic_on_boundaries_and_strided_views() {
    let mut grid = grid_with_factors(
        vec![vec![0.0, 1.2, 2.7], vec![0.7, 1.1], vec![3.0]],
        vec![vec![-1.7, 0.25, 2.0], vec![0.5, -0.75], vec![0.125]],
    );
    grid.scaling = 891.0;
    let values = [f64::NEG_INFINITY, -0.5, 0.5, 3.0, f64::INFINITY, f64::NAN];
    let x = Array2::from_shape_fn((values.len(), 3), |(row, col)| {
        values[(row + col) % values.len()]
    });
    assert_predictions(&grid, x.view());
    let padded = Array2::from_shape_fn((2 * x.nrows(), 6), |(row, col)| x[[row / 2, col / 2]]);
    assert_predictions(&grid, padded.slice(s![..;2, ..;2]));
    assert_predictions(&grid, padded.slice(s![..;-2, ..;-2]));
    assert_predictions(&grid, x.slice(s![0..0, ..]));
    assert_predictions(&grid, x.slice(s![0..1, ..]));
}

#[test]
fn batch_predictions_preserve_zero_underflow_and_overflow_arithmetic() {
    let grid = grid_with_factors(
        vec![vec![0.0, 1e-200, 1.0], vec![1e-200, 0.0, 1.0]],
        vec![vec![711.0, -710.0, 700.0], vec![700.0, -711.0, -700.0]],
    );
    let x = Array2::from_shape_fn((9, 2), |(row, axis)| {
        if axis == 0 {
            (row / 3) as f64 - 1.0
        } else {
            (row % 3) as f64 - 1.0
        }
    });
    assert_predictions(&grid, x.view());
}

#[test]
fn batch_factor_tables_reflect_grid_mutation_and_component_index_clamping() {
    let mut grid = grid_with_factors(vec![vec![1.2, 2.7]], vec![vec![-0.25, 0.5]]);
    let x = Array2::from_shape_vec((3, 1), vec![-2.0, -0.5, 2.0]).unwrap();
    assert_predictions(&grid, x.view());
    grid.backbone_values[0][1] = 4.0;
    grid.tilt_values[0][0] = 0.75;
    grid.lambda_plus = 0.0;
    assert_predictions(&grid, x.view());
    grid.splits[0].push(1.0);
    let (plus, minus) = extract_two_tensor_predictions_unscaled(&grid, x.view());
    let (expected_plus, expected_minus) = reference_components(&grid, x.view());
    assert_eq!(plus, expected_plus);
    assert_eq!(minus, expected_minus);
}

#[test]
#[ignore = "wall-clock comparison of cached and per-sample factor evaluation"]
fn benchmark_prediction_factor_tables() {
    use std::hint::black_box;
    use std::time::Instant;
    let grid = grid_with_factors(
        (0..8)
            .map(|axis| {
                (0..32)
                    .map(|bin| 0.7 + 0.02 * (axis + bin) as f64)
                    .collect()
            })
            .collect(),
        (0..8)
            .map(|axis| {
                (0..32)
                    .map(|bin| 0.01 * (axis as f64 - bin as f64))
                    .collect()
            })
            .collect(),
    );
    let x = Array2::from_shape_fn((20_640, 8), |(row, axis)| {
        ((row * 13 + axis * 17) % 35) as f64 - 2.0
    });
    let expected = reference_components(&grid, x.view());
    let actual = extract_two_tensor_predictions_unscaled(&grid, x.view());
    assert_eq!(actual, expected);
    let reps = 40;
    let start = Instant::now();
    for _ in 0..reps {
        black_box(reference_components(black_box(&grid), black_box(x.view())));
    }
    let reference = start.elapsed();
    let start = Instant::now();
    for _ in 0..reps {
        black_box(extract_two_tensor_predictions_unscaled(
            black_box(&grid),
            black_box(x.view()),
        ));
    }
    let cached = start.elapsed();
    println!(
        "component extraction: reference={reference:?}, cached={cached:?}, speedup={:.3}",
        reference.as_secs_f64() / cached.as_secs_f64()
    );
    let start = Instant::now();
    for _ in 0..reps {
        black_box(Array1::from_iter(x.rows().into_iter().map(|row| {
            grid.predict_single_unscaled(black_box(row.as_slice().unwrap()))
        })));
    }
    let scalar = start.elapsed();
    let start = Instant::now();
    for _ in 0..reps {
        black_box(grid.predict(black_box(x.view())));
    }
    let batch = start.elapsed();
    println!(
        "grid prediction: scalar={scalar:?}, batch={batch:?}, speedup={:.3}",
        scalar.as_secs_f64() / batch.as_secs_f64()
    );
}
