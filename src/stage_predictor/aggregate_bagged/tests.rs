use super::shape_selection::compute_component_shape_distance_precomputed;
use super::*;
use ndarray::Array2;

fn create_simple_grid(
    backbone: Vec<Vec<f64>>,
    tilt: Vec<Vec<f64>>,
    lambda_plus: f64,
    lambda_minus: f64,
    splits: Vec<Vec<f64>>,
    intervals: Vec<Vec<(f64, f64)>>,
) -> GridTensor {
    let observation_counts: Vec<Vec<usize>> =
        backbone.iter().map(|axis| vec![10; axis.len()]).collect();
    GridTensor::new_two_tensor(
        splits,
        observation_counts,
        intervals,
        backbone,
        tilt,
        lambda_plus,
        lambda_minus,
    )
}

#[test]
fn test_component_shape_distance_identical() {
    let intervals = vec![vec![(0.0, 1.0)], vec![(0.0, 1.0)]];
    let splits = vec![vec![], vec![]];
    let grid1 = create_simple_grid(
        vec![vec![1.0], vec![1.0]],
        vec![vec![0.0], vec![0.0]],
        1.0,
        0.5,
        splits.clone(),
        intervals.clone(),
    );
    let grid2 = create_simple_grid(
        vec![vec![1.0], vec![1.0]],
        vec![vec![0.0], vec![0.0]],
        1.0,
        0.5,
        splits,
        intervals.clone(),
    );
    let bin_weights = precompute_bin_weights(&grid1.observation_counts);
    let grid1_pre = precompute_log_components(&grid1);
    let grid2_pre = precompute_log_components(&grid2);
    let distance =
        compute_component_shape_distance_precomputed(&grid1_pre, &grid2_pre, &bin_weights);
    assert!(
        distance < 1e-10,
        "Identical grids should have distance ≈ 0, got {}",
        distance
    );
}

#[test]
fn test_select_median_single_bag() {
    let intervals = vec![vec![(0.0, 1.0)]];
    let splits = vec![vec![]];
    let grid = create_simple_grid(
        vec![vec![1.0]],
        vec![vec![0.0]],
        1.0,
        0.5,
        splits,
        intervals.clone(),
    );
    let grids = [grid];
    let observation_counts = vec![vec![10]];

    let bin_weights = precompute_bin_weights(&observation_counts);
    let precomputed_grids: Vec<PrecomputedGrid> =
        grids.iter().map(precompute_log_components).collect();
    let (median_index, _distance_matrix) =
        select_median_tensor_component_shape(&precomputed_grids, &bin_weights);
    assert_eq!(median_index, 0);
}

#[test]
fn test_trim_outliers() {
    let intervals = vec![vec![(0.0, 1.0)]];
    let splits = vec![vec![]];
    let grid1 = create_simple_grid(
        vec![vec![1.0]],
        vec![vec![0.0]],
        1.0,
        0.5,
        splits.clone(),
        intervals.clone(),
    );
    let grid2 = create_simple_grid(
        vec![vec![1.0]],
        vec![vec![0.0]],
        1.0,
        0.5,
        splits.clone(),
        intervals.clone(),
    );
    let grid3 = create_simple_grid(
        vec![vec![2.0]],
        vec![vec![0.5]],
        2.0,
        1.0,
        splits,
        intervals,
    );
    let grids = vec![grid1, grid2, grid3];
    let observation_counts = vec![vec![10]];

    let bin_weights = precompute_bin_weights(&observation_counts);
    let precomputed_grids: Vec<PrecomputedGrid> =
        grids.iter().map(precompute_log_components).collect();
    let (median_index, distance_matrix) =
        select_median_tensor_component_shape(&precomputed_grids, &bin_weights);
    let kept = trim_outliers(&grids, median_index, &distance_matrix, 0.9);

    assert!(kept.len() >= 2);
    assert!(kept.contains(&0) || kept.contains(&1));
}

#[test]
fn test_combine_lambdas_geometric_mean() {
    let lambda_plus = vec![1.0, 2.0, 4.0];
    let lambda_minus = vec![0.5, 1.0, 2.0];

    let (combined_plus, combined_minus) =
        combine_lambdas_geometric_mean(&lambda_plus, &lambda_minus, None);

    let product: f64 = 1.0 * 2.0 * 4.0;
    let expected_plus = product.powf(1.0 / 3.0);
    assert!((combined_plus - expected_plus).abs() < 1e-10);
    assert!((combined_minus - 1.0).abs() < 1e-10);
}

#[test]
fn test_aggregate_bagged_two_tensor_single_bag() {
    let intervals = vec![vec![(0.0, 1.0)]];
    let splits = vec![vec![]];
    let grid = create_simple_grid(
        vec![vec![1.0]],
        vec![vec![0.0]],
        1.0,
        0.5,
        splits,
        intervals,
    );
    let grids = vec![grid.clone()];
    let points = Array2::from_shape_vec(
        (10, 1),
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    )
    .unwrap();

    let aggregated = aggregate_bagged_two_tensor(&grids, points.view(), None, 0.9);

    assert_eq!(aggregated.backbone_values.len(), grid.backbone_values.len());
}
