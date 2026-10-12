//! Bagged two-tensor aggregation.
//!
//! 1. Align every bag to the union grid and count the training rows per union bin.
//! 2. Fix each bag's gauge per axis.
//! 3. Unless every bag is kept, select the medoid bag by component-shape distance and
//!    keep the bags closest to it.
//! 4. Combine the kept bags' `a±` factors by geometric mean and convert them to `(b, d)`.
//! 5. Identify the combined factors.
//! 6. Combine the kept bags' `λ±` by geometric mean.

use crate::grid_tensor::identification::l2_identify;
use crate::grid_tensor::GridTensor;
use crate::logging::log_combination_choice;
use crate::stage_predictor::combine_grids::refine_grids_to_union_two_tensor;
use ndarray::ArrayView2;

mod canonicalize;
mod shape_selection;

use canonicalize::{canonicalize_bag_per_axis, compute_a_factors_from_bd, convert_a_factors_to_bd};
use shape_selection::{
    precompute_bin_weights, precompute_log_components, select_median_tensor_component_shape,
    trim_outliers, PrecomputedGrid,
};

/// Numerical stability constants
const EPSILON: f64 = 1e-10;
const LOG_EPSILON: f64 = -23.025850929940457; // ln(1e-10)
const LOG_MAX: f64 = 23.025850929940457; // ln(1e10)
const EPSILON_N: f64 = 1e-12; // For empty bin weights

/// `ln x` with `x` floored at `EPSILON` and the logarithm clamped to `[LOG_EPSILON, LOG_MAX]`.
fn clamped_ln(x: f64) -> f64 {
    x.max(EPSILON).ln().clamp(LOG_EPSILON, LOG_MAX)
}

/// Geometric mean of `values`, averaging their `clamped_ln` logarithms.
fn geometric_mean(values: impl ExactSizeIterator<Item = f64>) -> f64 {
    let count = values.len() as f64;
    let log_sum: f64 = values.map(clamped_ln).sum();
    (log_sum / count).exp()
}

/// Bin-wise geometric mean of the candidates' per-axis factors.
fn geometric_mean_factors(candidates: &[Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
    candidates[0]
        .iter()
        .enumerate()
        .map(|(axis, bins)| {
            (0..bins.len())
                .map(|bin| geometric_mean(candidates.iter().map(|factors| factors[axis][bin])))
                .collect()
        })
        .collect()
}

pub fn aggregate_bagged_two_tensor(
    grids: &[GridTensor],
    points: ArrayView2<f64>,
    trim_percentage: f64,
) -> GridTensor {
    if grids.is_empty() {
        panic!("Cannot aggregate empty grid list");
    }
    if grids.len() == 1 {
        let mut grid = grids[0].clone();
        grid.coalesce_equal_bins();
        return grid;
    }

    // Step 1: align every bag to the union grid and count the rows in each union bin.
    let mut aligned_grids = refine_grids_to_union_two_tensor(grids);
    let union_observation_counts: Vec<Vec<usize>> = (0..aligned_grids[0].intervals.len())
        .map(|axis| {
            let splits = &aligned_grids[0].splits[axis];
            let mut counts = vec![0; aligned_grids[0].intervals[axis].len()];
            // Union splits are sorted; a row on a split falls in the bin to its right.
            for &value in points.column(axis).iter() {
                counts[splits.partition_point(|&split| split <= value)] += 1;
            }
            counts
        })
        .collect();

    // Step 2: per-bag gauge fixing.
    for grid in &mut aligned_grids {
        canonicalize_bag_per_axis(grid, &union_observation_counts);
    }

    // Step 3: keep every bag, or the bags closest to the medoid.
    let kept_grids: Vec<&GridTensor> = if trim_percentage >= 1.0 {
        log::info!(
            "trim_percentage is {:.2}, combining all {} grids without distance computation",
            trim_percentage,
            aligned_grids.len()
        );
        let candidate_indices: Vec<(usize, f64)> =
            (0..aligned_grids.len()).map(|idx| (idx, 0.0)).collect();
        log_combination_choice("BaggedTwoTensor", None, &candidate_indices);
        aligned_grids.iter().collect()
    } else {
        let bin_weights = precompute_bin_weights(&union_observation_counts);
        let precomputed_grids: Vec<PrecomputedGrid> = aligned_grids
            .iter()
            .map(precompute_log_components)
            .collect();

        let (median_index, distance_matrix) =
            select_median_tensor_component_shape(&precomputed_grids, &bin_weights);

        let kept_indices = trim_outliers(median_index, &distance_matrix, trim_percentage);

        let candidate_indices: Vec<(usize, f64)> = kept_indices
            .iter()
            .map(|&idx| (idx, distance_matrix[idx][median_index]))
            .collect();
        log_combination_choice("BaggedTwoTensor", Some(median_index), &candidate_indices);

        kept_indices.iter().map(|&i| &aligned_grids[i]).collect()
    };
    if kept_grids.is_empty() {
        panic!("Cannot combine empty candidate grids");
    }

    // Step 4: geometric mean of the kept a± factors, converted back to (b, d).
    let (a_plus_candidates, a_minus_candidates): (Vec<_>, Vec<_>) = kept_grids
        .iter()
        .map(|grid| compute_a_factors_from_bd(&grid.backbone_values, &grid.tilt_values))
        .unzip();
    let (mut backbone, mut tilt) = convert_a_factors_to_bd(
        &geometric_mean_factors(&a_plus_candidates),
        &geometric_mean_factors(&a_minus_candidates),
    );

    // Step 5: identify the combined factors. Step 6 supplies λ±, so the scale that
    // identification moves into λ± is dropped.
    let (mut identified_plus, mut identified_minus) = (1.0, 1.0);
    l2_identify(
        &mut backbone,
        &mut tilt,
        &union_observation_counts,
        &mut identified_plus,
        &mut identified_minus,
    );

    // Step 6: geometric mean of the kept λ±.
    let lambda_plus = geometric_mean(kept_grids.iter().map(|grid| grid.lambda_plus));
    let lambda_minus = geometric_mean(kept_grids.iter().map(|grid| grid.lambda_minus));

    let mut combined = GridTensor::new_two_tensor(
        aligned_grids[0].splits.clone(),
        union_observation_counts,
        aligned_grids[0].intervals.clone(),
        backbone,
        tilt,
        lambda_plus,
        lambda_minus,
    );
    combined.coalesce_equal_bins();
    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// One-axis grid with the given splits, per-bin factors and branch scales.
    fn grid(splits: &[f64], backbone: &[f64], tilt: &[f64], lambda: (f64, f64)) -> GridTensor {
        let edges: Vec<f64> = std::iter::once(f64::NEG_INFINITY)
            .chain(splits.iter().copied())
            .chain([f64::INFINITY])
            .collect();
        GridTensor::new_two_tensor(
            vec![splits.to_vec()],
            vec![vec![10; backbone.len()]],
            vec![edges.windows(2).map(|pair| (pair[0], pair[1])).collect()],
            vec![backbone.to_vec()],
            vec![tilt.to_vec()],
            lambda.0,
            lambda.1,
        )
    }

    fn points() -> Array2<f64> {
        Array2::from_shape_fn((20, 1), |(row, _)| (row as f64 + 0.5) / 20.0)
    }

    fn aggregate(bags: &[GridTensor], trim_percentage: f64) -> Array1<f64> {
        let points = points();
        aggregate_bagged_two_tensor(bags, points.view(), trim_percentage).predict(points.view())
    }

    #[test]
    fn aggregating_one_bag_keeps_its_predictions() {
        let bag = grid(
            &[0.25, 0.5, 0.75],
            &[1.0, 1.0, 2.0, 0.5],
            &[0.3, 0.3, -0.2, 0.1],
            (1.5, 0.5),
        );
        let expected = bag.predict(points().view());
        assert_eq!(aggregate(&[bag], 0.9), expected);
    }

    #[test]
    fn trimming_drops_the_bag_far_from_the_medoid() {
        let near = grid(&[0.5], &[1.0, 1.2], &[0.1, -0.1], (1.0, 0.5));
        let far = grid(&[0.5], &[1.0, 6.0], &[0.8, -0.8], (2.0, 1.5));
        let near_only = aggregate(&[near.clone(), near.clone()], 1.0);
        // The far bag comes first, so keeping a prefix of the input would keep it.
        let bags = [far, near.clone(), near];
        assert_eq!(aggregate(&bags, 0.6), near_only);
        assert_ne!(aggregate(&bags, 1.0), near_only);
    }

    #[test]
    fn geometric_mean_averages_logarithms() {
        assert!((geometric_mean([1.0, 2.0, 4.0].into_iter()) - 2.0).abs() < 1e-10);
        assert!((geometric_mean([0.5, 1.0, 2.0].into_iter()) - 1.0).abs() < 1e-10);
    }
}
