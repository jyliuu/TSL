//! Bagged Two-Tensor TSL Aggregation
//!
//! The procedure includes:
//! 1. Common grid alignment per axis
//! 2. Gauge fixing / canonicalization per bag
//! 3. Component-shape distance computation
//! 4. Median tensor selection (medoid)
//! 5. Robust averaging of kept components
//! 6. Post-aggregation normalization

use crate::grid_tensor::identification::l2_identify;
use crate::grid_tensor::GridTensor;
use crate::logging::log_combination_choice;
use crate::stage_predictor::combine_grids::refine_grids_to_union_two_tensor;
use ndarray::{ArrayView1, ArrayView2};

mod canonicalize;
mod shape_selection;
#[cfg(test)]
mod tests;

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

fn geometric_mean_combine_a_factors(
    a_plus_candidates: &[Vec<Vec<f64>>],
    a_minus_candidates: &[Vec<Vec<f64>>],
    weights: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    if a_plus_candidates.is_empty() {
        panic!("Cannot combine empty candidate grids");
    }

    let num_axes = a_plus_candidates[0].len();
    let total_weight: f64 = weights.iter().sum();

    if total_weight <= 0.0 {
        panic!("Total weight must be positive");
    }

    let mut combined_a_plus: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
    let mut combined_a_minus: Vec<Vec<f64>> = Vec::with_capacity(num_axes);

    for axis in 0..num_axes {
        let n_bins = a_plus_candidates[0][axis].len();
        let mut a_plus_axis = Vec::with_capacity(n_bins);
        let mut a_minus_axis = Vec::with_capacity(n_bins);

        for bin in 0..n_bins {
            let mut log_sum_plus = 0.0;
            for (a_plus_cand, &weight) in a_plus_candidates.iter().zip(weights) {
                let a_val = a_plus_cand[axis][bin].max(EPSILON);
                let log_val = a_val.ln().max(LOG_EPSILON).min(LOG_MAX);
                log_sum_plus += weight * log_val;
            }
            let combined_a_plus_val = (log_sum_plus / total_weight).exp();

            let mut log_sum_minus = 0.0;
            for (a_minus_cand, &weight) in a_minus_candidates.iter().zip(weights) {
                let a_val = a_minus_cand[axis][bin].max(EPSILON);
                let log_val = a_val.ln().max(LOG_EPSILON).min(LOG_MAX);
                log_sum_minus += weight * log_val;
            }
            let combined_a_minus_val = (log_sum_minus / total_weight).exp();

            a_plus_axis.push(combined_a_plus_val);
            a_minus_axis.push(combined_a_minus_val);
        }

        combined_a_plus.push(a_plus_axis);
        combined_a_minus.push(a_minus_axis);
    }

    (combined_a_plus, combined_a_minus)
}

fn combine_lambdas_geometric_mean(
    lambda_plus_candidates: &[f64],
    lambda_minus_candidates: &[f64],
    weights: Option<&[f64]>,
) -> (f64, f64) {
    if lambda_plus_candidates.is_empty() {
        panic!("Cannot combine empty lambda list");
    }

    let n = lambda_plus_candidates.len();
    let uniform_weights: Vec<f64> = vec![1.0; n];
    let weights = weights.unwrap_or(&uniform_weights);

    if weights.len() != n {
        panic!("Weights length must match lambdas length");
    }

    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        panic!("Total weight must be positive");
    }

    let log_sum_plus: f64 = lambda_plus_candidates
        .iter()
        .zip(weights.iter())
        .map(|(&l, &w)| {
            let log_l = l.max(EPSILON).ln().max(LOG_EPSILON).min(LOG_MAX);
            log_l * w
        })
        .sum();
    let combined_lambda_plus = (log_sum_plus / total_weight).exp();

    let log_sum_minus: f64 = lambda_minus_candidates
        .iter()
        .zip(weights.iter())
        .map(|(&l, &w)| {
            let log_l = l.max(EPSILON).ln().max(LOG_EPSILON).min(LOG_MAX);
            log_l * w
        })
        .sum();
    let combined_lambda_minus = (log_sum_minus / total_weight).exp();

    (combined_lambda_plus, combined_lambda_minus)
}

pub fn aggregate_bagged_two_tensor(
    grids: &[GridTensor],
    points: ArrayView2<f64>,
    _weights: Option<ArrayView1<f64>>,
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

    // Step 1: Align all grids to union grid
    let mut aligned_grids = refine_grids_to_union_two_tensor(grids);

    // Step 2: Recompute observation counts for the union grid
    let num_axes = aligned_grids[0].intervals.len();
    let mut union_observation_counts: Vec<Vec<usize>> = Vec::with_capacity(num_axes);
    for axis in 0..num_axes {
        let n_bins = aligned_grids[0].intervals[axis].len();
        let mut counts: Vec<usize> = vec![0; n_bins];
        let splits = &aligned_grids[0].splits[axis];

        let mut vals: Vec<f64> = points.column(axis).iter().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut s_idx: usize = 0;
        let b_len = splits.len();
        for v in vals {
            while s_idx < b_len && v >= splits[s_idx] {
                s_idx += 1;
            }
            counts[s_idx] += 1;
        }

        union_observation_counts.push(counts);
    }

    // Step 3: Per-bag canonicalization
    for grid in &mut aligned_grids {
        canonicalize_bag_per_axis(grid, &union_observation_counts);
    }

    let (_kept_indices, kept_grids) = if trim_percentage >= 1.0 {
        log::info!(
            "trim_percentage is {:.2}, combining all {} grids without distance computation",
            trim_percentage,
            aligned_grids.len()
        );
        let all_indices: Vec<usize> = (0..aligned_grids.len()).collect();
        let candidate_indices: Vec<(usize, f64)> =
            all_indices.iter().map(|&idx| (idx, 0.0)).collect();
        log_combination_choice("BaggedTwoTensor", None, &candidate_indices);
        let kept: Vec<&GridTensor> = aligned_grids.iter().collect();
        (all_indices, kept)
    } else {
        let bin_weights = precompute_bin_weights(&union_observation_counts);
        let precomputed_grids: Vec<PrecomputedGrid> = aligned_grids
            .iter()
            .map(precompute_log_components)
            .collect();

        let (median_index, distance_matrix) =
            select_median_tensor_component_shape(&precomputed_grids, &bin_weights);

        let kept_indices = trim_outliers(
            &aligned_grids,
            median_index,
            &distance_matrix,
            trim_percentage,
        );

        let candidate_indices: Vec<(usize, f64)> = kept_indices
            .iter()
            .map(|&idx| (idx, distance_matrix[idx][median_index]))
            .collect();
        log_combination_choice("BaggedTwoTensor", Some(median_index), &candidate_indices);

        let kept_grids: Vec<&GridTensor> =
            kept_indices.iter().map(|&i| &aligned_grids[i]).collect();
        (kept_indices, kept_grids)
    };

    // Step 7: Geometric mean of a_± factors on kept set
    let mut a_plus_candidates: Vec<Vec<Vec<f64>>> = Vec::with_capacity(kept_grids.len());
    let mut a_minus_candidates: Vec<Vec<Vec<f64>>> = Vec::with_capacity(kept_grids.len());
    let mut lambda_plus_candidates: Vec<f64> = Vec::with_capacity(kept_grids.len());
    let mut lambda_minus_candidates: Vec<f64> = Vec::with_capacity(kept_grids.len());

    for grid in &kept_grids {
        let (a_plus, a_minus) = compute_a_factors_from_bd(&grid.backbone_values, &grid.tilt_values);
        a_plus_candidates.push(a_plus);
        a_minus_candidates.push(a_minus);
        lambda_plus_candidates.push(grid.lambda_plus);
        lambda_minus_candidates.push(grid.lambda_minus);
    }

    let kept_weights: Vec<f64> = vec![1.0; kept_grids.len()];
    let (combined_a_plus, combined_a_minus) =
        geometric_mean_combine_a_factors(&a_plus_candidates, &a_minus_candidates, &kept_weights);

    // Step 8: Reconstruct (b, d)
    let (combined_backbone, combined_tilt) =
        convert_a_factors_to_bd(&combined_a_plus, &combined_a_minus);

    // Step 9: Post-aggregation normalization
    let mut temp_backbone = combined_backbone.clone();
    let mut temp_tilt = combined_tilt.clone();
    let mut temp_lambda_plus = 1.0;
    let mut temp_lambda_minus = 1.0;

    l2_identify(
        &mut temp_backbone,
        &mut temp_tilt,
        &union_observation_counts,
        &mut temp_lambda_plus,
        &mut temp_lambda_minus,
    );

    // Step 10: Geometric mean of lambdas
    let (combined_lambda_plus, combined_lambda_minus) =
        combine_lambdas_geometric_mean(&lambda_plus_candidates, &lambda_minus_candidates, None);

    let mut combined = GridTensor::new_two_tensor(
        aligned_grids[0].splits.clone(),
        union_observation_counts,
        aligned_grids[0].intervals.clone(),
        temp_backbone,
        temp_tilt,
        combined_lambda_plus,
        combined_lambda_minus,
    );
    combined.coalesce_equal_bins();
    combined
}
