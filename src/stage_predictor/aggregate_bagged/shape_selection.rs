//! Component-shape distances between bags, medoid selection and outlier trimming.

use crate::grid_tensor::GridTensor;

#[cfg(feature = "use-rayon")]
use rayon::prelude::*;

use super::{EPSILON, EPSILON_N, LOG_EPSILON, LOG_MAX};

#[derive(Clone)]
pub(super) struct PrecomputedGrid {
    log_f_plus: Vec<Vec<f64>>,
    log_f_minus: Vec<Vec<f64>>,
}

pub(super) fn precompute_bin_weights(observation_counts: &[Vec<usize>]) -> Vec<Vec<f64>> {
    observation_counts
        .iter()
        .map(|axis_counts| {
            axis_counts
                .iter()
                .map(|&count| if count == 0 { EPSILON_N } else { count as f64 })
                .collect()
        })
        .collect()
}

pub(super) fn precompute_log_components(grid: &GridTensor) -> PrecomputedGrid {
    let mut log_f_plus: Vec<Vec<f64>> = Vec::with_capacity(grid.backbone_values.len());
    let mut log_f_minus: Vec<Vec<f64>> = Vec::with_capacity(grid.backbone_values.len());

    for axis in 0..grid.backbone_values.len() {
        let backbone = &grid.backbone_values[axis];
        let tilt = &grid.tilt_values[axis];
        let mut plus_axis = Vec::with_capacity(backbone.len());
        let mut minus_axis = Vec::with_capacity(backbone.len());

        for bin in 0..backbone.len() {
            let b = backbone[bin].max(EPSILON);
            let d = tilt[bin];
            let log_b = b.ln();

            let p = (log_b + d.min(50.0)).max(LOG_EPSILON).min(LOG_MAX);
            let m = (log_b + (-d).min(50.0)).max(LOG_EPSILON).min(LOG_MAX);

            plus_axis.push(p);
            minus_axis.push(m);
        }

        log_f_plus.push(plus_axis);
        log_f_minus.push(minus_axis);
    }

    PrecomputedGrid {
        log_f_plus,
        log_f_minus,
    }
}

pub(super) fn compute_component_shape_distance_precomputed(
    grid_a: &PrecomputedGrid,
    grid_b: &PrecomputedGrid,
    bin_weights: &[Vec<f64>],
) -> f64 {
    debug_assert_eq!(grid_a.log_f_plus.len(), grid_b.log_f_plus.len());
    debug_assert_eq!(grid_a.log_f_plus.len(), bin_weights.len());

    let mut total_distance_sq = 0.0;

    for axis in 0..grid_a.log_f_plus.len() {
        let p_a = &grid_a.log_f_plus[axis];
        let m_a = &grid_a.log_f_minus[axis];
        let p_b = &grid_b.log_f_plus[axis];
        let m_b = &grid_b.log_f_minus[axis];
        let weights = &bin_weights[axis];

        debug_assert_eq!(p_a.len(), p_b.len());
        debug_assert_eq!(p_a.len(), weights.len());

        let mut axis_distance_sq = 0.0;

        for bin in 0..p_a.len() {
            let w = weights[bin];
            let delta_p = p_a[bin] - p_b[bin];
            let delta_m = m_a[bin] - m_b[bin];
            axis_distance_sq += w * (delta_p * delta_p + delta_m * delta_m);
        }

        total_distance_sq += axis_distance_sq;
    }

    total_distance_sq.sqrt()
}

#[cfg(not(feature = "use-rayon"))]
pub(super) fn select_median_tensor_component_shape(
    grids: &[PrecomputedGrid],
    bin_weights: &[Vec<f64>],
) -> (usize, Vec<Vec<f64>>) {
    if grids.is_empty() {
        panic!("Cannot select median from empty grid list");
    }
    if grids.len() == 1 {
        return (0, vec![vec![0.0; 1]; 1]);
    }

    let n = grids.len();
    let mut min_total_distance = f64::INFINITY;
    let mut median_index = 0;

    let mut distance_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    log::info!(
        "Computing pairwise component-shape distances for {} bags",
        n
    );
    for i in 0..n {
        for j in (i + 1)..n {
            let dist =
                compute_component_shape_distance_precomputed(&grids[i], &grids[j], bin_weights);
            distance_matrix[i][j] = dist;
            distance_matrix[j][i] = dist;
        }
    }

    log::info!("Finding median bag (medoid) from distance matrix");
    for i in 0..n {
        let total_distance: f64 = distance_matrix[i].iter().sum();
        if total_distance < min_total_distance {
            min_total_distance = total_distance;
            median_index = i;
        }
    }

    log::info!(
        "Selected grid {} as median (medoid) with total distance {:.6}",
        median_index,
        min_total_distance
    );
    (median_index, distance_matrix)
}

#[cfg(feature = "use-rayon")]
pub(super) fn select_median_tensor_component_shape(
    grids: &[PrecomputedGrid],
    bin_weights: &[Vec<f64>],
) -> (usize, Vec<Vec<f64>>) {
    if grids.is_empty() {
        panic!("Cannot select median from empty grid list");
    }
    if grids.len() == 1 {
        return (0, vec![vec![0.0; 1]; 1]);
    }

    let n = grids.len();
    let mut distance_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    log::info!(
        "Computing pairwise component-shape distances for {} bags (parallel)",
        n
    );

    let upper_triangle: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            ((i + 1)..n).into_par_iter().map(move |j| {
                let dist =
                    compute_component_shape_distance_precomputed(&grids[i], &grids[j], bin_weights);
                (i, j, dist)
            })
        })
        .collect();

    for (i, j, dist) in upper_triangle {
        distance_matrix[i][j] = dist;
        distance_matrix[j][i] = dist;
    }

    log::info!("Finding median bag (medoid) from distance matrix");
    let (median_index, min_total_distance) = (0..n)
        .map(|i| {
            let total_distance: f64 = distance_matrix[i].iter().sum();
            (i, total_distance)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    log::info!(
        "Selected grid {} as median (medoid) with total distance {:.6}",
        median_index,
        min_total_distance
    );
    (median_index, distance_matrix)
}

pub(super) fn trim_outliers(
    median_index: usize,
    distance_matrix: &[Vec<f64>],
    trim_percentage: f64,
) -> Vec<usize> {
    let n = distance_matrix.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let keep_count = (trim_percentage * n as f64).ceil() as usize;
    let keep_count = keep_count.min(n);

    log::info!(
        "Using pre-computed distances to median bag {}",
        median_index
    );
    let mut distances: Vec<(usize, f64)> = distance_matrix
        .iter()
        .enumerate()
        .map(|(i, row)| (i, row[median_index]))
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let kept: Vec<usize> = distances[..keep_count]
        .iter()
        .map(|(idx, _)| *idx)
        .collect();

    log::info!(
        "Selected {} out of {} grids (trim_percentage={:.2}, keep_count={})",
        kept.len(),
        n,
        trim_percentage,
        keep_count
    );

    kept
}
