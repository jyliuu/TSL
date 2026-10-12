//! Backbone and tilt feature importance of a fitted `TSL`, per stage and energy-weighted.

use numpy::ndarray::{Array1, Array2, ArrayView2};
use pyo3::{exceptions::PyValueError, prelude::*};
use tsl::forest::TSL;

/// Per-stage importance `(backbone, tilt)`, each of shape `(n_stages, n_features)`.
pub fn per_stage_feature_importance(
    tsl: &TSL,
    data_x: ArrayView2<f64>,
) -> PyResult<(Array2<f64>, Array2<f64>)> {
    let data_x_contiguous = data_x.as_standard_layout();
    per_stage_variances(tsl, data_x_contiguous.view())
}

/// Energy-weighted global importance `(backbone, tilt, stage_weights)`.
pub fn aggregated_feature_importance(
    tsl: &TSL,
    data_x: ArrayView2<f64>,
) -> PyResult<(Array1<f64>, Array1<f64>, Vec<f64>)> {
    let data_x_contiguous = data_x.as_standard_layout();

    // Step 1: Compute per-stage feature importance
    let (backbone_per_stage, tilt_per_stage) = per_stage_variances(tsl, data_x_contiguous.view())?;

    // Step 2: Compute stage weights
    let stage_weights = stage_weights(tsl, data_x_contiguous.view())?;

    // Step 3: Aggregate global importance
    let n_stages = backbone_per_stage.nrows();
    let n_features = backbone_per_stage.ncols();
    let mut global_backbone = Array1::<f64>::zeros(n_features);
    let mut global_tilt = Array1::<f64>::zeros(n_features);

    for stage_idx in 0..n_stages {
        let weight = stage_weights[stage_idx];
        for j in 0..n_features {
            global_backbone[j] += weight * backbone_per_stage[[stage_idx, j]];
            global_tilt[j] += weight * tilt_per_stage[[stage_idx, j]];
        }
    }

    Ok((global_backbone, global_tilt, stage_weights))
}

/// Computes per-stage feature importance: the variance over the rows of `data_x` of
/// `log b_j` and of `d_j`, for each stage and feature.
fn per_stage_variances(
    tsl: &TSL,
    data_x: ArrayView2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), PyErr> {
    let n_samples = data_x.nrows();
    let n_features = data_x.ncols();

    let stage_predictors = tsl.get_stage_predictors();
    let n_stages = stage_predictors.len();

    // Validate dimensions
    for tgf in stage_predictors.iter() {
        let grid = tgf.get_primary_grid_tensor();
        if grid.splits.len() != n_features {
            return Err(PyValueError::new_err(format!(
                "Feature dimension mismatch: data has {} features, but model has {}",
                n_features,
                grid.splits.len()
            )));
        }
    }

    // Initialize result arrays: (n_stages, n_features)
    let mut backbone_importance = Array2::<f64>::zeros((n_stages, n_features));
    let mut tilt_importance = Array2::<f64>::zeros((n_stages, n_features));

    for (stage_idx, tgf) in stage_predictors.iter().enumerate() {
        let grid = tgf.get_primary_grid_tensor();

        // For each feature j, evaluate b_j(x_j) and d_j(x_j) on all samples
        for j in 0..n_features {
            // Evaluate log(b_j(x_j)) and d_j(x_j) for all samples
            let mut log_b_values = Vec::with_capacity(n_samples);
            let mut d_values = Vec::with_capacity(n_samples);

            for i in 0..n_samples {
                let bin = grid
                    .bin_index(j, data_x[[i, j]])
                    .min(grid.backbone_values[j].len() - 1);
                log_b_values.push(grid.backbone_values[j][bin].ln());
                d_values.push(grid.tilt_values[j][bin]);
            }

            // Compute mean
            let log_b_mean = log_b_values.iter().sum::<f64>() / n_samples as f64;
            let d_mean = d_values.iter().sum::<f64>() / n_samples as f64;

            // Compute variance
            let backbone_var = log_b_values
                .iter()
                .map(|&val| (val - log_b_mean).powi(2))
                .sum::<f64>()
                / n_samples as f64;
            let tilt_var = d_values
                .iter()
                .map(|&val| (val - d_mean).powi(2))
                .sum::<f64>()
                / n_samples as f64;

            backbone_importance[[stage_idx, j]] = backbone_var;
            tilt_importance[[stage_idx, j]] = tilt_var;
        }
    }

    Ok((backbone_importance, tilt_importance))
}

/// Computes stage weights based on energy: each stage's share of the total
/// `||scaling_+ * f_+ - scaling_- * f_-||_n^2` over all stages.
fn stage_weights(tsl: &TSL, data_x: ArrayView2<f64>) -> Result<Vec<f64>, PyErr> {
    let stage_predictors = tsl.get_stage_predictors();
    let n_stages = stage_predictors.len();
    let n_samples = data_x.nrows();

    if n_stages == 0 {
        return Err(PyValueError::new_err("Model has no stages"));
    }

    // Energy of a stage: the mean square of its prediction scaling_+ * f_+ + scaling_- * (-f_-)
    let stage_energies: Vec<f64> = stage_predictors
        .iter()
        .map(|tgf| {
            let prediction = tgf.predict(data_x);
            prediction.iter().fold(0.0, |energy, p| energy + p * p) / n_samples as f64
        })
        .collect();

    // Normalize stage weights
    let total_energy: f64 = stage_energies.iter().sum();
    let stage_weights: Vec<f64> = if total_energy > 0.0 {
        stage_energies.iter().map(|&e| e / total_energy).collect()
    } else {
        // If total energy is zero, use uniform weights
        vec![1.0 / n_stages as f64; n_stages]
    };

    Ok(stage_weights)
}
