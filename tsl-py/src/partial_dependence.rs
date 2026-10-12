//! Partial dependence and ICE curves of a fitted `TSL`, per stage and per branch.

use std::collections::HashSet;

use numpy::{
    ndarray::{Array2, Array3, ArrayView1, ArrayView2},
    ToPyArray,
};
use pyo3::{
    prelude::*,
    types::{PyList, PyTuple},
};
use tsl::forest::TSL;

/// Per-epoch `(C_plus, C_minus)` constants and the `(n_observations, 2 * n_epochs)` values
/// with columns `[f+_epoch0, f-_epoch0, f+_epoch1, f-_epoch1, ...]`.
pub type PartialDependence = (Vec<(f64, f64)>, Array2<f64>);

/// Converts a partial dependence result to the Python tuple `(constants_per_epoch, pd_values)`,
/// with the constants as a list of `(C_plus, C_minus)` tuples.
pub fn to_py_tuple<'py>(
    py: Python<'py>,
    (constants_per_epoch, pd_values): &PartialDependence,
) -> PyResult<Bound<'py, PyTuple>> {
    let constants_list = PyList::empty(py);
    for (c_plus, c_minus) in constants_per_epoch {
        let constant_tuple = PyTuple::new(py, [*c_plus, *c_minus])?;
        constants_list.append(constant_tuple)?;
    }

    let pd_py = pd_values.to_pyarray(py);
    PyTuple::new(py, [constants_list.as_any(), pd_py.as_any()])
}

/// Partial dependence on the features `fixed_indices`, evaluated at the rows of `fixed_values`
/// and averaged over the rows of `data_x` for the remaining features.
pub fn partial_dependence(
    tsl: &TSL,
    fixed_indices: &[usize],
    fixed_values: ArrayView2<f64>,
    data_x: ArrayView2<f64>,
) -> PyResult<PartialDependence> {
    let fixed_values_contiguous = fixed_values.as_standard_layout();
    let data_x_contiguous = data_x.as_standard_layout();

    let n_observations = fixed_values_contiguous.nrows();
    let n_fixed = fixed_indices.len();
    let n_epochs = tsl.get_stage_predictors().len();

    // Validate dimensions
    if fixed_values_contiguous.ncols() != n_fixed {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "fixed_values must have {} columns (one per fixed_indices), got {}",
            n_fixed,
            fixed_values_contiguous.ncols()
        )));
    }

    // Convert fixed_indices to HashSet for fast lookup
    let fixed_set: HashSet<usize> = fixed_indices.iter().copied().collect();

    // Validate all indices are unique
    if fixed_set.len() != n_fixed {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "fixed_indices must contain unique values",
        ));
    }

    // Initialize result array: (n_observations, 2 * n_epochs)
    // Columns alternate: [f+_epoch0, f-_epoch0, f+_epoch1, f-_epoch1, ...]
    let mut results = Array2::<f64>::zeros((n_observations, 2 * n_epochs));
    let mut constants_per_epoch: Vec<(f64, f64)> = Vec::with_capacity(n_epochs);

    for (epoch_idx, tgf) in tsl.get_stage_predictors().iter().enumerate() {
        let grid = &tgf.primary_grid_tensor;
        let n_features = grid.splits.len();

        // Validate fixed indices are in bounds
        for &idx in fixed_indices {
            if idx >= n_features {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Feature index {} out of bounds (n_features={})",
                    idx, n_features
                )));
            }
        }

        let n_samples = data_x_contiguous.nrows();

        // Get OLS scaling coefficients and absorb into lambda
        // Prediction formula: scaling_plus * f_plus + scaling_minus * (-f_minus)
        // So we use: effective_lambda_plus = scaling_plus * lambda_plus
        //            effective_lambda_minus = -scaling_minus * lambda_minus
        let scaling_plus = tgf.scaling_plus.unwrap_or(1.0);
        let scaling_minus = tgf.scaling_minus.unwrap_or(0.0);
        let effective_lambda_plus = scaling_plus * grid.lambda_plus;
        let effective_lambda_minus = -scaling_minus * grid.lambda_minus;

        // Step 1: Compute E[∏_{j ∉ S} f_j(X_j)] once per epoch
        // Since f = λ_± * ∏_j a_±,j, we compute the expectation of the product over non-fixed features
        // Note: lambda values already include scaling
        let mut expected_marginalized_plus = 0.0;
        let mut expected_marginalized_minus = 0.0;

        for train_idx in 0..n_samples {
            let mut prod_plus = effective_lambda_plus;
            let mut prod_minus = effective_lambda_minus;

            // Only iterate over non-fixed features
            for j in 0..n_features {
                if fixed_set.contains(&j) {
                    continue; // Skip fixed features
                }

                let x_j = data_x_contiguous[[train_idx, j]];
                let col_idx = grid.splits[j].partition_point(|&split| split <= x_j);
                let col_idx = col_idx.min(grid.backbone_values[j].len() - 1);

                let b = grid.backbone_values[j][col_idx];
                let d = grid.tilt_values[j][col_idx];

                prod_plus *= b * d.exp();
                prod_minus *= b * (-d).exp();
            }

            expected_marginalized_plus += prod_plus;
            expected_marginalized_minus += prod_minus;
        }

        expected_marginalized_plus /= n_samples as f64;
        expected_marginalized_minus /= n_samples as f64;

        // Store constants for this epoch (expectation over marginalized features)
        constants_per_epoch.push((expected_marginalized_plus, expected_marginalized_minus));

        // Step 2: For each observation, compute ∏_{j ∈ S} f_j(x_j) and multiply
        for obs_idx in 0..n_observations {
            let mut prod_fixed_plus = 1.0;
            let mut prod_fixed_minus = 1.0;

            // Only iterate over fixed features
            for (col_idx, &feature_idx) in fixed_indices.iter().enumerate() {
                let x_j = fixed_values_contiguous[[obs_idx, col_idx]];

                let interval_idx = grid.splits[feature_idx].partition_point(|&split| split <= x_j);
                let interval_idx = interval_idx.min(grid.backbone_values[feature_idx].len() - 1);

                let b = grid.backbone_values[feature_idx][interval_idx];
                let d = grid.tilt_values[feature_idx][interval_idx];

                prod_fixed_plus *= b * d.exp();
                prod_fixed_minus *= b * (-d).exp();
            }

            // Combine: ∏_{j ∈ S} f_j(x_j) · E[∏_{j ∉ S} f_j(X_j)]
            // Scaling is already absorbed into effective_lambda values
            let f_plus = prod_fixed_plus * expected_marginalized_plus;
            let f_minus = prod_fixed_minus * expected_marginalized_minus;

            // Store f+ and f- separately (scaling already absorbed into lambda)
            // Columns alternate: [f+_epoch0, f-_epoch0, f+_epoch1, f-_epoch1, ...]
            // Final prediction per epoch = f+ + f- (add the two columns)
            results[[obs_idx, 2 * epoch_idx]] = f_plus;
            results[[obs_idx, 2 * epoch_idx + 1]] = f_minus;
        }
    }

    Ok((constants_per_epoch, results))
}

/// First-order partial dependence of every feature `j`, evaluated at column `j` of `values_x`
/// and averaged over the rows of `data_x` for the other features; one result per feature.
pub fn first_order_partial_dependence(
    tsl: &TSL,
    values_x: ArrayView2<f64>,
    data_x: ArrayView2<f64>,
) -> PyResult<Vec<PartialDependence>> {
    let values_x_contiguous = values_x.as_standard_layout();
    let data_x_contiguous = data_x.as_standard_layout();

    let n_observations = values_x_contiguous.nrows();
    let n_features = values_x_contiguous.ncols();
    let n_samples = data_x_contiguous.nrows();
    let n_epochs = tsl.get_stage_predictors().len();

    if n_samples == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "data_x must contain at least one sample",
        ));
    }
    if data_x_contiguous.ncols() != n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "values_x and data_x must have the same number of columns, got {} and {}",
            n_features,
            data_x_contiguous.ncols()
        )));
    }

    let mut constants_per_dim: Vec<Vec<(f64, f64)>> =
        vec![Vec::with_capacity(n_epochs); n_features];
    let mut pd_values_per_dim: Vec<Array2<f64>> = (0..n_features)
        .map(|_| Array2::<f64>::zeros((n_observations, 2 * n_epochs)))
        .collect();

    for (epoch_idx, tgf) in tsl.get_stage_predictors().iter().enumerate() {
        let grid = &tgf.primary_grid_tensor;
        let grid_features = grid.splits.len();

        if grid_features != n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Feature count mismatch: values_x has {} columns but grid has {}",
                n_features, grid_features
            )));
        }

        // Get OLS scaling coefficients and absorb into lambda
        // Prediction formula: scaling_plus * f_plus + scaling_minus * (-f_minus)
        // So we use: effective_lambda_plus = scaling_plus * lambda_plus
        //            effective_lambda_minus = -scaling_minus * lambda_minus
        let scaling_plus = tgf.scaling_plus.unwrap_or(1.0);
        let scaling_minus = tgf.scaling_minus.unwrap_or(0.0);
        let effective_lambda_plus = scaling_plus * grid.lambda_plus;
        let effective_lambda_minus = -scaling_minus * grid.lambda_minus;

        let mut sum_plus = vec![0.0; n_features];
        let mut sum_minus = vec![0.0; n_features];

        let mut factors_plus = vec![0.0; n_features];
        let mut factors_minus = vec![0.0; n_features];
        let mut prefix_plus = vec![1.0; n_features + 1];
        let mut prefix_minus = vec![1.0; n_features + 1];
        let mut suffix_plus = vec![1.0; n_features + 1];
        let mut suffix_minus = vec![1.0; n_features + 1];

        for sample_idx in 0..n_samples {
            for j in 0..n_features {
                let x_j = data_x_contiguous[[sample_idx, j]];
                let interval_idx = grid.splits[j].partition_point(|&split| split <= x_j);
                let interval_idx =
                    interval_idx.min(grid.backbone_values[j].len().saturating_sub(1));

                let b = grid.backbone_values[j][interval_idx];
                let d = grid.tilt_values[j][interval_idx];

                factors_plus[j] = b * d.exp();
                factors_minus[j] = b * (-d).exp();
            }

            prefix_plus[0] = 1.0;
            prefix_minus[0] = 1.0;
            for j in 0..n_features {
                prefix_plus[j + 1] = prefix_plus[j] * factors_plus[j];
                prefix_minus[j + 1] = prefix_minus[j] * factors_minus[j];
            }

            suffix_plus[n_features] = 1.0;
            suffix_minus[n_features] = 1.0;
            for j in (0..n_features).rev() {
                suffix_plus[j] = suffix_plus[j + 1] * factors_plus[j];
                suffix_minus[j] = suffix_minus[j + 1] * factors_minus[j];
            }

            for j in 0..n_features {
                // Use effective lambdas (scaling already absorbed)
                sum_plus[j] += effective_lambda_plus * prefix_plus[j] * suffix_plus[j + 1];
                sum_minus[j] += effective_lambda_minus * prefix_minus[j] * suffix_minus[j + 1];
            }
        }

        let mut constants_plus = vec![0.0; n_features];
        let mut constants_minus = vec![0.0; n_features];
        let denom = n_samples as f64;
        for j in 0..n_features {
            let c_plus = sum_plus[j] / denom;
            let c_minus = sum_minus[j] / denom;
            constants_plus[j] = c_plus;
            constants_minus[j] = c_minus;
            constants_per_dim[j].push((c_plus, c_minus));
        }

        for obs_idx in 0..n_observations {
            for j in 0..n_features {
                let x_j = values_x_contiguous[[obs_idx, j]];
                let interval_idx = grid.splits[j].partition_point(|&split| split <= x_j);
                let interval_idx =
                    interval_idx.min(grid.backbone_values[j].len().saturating_sub(1));

                let b = grid.backbone_values[j][interval_idx];
                let d = grid.tilt_values[j][interval_idx];

                let m_plus = b * d.exp();
                let m_minus = b * (-d).exp();

                // Compute PD values (scaling already absorbed into constants via effective_lambda)
                let col_plus = 2 * epoch_idx;
                let col_minus = 2 * epoch_idx + 1;
                pd_values_per_dim[j][[obs_idx, col_plus]] = constants_plus[j] * m_plus;
                pd_values_per_dim[j][[obs_idx, col_minus]] = constants_minus[j] * m_minus;
            }
        }
    }

    Ok(constants_per_dim
        .into_iter()
        .zip(pd_values_per_dim)
        .collect())
}

/// ICE curves of `feature_index` over `x_range` for each row of `observations`, shaped
/// `(n_obs, n_range_values, 2 * n_epochs)` with scaled `f+` and `-f-` per epoch.
pub fn ice_curves(
    tsl: &TSL,
    observations: ArrayView2<f64>,
    feature_index: usize,
    x_range: ArrayView1<f64>,
    data_x: ArrayView2<f64>,
) -> PyResult<Array3<f64>> {
    let observations_contiguous = observations.as_standard_layout();
    let x_range_contiguous = x_range.as_standard_layout();
    let data_x_contiguous = data_x.as_standard_layout();

    let n_obs = observations_contiguous.nrows();
    let n_features = observations_contiguous.ncols();
    let n_range_values = x_range_contiguous.len();
    let n_epochs = tsl.get_stage_predictors().len();

    // Validate dimensions
    if feature_index >= n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "feature_index {} out of bounds (n_features={})",
            feature_index, n_features
        )));
    }

    if data_x_contiguous.ncols() != n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "observations and data_x must have the same number of columns, got {} and {}",
            n_features,
            data_x_contiguous.ncols()
        )));
    }

    if n_range_values == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_range must contain at least one value",
        ));
    }

    // Initialize result array: (n_obs, n_range_values, 2 * n_epochs)
    let mut results = Array3::<f64>::zeros((n_obs, n_range_values, 2 * n_epochs));

    for (epoch_idx, tgf) in tsl.get_stage_predictors().iter().enumerate() {
        let grid = &tgf.primary_grid_tensor;
        let grid_features = grid.splits.len();

        if grid_features != n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Feature count mismatch: observations has {} columns but grid has {}",
                n_features, grid_features
            )));
        }

        // Get OLS scaling coefficients
        let scaling_plus = tgf.scaling_plus.unwrap_or(1.0);
        let scaling_minus = tgf.scaling_minus.unwrap_or(0.0);

        // For each observation
        for obs_idx in 0..n_obs {
            // Pre-compute constant factors for all features except the varying one
            let mut const_plus = grid.lambda_plus;
            let mut const_minus = grid.lambda_minus;

            for j in 0..n_features {
                if j == feature_index {
                    continue; // Skip the varying feature
                }

                let x_j = observations_contiguous[[obs_idx, j]];
                let interval_idx = grid.splits[j].partition_point(|&split| split <= x_j);
                let interval_idx =
                    interval_idx.min(grid.backbone_values[j].len().saturating_sub(1));

                let b = grid.backbone_values[j][interval_idx];
                let d = grid.tilt_values[j][interval_idx];

                const_plus *= b * d.exp();
                const_minus *= b * (-d).exp();
            }

            // For each value in x_range
            for range_idx in 0..n_range_values {
                let v = x_range_contiguous[range_idx];

                // Find interval for the varying feature at value v
                let interval_idx = grid.splits[feature_index].partition_point(|&split| split <= v);
                let interval_idx =
                    interval_idx.min(grid.backbone_values[feature_index].len().saturating_sub(1));

                let b = grid.backbone_values[feature_index][interval_idx];
                let d = grid.tilt_values[feature_index][interval_idx];

                // Compute f+ and f- with the varying feature
                let f_plus_unscaled = const_plus * (b * d.exp());
                let f_minus_unscaled = const_minus * (b * (-d).exp());

                // Apply scaling (matching prediction formula: scaling_plus * f_+ + scaling_minus * (-f_-))
                // For ICE curves, return scaled f+ and scaled (-f-) separately
                let f_plus_scaled = scaling_plus * f_plus_unscaled;
                let f_minus_scaled = scaling_minus * (-f_minus_unscaled);

                // Store results directly in 3D array
                results[[obs_idx, range_idx, 2 * epoch_idx]] = f_plus_scaled;
                results[[obs_idx, range_idx, 2 * epoch_idx + 1]] = f_minus_scaled;
            }
        }
    }

    Ok(results)
}
