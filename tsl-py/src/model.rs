//! The `TSL` Python class: the boosted model's fit, prediction, diagnostics and persistence.

use numpy::{
    ndarray::Array1, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::{
    prelude::*,
    types::{PyList, PyTuple, PyType},
};
use tsl::{
    forest::{fit_boosted, TSL},
    stage_predictor::StagePredictor,
};

use crate::{
    feature_importance, fit_result::FitResultPy, params::BoostedFitOptions, partial_dependence,
    persistence, stage_predictor::StagePredictorPy,
};

#[derive(Debug)]
#[pyclass(name = "TSL")]
pub struct TSLPy(TSL);

impl From<TSL> for TSLPy {
    fn from(tsl: TSL) -> Self {
        TSLPy(tsl)
    }
}

#[pymethods]
impl TSLPy {
    #[getter]
    pub fn get_stage_predictors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let stage_predictors_py: Vec<StagePredictorPy> = self
            .0
            .get_stage_predictors()
            .iter()
            .map(|tgf: &StagePredictor| StagePredictorPy::from(tgf.clone()))
            .collect();

        PyList::new(py, stage_predictors_py)
    }

    #[pyo3(name = "predict")]
    pub fn _predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = x.as_array();
        let y_hat = self.0.predict(x);
        Ok(y_hat.to_pyarray(py))
    }

    /// Compute partial dependence function: E_{X_S^c}[f(X_S, X_S^c)]
    ///
    /// For each tree grid family (epoch), computes the partial dependence where
    /// some features are fixed and others are marginalized over their empirical joint distribution.
    ///
    /// Computed as: (1/n) ∑_{i=1}^n f(x_S, x_{i,S^c}) where we average over training samples,
    /// preserving the joint distribution of marginalized features (not assuming independence).
    ///
    /// Parameters:
    /// - fixed_indices: Vector of feature indices that are fixed (in order)
    /// - fixed_values: Array2 of shape (n_observations, len(fixed_indices))
    ///                 Each row is one observation, columns correspond to fixed_indices order
    /// - data_x: Training data for estimating marginal distributions (n_samples, n_features)
    ///
    /// Returns:
    /// - Tuple of (constants_per_epoch, pd_values):
    ///   - constants_per_epoch: list of (C_plus, C_minus) per epoch
    ///     Constants are E[∏_{j ∉ S} f_j(X_j)] (expectation over marginalized features)
    ///     Constants include OLS scaling (computed with effective_lambda = scaling * lambda)
    ///   - pd_values: Array2 of shape (n_observations, 2 * n_epochs) with columns
    ///     [f+_epoch0, f-_epoch0, f+_epoch1, f-_epoch1, ...]
    ///     Values have scaling absorbed into lambda, so final prediction per epoch = f+ + f- (add columns)
    #[pyo3(name = "compute_partial_dependence_function")]
    pub fn _compute_partial_dependence_function<'py>(
        &self,
        py: Python<'py>,
        fixed_indices: Vec<usize>,
        fixed_values: PyReadonlyArray2<'py, f64>,
        data_x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let pd = partial_dependence::partial_dependence(
            &self.0,
            &fixed_indices,
            fixed_values.as_array(),
            data_x.as_array(),
        )?;
        partial_dependence::to_py_tuple(py, &pd)
    }

    /// Compute first-order partial dependence functions for every feature.
    ///
    /// For each epoch and feature j, computes a constant C_{+,j} and C_{-,j}:
    ///   C_{+,j} = E[lambda_+ * ∏_{k != j} a_{+,k}(X_k)]
    ///   C_{-,j} = E[lambda_- * ∏_{k != j} a_{-,k}(X_k)]
    ///
    /// Then PD_{+,j}(x_j) = C_{+,j} * a_{+,j}(x_j) and
    ///      PD_{-,j}(x_j) = C_{-,j} * a_{-,j}(x_j).
    ///
    /// Returns a list of length p where each entry is:
    ///   (constants_per_epoch, pd_values)
    ///
    /// - constants_per_epoch: Vec of (C_plus, C_minus) for each epoch
    ///   Constants include OLS scaling (computed with effective_lambda = scaling * lambda)
    /// - pd_values: Array2 of shape (n_observations, 2 * n_epochs) with columns
    ///   [f+_epoch0, f-_epoch0, f+_epoch1, f-_epoch1, ...] for feature j.
    ///   Values have scaling absorbed into constants, so final prediction per epoch = f+ + f- (add columns)
    #[pyo3(name = "compute_first_order_partial_dependence_functions")]
    pub fn _compute_first_order_partial_dependence_functions<'py>(
        &self,
        py: Python<'py>,
        values_x: PyReadonlyArray2<'py, f64>,
        data_x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let pd_per_feature = partial_dependence::first_order_partial_dependence(
            &self.0,
            values_x.as_array(),
            data_x.as_array(),
        )?;

        let mut output_items: Vec<Bound<'_, PyAny>> = Vec::with_capacity(pd_per_feature.len());
        for pd in &pd_per_feature {
            let tuple = partial_dependence::to_py_tuple(py, pd)?;
            output_items.push(tuple.as_any().clone());
        }

        PyList::new(py, output_items)
    }

    /// Compute Individual Conditional Expectation (ICE) curves for a single feature.
    ///
    /// For each observation, varies the specified feature over the provided range while
    /// keeping all other features fixed at that observation's values. Computes f+ and f-
    /// separately for each epoch.
    ///
    /// Parameters:
    /// - observations: Array2 of shape (n_obs, n_features) - observations to compute ICE for
    /// - feature_index: Index of the feature to vary
    /// - x_range: Array1 of values to evaluate for the varying feature
    /// - data_x: Training data (n_samples, n_features) - used for validation only
    ///
    /// Returns:
    /// - Array3 of shape (n_obs, n_range_values, 2 * n_epochs)
    ///   Last dimension: [f+_epoch0, f-_epoch0, f+_epoch1, f-_epoch1, ...]
    ///   Values have scaling applied: scaling_plus * f_+ and scaling_minus * (-f_-)
    #[pyo3(name = "compute_ice_curves")]
    pub fn _compute_ice_curves<'py>(
        &self,
        py: Python<'py>,
        observations: PyReadonlyArray2<'py, f64>,
        feature_index: usize,
        x_range: PyReadonlyArray1<'py, f64>,
        data_x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let results = partial_dependence::ice_curves(
            &self.0,
            observations.as_array(),
            feature_index,
            x_range.as_array(),
            data_x.as_array(),
        )?;

        // Return as 3D array: (n_obs, n_range_values, 2 * n_epochs)
        Ok(results.to_pyarray(py))
    }

    /// Compute per-stage feature importance metrics.
    ///
    /// For each stage ℓ and feature j, computes:
    /// - Backbone variance: Var_n[log b_j^(ℓ)(X_j)]
    /// - Tilt variance: Var_n[d_j^(ℓ)(X_j)]
    ///
    /// Returns:
    /// - Array2 of shape (n_stages, n_features) for backbone importance
    /// - Array2 of shape (n_stages, n_features) for tilt importance
    #[pyo3(name = "compute_per_stage_feature_importance")]
    pub fn _compute_per_stage_feature_importance<'py>(
        &self,
        py: Python<'py>,
        data_x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (backbone_importance, tilt_importance) =
            feature_importance::per_stage_feature_importance(&self.0, data_x.as_array())?;

        Ok((
            backbone_importance.to_pyarray(py),
            tilt_importance.to_pyarray(py),
        ))
    }

    /// Compute aggregated global feature importance.
    ///
    /// Computes stage weights based on energy:
    /// ω_ℓ = ||λ_{+,ℓ} m̂_{+}^{(ℓ)} - λ_{-,ℓ} m̂_{-}^{(ℓ)}||_n^2 / sum_k ||λ_{+,k} m̂_{+}^{(k)} - λ_{-,k} m̂_{-}^{(k)}||_n^2
    ///
    /// Then aggregates per-stage importance:
    /// I_j^b = sum_ℓ ω_ℓ I_j^{b,(ℓ)}
    /// I_j^d = sum_ℓ ω_ℓ I_j^{d,(ℓ)}
    ///
    /// Returns:
    /// - Array1 of shape (n_features,) for global backbone importance
    /// - Array1 of shape (n_features,) for global tilt importance
    /// - Array1 of shape (n_stages,) for stage weights
    #[pyo3(name = "compute_aggregated_feature_importance")]
    pub fn _compute_aggregated_feature_importance<'py>(
        &self,
        py: Python<'py>,
        data_x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let (global_backbone, global_tilt, stage_weights) =
            feature_importance::aggregated_feature_importance(&self.0, data_x.as_array())?;

        Ok((
            global_backbone.to_pyarray(py),
            global_tilt.to_pyarray(py),
            Array1::from_vec(stage_weights).to_pyarray(py),
        ))
    }

    /// Compute combined feature importance score.
    ///
    /// Combines backbone and tilt importance:
    /// I_j = I_j^b + γ * I_j^d
    ///
    /// Parameters:
    /// - data_x: Training data (n_samples, n_features)
    /// - gamma: Weight for tilt importance (default: 1.0)
    ///
    /// Returns:
    /// - Array1 of shape (n_features,) for combined importance
    /// - Array1 of shape (n_features,) for backbone importance
    /// - Array1 of shape (n_features,) for tilt importance
    #[pyo3(name = "compute_combined_feature_importance", signature = (data_x, gamma=None))]
    pub fn _compute_combined_feature_importance<'py>(
        &self,
        py: Python<'py>,
        data_x: PyReadonlyArray2<'py, f64>,
        gamma: Option<f64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let gamma = gamma.unwrap_or(1.0);

        let (global_backbone, global_tilt, _stage_weights) =
            feature_importance::aggregated_feature_importance(&self.0, data_x.as_array())?;

        let combined = &global_backbone + &(gamma * &global_tilt);

        Ok((
            combined.to_pyarray(py),
            global_backbone.to_pyarray(py),
            global_tilt.to_pyarray(py),
        ))
    }

    /// Save the TSL model to a binary file (preserves exact floating point values).
    ///
    /// **Note**: Binary format is same-version-only. Models saved with one version of tsl-py
    /// may not load with a different version due to schema changes. For portability across
    /// versions, consider exporting model parameters or predictions instead.
    #[pyo3(name = "save")]
    pub fn save(&self, path: &str) -> PyResult<()> {
        persistence::save(&self.0, path)
    }

    /// Load an TSL model from a binary file.
    ///
    /// **Note**: Binary format is same-version-only. Models saved with one version of tsl-py
    /// may not load with a different version due to schema changes.
    #[classmethod]
    #[pyo3(name = "load")]
    pub fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<TSLPy> {
        Ok(TSLPy(persistence::load(path)?))
    }

    #[classmethod]
    #[pyo3(name = "fit")]
    #[pyo3(signature = (
        x, y, epochs,
        decay=1.0,
        n_trees=10,
        n_iter=10,
        split_try=10,
        colsample_bytree=1.0,
        alpha=0.0,
        complexity_penalty=0.0,
        min_split_loss=0.0,
        min_interval_samples=1,
        refinement_strategy="l2",
        update_clamp=f64::INFINITY,
        tilt_tau=0.01,
        tilt_rho=0.0,
        split_strategy="random",
        top_k=10,
        must_fill_all_k=true,
        similarity_threshold=0.0,
        consensus=0.0,
        seed=42,
        verbosity=1,
        visualdb=None,
        *,
        max_bins=None,
        consensus_interval=1,
        nonnegative_scaling=false,
        consensus_anchors=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn _fit<'py>(
        _cls: &Bound<'_, PyType>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        epochs: usize,
        decay: f64,
        n_trees: usize,
        n_iter: usize,
        split_try: usize,
        colsample_bytree: f64,
        alpha: f64,
        complexity_penalty: f64,
        min_split_loss: f64,
        min_interval_samples: usize,
        refinement_strategy: &str,
        update_clamp: f64,
        tilt_tau: f64,
        tilt_rho: f64,
        split_strategy: &str,
        top_k: usize,
        must_fill_all_k: bool,
        similarity_threshold: f64,
        consensus: f64,
        seed: u64,
        verbosity: u8,
        visualdb: Option<String>,
        max_bins: Option<u16>,
        consensus_interval: usize,
        nonnegative_scaling: bool,
        consensus_anchors: Option<usize>,
    ) -> PyResult<(TSLPy, FitResultPy)> {
        let options = BoostedFitOptions {
            epochs,
            decay,
            n_trees,
            n_iter,
            split_try,
            colsample_bytree,
            alpha,
            complexity_penalty,
            min_split_loss,
            min_interval_samples,
            refinement_strategy,
            update_clamp,
            tilt_tau,
            tilt_rho,
            split_strategy,
            top_k,
            must_fill_all_k,
            similarity_threshold,
            consensus,
            seed,
            verbosity,
            visualdb,
            max_bins,
            consensus_interval,
            nonnegative_scaling,
            consensus_anchors,
        };
        options.validate()?;
        let x = x.as_array();
        let y = y.as_array();
        let x_contiguous = x.as_standard_layout();
        let y_contiguous = y.as_standard_layout();

        let params = options.into_params()?;

        let (fit_result, tsl) = fit_boosted(x_contiguous.view(), y_contiguous.view(), &params);
        Ok((tsl.into(), FitResultPy::from(fit_result)))
    }
}
