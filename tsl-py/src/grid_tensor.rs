//! The `GridTensor` Python class: one fitted separable component and its standalone fit.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{
    prelude::*,
    types::{PyList, PyType},
};
use rand::{rngs::StdRng, SeedableRng};
use tsl::grid_tensor::{self, GridTensor};

use crate::{fit_result::FitResultPy, params};

#[derive(Debug)]
#[pyclass(name = "GridTensor")]
pub struct GridTensorPy(GridTensor);

impl From<GridTensor> for GridTensorPy {
    fn from(tg: GridTensor) -> Self {
        GridTensorPy(tg)
    }
}

#[pymethods]
impl GridTensorPy {
    #[getter]
    pub fn get_scaling(&self) -> f64 {
        self.0.scaling
    }

    #[getter]
    pub fn get_splits<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Convert the reference to a PyList
        PyList::new(py, &self.0.splits)
    }

    #[getter]
    pub fn get_intervals<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Convert the reference to a PyList
        PyList::new(py, &self.0.intervals)
    }

    #[getter]
    pub fn mean_factor<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mean_factor = self.0.get_mean_factor();
        PyList::new(py, mean_factor)
    }

    #[getter]
    pub fn grid_values<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.mean_factor(py)
    }

    #[getter]
    pub fn backbone_values<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, &self.0.backbone_values)
    }

    #[getter]
    pub fn tilt_values<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, &self.0.tilt_values)
    }

    #[getter]
    pub fn lambda_plus(&self) -> f64 {
        self.0.lambda_plus
    }

    #[getter]
    pub fn lambda_minus(&self) -> f64 {
        self.0.lambda_minus
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

    #[classmethod]
    #[pyo3(name = "fit")]
    #[pyo3(signature = (x, y, n_iter, split_try, colsample_bysplit, complexity_penalty=0.0, seed=42))]
    pub fn _fit<'py>(
        _cls: &Bound<'_, PyType>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        n_iter: usize,
        split_try: usize,
        colsample_bysplit: f64,
        complexity_penalty: f64,
        seed: u64,
    ) -> PyResult<(GridTensorPy, FitResultPy)> {
        let x = x.as_array();
        let y = y.as_array();
        let x_contiguous = x.as_standard_layout();
        let y_contiguous = y.as_standard_layout();
        let params =
            params::grid_tensor_params(n_iter, split_try, colsample_bysplit, complexity_penalty);
        let mut rng = StdRng::seed_from_u64(seed);
        let (fit_result, tg) =
            grid_tensor::fit(x_contiguous.view(), y_contiguous.view(), &params, &mut rng);
        Ok((tg.into(), FitResultPy::from(fit_result)))
    }
}
