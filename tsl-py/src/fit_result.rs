//! The `FitResult` Python class: training error, residuals and fitted values of a fit.

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use tsl::FitResult;

#[derive(Debug)]
#[pyclass(name = "FitResult")]
pub struct FitResultPy {
    #[pyo3(get)]
    err: f64,
    #[pyo3(get)]
    residuals: Py<PyArray1<f64>>,
    #[pyo3(get)]
    y_hat: Py<PyArray1<f64>>,
}

impl From<FitResult> for FitResultPy {
    fn from(fit_result: FitResult) -> Self {
        Python::with_gil(|py| FitResultPy {
            err: fit_result.err,
            residuals: fit_result.residuals.to_pyarray(py).unbind(),
            y_hat: fit_result.y_hat.to_pyarray(py).unbind(),
        })
    }
}

#[pymethods]
impl FitResultPy {
    fn __repr__(&self) -> String {
        format!(
            "FitResult(error={}, residuals={}, y_hat={})",
            self.err, self.residuals, self.y_hat
        )
    }
}
