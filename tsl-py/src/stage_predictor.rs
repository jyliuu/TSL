//! The `StagePredictor` Python class: one boosting stage, its bagged grids and OLS scaling.

use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, types::PyList};
use tsl::stage_predictor::StagePredictor;

use crate::grid_tensor::GridTensorPy;

#[derive(Debug)]
#[pyclass(name = "StagePredictor")]
pub struct StagePredictorPy(StagePredictor);

impl From<StagePredictor> for StagePredictorPy {
    fn from(stage_predictor: StagePredictor) -> Self {
        StagePredictorPy(stage_predictor)
    }
}

#[pymethods]
impl StagePredictorPy {
    #[getter]
    pub fn get_grid_tensors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let grid_tensors_py: Vec<GridTensorPy> = self
            .0
            .get_grid_tensors()
            .iter()
            .map(|tg| GridTensorPy::from(tg.clone()))
            .collect();

        PyList::new(py, grid_tensors_py)
    }

    #[getter]
    pub fn get_combined_grid_tensor(&self) -> PyResult<GridTensorPy> {
        Ok(GridTensorPy::from(self.0.get_primary_grid_tensor().clone()))
    }

    #[getter]
    pub fn get_candidate_indices(&self) -> PyResult<Vec<usize>> {
        Ok(self.0.get_candidate_indices().unwrap_or_default().to_vec())
    }

    #[getter]
    pub fn get_scaling_plus(&self) -> Option<f64> {
        self.0.scaling_plus
    }

    #[getter]
    pub fn get_scaling_minus(&self) -> Option<f64> {
        self.0.scaling_minus
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
}
