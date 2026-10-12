//! The `tensorsl._tensorsl` extension module, registering its four Python classes.

use pyo3::prelude::*;

mod feature_importance;
mod fit_result;
mod grid_tensor;
mod model;
mod params;
mod partial_dependence;
mod persistence;
mod stage_predictor;

use fit_result::FitResultPy;
use grid_tensor::GridTensorPy;
use model::TSLPy;
use stage_predictor::StagePredictorPy;

#[pymodule]
fn _tensorsl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GridTensorPy>()?;
    m.add_class::<FitResultPy>()?;
    m.add_class::<StagePredictorPy>()?;
    m.add_class::<TSLPy>()?;
    Ok(())
}
