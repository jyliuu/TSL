use ndarray::{Array1, ArrayView1};

pub mod forest;
pub mod grid_tensor;
pub mod logging;
pub mod stage_predictor;

#[derive(Debug)]
pub struct FitResult {
    /// Mean squared residual, weighted by the sample weights of a weighted fit.
    pub err: f64,
    pub residuals: Array1<f64>,
    pub y_hat: Array1<f64>,
}

/// `Σᵢ sᵢ rᵢ² / Σᵢ sᵢ`, or the plain mean of `rᵢ²` without sample weights.
pub(crate) fn weighted_mean_squared_error(
    residuals: ArrayView1<f64>,
    sample_weight: Option<ArrayView1<f64>>,
) -> f64 {
    match sample_weight {
        None => residuals.pow2().mean().unwrap(),
        Some(sample_weight) => {
            let total: f64 = sample_weight.sum();
            residuals
                .iter()
                .zip(sample_weight)
                .map(|(r, s)| s * r * r)
                .sum::<f64>()
                / total
        }
    }
}
