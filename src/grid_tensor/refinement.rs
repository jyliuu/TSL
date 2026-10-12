//! Refinement module for updating tree statistics after splits
//!
//! This module contains the refinement strategies that update precomputed statistics
//! after each split/resplit/merge operation. The actual tree updates are handled by
//! the reducer in `grid/reducer.rs`.

mod binned_curvature;
mod boundary_scores;
mod gain;
mod incremental;
mod initialize;
mod prefix;
mod refresh;
mod scores;
mod split_scores;
#[cfg(test)]
mod tests;

pub use gain::{huber_weighting, l2_gain_raw, l2_update_unanchored};
pub use prefix::{
    binned_prefix_stats_at, binned_prefix_stats_range, prefix_range, prefix_stats_range,
};

use crate::grid_tensor::two_tensor_solver::{DEFAULT_V_MAX, DEFAULT_V_MIN};

pub enum RefinementStrategy {
    L2Refinement {
        /// L2 penalty on the log-backbone update.
        alpha: f64,
        /// L2 penalty on the tilt update.
        tilt_tau: f64,
        /// L1 penalty on the tilt update.
        tilt_rho: f64,
        /// Prior sample size for parent anchoring (tau_0).
        /// Interpreted as "how many samples worth of confidence in the parent".
        prior_sample_size: f64,
        update_clamp: f64,
    },
    HuberRefinement {
        alpha: f64,
        c: f64,
        tilt_tau: f64,
        tilt_rho: f64,
        prior_sample_size: f64,
        update_clamp: f64,
    },
}

impl RefinementStrategy {
    /// Multiplier limits combine the numerical box with the configured log-step cap.
    #[inline]
    pub(crate) fn multiplier_bounds(&self) -> (f64, f64) {
        let clamp = self.update_clamp();
        assert!(
            clamp >= 0.0,
            "update_clamp must be nonnegative or positive infinity"
        );
        if clamp >= DEFAULT_V_MAX.ln() {
            (DEFAULT_V_MIN, DEFAULT_V_MAX)
        } else {
            (
                (-clamp).exp().max(DEFAULT_V_MIN),
                clamp.exp().min(DEFAULT_V_MAX),
            )
        }
    }

    #[inline]
    pub fn tilt_tau(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { tilt_tau, .. }
            | RefinementStrategy::HuberRefinement { tilt_tau, .. } => *tilt_tau,
        }
    }

    #[inline]
    pub fn tilt_rho(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { tilt_rho, .. }
            | RefinementStrategy::HuberRefinement { tilt_rho, .. } => *tilt_rho,
        }
    }

    pub fn alpha(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { alpha, .. } => *alpha,
            RefinementStrategy::HuberRefinement { alpha, .. } => *alpha,
        }
    }

    #[inline]
    pub fn weight(&self, res: f64) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { .. } => 1.0,
            RefinementStrategy::HuberRefinement { c, .. } => huber_weighting(res, *c),
        }
    }

    /// Current IRLS quadratic surrogate used to compare every structural action
    /// and its stopping tolerance on one loss scale.
    pub(crate) fn weighted_squared_error(&self, residuals: &ndarray::Array1<f64>) -> f64 {
        residuals
            .iter()
            .map(|&residual| self.weight(residual) * residual * residual)
            .sum()
    }

    /// Returns the prior sample size (tau_0) for parent anchoring.
    ///
    /// This represents "how many samples worth of confidence we have that
    /// children should equal their parent". With tau_0 = 30, a child interval
    /// with 10 samples will be heavily shrunk toward the parent, while a child
    /// with 100 samples will mostly trust its own data.
    pub fn prior_sample_size(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement {
                prior_sample_size, ..
            } => *prior_sample_size,
            RefinementStrategy::HuberRefinement {
                prior_sample_size, ..
            } => *prior_sample_size,
        }
    }

    pub fn update_clamp(&self) -> f64 {
        match self {
            RefinementStrategy::L2Refinement { update_clamp, .. } => *update_clamp,
            RefinementStrategy::HuberRefinement { update_clamp, .. } => *update_clamp,
        }
    }
}
