//! Maps the flat Python fit keyword arguments onto the core parameter builders.

use pyo3::{exceptions::PyValueError, prelude::*};
use tsl::{
    forest::params::{TSLBoostedParams, TSLBoostedParamsBuilder},
    grid_tensor::{
        params::{
            RefinementStrategyParams, RefinementStrategyParamsBuilder, SplitStrategyParams,
            SplitStrategyParamsBuilder,
        },
        GridTensorParams, GridTensorParamsBuilder,
    },
};

/// The keyword arguments of `TSL.fit` that configure the boosted fit.
pub struct BoostedFitOptions<'a> {
    pub epochs: usize,
    pub decay: f64,
    pub n_trees: usize,
    pub n_iter: usize,
    pub split_try: usize,
    pub colsample_bysplit: f64,
    pub alpha: f64,
    pub complexity_penalty: f64,
    pub min_split_loss: f64,
    pub min_interval_samples: usize,
    pub refinement_strategy: &'a str,
    pub update_clamp: f64,
    pub tilt_tau: f64,
    pub tilt_rho: f64,
    pub split_strategy: &'a str,
    pub top_k: usize,
    pub must_fill_all_k: bool,
    pub similarity_threshold: f64,
    pub consensus: f64,
    pub seed: u64,
    pub verbosity: u8,
    pub visualdb: Option<String>,
    pub max_bins: Option<u16>,
    pub consensus_interval: usize,
    pub nonnegative_scaling: bool,
    pub consensus_anchors: Option<usize>,
    pub bootstrap: bool,
}

impl BoostedFitOptions<'_> {
    /// Rejects numeric hyperparameters outside their valid ranges.
    pub fn validate(&self) -> PyResult<()> {
        let consensus = self.consensus;
        if !(consensus.is_finite() && consensus >= 0.0) {
            return Err(PyValueError::new_err(format!(
                "consensus must be finite and nonnegative, got {consensus}"
            )));
        }
        if self.max_bins.is_some_and(|bins| bins < 2) {
            return Err(PyValueError::new_err("max_bins must be at least 2"));
        }
        if self.consensus_interval == 0 {
            return Err(PyValueError::new_err(
                "consensus_interval must be at least 1",
            ));
        }
        if self.consensus_anchors == Some(0) {
            return Err(PyValueError::new_err("consensus_anchors must be positive"));
        }
        Ok(())
    }

    /// Builds the core boosted-fit parameters, rejecting unknown strategy names.
    pub fn into_params(self) -> PyResult<TSLBoostedParams> {
        let split_strategy_params = self.split_strategy_params()?;
        let refinement_strategy_params = self.refinement_strategy_params()?;

        Ok(TSLBoostedParamsBuilder::new()
            .epochs(self.epochs)
            .decay(self.decay)
            .n_trees(self.n_trees)
            .n_iter(self.n_iter)
            .split_strategy(split_strategy_params)
            .refinement_strategy(refinement_strategy_params)
            .similarity_threshold(self.similarity_threshold)
            .consensus(self.consensus)
            .consensus_interval(self.consensus_interval)
            .nonnegative_scaling(self.nonnegative_scaling)
            .consensus_anchors(self.consensus_anchors)
            .bootstrap(self.bootstrap)
            .max_bins(self.max_bins)
            .seed(self.seed)
            .log_level(match self.verbosity {
                0 => "off",
                1 => "info",
                2 => "debug",
                3 => "trace",
                _ => "info",
            })
            .visualdb_path(self.visualdb)
            .build())
    }

    fn split_strategy_params(&self) -> PyResult<SplitStrategyParams> {
        Ok(match self.split_strategy {
            "random" => SplitStrategyParamsBuilder::new()
                .random_split()
                .split_try(self.split_try)
                .colsample_bysplit(self.colsample_bysplit)
                .min_interval_samples(self.min_interval_samples)
                .min_split_loss(self.min_split_loss)
                .complexity_penalty(self.complexity_penalty)
                .build(),
            "best_split" => SplitStrategyParamsBuilder::new()
                .best_split()
                .min_interval_samples(self.min_interval_samples)
                .min_split_loss(self.min_split_loss)
                .complexity_penalty(self.complexity_penalty)
                .build(),
            "top_k" => SplitStrategyParamsBuilder::new()
                .top_k_splits()
                .top_k(self.top_k)
                .must_fill_all_k(self.must_fill_all_k)
                .min_interval_samples(self.min_interval_samples)
                .min_split_loss(self.min_split_loss)
                .complexity_penalty(self.complexity_penalty)
                .build(),
            s => {
                return Err(PyValueError::new_err(format!(
                    "Unknown split_strategy: '{s}'. Valid: 'random', 'best_split', 'top_k'"
                )))
            }
        })
    }

    fn refinement_strategy_params(&self) -> PyResult<RefinementStrategyParams> {
        let builder = match self.refinement_strategy {
            "l2" => RefinementStrategyParamsBuilder::new().l2(),
            "huber" => RefinementStrategyParamsBuilder::new().huber(),
            s => {
                return Err(PyValueError::new_err(format!(
                    "Unknown refinement_strategy: '{s}'. Valid: 'l2', 'huber'"
                )))
            }
        };
        Ok(builder
            .alpha(self.alpha)
            .update_clamp(self.update_clamp)
            .tilt_tau(self.tilt_tau)
            .tilt_rho(self.tilt_rho)
            .build())
    }
}

/// Builds the parameters of a standalone `GridTensor.fit`, which always splits at random.
pub fn grid_tensor_params(
    n_iter: usize,
    split_try: usize,
    colsample_bysplit: f64,
    complexity_penalty: f64,
) -> GridTensorParams {
    GridTensorParamsBuilder::new()
        .n_iter(n_iter)
        .split_strategy(
            SplitStrategyParamsBuilder::new()
                .random_split()
                .split_try(split_try)
                .colsample_bysplit(colsample_bysplit)
                .min_interval_samples(1)
                .complexity_penalty(complexity_penalty)
                .build(),
        )
        .build()
}
