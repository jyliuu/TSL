use crate::{
    grid_tensor::params::{
        GridTensorParams, GridTensorParamsBuilder, RefinementStrategyParams, SplitStrategyParams,
    },
    stage_predictor::Aggregation,
};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct StagePredictorParams {
    pub n_trees: usize,
    pub tg_params: GridTensorParams,
    pub similarity_threshold: f64,
    pub aggregation_method: Aggregation,
    /// Strength ρ of the pull of each grid's branches toward the mean branches of the
    /// other grids; 0 fits the grids independently.
    pub consensus: f64,
    /// Maximum local action attempts between peer-reference updates.
    pub consensus_interval: usize,
    /// Optional uniform training-row sample for the branch-consensus penalty.
    pub consensus_anchors: Option<usize>,
    /// Fit every grid of an independent bag to its own bootstrap resample of the rows.
    pub bootstrap: bool,
}

#[derive(Debug)]
pub struct StagePredictorParamsBuilder {
    n_trees: usize,
    tg_params_builder: GridTensorParamsBuilder,
    similarity_threshold: f64,
    aggregation_method: Aggregation,
    consensus: f64,
    consensus_interval: usize,
    consensus_anchors: Option<usize>,
    bootstrap: bool,
}

impl StagePredictorParamsBuilder {
    pub fn new() -> Self {
        Self {
            n_trees: 100,
            tg_params_builder: GridTensorParamsBuilder::new(),
            similarity_threshold: 0.0,
            aggregation_method: Aggregation::Combined,
            consensus: 0.0,
            consensus_interval: 1,
            consensus_anchors: None,
            bootstrap: true,
        }
    }

    pub fn n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = n_trees;
        self
    }

    pub fn aggregation_method(mut self, aggregation_method: Aggregation) -> Self {
        self.aggregation_method = aggregation_method;
        self
    }

    pub fn similarity_threshold(mut self, similarity_threshold: f64) -> Self {
        self.similarity_threshold = similarity_threshold;
        self
    }

    /// Fit the bag in lockstep, adding `ρ (‖f₊ − c₊‖² + ‖f₋ − c₋‖²)` to each grid's
    /// objective, where `c±` are the mean branches of the other grids and `ρ = consensus`.
    pub fn consensus(mut self, consensus: f64) -> Self {
        self.consensus = consensus;
        self
    }

    pub fn consensus_interval(mut self, consensus_interval: usize) -> Self {
        assert!(
            consensus_interval >= 1,
            "consensus_interval must be at least 1"
        );
        self.consensus_interval = consensus_interval;
        self
    }

    pub fn consensus_anchors(mut self, consensus_anchors: Option<usize>) -> Self {
        assert!(
            consensus_anchors != Some(0),
            "consensus_anchors must be positive"
        );
        self.consensus_anchors = consensus_anchors;
        self
    }

    /// Fit each grid of an independent bag (`consensus` 0, at least two grids) to a
    /// bootstrap resample: row `i` is weighted by the number of times it is drawn in `n`
    /// draws with replacement. A consensus bag fits every grid to all rows.
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.tg_params_builder = self.tg_params_builder.n_iter(n_iter);
        self
    }

    pub fn split_strategy(mut self, strategy: SplitStrategyParams) -> Self {
        self.tg_params_builder = self.tg_params_builder.split_strategy(strategy);
        self
    }

    pub fn refinement_strategy(mut self, strategy: RefinementStrategyParams) -> Self {
        self.tg_params_builder = self.tg_params_builder.refinement_strategy(strategy);
        self
    }

    pub fn max_bins(mut self, max_bins: Option<u16>) -> Self {
        self.tg_params_builder = self.tg_params_builder.max_bins(max_bins);
        self
    }

    pub fn build(self) -> StagePredictorParams {
        StagePredictorParams {
            n_trees: self.n_trees,
            tg_params: self.tg_params_builder.build(),
            similarity_threshold: self.similarity_threshold,
            aggregation_method: self.aggregation_method,
            consensus: self.consensus,
            consensus_interval: self.consensus_interval,
            consensus_anchors: self.consensus_anchors,
            bootstrap: self.bootstrap,
        }
    }
}

impl Default for StagePredictorParamsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for StagePredictorParams {
    fn default() -> Self {
        StagePredictorParamsBuilder::new().build()
    }
}
