//! Mutable state of one grid-tensor fit: grid parameters, per-row caches and working statistics.

mod consensus;
mod intervals;
mod loop_state;
mod statistics;

pub(crate) use consensus::same_consensus_support;
pub use consensus::{BranchConsensus, ConsensusAnchors};
pub use intervals::AffectedRange;
pub use loop_state::{LoopState, MAX_CONSECUTIVE_RESPLIT};
pub use statistics::{
    BinEdges, CurvatureCache, IntervalStats, PrecomputedStatistics, PrefixStats,
    MAX_RANK_STREAM_ENTRIES,
};

#[cfg(feature = "evo-logging")]
use crate::grid_tensor::logging_helpers::LoggingState;
use crate::grid_tensor::splitting::SplitStrategyState;
use ndarray::{Array1, ArrayView1, ArrayView2};

#[derive(Debug)]
pub struct FittingState<'a> {
    // Core two-tensor grid state
    pub backbone_values: Vec<Vec<f64>>, // b_j^k per axis per interval
    pub tilt_values: Vec<Vec<f64>>,     // d_j^k per axis per interval
    pub boundaries: Vec<Vec<usize>>,
    pub scaling: f64,

    // Two-tensor stage scalars
    pub lambda_plus: f64,  // λ_+ ≥ 0
    pub lambda_minus: f64, // λ_- ≥ 0

    // Per-point caches
    pub f_plus: Array1<f64>,  // f_+(x_i) for each point i
    pub f_minus: Array1<f64>, // f_-(x_i) for each point i
    pub f: Array1<f64>,       // f(x_i) = f_plus[i] - f_minus[i]
    pub r_tilde: Array1<f64>, // within-stage residual: R_i - f(x_i)

    // Predictions and residuals (outer residuals for forest-level)
    pub y_hat: Array1<f64>,
    pub residuals: Array1<f64>,
    /// Current squared error, or the current IRLS quadratic surrogate for Huber refinement.
    pub current_error: f64,

    // Data references (immutable)
    pub x: ArrayView2<'a, f64>,
    pub labels: ArrayView1<'a, f64>,
    pub n: usize,
    pub p: usize,

    // Working buffers (performance optimization)
    pub precomputed_statistics: PrecomputedStatistics,

    // Loop control state
    pub loop_state: LoopState,

    // Split strategy state
    pub split_strategy_state: SplitStrategyState,

    // Fitting metadata
    pub iteration: usize,

    pub terminated: bool,

    /// Proximal pull of each branch toward reference branch functions.
    pub branch_consensus: Option<BranchConsensus>,

    /// Consensus Random proposals evaluate split scores at their sampled positions.
    pub(crate) sampled_split_scoring: bool,
    /// Some split scores require evaluation against the current prefix statistics.
    pub(crate) split_scores_stale: bool,
    /// Linear row/prefix statistics require a full rebuild before another action.
    pub(crate) linear_statistics_stale: bool,

    // Logging state (optional: only when logging enabled)
    #[cfg(feature = "evo-logging")]
    pub logging_state: Option<LoggingState>,
}

// LoggingState holds a channel sender, so a clone starts without logging.
impl<'a> Clone for FittingState<'a> {
    fn clone(&self) -> Self {
        Self {
            backbone_values: self.backbone_values.clone(),
            tilt_values: self.tilt_values.clone(),
            boundaries: self.boundaries.clone(),
            scaling: self.scaling,
            lambda_plus: self.lambda_plus,
            lambda_minus: self.lambda_minus,
            f_plus: self.f_plus.clone(),
            f_minus: self.f_minus.clone(),
            f: self.f.clone(),
            r_tilde: self.r_tilde.clone(),
            y_hat: self.y_hat.clone(),
            residuals: self.residuals.clone(),
            current_error: self.current_error,
            x: self.x,
            labels: self.labels,
            n: self.n,
            p: self.p,
            precomputed_statistics: self.precomputed_statistics.clone(),
            loop_state: self.loop_state.clone(),
            split_strategy_state: self.split_strategy_state.clone(),
            iteration: self.iteration,
            terminated: self.terminated,
            branch_consensus: self.branch_consensus.clone(),
            sampled_split_scoring: self.sampled_split_scoring,
            split_scores_stale: self.split_scores_stale,
            linear_statistics_stale: self.linear_statistics_stale,
            #[cfg(feature = "evo-logging")]
            logging_state: None,
        }
    }
}

/// A zero-filled contiguous array whose data begins `slot` KiB into its allocation.
///
/// Row caches built with distinct slots keep one row's elements in distinct L1 sets
/// when their allocations share a page alignment; the padding is never read.
pub(crate) fn staggered_zeros(n: usize, slot: usize) -> Array1<f64> {
    let pad = slot * 128;
    Array1::zeros(n + pad).slice_move(ndarray::s![pad..])
}

impl<'a> FittingState<'a> {
    pub fn new(x: ArrayView2<'a, f64>, y: ArrayView1<'a, f64>) -> Self {
        let split_strategy_state = SplitStrategyState::default();
        let loop_state = LoopState::default();
        let precomputed_statistics = PrecomputedStatistics::new(x.nrows(), x.ncols());
        let n = x.nrows();
        let p = x.ncols();

        // Initialize two-tensor grid: backbone = 1, tilt = 0 for all intervals
        let backbone_values = vec![vec![1.0]; p];
        let tilt_values = vec![vec![0.0]; p];

        // Small positive placeholders until the refinement initializes λ± from the residuals.
        const EPS_LAMBDA: f64 = 1e-10;
        let lambda_plus = EPS_LAMBDA;
        let lambda_minus = EPS_LAMBDA;

        // Initialize per-point caches
        // Initially f_plus = lambda_plus, f_minus = lambda_minus (since all b=1, d=0)
        // Each cache starts a different number of KiB into its own allocation, so one
        // row's caches occupy distinct L1 sets even when the allocations are page-aligned.
        let mut f_plus = staggered_zeros(n, 1);
        f_plus.fill(lambda_plus);
        let mut f_minus = staggered_zeros(n, 2);
        f_minus.fill(lambda_minus);
        let mut f = staggered_zeros(n, 3);
        ndarray::Zip::from(&mut f)
            .and(&f_plus)
            .and(&f_minus)
            .for_each(|f, &plus, &minus| *f = plus - minus);

        // Initially y_hat = f (stage prediction)
        let mut y_hat = staggered_zeros(n, 4);
        y_hat.assign(&f);
        let mut residuals = staggered_zeros(n, 5);
        ndarray::Zip::from(&mut residuals)
            .and(&y)
            .and(&y_hat)
            .for_each(|residual, &label, &prediction| *residual = label - prediction);
        let current_error = residuals.iter().map(|r| r * r).sum();

        // r_tilde = labels - f, which equals the residuals since y_hat = f.
        let mut r_tilde = staggered_zeros(n, 6);
        r_tilde.assign(&residuals);

        Self {
            x,
            labels: y,
            backbone_values,
            tilt_values,
            boundaries: vec![vec![]; p],
            scaling: 1.0,
            lambda_plus,
            lambda_minus,
            f_plus,
            f_minus,
            f,
            r_tilde,
            y_hat,
            residuals,
            current_error,
            n,
            p,
            precomputed_statistics,
            loop_state,
            split_strategy_state,
            iteration: 0,
            terminated: false,
            branch_consensus: None,
            sampled_split_scoring: false,
            split_scores_stale: false,
            linear_statistics_stale: false,
            #[cfg(feature = "evo-logging")]
            logging_state: None,
        }
    }

    /// Per-row sufficient statistics of the penalized local objective.
    ///
    /// With branch consensus `ρ, c₊, c₋`, the objective adds
    /// `ρ (f₊ − c₊)² + ρ (f₋ − c₋)²` per row, which contributes to `s11`, `s22`, `t1` and `t2`.
    #[inline]
    pub(crate) fn point_statistics(&self, row: usize, weight: f64) -> PrefixStats {
        self.point_statistics_impl::<true>(row, weight)
    }

    #[inline]
    pub(crate) fn point_curvature_statistics(&self, row: usize, weight: f64) -> PrefixStats {
        self.point_statistics_impl::<false>(row, weight)
    }

    #[inline]
    fn point_statistics_impl<const LINEAR: bool>(&self, row: usize, weight: f64) -> PrefixStats {
        let (fp, fm, rt) = (self.f_plus[row], self.f_minus[row], self.r_tilde[row]);
        let mut stats = PrefixStats {
            s11: weight * fp * fp,
            s22: weight * fm * fm,
            s12: weight * fp * fm,
            t1: if LINEAR { weight * rt * fp } else { 0.0 },
            t2: if LINEAR { weight * rt * fm } else { 0.0 },
        };
        if let Some((rho, target_plus, target_minus)) = self
            .branch_consensus
            .as_ref()
            .and_then(|consensus| consensus.target(row))
        {
            stats.s11 += rho * fp * fp;
            stats.s22 += rho * fm * fm;
            if LINEAR {
                stats.t1 += rho * (target_plus - fp) * fp;
                stats.t2 += rho * (fm - target_minus) * fm;
            }
        }
        stats
    }

    pub(crate) fn require_current_statistics(&self) {
        assert!(
            !self.linear_statistics_stale,
            "linear statistics require a full refresh"
        );
    }

    /// The fineness budget is spent and merges are disabled, so no structural action remains.
    #[inline]
    pub fn structure_is_final(&self, n_iter: usize) -> bool {
        !self.split_strategy_state.merge_enabled && self.loop_state.fineness >= n_iter
    }

    /// Check if we're in Stage 1 positive-only mode
    /// Stage 1 is active when λ_- = 0 and all d_j = 0
    pub fn is_stage1_positive_only(&self) -> bool {
        // Check lambda_minus is zero (or very close to zero)
        const EPS: f64 = 1e-10;
        if self.lambda_minus > EPS {
            return false;
        }
        // Check all tilt values are zero
        self.tilt_values
            .iter()
            .all(|d_vec| d_vec.iter().all(|&d| d.abs() < EPS))
    }
}
