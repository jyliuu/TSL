#[cfg(feature = "evo-logging")]
use crate::grid_tensor::logging_helpers::LoggingState;
use crate::grid_tensor::splitting::SplitStrategyState;
use ndarray::{Array1, ArrayView1, ArrayView2};
use std::sync::Arc;

#[derive(Debug)]
pub struct FittingState<'a> {
    // Core two-tensor grid state
    pub backbone_values: Vec<Vec<f64>>, // b_j^k per axis per interval
    pub tilt_values: Vec<Vec<f64>>,     // d_j^k per axis per interval
    pub boundaries: Vec<Vec<usize>>,
    pub scaling: f64,

    // Two-tensor stage scalars
    pub lambda_plus: f64,  // λ_+ > 0
    pub lambda_minus: f64, // λ_- > 0

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
    // Note: LoggingState is not Clone (contains Sender), so FittingState is not Clone
    #[cfg(feature = "evo-logging")]
    pub logging_state: Option<LoggingState>,
}

// Manual Clone implementation that excludes logging_state (can't clone Sender)
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
            logging_state: None, // Can't clone LoggingState (contains Sender)
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
        let split_strategy_state = SplitStrategyState::new();
        let loop_state = LoopState::new();
        let precomputed_statistics = PrecomputedStatistics::new(x.nrows(), x.ncols());
        let n = x.nrows();
        let p = x.ncols();

        // Initialize two-tensor grid: backbone = 1, tilt = 0 for all intervals
        let backbone_values = vec![vec![1.0]; p];
        let tilt_values = vec![vec![0.0]; p];

        // Initialize lambdas from residuals (will be properly initialized in refinement::initialize)
        // For now, set to small positive values
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

        // Initially r_tilde = R (within-stage residual equals outer residual when f=0)
        // But since we initialized f, we need to compute r_tilde = R - f
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

    /// Index of the interval on `col` that holds `row`: the number of boundaries at or
    /// below the row's rank, since an interval begins at its boundary rank.
    pub fn interval_index(&self, col: usize, row: usize) -> usize {
        let rank = self.precomputed_statistics.sort_order[col][row];
        self.boundaries[col].partition_point(|&b| b <= rank)
    }

    pub fn interval_range(&self, col: usize, interval_idx: usize) -> (usize, usize) {
        let start = if interval_idx == 0 {
            0
        } else {
            self.boundaries[col][interval_idx - 1]
        };
        let end = self.boundaries[col]
            .get(interval_idx)
            .copied()
            .unwrap_or(self.x.nrows());
        (start, end)
    }

    pub fn interval_range_left_and_right(
        &self,
        col: usize,
        interval_idx: usize,
    ) -> (usize, usize, usize) {
        let start = if interval_idx == 0 {
            0
        } else {
            self.boundaries[col][interval_idx - 1]
        };

        let index = if interval_idx < self.boundaries[col].len() {
            self.boundaries[col][interval_idx]
        } else {
            self.x.nrows()
        };

        let end = if interval_idx + 1 < self.boundaries[col].len() {
            self.boundaries[col][interval_idx + 1]
        } else {
            self.x.nrows()
        };
        (start, index, end)
    }

    /// Get affected ranges for all columns after a transformation
    /// Returns both tight point ranges (for statistics) and interval ranges (for filtering)
    pub(crate) fn get_affected_ranges(
        &self,
        col: usize,
        interval: (usize, usize),
    ) -> Vec<AffectedRange> {
        let (start, end) = interval;
        let updated_points = &self.precomputed_statistics.sorted_indices[col][start..end];

        let mut affected_ranges = Vec::with_capacity(self.x.ncols());

        // For the split column, compute which allowed_intervals intersect with the affected interval
        let point_range = (start, end - 1);
        // Compute the allowed_interval index range that intersects with the affected point range
        let interval_range = self.compute_allowed_interval_index_range(col, start, end - 1);
        affected_ranges.push(AffectedRange {
            col,
            point_range,
            interval_range,
        });

        // For other columns, compute tight point ranges and map to interval ranges
        for other_col in (0..self.x.ncols()).filter(|&c| c != col) {
            let mut min_pos = usize::MAX;
            let mut max_pos = 0usize;
            for &pt_idx in updated_points.iter() {
                let pos = self.precomputed_statistics.sort_order[other_col][pt_idx];
                if pos < min_pos {
                    min_pos = pos;
                }
                if pos > max_pos {
                    max_pos = pos;
                }
            }
            if min_pos == usize::MAX {
                // Empty range - skip this column
                continue;
            }

            let point_range = (min_pos, max_pos);

            // Expand the point range to include the full data intervals that were touched
            // This ensures that all allowed intervals within the recomputed data intervals are filtered
            let (d_lo, d_hi) = self.compute_boundary_index_range(other_col, min_pos, max_pos);
            let start_expanded = self.interval_range(other_col, d_lo).0;
            let end_expanded = self.interval_range(other_col, d_hi).1;

            // Compute the allowed_interval index range that intersects with the EXPANDED range
            let interval_range = self.compute_allowed_interval_index_range(
                other_col,
                start_expanded,
                end_expanded.saturating_sub(1),
            );

            affected_ranges.push(AffectedRange {
                col: other_col,
                point_range,
                interval_range,
            });
        }

        affected_ranges
    }

    /// Convert datapoint position range to interval index range
    /// Given boundaries and a range of affected datapoint positions [lo_pos, hi_pos],
    /// returns the range of interval indices [lo_idx, hi_idx] whose data ranges intersect
    /// with the position range.
    /// When boundaries is empty, there is one interval (index 0) covering all points.
    pub(crate) fn compute_boundary_index_range(
        &self,
        col: usize,
        lo_pos: usize,
        hi_pos: usize,
    ) -> (usize, usize) {
        let boundaries = &self.boundaries[col];

        if boundaries.is_empty() {
            // No boundaries means one interval (index 0) covering all points
            // If the position range is non-empty, it intersects with interval 0
            if lo_pos <= hi_pos {
                return (0, 0);
            } else {
                // Empty range - return (1, 0) to indicate no intervals
                return (1, 0);
            }
        }

        if lo_pos > hi_pos {
            // Empty range - return (1, 0) to indicate no intervals
            return (1, 0);
        }

        // Find the lowest interval index whose range intersects [lo_pos, hi_pos]
        // Interval k covers [boundaries[k-1], boundaries[k]).
        // We want k such that boundaries[k] > lo_pos.
        let lo = boundaries.partition_point(|&b| b <= lo_pos);

        // Find the highest interval index whose range intersects [lo_pos, hi_pos]
        // We want the interval containing hi_pos.
        let hi = boundaries.partition_point(|&b| b <= hi_pos);

        (lo, hi)
    }

    /// Convert datapoint position range to allowed_interval index range
    /// Given allowed_intervals and a range of affected datapoint positions [lo_pos, hi_pos],
    /// returns the range of allowed_interval indices [lo_idx, hi_idx] that intersect
    /// with the position range.
    /// Returns (1, 0) to indicate no intervals if the range is empty or no intervals intersect.
    ///
    /// An interval intersects [lo_pos, hi_pos] if: interval.start < hi_pos + 1 && interval.end() > lo_pos
    /// Since allowed_intervals are sorted by start, intersecting intervals are consecutive.
    pub(crate) fn compute_allowed_interval_index_range(
        &self,
        col: usize,
        lo_pos: usize,
        hi_pos: usize,
    ) -> (usize, usize) {
        let allowed_intervals = &self.split_strategy_state.allowed_intervals[col];

        if allowed_intervals.is_empty() || lo_pos > hi_pos {
            return (1, 0);
        }

        // Find the first interval that could intersect: first where interval.end() > lo_pos
        // This is the first interval that ends after lo_pos
        let lo_idx = allowed_intervals.partition_point(|interval| interval.end() <= lo_pos);

        // Find the last interval that could intersect: last where interval.start <= hi_pos
        // This is the rightmost interval that starts at or before hi_pos
        let hi_candidate = allowed_intervals
            .partition_point(|interval| interval.start <= hi_pos)
            .saturating_sub(1);

        // Check if any intervals actually intersect
        if lo_idx > hi_candidate {
            return (1, 0);
        }

        // Verify the last candidate actually intersects (it might start before but end before lo_pos)
        let hi_idx = if allowed_intervals[hi_candidate].end() > lo_pos {
            hi_candidate
        } else {
            // Last candidate doesn't intersect, find the actual last intersecting one
            // We need to search backwards from hi_candidate
            let mut idx = hi_candidate;
            while idx >= lo_idx {
                if allowed_intervals[idx].end() > lo_pos {
                    break;
                }
                if idx == 0 {
                    return (1, 0);
                }
                idx -= 1;
            }
            idx
        };

        (lo_idx, hi_idx)
    }
}

/// Fixed training rows on which branch consensus is evaluated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsensusAnchors {
    pub rows: Vec<usize>,
    pub row_to_anchor: Vec<usize>,
}

impl ConsensusAnchors {
    pub fn new(n: usize, rows: Vec<usize>) -> Self {
        assert!(!rows.is_empty(), "consensus anchors must not be empty");
        assert!(
            rows.windows(2).all(|pair| pair[0] < pair[1]),
            "anchor rows must be sorted and unique"
        );
        assert!(
            rows.last().is_some_and(|&row| row < n),
            "anchor row is outside the training data"
        );
        let mut row_to_anchor = vec![usize::MAX; n];
        for (index, &row) in rows.iter().enumerate() {
            row_to_anchor[row] = index;
        }
        Self {
            rows,
            row_to_anchor,
        }
    }
}

pub(crate) fn same_consensus_support(
    left: &Option<Arc<ConsensusAnchors>>,
    right: &Option<Arc<ConsensusAnchors>>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        _ => false,
    }
}

/// Reference branches and their per-row penalty weight.
/// Anchored references have one entry per anchor and weight `ρ n / m`.
#[derive(Debug, Clone)]
pub struct BranchConsensus {
    pub weight: f64,
    pub plus: Array1<f64>,
    pub minus: Array1<f64>,
    pub anchors: Option<Arc<ConsensusAnchors>>,
}

impl BranchConsensus {
    pub fn target(&self, row: usize) -> Option<(f64, f64, f64)> {
        let index = match &self.anchors {
            Some(anchors) => {
                let index = anchors.row_to_anchor[row];
                if index == usize::MAX {
                    return None;
                }
                index
            }
            None => row,
        };
        Some((self.weight, self.plus[index], self.minus[index]))
    }
}

/// Per-interval aggregate statistics for O(1) resplit/merge lookups
#[derive(Debug, Clone, Copy, Default)]
pub struct IntervalStats {
    /// Sum of S11 (f_plus^2) in this interval
    pub sum_s11: f64,
    /// Sum of S22 (f_minus^2) in this interval
    pub sum_s22: f64,
    /// Sum of S12 (f_plus * f_minus) in this interval
    pub sum_s12: f64,
    /// Sum of t1 (r_tilde * f_plus) in this interval
    pub sum_t1: f64,
    /// Sum of t2 (r_tilde * f_minus) in this interval
    pub sum_t2: f64,
    /// Number of points in this interval
    pub n: usize,
}

impl IntervalStats {
    /// Compute union stats by adding two interval stats.
    ///
    /// Uses the union additivity property: S^U = S^A + S^B, t^U = t^A + t^B
    ///
    /// # Arguments
    /// * `left` - Stats for left interval A
    /// * `right` - Stats for right interval B
    ///
    /// # Returns
    /// Union stats U = A ∪ B
    pub fn union(left: &IntervalStats, right: &IntervalStats) -> IntervalStats {
        IntervalStats {
            sum_s11: left.sum_s11 + right.sum_s11,
            sum_s22: left.sum_s22 + right.sum_s22,
            sum_s12: left.sum_s12 + right.sum_s12,
            sum_t1: left.sum_t1 + right.sum_t1,
            sum_t2: left.sum_t2 + right.sum_t2,
            n: left.n + right.n,
        }
    }
}

/// Represents affected ranges for a column after a transformation
/// Contains both the tight point range (for statistics updates) and
/// the interval range (for cache refreshing)
#[derive(Debug, Clone, Copy)]
pub struct AffectedRange {
    /// Column index
    pub col: usize,
    /// Tight point position range (lo, hi) for statistics updates
    /// This is the minimal range covering all affected data points
    pub point_range: (usize, usize),
    /// Interval index range (blo, bhi) for cache refreshing and filtering
    /// This is the range of interval indices whose data ranges intersect with point_range
    pub interval_range: (usize, usize),
}

/// Per-feature bin-edge positions for histogram binning. Empty `Vec<usize>`
/// means "no binning for this feature" (exact path); otherwise contains
/// sorted bin-boundary positions in sorted-index space.
pub type BinEdges = Vec<Vec<usize>>;

/// Packed (s11, s22, s12, t1, t2) tuple. Used as the element type of
/// per-column prefix-sum arrays so that the segment-prefix sweep touches one
/// 40-byte struct per index (one cache line per two indices) instead of
/// five independent `Vec<f64>`. Order of summation and arithmetic is
/// unchanged — bit-for-bit parity with the prior SoA-of-5 layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct PrefixStats {
    pub s11: f64,
    pub s22: f64,
    pub s12: f64,
    pub t1: f64,
    pub t2: f64,
}

impl PrefixStats {
    pub const ZERO: PrefixStats = PrefixStats {
        s11: 0.0,
        s22: 0.0,
        s12: 0.0,
        t1: 0.0,
        t2: 0.0,
    };
}

impl std::ops::AddAssign for PrefixStats {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.s11 += rhs.s11;
        self.s22 += rhs.s22;
        self.s12 += rhs.s12;
        self.t1 += rhs.t1;
        self.t2 += rhs.t2;
    }
}

/// Entry budget for the rank-ordered bin streams (32 MiB of `u16`).
pub const MAX_RANK_STREAM_ENTRIES: usize = 1 << 24;

/// One row's curvature contributions, `s11 = wᵢ f₊²`, `s22 = wᵢ f₋²` and `s12 = wᵢ f₊ f₋`,
/// with the consensus weight folded into `s11` and `s22`; stored together so a row's
/// cache is one cache line.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct CurvatureCache {
    pub s11: f64,
    pub s22: f64,
    pub s12: f64,
}

/// Working buffers containing pre-computed performance caches
#[derive(Debug, Clone)]
pub struct PrecomputedStatistics {
    /// Prefix sums of (s11, s22, s12, t1, t2) by sorted position [col][pos].
    /// Empty for binned columns (`binned_prefix_sums[col]` used instead).
    pub prefix_sums: Vec<Vec<PrefixStats>>,
    /// Per-point curvature contributions, stored once since they are point-specific.
    pub curvature: Vec<CurvatureCache>,
    /// Per-point t1 contributions: c_t1[i] = w_i * r_tilde[i] * f_plus[i]
    /// Stored once (not per-column) since these are point-specific
    pub c_t1: Vec<f64>,
    /// Per-point t2 contributions: c_t2[i] = w_i * r_tilde[i] * f_minus[i]
    /// Stored once (not per-column) since these are point-specific
    pub c_t2: Vec<f64>,
    /// Per-interval statistics [col][interval_idx] for O(1) resplit/merge lookups
    /// Each column starts with one interval covering all points
    pub interval_stats: Vec<Vec<IntervalStats>>,
    /// Pre-computed split update factors [col][pos]
    pub update_pairs_split_left: Vec<Vec<(f64, f64)>>,
    pub update_pairs_split_right: Vec<Vec<(f64, f64)>>,
    /// Pre-computed resplit update factors for left side [col][boundary_pos] -> (u_plus_L, u_minus_L)
    pub update_pairs_resplit_left: Vec<Vec<(f64, f64)>>,
    /// Pre-computed resplit update factors for right side [col][boundary_pos] -> (u_plus_R, u_minus_R)
    pub update_pairs_resplit_right: Vec<Vec<(f64, f64)>>,
    /// Absolute axis factors for a merged interval [col][boundary_pos] -> (a_plus, a_minus)
    pub update_pairs_merge: Vec<Vec<(f64, f64)>>,
    /// Pre-computed split error reductions [col][pos]
    pub error_reductions_split: Vec<Vec<f64>>,
    /// Pre-computed split error reductions per interval [col][pos] -> (left, right)
    pub error_reductions_split_pairs: Vec<Vec<(f64, f64)>>,
    /// Pre-computed resplit error reductions [col][boundary_pos]
    pub error_reductions_resplit: Vec<Vec<f64>>,
    /// Pre-computed resplit error reductions per interval [col][boundary_pos] -> (left, right)
    pub error_reductions_resplit_pairs: Vec<Vec<(f64, f64)>>,
    /// Pre-computed merge error reductions [col][boundary_pos]
    pub error_reductions_merge: Vec<Vec<f64>>,
    /// Sorted row indices [col][rank] -> row mapping
    pub sorted_indices: Vec<Vec<usize>>,
    /// Rank mapping [col][row] -> rank mapping
    pub sort_order: Vec<Vec<usize>>,

    /// Per-feature bin-edge positions for histogram binning.
    /// `bin_edges[col]` is sorted in ascending order. Empty (default) means
    /// "no binning for this column — iterate all positions as candidates".
    pub bin_edges: BinEdges,

    /// Immutable row-to-bin mappings shared by cloned grid fitting states.
    /// Exact and constant columns have empty mappings.
    pub bin_indices: Vec<std::sync::Arc<[u16]>>,
    /// Every feature's bin mapping in each column's sorted-row order,
    /// `bin_ranks[col][feature * n + rank]`, so an action's affected ranks read each
    /// feature's bins as one contiguous stream. Empty unless every column is binned and
    /// the `p * p * n` entries fit [`MAX_RANK_STREAM_ENTRIES`].
    pub bin_ranks: Vec<std::sync::Arc<[u16]>>,
    /// Reusable per-action curvature deltas in affected-row order; empty between actions.
    pub curvature_deltas: Vec<[f64; 3]>,

    /// Bin-cumulative prefix sums (length B = bin_edges[col].len() + 1) used in
    /// the binned path. `binned_prefix_sums[col][k]` equals the cumulative
    /// stats sum for all sorted positions in bins `0..=k`. Empty when
    /// `bin_edges[col]` is empty (exact path uses `prefix_sums` instead).
    pub binned_prefix_sums: Vec<Vec<PrefixStats>>,

    pub initialized: bool,
    /// Branch-consensus weight folded into the stored `s11`/`s22` statistics.
    pub consensus_weight: f64,
    /// Anchor support folded into the stored curvature statistics.
    pub consensus_anchors: Option<Arc<ConsensusAnchors>>,
}

impl PrecomputedStatistics {
    pub fn new(n: usize, p: usize) -> Self {
        Self {
            prefix_sums: vec![vec![]; p],
            curvature: vec![CurvatureCache::default(); n],
            c_t1: vec![0.0; n],
            c_t2: vec![0.0; n],
            // Each column starts with one interval covering all points
            interval_stats: vec![vec![IntervalStats::default()]; p],
            update_pairs_split_left: vec![vec![(f64::NAN, f64::NAN); n]; p],
            update_pairs_split_right: vec![vec![(f64::NAN, f64::NAN); n]; p],
            update_pairs_resplit_left: vec![vec![]; p],
            update_pairs_resplit_right: vec![vec![]; p],
            update_pairs_merge: vec![vec![]; p],
            error_reductions_split: vec![vec![0.0; n]; p],
            error_reductions_split_pairs: vec![vec![(0.0, 0.0); n]; p],
            error_reductions_resplit: vec![vec![]; p],
            error_reductions_resplit_pairs: vec![vec![]; p],
            error_reductions_merge: vec![vec![]; p],
            sorted_indices: vec![vec![]; p],
            sort_order: vec![vec![0; n]; p],
            bin_edges: vec![vec![]; p],
            bin_indices: vec![std::sync::Arc::from([]); p],
            bin_ranks: Vec::new(),
            curvature_deltas: Vec::new(),
            binned_prefix_sums: vec![vec![]; p],
            initialized: false,
            consensus_weight: 0.0,
            consensus_anchors: None,
        }
    }
}

/// Maximum number of consecutive resplits allowed before forcing a different action
pub const MAX_CONSECUTIVE_RESPLIT: usize = 5;

/// Loop control state for tracking algorithm progress
#[derive(Debug, Clone)]
pub struct LoopState {
    /// Net number of splits performed (splits +1, resplits 0, merges -1)
    pub fineness: usize,
    /// Total number of splits performed
    pub split_count: usize,
    /// Total number of resplits performed
    pub resplit_count: usize,
    /// Total number of merges performed
    pub merge_count: usize,
    /// Number of consecutive resplits (for loop prevention)
    pub consecutive_resplits: usize,
    /// Current iteration number
    pub iteration: usize,
}

impl Default for LoopState {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopState {
    pub fn new() -> Self {
        Self {
            fineness: 0,
            split_count: 0,
            resplit_count: 0,
            merge_count: 0,
            consecutive_resplits: 0,
            iteration: 0,
        }
    }
}
