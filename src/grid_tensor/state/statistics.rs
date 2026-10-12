//! Sufficient statistics and candidate caches that score split, resplit and merge proposals.

use super::ConsensusAnchors;
use std::sync::Arc;

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

/// Per-feature bin-edge positions for histogram binning. Empty `Vec<usize>`
/// means "no binning for this feature" (exact path); otherwise contains
/// sorted bin-boundary positions in sorted-index space.
pub type BinEdges = Vec<Vec<usize>>;

/// Packed (s11, s22, s12, t1, t2) tuple: the element type of the per-column
/// prefix-sum arrays, so the segment-prefix sweep reads one 40-byte struct per
/// index (one cache line per two indices).
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
    /// Per-point t1 contributions: c_t1[i] = w_i * r_tilde[i] * f_plus[i],
    /// shared by every column
    pub c_t1: Vec<f64>,
    /// Per-point t2 contributions: c_t2[i] = w_i * r_tilde[i] * f_minus[i],
    /// shared by every column
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
