//! Range sums over exact and bin-cumulative prefix statistics.

use crate::grid_tensor::state::PrefixStats;

#[inline]
pub(super) fn add_statistics<const LINEAR: bool>(target: &mut PrefixStats, value: PrefixStats) {
    target.s11 += value.s11;
    target.s22 += value.s22;
    target.s12 += value.s12;
    if LINEAR {
        target.t1 += value.t1;
        target.t2 += value.t2;
    }
}

/// Subtract two `PrefixStats` componentwise.
#[inline]
pub(super) fn sub_stats(a: PrefixStats, b: PrefixStats) -> PrefixStats {
    PrefixStats {
        s11: a.s11 - b.s11,
        s22: a.s22 - b.s22,
        s12: a.s12 - b.s12,
        t1: a.t1 - b.t1,
        t2: a.t2 - b.t2,
    }
}

/// Sum the per-stat prefix sums over the half-open sorted range `[start, end)`.
#[inline]
pub fn prefix_stats_range(prefix: &[PrefixStats], start: usize, end: usize) -> PrefixStats {
    if start == 0 {
        prefix[end - 1]
    } else {
        sub_stats(prefix[end - 1], prefix[start - 1])
    }
}

/// Read a bin-cumulative prefix sum at sorted position `pos`. `pos` must be
/// the last position of some bin (i.e., `pos = bin_edges[k] - 1` for some `k`
/// or `pos = n - 1`).
#[inline]
pub fn binned_prefix_stats_at(
    binned: &[PrefixStats],
    bin_edges: &[usize],
    pos: usize,
) -> PrefixStats {
    let k = bin_edges.partition_point(|&e| e <= pos);
    binned[k]
}

/// Bin-cumulative equivalent of `prefix_stats_range`: sum over sorted
/// positions `[start, end)`. Both endpoints must be bin-aligned.
#[inline]
pub fn binned_prefix_stats_range(
    binned: &[PrefixStats],
    bin_edges: &[usize],
    start: usize,
    end: usize,
) -> PrefixStats {
    let end_v = binned_prefix_stats_at(binned, bin_edges, end - 1);
    if start == 0 {
        end_v
    } else {
        sub_stats(end_v, binned_prefix_stats_at(binned, bin_edges, start - 1))
    }
}
