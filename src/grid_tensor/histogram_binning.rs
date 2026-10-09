//! Histogram binning prologue for the per-tree split scan.
//!
//! The exact per-tree fit scans every sorted position as a candidate split
//! threshold. Histogram binning restricts the candidate set to a deterministic
//! set of quantile bin boundaries per feature, reducing the inner-loop solver
//! call count from O(n_interval) to O(n_bins).
//!
//! In addition, the per-tree sufficient-stat prefix sums (s11, s22, s12, t1,
//! t2) are *replaced* by length-B bin-cumulative arrays for binned columns:
//! `binned_prefix_sums_*[col][k]` = sum of `c_*` over sorted positions in
//! bins `0..=k`. This shrinks the per-call bucket scatter + segment
//! prefix-sum sweep inside `update_statistics` from O(span) ≈ O(n) to O(B),
//! which is the actual hot path identified by profiling (~68% self-time).
//!
//! For non-binned columns (`bin_edges[col].is_empty()`, including the entire
//! `max_bins = None` parity path), the length-n `prefix_sums_*[col]` storage
//! is retained and used unchanged.
//!
//! Determinism: bin-edge positions are derived from sorted-position indices,
//! so the same data + the same max_bins yield the same bins on every run.

use crate::grid_tensor::state::FittingState;

/// Compute deterministic bin-edge positions in sorted order for a feature.
///
/// Returns sorted-position indices separating distinct feature values.
/// Quantile targets snap to the nearest distinct-value boundary; equidistant
/// targets choose the lower rank. Repeated targets share one boundary.
///
/// When `n_bins >= unique-value count`, every distinct boundary is a
/// candidate (effectively the exact path).
pub(crate) fn quantile_bin_edge_positions(
    sorted_indices: &[usize],
    feature_values: &[f64],
    n_bins: usize,
) -> Vec<usize> {
    let n = sorted_indices.len();
    if n < 2 || n_bins < 2 {
        return Vec::new();
    }
    let n_bins = n_bins.min(n);

    let distinct: Vec<usize> = (1..n)
        .filter(|&pos| {
            feature_values[sorted_indices[pos]] != feature_values[sorted_indices[pos - 1]]
        })
        .collect();
    if distinct.len() < n_bins {
        return distinct;
    }

    let mut edges = Vec::with_capacity(n_bins - 1);
    for k in 1..n_bins {
        let target = (k * n) / n_bins;
        let right = distinct.partition_point(|&boundary| boundary < target);
        let boundary = if right == 0 {
            distinct[0]
        } else if right == distinct.len() {
            distinct[right - 1]
        } else if target - distinct[right - 1] <= distinct[right] - target {
            distinct[right - 1]
        } else {
            distinct[right]
        };
        if edges.last() != Some(&boundary) {
            edges.push(boundary);
        }
    }
    edges
}

/// Apply histogram-binning NaN mask to the split-candidate caches.
///
/// For each feature, marks every sorted position that is NOT a quantile
/// bin boundary as NaN in `error_reductions_split`, `update_pairs_split_left`,
/// `update_pairs_split_right`, and `error_reductions_split_pairs`. Bin
/// boundary positions retain their values (just computed by
/// `RefinementStrategy::initialize`).
///
/// Must be called AFTER `RefinementStrategy::initialize` and BEFORE
/// `SplitStrategy::initialize` (so that `allowed_intervals` is built from the
/// binned mask).
pub fn apply_histogram_binning_mask<'a>(
    state: FittingState<'a>,
    max_bins: Option<u16>,
) -> FittingState<'a> {
    apply_histogram_binning_mask_with_input(state, max_bins, None)
}

pub(crate) fn row_bin_indices(sorted_indices: &[usize], edges: &[usize]) -> std::sync::Arc<[u16]> {
    if edges.is_empty() {
        return std::sync::Arc::from([]);
    }
    assert!(edges.len() <= u16::MAX as usize);
    let mut membership = vec![0; sorted_indices.len()];
    let mut bin = 0;
    for (rank, &row) in sorted_indices.iter().enumerate() {
        while bin < edges.len() && edges[bin] <= rank {
            bin += 1;
        }
        membership[row] = bin as u16;
    }
    membership.into()
}

/// Per-column rank-ordered bin streams for every feature.
///
/// Empty unless every column maps all `n` rows and `p * p * n` entries fit `budget`.
pub(crate) fn rank_ordered_bin_streams_within(
    sorted_indices: &[Vec<usize>],
    bin_indices: &[std::sync::Arc<[u16]>],
    n: usize,
    budget: usize,
) -> Vec<std::sync::Arc<[u16]>> {
    let p = bin_indices.len();
    if p == 0
        || sorted_indices.len() != p
        || p.saturating_mul(p).saturating_mul(n) > budget
        || bin_indices.iter().any(|column| column.len() != n)
        || sorted_indices.iter().any(|order| order.len() != n)
    {
        return Vec::new();
    }
    sorted_indices
        .iter()
        .map(|order| {
            let mut table = vec![0u16; p * n];
            for (feature, bins) in bin_indices.iter().enumerate() {
                let stream = &mut table[feature * n..(feature + 1) * n];
                for (slot, &row) in stream.iter_mut().zip(order) {
                    *slot = bins[row];
                }
            }
            table.into()
        })
        .collect()
}

pub(crate) fn rank_ordered_bin_streams(
    sorted_indices: &[Vec<usize>],
    bin_indices: &[std::sync::Arc<[u16]>],
    n: usize,
) -> Vec<std::sync::Arc<[u16]>> {
    rank_ordered_bin_streams_within(
        sorted_indices,
        bin_indices,
        n,
        super::state::MAX_RANK_STREAM_ENTRIES,
    )
}

pub(crate) fn apply_histogram_binning_mask_with_input<'a>(
    mut state: FittingState<'a>,
    max_bins: Option<u16>,
    prepared: Option<&super::prepared_input::PreparedInput<'_>>,
) -> FittingState<'a> {
    let Some(max_bins) = max_bins else {
        return state;
    };
    let n_bins = max_bins as usize;
    if n_bins < 2 {
        return state;
    }

    let p = state.p;
    let n = state.n;

    for col in 0..p {
        let sorted_indices = &state.precomputed_statistics.sorted_indices[col];
        // Build a boolean keep-mask over [0, n).
        // Position 0 and position n-1 are always NaN (no split before-first /
        // after-last), so we just need to mark the interior.
        let edges = if let Some(prepared) = prepared {
            prepared.bin_edges[col].clone()
        } else {
            let feature_values: Vec<f64> = (0..n).map(|i| state.x[[i, col]]).collect();
            quantile_bin_edge_positions(sorted_indices, &feature_values, n_bins)
        };

        // Constant columns have no valid split boundaries.
        if edges.is_empty() {
            continue;
        }
        state.precomputed_statistics.bin_indices[col] = if let Some(prepared) = prepared {
            prepared.bin_indices[col].clone()
        } else {
            row_bin_indices(sorted_indices, &edges)
        };

        let mut keep = vec![false; n];
        for &e in &edges {
            keep[e] = true;
        }

        let er = &mut state.precomputed_statistics.error_reductions_split[col];
        let pl = &mut state.precomputed_statistics.update_pairs_split_left[col];
        let pr = &mut state.precomputed_statistics.update_pairs_split_right[col];
        let pp = &mut state.precomputed_statistics.error_reductions_split_pairs[col];
        for pos in 0..n {
            if !keep[pos] {
                er[pos] = f64::NAN;
                pl[pos] = (f64::NAN, f64::NAN);
                pr[pos] = (f64::NAN, f64::NAN);
                pp[pos] = (f64::NAN, f64::NAN);
            }
        }

        // Populate length-B bin-cumulative prefix sums by sampling the
        // length-n prefix sums at each bin's last position. These will be the
        // sole storage maintained by `update_statistics` for binned columns.
        let n_bins = edges.len() + 1;
        let mut binned = Vec::with_capacity(n_bins);
        {
            let ps = &state.precomputed_statistics.prefix_sums[col];
            for k in 0..n_bins {
                let bin_end_pos = if k < edges.len() { edges[k] } else { n };
                let idx = bin_end_pos - 1;
                binned.push(ps[idx]);
            }
        }
        state.precomputed_statistics.binned_prefix_sums[col] = binned;

        // Free the length-n prefix sums for binned columns — they are no
        // longer read once `bin_edges[col]` is non-empty.
        state.precomputed_statistics.prefix_sums[col] = Vec::new();

        // Publish bin-edge positions so the inner split-scan can iterate them
        // directly instead of walking every sorted position.
        state.precomputed_statistics.bin_edges[col] = edges;
    }

    state.precomputed_statistics.bin_ranks = match prepared {
        Some(prepared) => prepared.bin_ranks.clone(),
        None => rank_ordered_bin_streams(
            &state.precomputed_statistics.sorted_indices,
            &state.precomputed_statistics.bin_indices,
            n,
        ),
    };
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_row_bins_match_rank_search_and_share_immutable_storage() {
        let sorted: Vec<usize> = (0..127).map(|i| i * 29 % 127).collect();
        for edges in [vec![1, 11, 31, 90, 126], vec![63], (1..127).collect()] {
            let cached = row_bin_indices(&sorted, &edges);
            for (rank, &row) in sorted.iter().enumerate() {
                assert_eq!(cached[row] as usize, edges.partition_point(|&edge| edge <= rank));
            }
            let mut statistics = crate::grid_tensor::state::PrecomputedStatistics::new(127, 1);
            statistics.bin_indices[0] = cached;
            let cloned = statistics.clone();
            assert!(std::sync::Arc::ptr_eq(&statistics.bin_indices[0], &cloned.bin_indices[0]));
        }
        assert!(row_bin_indices(&sorted, &[]).is_empty());
        let sorted: Vec<_> = (0..=u16::MAX as usize).rev().collect();
        let edges: Vec<_> = (1..=u16::MAX as usize).collect();
        let cached = row_bin_indices(&sorted, &edges);
        assert_eq!(cached[0], u16::MAX);
        assert_eq!(cached[u16::MAX as usize], 0);
    }

    #[test]
    fn bin_edge_positions_uniform_data() {
        // Feature values 0..10 sorted -> sorted_indices = [0,1,2,...,9].
        let sorted = (0..10).collect::<Vec<_>>();
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        // 5 bins over 10 positions -> edges at positions 2, 4, 6, 8.
        let edges = quantile_bin_edge_positions(&sorted, &values, 5);
        assert_eq!(edges, vec![2, 4, 6, 8]);
    }

    #[test]
    fn bin_edge_positions_skip_duplicates() {
        // Feature values: 0, 0, 0, 1, 2, 3, 4, 5, 6, 7
        // sorted_indices preserves order.
        let sorted = (0..10).collect::<Vec<_>>();
        let values: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        // The target at rank2 snaps to the boundary after the repeated zero values.
        let edges = quantile_bin_edge_positions(&sorted, &values, 5);
        assert!(!edges.contains(&2));
        assert!(edges.contains(&3));
        assert!(edges.contains(&4));
        assert!(edges.contains(&6));
        assert!(edges.contains(&8));
    }

    #[test]
    fn every_low_cardinality_boundary_is_retained_with_uneven_counts() {
        let values = [vec![0.0; 37], vec![1.0; 53], vec![2.0; 29], vec![3.0; 11]].concat();
        let sorted: Vec<usize> = (0..values.len()).collect();
        assert_eq!(
            quantile_bin_edge_positions(&sorted, &values, 64),
            vec![37, 90, 119]
        );
        assert_eq!(
            quantile_bin_edge_positions(&sorted, &values, 4),
            vec![37, 90, 119]
        );
    }

    #[test]
    fn quantiles_inside_duplicate_runs_snap_to_valid_boundaries() {
        let values: Vec<f64> = (0..10)
            .flat_map(|value| std::iter::repeat_n(value as f64, 10))
            .collect();
        let sorted: Vec<usize> = (0..values.len()).collect();
        // Targets 25, 50, 75 land inside tied runs; lower-rank ties are deterministic.
        assert_eq!(
            quantile_bin_edge_positions(&sorted, &values, 4),
            vec![20, 50, 70]
        );
    }

    #[test]
    fn repeated_quantile_targets_share_one_distinct_boundary() {
        let values = [
            vec![0.0; 80],
            vec![1.0; 3],
            vec![2.0; 3],
            vec![3.0; 3],
            vec![4.0; 3],
            vec![5.0; 8],
        ]
        .concat();
        let sorted: Vec<usize> = (0..values.len()).collect();
        let edges = quantile_bin_edge_positions(&sorted, &values, 4);
        assert_eq!(edges, vec![80]);
        assert!(edges.len() < 4);
    }

    #[test]
    fn replicated_discrete_data_preserves_boundary_values_and_scales_counts() {
        let values: Vec<f64> = (0..10)
            .flat_map(|value| std::iter::repeat_n(value as f64, 10))
            .collect();
        let sorted: Vec<usize> = (0..values.len()).collect();
        let edges = quantile_bin_edge_positions(&sorted, &values, 4);
        let replicated: Vec<f64> = values
            .iter()
            .flat_map(|&value| std::iter::repeat_n(value, 3))
            .collect();
        let replicated_sorted: Vec<usize> = (0..replicated.len()).rev().collect();
        let mut replicated_sorted = replicated_sorted;
        replicated_sorted.sort_by(|&a, &b| replicated[a].total_cmp(&replicated[b]));
        let replicated_edges = quantile_bin_edge_positions(&replicated_sorted, &replicated, 4);
        assert_eq!(
            replicated_edges,
            edges.iter().map(|edge| edge * 3).collect::<Vec<_>>()
        );
        for (edge, replicated_edge) in edges.iter().zip(replicated_edges) {
            assert_eq!(
                values[sorted[*edge]],
                replicated[replicated_sorted[replicated_edge]]
            );
        }
    }

    #[test]
    fn candidate_boundaries_obey_bin_budget_for_repeated_value_patterns() {
        for unique in 1..30 {
            let values: Vec<f64> = (0..unique)
                .flat_map(|value| std::iter::repeat_n(value as f64, 1 + (value * 7 + unique) % 13))
                .collect();
            let sorted: Vec<usize> = (0..values.len()).collect();
            for bins in 2..35 {
                let edges = quantile_bin_edge_positions(&sorted, &values, bins);
                assert!(edges.len() < bins);
                assert!(edges.windows(2).all(|pair| pair[0] < pair[1]));
                assert!(edges.iter().all(|&edge| edge > 0
                    && edge < values.len()
                    && values[edge - 1] < values[edge]));
                if unique <= bins {
                    assert_eq!(edges.len(), unique - 1);
                }
            }
        }
    }

    #[test]
    fn bin_edge_positions_max_bins_too_large() {
        // 4 points, n_bins = 100 -> n_bins capped at 4 -> 3 edges at floor(k*4/4)
        let sorted = vec![0, 1, 2, 3];
        let values: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let edges = quantile_bin_edge_positions(&sorted, &values, 100);
        // For n=4, n_bins=4: edges at k*4/4 = 1, 2, 3.
        assert_eq!(edges, vec![1, 2, 3]);
    }
}
