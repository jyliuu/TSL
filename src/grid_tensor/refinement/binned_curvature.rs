//! Full-row L2 curvature deltas scattered into the binned prefix sums of every feature.

use std::iter::once;

use super::RefinementStrategy;
use crate::grid_tensor::state::{CurvatureCache, FittingState, PrefixStats};

impl RefinementStrategy {
    /// Scatter curvature deltas while locating the touched bins in the same row traversal.
    pub(crate) fn update_binned_curvature_statistics<'a>(
        &self,
        mut state: FittingState<'a>,
        col: usize,
        start: usize,
        end: usize,
    ) -> FittingState<'a> {
        state.require_current_statistics();
        assert!(matches!(self, Self::L2Refinement { .. }));
        assert!(state
            .branch_consensus
            .as_ref()
            .is_some_and(|reference| reference.anchors.is_none()));
        assert!(state
            .precomputed_statistics
            .bin_edges
            .iter()
            .all(|edges| !edges.is_empty()));
        let updated_points = &state.precomputed_statistics.sorted_indices[col][start..end];
        if updated_points.is_empty() {
            return state;
        }

        let mut deltas = self.collect_binned_curvature_deltas(&mut state, col, start, end);
        let (n, p) = (state.n, state.p);
        let statistics = &mut state.precomputed_statistics;
        let mut buckets = Vec::new();
        if let Some(streams) = statistics.bin_ranks.get(col) {
            // Each feature's bins over the affected ranks form one contiguous stream. Features
            // are scattered in pairs.
            assert_eq!(streams.len(), p * n);
            let stream = |feature: usize| &streams[feature * n + start..feature * n + end];
            let prefixes = &mut statistics.binned_prefix_sums;
            let mut feature = 0;
            while feature + 1 < p {
                let bins = stream(feature)
                    .iter()
                    .zip(stream(feature + 1))
                    .map(|(&first, &second)| (first as usize, second as usize));
                scatter_pair(
                    prefixes,
                    (feature, feature + 1),
                    bins,
                    &deltas,
                    &mut buckets,
                );
                feature += 2;
            }
            if feature < p {
                let bins = stream(feature).iter().map(|&bin| bin as usize);
                scatter(&mut prefixes[feature], bins, &deltas, &mut buckets);
            }
        } else {
            let updated_points = &statistics.sorted_indices[col][start..end];
            let mut features = (0..p).filter(|&feature| feature != col);
            let mut unpaired = None;
            while let Some(first) = features.next() {
                let Some(second) = features.next() else {
                    unpaired = Some(first);
                    break;
                };
                let (first_bins, second_bins) = (
                    &statistics.bin_indices[first],
                    &statistics.bin_indices[second],
                );
                let bins = updated_points
                    .iter()
                    .map(|&row| (first_bins[row] as usize, second_bins[row] as usize));
                scatter_pair(
                    &mut statistics.binned_prefix_sums,
                    (first, second),
                    bins,
                    &deltas,
                    &mut buckets,
                );
            }
            for feature in unpaired.into_iter().chain(once(col)) {
                let row_bins = &statistics.bin_indices[feature];
                let bins = updated_points.iter().map(|&row| row_bins[row] as usize);
                scatter(
                    &mut statistics.binned_prefix_sums[feature],
                    bins,
                    &deltas,
                    &mut buckets,
                );
            }
        }
        deltas.clear();
        statistics.curvature_deltas = deltas;
        state.linear_statistics_stale = true;
        state
    }

    /// Collect full-row L2 curvature deltas in affected-row order.
    fn collect_binned_curvature_deltas(
        &self,
        state: &mut FittingState<'_>,
        col: usize,
        start: usize,
        end: usize,
    ) -> Vec<[f64; 3]> {
        let updated_points = &state.precomputed_statistics.sorted_indices[col][start..end];
        // The scratch keeps its capacity across actions; it is returned empty afterwards.
        let mut deltas = std::mem::take(&mut state.precomputed_statistics.curvature_deltas);
        deltas.clear();
        deltas.reserve(updated_points.len());
        let n = state.n;
        if let (Some(f_plus), Some(f_minus), Some(r_tilde), Some(residuals), Some(reference)) = (
            state.f_plus.as_slice(),
            state.f_minus.as_slice(),
            state.r_tilde.as_slice(),
            state.residuals.as_slice(),
            state.branch_consensus.as_ref(),
        ) {
            if let (Some(target_plus), Some(target_minus)) =
                (reference.plus.as_slice(), reference.minus.as_slice())
            {
                if [
                    f_plus.len(),
                    f_minus.len(),
                    r_tilde.len(),
                    residuals.len(),
                    target_plus.len(),
                    target_minus.len(),
                    state.precomputed_statistics.curvature.len(),
                ]
                .into_iter()
                .all(|length| length == n)
                    && state.sample_weight.is_none()
                {
                    let rho = reference.weight;
                    let curvature = &mut state.precomputed_statistics.curvature[..n];
                    for &row in updated_points {
                        let fp = f_plus[row];
                        let fm = f_minus[row];
                        let mut s11 = 1.0 * fp * fp;
                        let mut s22 = 1.0 * fm * fm;
                        let s12 = 1.0 * fp * fm;
                        s11 += rho * fp * fp;
                        s22 += rho * fm * fm;
                        let cache = &mut curvature[row];
                        deltas.push([s11 - cache.s11, s22 - cache.s22, s12 - cache.s12]);
                        *cache = CurvatureCache { s11, s22, s12 };
                    }
                    return deltas;
                }
            }
        }
        for &row in updated_points {
            let point = state.point_curvature_statistics(row, self.row_weight(state, row));
            deltas.push([
                point.s11 - state.precomputed_statistics.curvature[row].s11,
                point.s22 - state.precomputed_statistics.curvature[row].s22,
                point.s12 - state.precomputed_statistics.curvature[row].s12,
            ]);
            // Row statistics depend on predictions and the reference, not another row's cache.
            state.precomputed_statistics.curvature[row].s11 = point.s11;
            state.precomputed_statistics.curvature[row].s22 = point.s22;
            state.precomputed_statistics.curvature[row].s12 = point.s12;
        }
        deltas
    }
}

/// Scatter the deltas of two features in one pass over the affected rows, then sweep each
/// feature's buckets into its prefix sums. Each histogram receives its additions in
/// affected-row order.
fn scatter_pair(
    prefixes: &mut [Vec<PrefixStats>],
    (first, second): (usize, usize),
    bins: impl Iterator<Item = (usize, usize)>,
    deltas: &[[f64; 3]],
    buckets: &mut Vec<[f64; 3]>,
) {
    let first_count = prefixes[first].len();
    buckets.clear();
    buckets.resize(first_count + prefixes[second].len(), [0.0; 3]);
    let (first_buckets, second_buckets) = buckets.split_at_mut(first_count);
    let (mut first_range, mut second_range) = ((usize::MAX, 0), (usize::MAX, 0));
    for ((first_bin, second_bin), delta) in bins.zip(deltas) {
        first_range = (first_range.0.min(first_bin), first_range.1.max(first_bin));
        second_range = (
            second_range.0.min(second_bin),
            second_range.1.max(second_bin),
        );
        add_delta(&mut first_buckets[first_bin], delta);
        add_delta(&mut second_buckets[second_bin], delta);
    }
    sweep(&mut prefixes[first], first_buckets, first_range);
    sweep(&mut prefixes[second], second_buckets, second_range);
}

/// Scatter one feature's deltas in affected-row order, then sweep them into `prefix`.
fn scatter(
    prefix: &mut [PrefixStats],
    bins: impl Iterator<Item = usize>,
    deltas: &[[f64; 3]],
    buckets: &mut Vec<[f64; 3]>,
) {
    buckets.clear();
    buckets.resize(prefix.len(), [0.0; 3]);
    let mut range = (usize::MAX, 0);
    for (bin, delta) in bins.zip(deltas) {
        // Every touched row contributes to the range, including zero deltas.
        range = (range.0.min(bin), range.1.max(bin));
        add_delta(&mut buckets[bin], delta);
    }
    sweep(prefix, buckets, range);
}

#[inline]
fn add_delta(bucket: &mut [f64; 3], delta: &[f64; 3]) {
    bucket[0] += delta[0];
    bucket[1] += delta[1];
    bucket[2] += delta[2];
}

/// Carry bucketed curvature deltas into bin-cumulative `prefix`: running totals over the
/// touched bins `first..=last` and the full total past `last`. Bin membership is monotone
/// in sorted rank, so the touched bins are those of the minimum and maximum affected ranks.
fn sweep(prefix: &mut [PrefixStats], buckets: &[[f64; 3]], (first, last): (usize, usize)) {
    let mut accumulated = [0.0; 3];
    for bin in first..=last {
        add_delta(&mut accumulated, &buckets[bin]);
        prefix[bin].s11 += accumulated[0];
        prefix[bin].s22 += accumulated[1];
        prefix[bin].s12 += accumulated[2];
    }
    for slot in &mut prefix[(last + 1)..] {
        slot.s11 += accumulated[0];
        slot.s22 += accumulated[1];
        slot.s12 += accumulated[2];
    }
}
