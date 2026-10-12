//! Full-row L2 curvature deltas scattered into the binned prefix sums of every feature.

use std::iter::once;

use super::RefinementStrategy;
use crate::grid_tensor::state::{CurvatureCache, FittingState};

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
        let n = state.n;
        if let Some(streams) = state.precomputed_statistics.bin_ranks.get(col).cloned() {
            // Each feature's bins over the affected ranks form one contiguous stream. Features
            // are scattered in pairs, and every histogram receives its additions in
            // affected-row order.
            assert_eq!(streams.len(), state.p * n);
            let prefixes = &mut state.precomputed_statistics.binned_prefix_sums;
            let mut buckets = Vec::new();
            let mut feature = 0;
            while feature < state.p {
                let paired = feature + 1 < state.p;
                let first_count = prefixes[feature].len();
                let second_count = if paired {
                    prefixes[feature + 1].len()
                } else {
                    0
                };
                buckets.clear();
                buckets.resize(first_count + second_count, [0.0; 3]);
                let (first_buckets, second_buckets) = buckets.split_at_mut(first_count);
                let first_stream = &streams[feature * n + start..feature * n + end];
                let (mut first_min, mut first_max) = (usize::MAX, 0);
                let (mut second_min, mut second_max) = (usize::MAX, 0);
                if paired {
                    let second_stream =
                        &streams[(feature + 1) * n + start..(feature + 1) * n + end];
                    for ((&first_bin, &second_bin), delta) in
                        first_stream.iter().zip(second_stream).zip(&deltas)
                    {
                        let first_bin = first_bin as usize;
                        let second_bin = second_bin as usize;
                        first_min = first_min.min(first_bin);
                        first_max = first_max.max(first_bin);
                        second_min = second_min.min(second_bin);
                        second_max = second_max.max(second_bin);
                        let first_slot = &mut first_buckets[first_bin];
                        first_slot[0] += delta[0];
                        first_slot[1] += delta[1];
                        first_slot[2] += delta[2];
                        let second_slot = &mut second_buckets[second_bin];
                        second_slot[0] += delta[0];
                        second_slot[1] += delta[1];
                        second_slot[2] += delta[2];
                    }
                } else {
                    for (&bin, delta) in first_stream.iter().zip(&deltas) {
                        let bin = bin as usize;
                        // Every touched row contributes to the range, including zero deltas.
                        first_min = first_min.min(bin);
                        first_max = first_max.max(bin);
                        let slot = &mut first_buckets[bin];
                        slot[0] += delta[0];
                        slot[1] += delta[1];
                        slot[2] += delta[2];
                    }
                }
                // Bin membership is monotone in sorted rank. These bounds equal the bins
                // of the minimum and maximum affected ranks, without a separate rank scan.
                let histograms = [
                    (feature, &*first_buckets, first_min, first_max),
                    (feature + 1, &*second_buckets, second_min, second_max),
                ];
                for &(feature, buckets, first, last) in histograms.iter().take(1 + paired as usize)
                {
                    let prefix = &mut prefixes[feature];
                    let mut accumulated = [0.0; 3];
                    for bin in first..=last {
                        accumulated[0] += buckets[bin][0];
                        accumulated[1] += buckets[bin][1];
                        accumulated[2] += buckets[bin][2];
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
                feature += 1 + paired as usize;
            }
            deltas.clear();
            state.precomputed_statistics.curvature_deltas = deltas;
            state.linear_statistics_stale = true;
            return state;
        }
        let updated_points = &state.precomputed_statistics.sorted_indices[col][start..end];

        let mut buckets = Vec::new();
        let mut features = (0..state.p).filter(|&feature| feature != col);
        let mut unpaired = None;
        while let Some(first_feature) = features.next() {
            let Some(second_feature) = features.next() else {
                unpaired = Some(first_feature);
                break;
            };
            let first_bins = &state.precomputed_statistics.bin_indices[first_feature];
            let second_bins = &state.precomputed_statistics.bin_indices[second_feature];
            let first_count = state.precomputed_statistics.binned_prefix_sums[first_feature].len();
            let second_count =
                state.precomputed_statistics.binned_prefix_sums[second_feature].len();
            buckets.clear();
            buckets.resize(first_count + second_count, [0.0; 3]);
            let (first_buckets, second_buckets) = buckets.split_at_mut(first_count);
            let (mut first_min, mut first_max) = (usize::MAX, 0);
            let (mut second_min, mut second_max) = (usize::MAX, 0);
            // Each histogram receives additions in affected-row order.
            for (&row, delta) in updated_points.iter().zip(&deltas) {
                let first_bin = first_bins[row] as usize;
                let second_bin = second_bins[row] as usize;
                first_min = first_min.min(first_bin);
                first_max = first_max.max(first_bin);
                second_min = second_min.min(second_bin);
                second_max = second_max.max(second_bin);
                let first_slot = &mut first_buckets[first_bin];
                first_slot[0] += delta[0];
                first_slot[1] += delta[1];
                first_slot[2] += delta[2];
                let second_slot = &mut second_buckets[second_bin];
                second_slot[0] += delta[0];
                second_slot[1] += delta[1];
                second_slot[2] += delta[2];
            }
            for (feature, buckets, first, last) in [
                (first_feature, &*first_buckets, first_min, first_max),
                (second_feature, &*second_buckets, second_min, second_max),
            ] {
                let prefix = &mut state.precomputed_statistics.binned_prefix_sums[feature];
                let mut accumulated = [0.0; 3];
                for bin in first..=last {
                    accumulated[0] += buckets[bin][0];
                    accumulated[1] += buckets[bin][1];
                    accumulated[2] += buckets[bin][2];
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
        }
        for feature in unpaired.into_iter().chain(once(col)) {
            let bins = &state.precomputed_statistics.bin_indices[feature];
            let bin_count = state.precomputed_statistics.binned_prefix_sums[feature].len();
            buckets.clear();
            buckets.resize(bin_count, [0.0; 3]);
            let (mut first, mut last) = (usize::MAX, 0);
            for (&row, delta) in updated_points.iter().zip(&deltas) {
                let bin = bins[row] as usize;
                // Every touched row contributes to the range, including zero deltas.
                first = first.min(bin);
                last = last.max(bin);
                let slot = &mut buckets[bin];
                slot[0] += delta[0];
                slot[1] += delta[1];
                slot[2] += delta[2];
            }

            // Bin membership is monotone in sorted rank. These bounds equal the bins
            // of the minimum and maximum affected ranks, without a separate rank scan.
            let prefix = &mut state.precomputed_statistics.binned_prefix_sums[feature];
            let mut accumulated = [0.0; 3];
            for bin in first..=last {
                accumulated[0] += buckets[bin][0];
                accumulated[1] += buckets[bin][1];
                accumulated[2] += buckets[bin][2];
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
        deltas.clear();
        state.precomputed_statistics.curvature_deltas = deltas;
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
                if matches!(self, Self::L2Refinement { .. })
                    && reference.anchors.is_none()
                    && [
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
                {
                    let rho = reference.weight;
                    let curvature = &mut state.precomputed_statistics.curvature[..n];
                    for &row in updated_points {
                        if row >= n {
                            // The first failing row uses the residual array's indexing behavior.
                            let _ = state.residuals[row];
                            unreachable!();
                        }
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
            let point = state.point_curvature_statistics(row, self.weight(state.residuals[row]));
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
