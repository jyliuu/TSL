use std::sync::Arc;

use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, SeedableRng};

use super::RefinementStrategy;
use crate::grid_tensor::{
    fit::{fit_step, initialize_state_with_input},
    params::{GridTensorParamsBuilder, RefinementStrategyParamsBuilder},
    state::{AffectedRange, BranchConsensus, ConsensusAnchors, FittingState, PrefixStats},
};

// The two-pass oracle separates row evaluation from writes to the statistic caches.
pub(super) fn two_pass_refresh<'a>(
    strategy: &RefinementStrategy,
    mut state: FittingState<'a>,
) -> FittingState<'a> {
    let n = state.n;
    // L2 weights are constant, so while the consensus weight is unchanged only the
    // reference-dependent t1/t2 sums move.
    let consensus_weight = state.branch_consensus.as_ref().map_or(0.0, |c| c.weight);
    let consensus_anchors = state
        .branch_consensus
        .as_ref()
        .and_then(|c| c.anchors.clone());
    let weights_fixed = matches!(strategy, RefinementStrategy::L2Refinement { .. })
        && state.precomputed_statistics.consensus_weight == consensus_weight
        && crate::grid_tensor::state::same_consensus_support(
            &state.precomputed_statistics.consensus_anchors,
            &consensus_anchors,
        );
    let points: Vec<PrefixStats> = (0..n)
        .map(|i| state.point_statistics(i, strategy.weight(state.residuals[i])))
        .collect();
    let statistics = &mut state.precomputed_statistics;
    statistics.consensus_weight = consensus_weight;
    statistics.consensus_anchors = consensus_anchors;
    for (i, point) in points.iter().enumerate() {
        if !weights_fixed {
            statistics.curvature[i].s11 = point.s11;
            statistics.curvature[i].s22 = point.s22;
            statistics.curvature[i].s12 = point.s12;
        }
        statistics.c_t1[i] = point.t1;
        statistics.c_t2[i] = point.t2;
    }
    let mut bins_t1 = Vec::new();
    let mut bins_t2 = Vec::new();
    for col in 0..state.p {
        if statistics.bin_edges[col].is_empty() {
            let prefix = &mut statistics.prefix_sums[col];
            let mut acc = PrefixStats::ZERO;
            for (slot, &i) in prefix.iter_mut().zip(&statistics.sorted_indices[col]) {
                if !weights_fixed {
                    acc.s11 += statistics.curvature[i].s11;
                    acc.s22 += statistics.curvature[i].s22;
                    acc.s12 += statistics.curvature[i].s12;
                    slot.s11 = acc.s11;
                    slot.s22 = acc.s22;
                    slot.s12 = acc.s12;
                }
                acc.t1 += statistics.c_t1[i];
                acc.t2 += statistics.c_t2[i];
                slot.t1 = acc.t1;
                slot.t2 = acc.t2;
            }
        } else if weights_fixed {
            let binned = &mut statistics.binned_prefix_sums[col];
            bins_t1.clear();
            bins_t1.resize(binned.len(), 0.0);
            bins_t2.clear();
            bins_t2.resize(binned.len(), 0.0);
            for (i, &bin) in statistics.bin_indices[col].iter().enumerate() {
                bins_t1[bin as usize] += statistics.c_t1[i];
                bins_t2[bin as usize] += statistics.c_t2[i];
            }
            let (mut t1, mut t2) = (0.0, 0.0);
            for (slot, (b1, b2)) in binned.iter_mut().zip(bins_t1.iter().zip(&bins_t2)) {
                t1 += b1;
                t2 += b2;
                slot.t1 = t1;
                slot.t2 = t2;
            }
        } else {
            let binned = &mut statistics.binned_prefix_sums[col];
            binned.fill(PrefixStats::ZERO);
            for (i, &bin) in statistics.bin_indices[col].iter().enumerate() {
                binned[bin as usize] += PrefixStats {
                    s11: statistics.curvature[i].s11,
                    s22: statistics.curvature[i].s22,
                    s12: statistics.curvature[i].s12,
                    t1: statistics.c_t1[i],
                    t2: statistics.c_t2[i],
                };
            }
            let mut acc = PrefixStats::ZERO;
            for bin in binned.iter_mut() {
                acc += *bin;
                *bin = acc;
            }
        }
    }

    state.linear_statistics_stale = false;
    for col in 0..state.p {
        let last_interval = state.boundaries[col].len();
        RefinementStrategy::update_interval_stats_for_col_range(&mut state, col, 0, last_interval);
    }
    let ranges: Vec<AffectedRange> = (0..state.p)
        .map(|col| AffectedRange {
            col,
            point_range: (0, n - 1),
            interval_range: (0, 0),
        })
        .collect();
    strategy.refresh_error_reduction_caches_for_affected(state, &ranges)
}

fn fixture(mixed: bool) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((127, 4), |(row, col)| {
        if mixed && col == 3 {
            2.0
        } else {
            ((row * [1, 31, 53, 71][col]) % 127) as f64 / 17.0
        }
    });
    let y = Array1::from_shape_fn(127, |row| {
        x[(row, 0)].sin() * (1.0 + x[(row, 1)]) - 0.3 * x[(row, 2)]
    });
    (x, y)
}

fn reference(
    n: usize,
    anchors: Option<Arc<ConsensusAnchors>>,
    weight: f64,
    phase: usize,
) -> BranchConsensus {
    let rows: Vec<_> = anchors
        .as_ref()
        .map_or_else(|| (0..n).collect(), |a| a.rows.clone());
    BranchConsensus {
        weight,
        plus: Array1::from_iter(
            rows.iter()
                .map(|&row| 1.2 + (row + phase) as f64 / n as f64),
        ),
        minus: Array1::from_iter(
            rows.iter()
                .map(|&row| 0.4 + 0.1 * ((row + phase) as f64).cos()),
        ),
        anchors,
    }
}

fn bits(actual: f64, expected: f64) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{actual:?} != {expected:?}"
    );
}

fn slice_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        bits(actual, expected);
    }
}

pub(super) fn assert_statistics_bits(actual: &FittingState, expected: &FittingState) {
    let (a, e) = (
        &actual.precomputed_statistics,
        &expected.precomputed_statistics,
    );
    crate::grid_tensor::state::assert_curvature_bits(&a.curvature, &e.curvature);
    for (a, e) in [(&a.c_t1, &e.c_t1), (&a.c_t2, &e.c_t2)] {
        slice_bits(a, e);
    }
    for (a, e) in [
        (&a.prefix_sums, &e.prefix_sums),
        (&a.binned_prefix_sums, &e.binned_prefix_sums),
    ] {
        assert_eq!(a.len(), e.len());
        for (a, e) in a.iter().zip(e) {
            assert_eq!(a.len(), e.len());
            for (a, e) in a.iter().zip(e) {
                slice_bits(
                    &[a.s11, a.s22, a.s12, a.t1, a.t2],
                    &[e.s11, e.s22, e.s12, e.t1, e.t2],
                );
            }
        }
    }
    for (a, e) in a.interval_stats.iter().zip(&e.interval_stats) {
        assert_eq!(a.len(), e.len());
        for (a, e) in a.iter().zip(e) {
            assert_eq!(a.n, e.n);
            slice_bits(
                &[a.sum_s11, a.sum_s22, a.sum_s12, a.sum_t1, a.sum_t2],
                &[e.sum_s11, e.sum_s22, e.sum_s12, e.sum_t1, e.sum_t2],
            );
        }
    }
    for (a, e) in [
        (&a.error_reductions_split, &e.error_reductions_split),
        (&a.error_reductions_resplit, &e.error_reductions_resplit),
        (&a.error_reductions_merge, &e.error_reductions_merge),
    ] {
        assert_eq!(a.len(), e.len());
        for (a, e) in a.iter().zip(e) {
            slice_bits(a, e);
        }
    }
    for (a, e) in [
        (&a.update_pairs_split_left, &e.update_pairs_split_left),
        (&a.update_pairs_split_right, &e.update_pairs_split_right),
        (&a.update_pairs_resplit_left, &e.update_pairs_resplit_left),
        (&a.update_pairs_resplit_right, &e.update_pairs_resplit_right),
        (&a.update_pairs_merge, &e.update_pairs_merge),
        (
            &a.error_reductions_split_pairs,
            &e.error_reductions_split_pairs,
        ),
        (
            &a.error_reductions_resplit_pairs,
            &e.error_reductions_resplit_pairs,
        ),
    ] {
        assert_eq!(a.len(), e.len());
        for (a, e) in a.iter().zip(e) {
            assert_eq!(a.len(), e.len());
            for (a, e) in a.iter().zip(e) {
                slice_bits(&[a.0, a.1], &[e.0, e.1]);
            }
        }
    }
    bits(a.consensus_weight, e.consensus_weight);
    assert_eq!(a.consensus_anchors, e.consensus_anchors);
    assert_eq!(format!("{a:?}"), format!("{e:?}"));
    assert_eq!(
        actual.linear_statistics_stale,
        expected.linear_statistics_stale
    );
    assert_eq!(actual.split_scores_stale, expected.split_scores_stale);
    assert_eq!(actual.sampled_split_scoring, expected.sampled_split_scoring);
    assert_eq!(actual.boundaries, expected.boundaries);
    assert_eq!(actual.backbone_values, expected.backbone_values);
    assert_eq!(actual.tilt_values, expected.tilt_values);
    bits(actual.current_error, expected.current_error);
}

#[test]
fn streamed_refresh_matches_two_pass_for_weights_support_and_storage() {
    for (bins, mixed) in [(None, false), (Some(8), false), (Some(8), true)] {
        for huber in [false, true] {
            let (x, y) = fixture(mixed);
            let refinement = RefinementStrategyParamsBuilder::new()
                .alpha(0.01)
                .tilt_tau(0.01);
            let params = GridTensorParamsBuilder::new()
                .n_iter(12)
                .max_bins(bins)
                .refinement_strategy(
                    if huber {
                        refinement.huber()
                    } else {
                        refinement.l2()
                    }
                    .build(),
                )
                .build();
            let strategy = params.refinement_strategy_params.get_refinement_strategy();
            let split = params.split_strategy_params.get_split_strategy();
            let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
            let mut rng = StdRng::seed_from_u64(29);
            for iter in 0..5 {
                let advanced;
                (state, advanced) =
                    fit_step(state, iter, params.n_iter, &strategy, &split, &mut rng);
                if !advanced {
                    break;
                }
            }
            assert!(state.boundaries.iter().any(|column| !column.is_empty()));
            assert_eq!(
                state
                    .precomputed_statistics
                    .bin_edges
                    .iter()
                    .any(Vec::is_empty),
                bins.is_none() || mixed
            );
            assert_eq!(
                state
                    .precomputed_statistics
                    .bin_edges
                    .iter()
                    .any(|col| !col.is_empty()),
                bins.is_some()
            );
            let anchors = Arc::new(ConsensusAnchors::new(
                state.n,
                (0..state.n).step_by(7).collect(),
            ));
            let other_anchors = Arc::new(ConsensusAnchors::new(
                state.n,
                (1..state.n).step_by(7).collect(),
            ));
            let references = [
                None,
                Some(reference(state.n, None, 0.3, 0)),
                Some(reference(state.n, None, 0.3, 3)),
                Some(reference(state.n, None, 0.7, 3)),
                Some(reference(state.n, Some(anchors.clone()), 0.7, 0)),
                Some(reference(state.n, Some(anchors.clone()), 0.7, 5)),
                Some(reference(state.n, Some(other_anchors), 0.7, 5)),
                Some(reference(state.n, None, 0.7, 7)),
                None,
            ];
            for (index, reference) in references.into_iter().enumerate() {
                state.branch_consensus = reference;
                for sampled in [false, true] {
                    let mut input = state.clone();
                    input.sampled_split_scoring = sampled;
                    input.split_scores_stale = sampled;
                    // A deferred refresh rebuilds both linear caches from row inputs.
                    input.linear_statistics_stale = index % 2 == 0;
                    if input.linear_statistics_stale {
                        input.precomputed_statistics.c_t1.fill(f64::NAN);
                        input.precomputed_statistics.c_t2.fill(f64::NAN);
                    }
                    let expected = two_pass_refresh(&strategy, input.clone());
                    let actual = strategy.refresh_statistics(input);
                    assert_statistics_bits(&actual, &expected);
                }
                state = two_pass_refresh(&strategy, state);
            }
        }
    }
}

#[test]
fn streamed_refresh_preserves_curvature_and_signed_zero_bits() {
    for bins in [None, Some(8)] {
        let (x, y) = fixture(false);
        let params = GridTensorParamsBuilder::new().max_bins(bins).build();
        let strategy = params.refinement_strategy_params.get_refinement_strategy();
        let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
        for context in 0..3 {
            state.branch_consensus = match context {
                0 => None,
                1 => Some(reference(state.n, None, 0.3, 0)),
                _ => Some(reference(
                    state.n,
                    Some(Arc::new(ConsensusAnchors::new(
                        state.n,
                        (0..state.n).step_by(3).collect(),
                    ))),
                    0.3,
                    0,
                )),
            };
            state = two_pass_refresh(&strategy, state);
            for row in 0..state.n {
                state.f_plus[row] = if row % 2 == 0 { -0.0 } else { 0.0 };
                state.f_minus[row] = if row % 3 == 0 { -0.0 } else { 0.0 };
                state.r_tilde[row] = if row % 5 == 0 {
                    -0.0
                } else {
                    row as f64 - 64.0
                };
                state.residuals[row] = state.r_tilde[row];
                state.precomputed_statistics.curvature[row].s11 = -0.0;
                state.precomputed_statistics.curvature[row].s22 = 0.0;
                state.precomputed_statistics.curvature[row].s12 = -0.0;
            }
            let expected = two_pass_refresh(&strategy, state.clone());
            let actual = strategy.refresh_statistics(state.clone());
            assert_statistics_bits(&actual, &expected);
            crate::grid_tensor::state::assert_curvature_bits(
                &actual.precomputed_statistics.curvature,
                &state.precomputed_statistics.curvature,
            );
            assert!(!actual.linear_statistics_stale);
        }
    }
}
