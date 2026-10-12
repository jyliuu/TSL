use std::sync::Arc;

use ndarray::{s, Array1, Array2, ArrayView1};
use rand::{rngs::StdRng, SeedableRng};

use super::{
    fit_members_with_candidate_refresh, reuse_full_row_reference,
    tests::{assert_members_equal, fit_members_single_action_reference},
    Member,
};
use crate::grid_tensor::{
    fit::initialize_state_with_input,
    params::{
        GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
    },
    reducer::CandidateRefresh,
    state::{BranchConsensus, ConsensusAnchors, FittingState},
    GridTensorParams,
};

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

fn assert_statistics_bits(actual: &FittingState, expected: &FittingState) {
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

fn array_bits(actual: &Array1<f64>, expected: &Array1<f64>) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        bits(actual, expected);
    }
}

fn reference_bits(actual: &BranchConsensus, expected: &BranchConsensus) {
    bits(actual.weight, expected.weight);
    array_bits(&actual.plus, &expected.plus);
    array_bits(&actual.minus, &expected.minus);
    assert_eq!(actual.anchors, expected.anchors);
}

fn fresh_reference(
    current: &FittingState<'_>,
    total_plus: &ArrayView1<'_, f64>,
    total_minus: &ArrayView1<'_, f64>,
    others: f64,
    consensus: f64,
) -> BranchConsensus {
    BranchConsensus {
        weight: consensus,
        plus: (total_plus - &current.f_plus) / others,
        minus: (total_minus - &current.f_minus) / others,
        anchors: None,
    }
}

fn install_reference(
    current: &mut FittingState<'_>,
    totals: &[Array1<f64>; 2],
    others: f64,
    consensus: f64,
) -> bool {
    let (plus, minus) = (totals[0].view(), totals[1].view());
    if reuse_full_row_reference(current, &plus, &minus, others, consensus) {
        true
    } else {
        current.branch_consensus = Some(fresh_reference(current, &plus, &minus, others, consensus));
        false
    }
}

fn fixture(mixed: bool) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((71, 3), |(row, col)| {
        if mixed && col == 2 {
            1.0
        } else {
            ((row * [1, 29, 43][col]) % 71) as f64 / 11.0 - 3.0
        }
    });
    let y = Array1::from_shape_fn(71, |row| {
        x[(row, 0)].sin() * (1.0 + x[(row, 1)]) - 0.4 * x[(row, 2)]
    });
    (x, y)
}

fn params(huber: bool, bins: Option<u16>) -> GridTensorParams {
    let refinement = RefinementStrategyParamsBuilder::new()
        .alpha(0.01)
        .tilt_tau(0.01);
    GridTensorParamsBuilder::new()
        .n_iter(4)
        .max_bins(bins)
        .refinement_strategy(
            if huber {
                refinement.huber()
            } else {
                refinement.l2()
            }
            .build(),
        )
        .split_strategy(
            SplitStrategyParamsBuilder::new()
                .random_split()
                .split_try(7)
                .min_interval_samples(2)
                .complexity_penalty(0.0)
                .build(),
        )
        .build()
}

fn with_reference(mut state: FittingState<'_>, weight: f64) -> FittingState<'_> {
    state.branch_consensus = Some(BranchConsensus {
        weight,
        plus: Array1::from_elem(state.n, 7.0),
        minus: Array1::from_elem(state.n, 9.0),
        anchors: None,
    });
    state
}

fn totals(state: &FittingState<'_>) -> [Array1<f64>; 2] {
    [&state.f_plus * 4.0 + 0.125, &state.f_minus * 4.0 + 0.25]
}

#[test]
fn reference_reuse_reuses_targets_with_exact_subtraction_and_division() {
    let (x, y) = fixture(false);
    let initial = initialize_state_with_input(x.view(), y.view(), &params(false, Some(8)), None);
    for others in [3.0_f64, 49.0] {
        assert_ne!((5.0 / others).to_bits(), (5.0 * (1.0 / others)).to_bits());
    }
    for others in [1.0, 3.0, 49.0] {
        for weight in [0.0, -0.0, 0.3, 0.7] {
            let mut state = with_reference(initial.clone(), 0.13);
            let cases = [
                (0.0, -0.0),
                (-0.0, 0.0),
                (f64::from_bits(1), 0.0),
                (f64::from_bits(3), f64::from_bits(1)),
                (1e150, 1e150 * (1.0 - f64::EPSILON)),
                (1.0, 1.0 + f64::EPSILON),
                (1e-150, -1e-150),
                (f64::MAX / 8.0, -f64::MAX / 8.0),
                (-1e120, 1e120),
                (6.0, 1.0),
                (1e16, 1.0),
                (2.25, 0.1),
                (0.7, 0.1),
            ];
            let mut totals = [Array1::zeros(state.n), Array1::zeros(state.n)];
            for row in 0..state.n {
                let (total, current) = cases[row % cases.len()];
                totals[0][row] = total;
                totals[1][row] = current;
                state.f_plus[row] = current;
                state.f_minus[row] = total;
            }
            let inputs = [
                totals[0].clone(),
                totals[1].clone(),
                state.f_plus.clone(),
                state.f_minus.clone(),
            ];
            let expected =
                fresh_reference(&state, &totals[0].view(), &totals[1].view(), others, weight);
            let reference = state.branch_consensus.as_mut().unwrap();
            reference.plus.fill(f64::NAN);
            reference.minus.fill(f64::NAN);
            let pointers = (reference.plus.as_ptr(), reference.minus.as_ptr());
            let metadata = format!("{:?}", state.precomputed_statistics);
            assert!(install_reference(&mut state, &totals, others, weight));
            let actual = state.branch_consensus.as_ref().unwrap();
            reference_bits(actual, &expected);
            assert_eq!((actual.plus.as_ptr(), actual.minus.as_ptr()), pointers);
            assert_eq!(format!("{:?}", state.precomputed_statistics), metadata);
            for (actual, expected) in [&totals[0], &totals[1], &state.f_plus, &state.f_minus]
                .into_iter()
                .zip(&inputs)
            {
                array_bits(actual, expected);
            }
        }
    }
}

fn selected_array<'s>(
    state: &'s mut FittingState<'_>,
    totals: &'s mut [Array1<f64>; 2],
    index: usize,
) -> &'s mut Array1<f64> {
    match index {
        0 | 1 => &mut totals[index],
        2 => &mut state.f_plus,
        3 => &mut state.f_minus,
        4 => &mut state.branch_consensus.as_mut().unwrap().plus,
        5 => &mut state.branch_consensus.as_mut().unwrap().minus,
        _ => unreachable!(),
    }
}

fn variant(values: &Array1<f64>, kind: usize) -> Array1<f64> {
    match kind {
        0 => Array1::from_shape_fn(2 * values.len(), |row| {
            if row % 2 == 0 {
                values[row / 2]
            } else {
                123.0
            }
        })
        .slice_move(s![..;2]),
        1 => Array1::from_iter(values.iter().rev().copied()).slice_move(s![..;-1]),
        2 => values.clone().slice_move(s![..-1]),
        3 => Array1::from_iter(values.iter().copied().chain([123.0])),
        4 => values.clone().slice_move(s![..1]),
        _ => Array1::zeros(0),
    }
}

fn assert_declines_without_mutation(state: &mut FittingState<'_>, totals: &[Array1<f64>; 2]) {
    let before = state.clone();
    let reference = state
        .branch_consensus
        .as_ref()
        .map(|r| (r.plus.as_ptr(), r.minus.as_ptr()));
    let total_copies = [totals[0].clone(), totals[1].clone()];
    assert!(!reuse_full_row_reference(
        state,
        &totals[0].view(),
        &totals[1].view(),
        3.0,
        0.7
    ));
    assert_eq!(format!("{state:?}"), format!("{before:?}"));
    assert_eq!(
        state
            .branch_consensus
            .as_ref()
            .map(|r| (r.plus.as_ptr(), r.minus.as_ptr())),
        reference
    );
    if let (Some(actual), Some(expected)) = (&state.branch_consensus, &before.branch_consensus) {
        reference_bits(actual, expected);
    }
    array_bits(&totals[0], &total_copies[0]);
    array_bits(&totals[1], &total_copies[1]);
}

fn assert_fallback(mut state: FittingState<'_>, totals: &[Array1<f64>; 2], success: bool) {
    assert_declines_without_mutation(&mut state, totals);
    let expected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fresh_reference(&state, &totals[0].view(), &totals[1].view(), 3.0, 0.7)
    }));
    let actual = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert!(!install_reference(&mut state, totals, 3.0, 0.7));
    }));
    assert_eq!(expected.is_ok(), success);
    assert_eq!(actual.is_ok(), success);
    if let Ok(expected) = expected {
        reference_bits(state.branch_consensus.as_ref().unwrap(), &expected);
    }
}

#[test]
fn reference_reuse_guards_preserve_layout_shape_and_broadcast_fallbacks() {
    let (x, y) = fixture(false);
    let initial = with_reference(
        initialize_state_with_input(x.view(), y.view(), &params(false, Some(8)), None),
        0.13,
    );
    for index in 0..6 {
        for kind in 0..6 {
            let mut state = initial.clone();
            let mut totals = totals(&state);
            let values = selected_array(&mut state, &mut totals, index);
            *values = variant(values, kind);
            assert_fallback(state, &totals, index >= 4 || kind <= 1 || kind == 4);
        }
    }
    for anchored in [false, true] {
        let mut state = initial.clone();
        let totals = totals(&state);
        if anchored {
            state.branch_consensus.as_mut().unwrap().anchors = Some(Arc::new(
                ConsensusAnchors::new(state.n, (0..state.n).step_by(3).collect()),
            ));
        } else {
            state.branch_consensus = None;
        }
        assert_fallback(state, &totals, true);
    }
}

#[test]
fn reference_reuse_refresh_preserves_all_caches_and_weight_metadata() {
    for huber in [false, true] {
        for (bins, mixed) in [(None, false), (Some(8), false), (Some(8), true)] {
            let (x, y) = fixture(mixed);
            let params = params(huber, bins);
            let refinement = params.refinement_strategy_params.get_refinement_strategy();
            let initial = initialize_state_with_input(x.view(), y.view(), &params, None);
            for (old_weight, weight) in [(0.3, 0.3), (0.3, 0.7), (0.3, 0.0), (0.0, -0.0)] {
                let mut actual =
                    refinement.refresh_statistics(with_reference(initial.clone(), old_weight));
                actual.linear_statistics_stale = true;
                actual.precomputed_statistics.c_t1.fill(f64::NAN);
                actual.precomputed_statistics.c_t2.fill(f64::NAN);
                let totals = totals(&actual);
                let mut expected = actual.clone();
                expected.branch_consensus = Some(fresh_reference(
                    &expected,
                    &totals[0].view(),
                    &totals[1].view(),
                    3.0,
                    weight,
                ));
                let cached = actual.clone();
                assert!(install_reference(&mut actual, &totals, 3.0, weight));
                assert_statistics_bits(&actual, &cached);
                bits(actual.precomputed_statistics.consensus_weight, old_weight);
                reference_bits(
                    actual.branch_consensus.as_ref().unwrap(),
                    expected.branch_consensus.as_ref().unwrap(),
                );
                let expected = refinement.refresh_statistics(expected);
                let actual = refinement.refresh_statistics(actual);
                assert_statistics_bits(&actual, &expected);
                bits(actual.precomputed_statistics.consensus_weight, weight);
            }
        }
    }
}

fn mixed_members<'a>(initial: &FittingState<'a>, params: &GridTensorParams) -> Vec<Member<'a>> {
    (0..4)
        .map(|tree_id| {
            let mut state = with_reference(initial.clone(), 0.13);
            if tree_id == 1 {
                state.terminated = true;
                state.iteration = 77;
            }
            if tree_id == 2 {
                state.loop_state.fineness = params.n_iter;
                state.iteration = 77;
            }
            #[cfg(feature = "evo-logging")]
            {
                state.logging_state =
                    Some(crate::grid_tensor::logging_helpers::LoggingState::new());
            }
            Member {
                tree_id,
                rng: StdRng::seed_from_u64(17 + tree_id as u64),
                state: Some(state),
                next_iteration: 0,
            }
        })
        .collect()
}

#[test]
fn reference_reuse_frozen_rounds_include_inactive_members_in_order() {
    let (x, y) = fixture(false);
    for huber in [false, true] {
        for bins in [None, Some(8)] {
            let params = params(huber, bins);
            let initial = initialize_state_with_input(x.view(), y.view(), &params, None);
            let mut expected = mixed_members(&initial, &params);
            let mut actual = mixed_members(&initial, &params);
            let pointers: Vec<_> = actual
                .iter()
                .map(|m| {
                    let r = m.state.as_ref().unwrap().branch_consensus.as_ref().unwrap();
                    (r.plus.as_ptr(), r.minus.as_ptr())
                })
                .collect();
            fit_members_single_action_reference(
                &mut expected,
                &params,
                0.3,
                None,
                CandidateRefresh::Deferred,
            );
            fit_members_with_candidate_refresh(
                &mut actual,
                &params,
                0.3,
                1,
                None,
                None,
                CandidateRefresh::Deferred,
            );
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                let state = actual.state.as_ref().unwrap();
                let reference = state.branch_consensus.as_ref().unwrap();
                reference_bits(
                    reference,
                    expected
                        .state
                        .as_ref()
                        .unwrap()
                        .branch_consensus
                        .as_ref()
                        .unwrap(),
                );
                assert_eq!(
                    (reference.plus.as_ptr(), reference.minus.as_ptr()),
                    pointers[index]
                );
                if index == 1 || index == 2 {
                    assert_eq!(actual.next_iteration, 0);
                    assert_eq!(state.iteration, 77);
                    array_bits(&state.f_plus, &initial.f_plus);
                    array_bits(&state.f_minus, &initial.f_minus);
                    bits(reference.weight, 0.13);
                } else {
                    assert!(actual.next_iteration > 0);
                    bits(reference.weight, 0.3);
                }
            }
            assert_members_equal(&initial, &params, expected, actual);
        }
    }
}
