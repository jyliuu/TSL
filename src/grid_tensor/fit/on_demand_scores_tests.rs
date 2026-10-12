use std::sync::Arc;

use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, seq::index::sample, RngCore, SeedableRng};

use super::{finish_fit, fit_step, initialize_state_with_input, max_iterations};
use crate::grid_tensor::{
    action::FittingAction,
    params::{
        GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
    },
    refinement::RefinementStrategy,
    state::{AffectedRange, BranchConsensus, ConsensusAnchors, FittingState, PrefixStats},
    GridTensorParams,
};

fn fixture(only_one_variable: bool, positive: bool) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((96, 5), |(i, j)| {
        if j == 4 || (only_one_variable && j > 0) {
            1.0
        } else if j == 3 {
            ((i / 8) % 3) as f64
        } else {
            ((i * (j * 2 + 3) + j) % 61) as f64 / 11.0
        }
    });
    let y = Array1::from_shape_fn(x.nrows(), |i| {
        let value = (x[(i, 0)] * 1.3).sin() * (1.0 + x[(i, 1)]) - 0.2 * x[(i, 2)];
        if positive {
            value.abs() + 0.5
        } else {
            value
        }
    });
    (x, y)
}

fn parameters(
    bins: Option<u16>,
    huber: bool,
    split_try: usize,
    colsample: f64,
    margin: usize,
    merges: bool,
) -> GridTensorParams {
    let refinement = RefinementStrategyParamsBuilder::new()
        .alpha(0.01)
        .tilt_tau(0.01);
    let refinement = if huber {
        refinement.huber()
    } else {
        refinement.l2()
    }
    .build();
    GridTensorParamsBuilder::new()
        .n_iter(12)
        .max_bins(bins)
        .refinement_strategy(refinement)
        .split_strategy(
            SplitStrategyParamsBuilder::new()
                .random_split()
                .split_try(split_try)
                .colsample_bytree(colsample)
                .min_interval_samples(margin)
                .complexity_penalty(if merges { 0.5 } else { 0.0 })
                .build(),
        )
        .build()
}

fn bits(actual: f64, expected: f64) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{actual} != {expected}"
    );
}

fn slice_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        bits(actual, expected);
    }
}

fn matrix_bits(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        slice_bits(actual, expected);
    }
}

fn pair_bits(actual: &[Vec<(f64, f64)>], expected: &[Vec<(f64, f64)>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            bits(actual.0, expected.0);
            bits(actual.1, expected.1);
        }
    }
}

fn prefix_bits(actual: &[Vec<PrefixStats>], expected: &[Vec<PrefixStats>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            slice_bits(
                &[a.s11, a.s22, a.s12, a.t1, a.t2],
                &[e.s11, e.s22, e.s12, e.t1, e.t2],
            );
        }
    }
}

fn assert_states_identical(actual: &FittingState, expected: &FittingState) {
    assert_eq!(actual.boundaries, expected.boundaries);
    matrix_bits(&actual.backbone_values, &expected.backbone_values);
    matrix_bits(&actual.tilt_values, &expected.tilt_values);
    bits(actual.lambda_plus, expected.lambda_plus);
    bits(actual.lambda_minus, expected.lambda_minus);
    bits(actual.current_error, expected.current_error);
    for (actual, expected) in [
        (&actual.f_plus, &expected.f_plus),
        (&actual.f_minus, &expected.f_minus),
        (&actual.f, &expected.f),
        (&actual.residuals, &expected.residuals),
        (&actual.r_tilde, &expected.r_tilde),
    ] {
        slice_bits(actual.as_slice().unwrap(), expected.as_slice().unwrap());
    }
    assert_eq!(actual.loop_state.fineness, expected.loop_state.fineness);
    assert_eq!(
        actual.loop_state.split_count,
        expected.loop_state.split_count
    );
    assert_eq!(
        actual.loop_state.resplit_count,
        expected.loop_state.resplit_count
    );
    assert_eq!(
        actual.loop_state.merge_count,
        expected.loop_state.merge_count
    );
    assert_eq!(
        actual.split_strategy_state.allowed_intervals,
        expected.split_strategy_state.allowed_intervals
    );
    assert_eq!(
        actual.split_strategy_state.total_positions,
        expected.split_strategy_state.total_positions
    );
    let (a, e) = (
        &actual.precomputed_statistics,
        &expected.precomputed_statistics,
    );
    crate::grid_tensor::state::assert_curvature_bits(&a.curvature, &e.curvature);
    for (a, e) in [(&a.c_t1, &e.c_t1), (&a.c_t2, &e.c_t2)] {
        slice_bits(a, e);
    }
    prefix_bits(&a.prefix_sums, &e.prefix_sums);
    prefix_bits(&a.binned_prefix_sums, &e.binned_prefix_sums);
    matrix_bits(&a.error_reductions_resplit, &e.error_reductions_resplit);
    matrix_bits(&a.error_reductions_merge, &e.error_reductions_merge);
    pair_bits(&a.update_pairs_resplit_left, &e.update_pairs_resplit_left);
    pair_bits(&a.update_pairs_resplit_right, &e.update_pairs_resplit_right);
    pair_bits(&a.update_pairs_merge, &e.update_pairs_merge);
    for (a, e) in a
        .error_reductions_split
        .iter()
        .zip(&e.error_reductions_split)
    {
        assert_eq!(a.len(), e.len());
        for (a, e) in a.iter().zip(e) {
            assert_eq!(a.is_nan(), e.is_nan());
        }
    }
}

fn changing_reference(state: &FittingState, kind: usize, round: usize) -> Option<BranchConsensus> {
    if kind == 0 {
        return None;
    }
    let anchors = if kind == 2 {
        state
            .branch_consensus
            .as_ref()
            .and_then(|reference| reference.anchors.clone())
            .or_else(|| {
                Some(Arc::new(ConsensusAnchors::new(
                    state.n,
                    (0..state.n).step_by(5).collect(),
                )))
            })
    } else {
        None
    };
    let rows: Vec<usize> = anchors
        .as_ref()
        .map_or_else(|| (0..state.n).collect(), |anchors| anchors.rows.clone());
    Some(BranchConsensus {
        weight: 0.3 * state.n as f64 / rows.len() as f64,
        plus: Array1::from_iter(
            rows.iter()
                .map(|row| 1.3 + 0.2 * (*row as f64 + round as f64).cos()),
        ),
        minus: Array1::from_iter(
            rows.iter()
                .map(|row| 0.8 + 0.1 * (*row as f64 - round as f64).sin()),
        ),
        anchors,
    })
}

fn refresh_reference<'a>(
    mut state: FittingState<'a>,
    strategy: &RefinementStrategy,
    reference: Option<BranchConsensus>,
) -> FittingState<'a> {
    if let Some(reference) = reference {
        state = strategy.refresh_consensus_reference(state, reference);
    }
    state
}

fn reference_random_choice(
    state: &FittingState,
    split_try: usize,
    colsample: f64,
    rng: &mut StdRng,
) -> Option<(usize, usize, usize, f64)> {
    let eligible: Vec<usize> = state
        .split_strategy_state
        .total_positions
        .iter()
        .enumerate()
        .filter_map(|(col, &count)| (count > 0).then_some(col))
        .collect();
    let requested = (colsample * state.p as f64) as usize;
    let columns = if eligible.len() < requested {
        sample(rng, state.p, requested).into_vec()
    } else {
        eligible
    };
    let mut best: Option<(usize, usize, usize, f64)> = None;
    for col in columns {
        let count = state.split_strategy_state.total_positions[col];
        let positions = if count < split_try {
            (0..count).collect()
        } else {
            sample(rng, count, split_try).into_vec()
        };
        for abstract_position in positions {
            let mut remaining = abstract_position;
            let mut concrete = None;
            for (allowed, interval) in state.split_strategy_state.allowed_intervals[col]
                .iter()
                .enumerate()
            {
                if remaining < interval.length {
                    concrete = Some((interval.start + remaining, allowed));
                    break;
                }
                remaining -= interval.length;
            }
            let (position, allowed) = concrete.unwrap();
            let gain = state.precomputed_statistics.error_reductions_split[col][position];
            if !gain.is_nan() && best.is_none_or(|entry| gain > entry.3) {
                best = Some((col, position, allowed, gain));
            }
        }
    }
    best.filter(|choice| {
        choice.3.is_finite() && choice.3 > 1e-12 * state.current_error.abs().max(1.0)
    })
}

#[test]
fn on_demand_scores_random_sampling_matches_the_independent_reference() {
    let (x, y) = fixture(false, false);
    for pattern in 0..5 {
        for colsample in [0.0, 0.5, 1.0] {
            for split_try in [0, 1, 10, usize::MAX] {
                let params = parameters(None, false, split_try, colsample, 1, false);
                let split = params.split_strategy_params.get_split_strategy();
                let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
                for col in 0..state.p {
                    for position in 0..state.n {
                        let allowed = position > 0
                            && position + 1 < state.n
                            && position % 3 != 0
                            && (pattern != 1 || col == 0);
                        let gain = if !allowed {
                            f64::NAN
                        } else if pattern == 2 {
                            1.0
                        } else if pattern == 3 && position % 5 == 0 {
                            f64::INFINITY
                        } else if pattern == 4 {
                            f64::NEG_INFINITY
                        } else {
                            (col * state.n + position + 1) as f64
                        };
                        state.precomputed_statistics.error_reductions_split[col][position] = gain;
                        state.precomputed_statistics.update_pairs_split_left[col][position] =
                            (col as f64, position as f64);
                        state.precomputed_statistics.update_pairs_split_right[col][position] =
                            (position as f64, col as f64);
                    }
                }
                state = split.initialize(state);
                for seed in 0..8 {
                    let mut reference_rng = StdRng::seed_from_u64(seed);
                    let mut actual_rng = reference_rng.clone();
                    let expected =
                        reference_random_choice(&state, split_try, colsample, &mut reference_rng);
                    let actual = split.propose_next_action(&state, &mut actual_rng);
                    assert_eq!(actual_rng.next_u64(), reference_rng.next_u64());
                    match (actual, expected) {
                        (None, None) => {}
                        (
                            Some(FittingAction::ApplySplit { split: actual }),
                            Some((col, index, allowed, gain)),
                        ) => {
                            assert_eq!(
                                (actual.col, actual.index, actual.allowed_interval_idx),
                                (col, index, allowed)
                            );
                            bits(actual.error_reduction, gain);
                            assert_eq!(actual.update_left, (col as f64, index as f64));
                            assert_eq!(actual.update_right, (index as f64, col as f64));
                        }
                        (actual, expected) => {
                            panic!("unexpected choice: {actual:?} vs {expected:?}")
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn on_demand_scores_preserves_random_trajectories_masks_and_rng() {
    let mut case = 0;
    let mut sampled_steps = 0;
    for bins in [None, Some(1), Some(8), Some(64)] {
        for huber in [false, true] {
            for positive in [false, true] {
                for reference_kind in 0..3 {
                    let (x, y) = fixture(case % 3 == 0, positive);
                    let params = parameters(
                        bins,
                        huber,
                        [0, 1, 10, usize::MAX][case % 4],
                        [0.0, 0.5, 1.0][case % 3],
                        if case % 2 == 0 { 1 } else { 3 },
                        case % 2 == 1,
                    );
                    let strategy = params.refinement_strategy_params.get_refinement_strategy();
                    let split = params.split_strategy_params.get_split_strategy();
                    let mut eager = initialize_state_with_input(x.view(), y.view(), &params, None);
                    let mut lazy = eager.clone();
                    lazy.sampled_split_scoring = true;
                    let mut eager_rng = StdRng::seed_from_u64(17 + case as u64);
                    let mut lazy_rng = eager_rng.clone();
                    for iteration in 0..max_iterations(params.n_iter) {
                        let reference = changing_reference(&eager, reference_kind, iteration);
                        eager = refresh_reference(eager, &strategy, reference.clone());
                        lazy = refresh_reference(lazy, &strategy, reference);
                        let expected_action = split.propose_next_action_with_split_budget(
                            &eager,
                            &mut eager_rng.clone(),
                            eager.loop_state.fineness < params.n_iter,
                        );
                        let mut scored = lazy.clone();
                        let actual_action = split.propose_next_action_with_sampled_scores(
                            &mut scored,
                            &strategy,
                            &mut lazy_rng.clone(),
                            lazy.loop_state.fineness < params.n_iter,
                        );
                        if format!("{actual_action:?}") != format!("{expected_action:?}") {
                            if let Some(FittingAction::ApplySplit { split: candidate }) =
                                &expected_action
                            {
                                let (col, pos) = (candidate.col, candidate.index);
                                let interval = eager.boundaries[col]
                                    .partition_point(|&boundary| boundary <= pos);
                                let mut rescored = eager.clone();
                                strategy.refresh_sampled_split(&mut rescored, col, pos);
                                eprintln!(
                                    "first proposal difference: case={case}, iteration={iteration}, bins={bins:?}, huber={huber}, positive={positive}, reference={reference_kind}, coordinate=({col},{pos}), interval={:?}, boundaries={:?}, stored_gain={:?}, rescored_gain={:?}, stored_left={:?}, rescored_left={:?}, stored_right={:?}, rescored_right={:?}",
                                    eager.interval_range(col, interval),
                                    eager.boundaries[col],
                                    eager.precomputed_statistics.error_reductions_split[col][pos],
                                    rescored.precomputed_statistics.error_reductions_split[col][pos],
                                    eager.precomputed_statistics.update_pairs_split_left[col][pos],
                                    rescored.precomputed_statistics.update_pairs_split_left[col][pos],
                                    eager.precomputed_statistics.update_pairs_split_right[col][pos],
                                    rescored.precomputed_statistics.update_pairs_split_right[col][pos],
                                );
                            }
                            panic!("case {case}, iteration {iteration}: {actual_action:?} != {expected_action:?}");
                        }
                        let (next_eager, eager_advanced) = fit_step(
                            eager,
                            iteration,
                            params.n_iter,
                            &strategy,
                            &split,
                            &mut eager_rng,
                        );
                        let (next_lazy, lazy_advanced) = fit_step(
                            lazy,
                            iteration,
                            params.n_iter,
                            &strategy,
                            &split,
                            &mut lazy_rng,
                        );
                        eager = next_eager;
                        lazy = next_lazy;
                        assert_eq!(
                            lazy_advanced, eager_advanced,
                            "case {case}, iteration {iteration}"
                        );
                        assert_eq!(lazy_rng.clone().next_u64(), eager_rng.clone().next_u64());
                        assert_states_identical(&lazy, &eager);
                        sampled_steps += usize::from(lazy_advanced && !positive);
                        if !eager_advanced {
                            break;
                        }
                    }
                    lazy.sampled_split_scoring = false;
                    eager = strategy.refresh_statistics(eager);
                    lazy = strategy.refresh_statistics(lazy);
                    matrix_bits(
                        &lazy.precomputed_statistics.error_reductions_split,
                        &eager.precomputed_statistics.error_reductions_split,
                    );
                    pair_bits(
                        &lazy.precomputed_statistics.update_pairs_split_left,
                        &eager.precomputed_statistics.update_pairs_split_left,
                    );
                    pair_bits(
                        &lazy.precomputed_statistics.update_pairs_split_right,
                        &eager.precomputed_statistics.update_pairs_split_right,
                    );
                    let (eager_fit, eager_grid) = finish_fit(eager);
                    let (lazy_fit, lazy_grid) = finish_fit(lazy);
                    assert_eq!(
                        bincode::serialize(&lazy_grid).unwrap(),
                        bincode::serialize(&eager_grid).unwrap()
                    );
                    assert_eq!(lazy_fit.y_hat, eager_fit.y_hat);
                    case += 1;
                }
            }
        }
    }
    assert_eq!(case, 48);
    assert!(sampled_steps > 20);
}

#[test]
fn on_demand_scores_single_candidate_values_match_eager_bits() {
    for bins in [None, Some(8), Some(64)] {
        for huber in [false, true] {
            let (x, y) = fixture(false, false);
            let params = parameters(bins, huber, 10, 1.0, 1, true);
            let strategy = params.refinement_strategy_params.get_refinement_strategy();
            let split = params.split_strategy_params.get_split_strategy();
            let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
            let mut rng = StdRng::seed_from_u64(53);
            for iteration in 0..6 {
                let (next, advanced) =
                    fit_step(state, iteration, params.n_iter, &strategy, &split, &mut rng);
                state = next;
                if !advanced {
                    break;
                }
            }
            let reference = changing_reference(&state, 1, 7).unwrap();
            let eager = strategy.refresh_consensus_reference(state.clone(), reference.clone());
            state.sampled_split_scoring = true;
            let mut lazy = strategy.refresh_consensus_reference(state, reference);
            for col in 0..lazy.p {
                for position in 0..lazy.n {
                    strategy.refresh_sampled_split(&mut lazy, col, position);
                }
            }
            matrix_bits(
                &lazy.precomputed_statistics.error_reductions_split,
                &eager.precomputed_statistics.error_reductions_split,
            );
            pair_bits(
                &lazy.precomputed_statistics.update_pairs_split_left,
                &eager.precomputed_statistics.update_pairs_split_left,
            );
            pair_bits(
                &lazy.precomputed_statistics.update_pairs_split_right,
                &eager.precomputed_statistics.update_pairs_split_right,
            );
            pair_bits(
                &lazy.precomputed_statistics.error_reductions_split_pairs,
                &eager.precomputed_statistics.error_reductions_split_pairs,
            );
        }
    }
}

#[test]
fn on_demand_scores_best_and_top_k_preserve_local_cached_scores() {
    let (x, y) = fixture(false, false);
    for top_k in [false, true] {
        for bins in [None, Some(8)] {
            for huber in [false, true] {
                for periodic_reference in [false, true] {
                    let mut params = parameters(bins, huber, 10, 1.0, 1, true);
                    let split_params = SplitStrategyParamsBuilder::new()
                        .min_interval_samples(1)
                        .complexity_penalty(0.5);
                    params.split_strategy_params = if top_k {
                        split_params.top_k_splits().top_k(2)
                    } else {
                        split_params.best_split()
                    }
                    .build();
                    let strategy = params.refinement_strategy_params.get_refinement_strategy();
                    let split = params.split_strategy_params.get_split_strategy();
                    let mut eager = initialize_state_with_input(x.view(), y.view(), &params, None);
                    let mut lazy = eager.clone();
                    lazy.sampled_split_scoring = true;
                    let mut eager_rng = StdRng::seed_from_u64(73);
                    let mut lazy_rng = eager_rng.clone();
                    let mut advanced_steps = 0;
                    for iteration in 0..max_iterations(params.n_iter) {
                        if periodic_reference && iteration % 3 == 0 {
                            let reference = changing_reference(&eager, 1, iteration);
                            eager = refresh_reference(eager, &strategy, reference.clone());
                            lazy = refresh_reference(lazy, &strategy, reference);
                        }
                        let expected_action = split.propose_next_action_with_split_budget(
                            &eager,
                            &mut eager_rng.clone(),
                            eager.loop_state.fineness < params.n_iter,
                        );
                        let mut scored = lazy.clone();
                        let actual_action = split.propose_next_action_with_sampled_scores(
                            &mut scored,
                            &strategy,
                            &mut lazy_rng.clone(),
                            lazy.loop_state.fineness < params.n_iter,
                        );
                        assert_eq!(
                            format!("{actual_action:?}"),
                            format!("{expected_action:?}"),
                            "mode={}, bins={bins:?}, huber={huber}, reference={periodic_reference}, iteration={iteration}",
                            if top_k { "TopK" } else { "Best" },
                        );
                        let (next_eager, eager_advanced) = fit_step(
                            eager,
                            iteration,
                            params.n_iter,
                            &strategy,
                            &split,
                            &mut eager_rng,
                        );
                        let (next_lazy, lazy_advanced) = fit_step(
                            lazy,
                            iteration,
                            params.n_iter,
                            &strategy,
                            &split,
                            &mut lazy_rng,
                        );
                        eager = next_eager;
                        lazy = next_lazy;
                        assert_eq!(lazy_advanced, eager_advanced);
                        assert_eq!(lazy_rng.clone().next_u64(), eager_rng.clone().next_u64());
                        assert_states_identical(&lazy, &eager);
                        matrix_bits(
                            &lazy.precomputed_statistics.error_reductions_split,
                            &eager.precomputed_statistics.error_reductions_split,
                        );
                        if !eager_advanced {
                            break;
                        }
                        advanced_steps += 1;
                    }
                    assert!(advanced_steps > 1);
                    let (_, eager_grid) = finish_fit(eager);
                    let (_, lazy_grid) = finish_fit(lazy);
                    assert_eq!(
                        bincode::serialize(&lazy_grid).unwrap(),
                        bincode::serialize(&eager_grid).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn on_demand_scores_positive_only_transition_refreshes_unsampled_columns() {
    let (x, y) = fixture(false, false);
    let params = parameters(Some(8), false, 1, 1.0, 1, false);
    let strategy = params.refinement_strategy_params.get_refinement_strategy();
    let split = params.split_strategy_params.get_split_strategy();
    let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
    assert!(!state.is_stage1_positive_only());
    state.sampled_split_scoring = true;
    let reference = changing_reference(&state, 1, 3).unwrap();
    state = strategy.refresh_consensus_reference(state, reference);
    assert!(state.split_scores_stale);
    let stale = state.precomputed_statistics.clone();

    state.lambda_minus = 0.0;
    for tilts in &mut state.tilt_values {
        tilts.fill(0.0);
    }
    state.f_minus.fill(0.0);
    state.f.assign(&state.f_plus);
    state.y_hat.assign(&state.f);
    state.residuals.assign(&(&y - &state.f));
    state.r_tilde.assign(&state.residuals);
    state.sampled_split_scoring = false;
    state.precomputed_statistics.consensus_weight = f64::NAN;
    let expected = strategy.refresh_statistics(state);
    assert!(expected.is_stage1_positive_only());
    assert!(stale.error_reductions_split[1]
        .iter()
        .zip(&expected.precomputed_statistics.error_reductions_split[1])
        .any(|(a, e)| a.is_finite() && e.is_finite() && a.to_bits() != e.to_bits()));

    let mut pending = expected.clone();
    pending.sampled_split_scoring = true;
    pending.split_scores_stale = true;
    pending.precomputed_statistics.error_reductions_split = stale.error_reductions_split;
    pending.precomputed_statistics.error_reductions_split_pairs =
        stale.error_reductions_split_pairs;
    pending.precomputed_statistics.update_pairs_split_left = stale.update_pairs_split_left;
    pending.precomputed_statistics.update_pairs_split_right = stale.update_pairs_split_right;
    let mut proposed = pending.clone();
    let partial = strategy.refresh_error_reduction_caches_for_affected(
        pending,
        &[AffectedRange {
            col: 0,
            point_range: (0, expected.n - 1),
            interval_range: (0, 0),
        }],
    );
    assert!(!partial.split_scores_stale);
    matrix_bits(
        &partial.precomputed_statistics.error_reductions_split,
        &expected.precomputed_statistics.error_reductions_split,
    );
    pair_bits(
        &partial.precomputed_statistics.update_pairs_split_left,
        &expected.precomputed_statistics.update_pairs_split_left,
    );
    pair_bits(
        &partial.precomputed_statistics.update_pairs_split_right,
        &expected.precomputed_statistics.update_pairs_split_right,
    );

    let mut actual_rng = StdRng::seed_from_u64(11);
    let mut expected_rng = actual_rng.clone();
    let actual_action = split.propose_next_action_with_sampled_scores(
        &mut proposed,
        &strategy,
        &mut actual_rng,
        true,
    );
    let expected_action = split.propose_next_action(&expected, &mut expected_rng);
    assert_eq!(format!("{actual_action:?}"), format!("{expected_action:?}"));
    assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());
    assert!(!proposed.split_scores_stale);
    matrix_bits(
        &proposed.precomputed_statistics.error_reductions_split,
        &expected.precomputed_statistics.error_reductions_split,
    );
}

#[test]
fn on_demand_scores_two_branch_scoring_preserves_masks_with_extreme_inputs() {
    let (x, y) = fixture(false, false);
    let params = parameters(None, false, 10, 1.0, 1, false);
    let initial = initialize_state_with_input(x.view(), y.view(), &params, None);
    assert!(!initial.is_stage1_positive_only());
    let position = initial.precomputed_statistics.error_reductions_split[0]
        .iter()
        .position(|gain| gain.is_finite())
        .unwrap();
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -1e300,
        0.0,
        1e-300,
        1e300,
    ] {
        for alpha in [0.01, -1.0, f64::INFINITY, f64::NAN] {
            let strategy = RefinementStrategy::L2Refinement {
                alpha,
                tilt_tau: 0.01,
                tilt_rho: 0.0,
                prior_sample_size: 0.0,
                update_clamp: f64::INFINITY,
            };
            let mut lazy = initial.clone();
            lazy.sampled_split_scoring = true;
            let last = lazy.n - 1;
            lazy.precomputed_statistics.prefix_sums[0][position - 1] = PrefixStats {
                s11: value,
                s22: value,
                s12: value,
                t1: value,
                t2: value,
            };
            lazy.precomputed_statistics.prefix_sums[0][last] = PrefixStats {
                s11: value * 2.0,
                s22: value * 2.0,
                s12: value * 2.0,
                t1: value * 2.0,
                t2: value * 2.0,
            };
            let mut eager = lazy.clone();
            eager.sampled_split_scoring = false;
            strategy.refresh_all_split_scores(&mut eager);
            strategy.refresh_sampled_split(&mut lazy, 0, position);
            let actual = &lazy.precomputed_statistics;
            let expected = &eager.precomputed_statistics;
            assert!(!actual.error_reductions_split[0][position].is_nan());
            let gains = actual.error_reductions_split_pairs[0][position];
            assert!(gains.0.is_finite() && gains.1.is_finite());
            bits(
                actual.error_reductions_split[0][position],
                expected.error_reductions_split[0][position],
            );
            assert!(actual.error_reductions_split[0][0].is_nan());
            assert!(actual.error_reductions_split[0][last].is_nan());
        }
    }
}
