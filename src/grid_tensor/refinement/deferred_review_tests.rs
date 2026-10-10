use std::sync::Arc;

use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, RngCore, SeedableRng};

use crate::grid_tensor::{
    action::FittingAction,
    fit::{finish_fit, initialize_state_with_input},
    params::{
        GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
    },
    reducer::{fitting_reducer, fitting_reducer_with_candidate_refresh, CandidateRefresh},
    splitting::{MergeCandidate, ResplitCandidate, SplitCandidate},
    state::{BranchConsensus, ConsensusAnchors, FittingState, PrefixStats},
    GridTensorParams,
};

fn data(constant_column: bool) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((128, 4), |(row, col)| {
        if constant_column && col == 3 {
            1.0
        } else {
            ((row * (2 * col + 3) + col) % 97) as f64 / 13.0
        }
    });
    let y = Array1::from_shape_fn(x.nrows(), |row| {
        x[(row, 0)].sin() * (1.0 + x[(row, 1)]) - 0.4 * x[(row, 2)]
    });
    (x, y)
}

fn parameters(bins: Option<u16>, huber: bool) -> GridTensorParams {
    let refinement = RefinementStrategyParamsBuilder::new()
        .alpha(0.01)
        .tilt_tau(0.01);
    GridTensorParamsBuilder::new()
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
        .split_strategy(
            SplitStrategyParamsBuilder::new()
                .random_split()
                .min_interval_samples(1)
                .split_try(10)
                .complexity_penalty(0.1)
                .build(),
        )
        .build()
}

fn reference(n: usize, anchored: bool, phase: usize) -> BranchConsensus {
    let anchors = anchored.then(|| Arc::new(ConsensusAnchors::new(n, (0..n).step_by(5).collect())));
    let rows: Vec<usize> = anchors
        .as_ref()
        .map_or_else(|| (0..n).collect(), |a| a.rows.clone());
    BranchConsensus {
        weight: 0.3 * n as f64 / rows.len() as f64,
        plus: Array1::from_iter(
            rows.iter()
                .map(|&row| 1.7 + 0.2 * (row + phase) as f64 / n as f64),
        ),
        minus: Array1::from_iter(
            rows.iter()
                .map(|&row| 0.9 + 0.1 * (row as f64 - phase as f64).sin()),
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
    for (&a, &e) in actual.iter().zip(expected) {
        bits(a, e);
    }
}

fn prefix_bits(actual: &[Vec<PrefixStats>], expected: &[Vec<PrefixStats>], linear: bool) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            slice_bits(&[a.s11, a.s22, a.s12], &[e.s11, e.s22, e.s12]);
            if linear {
                slice_bits(&[a.t1, a.t2], &[e.t1, e.t2]);
            }
        }
    }
}

fn statistics_bits(actual: &FittingState, expected: &FittingState, linear: bool) {
    let (a, e) = (
        &actual.precomputed_statistics,
        &expected.precomputed_statistics,
    );
    crate::grid_tensor::state::assert_curvature_bits(&a.curvature, &e.curvature);
    if linear {
        slice_bits(&a.c_t1, &e.c_t1);
        slice_bits(&a.c_t2, &e.c_t2);
    }
    prefix_bits(&a.prefix_sums, &e.prefix_sums, linear);
    prefix_bits(&a.binned_prefix_sums, &e.binned_prefix_sums, linear);
}

fn prediction_bits(actual: &FittingState, expected: &FittingState) {
    assert_eq!(actual.boundaries, expected.boundaries);
    assert_eq!(actual.interval_id, expected.interval_id);
    for (actual, expected) in [
        (&actual.backbone_values, &expected.backbone_values),
        (&actual.tilt_values, &expected.tilt_values),
    ] {
        for (actual, expected) in actual.iter().zip(expected) {
            slice_bits(actual, expected);
        }
    }
    for (actual, expected) in [
        (&actual.f_plus, &expected.f_plus),
        (&actual.f_minus, &expected.f_minus),
        (&actual.f, &expected.f),
        (&actual.y_hat, &expected.y_hat),
        (&actual.residuals, &expected.residuals),
        (&actual.r_tilde, &expected.r_tilde),
    ] {
        slice_bits(actual.as_slice().unwrap(), expected.as_slice().unwrap());
    }
    bits(actual.current_error, expected.current_error);
    bits(actual.lambda_plus, expected.lambda_plus);
    bits(actual.lambda_minus, expected.lambda_minus);
}

fn best_split(state: &FittingState) -> SplitCandidate {
    let mut best: Option<SplitCandidate> = None;
    for col in 0..state.p {
        for (allowed_interval_idx, interval) in state.split_strategy_state.allowed_intervals[col]
            .iter()
            .enumerate()
        {
            for index in interval.start..interval.end() {
                let gain = state.precomputed_statistics.error_reductions_split[col][index];
                if gain.is_finite() && best.as_ref().is_none_or(|best| gain > best.error_reduction)
                {
                    best = Some(SplitCandidate {
                        col,
                        index,
                        allowed_interval_idx,
                        error_reduction: gain,
                        update_left: state.precomputed_statistics.update_pairs_split_left[col]
                            [index],
                        update_right: state.precomputed_statistics.update_pairs_split_right[col]
                            [index],
                    });
                }
            }
        }
    }
    best.expect("fixture has a finite split")
}

fn actions(state: &FittingState) -> Vec<FittingAction> {
    let split = FittingAction::ApplySplit {
        split: best_split(state),
    };
    let (col, interval_idx) = state
        .boundaries
        .iter()
        .enumerate()
        .find_map(|(col, boundaries)| {
            (0..boundaries.len()).find_map(|interval_idx| {
                let stats = &state.precomputed_statistics;
                (stats.error_reductions_resplit[col][interval_idx].is_finite()
                    && stats.error_reductions_merge[col][interval_idx].is_finite())
                .then_some((col, interval_idx))
            })
        })
        .expect("fixture has finite resplit and merge candidates");
    let index = state.boundaries[col][interval_idx];
    vec![
        split,
        FittingAction::ApplyResplit {
            resplit: ResplitCandidate {
                col,
                index,
                interval_idx,
                error_reduction: state.precomputed_statistics.error_reductions_resplit[col]
                    [interval_idx],
                update_left: state.precomputed_statistics.update_pairs_resplit_left[col]
                    [interval_idx],
                update_right: state.precomputed_statistics.update_pairs_resplit_right[col]
                    [interval_idx],
            },
        },
        FittingAction::ApplyMerge {
            merge: MergeCandidate {
                col,
                index,
                interval_idx,
                error_reduction: state.precomputed_statistics.error_reductions_merge[col]
                    [interval_idx],
            },
        },
    ]
}

#[test]
fn deferred_review_curvature_keeps_the_eager_arithmetic() {
    for bins in [None, Some(8), Some(64)] {
        for constant in [false, true] {
            let (x, y) = data(constant);
            let params = parameters(bins, false);
            let strategy = params.refinement_strategy_params.get_refinement_strategy();
            let initial = initialize_state_with_input(x.view(), y.view(), &params, None);
            let mut state =
                strategy.refresh_consensus_reference(initial, reference(x.nrows(), false, 0));
            for col in 0..2 {
                let edges = &state.precomputed_statistics.bin_edges[col];
                let (start, end) = if edges.len() >= 4 {
                    (edges[1], edges[3])
                } else {
                    (17, 93)
                };
                let rows = state.precomputed_statistics.sorted_indices[col][start..end].to_vec();
                for row in rows {
                    state.f_plus[row] *= 1.007;
                    state.f_minus[row] *= 0.996;
                    state.f[row] = state.f_plus[row] - state.f_minus[row];
                    state.y_hat[row] = state.f[row];
                    state.residuals[row] = state.labels[row] - state.f[row];
                    state.r_tilde[row] = state.residuals[row];
                }
                let affected = state.get_affected_ranges(col, (start, end));
                let mut ranges = vec![(1, 0); state.p];
                for range in &affected {
                    ranges[range.col] = range.point_range;
                }
                let old_t1 = state.precomputed_statistics.c_t1.clone();
                let old_t2 = state.precomputed_statistics.c_t2.clone();
                let expected = strategy.update_statistics(state.clone(), col, start, end, &ranges);
                let actual = strategy.update_curvature_statistics(state, col, start, end, &ranges);
                assert!(actual.linear_statistics_stale);
                statistics_bits(&actual, &expected, false);
                prediction_bits(&actual, &expected);
                slice_bits(&actual.precomputed_statistics.c_t1, &old_t1);
                slice_bits(&actual.precomputed_statistics.c_t2, &old_t2);
                assert!(expected
                    .precomputed_statistics
                    .c_t1
                    .iter()
                    .zip(&old_t1)
                    .any(|(a, b)| a.to_bits() != b.to_bits()));
                let expected = strategy.refresh_statistics(expected);
                state = strategy.refresh_statistics(actual);
                assert!(!state.linear_statistics_stale);
                statistics_bits(&state, &expected, true);
                assert_eq!(
                    format!("{:?}", state.precomputed_statistics),
                    format!("{:?}", expected.precomputed_statistics)
                );
            }
        }
    }
}

#[test]
fn deferred_review_actions_refresh_finish_and_log_exactly() {
    for bins in [None, Some(8), Some(64)] {
        for context in 0..4 {
            let (x, y) = data(true);
            let params = parameters(bins, context == 2);
            let strategy = params.refinement_strategy_params.get_refinement_strategy();
            let split_strategy = params.split_strategy_params.get_split_strategy();
            let mut initial = initialize_state_with_input(x.view(), y.view(), &params, None);
            if context != 3 {
                initial = strategy
                    .refresh_consensus_reference(initial, reference(x.nrows(), context == 1, 0));
            }
            for _ in 0..3 {
                let split = best_split(&initial);
                initial = fitting_reducer(
                    initial,
                    FittingAction::ApplySplit { split },
                    &strategy,
                    &split_strategy,
                );
            }
            for action in actions(&initial) {
                let expected =
                    fitting_reducer(initial.clone(), action.clone(), &strategy, &split_strategy);
                let actual = fitting_reducer_with_candidate_refresh(
                    initial.clone(),
                    action.clone(),
                    &strategy,
                    &split_strategy,
                    CandidateRefresh::Deferred,
                );
                assert_eq!(actual.linear_statistics_stale, context == 0);
                prediction_bits(&actual, &expected);
                statistics_bits(&actual, &expected, context != 0);
                assert!(actual
                    .f_plus
                    .iter()
                    .chain(actual.f_minus.iter())
                    .all(|v| *v >= 0.0 && v.is_finite()));

                #[cfg(feature = "evo-logging")]
                {
                    use crate::grid_tensor::logging_helpers::create_simple_split_event;
                    let actual_event =
                        create_simple_split_event(&actual, &action, initial.current_error);
                    let expected_event =
                        create_simple_split_event(&expected, &action, initial.current_error);
                    assert_eq!(
                        serde_json::to_string(&actual_event).unwrap(),
                        serde_json::to_string(&expected_event).unwrap()
                    );
                }

                let (actual_fit, actual_grid) = finish_fit(actual.clone());
                let (expected_fit, expected_grid) = finish_fit(expected.clone());
                assert_eq!(
                    bincode::serialize(&actual_grid).unwrap(),
                    bincode::serialize(&expected_grid).unwrap()
                );
                slice_bits(
                    actual_fit.y_hat.as_slice().unwrap(),
                    expected_fit.y_hat.as_slice().unwrap(),
                );
                slice_bits(
                    actual_fit.residuals.as_slice().unwrap(),
                    expected_fit.residuals.as_slice().unwrap(),
                );
                bits(actual_fit.err, expected_fit.err);

                for anchored in [false, true] {
                    let next = reference(x.nrows(), anchored, 7);
                    let mut actual =
                        strategy.refresh_consensus_reference(actual.clone(), next.clone());
                    let expected = strategy.refresh_consensus_reference(expected.clone(), next);
                    assert!(!actual.linear_statistics_stale);
                    statistics_bits(&actual, &expected, true);
                    assert_eq!(
                        format!("{:?}", actual.precomputed_statistics),
                        format!("{:?}", expected.precomputed_statistics)
                    );
                    let mut actual_rng = StdRng::seed_from_u64(31);
                    let mut expected_rng = actual_rng.clone();
                    let a = split_strategy.propose_next_action_with_sampled_scores(
                        &mut actual,
                        &strategy,
                        &mut actual_rng,
                        true,
                    );
                    let e = split_strategy.propose_next_action(&expected, &mut expected_rng);
                    assert_eq!(format!("{a:?}"), format!("{e:?}"));
                    assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "linear statistics require a full refresh")]
fn deferred_review_stale_statistics_reject_a_second_incremental_update() {
    let (x, y) = data(false);
    let params = parameters(Some(8), false);
    let strategy = params.refinement_strategy_params.get_refinement_strategy();
    let state = initialize_state_with_input(x.view(), y.view(), &params, None);
    let state = strategy.refresh_consensus_reference(state, reference(x.nrows(), false, 0));
    let ranges = vec![(0, state.n - 1); state.p];
    let n = state.n;
    let state = strategy.update_curvature_statistics(state, 0, 0, n, &ranges);
    let _ = strategy.update_statistics(state, 0, 0, n, &ranges);
}

#[test]
#[should_panic(expected = "linear statistics require a full refresh")]
fn deferred_review_stale_statistics_reject_an_eager_proposal() {
    let (x, y) = data(false);
    let params = parameters(Some(8), false);
    let strategy = params.refinement_strategy_params.get_refinement_strategy();
    let split = params.split_strategy_params.get_split_strategy();
    let state = initialize_state_with_input(x.view(), y.view(), &params, None);
    let state = strategy.refresh_consensus_reference(state, reference(x.nrows(), false, 0));
    let ranges = vec![(0, state.n - 1); state.p];
    let n = state.n;
    let state = strategy.update_curvature_statistics(state, 0, 0, n, &ranges);
    let _ = split.propose_next_action(&state, &mut StdRng::seed_from_u64(3));
}
