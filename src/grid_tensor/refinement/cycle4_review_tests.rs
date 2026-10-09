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

fn fixture(constant: bool) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((127, 4), |(row, col)| {
        if constant && col == 3 {
            2.0
        } else {
            ((row * [1, 31, 53, 71][col]) % 127) as f64 / 17.0
        }
    });
    let y = Array1::from_shape_fn(127, |row| {
        x[(row, 0)].sin() * (1.0 + x[(row, 1)]) - x[(row, 2)] * 0.3
    });
    (x, y)
}

fn params(bins: Option<u16>, huber: bool) -> GridTensorParams {
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

fn reference(n: usize, anchors: bool) -> BranchConsensus {
    let anchors = anchors.then(|| Arc::new(ConsensusAnchors::new(n, (0..n).step_by(7).collect())));
    let rows: Vec<_> = anchors
        .as_ref()
        .map_or_else(|| (0..n).collect(), |a| a.rows.clone());
    BranchConsensus {
        weight: 0.3 * n as f64 / rows.len() as f64,
        plus: Array1::from_iter(rows.iter().map(|&row| 1.2 + row as f64 / n as f64)),
        minus: Array1::from_iter(rows.iter().map(|&row| 0.4 + 0.1 * (row as f64).cos())),
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

fn statistics_bits(actual: &FittingState, expected: &FittingState, linear: bool) {
    let (a, e) = (
        &actual.precomputed_statistics,
        &expected.precomputed_statistics,
    );
    for (a, e) in [
        (&a.c_s11, &e.c_s11),
        (&a.c_s22, &e.c_s22),
        (&a.c_s12, &e.c_s12),
    ] {
        slice_bits(a, e);
    }
    if linear {
        slice_bits(&a.c_t1, &e.c_t1);
        slice_bits(&a.c_t2, &e.c_t2);
    }
    for (a, e) in [
        (&a.prefix_sums, &e.prefix_sums),
        (&a.binned_prefix_sums, &e.binned_prefix_sums),
    ] {
        assert_eq!(a.len(), e.len());
        for (a, e) in a.iter().zip(e) {
            assert_eq!(a.len(), e.len());
            for (a, e) in a.iter().zip(e) {
                slice_bits(&[a.s11, a.s22, a.s12], &[e.s11, e.s22, e.s12]);
                if linear {
                    slice_bits(&[a.t1, a.t2], &[e.t1, e.t2]);
                }
            }
        }
    }
}

fn ranges(state: &FittingState, col: usize, start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut result = vec![(1, 0); state.p];
    for range in state.get_affected_ranges(col, (start, end)) {
        result[range.col] = range.point_range;
    }
    result
}

fn update_prediction(state: &mut FittingState, row: usize, plus: f64, minus: f64) {
    state.f_plus[row] = plus;
    state.f_minus[row] = minus;
    state.f[row] = plus - minus;
    state.y_hat[row] = state.f[row];
    state.residuals[row] = state.labels[row] - state.f[row];
    state.r_tilde[row] = state.residuals[row];
}

#[test]
fn cycle4_review_binned_kernel_matches_rank_range_oracle() {
    let (x, y) = fixture(false);
    for bins in [8, 64] {
        let params = params(Some(bins), false);
        let refinement = params.refinement_strategy_params.get_refinement_strategy();
        let state = initialize_state_with_input(x.view(), y.view(), &params, None);
        let initial = refinement.refresh_consensus_reference(state, reference(x.nrows(), false));
        for col in 0..x.ncols() {
            let edges = &initial.precomputed_statistics.bin_edges[col];
            for (start, end) in [
                (0, 127),
                (edges[1], edges[3]),
                (*edges.last().unwrap(), 127),
                (31, 32),
            ] {
                for scenario in 0..3 {
                    let mut state = initial.clone();
                    let rows =
                        state.precomputed_statistics.sorted_indices[col][start..end].to_vec();
                    for (offset, row) in rows.into_iter().enumerate() {
                        let (fp, fm) = match scenario {
                            0 => (state.f_plus[row] * 1.007, state.f_minus[row] * 0.996),
                            1 => (state.f_plus[row], state.f_minus[row]),
                            _ => {
                                let magnitude = [1e-120, 1e120, 1e-40, 1e40][offset % 4];
                                (magnitude, magnitude * 0.5)
                            }
                        };
                        update_prediction(&mut state, row, fp, fm);
                    }
                    let affected = ranges(&state, col, start, end);
                    let expected = refinement.update_curvature_statistics(
                        state.clone(),
                        col,
                        start,
                        end,
                        &affected,
                    );
                    let actual =
                        refinement.update_binned_curvature_statistics(state, col, start, end);
                    statistics_bits(&actual, &expected, true);
                    assert_eq!(
                        actual.linear_statistics_stale,
                        expected.linear_statistics_stale
                    );
                    assert_eq!(
                        format!("{:?}", actual.precomputed_statistics),
                        format!("{:?}", expected.precomputed_statistics)
                    );
                    let expected = refinement.refresh_statistics(expected);
                    let actual = refinement.refresh_statistics(actual);
                    statistics_bits(&actual, &expected, true);
                    assert_eq!(
                        format!("{:?}", actual.precomputed_statistics),
                        format!("{:?}", expected.precomputed_statistics)
                    );
                }
            }
        }
        let actual = refinement.update_binned_curvature_statistics(initial.clone(), 0, 31, 31);
        statistics_bits(&actual, &initial, true);
        assert!(!actual.linear_statistics_stale);
    }
}

#[test]
fn cycle4_review_zero_deltas_keep_touched_range_and_signed_zero() {
    let (x, y) = fixture(false);
    let params = params(Some(8), false);
    let refinement = params.refinement_strategy_params.get_refinement_strategy();
    let state = initialize_state_with_input(x.view(), y.view(), &params, None);
    let mut state = refinement.refresh_consensus_reference(state, reference(x.nrows(), false));
    for row in 0..state.n {
        update_prediction(&mut state, row, -0.0, 0.0);
    }
    state.precomputed_statistics.c_s11.fill(0.0);
    state.precomputed_statistics.c_s22.fill(0.0);
    state.precomputed_statistics.c_s12.fill(-0.0);
    for column in &mut state.precomputed_statistics.binned_prefix_sums {
        for value in column {
            *value = PrefixStats {
                s11: -0.0,
                s22: -0.0,
                s12: -0.0,
                ..*value
            };
        }
    }
    let (col, start, end) = (0, 43, 44);
    let affected = ranges(&state, col, start, end);
    let expected =
        refinement.update_curvature_statistics(state.clone(), col, start, end, &affected);
    let actual = refinement.update_binned_curvature_statistics(state.clone(), col, start, end);
    statistics_bits(&actual, &expected, true);
    let touched = state.precomputed_statistics.bin_indices[col]
        [state.precomputed_statistics.sorted_indices[col][start]] as usize;
    assert!(touched > 0);
    bits(
        actual.precomputed_statistics.binned_prefix_sums[col][touched - 1].s11,
        -0.0,
    );
    bits(
        actual.precomputed_statistics.binned_prefix_sums[col][touched].s11,
        0.0,
    );
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
    let statistics = &state.precomputed_statistics;
    vec![
        FittingAction::ApplySplit {
            split: best_split(state),
        },
        FittingAction::ApplyResplit {
            resplit: ResplitCandidate {
                col,
                interval_idx,
                index,
                error_reduction: statistics.error_reductions_resplit[col][interval_idx],
                update_left: statistics.update_pairs_resplit_left[col][interval_idx],
                update_right: statistics.update_pairs_resplit_right[col][interval_idx],
            },
        },
        FittingAction::ApplyMerge {
            merge: MergeCandidate {
                col,
                interval_idx,
                index,
                error_reduction: statistics.error_reductions_merge[col][interval_idx],
            },
        },
    ]
}

#[test]
fn cycle4_review_structural_actions_fallbacks_logging_and_reference_transitions() {
    for bins in [None, Some(8), Some(64)] {
        for constant in [false, true] {
            for context in 0..4 {
                let (x, y) = fixture(constant);
                let params = params(bins, context == 2);
                let refinement = params.refinement_strategy_params.get_refinement_strategy();
                let split = params.split_strategy_params.get_split_strategy();
                let mut initial = initialize_state_with_input(x.view(), y.view(), &params, None);
                if context != 3 {
                    initial = refinement
                        .refresh_consensus_reference(initial, reference(x.nrows(), context == 1));
                }
                for _ in 0..3 {
                    let action = FittingAction::ApplySplit {
                        split: best_split(&initial),
                    };
                    initial = fitting_reducer(initial, action, &refinement, &split);
                }
                for action in actions(&initial) {
                    let expected =
                        fitting_reducer(initial.clone(), action.clone(), &refinement, &split);
                    let mut logged = initial.clone();
                    #[cfg(feature = "evo-logging")]
                    {
                        logged.logging_state =
                            Some(crate::grid_tensor::logging_helpers::LoggingState::new());
                    }
                    let actual = fitting_reducer_with_candidate_refresh(
                        logged,
                        action.clone(),
                        &refinement,
                        &split,
                        CandidateRefresh::Deferred,
                    );
                    assert_eq!(actual.linear_statistics_stale, context == 0);
                    statistics_bits(&actual, &expected, context != 0);
                    for (a, e) in [
                        (&actual.f_plus, &expected.f_plus),
                        (&actual.f_minus, &expected.f_minus),
                        (&actual.residuals, &expected.residuals),
                    ] {
                        slice_bits(a.as_slice().unwrap(), e.as_slice().unwrap());
                    }
                    bits(actual.current_error, expected.current_error);
                    #[cfg(feature = "evo-logging")]
                    {
                        use crate::grid_tensor::logging_helpers::{
                            create_simple_split_event, RawLoggingEvent,
                        };
                        let events = &actual.logging_state.as_ref().unwrap().events;
                        assert_eq!(events.len(), 1);
                        let RawLoggingEvent::Split(event) = &events[0] else {
                            panic!("expected a split event");
                        };
                        let expected_event =
                            create_simple_split_event(&expected, &action, initial.current_error);
                        assert_eq!(
                            serde_json::to_string(event).unwrap(),
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
                    for next in 0..3 {
                        let (mut a, mut e) = (actual.clone(), expected.clone());
                        if next == 2 {
                            a.branch_consensus = None;
                            e.branch_consensus = None;
                            a = refinement.refresh_statistics(a);
                            e = refinement.refresh_statistics(e);
                        } else {
                            a = refinement
                                .refresh_consensus_reference(a, reference(x.nrows(), next == 1));
                            e = refinement
                                .refresh_consensus_reference(e, reference(x.nrows(), next == 1));
                        }
                        statistics_bits(&a, &e, true);
                        assert_eq!(
                            format!("{:?}", a.precomputed_statistics),
                            format!("{:?}", e.precomputed_statistics)
                        );
                        let mut arng = StdRng::seed_from_u64(31);
                        let mut erng = arng.clone();
                        let aa = split.propose_next_action(&a, &mut arng);
                        let ea = split.propose_next_action(&e, &mut erng);
                        assert_eq!(format!("{aa:?}"), format!("{ea:?}"));
                        assert_eq!(arng.next_u64(), erng.next_u64());
                    }
                }
            }
        }
    }
}
