use std::sync::Arc;

use ndarray::{Array1, Array2};

use super::RefinementStrategy;
use crate::grid_tensor::{
    action::FittingAction,
    fit::initialize_state_with_input,
    params::{
        GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
    },
    reducer::fitting_reducer,
    splitting::{MergeCandidate, SplitCandidate, SplitStrategy},
    state::{AffectedRange, BranchConsensus, ConsensusAnchors, FittingState, PrefixStats},
    GridTensorParams,
};

fn data(constant_column: bool) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((96, 3), |(i, j)| {
        if constant_column && j == 2 {
            1.0
        } else {
            ((i * (2 * j + 3) + j) % 61) as f64 / 7.0
        }
    });
    let y = Array1::from_shape_fn(x.nrows(), |i| x[(i, 0)].sin() * x[(i, 1)] - 0.3 * x[(i, 2)]);
    (x, y)
}

fn parameters(max_bins: Option<u16>, huber: bool, merges: bool) -> GridTensorParams {
    let refinement = RefinementStrategyParamsBuilder::new()
        .alpha(0.0)
        .tilt_tau(0.0);
    let refinement = if huber {
        refinement.huber()
    } else {
        refinement.l2()
    }
    .build();
    GridTensorParamsBuilder::new()
        .n_iter(30)
        .max_bins(max_bins)
        .refinement_strategy(refinement)
        .split_strategy(
            SplitStrategyParamsBuilder::new()
                .best_split()
                .min_interval_samples(2)
                .complexity_penalty(if merges { 0.1 } else { 0.0 })
                .build(),
        )
        .build()
}

fn reference(n: usize, rows: Vec<usize>, rho: f64, phase: f64) -> BranchConsensus {
    let anchors = Arc::new(ConsensusAnchors::new(n, rows));
    BranchConsensus {
        weight: rho * n as f64 / anchors.rows.len() as f64,
        plus: Array1::from_iter(
            anchors
                .rows
                .iter()
                .map(|i| 1.5 + 0.3 * (*i as f64 + phase).cos()),
        ),
        minus: Array1::from_iter(
            anchors
                .rows
                .iter()
                .map(|i| 0.8 + 0.2 * (*i as f64 - phase).sin()),
        ),
        anchors: Some(anchors),
    }
}

fn direct_target(reference: &BranchConsensus, row: usize) -> Option<(f64, f64)> {
    let index = match &reference.anchors {
        Some(anchors) => anchors.rows.binary_search(&row).ok()?,
        None => row,
    };
    Some((reference.plus[index], reference.minus[index]))
}

fn direct_statistics(
    state: &FittingState,
    strategy: &RefinementStrategy,
    row: usize,
) -> PrefixStats {
    let (fp, fm) = (state.f_plus[row], state.f_minus[row]);
    let residual = state.labels[row] - fp + fm;
    let weight = strategy.weight(residual);
    let mut point = PrefixStats {
        s11: weight * fp * fp,
        s22: weight * fm * fm,
        s12: weight * fp * fm,
        t1: weight * residual * fp,
        t2: weight * residual * fm,
    };
    if let Some(reference) = &state.branch_consensus {
        if let Some((plus, minus)) = direct_target(reference, row) {
            point.s11 += reference.weight * fp * fp;
            point.s22 += reference.weight * fm * fm;
            point.t1 += reference.weight * (plus - fp) * fp;
            point.t2 += reference.weight * (fm - minus) * fm;
        }
    }
    point
}

#[track_caller]
fn close(actual: f64, expected: f64) {
    assert_eq!(actual.is_nan(), expected.is_nan());
    if expected.is_finite() {
        assert!(
            (actual - expected).abs() <= 2e-9 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    } else if !expected.is_nan() {
        assert_eq!(actual, expected);
    }
}

#[track_caller]
fn point_close(actual: PrefixStats, expected: PrefixStats) {
    for (actual, expected) in [actual.s11, actual.s22, actual.s12, actual.t1, actual.t2]
        .into_iter()
        .zip([
            expected.s11,
            expected.s22,
            expected.s12,
            expected.t1,
            expected.t2,
        ])
    {
        close(actual, expected);
    }
}

fn assert_direct_statistics(state: &FittingState, strategy: &RefinementStrategy) {
    let statistics = &state.precomputed_statistics;
    for row in 0..state.n {
        point_close(
            PrefixStats {
                s11: statistics.c_s11[row],
                s22: statistics.c_s22[row],
                s12: statistics.c_s12[row],
                t1: statistics.c_t1[row],
                t2: statistics.c_t2[row],
            },
            direct_statistics(state, strategy, row),
        );
    }
    for col in 0..state.p {
        let mut sum = PrefixStats::ZERO;
        if statistics.bin_edges[col].is_empty() {
            for (position, &row) in statistics.sorted_indices[col].iter().enumerate() {
                sum += direct_statistics(state, strategy, row);
                point_close(statistics.prefix_sums[col][position], sum);
            }
        } else {
            for bin in 0..statistics.binned_prefix_sums[col].len() {
                for row in 0..state.n {
                    if statistics.bin_indices[col][row] as usize == bin {
                        sum += direct_statistics(state, strategy, row);
                    }
                }
                point_close(statistics.binned_prefix_sums[col][bin], sum);
            }
        }
    }
}

fn assert_scores_match<'a>(actual: &FittingState<'a>, expected: &FittingState<'a>) {
    let (a, e) = (
        &actual.precomputed_statistics,
        &expected.precomputed_statistics,
    );
    for (kind, a, e) in [
        (
            "split",
            &a.error_reductions_split,
            &e.error_reductions_split,
        ),
        (
            "resplit",
            &a.error_reductions_resplit,
            &e.error_reductions_resplit,
        ),
        (
            "merge",
            &a.error_reductions_merge,
            &e.error_reductions_merge,
        ),
    ] {
        for (col, (a, e)) in a.iter().zip(e.iter()).enumerate() {
            assert_eq!(a.len(), e.len());
            for (index, (&a, &e)) in a.iter().zip(e.iter()).enumerate() {
                if a.is_finite() && e.is_finite() {
                    assert!(
                        (a - e).abs() <= 2e-9 * e.abs().max(1.0),
                        "{kind} col={col} index={index}: {a} != {e}"
                    );
                } else {
                    close(a, e);
                }
            }
        }
    }
}

fn full_refresh<'a>(
    mut state: FittingState<'a>,
    strategy: &RefinementStrategy,
    reference: BranchConsensus,
) -> FittingState<'a> {
    state.branch_consensus = Some(reference);
    state.precomputed_statistics.consensus_weight = f64::NAN;
    strategy.refresh_statistics(state)
}

fn dense_reference_refresh<'a>(
    state: &FittingState<'a>,
    strategy: &RefinementStrategy,
    reference: BranchConsensus,
) -> FittingState<'a> {
    let mut expected = full_refresh(state.clone(), strategy, reference);
    // A reference change preserves curvature. Dense linear-term rebuilding
    // isolates that update from rounding in an independent curvature rebuild.
    let (before, after) = (
        &state.precomputed_statistics,
        &mut expected.precomputed_statistics,
    );
    after.c_s11.clone_from(&before.c_s11);
    after.c_s22.clone_from(&before.c_s22);
    after.c_s12.clone_from(&before.c_s12);
    for (source, target) in before.prefix_sums.iter().zip(&mut after.prefix_sums).chain(
        before
            .binned_prefix_sums
            .iter()
            .zip(&mut after.binned_prefix_sums),
    ) {
        for (source, target) in source.iter().zip(target.iter_mut()) {
            target.s11 = source.s11;
            target.s22 = source.s22;
            target.s12 = source.s12;
        }
    }
    for col in 0..expected.p {
        let last_interval = expected.boundaries[col].len();
        RefinementStrategy::update_interval_stats_for_col_range(
            &mut expected,
            col,
            0,
            last_interval,
        );
    }
    let ranges: Vec<_> = (0..expected.p)
        .map(|col| AffectedRange {
            col,
            point_range: (0, expected.n - 1),
            interval_range: (0, 0),
        })
        .collect();
    strategy.refresh_error_reduction_caches_for_affected(expected, &ranges)
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
                if gain.is_finite()
                    && best
                        .as_ref()
                        .is_none_or(|candidate| gain > candidate.error_reduction)
                {
                    best = Some(SplitCandidate {
                        col,
                        error_reduction: gain,
                        allowed_interval_idx,
                        index,
                        update_left: state.precomputed_statistics.update_pairs_split_left[col]
                            [index],
                        update_right: state.precomputed_statistics.update_pairs_split_right[col]
                            [index],
                    });
                }
            }
        }
    }
    best.expect("fixture has a finite split candidate")
}

fn frozen_objective(
    reference: &FittingState,
    evaluated: &FittingState,
    strategy: &RefinementStrategy,
) -> f64 {
    (0..reference.n)
        .map(|row| {
            let residual = reference.labels[row] - evaluated.f_plus[row] + evaluated.f_minus[row];
            let mut loss = strategy.weight(reference.residuals[row]) * residual * residual;
            if let Some(consensus) = &reference.branch_consensus {
                if let Some((plus, minus)) = direct_target(consensus, row) {
                    loss += consensus.weight
                        * ((evaluated.f_plus[row] - plus).powi(2)
                            + (evaluated.f_minus[row] - minus).powi(2));
                }
            }
            loss
        })
        .sum()
}

#[test]
fn anchor_review_statistics_retain_every_data_row() {
    let (x, y) = data(false);
    for huber in [false, true] {
        let params = parameters(Some(16), huber, false);
        let strategy = params.refinement_strategy_params.get_refinement_strategy();
        let state = initialize_state_with_input(x.view(), y.view(), &params, None);
        let reference = reference(state.n, (0..state.n).step_by(7).collect(), 0.6, 0.0);
        close(
            reference.weight * reference.anchors.as_ref().unwrap().rows.len() as f64,
            0.6 * state.n as f64,
        );
        let state = strategy.refresh_consensus_reference(state, reference);
        assert_direct_statistics(&state, &strategy);
        assert!(state
            .precomputed_statistics
            .bin_edges
            .iter()
            .all(|bins| !bins.is_empty()));
    }
}

#[test]
fn anchor_review_reference_deltas_track_actions_and_full_rebuilds() {
    let (x, y) = data(false);
    for huber in [false, true] {
        let params = parameters(Some(16), huber, false);
        let strategy = params.refinement_strategy_params.get_refinement_strategy();
        let split = params.split_strategy_params.get_split_strategy();
        let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
        let initial = reference(state.n, (0..state.n).step_by(5).collect(), 0.3, 0.0);
        state = strategy.refresh_consensus_reference(state, initial);
        for round in 0..10 {
            let candidate = best_split(&state);
            let predicted = candidate.error_reduction;
            let before = state.clone();
            state = fitting_reducer(
                state,
                FittingAction::ApplySplit { split: candidate },
                &strategy,
                &split,
            );
            close(
                frozen_objective(&before, &before, &strategy)
                    - frozen_objective(&before, &state, &strategy),
                predicted,
            );
            let mut changed = state.branch_consensus.as_ref().unwrap().clone();
            changed
                .plus
                .mapv_inplace(|value| value + 0.01 * (round as f64 + 0.3).sin());
            changed
                .minus
                .mapv_inplace(|value| value + 0.01 * (round as f64 + 0.8).cos());
            let expected = dense_reference_refresh(&state, &strategy, changed.clone());
            let previous = state.clone();
            state = strategy.refresh_consensus_reference(state, changed);
            assert_eq!(state.f_plus, previous.f_plus);
            assert_eq!(state.f_minus, previous.f_minus);
            assert_eq!(state.residuals, previous.residuals);
            assert_eq!(
                state.precomputed_statistics.c_s11,
                previous.precomputed_statistics.c_s11
            );
            assert_eq!(
                state.precomputed_statistics.c_s22,
                previous.precomputed_statistics.c_s22
            );
            assert_direct_statistics(&state, &strategy);
            assert_scores_match(&state, &expected);
        }
    }
}

#[test]
fn anchor_review_support_and_strength_changes_refresh_curvature() {
    let (x, y) = data(false);
    for huber in [false, true] {
        let params = parameters(Some(16), huber, false);
        let strategy = params.refinement_strategy_params.get_refinement_strategy();
        let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
        let references = [
            reference(state.n, (0..state.n).step_by(4).collect(), 0.5, 0.0),
            reference(state.n, (1..state.n).step_by(4).collect(), 0.5, 0.2),
            reference(state.n, (1..state.n).step_by(4).collect(), 1.0, 0.5),
            BranchConsensus {
                weight: 4.0,
                plus: Array1::from_elem(state.n, 1.7),
                minus: Array1::from_elem(state.n, 0.9),
                anchors: None,
            },
            reference(state.n, (2..state.n).step_by(4).collect(), 1.0, 0.8),
        ];
        for reference in references {
            let expected = full_refresh(state.clone(), &strategy, reference.clone());
            state = strategy.refresh_consensus_reference(state, reference);
            assert_direct_statistics(&state, &strategy);
            assert_scores_match(&state, &expected);
        }
    }
}

#[test]
fn anchor_review_exact_and_constant_columns_have_correct_fallbacks() {
    for constant_column in [false, true] {
        let (x, y) = data(constant_column);
        for bins in [None, Some(16)] {
            let params = parameters(bins, false, false);
            let strategy = params.refinement_strategy_params.get_refinement_strategy();
            let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
            let initial = reference(state.n, (0..state.n).step_by(6).collect(), 0.7, 0.0);
            state = strategy.refresh_consensus_reference(state, initial);
            let mut changed = state.branch_consensus.as_ref().unwrap().clone();
            changed.plus.mapv_inplace(|value| value + 0.2);
            let expected = full_refresh(state.clone(), &strategy, changed.clone());
            state = strategy.refresh_consensus_reference(state, changed);
            assert_direct_statistics(&state, &strategy);
            assert_scores_match(&state, &expected);
        }
    }
}

#[test]
fn anchor_review_merge_gains_match_the_weighted_all_row_objective() {
    let (x, y) = data(false);
    for huber in [false, true] {
        for bins in [None, Some(16)] {
            let params = parameters(bins, huber, true);
            let strategy = params.refinement_strategy_params.get_refinement_strategy();
            let split: SplitStrategy = params.split_strategy_params.get_split_strategy();
            let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
            let initial = reference(state.n, (0..state.n).step_by(5).collect(), 0.4, 0.0);
            state = strategy.refresh_consensus_reference(state, initial);
            for _ in 0..4 {
                let candidate = best_split(&state);
                state = fitting_reducer(
                    state,
                    FittingAction::ApplySplit { split: candidate },
                    &strategy,
                    &split,
                );
            }
            let mut changed = state.branch_consensus.as_ref().unwrap().clone();
            changed.minus.mapv_inplace(|value| value + 0.2);
            state = strategy.refresh_consensus_reference(state, changed);
            assert_direct_statistics(&state, &strategy);
            let mut tested = 0;
            for col in 0..state.p {
                for (interval_idx, &index) in state.boundaries[col].iter().enumerate() {
                    let predicted =
                        state.precomputed_statistics.error_reductions_merge[col][interval_idx];
                    assert!(predicted.is_finite());
                    let merge = MergeCandidate {
                        col,
                        error_reduction: predicted,
                        interval_idx,
                        index,
                    };
                    let merged = fitting_reducer(
                        state.clone(),
                        FittingAction::ApplyMerge { merge },
                        &strategy,
                        &split,
                    );
                    let realized = frozen_objective(&state, &state, &strategy)
                        - frozen_objective(&state, &merged, &strategy);
                    close(realized, predicted);
                    tested += 1;
                }
            }
            assert!(tested > 0);
        }
    }
}
