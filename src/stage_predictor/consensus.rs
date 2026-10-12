//! Lockstep bag fitting toward shared branch functions.
//!
//! Every grid takes a bounded batch of structural actions per round. Grid `b`'s local
//! objective gains the proximal term `ρ (‖f₊ − c₊‖² + ‖f₋ − c₋‖²)` on the training rows,
//! where `c±` are the mean branches of the other grids. The term encourages agreement
//! of the positive branches on observed training rows. Factor identification also depends
//! on feature support; branch agreement on these rows does not establish unique factors.
//! Agreement on `f = f₊ − f₋` alone leaves common changes of the two branches unpenalized.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::{rngs::StdRng, seq::index::sample, SeedableRng};
use std::sync::Arc;

#[cfg(feature = "use-rayon")]
use rayon::prelude::*;

use crate::{
    grid_tensor::{
        fit::{
            finish_fit, finish_grid, fit_step_with_candidate_refresh, initialize_state_with_input,
            max_iterations,
        },
        prepared_input::PreparedInput,
        reducer::CandidateRefresh,
        state::{BranchConsensus, ConsensusAnchors, FittingState},
        GridTensor, GridTensorParams,
    },
    logging, FitResult,
};

struct Member<'a> {
    tree_id: usize,
    rng: StdRng,
    state: Option<FittingState<'a>>,
    next_iteration: usize,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fit_consensus_bag<'a>(
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, f64>,
    tg_params: &GridTensorParams,
    consensus: f64,
    consensus_interval: usize,
    consensus_anchors: Option<usize>,
    seeds: &[u64],
    initial_state: Option<&FittingState<'a>>,
    prepared: Option<&PreparedInput<'_>>,
    epoch: Option<usize>,
    fit_results: bool,
) -> Vec<(Option<FitResult>, GridTensor)> {
    assert!(
        consensus.is_finite() && consensus >= 0.0,
        "consensus must be finite and nonnegative"
    );
    // Each member's training evaluation is kept when a caller asks for it or when logging
    // reports it; otherwise the member only yields its identified grid.
    let fit_results = fit_results || logging::is_logging_enabled();
    // Member setup and teardown have no observable order while logging is disabled, so
    // a shared initial state is cloned and the fitted grids are converted on the pool.
    let shared_setup = initial_state.filter(|_| !logging::is_logging_enabled());
    let member = |tree_id: usize, seed: u64, state: FittingState<'a>| Member {
        tree_id,
        rng: StdRng::seed_from_u64(seed),
        next_iteration: 0,
        state: Some(state),
    };
    let mut members: Vec<Member<'a>> = match shared_setup {
        #[cfg(feature = "use-rayon")]
        Some(initial_state) => seeds
            .par_iter()
            .enumerate()
            .map(|(tree_id, &seed)| {
                member(
                    tree_id,
                    seed,
                    in_context(tree_id, epoch, || initial_state.clone()),
                )
            })
            .collect(),
        _ => seeds
            .iter()
            .enumerate()
            .map(|(tree_id, &seed)| {
                let state = in_context(tree_id, epoch, || match initial_state {
                    Some(initial_state) => initial_state.clone(),
                    None => initialize_state_with_input(x, y, tg_params, prepared),
                });
                member(tree_id, seed, state)
            })
            .collect(),
    };

    let anchors = sample_consensus_anchors(
        x.nrows(),
        consensus_anchors,
        seeds.first().copied().unwrap_or(0),
    );
    if anchors.is_some() {
        fit_members_with_split_scoring(
            &mut members,
            tg_params,
            consensus,
            consensus_interval,
            anchors.as_ref(),
            epoch,
            CandidateRefresh::Deferred,
            true,
        );
    } else {
        fit_members(
            &mut members,
            tg_params,
            consensus,
            consensus_interval,
            epoch,
        );
    }

    let finish = |member: Member<'a>| {
        let mut state = member.state.expect("member state");
        state.branch_consensus = None;
        in_context(member.tree_id, epoch, || {
            if !fit_results {
                return (None, finish_grid(state));
            }
            let (fit_res, grid) = finish_fit(state);
            logging::log_grid_error_fitted(fit_res.err, logging::GridErrorVariant::Train);
            (Some(fit_res), grid)
        })
    };
    #[cfg(feature = "use-rayon")]
    if shared_setup.is_some() {
        return members.into_par_iter().map(finish).collect();
    }
    members.into_iter().map(finish).collect()
}

/// Sum every member's full-row branches over parallel row chunks.
///
/// Each row total starts from zero and adds the members in order, so the result is
/// identical to the sequential elementwise sum. Returns `false`, leaving the totals
/// untouched, unless every branch is a contiguous array of the totals' length.
fn accumulate_full_row_totals(
    members: &[Member<'_>],
    total_plus: &mut Array1<f64>,
    total_minus: &mut Array1<f64>,
) -> bool {
    let n = total_plus.len();
    if total_minus.len() != n {
        return false;
    }
    let Some(branches) = members
        .iter()
        .map(|member| {
            let state = member.state.as_ref().expect("member state");
            match (state.f_plus.as_slice(), state.f_minus.as_slice()) {
                (Some(plus), Some(minus)) if plus.len() == n && minus.len() == n => {
                    Some((plus, minus))
                }
                _ => None,
            }
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let (Some(total_plus), Some(total_minus)) =
        (total_plus.as_slice_mut(), total_minus.as_slice_mut())
    else {
        return false;
    };
    let chunk = (n / 32).max(512);
    let accumulate = |start: usize, plus_chunk: &mut [f64], minus_chunk: &mut [f64]| {
        plus_chunk.fill(0.0);
        minus_chunk.fill(0.0);
        let end = start + plus_chunk.len();
        for &(plus, minus) in &branches {
            for (total, &value) in plus_chunk.iter_mut().zip(&plus[start..end]) {
                *total += value;
            }
            for (total, &value) in minus_chunk.iter_mut().zip(&minus[start..end]) {
                *total += value;
            }
        }
    };
    #[cfg(feature = "use-rayon")]
    total_plus
        .par_chunks_mut(chunk)
        .zip(total_minus.par_chunks_mut(chunk))
        .enumerate()
        .for_each(|(index, (plus_chunk, minus_chunk))| {
            accumulate(index * chunk, plus_chunk, minus_chunk)
        });
    #[cfg(not(feature = "use-rayon"))]
    total_plus
        .chunks_mut(chunk)
        .zip(total_minus.chunks_mut(chunk))
        .enumerate()
        .for_each(|(index, (plus_chunk, minus_chunk))| {
            accumulate(index * chunk, plus_chunk, minus_chunk)
        });
    true
}

fn sample_consensus_anchors(
    n: usize,
    count: Option<usize>,
    seed: u64,
) -> Option<Arc<ConsensusAnchors>> {
    let count = count?;
    assert!(count > 0, "consensus_anchors must be positive");
    if count >= n {
        return None;
    }
    // A separate stream leaves every bag's proposal RNG unchanged.
    let mut rng = StdRng::seed_from_u64(seed ^ 0x414E_4348_4F52_535F);
    let mut rows = sample(&mut rng, n, count).into_vec();
    rows.sort_unstable();
    Some(Arc::new(ConsensusAnchors::new(n, rows)))
}

fn fit_members<'a>(
    members: &mut [Member<'a>],
    tg_params: &GridTensorParams,
    consensus: f64,
    consensus_interval: usize,
    epoch: Option<usize>,
) {
    fit_members_with_split_scoring(
        members,
        tg_params,
        consensus,
        consensus_interval,
        None,
        epoch,
        CandidateRefresh::Deferred,
        true,
    );
}

/// Reuse full-row reference storage with matching contiguous branch inputs.
fn reuse_full_row_reference(
    current: &mut FittingState<'_>,
    total_plus: &ArrayView1<'_, f64>,
    total_minus: &ArrayView1<'_, f64>,
    others: f64,
    consensus: f64,
) -> bool {
    let Some(reference) = current.branch_consensus.as_mut() else {
        return false;
    };
    if reference.anchors.is_some()
        || [
            total_plus.len(),
            total_minus.len(),
            current.f_plus.len(),
            current.f_minus.len(),
            reference.plus.len(),
            reference.minus.len(),
        ]
        .into_iter()
        .any(|len| len != current.n)
    {
        return false;
    }
    let (
        Some(total_plus),
        Some(total_minus),
        Some(current_plus),
        Some(current_minus),
        Some(reference_plus),
        Some(reference_minus),
    ) = (
        total_plus.as_slice(),
        total_minus.as_slice(),
        current.f_plus.as_slice(),
        current.f_minus.as_slice(),
        reference.plus.as_slice_mut(),
        reference.minus.as_slice_mut(),
    )
    else {
        return false;
    };
    for row in 0..current.n {
        let plus_difference = total_plus[row] - current_plus[row];
        let minus_difference = total_minus[row] - current_minus[row];
        reference_plus[row] = plus_difference / others;
        reference_minus[row] = minus_difference / others;
    }
    reference.weight = consensus;
    true
}

fn fit_members_with_split_scoring<'a>(
    members: &mut [Member<'a>],
    tg_params: &GridTensorParams,
    consensus: f64,
    consensus_interval: usize,
    anchors: Option<&Arc<ConsensusAnchors>>,
    epoch: Option<usize>,
    candidate_refresh: CandidateRefresh,
    sampled_split_scoring: bool,
) {
    assert!(
        consensus_interval > 0,
        "consensus_interval must be positive"
    );
    let refinement = tg_params
        .refinement_strategy_params
        .get_refinement_strategy();
    let split = tg_params.split_strategy_params.get_split_strategy();
    let others = (members.len().max(2) - 1) as f64;
    let n = members
        .first()
        .map_or(0, |member| member.state.as_ref().expect("member state").n);
    let reference_len = anchors.map_or(n, |anchors| anchors.rows.len());
    let mut total_plus = Array1::<f64>::zeros(reference_len);
    let mut total_minus = Array1::<f64>::zeros(reference_len);
    let iteration_budget = max_iterations(tg_params.n_iter);
    for _ in 0..iteration_budget {
        if anchors.is_some()
            || !accumulate_full_row_totals(members, &mut total_plus, &mut total_minus)
        {
            total_plus.fill(0.0);
            total_minus.fill(0.0);
            for member in members.iter() {
                let state = member.state.as_ref().expect("member state");
                if let Some(anchors) = anchors {
                    for (index, &row) in anchors.rows.iter().enumerate() {
                        total_plus[index] += state.f_plus[row];
                        total_minus[index] += state.f_minus[row];
                    }
                } else {
                    total_plus += &state.f_plus;
                    total_minus += &state.f_minus;
                }
            }
        }
        let (total_plus, total_minus) = (total_plus.view(), total_minus.view());
        let advance = |member: &mut Member<'a>| {
            let current = member.state.as_ref().expect("member state");
            if member.next_iteration >= iteration_budget
                || current.terminated
                || (!current.split_strategy_state.merge_enabled
                    && current.loop_state.fineness >= tg_params.n_iter)
            {
                return false;
            }
            let Member {
                tree_id,
                rng,
                state,
                next_iteration,
            } = member;
            in_context(*tree_id, epoch, || {
                let mut current = state.take().expect("member state");
                current.sampled_split_scoring = sampled_split_scoring
                    && matches!(
                        split,
                        crate::grid_tensor::splitting::SplitStrategy::Random { .. }
                    );
                let mut current = if let Some(anchors) = anchors {
                    let reference = BranchConsensus {
                        weight: consensus * (n as f64 / reference_len as f64),
                        plus: Array1::from_shape_fn(reference_len, |index| {
                            (total_plus[index] - current.f_plus[anchors.rows[index]]) / others
                        }),
                        minus: Array1::from_shape_fn(reference_len, |index| {
                            (total_minus[index] - current.f_minus[anchors.rows[index]]) / others
                        }),
                        anchors: Some(anchors.clone()),
                    };
                    refinement.refresh_consensus_reference(current, reference)
                } else {
                    if !reuse_full_row_reference(
                        &mut current,
                        &total_plus,
                        &total_minus,
                        others,
                        consensus,
                    ) {
                        current.branch_consensus = Some(BranchConsensus {
                            weight: consensus,
                            plus: (&total_plus - &current.f_plus) / others,
                            minus: (&total_minus - &current.f_minus) / others,
                            anchors: None,
                        });
                    }
                    refinement.refresh_statistics(current)
                };
                let batch_size = consensus_interval.min(iteration_budget - *next_iteration);
                let mut advanced = false;
                for offset in 0..batch_size {
                    if current.terminated
                        || (!current.split_strategy_state.merge_enabled
                            && current.loop_state.fineness >= tg_params.n_iter)
                    {
                        break;
                    }
                    // Intermediate proposals consume scores for the same frozen reference.
                    let refresh = if offset + 1 == batch_size {
                        candidate_refresh
                    } else {
                        CandidateRefresh::Eager
                    };
                    let iter = *next_iteration;
                    *next_iteration += 1;
                    let step_advanced;
                    (current, step_advanced) = fit_step_with_candidate_refresh(
                        current,
                        iter,
                        tg_params.n_iter,
                        &refinement,
                        &split,
                        rng,
                        refresh,
                    );
                    advanced |= step_advanced;
                    if !step_advanced {
                        break;
                    }
                }
                *state = Some(current);
                advanced
            })
        };
        // Every member sees the round's frozen reference before progress is combined.
        #[cfg(feature = "use-rayon")]
        let advanced = members
            .par_iter_mut()
            .map(advance)
            .reduce(|| false, |left, right| left | right);
        #[cfg(not(feature = "use-rayon"))]
        let advanced = members
            .iter_mut()
            .map(advance)
            .fold(false, |left, right| left | right);
        if !advanced {
            break;
        }
    }
}

fn in_context<R>(tree_id: usize, epoch: Option<usize>, f: impl FnOnce() -> R) -> R {
    logging::with_tree_id(tree_id, || match epoch {
        Some(epoch) => logging::with_epoch(epoch, f),
        None => f(),
    })
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    use super::{
        fit_consensus_bag, fit_members, fit_members_with_split_scoring, max_iterations,
        sample_consensus_anchors, Member,
    };
    use rand::{rngs::StdRng, SeedableRng};

    use crate::{
        grid_tensor::{
            fit::{finish_fit, fit_step, initialize_state_with_input},
            params::{
                GridTensorParamsBuilder, RefinementStrategyParamsBuilder,
                SplitStrategyParamsBuilder,
            },
            reducer::CandidateRefresh,
            state::{BranchConsensus, FittingState},
            GridTensor, GridTensorParams,
        },
        stage_predictor::{
            extract_two_tensor_predictions_unscaled, fit_grid_bag,
            params::StagePredictorParamsBuilder,
        },
    };

    fn data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((600, 2), |(i, j)| {
            ((i * (7 * j + 3) + 11 * j) % 97) as f64 / 12.0 - 4.0
        });
        let y = Array1::from_shape_fn(600, |i| {
            (x[[i, 0]].sin() * x[[i, 1]].cos()).exp() + x[[i, 0]]
        });
        (x, y)
    }

    fn bag(x: ArrayView2<f64>, y: &Array1<f64>, consensus: f64) -> Vec<GridTensor> {
        let params = StagePredictorParamsBuilder::new()
            .n_trees(12)
            .n_iter(15)
            .max_bins(Some(32))
            .consensus(consensus)
            .build();
        let mut rng = StdRng::seed_from_u64(3);
        fit_grid_bag(x, y.view(), &params, &mut rng)
            .into_iter()
            .map(|(fit, grid)| {
                let predictions = grid.predict(x);
                assert_eq!(fit.y_hat, predictions);
                assert_eq!(fit.residuals, y - &predictions);
                grid
            })
            .collect()
    }

    /// Mean pairwise relative squared distance between the grids' branches.
    fn branch_disagreement(grids: &[GridTensor], x: ArrayView2<f64>) -> f64 {
        let branches: Vec<_> = grids
            .iter()
            .map(|g| extract_two_tensor_predictions_unscaled(g, x))
            .collect();
        let relative = |a: &Array1<f64>, b: &Array1<f64>| {
            let scale =
                (a.mapv(|v| v * v).mean().unwrap() + b.mapv(|v| v * v).mean().unwrap()) / 2.0;
            (a - b).mapv(|v| v * v).mean().unwrap() / scale
        };
        let mut total = 0.0;
        let mut pairs = 0.0;
        for i in 0..branches.len() {
            for j in 0..i {
                total += relative(&branches[i].0, &branches[j].0)
                    + relative(&branches[i].1, &branches[j].1);
                pairs += 2.0;
            }
        }
        total / pairs
    }

    fn members_from_state<'a>(initial: &FittingState<'a>) -> Vec<Member<'a>> {
        (0..4)
            .map(|tree_id| {
                #[allow(unused_mut)]
                let mut state = initial.clone();
                #[cfg(feature = "evo-logging")]
                {
                    state.logging_state =
                        Some(crate::grid_tensor::logging_helpers::LoggingState::new());
                }
                Member {
                    tree_id,
                    next_iteration: 0,
                    rng: StdRng::seed_from_u64(17 + tree_id as u64),
                    state: Some(state),
                }
            })
            .collect()
    }

    fn member_params(huber: bool, max_bins: Option<u16>, penalty: f64) -> GridTensorParams {
        let refinement = RefinementStrategyParamsBuilder::new()
            .alpha(0.1)
            .tilt_tau(0.01)
            .tilt_rho(0.001);
        GridTensorParamsBuilder::new()
            .n_iter(8)
            .max_bins(max_bins)
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
                    .complexity_penalty(penalty)
                    .build(),
            )
            .build()
    }

    #[test]
    #[should_panic(expected = "consensus_anchors must be positive")]
    fn training_anchor_count_must_be_positive() {
        sample_consensus_anchors(100, Some(0), 42);
    }

    #[test]
    fn all_training_anchors_match_the_full_row_model_exactly() {
        let (x, y) = data();
        for interval in [1, 4] {
            for huber in [false, true] {
                let params = member_params(huber, Some(16), 0.0);
                let initial = initialize_state_with_input(x.view(), y.view(), &params, None);
                let fit = |anchors| {
                    fit_consensus_bag(
                        x.view(),
                        y.view(),
                        &params,
                        0.3,
                        interval,
                        anchors,
                        &[17, 18, 19, 20],
                        Some(&initial),
                        None,
                        None,
                        true,
                    )
                };
                let baseline = fit(None);
                for count in [x.nrows(), x.nrows() + 5] {
                    let requested = fit(Some(count));
                    for ((a_fit, a_grid), (b_fit, b_grid)) in baseline.iter().zip(requested.iter())
                    {
                        assert_eq!(
                            bincode::serialize(a_grid).unwrap(),
                            bincode::serialize(b_grid).unwrap()
                        );
                        let (a_fit, b_fit) = (a_fit.as_ref().unwrap(), b_fit.as_ref().unwrap());
                        assert_eq!(a_fit.y_hat, b_fit.y_hat);
                        assert_eq!(a_fit.residuals, b_fit.residuals);
                        assert_eq!(a_fit.err.to_bits(), b_fit.err.to_bits());
                    }
                }
            }
        }
    }

    fn assert_member_budget_and_logging(member: &Member<'_>, params: &GridTensorParams) {
        let state = member.state.as_ref().unwrap();
        let actions = state.loop_state.split_count
            + state.loop_state.resplit_count
            + state.loop_state.merge_count;
        assert!(actions <= member.next_iteration);
        assert!(member.next_iteration <= max_iterations(params.n_iter));
        assert!(state.loop_state.fineness <= params.n_iter);
        assert_eq!(state.iteration + 1, member.next_iteration);
        assert!(state
            .f_plus
            .iter()
            .chain(&state.f_minus)
            .all(|value| value.is_finite() && *value >= 0.0));
        let mut finished = state.clone();
        finished.branch_consensus = None;
        let (_, grid) = finish_fit(finished);
        assert!(grid.lambda_plus.is_finite() && grid.lambda_plus >= 0.0);
        assert!(grid.lambda_minus.is_finite() && grid.lambda_minus >= 0.0);
        assert!(grid
            .backbone_values
            .iter()
            .flatten()
            .all(|value| value.is_finite() && *value > 0.0));
        assert!(grid
            .tilt_values
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
        let (plus, minus) = extract_two_tensor_predictions_unscaled(&grid, state.x);
        for (&cached, &rebuilt) in state
            .f_plus
            .iter()
            .chain(&state.f_minus)
            .zip(plus.iter().chain(&minus))
        {
            assert!((cached - rebuilt).abs() <= 1e-10 * (1.0 + cached.abs()));
        }
        #[cfg(feature = "evo-logging")]
        {
            use crate::grid_tensor::logging_helpers::RawLoggingEvent;
            let logging = state.logging_state.as_ref().unwrap();
            let iterations: Vec<_> = logging
                .events
                .iter()
                .filter_map(|event| match event {
                    RawLoggingEvent::Split(event) => Some(event.iter_no),
                    _ => None,
                })
                .collect();
            assert_eq!(iterations.len(), actions);
            assert!(iterations.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(iterations.iter().all(|&iter| iter < member.next_iteration));
            assert_eq!(
                iterations,
                logging
                    .f_component_stats
                    .iter()
                    .map(|stats| stats.0)
                    .collect::<Vec<_>>()
            );
            let snapshot_iterations: Vec<_> = logging
                .events
                .iter()
                .filter_map(|event| match event {
                    RawLoggingEvent::Component(snapshot) => Some(snapshot.iter_no),
                    _ => None,
                })
                .collect();
            assert_eq!(
                snapshot_iterations,
                iterations
                    .iter()
                    .flat_map(|&iter| std::iter::repeat_n(iter, state.p))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn consensus_members_respect_budgets_and_branch_invariants() {
        let (x, y) = data();
        let anchors = sample_consensus_anchors(x.nrows(), Some(17), 17).unwrap();
        let mut reached_attempt_cap = false;
        for interval in [1, 4, usize::MAX] {
            for anchors in [None, Some(&anchors)] {
                for huber in [false, true] {
                    for (max_bins, penalty) in [(None, 0.0), (Some(16), 0.0), (Some(16), 0.001)] {
                        let params = member_params(huber, max_bins, penalty);
                        let initial =
                            initialize_state_with_input(x.view(), y.view(), &params, None);
                        let mut members = members_from_state(&initial);
                        fit_members_with_split_scoring(
                            &mut members,
                            &params,
                            0.3,
                            interval,
                            anchors,
                            None,
                            CandidateRefresh::Deferred,
                            true,
                        );
                        for member in &members {
                            assert_member_budget_and_logging(member, &params);
                            reached_attempt_cap |=
                                member.next_iteration == max_iterations(params.n_iter);
                        }
                    }
                }
            }
        }
        assert!(
            reached_attempt_cap,
            "fixture must exercise the per-member attempt limit"
        );
    }

    #[test]
    #[should_panic(expected = "consensus_interval must be positive")]
    fn consensus_interval_must_be_positive() {
        fit_members(&mut [], &lifecycle_params(0), 0.3, 0, None);
    }

    #[test]
    fn consensus_bags_are_reproducible() {
        let (x, y) = data();
        let first = bag(x.view(), &y, 0.3);
        let second = bag(x.view(), &y, 0.3);
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.splits, b.splits);
            assert_eq!(a.backbone_values, b.backbone_values);
            assert_eq!(a.tilt_values, b.tilt_values);
        }
    }

    #[test]
    fn consensus_pulls_grid_branches_together() {
        let (x, y) = data();
        let independent = branch_disagreement(&bag(x.view(), &y, 0.0), x.view());
        let pulled = branch_disagreement(&bag(x.view(), &y, 0.3), x.view());
        assert!(
            pulled < 0.25 * independent,
            "independent {independent}, consensus {pulled}"
        );
    }

    fn lifecycle_params(n_iter: usize) -> GridTensorParams {
        GridTensorParamsBuilder::new()
            .n_iter(n_iter)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .best_split()
                    .min_interval_samples(2)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(0.0)
                    .tilt_tau(0.0)
                    .tilt_rho(0.0)
                    .build(),
            )
            .build()
    }

    fn constant_members<'a>(
        x: ArrayView2<'a, f64>,
        y: ArrayView1<'a, f64>,
        params: &GridTensorParams,
        branches: &[(f64, f64)],
    ) -> Vec<Member<'a>> {
        let refinement = params.refinement_strategy_params.get_refinement_strategy();
        let total_plus: f64 = branches.iter().map(|pair| pair.0).sum();
        let total_minus: f64 = branches.iter().map(|pair| pair.1).sum();
        let others = (branches.len() - 1) as f64;
        branches
            .iter()
            .enumerate()
            .map(|(tree_id, &(plus, minus))| {
                let mut state = initialize_state_with_input(x, y, params, None);
                state.lambda_plus = plus;
                state.lambda_minus = minus;
                state.f_plus.fill(plus);
                state.f_minus.fill(minus);
                state.f.fill(plus - minus);
                state.y_hat.assign(&state.f);
                state.residuals.assign(&(&y - &state.f));
                state.r_tilde.assign(&state.residuals);
                state.current_error = state.residuals.iter().map(|r| r * r).sum();
                state.branch_consensus = Some(BranchConsensus {
                    anchors: None,
                    weight: 1.0,
                    plus: Array1::from_elem(y.len(), (total_plus - plus) / others),
                    minus: Array1::from_elem(y.len(), (total_minus - minus) / others),
                });
                Member {
                    tree_id,
                    next_iteration: 0,
                    rng: StdRng::seed_from_u64(tree_id as u64 + 1),
                    state: Some(refinement.refresh_statistics(state)),
                }
            })
            .collect()
    }

    #[test]
    fn consensus_retries_a_stalled_member_after_peer_branches_move() {
        let x = Array2::from_shape_fn((16, 1), |(row, _)| row as f64);
        let y = Array1::zeros(x.nrows());
        let params = lifecycle_params(2);
        let refinement = params.refinement_strategy_params.get_refinement_strategy();
        let split = params.split_strategy_params.get_split_strategy();
        let members = constant_members(x.view(), y.view(), &params, &[(3.0, 2.0), (4.0, 1.0)]);

        // A=(3,2) is stationary against B=(4,1), but B improves against A.
        let mut rng = members[0].rng.clone();
        let (stalled, advanced) = fit_step(
            members[0].state.as_ref().unwrap().clone(),
            0,
            params.n_iter,
            &refinement,
            &split,
            &mut rng,
        );
        assert!(!advanced);
        assert_eq!(stalled.loop_state.fineness, 0);
        let mut rng = members[1].rng.clone();
        let (peer, advanced) = fit_step(
            members[1].state.as_ref().unwrap().clone(),
            0,
            params.n_iter,
            &refinement,
            &split,
            &mut rng,
        );
        assert!(advanced);
        for (&plus, &minus) in peer.f_plus.iter().zip(peer.f_minus.iter()) {
            assert!((plus - 8.0 / 3.0).abs() < 1e-12);
            assert!((minus - 7.0 / 3.0).abs() < 1e-12);
        }

        let mut retried = stalled;
        retried.branch_consensus = Some(BranchConsensus {
            anchors: None,
            weight: 1.0,
            plus: peer.f_plus.clone(),
            minus: peer.f_minus.clone(),
        });
        let retried = refinement.refresh_statistics(retried);
        let (retried, advanced) =
            fit_step(retried, 1, params.n_iter, &refinement, &split, &mut rng);
        assert!(advanced);
        for (&plus, &minus) in retried.f_plus.iter().zip(retried.f_minus.iter()) {
            assert!((plus - 23.0 / 9.0).abs() < 1e-12);
            assert!((minus - 22.0 / 9.0).abs() < 1e-12);
        }

        for interval in [1, 2, usize::MAX] {
            let mut members =
                constant_members(x.view(), y.view(), &params, &[(3.0, 2.0), (4.0, 1.0)]);
            fit_members(&mut members, &params, 1.0, interval, None);
            for member in members {
                let mut state = member.state.unwrap();
                assert!(
                    state.loop_state.fineness > 0,
                    "member {} must advance",
                    member.tree_id
                );
                assert!(state.loop_state.fineness <= params.n_iter);
                let actions = state.loop_state.split_count
                    + state.loop_state.resplit_count
                    + state.loop_state.merge_count;
                assert!(actions <= super::max_iterations(params.n_iter));
                assert!(state.iteration < super::max_iterations(params.n_iter));
                let cached_plus = state.f_plus.clone();
                let cached_minus = state.f_minus.clone();
                state.branch_consensus = None;
                let (_, grid) = finish_fit(state);
                let (plus, minus) = extract_two_tensor_predictions_unscaled(&grid, x.view());
                for ((&actual_plus, &actual_minus), (&expected_plus, &expected_minus)) in plus
                    .iter()
                    .zip(minus.iter())
                    .zip(cached_plus.iter().zip(cached_minus.iter()))
                {
                    assert!(actual_plus.is_finite() && actual_plus > 0.0);
                    assert!(actual_minus.is_finite() && actual_minus > 0.0);
                    assert!((actual_plus - expected_plus).abs() < 1e-12);
                    assert!((actual_minus - expected_minus).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn consensus_stops_after_a_whole_round_without_progress() {
        let x = Array2::from_shape_fn((16, 1), |(row, _)| row as f64);
        let y = Array1::zeros(x.nrows());
        let params = lifecycle_params(5);
        for interval in [1, 3, usize::MAX] {
            let mut members =
                constant_members(x.view(), y.view(), &params, &[(2.0, 2.0), (2.0, 2.0)]);
            fit_members(&mut members, &params, 1.0, interval, None);
            for member in members {
                assert_eq!(member.next_iteration, 1);
                let state = member.state.unwrap();
                assert_eq!(state.iteration, 0);
                assert_eq!(state.loop_state.fineness, 0);
                assert!(state.f.iter().all(|&prediction| prediction == 0.0));
            }
        }
    }

    #[test]
    fn consensus_keeps_exhausted_and_terminated_members_fixed() {
        let x = Array2::from_shape_fn((16, 1), |(row, _)| row as f64);
        let y = Array1::zeros(x.nrows());
        let params = lifecycle_params(2);
        let mut members = constant_members(x.view(), y.view(), &params, &[(3.0, 2.0), (4.0, 1.0)]);
        members[0].state.as_mut().unwrap().loop_state.fineness = params.n_iter;
        members[1].state.as_mut().unwrap().terminated = true;
        for member in &mut members {
            member.state.as_mut().unwrap().iteration = 7;
        }
        fit_members(&mut members, &params, 1.0, 1, None);
        for (member, (plus, minus)) in members.into_iter().zip([(3.0, 2.0), (4.0, 1.0)]) {
            let state = member.state.unwrap();
            assert_eq!(state.iteration, 7);
            assert_eq!(state.loop_state.split_count, 0);
            assert!(state.f_plus.iter().all(|&value| value == plus));
            assert!(state.f_minus.iter().all(|&value| value == minus));
        }
    }
}
