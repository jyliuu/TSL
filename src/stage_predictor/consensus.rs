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
        splitting::SplitStrategy,
        state::{BranchConsensus, ConsensusAnchors, FittingState},
        GridTensor, GridTensorParams,
    },
    logging,
    stage_predictor::params::StagePredictorParams,
    FitResult,
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
    params: &StagePredictorParams,
    seeds: &[u64],
    initial_state: Option<&FittingState<'a>>,
    prepared: Option<&PreparedInput<'_>>,
    epoch: Option<usize>,
    fit_results: bool,
) -> Vec<(Option<FitResult>, GridTensor)> {
    let StagePredictorParams {
        ref tg_params,
        consensus,
        consensus_interval,
        consensus_anchors,
        ..
    } = *params;
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
    fit_members(
        &mut members,
        tg_params,
        consensus,
        consensus_interval,
        anchors.as_ref(),
        epoch,
    );

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

fn fit_members<'a>(
    members: &mut [Member<'a>],
    tg_params: &GridTensorParams,
    consensus: f64,
    consensus_interval: usize,
    anchors: Option<&Arc<ConsensusAnchors>>,
    epoch: Option<usize>,
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
                || current.structure_is_final(tg_params.n_iter)
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
                current.sampled_split_scoring = matches!(split, SplitStrategy::Random { .. });
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
                    if current.terminated || current.structure_is_final(tg_params.n_iter) {
                        break;
                    }
                    // Intermediate proposals consume scores for the same frozen reference;
                    // the batch's last action leaves them to the next round's refresh.
                    let refresh = if offset + 1 == batch_size {
                        CandidateRefresh::Deferred
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
mod tests;
