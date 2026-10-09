//! Lockstep bag fitting toward shared branch functions.
//!
//! Every grid takes one structural action per round. Before each round, grid `b`'s local
//! objective gains the proximal term `ρ (‖f₊ − c₊‖² + ‖f₋ − c₋‖²)` on the training rows,
//! where `c±` are the mean branches of the other grids. The term encourages agreement
//! of the positive branches on observed training rows. Factor identification also depends
//! on feature support; branch agreement on these rows does not establish unique factors.
//! Agreement on `f = f₊ − f₋` alone leaves common changes of the two branches unpenalized.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::{rngs::StdRng, SeedableRng};

#[cfg(feature = "use-rayon")]
use rayon::prelude::*;

use crate::{
    grid_tensor::{
        fit::{
            finish_fit, fit_step_with_candidate_refresh, initialize_state_with_input, max_iterations,
        },
        prepared_input::PreparedInput,
        reducer::CandidateRefresh,
        state::{BranchConsensus, FittingState},
        GridTensor, GridTensorParams,
    },
    logging, FitResult,
};

struct Member<'a> {
    tree_id: usize,
    rng: StdRng,
    state: Option<FittingState<'a>>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fit_consensus_bag<'a>(
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, f64>,
    tg_params: &GridTensorParams,
    consensus: f64,
    seeds: &[u64],
    initial_state: Option<&FittingState<'a>>,
    prepared: Option<&PreparedInput<'_>>,
    epoch: Option<usize>,
) -> Vec<(FitResult, GridTensor)> {
    assert!(
        consensus.is_finite() && consensus >= 0.0,
        "consensus must be finite and nonnegative"
    );
    let mut members: Vec<Member<'a>> = seeds
        .iter()
        .enumerate()
        .map(|(tree_id, &seed)| Member {
            tree_id,
            rng: StdRng::seed_from_u64(seed),
            state: Some(in_context(tree_id, epoch, || match initial_state {
                Some(initial_state) => initial_state.clone(),
                None => initialize_state_with_input(x, y, tg_params, prepared),
            })),
        })
        .collect();

    fit_members(&mut members, tg_params, consensus, epoch);

    members
        .into_iter()
        .map(|member| {
            let mut state = member.state.expect("member state");
            state.branch_consensus = None;
            in_context(member.tree_id, epoch, || {
                let (fit_res, grid) = finish_fit(state);
                logging::log_grid_error_fitted(fit_res.err, logging::GridErrorVariant::Train);
                (fit_res, grid)
            })
        })
        .collect()
}

fn fit_members<'a>(
    members: &mut [Member<'a>],
    tg_params: &GridTensorParams,
    consensus: f64,
    epoch: Option<usize>,
) {
    fit_members_with_candidate_refresh(
        members,
        tg_params,
        consensus,
        epoch,
        CandidateRefresh::Deferred,
    );
}

fn fit_members_with_candidate_refresh<'a>(
    members: &mut [Member<'a>],
    tg_params: &GridTensorParams,
    consensus: f64,
    epoch: Option<usize>,
    candidate_refresh: CandidateRefresh,
) {
    let refinement = tg_params
        .refinement_strategy_params
        .get_refinement_strategy();
    let split = tg_params.split_strategy_params.get_split_strategy();
    let others = (members.len().max(2) - 1) as f64;
    let n = members
        .first()
        .map_or(0, |member| member.state.as_ref().expect("member state").n);
    let mut total_plus = Array1::<f64>::zeros(n);
    let mut total_minus = Array1::<f64>::zeros(n);
    for iter in 0..max_iterations(tg_params.n_iter) {
        total_plus.fill(0.0);
        total_minus.fill(0.0);
        for member in members.iter() {
            let state = member.state.as_ref().expect("member state");
            total_plus += &state.f_plus;
            total_minus += &state.f_minus;
        }
        let (total_plus, total_minus) = (total_plus.view(), total_minus.view());
        let advance = |member: &mut Member<'a>| {
            let current = member.state.as_ref().expect("member state");
            if current.terminated
                || (!current.split_strategy_state.merge_enabled
                    && current.loop_state.fineness >= tg_params.n_iter)
            {
                return false;
            }
            let Member {
                tree_id,
                rng,
                state,
            } = member;
            in_context(*tree_id, epoch, || {
                let mut current = state.take().expect("member state");
                current.branch_consensus = Some(BranchConsensus {
                    weight: consensus,
                    plus: (&total_plus - &current.f_plus) / others,
                    minus: (&total_minus - &current.f_minus) / others,
                });
                let current = refinement.refresh_statistics(current);
                // References change each round, so every proposal uses a full refresh.
                let (current, advanced) = fit_step_with_candidate_refresh(
                    current,
                    iter,
                    tg_params.n_iter,
                    &refinement,
                    &split,
                    rng,
                    candidate_refresh,
                );
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

    use super::{fit_members, fit_members_with_candidate_refresh, Member};
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    use crate::{
        grid_tensor::{
            action::FittingAction,
            fit::{finish_fit, fit_step, initialize_state_with_input},
            params::{
                GridTensorParamsBuilder, RefinementStrategyParamsBuilder,
                SplitStrategyParamsBuilder,
            },
            reducer::{fitting_reducer, CandidateRefresh},
            splitting::SplitCandidate,
            state::{BranchConsensus, FittingState},
            GridTensor, GridTensorParams,
        },
        stage_predictor::{
            aggregate_bagged_two_tensor, extract_two_tensor_predictions_unscaled, fit_grid_bag,
            params::StagePredictorParamsBuilder, Aggregation, StagePredictor,
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
                    rng: StdRng::seed_from_u64(17 + tree_id as u64),
                    state: Some(state),
                }
            })
            .collect()
    }

    fn assert_candidate_refresh_parity(
        initial: &FittingState<'_>,
        params: &GridTensorParams,
    ) -> [usize; 3] {
        let mut eager = members_from_state(initial);
        let mut deferred = members_from_state(initial);
        fit_members_with_candidate_refresh(
            &mut eager,
            params,
            0.3,
            None,
            CandidateRefresh::Eager,
        );
        fit_members_with_candidate_refresh(
            &mut deferred,
            params,
            0.3,
            None,
            CandidateRefresh::Deferred,
        );
        let refinement = params.refinement_strategy_params.get_refinement_strategy();
        let mut counts = [0; 3];
        for (a, b) in eager.iter_mut().zip(&mut deferred) {
            assert_eq!(a.rng.next_u64(), b.rng.next_u64());
            let a = a.state.as_ref().unwrap();
            let b = b.state.as_ref().unwrap();
            assert_eq!(a.f_plus, b.f_plus);
            assert_eq!(a.f_minus, b.f_minus);
            assert_eq!(a.residuals, b.residuals);
            assert_eq!(a.current_error.to_bits(), b.current_error.to_bits());
            assert_eq!(a.boundaries, b.boundaries);
            assert_eq!(a.backbone_values, b.backbone_values);
            assert_eq!(a.tilt_values, b.tilt_values);
            assert_eq!(a.iteration, b.iteration);
            assert_eq!(a.terminated, b.terminated);
            assert_eq!(format!("{:?}", a.loop_state), format!("{:?}", b.loop_state));
            counts[0] += a.loop_state.split_count - initial.loop_state.split_count;
            counts[1] += a.loop_state.resplit_count - initial.loop_state.resplit_count;
            counts[2] += a.loop_state.merge_count - initial.loop_state.merge_count;

            // Buffered events include the ordered actions, gains, row updates, and snapshots.
            #[cfg(feature = "evo-logging")]
            assert_eq!(
                format!("{:?}", a.logging_state),
                format!("{:?}", b.logging_state)
            );
            // The next full refresh must reconstruct identical candidate and interval caches.
            let refreshed_a = refinement.refresh_statistics(a.clone());
            let refreshed_b = refinement.refresh_statistics(b.clone());
            assert_eq!(
                format!("{:?}", refreshed_a.precomputed_statistics),
                format!("{:?}", refreshed_b.precomputed_statistics)
            );
        }
        let finish_model = |members: Vec<Member<'_>>| {
            let grids: Vec<_> = members
                .into_iter()
                .map(|member| {
                    let mut state = member.state.unwrap();
                    state.branch_consensus = None;
                    finish_fit(state).1
                })
                .collect();
            let primary = aggregate_bagged_two_tensor(&grids, initial.x, None, 1.0);
            let indices = (0..grids.len()).collect();
            crate::forest::TSL::new(vec![StagePredictor::new_ensemble(
                grids,
                primary,
                indices,
                Aggregation::Combined,
            )])
        };
        // Finalization operates directly on the final states, including a deferred final round.
        let eager = finish_model(eager);
        let deferred = finish_model(deferred);
        assert_eq!(
            bincode::serialize(&eager).unwrap(),
            bincode::serialize(&deferred).unwrap()
        );
        assert_eq!(eager.predict(initial.x), deferred.predict(initial.x));
        counts
    }

    fn parity_params(huber: bool, max_bins: Option<u16>, penalty: f64) -> GridTensorParams {
        let refinement = RefinementStrategyParamsBuilder::new()
            .alpha(0.1)
            .tilt_tau(0.01)
            .tilt_rho(0.001);
        GridTensorParamsBuilder::new()
            .n_iter(8)
            .max_bins(max_bins)
            .refinement_strategy(if huber { refinement.huber() } else { refinement.l2() }.build())
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
    fn deferred_consensus_matches_eager_models_actions_and_caches() {
        let x = Array2::from_shape_fn((128, 3), |(row, col)| {
            ((row * (2 * col + 3) + col * 11) % 127) as f64 / 32.0 - 2.0
        });
        let y = Array1::from_shape_fn(x.nrows(), |row| {
            let a = x[[row, 0]];
            let b = x[[row, 1]];
            a.sin() * b.exp() + 0.4 * x[[row, 2]] + if row % 23 == 0 { 3.0 } else { 0.0 }
        });
        let mut counts = [0; 3];
        for huber in [false, true] {
            for max_bins in [None, Some(16)] {
                for penalty in [0.0, 0.001] {
                    let params = parity_params(huber, max_bins, penalty);
                    let initial = initialize_state_with_input(x.view(), y.view(), &params, None);
                    let actions = assert_candidate_refresh_parity(&initial, &params);
                    for (count, actions) in counts.iter_mut().zip(actions) {
                        *count += actions;
                    }
                }
            }
        }
        assert!(counts[0] > 0 && counts[1] > 0);
    }

    #[test]
    fn deferred_consensus_matches_eager_merges() {
        let x = Array2::from_shape_fn((32, 1), |(row, _)| row as f64);
        let y = Array1::from_shape_fn(32, |row| if row < 16 { -1.0 } else { 2.0 });
        for huber in [false, true] {
            for max_bins in [None, Some(8)] {
                let params = parity_params(huber, max_bins, 1e6);
                let refinement = params.refinement_strategy_params.get_refinement_strategy();
                let split_strategy = params.split_strategy_params.get_split_strategy();
                let mut initial = initialize_state_with_input(x.view(), y.view(), &params, None);
                // A pre-existing boundary exposes a beneficial complexity-reducing merge.
                let split = SplitCandidate {
                    col: 0,
                    index: 16,
                    allowed_interval_idx: 0,
                    error_reduction: initial.precomputed_statistics.error_reductions_split[0][16],
                    update_left: initial.precomputed_statistics.update_pairs_split_left[0][16],
                    update_right: initial.precomputed_statistics.update_pairs_split_right[0][16],
                };
                initial = fitting_reducer(
                    initial,
                    FittingAction::ApplySplit { split },
                    &refinement,
                    &split_strategy,
                );
                let counts = assert_candidate_refresh_parity(&initial, &params);
                assert_eq!(counts[2], 4, "each member must merge its initial boundary");
            }
        }
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
                    weight: 1.0,
                    plus: Array1::from_elem(y.len(), (total_plus - plus) / others),
                    minus: Array1::from_elem(y.len(), (total_minus - minus) / others),
                });
                Member {
                    tree_id,
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
        let mut members = constant_members(x.view(), y.view(), &params, &[(3.0, 2.0), (4.0, 1.0)]);

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

        fit_members(&mut members, &params, 1.0, None);
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

    #[test]
    fn consensus_stops_after_a_whole_round_without_progress() {
        let x = Array2::from_shape_fn((16, 1), |(row, _)| row as f64);
        let y = Array1::zeros(x.nrows());
        let params = lifecycle_params(5);
        let mut members = constant_members(x.view(), y.view(), &params, &[(2.0, 2.0), (2.0, 2.0)]);
        fit_members(&mut members, &params, 1.0, None);
        for member in members {
            let state = member.state.unwrap();
            assert_eq!(state.iteration, 0);
            assert_eq!(state.loop_state.fineness, 0);
            assert!(state.f.iter().all(|&prediction| prediction == 0.0));
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
        fit_members(&mut members, &params, 1.0, None);
        for (member, (plus, minus)) in members.into_iter().zip([(3.0, 2.0), (4.0, 1.0)]) {
            let state = member.state.unwrap();
            assert_eq!(state.iteration, 7);
            assert_eq!(state.loop_state.split_count, 0);
            assert!(state.f_plus.iter().all(|&value| value == plus));
            assert!(state.f_minus.iter().all(|&value| value == minus));
        }
    }
}
