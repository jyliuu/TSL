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
        fit::{finish_fit, fit_step, initialize_state_with_input, max_iterations},
        prepared_input::PreparedInput,
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
                let (current, advanced) =
                    fit_step(current, iter, tg_params.n_iter, &refinement, &split, rng);
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

    use super::{fit_members, Member};
    use rand::{rngs::StdRng, SeedableRng};

    use crate::{
        grid_tensor::{
            fit::{finish_fit, fit_step, initialize_state_with_input},
            params::{
                GridTensorParamsBuilder, RefinementStrategyParamsBuilder,
                SplitStrategyParamsBuilder,
            },
            state::BranchConsensus,
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
