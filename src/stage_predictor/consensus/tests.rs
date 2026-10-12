use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::{fit_consensus_bag, fit_members, max_iterations, sample_consensus_anchors, Member};
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    grid_tensor::{
        fit::{finish_fit, fit_step, initialize_state_with_input},
        params::{
            GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
        },
        state::{BranchConsensus, FittingState},
        GridTensor, GridTensorParams,
    },
    stage_predictor::{
        extract_two_tensor_predictions_unscaled, fit_grid_bag,
        params::{StagePredictorParams, StagePredictorParamsBuilder},
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
        let scale = (a.mapv(|v| v * v).mean().unwrap() + b.mapv(|v| v * v).mean().unwrap()) / 2.0;
        (a - b).mapv(|v| v * v).mean().unwrap() / scale
    };
    let mut total = 0.0;
    let mut pairs = 0.0;
    for i in 0..branches.len() {
        for j in 0..i {
            total +=
                relative(&branches[i].0, &branches[j].0) + relative(&branches[i].1, &branches[j].1);
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
                let stage_params = StagePredictorParams {
                    tg_params: params.clone(),
                    consensus: 0.3,
                    consensus_interval: interval,
                    consensus_anchors: anchors,
                    ..StagePredictorParams::default()
                };
                fit_consensus_bag(
                    x.view(),
                    y.view(),
                    &stage_params,
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
                for ((a_fit, a_grid), (b_fit, b_grid)) in baseline.iter().zip(requested.iter()) {
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
                    let initial = initialize_state_with_input(x.view(), y.view(), &params, None);
                    let mut members = members_from_state(&initial);
                    fit_members(&mut members, &params, 0.3, interval, anchors, None);
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
    fit_members(&mut [], &lifecycle_params(0), 0.3, 0, None, None);
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
    let (retried, advanced) = fit_step(retried, 1, params.n_iter, &refinement, &split, &mut rng);
    assert!(advanced);
    for (&plus, &minus) in retried.f_plus.iter().zip(retried.f_minus.iter()) {
        assert!((plus - 23.0 / 9.0).abs() < 1e-12);
        assert!((minus - 22.0 / 9.0).abs() < 1e-12);
    }

    for interval in [1, 2, usize::MAX] {
        let mut members = constant_members(x.view(), y.view(), &params, &[(3.0, 2.0), (4.0, 1.0)]);
        fit_members(&mut members, &params, 1.0, interval, None, None);
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
        let mut members = constant_members(x.view(), y.view(), &params, &[(2.0, 2.0), (2.0, 2.0)]);
        fit_members(&mut members, &params, 1.0, interval, None, None);
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
fn consensus_keeps_exhausted_members_fixed() {
    let x = Array2::from_shape_fn((16, 1), |(row, _)| row as f64);
    let y = Array1::zeros(x.nrows());
    let params = lifecycle_params(2);
    let mut members = constant_members(x.view(), y.view(), &params, &[(3.0, 2.0), (4.0, 1.0)]);
    for member in &mut members {
        let state = member.state.as_mut().unwrap();
        state.loop_state.fineness = params.n_iter;
        state.iteration = 7;
    }
    fit_members(&mut members, &params, 1.0, 1, None, None);
    for (member, (plus, minus)) in members.into_iter().zip([(3.0, 2.0), (4.0, 1.0)]) {
        let state = member.state.unwrap();
        assert_eq!(state.iteration, 7);
        assert_eq!(state.loop_state.split_count, 0);
        assert!(state.f_plus.iter().all(|&value| value == plus));
        assert!(state.f_minus.iter().all(|&value| value == minus));
    }
}
