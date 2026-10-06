//! Lockstep bag fitting toward shared branch functions.
//!
//! Every grid takes one structural action per round. Before each round, grid `b`'s local
//! objective gains the proximal term `ρ (‖f₊ − c₊‖² + ‖f₋ − c₋‖²)` on the training rows,
//! where `c±` are the mean branches of the other grids. Each branch is a single product
//! of univariate factors, so grids that agree on both branches agree on their factors up
//! to the per-feature scale that identification removes. Agreement on `f = f₊ − f₋`
//! alone would leave every common-mode change of the two branches free.

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
    active: bool,
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
    let refinement = tg_params.refinement_strategy_params.get_refinement_strategy();
    let split = tg_params.split_strategy_params.get_split_strategy();
    let others = (seeds.len().max(2) - 1) as f64;

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
            active: true,
        })
        .collect();

    let mut total_plus = Array1::<f64>::zeros(y.len());
    let mut total_minus = Array1::<f64>::zeros(y.len());
    for iter in 0..max_iterations(tg_params.n_iter) {
        total_plus.fill(0.0);
        total_minus.fill(0.0);
        for member in &members {
            let state = member.state.as_ref().expect("member state");
            total_plus += &state.f_plus;
            total_minus += &state.f_minus;
        }
        let (total_plus, total_minus) = (total_plus.view(), total_minus.view());
        let advance = |member: &mut Member<'a>| {
            if !member.active {
                return;
            }
            let Member {
                tree_id,
                rng,
                state,
                active,
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
                *active = advanced;
            });
        };
        #[cfg(feature = "use-rayon")]
        members.par_iter_mut().for_each(advance);
        #[cfg(not(feature = "use-rayon"))]
        members.iter_mut().for_each(advance);
        if members.iter().all(|member| !member.active) {
            break;
        }
    }

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

fn in_context<R>(tree_id: usize, epoch: Option<usize>, f: impl FnOnce() -> R) -> R {
    logging::with_tree_id(tree_id, || match epoch {
        Some(epoch) => logging::with_epoch(epoch, f),
        None => f(),
    })
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, ArrayView2};
    use rand::{rngs::StdRng, SeedableRng};

    use crate::{
        grid_tensor::GridTensor,
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
}
