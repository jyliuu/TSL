use ndarray::{ArrayView1, ArrayView2};
use rand::{Rng, SeedableRng};

use crate::{
    grid_tensor::{self},
    logging::{self},
    stage_predictor::aggregate_bagged::aggregate_bagged_two_tensor,
    FitResult,
};

use super::super::grid_tensor::GridTensor;
use super::params::StagePredictorParams;
use super::StagePredictor;

#[cfg(feature = "use-rayon")]
use rayon::prelude::*;

fn fit_grid_tensor_with_context(
    tree_id: usize,
    seed: u64,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    tg_params: &grid_tensor::GridTensorParams,
    initial_state: Option<&grid_tensor::state::FittingState<'_>>,
) -> (FitResult, GridTensor) {
    logging::with_tree_id(tree_id, || {
        let mut thread_rng = rand::rngs::StdRng::seed_from_u64(seed);

        let (fit_res, tg) = if let Some(initial_state) = initial_state {
            grid_tensor::fit::fit_initialized(initial_state.clone(), tg_params, &mut thread_rng)
        } else {
            grid_tensor::fit(x.view(), y.view(), tg_params, &mut thread_rng)
        };

        log::debug!(
            "Fitted tree grid, lambda+: {:?}, lambda-: {:?}, error: {:?}",
            tg.lambda_plus,
            tg.lambda_minus,
            fit_res.err
        );

        use crate::logging::{log_grid_error_fitted, GridErrorVariant};
        log_grid_error_fitted(fit_res.err, GridErrorVariant::Train);

        (fit_res, tg)
    })
}

pub fn fit_ensemble<R: Rng + ?Sized>(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &StagePredictorParams,
    rng: &mut R,
) -> (FitResult, StagePredictor) {
    fit_ensemble_with_input(x, y, hyperparameters, rng, None)
}

pub(crate) fn fit_ensemble_with_input<R: Rng + ?Sized>(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &StagePredictorParams,
    rng: &mut R,
    prepared: Option<&grid_tensor::prepared_input::PreparedInput<'_>>,
) -> (FitResult, StagePredictor) {
    let grid_tensors = fit_grid_bag_with_input(x, y, hyperparameters, rng, prepared)
        .into_iter()
        .map(|(_, grid)| grid)
        .collect();
    combine_fitted_grids(x, y, hyperparameters, grid_tensors)
}

/// Fit the ordered bag and retain each grid's normalized training predictions.
pub fn fit_grid_bag<R: Rng + ?Sized>(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &StagePredictorParams,
    rng: &mut R,
) -> Vec<(FitResult, GridTensor)> {
    fit_grid_bag_with_input(x, y, hyperparameters, rng, None)
}

fn fit_grid_bag_with_input<R: Rng + ?Sized>(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &StagePredictorParams,
    rng: &mut R,
    prepared: Option<&grid_tensor::prepared_input::PreparedInput<'_>>,
) -> Vec<(FitResult, GridTensor)> {
    let StagePredictorParams {
        n_trees, tg_params, ..
    } = hyperparameters;

    #[cfg(feature = "use-rayon")]
    let epoch_opt = logging::current_epoch();
    let seeds: Vec<u64> = (0..*n_trees).map(|_| rng.gen()).collect();
    let initial_state = if *n_trees > 1 && !logging::is_logging_enabled() {
        Some(grid_tensor::fit::initialize_state_with_input(
            x.view(),
            y.view(),
            tg_params,
            prepared,
        ))
    } else {
        None
    };

    let fitted_grids: Vec<(FitResult, GridTensor)>;

    #[cfg(not(feature = "use-rayon"))]
    {
        fitted_grids = seeds
            .iter()
            .enumerate()
            .map(|(tree_id, &seed)| {
                fit_grid_tensor_with_context(tree_id, seed, x, y, tg_params, initial_state.as_ref())
            })
            .collect();
    }

    #[cfg(feature = "use-rayon")]
    {
        let tree_seed_pairs: Vec<(usize, u64)> = (0..*n_trees).zip(seeds).collect();
        let mut results: Vec<(usize, FitResult, GridTensor)> = tree_seed_pairs
            .into_par_iter()
            .map(|(tree_id, seed)| {
                if let Some(epoch) = epoch_opt {
                    logging::with_epoch(epoch, || {
                        let (fit_res, grid) = fit_grid_tensor_with_context(
                            tree_id,
                            seed,
                            x,
                            y,
                            tg_params,
                            initial_state.as_ref(),
                        );
                        (tree_id, fit_res, grid)
                    })
                } else {
                    let (fit_res, grid) = fit_grid_tensor_with_context(
                        tree_id,
                        seed,
                        x,
                        y,
                        tg_params,
                        initial_state.as_ref(),
                    );
                    (tree_id, fit_res, grid)
                }
            })
            .collect();
        results.sort_by_key(|(tree_id, _, _)| *tree_id);
        fitted_grids = results
            .into_iter()
            .map(|(_, fit, grid)| (fit, grid))
            .collect();
    }

    drop(initial_state);
    fitted_grids
}

fn combine_fitted_grids(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &StagePredictorParams,
    grid_tensors: Vec<GridTensor>,
) -> (FitResult, StagePredictor) {
    let StagePredictorParams {
        similarity_threshold,
        aggregation_method,
        ..
    } = hyperparameters;

    let trim_percentage = 1.0 - *similarity_threshold;
    log::info!(
        "Using bagged two-tensor aggregation with similarity_threshold={}, trim_percentage={}",
        similarity_threshold,
        trim_percentage
    );

    let primary_grid_tensor =
        aggregate_bagged_two_tensor(&grid_tensors, x.view(), None, trim_percentage);

    let candidate_indices: Vec<usize> = (0..grid_tensors.len()).collect();

    let tgf = StagePredictor::new_ensemble(
        grid_tensors,
        primary_grid_tensor,
        candidate_indices,
        aggregation_method.clone(),
    );

    let preds = tgf.predict(x);
    log::debug!("Combined preds mean: {:?}", preds.mean());

    let residuals = &y - &preds;
    let err = residuals.pow2().mean().unwrap();
    log::info!(
        "Combined tree grid error: {:?}, lambda+: {:?}, lambda-: {:?}",
        err,
        tgf.primary_grid_tensor.lambda_plus,
        tgf.primary_grid_tensor.lambda_minus
    );

    use crate::logging::{log_grid_error_combined, GridErrorVariant};
    log_grid_error_combined(err, GridErrorVariant::Train);

    (
        FitResult {
            err,
            residuals: residuals.to_owned(),
            y_hat: preds,
        },
        tgf,
    )
}
