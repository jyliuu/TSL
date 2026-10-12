use ndarray::{ArrayView1, ArrayView2};
use rand::{Rng, SeedableRng};

use crate::{
    grid_tensor::{self, fit, state::FittingState, GridTensor, GridTensorParams},
    logging::{self},
    stage_predictor::aggregate_bagged::aggregate_bagged_two_tensor,
    FitResult,
};

use super::params::StagePredictorParams;
use super::StagePredictor;

#[cfg(feature = "use-rayon")]
use rayon::prelude::*;

/// Fit one bag member; `fit_results` keeps its training evaluation.
fn fit_grid_tensor_with_context<'a>(
    tree_id: usize,
    seed: u64,
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, f64>,
    tg_params: &GridTensorParams,
    initial_state: Option<&FittingState<'a>>,
    fit_results: bool,
) -> (Option<FitResult>, GridTensor) {
    logging::with_tree_id(tree_id, || {
        let mut thread_rng = rand::rngs::StdRng::seed_from_u64(seed);
        let state = match initial_state {
            Some(initial_state) => initial_state.clone(),
            None => fit::initialize_state(x, y, tg_params),
        };
        let state = fit::fit_structure(state, tg_params, &mut thread_rng);

        let (fit_res, tg) = if fit_results {
            let (fit_res, tg) = fit::finish_fit(state);
            logging::log_grid_error_fitted(fit_res.err, logging::GridErrorVariant::Train);
            (Some(fit_res), tg)
        } else {
            (None, fit::finish_grid(state))
        };
        log::debug!(
            "Fitted tree grid, lambda+: {:?}, lambda-: {:?}, error: {:?}",
            tg.lambda_plus,
            tg.lambda_minus,
            fit_res.as_ref().map(|fit_res| fit_res.err)
        );
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
    let grid_tensors = fit_grid_bag_with_input(x, y, hyperparameters, rng, prepared, false)
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
    fit_grid_bag_with_input(x, y, hyperparameters, rng, None, true)
        .into_iter()
        .map(|(fit, grid)| (fit.expect("bag members keep their fit results"), grid))
        .collect()
}

/// Fit the bag; `fit_results` keeps each member's training evaluation, which the
/// ensemble path does not consume.
fn fit_grid_bag_with_input<R: Rng + ?Sized>(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &StagePredictorParams,
    rng: &mut R,
    prepared: Option<&grid_tensor::prepared_input::PreparedInput<'_>>,
    fit_results: bool,
) -> Vec<(Option<FitResult>, GridTensor)> {
    let StagePredictorParams {
        n_trees,
        tg_params,
        consensus,
        ..
    } = hyperparameters;

    // Each member's training evaluation is kept when a caller asks for it or when logging
    // reports it; otherwise the member only yields its identified grid.
    let fit_results = fit_results || logging::is_logging_enabled();
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

    if *consensus > 0.0 && *n_trees > 1 {
        return super::consensus::fit_consensus_bag(
            x.view(),
            y.view(),
            hyperparameters,
            &seeds,
            initial_state.as_ref(),
            prepared,
            epoch_opt,
            fit_results,
        );
    }

    let fit_member = |tree_id: usize, seed: u64| {
        fit_grid_tensor_with_context(
            tree_id,
            seed,
            x.view(),
            y.view(),
            tg_params,
            initial_state.as_ref(),
            fit_results,
        )
    };

    #[cfg(not(feature = "use-rayon"))]
    let fitted_grids = seeds
        .iter()
        .enumerate()
        .map(|(tree_id, &seed)| fit_member(tree_id, seed))
        .collect();

    // Rayon workers carry no logging context, so each member re-enters the epoch.
    #[cfg(feature = "use-rayon")]
    let fitted_grids = seeds
        .par_iter()
        .enumerate()
        .map(|(tree_id, &seed)| match epoch_opt {
            Some(epoch) => logging::with_epoch(epoch, || fit_member(tree_id, seed)),
            None => fit_member(tree_id, seed),
        })
        .collect();

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

    let primary_grid_tensor = aggregate_bagged_two_tensor(&grid_tensors, x.view(), trim_percentage);

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
