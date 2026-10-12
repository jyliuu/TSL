use crate::logging::{self, init_logging};

#[cfg(feature = "evo-logging")]
use crate::logging::{create_logger, LoggingConfig};
use crate::FitResult;
use ndarray::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

use super::{params::TSLBoostedParams, TSL};
use crate::stage_predictor::fit_ensemble_with_input;
use nalgebra::{DMatrix, DVector};

mod nonnegative;

type Logger = Option<std::rc::Rc<std::cell::RefCell<Box<dyn crate::logging::EvoLogger>>>>;

/// Minimum-norm least-squares solution of `x w ≈ y` via SVD.
///
/// Singular values at or below `σ_max · max(n, k) · ε` are treated as zero, so a
/// rank-deficient design — which arises once stages contribute near-collinear
/// `f₊`/`−f₋` columns — yields the pseudo-inverse solution instead of diverging.
/// Returns `None` only when the SVD lacks the factors needed to back-substitute.
fn solve_least_squares(x: ArrayView2<f64>, y: ArrayView1<f64>) -> Option<Array1<f64>> {
    let (n, k) = (x.nrows(), x.ncols());
    let a = DMatrix::from_row_iterator(n, k, x.iter().copied());
    let b = DVector::from_iterator(n, y.iter().copied());
    let svd = a.svd(true, true);
    let sigma_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = sigma_max * (n.max(k) as f64) * f64::EPSILON;
    let w = svd.solve(&b, cutoff).ok()?;
    Some(Array1::from_iter(w.iter().copied()))
}

/// Rows of `x` and `y` scaled by `√s`, so least squares on them minimises `Σᵢ sᵢ rᵢ²`.
fn scale_rows_by_root_weight(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    sample_weight: ArrayView1<f64>,
) -> (Array2<f64>, Array1<f64>) {
    let root = sample_weight.mapv(f64::sqrt);
    (&x * &root.view().insert_axis(Axis(1)), &y * &root)
}

/// Set up logging and event channel if visualdb_path is specified
fn setup_logging(
    db_path: Option<&String>,
    hyperparameters: &TSLBoostedParams,
    n_rows: usize,
    n_cols: usize,
) -> Option<crate::logging::EventChannel> {
    if let Some(db_path) = db_path {
        #[cfg(feature = "evo-logging")]
        {
            log::info!("Will save split events to database: {}", db_path);
            let config = LoggingConfig {
                db_path: db_path.clone(),
                run_label: None,
                record_residual_updates: false,
                pack_updates_as_blob: false,
            };
            let boxed = create_logger(Some(config));
            let logger_rc = std::rc::Rc::new(std::cell::RefCell::new(boxed));

            // Start the run
            let params_json = serde_json::to_string(hyperparameters).unwrap();
            logger_rc
                .borrow_mut()
                .start_run(&params_json, n_rows, n_cols);

            // Set up event channel
            let (tx, rx) = std::sync::mpsc::channel();
            crate::logging::set_event_sender(tx);

            Some((rx, logger_rc))
        }

        #[cfg(not(feature = "evo-logging"))]
        {
            log::warn!(
                "visualdb parameter specified but evo-logging feature is not enabled. Split events will not be saved to database: {}",
                db_path
            );
            None
        }
    } else {
        None
    }
}

/// Log combined grid snapshot for visualization
#[cfg(feature = "evo-logging")]
fn log_combined_grid_snapshot(
    logger: &Logger,
    stage_predictor: &crate::stage_predictor::StagePredictor,
    epoch: usize,
    f_plus: Option<&ndarray::Array1<f64>>,
    f_minus: Option<&ndarray::Array1<f64>>,
) {
    if logger.is_some() {
        use crate::grid_tensor::GridTensor;
        use serde_json::json;
        let primary: &GridTensor = &stage_predictor.primary_grid_tensor;

        // Build JSON with legacy fields and two-tensor fields
        let mean_factor = primary.get_mean_factor();
        let grid_json = json!({
            "grid_values": mean_factor,  // Computed: b * cosh(d)
            "splits": primary.splits,
            "intervals": primary.intervals,
            // Two-tensor fields
            "backbone_values": primary.backbone_values,
            "tilt_values": primary.tilt_values,
            "lambda_plus": primary.lambda_plus,
            "lambda_minus": primary.lambda_minus,
        })
        .to_string();

        // Convert f+ and f- arrays to Vec if provided
        let f_plus_vec = f_plus.map(|arr| arr.to_vec());
        let f_minus_vec = f_minus.map(|arr| arr.to_vec());

        logging::log_combined_grid(logging::CombinedGridSnapshot {
            epoch,
            energy: stage_predictor.energy,
            scaling: stage_predictor.scaling_plus, // Legacy field for backward compatibility
            scaling_plus: stage_predictor.scaling_plus,
            scaling_minus: stage_predictor.scaling_minus,
            grid_json,
            f_plus: f_plus_vec,
            f_minus: f_minus_vec,
        });
    }
}

/// Flush events to database if logging was set up
fn flush_logger(logger: &Logger) {
    #[cfg(feature = "evo-logging")]
    if let Some(ref logger_rc) = logger {
        if let Err(e) = logger_rc.borrow_mut().flush() {
            log::warn!("Failed to flush events to database: {}", e);
        } else {
            log::info!("Successfully saved split events to database");
        }
    }
}

/// Generic training loop for boosted model
fn train_boosted_model<F>(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &TSLBoostedParams,
    on_epoch_end: F,
) -> (FitResult, TSL)
where
    F: FnMut(
        usize,
        &crate::stage_predictor::StagePredictor,
        &[crate::stage_predictor::StagePredictor],
        &Logger,
    ),
{
    train_boosted_model_with_preparation(x, y, None, hyperparameters, true, on_epoch_end)
}

fn train_boosted_model_with_preparation<F>(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    sample_weight: Option<ArrayView1<f64>>,
    hyperparameters: &TSLBoostedParams,
    reuse_input: bool,
    mut on_epoch_end: F,
) -> (FitResult, TSL)
where
    F: FnMut(
        usize,
        &crate::stage_predictor::StagePredictor,
        &[crate::stage_predictor::StagePredictor],
        &Logger,
    ),
{
    init_logging(&hyperparameters.log_level);
    log::info!(
        "Fitting boosted model with hyperparameters: {:#?}",
        hyperparameters
    );

    // Set up logging and event channel if visualdb_path is specified
    let event_channel = setup_logging(
        hyperparameters.visualdb_path.as_ref(),
        hyperparameters,
        x.nrows(),
        x.ncols(),
    );

    let mut sp_params = hyperparameters.sp_params.clone();
    let prepared_input = (reuse_input
        && hyperparameters.epochs > 1
        && sp_params.n_trees > 1
        && !logging::is_logging_enabled())
    .then(|| {
        crate::grid_tensor::prepared_input::PreparedInput::new(x, sp_params.tg_params.max_bins)
    });

    let mut rng = StdRng::seed_from_u64(hyperparameters.seed);
    let mut y_new = y.to_owned();
    let mut stage_predictors: Vec<crate::stage_predictor::StagePredictor> = Vec::new();
    // Pre-allocate matrix to store unscaled predictions as columns for Orthogonal Greedy
    // Always use 2 columns per epoch: f+ and -f- (will be populated in Step 4)
    let n_samples = x.nrows();
    let n_components_per_epoch = 2; // f+ and -f- columns per epoch
    let mut predictions_matrix =
        Array2::<f64>::zeros((n_samples, hyperparameters.epochs * n_components_per_epoch));

    for i in 0..hyperparameters.epochs {
        logging::with_epoch(i, || {
            log::info!("Fitting epoch {}", i);
            if i > 0 {
                let current_n_iter = sp_params.tg_params.n_iter as f64;
                sp_params.tg_params.n_iter =
                    ((current_n_iter * hyperparameters.decay).round() as usize).max(1);
            }

            // Single-family fitting (split_residuals removed - replaced by two-tensor f+ and -f- columns)
            {
                // Original single-family fitting logic
                let (_fit_result, mut stage_predictor) = fit_ensemble_with_input(
                    x.view(),
                    y_new.view(),
                    sample_weight
                        .as_ref()
                        .map(|sample_weight| sample_weight.view()),
                    &sp_params,
                    &mut rng,
                    prepared_input.as_ref(),
                );

                // Always use OrthogonalGreedy scaling strategy
                // Store f+ and f- for logging
                use crate::stage_predictor::extract_two_tensor_predictions_unscaled;

                // Extract UNSCALED f+ and f- predictions from combined grid
                // CRITICAL: Extract directly from two-tensor fields, do NOT apply any scaling
                // The grid's scaling field is ignored - we only use scaling_plus/scaling_minus from OLS
                let (f_plus, f_minus) = extract_two_tensor_predictions_unscaled(
                    &stage_predictor.primary_grid_tensor,
                    x,
                );

                // Store for logging
                let f_plus_opt: Option<ndarray::Array1<f64>> = Some(f_plus.clone());
                let f_minus_opt: Option<ndarray::Array1<f64>> = Some(f_minus.clone());

                // Compute energy for logging
                let energy_plus = f_plus.pow2().mean().unwrap();
                let energy_minus = f_minus.pow2().mean().unwrap();
                stage_predictor.energy = Some(energy_plus + energy_minus);
                log::info!(
                    "Energy of epoch {}: f+={:.6}, f-={:.6}, total={:.6}",
                    i,
                    energy_plus,
                    energy_minus,
                    energy_plus + energy_minus
                );

                // Store in predictions matrix: columns i*2 (f+) and i*2+1 (-f-)
                // Note: We store -f_minus in the design matrix (not f_minus)
                // This is because the model is: f = scaling_plus * f_+ + scaling_minus * (-f_-)
                predictions_matrix.column_mut(i * 2).assign(&f_plus);
                let f_minus_neg = -&f_minus;
                predictions_matrix
                    .column_mut(i * 2 + 1)
                    .assign(&f_minus_neg);

                // Slice matrix up to current epoch: columns 0..(2*(i+1))
                // At epoch i, we have 2*(i+1) columns total (2 columns per epoch: f+ and -f-)
                let k = 2 * (i + 1);
                let x_mat = predictions_matrix.slice(s![.., 0..k]);

                // Refit every branch coefficient against the original target.
                let scaled = sample_weight
                    .map(|sample_weight| scale_rows_by_root_weight(x_mat, y, sample_weight));
                let (design, target) = match &scaled {
                    Some((design, target)) => (design.view(), target.view()),
                    None => (x_mat.view(), y.view()),
                };
                let weights = if hyperparameters.nonnegative_scaling {
                    let mut prior = Array1::zeros(k);
                    for (j, stage) in stage_predictors.iter().enumerate() {
                        prior[2 * j] = stage.scaling_plus.unwrap_or(0.0);
                        prior[2 * j + 1] = stage.scaling_minus.unwrap_or(0.0);
                    }
                    nonnegative::solve(design, target, prior.view())
                } else {
                    solve_least_squares(design, target)
                };
                let (err, residuals) = if let Some(w) = weights {
                    log::info!("New projected weights (two-tensor mode): {:?}", w);

                    // Update scalings for all families based on OLS solution
                    // w has length k = 2*(i+1), with 2 coefficients per epoch
                    // Each family now needs TWO scalings: one for f+ and one for -f-
                    for (j, prev_tgf) in stage_predictors.iter_mut().enumerate() {
                        // j is the family index (0..i)
                        // Family j's coefficients are at w[2*j] (f+) and w[2*j+1] (-f-)
                        if 2 * j + 1 < w.len() {
                            prev_tgf.scaling_plus = Some(w[2 * j]); // f+ coefficient
                            prev_tgf.scaling_minus = Some(w[2 * j + 1]);
                            // -f- coefficient (note: already negative in design matrix)
                        }
                    }

                    // Current epoch's scalings
                    stage_predictor.scaling_plus = Some(w[k - 2]); // f+ coefficient
                    stage_predictor.scaling_minus = Some(w[k - 1]); // -f- coefficient

                    // Update residuals and error
                    let preds_all = x_mat.dot(&w);
                    let residuals = &y - &preds_all;
                    let err = crate::weighted_mean_squared_error(residuals.view(), sample_weight);
                    let residuals_owned = residuals.to_owned();
                    (err, residuals_owned)
                } else {
                    log::warn!(
                        "OLS solve failed for OrthogonalGreedy, falling back to unscaled predictions"
                    );
                    // Fallback: use unscaled predictions (f+ - f-), no scaling applied
                    // This is the raw two-tensor prediction without OLS coefficients
                    let combined_preds = &f_plus - &f_minus;
                    let residuals = &y - &combined_preds;
                    let err = crate::weighted_mean_squared_error(residuals.view(), sample_weight);
                    (err, residuals.to_owned())
                };

                // Log epoch scalings (always using OrthogonalGreedy)
                if event_channel.is_some() {
                    // Log all scalings that were updated (epochs 0..i)
                    for (j, prev_tgf) in stage_predictors.iter().enumerate() {
                        if let Some(scaling_plus) = prev_tgf.scaling_plus {
                            logging::log_epoch_scaling(logging::EpochScalingSnapshot {
                                epoch: j,
                                scaling: scaling_plus, // TODO: Update logging to show both scalings
                                optimization_epoch: i, // Re-optimized at epoch i
                            });
                        }
                    }
                    // Log current epoch's scaling
                    if let Some(scaling_plus) = stage_predictor.scaling_plus {
                        logging::log_epoch_scaling(logging::EpochScalingSnapshot {
                            epoch: i,
                            scaling: scaling_plus, // TODO: Update logging to show both scalings
                            optimization_epoch: i,
                        });
                    }
                }

                y_new = residuals;

                // Log the combined/primary tree grid snapshot for visualization
                #[cfg(feature = "evo-logging")]
                {
                    if let Some((_, ref logger_rc)) = event_channel {
                        log_combined_grid_snapshot(
                            &Some(logger_rc.clone()),
                            &stage_predictor,
                            i,
                            f_plus_opt.as_ref(),
                            f_minus_opt.as_ref(),
                        );
                    }
                }

                // Callback for specific logic (test error, etc)
                // Pass stage_predictors so callback can access all previous families with updated scalings
                #[cfg(feature = "evo-logging")]
                let logger_for_callback = event_channel
                    .as_ref()
                    .map(|(_, logger_rc)| logger_rc.clone());
                #[cfg(not(feature = "evo-logging"))]
                let logger_for_callback = None;
                on_epoch_end(i, &stage_predictor, &stage_predictors, &logger_for_callback);

                log::info!("Epoch {}, error: {:?}", i, err);
                stage_predictors.push(stage_predictor);
            }
        });
    }

    let err = match sample_weight {
        Some(_) => crate::weighted_mean_squared_error(y_new.view(), sample_weight),
        None => y_new.iter().map(|&x| x * x).sum::<f64>() / y_new.len() as f64,
    };
    let fit_result = FitResult {
        err,
        residuals: y_new.clone(),
        y_hat: -y_new + y,
    };

    // Drain all events and flush to database if logging was set up
    #[cfg(feature = "evo-logging")]
    if let Some(ref event_channel) = event_channel {
        crate::logging::drain_logging_events(event_channel);
        crate::logging::clear_event_sender();
        flush_logger(&Some(event_channel.1.clone()));
    }

    (fit_result, TSL::new(stage_predictors))
}

pub fn fit_boosted(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    hyperparameters: &TSLBoostedParams,
) -> (FitResult, TSL) {
    train_boosted_model(
        x,
        y,
        hyperparameters,
        |_epoch, _stage_predictor, _prev_families, _logger| {
            // Logging is now done before the callback, so nothing to do here
        },
    )
}

/// Fit a boosted TSL model to the weighted squared error `Σᵢ sᵢ (yᵢ − F(xᵢ))²`.
///
/// Every stage's grid tensors and the per-epoch refit of all stage coefficients minimise
/// this objective, and a row's branch-consensus term carries the same weight `sᵢ`. The
/// weights are rescaled to mean one, so the penalties and `min_split_loss` keep the scale
/// they have in an unweighted fit.
///
/// # Panics
/// If `sample_weight` does not hold one finite, nonnegative entry per row of `x` with a
/// positive sum.
pub fn fit_boosted_weighted(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    sample_weight: ArrayView1<f64>,
    hyperparameters: &TSLBoostedParams,
) -> (FitResult, TSL) {
    assert_eq!(
        sample_weight.len(),
        x.nrows(),
        "sample_weight must have one entry per row"
    );
    assert!(
        sample_weight.iter().all(|s| s.is_finite() && *s >= 0.0),
        "sample_weight must be finite and nonnegative"
    );
    let total = sample_weight.sum();
    assert!(total > 0.0, "sample_weight must have a positive sum");
    let sample_weight = sample_weight.mapv(|s| s * (x.nrows() as f64 / total));
    train_boosted_model_with_preparation(
        x,
        y,
        Some(sample_weight.view()),
        hyperparameters,
        true,
        |_epoch, _stage_predictor, _prev_families, _logger| {},
    )
}

/// Fit a boosted TSL model while tracking per-epoch test error on a provided
/// evaluation dataset. The stage-wise predictor at epoch `i` is the sum of
/// predictions from the first `i+1` fitted `StagePredictor`s.
pub fn fit_boosted_with_test_error(
    x_train: ArrayView2<f64>,
    y_train: ArrayView1<f64>,
    x_test: ArrayView2<f64>,
    y_test: ArrayView1<f64>,
    hyperparameters: &TSLBoostedParams,
) -> (FitResult, TSL, Array1<f64>) {
    // Maintain cumulative test predictions of the stage-wise model
    let mut test_errors: Vec<f64> = Vec::with_capacity(hyperparameters.epochs);

    let (fit_res, tsl) = train_boosted_model(
        x_train,
        y_train,
        hyperparameters,
        |epoch, stage_predictor, prev_families, logger| {
            // Scalings are updated retroactively, so we need to
            // recompute cumulative predictions from scratch using all families with updated scalings
            // prev_families contains families 0..epoch-1 with updated scalings
            // stage_predictor has the updated scaling for the current epoch

            // Recompute cumulative test predictions from scratch with current scalings
            let mut yhat_test_epoch = Array1::<f64>::zeros(x_test.nrows());
            // Add predictions from all previous families (with updated scalings)
            for family in prev_families {
                yhat_test_epoch += &family.predict(x_test.view());
            }

            // Add prediction from current family (with updated scaling)
            yhat_test_epoch += &stage_predictor.predict(x_test.view());

            let test_err = (y_test.to_owned() - &yhat_test_epoch)
                .pow2()
                .mean()
                .unwrap();

            // Log test error and combined grid snapshot
            #[cfg(feature = "evo-logging")]
            if logger.is_some() {
                use crate::logging::{
                    log_grid_error_combined, log_grid_error_fitted, GridErrorVariant,
                };

                // Log family-level test error
                log_grid_error_combined(test_err, GridErrorVariant::Test);
                // Also log per-tree test error for trees in current family
                // Note: This computes error if we replaced the current family with just this tree
                let mut per_tree_test_errs: Vec<f64> = Vec::new();
                for (tree_id, tg) in stage_predictor.get_grid_tensors().iter().enumerate() {
                    logging::with_tree_id(tree_id, || {
                        // Compute cumulative predictions: all previous families + this tree
                        let mut yhat_test_tree = Array1::<f64>::zeros(x_test.nrows());
                        // Add predictions from all previous families
                        for family in prev_families {
                            yhat_test_tree += &family.predict(x_test.view());
                        }
                        // Replace current family with just this tree
                        yhat_test_tree += &tg.predict(x_test.view());
                        let tree_test_err =
                            (y_test.to_owned() - &yhat_test_tree).pow2().mean().unwrap();
                        per_tree_test_errs.push(tree_test_err);
                        log_grid_error_fitted(tree_test_err, GridErrorVariant::Test);
                    });
                }
                if !per_tree_test_errs.is_empty() {
                    let mut min_err = f64::INFINITY;
                    let mut max_err = f64::NEG_INFINITY;
                    let mut sum_err = 0.0;
                    for e in &per_tree_test_errs {
                        if *e < min_err {
                            min_err = *e;
                        }
                        if *e > max_err {
                            max_err = *e;
                        }
                        sum_err += *e;
                    }
                    let mean_err = sum_err / per_tree_test_errs.len() as f64;
                    log::info!(
                            "Epoch {} per-tree test error range: min={:.6}, max={:.6}, mean={:.6} (n={})",
                            epoch, min_err, max_err, mean_err, per_tree_test_errs.len()
                        );
                }
            }

            log::info!("Epoch {}, test error: {:?}", epoch, test_err);

            // Logging is now done before the callback in train_boosted_model, so nothing to do here

            test_errors.push(test_err);
        },
    );

    (fit_res, tsl, Array1::from(test_errors))
}

#[cfg(test)]
mod prepared_input_tests {
    use super::*;
    use crate::{
        forest::params::TSLBoostedParamsBuilder,
        grid_tensor::params::RefinementStrategyParamsBuilder,
    };

    fn assert_parity(x: ArrayView2<f64>, y: ArrayView1<f64>, params: &TSLBoostedParams) {
        let reference =
            train_boosted_model_with_preparation(x, y, None, params, false, |_, _, _, _| {});
        let actual =
            train_boosted_model_with_preparation(x, y, None, params, true, |_, _, _, _| {});
        assert_eq!(
            bincode::serialize(&actual.1).unwrap(),
            bincode::serialize(&reference.1).unwrap()
        );
        assert_eq!(actual.0.y_hat, reference.0.y_hat);
        assert_eq!(actual.0.residuals, reference.0.residuals);
        assert_eq!(actual.0.err.to_bits(), reference.0.err.to_bits());
    }

    #[test]
    fn repeated_stage_input_preparation_preserves_complete_forests() {
        let storage = Array2::from_shape_fn((166, 10), |(i, j)| {
            if j == 8 {
                1.
            } else {
                ((i * (j + 5)) % 47) as f64 / 13.
            }
        });
        let x = storage.slice(s![..;2,..;2]);
        let y = Array1::from_shape_fn(x.nrows(), |i| x[[i, 0]].sin() + x[[i, 1]] * x[[i, 2]] - 2.);
        for bins in [None, Some(1), Some(16), Some(256)] {
            for huber in [false, true] {
                let refinement = RefinementStrategyParamsBuilder::new()
                    .alpha(0.01)
                    .tilt_tau(0.01);
                let refinement = if huber {
                    refinement.huber()
                } else {
                    refinement.l2()
                }
                .build();
                let params = TSLBoostedParamsBuilder::new()
                    .epochs(13)
                    .n_trees(3)
                    .n_iter(8)
                    .decay(0.95)
                    .seed(79)
                    .max_bins(bins)
                    .log_level("off")
                    .refinement_strategy(refinement)
                    .build();
                assert_parity(x, y.view(), &params);
            }
        }
        let params = TSLBoostedParamsBuilder::new()
            .epochs(100)
            .n_trees(3)
            .n_iter(3)
            .decay(0.8)
            .seed(79)
            .max_bins(Some(16))
            .log_level("off")
            .build();
        assert_parity(x, y.view(), &params);
    }
}

#[cfg(test)]
mod nonnegative_scaling_tests {
    use super::*;
    use crate::forest::params::TSLBoostedParamsBuilder;
    use crate::stage_predictor::extract_two_tensor_predictions_unscaled;

    #[test]
    fn nonnegative_scaling_preserves_branch_signs_and_training_progress() {
        let x = Array2::from_shape_fn((96, 3), |(i, j)| ((i * (j + 3)) % 41) as f64 / 13.0);
        let y = Array1::from_shape_fn(x.nrows(), |i| x[(i, 0)].sin() + x[(i, 1)] * x[(i, 2)] - 2.0);
        let params = TSLBoostedParamsBuilder::new()
            .epochs(4)
            .n_trees(3)
            .n_iter(8)
            .max_bins(Some(16))
            .consensus(0.3)
            .nonnegative_scaling(true)
            .log_level("off")
            .build();
        let mut previous_loss = y.mapv(|v| v * v).sum();
        let (fit, model) =
            train_boosted_model(x.view(), y.view(), &params, |_, stage, earlier, _| {
                let mut prediction = Array1::<f64>::zeros(y.len());
                for component in earlier.iter().chain(std::iter::once(stage)) {
                    let plus = component.scaling_plus.unwrap();
                    let minus = component.scaling_minus.unwrap();
                    assert!(plus.is_finite() && plus >= 0.0);
                    assert!(minus.is_finite() && minus >= 0.0);
                    let (fp, fm) = extract_two_tensor_predictions_unscaled(
                        &component.primary_grid_tensor,
                        x.view(),
                    );
                    assert!(fp
                        .iter()
                        .chain(fm.iter())
                        .all(|v| v.is_finite() && *v >= 0.0));
                    let reconstructed = &fp * plus - &fm * minus;
                    assert_eq!(component.predict(x.view()), reconstructed);
                    prediction += &reconstructed;
                }
                assert!(prediction.iter().all(|v| v.is_finite()));
                let loss = (&prediction - &y).mapv(|v| v * v).sum();
                assert!(loss <= previous_loss + 1e-10 * previous_loss.max(1.0));
                previous_loss = loss;
            });
        let predicted = model.predict(x.view());
        assert!((&predicted - &fit.y_hat).iter().all(|v| v.abs() < 1e-11));
        assert!((fit.err * y.len() as f64 - previous_loss).abs() < 1e-10);
    }
}
