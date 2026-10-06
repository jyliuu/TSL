use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Mutex;
use tsl::{
    grid_tensor::{
        fit,
        params::{RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder},
        GridTensor,
    },
    logging,
    stage_predictor::{
        aggregate_bagged_two_tensor, fit_ensemble, fit_grid_bag,
        params::{StagePredictorParams, StagePredictorParamsBuilder},
        StagePredictor,
    },
    FitResult,
};

#[cfg(feature = "use-rayon")]
use rayon::prelude::*;

static TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn ordered_grid_bag_matches_ensemble_grids_and_training_predictions() {
    let _guard = TEST_LOCK.lock().unwrap();
    let (x, y) = data(127, 4, false);
    for bins in [None, Some(16)] {
        for huber in [false, true] {
            let parameters = params(false, huber, bins, 7);
            let mut bag_rng = StdRng::seed_from_u64(312);
            let mut ensemble_rng = bag_rng.clone();
            let bag = fit_grid_bag(x.view(), y.view(), &parameters, &mut bag_rng);
            let (_, ensemble) = fit_ensemble(x.view(), y.view(), &parameters, &mut ensemble_rng);
            assert_eq!(bag.len(), ensemble.grid_tensors.len());
            for ((fit, grid), expected) in bag.iter().zip(&ensemble.grid_tensors) {
                assert_eq!(
                    bincode::serialize(grid).unwrap(),
                    bincode::serialize(expected).unwrap()
                );
                assert_eq!(fit.y_hat, grid.predict(x.view()));
                assert_eq!(fit.residuals, &y - &fit.y_hat);
                assert_eq!(
                    fit.err.to_bits(),
                    fit.residuals.pow2().mean().unwrap().to_bits()
                );
            }
            assert_eq!(bag_rng.gen::<u64>(), ensemble_rng.gen::<u64>());
        }
    }
}

fn data(n: usize, p: usize, positive: bool) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((n, p), |(row, axis)| {
        ((row * (19 + axis * 12) + axis * 31) % 251) as f64 / 31.0 - 4.0
    });
    let y = Array1::from_shape_fn(n, |row| {
        x[[row, 0]].sin().exp() * x[[row, 1]].cos()
            + 0.5 * x[[row, 2]]
            + if positive { 6.0 } else { 0.0 }
    });
    (x, y)
}

fn params(best: bool, huber: bool, bins: Option<u16>, trees: usize) -> StagePredictorParams {
    let split = SplitStrategyParamsBuilder::new()
        .split_try(9)
        .colsample_bytree(0.8)
        .min_interval_samples(3)
        .complexity_penalty(0.001);
    let split = if best {
        split.best_split()
    } else {
        split.random_split()
    };
    let refinement = RefinementStrategyParamsBuilder::new()
        .alpha(0.1)
        .tilt_tau(0.01)
        .tilt_rho(0.001);
    let refinement = if huber {
        refinement.huber()
    } else {
        refinement.l2()
    };
    StagePredictorParamsBuilder::new()
        .n_trees(trees)
        .n_iter(12)
        .similarity_threshold(0.5)
        .split_strategy(split.build())
        .refinement_strategy(refinement.build())
        .max_bins(bins)
        .build()
}

fn independent_ensemble(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    params: &StagePredictorParams,
    rng: &mut StdRng,
) -> (FitResult, StagePredictor) {
    let seeds: Vec<u64> = (0..params.n_trees).map(|_| rng.gen()).collect();
    let epoch = logging::current_epoch();
    let fit_one = |(tree_id, seed): (usize, u64)| {
        let task = || {
            logging::with_tree_id(tree_id, || {
                let (result, grid) = fit(x, y, &params.tg_params, &mut StdRng::seed_from_u64(seed));
                logging::log_grid_error_fitted(result.err, logging::GridErrorVariant::Train);
                grid
            })
        };
        if let Some(epoch) = epoch {
            logging::with_epoch(epoch, task)
        } else {
            task()
        }
    };
    #[cfg(feature = "use-rayon")]
    let grids: Vec<GridTensor> = seeds.into_par_iter().enumerate().map(fit_one).collect();
    #[cfg(not(feature = "use-rayon"))]
    let grids: Vec<GridTensor> = seeds.into_iter().enumerate().map(fit_one).collect();
    let primary = aggregate_bagged_two_tensor(&grids, x, None, 1.0 - params.similarity_threshold);
    let stage = StagePredictor::new_ensemble(
        grids,
        primary,
        (0..params.n_trees).collect(),
        params.aggregation_method.clone(),
    );
    let predictions = stage.predict(x);
    let residuals = &y - &predictions;
    let err = residuals.pow2().mean().unwrap();
    logging::log_grid_error_combined(err, logging::GridErrorVariant::Train);
    (
        FitResult {
            err,
            residuals,
            y_hat: predictions,
        },
        stage,
    )
}

fn assert_fit_parity(actual: &(FitResult, StagePredictor), expected: &(FitResult, StagePredictor)) {
    assert_eq!(
        bincode::serialize(&actual.1).unwrap(),
        bincode::serialize(&expected.1).unwrap()
    );
    assert_eq!(actual.0.err.to_bits(), expected.0.err.to_bits());
    assert_eq!(actual.0.y_hat, expected.0.y_hat);
    assert_eq!(actual.0.residuals, expected.0.residuals);
}

#[test]
fn shared_initialization_matches_independent_seeded_fits() {
    let _guard = TEST_LOCK.lock().unwrap();
    let mut models = Vec::new();
    for positive in [false, true] {
        let (x, y) = data(251, 4, positive);
        for best in [false, true] {
            for huber in [false, true] {
                for bins in [None, Some(32)] {
                    let params = params(best, huber, bins, 5);
                    let mut expected_rng = StdRng::seed_from_u64(137);
                    let mut actual_rng = expected_rng.clone();
                    let expected =
                        independent_ensemble(x.view(), y.view(), &params, &mut expected_rng);
                    let actual = fit_ensemble(x.view(), y.view(), &params, &mut actual_rng);
                    assert_fit_parity(&actual, &expected);
                    assert_eq!(actual_rng.gen::<u64>(), expected_rng.gen::<u64>());
                    models.push(actual.1);
                }
            }
        }
    }
    if let Some(path) = std::env::var_os("TSL_INITIALIZATION_BASELINE") {
        let bytes = bincode::serialize(&models).unwrap();
        if std::env::var_os("TSL_CAPTURE_INITIALIZATION_BASELINE").is_some() {
            std::fs::write(path, &bytes).unwrap();
        } else {
            assert_eq!(bytes, std::fs::read(path).unwrap());
        }
    }
}

#[test]
fn shared_initialization_accepts_strided_data_and_single_grid() {
    let _guard = TEST_LOCK.lock().unwrap();
    let (x, y) = data(502, 6, false);
    for trees in [1, 4] {
        let params = params(false, false, Some(32), trees);
        let x = x.slice(s![..;2, ..;2]);
        let y = y.slice(s![..;2]);
        let expected = independent_ensemble(x, y, &params, &mut StdRng::seed_from_u64(99));
        let actual = fit_ensemble(x, y, &params, &mut StdRng::seed_from_u64(99));
        assert_fit_parity(&actual, &expected);
    }
}

#[cfg(feature = "evo-logging")]
#[test]
fn logged_ensemble_preserves_independent_tree_events() {
    let _guard = TEST_LOCK.lock().unwrap();
    struct SenderGuard;
    impl Drop for SenderGuard {
        fn drop(&mut self) {
            logging::clear_event_sender();
        }
    }
    let (x, y) = data(80, 4, false);
    let params = params(false, true, Some(16), 3);
    let run = |shared: bool| {
        let (tx, rx) = std::sync::mpsc::channel();
        logging::set_event_sender(tx);
        let guard = SenderGuard;
        let result = logging::with_epoch(7, || {
            let mut rng = StdRng::seed_from_u64(71);
            if shared {
                fit_ensemble(x.view(), y.view(), &params, &mut rng)
            } else {
                independent_ensemble(x.view(), y.view(), &params, &mut rng)
            }
        });
        drop(guard);
        let events: Vec<_> = rx.into_iter().collect();
        for expected_tree in 0..params.n_trees {
            assert!(events.iter().any(|event| matches!(event,
                logging::LoggingMessage::Split { epoch: 7, tree_id, .. }
                    if *tree_id == expected_tree)));
            assert!(events.iter().any(|event| matches!(event,
                logging::LoggingMessage::Component { epoch: 7, tree_id, .. }
                    if *tree_id == expected_tree)));
            assert!(events.iter().any(|event| matches!(event,
                logging::LoggingMessage::FComponentStats { epoch: 7, tree_id, .. }
                    if *tree_id == expected_tree)));
        }
        let mut events: Vec<String> = events.iter().map(|event| format!("{event:?}")).collect();
        events.sort();
        (result, events)
    };
    let (expected, expected_events) = run(false);
    let (actual, actual_events) = run(true);
    assert_fit_parity(&actual, &expected);
    assert!(!actual_events.is_empty());
    assert_eq!(actual_events, expected_events);
}

#[test]
#[ignore = "end-to-end stage fit timing with independent versus shared initialization"]
fn benchmark_shared_initialization() {
    use std::hint::black_box;
    use std::time::Instant;
    let _guard = TEST_LOCK.lock().unwrap();
    let mut data_rng = StdRng::seed_from_u64(1043);
    let x = Array2::from_shape_fn((20_640, 8), |_| data_rng.gen_range(-4.0_f64..4.0));
    let y = Array1::from_shape_fn(x.nrows(), |row| {
        (x[[row, 0]].sin() * x[[row, 1]].cos()).exp() + x[[row, 2]]
    });
    #[cfg(feature = "use-rayon")]
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let benchmark = || {
        for trees in [2, 24] {
            for bins in [None, Some(64)] {
                for iterations in [5, 40] {
                    let mut params = params(false, false, bins, trees);
                    params.tg_params.n_iter = iterations;
                    params.similarity_threshold = 0.0;
                    let expected = independent_ensemble(
                        x.view(),
                        y.view(),
                        &params,
                        &mut StdRng::seed_from_u64(81),
                    );
                    let actual =
                        fit_ensemble(x.view(), y.view(), &params, &mut StdRng::seed_from_u64(81));
                    assert_fit_parity(&actual, &expected);
                    let measure = |shared: bool| {
                        let start = Instant::now();
                        let mut rng = StdRng::seed_from_u64(81);
                        if shared {
                            black_box(fit_ensemble(x.view(), y.view(), &params, &mut rng));
                        } else {
                            black_box(independent_ensemble(x.view(), y.view(), &params, &mut rng));
                        }
                        start.elapsed()
                    };
                    for repeat in 0..3 {
                        let (independent, shared) = if repeat % 2 == 0 {
                            (measure(false), measure(true))
                        } else {
                            let shared = measure(true);
                            (measure(false), shared)
                        };
                        println!("shared initialization: trees={trees} bins={bins:?} iterations={iterations} repeat={repeat} independent_ms={:.3} shared_ms={:.3} speedup={:.3}", independent.as_secs_f64()*1000.0, shared.as_secs_f64()*1000.0, independent.as_secs_f64()/shared.as_secs_f64());
                    }
                }
            }
        }
    };
    #[cfg(feature = "use-rayon")]
    pool.install(benchmark);
    #[cfg(not(feature = "use-rayon"))]
    benchmark();
}
