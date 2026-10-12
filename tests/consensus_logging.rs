#![cfg(feature = "evo-logging")]

use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, SeedableRng};
use tsl::{
    grid_tensor::params::{RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder},
    logging,
    stage_predictor::{fit_ensemble, params::StagePredictorParamsBuilder},
};

#[test]
fn logged_consensus_ensemble_reports_every_member() {
    struct SenderGuard;
    impl Drop for SenderGuard {
        fn drop(&mut self) {
            logging::clear_event_sender();
        }
    }
    let x = Array2::from_shape_fn((80, 4), |(row, axis)| {
        ((row * (19 + axis * 12) + axis * 31) % 251) as f64 / 31.0 - 4.0
    });
    let y = Array1::from_shape_fn(80, |row| {
        x[[row, 0]].sin().exp() * x[[row, 1]].cos() + 0.5 * x[[row, 2]]
    });
    let params = StagePredictorParamsBuilder::new()
        .n_trees(3)
        .n_iter(12)
        .similarity_threshold(0.5)
        .consensus(0.3)
        .split_strategy(
            SplitStrategyParamsBuilder::new()
                .random_split()
                .split_try(9)
                .colsample_bytree(0.8)
                .min_interval_samples(3)
                .complexity_penalty(0.001)
                .build(),
        )
        .refinement_strategy(
            RefinementStrategyParamsBuilder::new()
                .huber()
                .alpha(0.1)
                .tilt_tau(0.01)
                .tilt_rho(0.001)
                .build(),
        )
        .max_bins(Some(16))
        .build();

    let (tx, rx) = std::sync::mpsc::channel();
    logging::set_event_sender(tx);
    let guard = SenderGuard;
    let (_, ensemble) = logging::with_epoch(7, || {
        fit_ensemble(x.view(), y.view(), &params, &mut StdRng::seed_from_u64(71))
    });
    drop(guard);
    let events: Vec<_> = rx.into_iter().collect();
    for expected_tree in 0..params.n_trees {
        assert!(events.iter().any(|event| matches!(event,
            logging::LoggingMessage::GridErrFitted { epoch: 7, tree_id, variant: logging::GridErrorVariant::Train, .. }
                if *tree_id == expected_tree)));
        assert!(events.iter().any(|event| matches!(event,
            logging::LoggingMessage::Component { epoch: 7, tree_id, .. }
                if *tree_id == expected_tree)));
    }

    // Logging observes the fit without changing it.
    let (_, unlogged) = fit_ensemble(x.view(), y.view(), &params, &mut StdRng::seed_from_u64(71));
    for (grid, expected) in ensemble.grid_tensors.iter().zip(&unlogged.grid_tensors) {
        assert_eq!(
            bincode::serialize(grid).unwrap(),
            bincode::serialize(expected).unwrap()
        );
    }
}
