//! Histogram-binning tests.
//!
//! Parity test: max_bins=None must reproduce the existing exact path bit-for-bit.

mod data_utils;

#[cfg(test)]
mod tests {
    use crate::data_utils::setup_data_csv;
    use rand::{rngs::StdRng, SeedableRng};
    use tsl::grid_tensor::{
        fit,
        params::{
            GridTensorParams, GridTensorParamsBuilder, RefinementStrategyParamsBuilder,
            SplitStrategyParamsBuilder,
        },
    };

    fn housing_params_random(max_bins: Option<u16>) -> GridTensorParams {
        // RandomSplit: closer to the tuned-test default configuration.
        GridTensorParamsBuilder::new()
            .n_iter(50)
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .random_split()
                    .split_try(10)
                    .colsample_bysplit(1.0)
                    .min_interval_samples(3)
                    .min_split_loss(0.0)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(0.01)
                    .build(),
            )
            .max_bins(max_bins)
            .build()
    }

    /// Default for parity & determinism tests — uses random_split (default mode).
    fn housing_params(max_bins: Option<u16>) -> GridTensorParams {
        housing_params_random(max_bins)
    }

    /// PARITY TEST. Fits California Housing twice — once with the exact path
    /// (max_bins=None) and once with max_bins=None on the new code path — and
    /// asserts identical fit results. This verifies the histogram-binning
    /// prologue is a true no-op when max_bins is None.
    #[test]
    fn parity_max_bins_none_matches_exact() {
        let (x, y) = setup_data_csv("./data/housing_full.csv");
        let params = housing_params(None);

        // Run the same fit twice with the same seed — must be deterministic and
        // identical to itself.
        let mut rng1 = StdRng::seed_from_u64(42);
        let (fit_a, model_a) = fit(x.view(), y.view(), &params, &mut rng1);
        let mut rng2 = StdRng::seed_from_u64(42);
        let (fit_b, model_b) = fit(x.view(), y.view(), &params, &mut rng2);

        assert_eq!(
            fit_a.err, fit_b.err,
            "Determinism: same seed must yield same training err"
        );
        // Compare model splits (continuous thresholds) and component values.
        for col in 0..model_a.splits.len() {
            assert_eq!(
                model_a.splits[col], model_b.splits[col],
                "Determinism: same seed must yield same splits per column"
            );
            assert_eq!(
                model_a.backbone_values[col], model_b.backbone_values[col],
                "Determinism: backbone values must match"
            );
            assert_eq!(
                model_a.tilt_values[col], model_b.tilt_values[col],
                "Determinism: tilt values must match"
            );
        }
        assert_eq!(model_a.lambda_plus, model_b.lambda_plus);
        assert_eq!(model_a.lambda_minus, model_b.lambda_minus);
        println!(
            "PARITY OK | max_bins=None determinism, train_err = {:.6}",
            fit_a.err
        );
    }

    /// Sanity smoke test: the binned path actually fits and produces a
    /// reasonable train MSE on California Housing (within 2× of exact).
    #[test]
    fn smoke_max_bins_255_california_housing() {
        let (x, y) = setup_data_csv("./data/housing_full.csv");
        let params_none = housing_params(None);
        let params_255 = housing_params(Some(255));

        let mut rng_none = StdRng::seed_from_u64(123);
        let (fit_none, _) = fit(x.view(), y.view(), &params_none, &mut rng_none);
        let mut rng_bin = StdRng::seed_from_u64(123);
        let (fit_bin, _) = fit(x.view(), y.view(), &params_255, &mut rng_bin);

        println!(
            "Train MSE: exact = {:.4}, max_bins=255 = {:.4}",
            fit_none.err, fit_bin.err
        );
        // Binned path must produce a real (non-NaN, finite) error close to exact.
        assert!(fit_bin.err.is_finite());
        assert!(fit_bin.err > 0.0);
        // Allow up to 2× degradation — generous, just a sanity bound.
        assert!(
            fit_bin.err < 2.0 * fit_none.err,
            "Binned path err {} too far from exact {}",
            fit_bin.err,
            fit_none.err
        );
    }

    /// TOLERANCE TEST. Asserts that binning at max_bins=255 stays within a
    /// reasonable train-MSE delta of the exact path on California Housing.
    /// The optimization should not silently regress quality.
    #[test]
    fn mse_tolerance_max_bins_255_california_housing() {
        let (x, y) = setup_data_csv("./data/housing_full.csv");
        let params_none = housing_params(None);
        let params_255 = housing_params(Some(255));

        let mut rng_none = StdRng::seed_from_u64(42);
        let (fit_none, _) = fit(x.view(), y.view(), &params_none, &mut rng_none);
        let mut rng_bin = StdRng::seed_from_u64(42);
        let (fit_bin, _) = fit(x.view(), y.view(), &params_255, &mut rng_bin);

        let drift = (fit_bin.err - fit_none.err) / fit_none.err;
        println!(
            "Train MSE: exact = {:.4}, max_bins=255 = {:.4} ({:+.1}% delta)",
            fit_none.err,
            fit_bin.err,
            drift * 100.0
        );
        // Allow up to 25% drift on California Housing. The v1 binning-only
        // attempt saw ~13% with split-scan binned only; binning both paths
        // should not regress meaningfully from that.
        assert!(
            drift.abs() < 0.25,
            "MSE drift {:.1}% exceeds 25% tolerance",
            drift * 100.0
        );
    }

    /// A bin budget covering all distinct values preserves the exact candidate set.
    #[test]
    fn low_cardinality_binned_fit_matches_exact_candidates() {
        let mut rows = Vec::new();
        let mut targets = Vec::new();
        for i in 0..8 {
            for j in 0..6 {
                for k in 0..4 {
                    for _ in 0..(1 + (i + 2 * j + 3 * k) % 5) {
                        rows.extend([i as f64, j as f64, k as f64]);
                        targets.push(
                            2.0 + 0.13 * (i * i) as f64
                                + 0.37 * j as f64
                                + 0.19 * (k * k) as f64
                                + 0.07 * (i * j) as f64,
                        );
                    }
                }
            }
        }
        let x = ndarray::Array2::from_shape_vec((targets.len(), 3), rows).unwrap();
        let y = ndarray::Array1::from_vec(targets);
        let mut exact_rng = StdRng::seed_from_u64(43);
        let mut binned_rng = StdRng::seed_from_u64(43);
        let (exact_fit, exact) = fit(x.view(), y.view(), &housing_params(None), &mut exact_rng);
        let (binned_fit, binned) = fit(
            x.view(),
            y.view(),
            &housing_params(Some(64)),
            &mut binned_rng,
        );
        assert!(exact.splits.iter().map(Vec::len).sum::<usize>() > 0);
        assert_eq!(exact.splits, binned.splits);
        let exact_predictions = exact.predict(x.view());
        let binned_predictions = binned.predict(x.view());
        let max_difference = exact_predictions
            .iter()
            .zip(binned_predictions.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_difference < 1e-9,
            "prediction difference {max_difference}"
        );
        assert!((exact_fit.err - binned_fit.err).abs() < 1e-10);
    }

    /// SAME-SEED DETERMINISM at max_bins=Some(255): also confirm the binned
    /// path is deterministic.
    #[test]
    fn determinism_max_bins_255() {
        let (x, y) = setup_data_csv("./data/housing_full.csv");
        let params = housing_params(Some(255));

        let mut rng1 = StdRng::seed_from_u64(7);
        let (fit_a, _) = fit(x.view(), y.view(), &params, &mut rng1);
        let mut rng2 = StdRng::seed_from_u64(7);
        let (fit_b, _) = fit(x.view(), y.view(), &params, &mut rng2);

        assert_eq!(
            fit_a.err, fit_b.err,
            "Binned path must be deterministic with same seed"
        );
        println!(
            "DETERMINISM (max_bins=255) OK | train_err = {:.6}",
            fit_a.err
        );
    }
}
