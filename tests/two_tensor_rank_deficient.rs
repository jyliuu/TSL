use ndarray::{Array1, Array2};
use tsl::forest::{fit_boosted, params::TSLBoostedParamsBuilder};
use tsl::grid_tensor::params::{RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder};

#[test]
fn centered_target_can_fit_from_equal_positive_branches() {
    let x = Array2::from_shape_fn((64, 1), |(i, _)| (i as f64 - 31.5) / 16.0);
    let centered = x.column(0).to_owned();
    for offset in [0.0, 2.0] {
        let y: Array1<f64> = &centered + offset;
        let params = TSLBoostedParamsBuilder::new()
            .epochs(1)
            .n_trees(1)
            .n_iter(8)
            .max_bins(None)
            .seed(42)
            .log_level("off")
            .split_strategy(
                SplitStrategyParamsBuilder::new()
                    .best_split()
                    .min_interval_samples(4)
                    .build(),
            )
            .refinement_strategy(
                RefinementStrategyParamsBuilder::new()
                    .l2()
                    .alpha(0.0)
                    .tilt_tau(0.01)
                    .build(),
            )
            .build();
        let (_, model) = fit_boosted(x.view(), y.view(), &params);
        let prediction = model.predict(x.view());
        let mse = (&prediction - &y).mapv(|r| r * r).mean().unwrap();
        assert!(mse < 0.025, "offset={offset}: MSE={mse}");
        let grid = &model.get_stage_predictors()[0].primary_grid_tensor;
        assert!(grid.splits.iter().map(Vec::len).sum::<usize>() >= 7);
        assert!(grid.lambda_plus >= 0.0 && grid.lambda_minus >= 0.0);
        assert!(grid
            .backbone_values
            .iter()
            .flatten()
            .all(|b| b.is_finite() && *b >= 0.0));
        assert!(grid.tilt_values.iter().flatten().all(|d| d.is_finite()));
    }
}
