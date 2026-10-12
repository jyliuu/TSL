mod data_utils;

#[cfg(test)]
mod tests {
    use crate::data_utils::{setup_data_csv, split_data};
    
    use tsl::{
        stage_predictor::Aggregation,
        forest::{fit_boosted, params::TSLBoostedParamsBuilder, TSL},
    };
    use std::fs;

    #[test]
    fn json_preserves_split_boundaries_and_predictions_bitwise() {
        use ndarray::Array2;
        use tsl::grid_tensor::GridTensor;

        for cut in [0.9619511996472115_f64, 0.11351420094652015, 24.217381236226846] {
            let grid = GridTensor::new_two_tensor(
                vec![vec![cut]], vec![vec![1, 2]],
                vec![vec![(f64::NEG_INFINITY, cut), (cut, f64::INFINITY)]],
                vec![vec![0.3, 1.234]], vec![vec![-0.2, 0.7]], 2.0, 0.1,
            );
            let points = Array2::from_shape_vec((3, 1), vec![
                f64::from_bits(cut.to_bits()-1), cut, f64::from_bits(cut.to_bits()+1),
            ]).unwrap();
            let restored: GridTensor = serde_json::from_slice(&serde_json::to_vec(&grid).unwrap()).unwrap();
            assert_eq!(bincode::serialize(&grid).unwrap(), bincode::serialize(&restored).unwrap());
            let expected = grid.predict(points.view());
            assert_ne!(expected[0], expected[1]);
            assert_eq!(expected, restored.predict(points.view()));
        }
    }

    #[test]
    fn test_tsl_save_and_load() {
        // Setup data
        let (x, y) = setup_data_csv("./data/2covars.csv");
        let (x_train, y_train, x_test, _y_test) = split_data(&x, &y);

        // Fit a model
        let params = TSLBoostedParamsBuilder::new()
            .epochs(3)
            .n_trees(5)
            .n_iter(10)
            .seed(42)
            .aggregation_method(Aggregation::Combined)
            .build();

        let (_fit_result, original_model) = fit_boosted(x_train.view(), y_train.view(), &params);

        // Get predictions from original model
        let original_preds = original_model.predict(x_test.view());

        // Serialize and save to file using bincode
        let bytes =
            bincode::serialize(&original_model).expect("Failed to serialize TSL model to bincode");
        fs::write("target/data.bin", bytes).expect("Failed to write to target/data.bin");

        // Read and deserialize from file
        let bytes = fs::read("target/data.bin").expect("Failed to read from target/data.bin");
        let deserialized_model: TSL =
            bincode::deserialize(&bytes).expect("Failed to deserialize TSL model from bincode");

        // Get predictions from deserialized model
        let deserialized_preds = deserialized_model.predict(x_test.view());

        // Verify predictions are identical
        let pred_diff = &original_preds - &deserialized_preds;
        let max_diff = pred_diff.iter().map(|x| x.abs()).fold(0.0, f64::max);

        assert!(
            max_diff < 1e-10,
            "Predictions from deserialized model differ from original (max diff: {})",
            max_diff
        );
    }
}
