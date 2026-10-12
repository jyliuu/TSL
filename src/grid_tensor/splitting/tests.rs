use super::*;
use crate::grid_tensor::state::FittingState;
use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, SeedableRng};

// Helper to create a mock FittingState for testing
fn create_mock_state(error_reductions: Vec<Vec<f64>>) -> FittingState<'static> {
    let n = 20;
    let p = error_reductions.len();

    // Create dummy data
    let x_data = Array2::zeros((n, p));
    let y_data = Array1::zeros(n);

    // Leak to get 'static lifetime (only for tests)
    let x_static = Box::leak(Box::new(x_data));
    let y_static = Box::leak(Box::new(y_data));

    let mut state = FittingState::new(x_static.view(), y_static.view());

    // Set up precomputed statistics with our error reductions
    state.precomputed_statistics.error_reductions_split = error_reductions;
    state.precomputed_statistics.initialized = true;

    state
}

fn best_strategy(complexity_penalty: f64) -> SplitStrategy {
    SplitStrategy::Best {
        min_interval_samples: 1,
        complexity_penalty,
        min_split_loss: 0.0,
    }
}

fn create_merge_state(
    complexity_penalty: f64,
    boundaries: Vec<usize>,
    raw_gains: Vec<f64>,
    merge_factors: Vec<(f64, f64)>,
    current_error: f64,
) -> (SplitStrategy, FittingState<'static>) {
    let strategy = best_strategy(complexity_penalty);
    let mut state = create_mock_state(vec![vec![f64::NAN; 20]]);
    state.current_error = current_error;
    state = strategy.initialize(state);
    state.boundaries[0] = boundaries;
    state.precomputed_statistics.error_reductions_merge[0] = raw_gains;
    state.precomputed_statistics.update_pairs_merge[0] = merge_factors;
    (strategy, state)
}

#[test]
fn initialize_enables_merges_only_for_positive_finite_penalties() {
    for penalty in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let strategy = best_strategy(penalty);
        let state = strategy.initialize(create_mock_state(vec![vec![1.0; 20]]));
        assert!(!state.split_strategy_state.merge_enabled);
    }

    let strategy = best_strategy(1.0);
    let state = strategy.initialize(create_mock_state(vec![vec![1.0; 20]]));
    assert!(state.split_strategy_state.merge_enabled);
}

#[test]
fn unified_selector_ranks_merge_against_split() {
    let strategy = best_strategy(1.0);
    let mut split_gains = vec![f64::NAN; 20];
    split_gains[5] = 1.0;
    let mut state = create_mock_state(vec![split_gains]);
    state = strategy.initialize(state);
    state.boundaries[0] = vec![10];
    state.precomputed_statistics.error_reductions_merge[0] = vec![100.0];
    state.precomputed_statistics.update_pairs_merge[0] = vec![(1.0, 1.0)];

    let mut rng = StdRng::seed_from_u64(42);
    assert!(matches!(
        strategy.propose_next_action(&state, &mut rng),
        Some(FittingAction::ApplyMerge { .. })
    ));
}

#[test]
fn boundary_penalty_uses_fixed_initial_scale_and_model_degrees_of_freedom() {
    let (_, state) = create_merge_state(1.0, vec![10], vec![0.0], vec![(1.0, 1.0)], 20.0);
    let expected_stage1 = 20.0_f64.ln();
    assert!((state.split_strategy_state.boundary_penalty - expected_stage1).abs() < 1e-12);

    let strategy = best_strategy(1.0);
    let mut state = create_mock_state(vec![vec![f64::NAN; 20]]);
    state.current_error = 20.0;
    state.lambda_minus = 1.0;
    let state = strategy.initialize(state);
    assert!((state.split_strategy_state.boundary_penalty - 2.0 * expected_stage1).abs() < 1e-12);
}

#[test]
fn boundary_penalty_is_fixed_after_initialization() {
    let (_, mut state) = create_merge_state(1.0, vec![10], vec![0.0], vec![(1.0, 1.0)], 20.0);
    let initial = state.split_strategy_state.boundary_penalty;
    state.current_error = 2_000.0;
    assert_eq!(state.split_strategy_state.boundary_penalty, initial);
}

#[test]
fn split_and_inverse_merge_pay_opposite_boundary_costs() {
    let strategy = best_strategy(1.0);
    let mut split_gains = vec![f64::NAN; 20];
    split_gains[5] = 4.0;
    let mut state = create_mock_state(vec![split_gains]);
    state.current_error = 20.0;
    state = strategy.initialize(state);
    state.boundaries[0] = vec![10];
    state.precomputed_statistics.error_reductions_merge[0] = vec![-4.0];
    state.precomputed_statistics.update_pairs_merge[0] = vec![(1.0, 1.0)];

    let mut rng = StdRng::seed_from_u64(42);
    assert!(matches!(
        strategy.propose_next_action(&state, &mut rng),
        Some(FittingAction::ApplySplit { .. })
    ));

    state.precomputed_statistics.error_reductions_split[0][5] = 2.0;
    state.precomputed_statistics.error_reductions_merge[0][0] = -2.0;
    assert!(matches!(
        strategy.propose_next_action(&state, &mut rng),
        Some(FittingAction::ApplyMerge { .. })
    ));
}

#[test]
fn unified_selector_rejects_negative_effective_merge_score() {
    let (strategy, state) = create_merge_state(1.0, vec![10], vec![-6.0], vec![(1.0, 1.0)], 20.0);

    assert!(strategy.propose_next_merge(&state).is_none());
}

#[test]
fn unified_selector_ranks_merge_score_and_skips_invalid_factors() {
    let (strategy, state) = create_merge_state(
        1.0,
        vec![1, 10, 15],
        vec![-5.0, 1.0, 100.0],
        vec![(2.0, 2.0), (4.0, 4.0), (f64::NAN, 7.0)],
        20.0,
    );

    let Some(FittingAction::ApplyMerge { merge }) = strategy.propose_next_merge(&state) else {
        panic!("expected a merge action");
    };
    assert_eq!(merge.col, 0);
    assert_eq!(merge.interval_idx, 1);
    assert_eq!(merge.index, 10);
    assert_eq!(merge.error_reduction, 1.0);
}

#[test]
fn test_top_k_splits_selects_from_top_k() {
    let error_reductions = vec![
        vec![f64::NAN, 1.0, 5.0, 2.0, 4.0, 3.0], // top 2 are: 5.0, 4.0
    ];
    let strategy = SplitStrategy::TopK {
        top_k: 2,
        must_fill_all_k: false,
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.0,
    };
    let mut state = create_mock_state(error_reductions);
    state = strategy.initialize(state);

    let mut rng = StdRng::seed_from_u64(42);
    let action = strategy.propose_next_action(&state, &mut rng);

    // Should select one of the top 2 splits
    assert!(action.is_some());
    if let Some(FittingAction::ApplySplit { split }) = action {
        let (col, index) = (split.col, split.index);
        assert_eq!(col, 0);
        assert!(index == 2 || index == 4); // Either 5.0 at index 2 or 4.0 at index 4
    }
}

#[test]
fn test_top_k_splits_must_fill_all_k_true() {
    let error_reductions = vec![
        vec![f64::NAN, 1.0, 2.0, 3.0, f64::NAN], // Only 3 valid splits
    ];
    let strategy = SplitStrategy::TopK {
        top_k: 5,
        must_fill_all_k: true,
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.0,
    };
    let mut state = create_mock_state(error_reductions);
    state = strategy.initialize(state);

    let mut rng = StdRng::seed_from_u64(42);
    let action = strategy.propose_next_action(&state, &mut rng);

    // Should return None because we can't fill all k=5 splits (only 3 valid)
    assert!(action.is_none());
}

#[test]
fn test_top_k_splits_must_fill_all_k_false() {
    let error_reductions = vec![
        vec![f64::NAN, 1.0, 2.0, 3.0, f64::NAN], // Only 3 valid splits
    ];
    let strategy = SplitStrategy::TopK {
        top_k: 5,
        must_fill_all_k: false,
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.0,
    };
    let mut state = create_mock_state(error_reductions);
    state = strategy.initialize(state);

    let mut rng = StdRng::seed_from_u64(42);
    let action = strategy.propose_next_action(&state, &mut rng);

    // Should return a result because must_fill_all_k is false
    assert!(action.is_some());
}

#[test]
fn test_top_k_splits_multiple_calls() {
    let error_reductions = vec![vec![f64::NAN, 1.0, 5.0, 2.0, 4.0, 3.0]];
    let strategy = SplitStrategy::TopK {
        top_k: 2,
        must_fill_all_k: false,
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.0,
    };
    let mut state = create_mock_state(error_reductions);
    state = strategy.initialize(state);

    let mut rng = StdRng::seed_from_u64(42);

    // First split from top-k
    let action1 = strategy.propose_next_action(&state, &mut rng);
    assert!(action1.is_some());
    if let Some(FittingAction::ApplySplit { split }) = action1 {
        // Forbid around the selected split
        state = strategy.forbid_around_split(state, &split);
    }

    // Second split should still work
    let action2 = strategy.propose_next_action(&state, &mut rng);
    assert!(action2.is_some());
}

#[test]
fn test_top_k_splits_empty_heap() {
    let error_reductions = vec![
        vec![f64::NAN, f64::NAN, f64::NAN], // No valid splits
    ];
    let strategy = SplitStrategy::TopK {
        top_k: 2,
        must_fill_all_k: false,
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.0,
    };
    let mut state = create_mock_state(error_reductions);
    state = strategy.initialize(state);

    let mut rng = StdRng::seed_from_u64(42);
    let action = strategy.propose_next_action(&state, &mut rng);

    assert!(action.is_none());
}

#[test]
fn test_top_k_splits_nan_handling() {
    let error_reductions = vec![vec![f64::NAN, 1.0, f64::NAN, 3.0, f64::NAN, 2.0]];
    let strategy = SplitStrategy::TopK {
        top_k: 2,
        must_fill_all_k: false,
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.0,
    };
    let mut state = create_mock_state(error_reductions);
    state = strategy.initialize(state);

    let mut rng = StdRng::seed_from_u64(42);
    let action = strategy.propose_next_action(&state, &mut rng);

    assert!(action.is_some());
    if let Some(FittingAction::ApplySplit { split }) = action {
        let (col, index) = (split.col, split.index);
        let err_reductions = &state.precomputed_statistics.error_reductions_split;
        assert!(!err_reductions[col][index].is_nan());
        // Should be one of the valid non-NaN positions
        assert!(index == 1 || index == 3 || index == 5);
    }
}

// random_split_tests

#[test]
fn test_random_split_exact_split_try_less_than_total_positions() {
    let error_reductions = vec![vec![f64::NAN, 1.0, 2.0, 3.0, 4.0]];
    let strategy = SplitStrategy::Random {
        split_try: 10,
        colsample_bytree: 1.0,
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.0,
    };
    let mut state = create_mock_state(error_reductions);
    state = strategy.initialize(state);

    let mut rng = StdRng::seed_from_u64(42);
    let action = strategy.propose_next_action(&state, &mut rng);
    assert!(action.is_some());
}

#[test]
fn test_random_split_exact_split_try() {
    let error_reductions = vec![vec![
        f64::NAN,
        1.0,
        2.0,
        3.0,
        4.0,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    ]];
    let min_interval_samples = 4;
    let (allowed_intervals, total_positions) =
        initialize_intervals(&error_reductions, min_interval_samples);

    // Check initialization creates correct intervals
    assert_eq!(total_positions, vec![3]);
    assert_eq!(
        allowed_intervals,
        vec![vec![Interval::new(4, 1), Interval::new(8, 2)]]
    );

    let mut rng = StdRng::seed_from_u64(42);
    let result = sample_random_split(
        &allowed_intervals,
        &total_positions,
        1.0,
        1,
        &mut rng,
        &error_reductions,
    );
    assert!(result.is_some());
}

#[test]
fn test_random_split_exact_initialization() {
    let error_reductions = vec![
        vec![
            f64::NAN,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            f64::NAN,
            f64::NAN,
            8.0,
            9.0,
            10.0,
        ], // column 0
        vec![f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], // column 1
    ];
    let min_interval_samples = 2;
    let (allowed_intervals, total_positions) =
        initialize_intervals(&error_reductions, min_interval_samples);

    // Should have 2 features, each with intervals
    assert_eq!(allowed_intervals.len(), 2);
    assert_eq!(allowed_intervals[0].len(), 2);
    assert_eq!(allowed_intervals[0][0], Interval::new(2, 4));
    assert_eq!(allowed_intervals[0][1], Interval::new(8, 2));
    assert_eq!(allowed_intervals[1].len(), 1);
    assert_eq!(allowed_intervals[1][0], Interval::new(2, 7));
    assert_eq!(total_positions, vec![6, 7]);
}

#[test]
fn test_random_split_exact_multiple_splits() {
    let error_reductions = vec![
        vec![f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], // column 0
    ];
    let min_interval_samples = 2;
    let (allowed_intervals, total_positions) =
        initialize_intervals(&error_reductions, min_interval_samples);

    let mut rng = StdRng::seed_from_u64(42);

    // Multiple splits should work
    let result1 = sample_random_split(
        &allowed_intervals,
        &total_positions,
        1.0,
        1,
        &mut rng,
        &error_reductions,
    );
    assert!(result1.is_some());

    let result2 = sample_random_split(
        &allowed_intervals,
        &total_positions,
        1.0,
        1,
        &mut rng,
        &error_reductions,
    );
    assert!(result2.is_some());

    // Third split should still work (we're not modifying intervals between calls)
    let result3 = sample_random_split(
        &allowed_intervals,
        &total_positions,
        1.0,
        1,
        &mut rng,
        &error_reductions,
    );
    assert!(result3.is_some());
}

#[test]
fn test_random_split_exact_nan_handling() {
    let error_reductions = vec![
        vec![f64::NAN, f64::NAN, 2.0, f64::NAN, 4.0], // column 0 with NaN values
        vec![f64::NAN, 1.0, f64::NAN, 3.0, f64::NAN], // column 1 with NaN values
    ];
    let min_interval_samples = 1;
    let (allowed_intervals, total_positions) =
        initialize_intervals(&error_reductions, min_interval_samples);

    let mut rng = StdRng::seed_from_u64(42);
    let result = sample_random_split(
        &allowed_intervals,
        &total_positions,
        1.0,
        2,
        &mut rng,
        &error_reductions,
    );
    let (col, index, _interval_idx, _err) = result.unwrap();
    assert!(!error_reductions[col][index].is_nan());
}

#[test]
fn test_random_split_filters_splits_below_min_split_loss() {
    // Create minimal test data
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    // Create state
    let mut state = FittingState::new(x.view(), y.view());

    // Manually set error reductions with some below min_split_loss
    state.precomputed_statistics.error_reductions_split[0] =
        vec![f64::NAN, 0.1, 0.2, f64::NAN, 0.4, 0.5, f64::NAN, 0.3];
    state.precomputed_statistics.initialized = true;

    // Create strategy with high min_split_loss
    let strategy = SplitStrategy::Random {
        split_try: 1,
        colsample_bytree: 1.0,
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.5, // Higher than all error reductions
    };

    state = strategy.initialize(state);

    let mut rng = StdRng::seed_from_u64(42);
    let action = strategy.propose_next_action(&state, &mut rng);

    // Should return None because all splits are below min_split_loss
    assert!(action.is_none());
}

// best_split_tests

#[test]
fn test_best_split_picks_resplit_if_better() {
    use crate::grid_tensor::state::FittingState;
    use ndarray::{Array1, Array2};

    // Create minimal test data
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    // Create state
    let mut state = FittingState::new(x.view(), y.view());

    // Set up boundaries to enable resplit
    state.boundaries[0] = vec![3, 7]; // Two intervals: [0,3) and [3,7) and [7,10)

    // Set error reductions: split=1.0, resplit=2.0 (better)
    state.precomputed_statistics.error_reductions_split[0] =
        vec![f64::NAN, 1.0, 1.0, f64::NAN, 1.0, 1.0, f64::NAN, 1.0];
    state.precomputed_statistics.error_reductions_resplit[0] = vec![2.0, 2.0]; // Two resplit candidates
    state.precomputed_statistics.update_pairs_resplit_left[0] = vec![(0.0, 0.0), (0.0, 0.0)];
    state.precomputed_statistics.update_pairs_resplit_right[0] = vec![(0.0, 0.0), (0.0, 0.0)];
    state.precomputed_statistics.initialized = true;

    // Create strategy
    let strategy = SplitStrategy::Best {
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.0,
    };

    state = strategy.initialize(state);

    let mut rng = StdRng::seed_from_u64(42);
    let action = strategy.propose_next_action(&state, &mut rng);

    // Should return a resplit action because resplit (2.0) > split (1.0)
    match action {
        Some(crate::grid_tensor::action::FittingAction::ApplyResplit { .. }) => {
            // Expected: resplit action selected
        }
        _ => panic!("Expected resplit action, got: {:?}", action),
    }
}

#[test]
fn test_best_split_prevents_consecutive_resplits() {
    use crate::grid_tensor::{action::FittingAction, state::FittingState};
    use ndarray::{Array1, Array2};

    // Create minimal test data
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    // Create state
    let mut state = FittingState::new(x.view(), y.view());

    // Set up boundaries to enable resplit
    state.boundaries[0] = vec![3]; // Two intervals: [0,3) and [3,10)

    // Set error reductions: split=3.0 (best), resplit=2.0 (but consecutive prevention should make split win)
    state.precomputed_statistics.error_reductions_split[0] =
        vec![f64::NAN, 3.0, 3.0, f64::NAN, 3.0, 3.0, f64::NAN, 3.0];
    state.precomputed_statistics.error_reductions_resplit[0] = vec![20.0]; // resplit with very high error reduction
    state.precomputed_statistics.update_pairs_resplit_left[0] = vec![(0.0, 0.0)]; // u_plus=0, u_minus=0 means no change
    state.precomputed_statistics.update_pairs_resplit_right[0] = vec![(0.0, 0.0)];
    state.precomputed_statistics.initialized = true;

    // The loop has just taken the maximum number of consecutive resplits
    state.loop_state.consecutive_resplits = crate::grid_tensor::state::MAX_CONSECUTIVE_RESPLIT;

    // Create strategy
    let strategy = SplitStrategy::Best {
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.0,
    };

    state = strategy.initialize(state);

    let mut rng = StdRng::seed_from_u64(42);
    let action = strategy.propose_next_action(&state, &mut rng);

    // Should return a split action instead of resplit because consecutive resplit is prevented
    match action {
        Some(FittingAction::ApplySplit { .. }) => {
            // Expected: split action selected instead of resplit
        }
        _ => panic!(
            "Expected split action (consecutive resplit prevented), got: {:?}",
            action
        ),
    }
}

#[test]
fn test_best_split_handles_forbidding_correctly() {
    use crate::grid_tensor::state::FittingState;
    use ndarray::{Array1, Array2};

    // Create minimal test data
    let x = Array2::from_shape_vec(
        (10, 1),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    // Create state
    let mut state = FittingState::new(x.view(), y.view());

    // Set error reductions with clear best choice
    state.precomputed_statistics.error_reductions_split[0] =
        vec![f64::NAN, 1.0, 5.0, 3.0, 4.0, 2.0, f64::NAN, 1.0];
    state.precomputed_statistics.initialized = true;

    // Create strategy
    let strategy = SplitStrategy::Best {
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.0,
    };

    state = strategy.initialize(state);

    let mut rng = StdRng::seed_from_u64(42);
    let action1 = strategy.propose_next_action(&state, &mut rng);

    // Should pick the best split (5.0 at index 2)
    match action1 {
        Some(crate::grid_tensor::action::FittingAction::ApplySplit { split }) => {
            assert_eq!(split.col, 0);
            assert_eq!(split.index, 2);
            assert_eq!(split.error_reduction, 5.0);
        }
        _ => panic!("Expected split action, got: {:?}", action1),
    }

    // After forbidding around the split, the same position should not be available
    let split_candidate = crate::grid_tensor::splitting::SplitCandidate {
        col: 0,
        index: 2,
        error_reduction: 5.0,
        allowed_interval_idx: 0,
        update_left: (0.0, 0.0),
        update_right: (0.0, 0.0),
    };
    state = strategy.forbid_around_split(state, &split_candidate);
    let action2 = strategy.propose_next_action(&state, &mut rng);

    // Should pick the next best available split (4.0 at index 4)
    match action2 {
        Some(crate::grid_tensor::action::FittingAction::ApplySplit { split }) => {
            assert_eq!(split.col, 0);
            assert_eq!(split.index, 4); // Next best after forbidding index 2
            assert_eq!(split.error_reduction, 4.0);
        }
        _ => panic!(
            "Expected split action at different position, got: {:?}",
            action2
        ),
    }
}

#[test]
fn test_best_split_nan_handling() {
    let error_reductions = vec![vec![f64::NAN, f64::NAN, 3.0, f64::NAN, 2.0]];
    let min_interval_samples = 1;
    let (allowed_intervals, _) = initialize_intervals(&error_reductions, min_interval_samples);

    let result = sample_best_split(&allowed_intervals, &error_reductions);
    assert_eq!(result, Some((0, 2, 0, 3.0))); // Should pick 3.0 as it's better than 2.0
}
