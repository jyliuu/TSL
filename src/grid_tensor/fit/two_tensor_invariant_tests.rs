use super::*;
use crate::grid_tensor::{
    action::FittingAction, reducer::fitting_reducer, refinement::RefinementStrategy,
    splitting::SplitCandidate, splitting::SplitStrategy,
};
use ndarray::{Array1, Array2};

/// Test that two-tensor invariants are maintained after fitting actions
/// Invariants: lambda_plus > 0, lambda_minus > 0, b > 0, f_plus >= 0, f_minus >= 0.
/// Mixed-sign y is required to keep the fitter out of stage-1 positive-only
/// mode (where lambda_minus is allowed to be 0).
#[test]
fn test_two_tensor_positivity_invariants() {
    // Create simple 2D dataset with mixed-sign y so we exercise the full
    // two-tensor path (not stage-1 positive-only).
    let n = 20;
    let p = 2;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    for i in 0..n {
        x_data.push(vec![i as f64 / n as f64, (i % 10) as f64 / 10.0]);
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        y_data.push(sign * ((i as f64 / n as f64) * 2.0 + 1.0));
    }
    let x = Array2::from_shape_vec((n, p), x_data.into_iter().flatten().collect()).unwrap();
    let y = Array1::from_vec(y_data);

    let refinement_strategy = RefinementStrategy::L2Refinement {
        alpha: 0.01,
        tilt_tau: crate::grid_tensor::two_tensor_solver::DEFAULT_TAU,
        tilt_rho: crate::grid_tensor::two_tensor_solver::DEFAULT_RHO,
        prior_sample_size: 0.0,
        update_clamp: f64::INFINITY,
    };
    let split_strategy = SplitStrategy::Best {
        min_interval_samples: 1,
        complexity_penalty: 0.0,
        min_split_loss: 0.0,
    };

    // Initialize state
    let mut state = FittingState::new(x.view(), y.view());
    state = refinement_strategy.initialize(state);
    state = split_strategy.initialize(state);

    // Check initial invariants
    assert!(state.lambda_plus > 0.0, "lambda_plus must be positive");
    assert!(state.lambda_minus > 0.0, "lambda_minus must be positive");
    for col in 0..state.p {
        for k in 0..state.backbone_values[col].len() {
            assert!(
                state.backbone_values[col][k] > 0.0,
                "backbone[{}][{}] = {} must be positive",
                col,
                k,
                state.backbone_values[col][k]
            );
        }
    }
    for i in 0..state.n {
        assert!(
            state.f_plus[i] >= 0.0,
            "f_plus[{}] = {} must be non-negative",
            i,
            state.f_plus[i]
        );
        assert!(
            state.f_minus[i] >= 0.0,
            "f_minus[{}] = {} must be non-negative",
            i,
            state.f_minus[i]
        );
        assert!(
            !state.f_plus[i].is_nan() && !state.f_plus[i].is_infinite(),
            "f_plus[{}] must be finite",
            i
        );
        assert!(
            !state.f_minus[i].is_nan() && !state.f_minus[i].is_infinite(),
            "f_minus[{}] must be finite",
            i
        );
    }

    // Apply a split
    if state.precomputed_statistics.error_reductions_split[0].len() > 5 {
        let idx = 5;
        let action = FittingAction::ApplySplit {
            split: {
                let update_left = state.precomputed_statistics.update_pairs_split_left[0][idx];
                let update_right = state.precomputed_statistics.update_pairs_split_right[0][idx];
                #[allow(deprecated)]
                SplitCandidate {
                    col: 0,
                    error_reduction: state.precomputed_statistics.error_reductions_split[0][idx],
                    allowed_interval_idx: 0,
                    index: idx,
                    update_left,
                    update_right,
                }
            },
        };
        state = fitting_reducer(state, action, &refinement_strategy, &split_strategy);

        // Check invariants after split
        assert!(
            state.lambda_plus > 0.0,
            "lambda_plus must remain positive after split"
        );
        assert!(
            state.lambda_minus > 0.0,
            "lambda_minus must remain positive after split"
        );
        for col in 0..state.p {
            for k in 0..state.backbone_values[col].len() {
                assert!(
                    state.backbone_values[col][k] > 0.0,
                    "backbone[{}][{}] = {} must remain positive after split",
                    col,
                    k,
                    state.backbone_values[col][k]
                );
            }
        }
        for i in 0..state.n {
            assert!(
                state.f_plus[i] >= 0.0,
                "f_plus[{}] = {} must remain non-negative after split",
                i,
                state.f_plus[i]
            );
            assert!(
                state.f_minus[i] >= 0.0,
                "f_minus[{}] = {} must remain non-negative after split",
                i,
                state.f_minus[i]
            );
            assert!(
                !state.f_plus[i].is_nan() && !state.f_minus[i].is_infinite(),
                "f_plus[{}] must remain finite after split",
                i
            );
            assert!(
                !state.f_minus[i].is_nan() && !state.f_minus[i].is_infinite(),
                "f_minus[{}] must remain finite after split",
                i
            );
        }
    }
}

/// Test that solver clamping enforces v_± ∈ [v_min, v_max]
#[test]
fn test_solver_clamping_enforces_bounds() {
    use crate::grid_tensor::two_tensor_solver::solve_two_tensor;

    // Test case that would produce v_+ > v_max without clamping
    let (u_plus, u_minus, _gain) = solve_two_tensor(
        100.0,  // s11
        100.0,  // s22
        0.0,    // s12
        1000.0, // t1 (large, would push u_+ high)
        0.0,    // t2
        0.01,   // alpha
        0.0,    // tau
        0.0,    // rho
        0.05,   // v_min
        20.0,   // v_max
    );

    let v_plus = 1.0 + u_plus;
    let v_minus = 1.0 + u_minus;

    assert!(
        (0.05..=20.0).contains(&v_plus),
        "v_plus = {} must be in [0.05, 20.0]",
        v_plus
    );
    assert!(
        (0.05..=20.0).contains(&v_minus),
        "v_minus = {} must be in [0.05, 20.0]",
        v_minus
    );
}
