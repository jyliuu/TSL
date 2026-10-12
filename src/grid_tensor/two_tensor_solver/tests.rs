use super::*;

#[test]
fn test_solve_rho_zero_simple() {
    let (u_plus, u_minus, gain) = solve_two_tensor(
        1.0,  // s11
        1.0,  // s22
        0.0,  // s12
        1.0,  // t1
        0.0,  // t2
        0.1,  // alpha
        0.01, // tau
        0.0,  // rho
        0.05, // v_min
        20.0, // v_max
    );

    assert!(u_plus > 0.8 && u_plus < 1.0);
    assert!(u_minus.abs() < 0.1);
    assert!(gain > 0.0);
}

#[test]
fn returned_gain_includes_exact_log_coordinate_penalty() {
    let (u_plus, u_minus, penalized_gain) =
        solve_two_tensor(2.0, 3.0, -0.25, 1.5, -0.5, 0.7, 0.4, 0.2, 0.05, 20.0);
    let data_gain = data_loss_gain(u_plus, u_minus, 2.0, 3.0, -0.25, 1.5, -0.5);
    let penalty = log_coordinate_penalty(1.0 + u_plus, 1.0 + u_minus, 0.7, 0.4, 0.2);

    assert!(data_gain.is_finite());
    assert!((data_gain - penalty - penalized_gain).abs() < 1e-12);
}

#[test]
fn backbone_and_tilt_penalties_are_scale_separated() {
    let tilt_only_at_unit_backbone = log_coordinate_penalty(2.0, 0.5, 0.0, 3.0, 0.0);
    let same_tilt_at_double_backbone = log_coordinate_penalty(4.0, 1.0, 0.0, 3.0, 0.0);
    assert!((tilt_only_at_unit_backbone - same_tilt_at_double_backbone).abs() < 1e-12);

    let reciprocal_backbone_penalty = log_coordinate_penalty(2.0, 0.5, 5.0, 0.0, 0.0);
    let equal_multiplier_tilt_penalty = log_coordinate_penalty(2.0, 2.0, 0.0, 5.0, 1.0);
    assert!(reciprocal_backbone_penalty.abs() < 1e-12);
    assert!(equal_multiplier_tilt_penalty.abs() < 1e-12);
}

#[test]
fn local_backbone_curvature_does_not_shrink_tilt_direction() {
    let unpenalized = solve_two_tensor(1.0, 1.0, 0.0, 0.2, -0.2, 0.0, 0.0, 0.0, 0.05, 20.0);
    let backbone_penalized =
        solve_two_tensor(1.0, 1.0, 0.0, 0.2, -0.2, 100.0, 0.0, 0.0, 0.05, 20.0);

    assert!((unpenalized.0 - backbone_penalized.0).abs() < 1e-12);
    assert!((unpenalized.1 - backbone_penalized.1).abs() < 1e-12);
}

#[test]
fn local_tilt_curvature_does_not_shrink_backbone_direction() {
    let unpenalized = solve_two_tensor(1.0, 1.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.05, 20.0);
    let tilt_penalized = solve_two_tensor(1.0, 1.0, 0.0, 0.2, 0.2, 0.0, 100.0, 0.0, 0.05, 20.0);

    assert!((unpenalized.0 - tilt_penalized.0).abs() < 1e-12);
    assert!((unpenalized.1 - tilt_penalized.1).abs() < 1e-12);
}

#[test]
fn rank_one_tilt_direction_has_a_minimum_norm_descent_update() {
    let (up, um, gain) = solve_two_tensor(2.0, 2.0, -2.0, 0.4, -0.4, 0.0, 0.01, 0.0, 0.05, 20.0);
    let expected_difference = 0.4 / (2.0 + 0.01 / 4.0);
    assert!((up - um - expected_difference).abs() < 1e-12);
    assert!((up + um).abs() < 1e-12);
    assert!(gain > 0.0);
}

#[test]
fn rank_one_tilt_l1_penalty_uses_soft_thresholding() {
    let rho = 0.2;
    let (up, um, gain) = solve_two_tensor(2.0, 2.0, -2.0, 0.4, -0.4, 0.0, 0.01, rho, 0.05, 20.0);
    let expected_difference = (0.4 - rho / 4.0) / (2.0 + 0.01 / 4.0);
    assert!((up - um - expected_difference).abs() < 1e-12);
    assert!((up + um).abs() < 1e-12);
    assert!(gain > 0.0);
    assert_eq!(
        solve_two_tensor(2.0, 2.0, -2.0, 0.4, -0.4, 0.0, 0.01, 2.0, 0.05, 20.0),
        (0.0, 0.0, 0.0)
    );
}

#[test]
fn quadratic_update_is_invariant_to_uniform_system_scale() {
    for (s11, s22, s12, alpha, tau) in [
        (2.0, 2.0, -2.0, 0.0, 0.01),
        (2.0, 3.0, -0.4, 0.3, 0.01),
        (1.002001, 1.0, -1.001, 0.0, 0.0),
    ] {
        for rho in [0.0, 0.001, 0.1] {
            let reference = solve_two_tensor(s11, s22, s12, 0.4, -0.4, alpha, tau, rho, 0.05, 20.0);
            for scale in [1e-100, 1e100] {
                let result = solve_two_tensor(
                    s11 * scale,
                    s22 * scale,
                    s12 * scale,
                    0.4 * scale,
                    -0.4 * scale,
                    alpha * scale,
                    tau * scale,
                    rho * scale,
                    0.05,
                    20.0,
                );
                assert!((result.0 - reference.0).abs() < 1e-10);
                assert!((result.1 - reference.1).abs() < 1e-10);
                assert!((result.2 / scale - reference.2).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn asymmetric_rank_one_system_uses_the_identified_direction() {
    let (up, um, gain) = solve_two_tensor(4.0, 1.0, -2.0, 0.8, -0.4, 0.0, 0.0, 0.0, 0.05, 20.0);
    assert!((up - 0.16).abs() < 1e-12);
    assert!((um + 0.08).abs() < 1e-12);
    assert!((gain - 0.16).abs() < 1e-12);
}

#[test]
fn rank_one_l1_candidates_are_compared_after_multiplier_bounds() {
    let p = 1.001;
    let (up, um, gain) = solve_two_tensor(p * p, 1.0, -p, p, -1.0, 0.0, 0.0, 0.001, 0.05, 20.0);
    assert!(up.abs() < 1.0 && um.abs() < 1.0);
    assert!(gain > 0.99);
}

#[test]
fn rank_deficient_clamped_update_has_exact_positive_gain() {
    for target in [0.2, 2.0, 20.0, -20.0] {
        let (up, um, gain) =
            solve_two_tensor(1.0, 1.0, -1.0, target, -target, 0.0, 5.0, 0.1, 0.05, 20.0);
        assert!((0.05 - 1e-14..=20.0).contains(&(1.0 + up)));
        assert!((0.05 - 1e-14..=20.0).contains(&(1.0 + um)));
        assert!(gain > 0.0);
        let expected = data_loss_gain(up, um, 1.0, 1.0, -1.0, target, -target)
            - log_coordinate_penalty(1.0 + up, 1.0 + um, 0.0, 5.0, 0.1);
        assert!((gain - expected).abs() < 1e-12);
    }
}

#[test]
fn rank_one_l1_solution_can_use_the_zero_tilt_direction() {
    let (up, um, gain) = solve_two_tensor(1.0, 4.0, -2.0, -0.2, 0.4, 0.0, 0.0, 10.0, 0.05, 20.0);
    assert!((up - um).abs() < 1e-12);
    assert!(up > 0.0 && gain > 0.0);
}

#[test]
fn invalid_and_indefinite_systems_have_no_update() {
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        assert_eq!(
            solve_two_tensor(bad, 1.0, -0.5, 0.2, -0.2, 0.0, 0.01, 0.0, 0.05, 20.0),
            (0.0, 0.0, 0.0)
        );
    }
    for rho in [0.0, 0.1] {
        for cross in [-2.0, 2.0] {
            assert_eq!(
                solve_two_tensor(1.0, 1.0, cross, 0.2, -0.2, 0.0, 0.0, rho, 0.05, 20.0),
                (0.0, 0.0, 0.0)
            );
        }
    }
    let roundoff = solve_two_tensor(1.0, 1.0, -1.0 - 1e-14, 0.2, -0.2, 0.0, 0.0, 0.0, 0.05, 20.0);
    assert!(roundoff.2 > 0.0);
}

#[test]
fn test_clamping() {
    // Case that would produce v_+ > v_max
    let (u_plus, _u_minus, _gain) = solve_two_tensor(
        100.0, // Large s11
        1.0, 0.0, 1000.0, // Large t1
        0.0, 0.0,  // alpha
        0.0,  // tau
        0.0,  // rho
        0.05, // v_min
        20.0, // v_max
    );

    let v_plus = 1.0 + u_plus;
    assert!(v_plus <= 20.0, "v_plus should be clamped to v_max");
    assert!(v_plus >= 0.05, "v_plus should be clamped to v_min");
}

#[test]
fn test_convert_multipliers_to_backbone_tilt() {
    let v_plus = 2.0;
    let v_minus = 0.5;

    let (v_b, delta_d) = convert_multipliers_to_backbone_tilt(v_plus, v_minus);

    // v_b = sqrt(2.0 * 0.5) = sqrt(1.0) = 1.0
    assert!((v_b - 1.0).abs() < 1e-10);

    // delta_d = 0.5 * ln(2.0 / 0.5) = 0.5 * ln(4.0) = 0.5 * 1.386... ≈ 0.693
    let expected_delta_d = 0.5 * (2.0f64 / 0.5f64).ln();
    assert!((delta_d - expected_delta_d).abs() < 1e-10);

    // Verify: v_b * exp(+delta_d) = v_+
    let v_plus_reconstructed = v_b * delta_d.exp();
    assert!((v_plus_reconstructed - v_plus).abs() < 1e-10);

    // Verify: v_b * exp(-delta_d) = v_-
    let v_minus_reconstructed = v_b * (-delta_d).exp();
    assert!((v_minus_reconstructed - v_minus).abs() < 1e-10);
}

#[test]
fn test_rho_positive_case() {
    // Test rho > 0 case
    let (u_plus, u_minus, gain) = solve_two_tensor(
        1.0, 1.0, 0.0, 1.0, 0.0, 0.1,  // alpha
        0.01, // tau
        0.1,  // rho > 0
        0.05, // v_min
        20.0, // v_max
    );

    // Should produce valid solution
    assert!(u_plus.is_finite());
    assert!(u_minus.is_finite());
    assert!(gain.is_finite());
}

#[test]
fn l1_tilt_penalty_can_select_an_exact_backbone_update() {
    let (u_plus, u_minus, gain) =
        solve_two_tensor(1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 10.0, 0.05, 20.0);

    assert!((u_plus - u_minus).abs() < 1e-12);
    assert!(gain.is_finite());
}
