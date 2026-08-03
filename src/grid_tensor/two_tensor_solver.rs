//! Two-Tensor Solver Module
//!
//!
//! Solves for `(u_+, u_-)`, where `v_± = 1 + u_±`, using the local
//! quadratic approximation to the per-side log-coordinate objective:
//!
//!   L_S = sum w_i (r_tilde - u_+ φ_1 - u_- φ_2)^2
//!         + α β^2 + τ δ^2 + ρ |δ|,
//!
//! with `β = 0.5 log(v_+ v_-)` (log-backbone update) and
//! `δ = 0.5 log(v_+ / v_-)` (tilt update). Candidate generation uses
//! `β ≈ (u_+ + u_-)/2` and `δ ≈ (u_+ - u_-)/2`; the returned gain
//! evaluates the exact log-coordinate penalty after multiplier clamping.
//!
//! Where:
//!   - φ_1 = f_+ * 1_S(i)
//!   - φ_2 = -f_- * 1_S(i)
//!   - S_{11} = sum w_i f_+^2
//!   - S_{22} = sum w_i f_-^2
//!   - S_{12} = -sum w_i f_+ f_-
//!   - t_1 = sum w_i r_tilde f_+
//!   - t_2 = -sum w_i r_tilde f_-

use std::f64;

/// Default hyperparameters for two-tensor solver
pub const DEFAULT_ALPHA: f64 = 0.1;
pub const DEFAULT_TAU: f64 = 0.01;
pub const DEFAULT_RHO: f64 = 0.0;
pub const DEFAULT_V_MIN: f64 = 0.05;
pub const DEFAULT_V_MAX: f64 = 20.0;

/// Condition number threshold for near-singular matrix detection
const COND_THRESHOLD: f64 = 1e12;

/// Solve the two-tensor 2×2 system
///
/// # Arguments
/// * `s11` - Sum of w_i * f_plus[i]^2 for side S
/// * `s22` - Sum of w_i * f_minus[i]^2 for side S
/// * `s12` - -Sum of w_i * f_plus[i] * f_minus[i] for side S (note: negative)
/// * `t1` - Sum of w_i * r_tilde[i] * f_plus[i] for side S
/// * `t2` - -Sum of w_i * r_tilde[i] * f_minus[i] for side S (note: negative)
/// * `alpha` - L2 log-backbone penalty strength (≥ 0)
/// * `tau` - L2 tilt penalty strength (≥ 0)
/// * `rho` - L1 tilt penalty strength (≥ 0)
/// * `v_min` - Minimum multiplier value (typically 0.05)
/// * `v_max` - Maximum multiplier value (typically 20.0)
///
/// # Returns
/// `(u_plus, u_minus, gain)` where:
/// - `u_plus` = v_plus - 1 (after clamping)
/// - `u_minus` = v_minus - 1 (after clamping)
/// - `gain` = exact penalized objective decrease at the clamped multipliers
///
/// # Panics
/// Never panics - returns (0.0, 0.0, 0.0) for near-singular or invalid cases
pub fn solve_two_tensor(
    s11: f64,
    s22: f64,
    s12: f64,
    t1: f64,
    t2: f64,
    alpha: f64,
    tau: f64,
    rho: f64,
    v_min: f64,
    v_max: f64,
) -> (f64, f64, f64) {
    // The local coordinates are m = (u_+ + u_-)/2 and
    // t = (u_+ - u_-)/2. Their quadratic penalty contributes
    // 1/4 [[α+τ, α-τ], [α-τ, α+τ]] to the system.
    let a11 = s11 + 0.25 * (alpha + tau);
    let a12 = s12 + 0.25 * (alpha - tau);
    let a21 = a12; // Symmetric
    let a22 = s22 + 0.25 * (alpha + tau);

    // Right-hand side vector t = [t1, t2]^T
    let t = [t1, t2];

    // Solve for u = [u_+, u_-]^T
    let linearized_rho = 0.5 * rho;
    let (u_plus, u_minus) = if linearized_rho == 0.0 {
        // Case 1: ρ = 0 (pure quadratic)
        solve_rho_zero(a11, a12, a21, a22, t[0], t[1])
    } else {
        // Case 2: ρ > 0 (L1 on tilt difference)
        solve_rho_positive(a11, a12, a21, a22, t[0], t[1], linearized_rho)
    };

    // Clamp multipliers to [v_min, v_max]
    let v_plus = (1.0 + u_plus).clamp(v_min, v_max);
    let v_minus = (1.0 + u_minus).clamp(v_min, v_max);

    // Recompute u after clamping
    let u_plus_clamped = v_plus - 1.0;
    let u_minus_clamped = v_minus - 1.0;

    let gain = data_loss_gain(u_plus_clamped, u_minus_clamped, s11, s22, s12, t1, t2)
        - log_coordinate_penalty(v_plus, v_minus, alpha, tau, rho);

    (u_plus_clamped, u_minus_clamped, gain)
}

/// Solve the 2×2 system when ρ = 0 (pure quadratic case)
fn solve_rho_zero(a11: f64, a12: f64, a21: f64, a22: f64, t1: f64, t2: f64) -> (f64, f64) {
    // Compute determinant
    let det = a11 * a22 - a12 * a21;

    // Check for near-singularity
    if det.abs() < 1e-15 || !det.is_finite() {
        // Near-singular: return no-op (u = 0)
        return (0.0, 0.0);
    }

    // Check condition number
    let norm_a = (a11 * a11 + a12 * a12 + a21 * a21 + a22 * a22).sqrt();
    let cond = norm_a / det.abs();
    if cond > COND_THRESHOLD {
        // Near-singular: return no-op (u = 0)
        return (0.0, 0.0);
    }

    // Solve: A u = t
    // u_+ = (a22 * t1 - a12 * t2) / det
    // u_- = (a11 * t2 - a21 * t1) / det
    let u_plus = (a22 * t1 - a12 * t2) / det;
    let u_minus = (a11 * t2 - a21 * t1) / det;

    // Check for NaN/Inf
    if !u_plus.is_finite() || !u_minus.is_finite() {
        return (0.0, 0.0);
    }

    (u_plus, u_minus)
}

/// Solve the 2×2 system when ρ > 0 (L1 penalty case)
///
/// Uses 3-case closed-form subgradient check:
/// - (+) Solve A u = t - (ρ/2) c, accept if c^T u > 0
/// - (−) Solve A u = t + (ρ/2) c, accept if c^T u < 0
/// - (0) Else project q = A^{-1} t onto hyperplane c^T u = 0
fn solve_rho_positive(
    a11: f64,
    a12: f64,
    a21: f64,
    a22: f64,
    t1: f64,
    t2: f64,
    rho: f64,
) -> (f64, f64) {
    // c = [1, -1]^T
    let c = [1.0, -1.0];
    let rho_half = rho / 2.0;

    // Compute determinant
    let det = a11 * a22 - a12 * a21;

    // Check for near-singularity
    if det.abs() < 1e-15 || !det.is_finite() {
        return (0.0, 0.0);
    }

    // Case (+): Solve A u = t - (ρ/2) c
    let t_plus = [t1 - rho_half * c[0], t2 - rho_half * c[1]];
    let u_plus_case = solve_2x2(a11, a12, a21, a22, t_plus[0], t_plus[1], det);
    if u_plus_case.is_some() {
        let (u_p, u_m) = u_plus_case.unwrap();
        let c_dot_u = c[0] * u_p + c[1] * u_m;
        if c_dot_u > 0.0 {
            return (u_p, u_m);
        }
    }

    // Case (−): Solve A u = t + (ρ/2) c
    let t_minus = [t1 + rho_half * c[0], t2 + rho_half * c[1]];
    let u_minus_case = solve_2x2(a11, a12, a21, a22, t_minus[0], t_minus[1], det);
    if u_minus_case.is_some() {
        let (u_p, u_m) = u_minus_case.unwrap();
        let c_dot_u = c[0] * u_p + c[1] * u_m;
        if c_dot_u < 0.0 {
            return (u_p, u_m);
        }
    }

    // Case (0): Project q = A^{-1} t onto hyperplane c^T u = 0
    // u^(0) = q - r * (c^T q) / (c^T r)
    // where q = A^{-1} t, r = A^{-1} c
    let q = solve_2x2(a11, a12, a21, a22, t1, t2, det);
    let r = solve_2x2(a11, a12, a21, a22, c[0], c[1], det);

    if q.is_some() && r.is_some() {
        let (q_p, q_m) = q.unwrap();
        let (r_p, r_m) = r.unwrap();
        let c_dot_q = c[0] * q_p + c[1] * q_m;
        let c_dot_r = c[0] * r_p + c[1] * r_m;

        if c_dot_r.abs() > 1e-15 {
            let u_p = q_p - r_p * (c_dot_q / c_dot_r);
            let u_m = q_m - r_m * (c_dot_q / c_dot_r);
            if u_p.is_finite() && u_m.is_finite() {
                return (u_p, u_m);
            }
        }
    }

    // Fallback: no-op
    (0.0, 0.0)
}

/// Helper to solve 2×2 system A u = t given precomputed determinant
fn solve_2x2(
    a11: f64,
    a12: f64,
    a21: f64,
    a22: f64,
    t1: f64,
    t2: f64,
    det: f64,
) -> Option<(f64, f64)> {
    if det.abs() < 1e-15 || !det.is_finite() {
        return None;
    }

    let u_plus = (a22 * t1 - a12 * t2) / det;
    let u_minus = (a11 * t2 - a21 * t1) / det;

    if u_plus.is_finite() && u_minus.is_finite() {
        Some((u_plus, u_minus))
    } else {
        None
    }
}

/// Return the frozen-weight squared-error decrease produced by an update.
///
/// The solver subtracts the exact log-coordinate penalty from this quantity
/// when it constructs the structural-action score.
#[allow(clippy::too_many_arguments)]
pub fn data_loss_gain(
    u_plus: f64,
    u_minus: f64,
    s11: f64,
    s22: f64,
    s12: f64,
    t1: f64,
    t2: f64,
) -> f64 {
    2.0 * (t1 * u_plus + t2 * u_minus)
        - (s11 * u_plus * u_plus + 2.0 * s12 * u_plus * u_minus + s22 * u_minus * u_minus)
}

/// Exact penalty on the multiplicative log-backbone and tilt coordinates.
///
/// Equal multipliers have zero tilt penalty, while reciprocal multipliers
/// have zero backbone penalty. The coordinates stay independent at every
/// multiplier scale rather than only near the no-update point.
pub fn log_coordinate_penalty(v_plus: f64, v_minus: f64, alpha: f64, tau: f64, rho: f64) -> f64 {
    if v_plus <= 0.0 || v_minus <= 0.0 {
        return f64::INFINITY;
    }
    let log_plus = v_plus.ln();
    let log_minus = v_minus.ln();
    let beta = 0.5 * (log_plus + log_minus);
    let delta = 0.5 * (log_plus - log_minus);
    alpha * beta * beta + tau * delta * delta + rho * delta.abs()
}

/// Convert (v_+, v_-) multipliers to (v_b, Δd) backbone/tilt updates
///
/// # Arguments
/// * `v_plus` - Multiplier for f_+ (must be > 0)
/// * `v_minus` - Multiplier for f_- (must be > 0)
///
/// # Returns
/// `(v_b, delta_d)` where:
/// - `v_b = sqrt(v_+ * v_-)` (backbone scaling factor)
/// - `delta_d = 0.5 * log(v_+ / v_-)` (tilt increment)
///
/// These satisfy:
/// - `v_b * exp(+delta_d) = v_+`
/// - `v_b * exp(-delta_d) = v_-`
pub fn convert_multipliers_to_backbone_tilt(v_plus: f64, v_minus: f64) -> (f64, f64) {
    debug_assert!(
        v_plus > 0.0 && v_minus > 0.0,
        "Multipliers must be positive"
    );

    let v_b = (v_plus * v_minus).sqrt();
    let delta_d = 0.5 * (v_plus / v_minus).ln();

    (v_b, delta_d)
}

#[cfg(test)]
mod tests {
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
    fn test_solve_near_singular() {
        // Near-singular case: very small determinant
        let (u_plus, u_minus, gain) = solve_two_tensor(
            1e-10, // s11
            1e-10, // s22
            1e-10, // s12 (makes det very small)
            1.0,   // t1
            1.0,   // t2
            0.0,   // alpha
            0.0,   // tau
            0.0,   // rho
            0.05,  // v_min
            20.0,  // v_max
        );

        // Should return no-op (0, 0, 0) for near-singular
        assert_eq!(u_plus, 0.0);
        assert_eq!(u_minus, 0.0);
        assert_eq!(gain, 0.0);
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
}
