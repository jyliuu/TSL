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

/// Relative eigenvalue threshold for numerically null quadratic directions
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
/// Invalid inputs and rank-deficient systems without a finite descent update return a no-op.
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
    if [s11, s22, s12, t1, t2, alpha, tau, rho, v_min, v_max]
        .iter()
        .any(|value| !value.is_finite())
        || s11 < 0.0
        || s22 < 0.0
        || alpha < 0.0
        || tau < 0.0
        || rho < 0.0
        || v_min <= 0.0
        || v_min > 1.0
        || v_max < 1.0
    {
        return (0.0, 0.0, 0.0);
    }

    // The local coordinates are m = (u_+ + u_-)/2 and
    // t = (u_+ - u_-)/2. Their quadratic penalty contributes
    // 1/4 [[α+τ, α-τ], [α-τ, α+τ]] to the system.
    let a11 = s11 + 0.25 * (alpha + tau);
    let a12 = s12 + 0.25 * (alpha - tau);
    let a21 = a12; // Symmetric
    let a22 = s22 + 0.25 * (alpha + tau);

    let Some(rank_deficient) = quadratic_rank(a11, a12, a22) else {
        return (0.0, 0.0, 0.0);
    };
    if rank_deficient {
        // Compare feasible range and zero-tilt candidates using the exact
        // log penalty. A minimum-norm range step controls nullspace motion.
        let mut best = (0.0, 0.0, 0.0);
        let mut consider = |up: f64, um: f64| {
            let up = (1.0 + up).clamp(v_min, v_max) - 1.0;
            let um = (1.0 + um).clamp(v_min, v_max) - 1.0;
            let mut fraction = 1.0;
            for _ in 0..32 {
                let up = fraction * up;
                let um = fraction * um;
                let gain = data_loss_gain(up, um, s11, s22, s12, t1, t2)
                    - log_coordinate_penalty(1.0 + up, 1.0 + um, alpha, tau, rho);
                if gain.is_finite() && gain > 0.0 {
                    if gain > best.2 {
                        best = (up, um, gain);
                    }
                    break;
                }
                fraction *= 0.5;
            }
        };
        if rho == 0.0 {
            if let Some((up, um)) = solve_2x2(a11, a12, a22, t1, t2) {
                consider(up, um);
            }
        } else {
            for sign in [1.0, -1.0] {
                if let Some((up, um)) =
                    solve_2x2(a11, a12, a22, t1 - sign * rho / 4.0, t2 + sign * rho / 4.0)
                {
                    if sign * (up - um) > 0.0 {
                        consider(up, um);
                    }
                }
            }
            if let Some(common) = zero_tilt_candidate(a11, a12, a22, t1, t2) {
                consider(common, common);
            }
        }
        return best;
    }

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

    if gain.is_finite() {
        (u_plus_clamped, u_minus_clamped, gain)
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Solve the 2×2 system when ρ = 0 (pure quadratic case)
fn solve_rho_zero(a11: f64, a12: f64, _a21: f64, a22: f64, t1: f64, t2: f64) -> (f64, f64) {
    solve_2x2(a11, a12, a22, t1, t2).unwrap_or((0.0, 0.0))
}

/// Solve both signed tilt regions and the zero-tilt boundary.
fn solve_rho_positive(
    a11: f64,
    a12: f64,
    _a21: f64,
    a22: f64,
    t1: f64,
    t2: f64,
    rho: f64,
) -> (f64, f64) {
    let mut best = (0.0, 0.0);
    let mut best_objective = 0.0;
    let mut consider = |up: f64, um: f64| {
        let objective = a11 * up * up + 2.0 * a12 * up * um + a22 * um * um
            - 2.0 * (t1 * up + t2 * um)
            + rho * (up - um).abs();
        if objective.is_finite() && objective < best_objective {
            best = (up, um);
            best_objective = objective;
        }
    };
    for sign in [1.0, -1.0] {
        if let Some((up, um)) =
            solve_2x2(a11, a12, a22, t1 - sign * rho / 2.0, t2 + sign * rho / 2.0)
        {
            if sign * (up - um) > 0.0 {
                consider(up, um);
            }
        }
    }
    if let Some(common) = zero_tilt_candidate(a11, a12, a22, t1, t2) {
        consider(common, common);
    }
    best
}

fn zero_tilt_candidate(a11: f64, a12: f64, a22: f64, t1: f64, t2: f64) -> Option<f64> {
    let scale = a11.abs().max(a12.abs()).max(a22.abs());
    if scale <= 0.0 || !scale.is_finite() {
        return None;
    }
    let trace = a11 / scale + a22 / scale;
    let curvature = trace + 2.0 * (a12 / scale);
    if curvature <= trace / COND_THRESHOLD {
        return None;
    }
    let common = (t1 / scale + t2 / scale) / curvature;
    common.is_finite().then_some(common)
}

/// Return whether a finite positive-semidefinite system has a null direction.
fn quadratic_rank(a11: f64, a12: f64, a22: f64) -> Option<bool> {
    let scale = a11.abs().max(a12.abs()).max(a22.abs());
    if scale <= 0.0 || !scale.is_finite() {
        return None;
    }
    let a = a11 / scale;
    let b = a12 / scale;
    let c = a22 / scale;
    let determinant = a * c - b * b;
    let tolerance = (a + c).powi(2) / COND_THRESHOLD;
    if a < -1.0 / COND_THRESHOLD || c < -1.0 / COND_THRESHOLD || determinant < -tolerance {
        None
    } else {
        Some(determinant <= tolerance)
    }
}

/// Minimum-norm solution with numerically null PSD directions truncated.
fn solve_2x2(a11: f64, a12: f64, a22: f64, t1: f64, t2: f64) -> Option<(f64, f64)> {
    let scale = a11.abs().max(a12.abs()).max(a22.abs());
    if scale <= 0.0 || !scale.is_finite() {
        return None;
    }
    let a = a11 / scale;
    let b = a12 / scale;
    let c = a22 / scale;
    let t1 = t1 / scale;
    let t2 = t2 / scale;
    let determinant = a * c - b * b;
    let tolerance = (a + c).powi(2) / COND_THRESHOLD;
    if a < -1.0 / COND_THRESHOLD || c < -1.0 / COND_THRESHOLD || determinant < -tolerance {
        return None;
    }
    let solution = if determinant > tolerance {
        (
            (c * t1 - b * t2) / determinant,
            (a * t2 - b * t1) / determinant,
        )
    } else {
        let eigenvalue = 0.5 * (a + c + (a - c).hypot(2.0 * b));
        if eigenvalue <= 0.0 || !eigenvalue.is_finite() {
            return None;
        }
        let left = (b, eigenvalue - a);
        let right = (eigenvalue - c, b);
        let (mut vx, mut vy) = if left.0.hypot(left.1) >= right.0.hypot(right.1) {
            left
        } else {
            right
        };
        let norm = vx.hypot(vy);
        vx /= norm;
        vy /= norm;
        let projected = (vx * t1 + vy * t2) / eigenvalue;
        (vx * projected, vy * projected)
    };
    if solution.0.is_finite() && solution.1.is_finite() {
        Some(solution)
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
mod tests;
