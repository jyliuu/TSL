//! Two-Tensor Solver Module
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

/// Default hyperparameters for two-tensor solver
pub const DEFAULT_TAU: f64 = 0.01;
pub const DEFAULT_RHO: f64 = 0.0;
pub const DEFAULT_V_MIN: f64 = 0.05;
pub const DEFAULT_V_MAX: f64 = 20.0;

/// Relative eigenvalue threshold for numerically null quadratic directions
const COND_THRESHOLD: f64 = 1e12;
/// Bounds on applied backbone values and tilts, shared by split scoring and the reducer
/// so a scored update and the applied update clamp to the same box.
pub(crate) const MIN_BACKBONE: f64 = 1e-10;
pub(crate) const MAX_BACKBONE: f64 = 1e10;
pub(crate) const MAX_TILT: f64 = 10.0;

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
    SolverParams::new(alpha, tau, rho, v_min, v_max).map_or((0.0, 0.0, 0.0), |params| {
        params.solve(s11, s22, s12, t1, t2)
    })
}

/// Validated penalties and multiplier bounds of [`solve_two_tensor`], so callers that
/// solve many sides with one configuration check it once.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SolverParams {
    alpha: f64,
    tau: f64,
    rho: f64,
    v_min: f64,
    v_max: f64,
    /// Entries of the penalty block 1/4 [[α+τ, α-τ], [α-τ, α+τ]].
    penalty_diagonal: f64,
    penalty_off_diagonal: f64,
}

impl SolverParams {
    /// `None` for a negative or non-finite penalty, or bounds that exclude the
    /// no-update multiplier 1.
    pub(crate) fn new(alpha: f64, tau: f64, rho: f64, v_min: f64, v_max: f64) -> Option<Self> {
        if [alpha, tau, rho, v_min, v_max]
            .iter()
            .any(|value| !value.is_finite())
            || alpha < 0.0
            || tau < 0.0
            || rho < 0.0
            || v_min <= 0.0
            || v_min > 1.0
            || v_max < 1.0
        {
            return None;
        }
        Some(Self {
            alpha,
            tau,
            rho,
            v_min,
            v_max,
            penalty_diagonal: 0.25 * (alpha + tau),
            penalty_off_diagonal: 0.25 * (alpha - tau),
        })
    }

    /// [`solve_two_tensor`] with these parameters.
    pub(crate) fn solve(&self, s11: f64, s22: f64, s12: f64, t1: f64, t2: f64) -> (f64, f64, f64) {
        let Self {
            alpha,
            tau,
            rho,
            v_min,
            v_max,
            ..
        } = *self;
        if [s11, s22, s12, t1, t2]
            .iter()
            .any(|value| !value.is_finite())
            || s11 < 0.0
            || s22 < 0.0
        {
            return (0.0, 0.0, 0.0);
        }

        // The local coordinates are m = (u_+ + u_-)/2 and
        // t = (u_+ - u_-)/2. Their quadratic penalty contributes
        // 1/4 [[α+τ, α-τ], [α-τ, α+τ]] to the system.
        let Some(system) = Quadratic::new(
            s11 + self.penalty_diagonal,
            s12 + self.penalty_off_diagonal,
            s22 + self.penalty_diagonal,
        ) else {
            return (0.0, 0.0, 0.0);
        };
        if system.rank_deficient() {
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
                if let Some((up, um)) = system.solve(t1, t2) {
                    consider(up, um);
                }
            } else {
                for sign in [1.0, -1.0] {
                    if let Some((up, um)) =
                        system.solve(t1 - sign * rho / 4.0, t2 + sign * rho / 4.0)
                    {
                        if sign * (up - um) > 0.0 {
                            consider(up, um);
                        }
                    }
                }
                if let Some(common) = system.zero_tilt(t1, t2) {
                    consider(common, common);
                }
            }
            return best;
        }

        let linearized_rho = 0.5 * rho;
        let (u_plus, u_minus) = if linearized_rho == 0.0 {
            system.solve(t1, t2).unwrap_or((0.0, 0.0))
        } else {
            system.solve_l1(t1, t2, linearized_rho)
        };

        let v_plus = (1.0 + u_plus).clamp(v_min, v_max);
        let v_minus = (1.0 + u_minus).clamp(v_min, v_max);
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
}

/// A finite positive-semidefinite system `[[a11, a12], [a12, a22]]` with its entries
/// normalized by the largest magnitude, which sets the null-direction tolerance.
struct Quadratic {
    a11: f64,
    a12: f64,
    a22: f64,
    scale: f64,
    a: f64,
    b: f64,
    c: f64,
    determinant: f64,
    tolerance: f64,
}

// `SolverParams::solve` runs once per candidate split position. Inlining these
// methods keeps the scaled system in registers instead of writing it to the
// stack on every call, which dominates exact-candidate fits.
impl Quadratic {
    /// `None` for a zero, non-finite or indefinite system.
    #[inline(always)]
    fn new(a11: f64, a12: f64, a22: f64) -> Option<Self> {
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
            return None;
        }
        Some(Self {
            a11,
            a12,
            a22,
            scale,
            a,
            b,
            c,
            determinant,
            tolerance,
        })
    }

    /// Whether the system has a numerically null direction.
    #[inline(always)]
    fn rank_deficient(&self) -> bool {
        self.determinant <= self.tolerance
    }

    /// Minimum-norm solution with numerically null directions truncated.
    #[inline(always)]
    fn solve(&self, t1: f64, t2: f64) -> Option<(f64, f64)> {
        let Self {
            a,
            b,
            c,
            determinant,
            ..
        } = *self;
        let t1 = t1 / self.scale;
        let t2 = t2 / self.scale;
        let solution = if determinant > self.tolerance {
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

    /// Common update `u_+ = u_-` minimizing the quadratic along the zero-tilt direction.
    #[inline(always)]
    fn zero_tilt(&self, t1: f64, t2: f64) -> Option<f64> {
        let trace = self.a + self.c;
        let curvature = trace + 2.0 * self.b;
        if curvature <= trace / COND_THRESHOLD {
            return None;
        }
        let common = (t1 / self.scale + t2 / self.scale) / curvature;
        common.is_finite().then_some(common)
    }

    /// Minimize the quadratic plus `rho |u_+ - u_-|` over both signed tilt regions and
    /// the zero-tilt boundary.
    fn solve_l1(&self, t1: f64, t2: f64, rho: f64) -> (f64, f64) {
        let Self { a11, a12, a22, .. } = *self;
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
            if let Some((up, um)) = self.solve(t1 - sign * rho / 2.0, t2 + sign * rho / 2.0) {
                if sign * (up - um) > 0.0 {
                    consider(up, um);
                }
            }
        }
        if let Some(common) = self.zero_tilt(t1, t2) {
            consider(common, common);
        }
        best
    }
}

/// Return the frozen-weight squared-error decrease produced by an update.
///
/// The solver subtracts the exact log-coordinate penalty from this quantity
/// when it constructs the structural-action score.
#[allow(clippy::too_many_arguments)]
#[inline]
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
#[inline]
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

/// [`log_coordinate_penalty`]`(v, v, alpha, 0, 0)` evaluated with one logarithm.
///
/// The tilt `0.5 (ln v − ln v)` and its zero-weighted terms stay in the sum so non-finite
/// logarithms and signed zeros propagate exactly as in the two-multiplier form.
#[allow(clippy::eq_op)]
#[inline]
pub(crate) fn backbone_log_penalty(v: f64, alpha: f64) -> f64 {
    if v <= 0.0 {
        return f64::INFINITY;
    }
    let log_v = v.ln();
    let beta = 0.5 * (log_v + log_v);
    let delta = 0.5 * (log_v - log_v);
    alpha * beta * beta + 0.0 * delta * delta + 0.0 * delta.abs()
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

    const NO_UPDATE: (f64, f64, f64) = (0.0, 0.0, 0.0);

    /// [`solve_two_tensor`] for `[s11, s22, s12, t1, t2]` and penalties `[α, τ, ρ]` within
    /// the default multiplier bounds.
    fn solve([s11, s22, s12, t1, t2]: [f64; 5], [alpha, tau, rho]: [f64; 3]) -> (f64, f64, f64) {
        solve_two_tensor(
            s11,
            s22,
            s12,
            t1,
            t2,
            alpha,
            tau,
            rho,
            DEFAULT_V_MIN,
            DEFAULT_V_MAX,
        )
    }

    /// The multipliers `1 + u±` lie within the bounds and the gain is the data gain less the
    /// exact log-coordinate penalty at those multipliers.
    fn assert_exact_gain(system: [f64; 5], penalties: [f64; 3], solution: (f64, f64, f64)) {
        let ([s11, s22, s12, t1, t2], [alpha, tau, rho]) = (system, penalties);
        let (u_plus, u_minus, gain) = solution;
        for v in [1.0 + u_plus, 1.0 + u_minus] {
            assert!((DEFAULT_V_MIN - 1e-12..=DEFAULT_V_MAX + 1e-12).contains(&v));
        }
        let expected = data_loss_gain(u_plus, u_minus, s11, s22, s12, t1, t2)
            - log_coordinate_penalty(1.0 + u_plus, 1.0 + u_minus, alpha, tau, rho);
        assert!(
            (gain - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{system:?} {penalties:?}: gain {gain}, expected {expected}"
        );
    }

    /// Full-rank, coupled and rank-one systems, with targets inside and beyond the multiplier
    /// bounds, under every penalty combination.
    #[test]
    fn gain_is_the_exact_penalized_decrease_at_bounded_multipliers() {
        let systems = [
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [2.0, 3.0, -0.25, 1.5, -0.5],
            [1.0, 1.0, -1.0, 0.2, -0.2],
            [100.0, 1.0, 0.0, 1e4, 0.0],
            [1.0, 1.0, 0.0, 0.0, -5.0],
        ];
        let penalties = [
            [0.0; 3],
            [0.1, 0.01, 0.0],
            [0.7, 0.4, 0.2],
            [0.0, 5.0, 0.1],
            [100.0, 100.0, 1.0],
        ];
        for system in systems {
            for penalties in penalties {
                assert_exact_gain(system, penalties, solve(system, penalties));
            }
        }
        let (u_plus, _, _) = solve([100.0, 1.0, 0.0, 1e4, 0.0], [0.0; 3]);
        assert!((1.0 + u_plus - DEFAULT_V_MAX).abs() < 1e-12);
        let (_, u_minus, _) = solve([1.0, 1.0, 0.0, 0.0, -5.0], [0.0; 3]);
        assert!((1.0 + u_minus - DEFAULT_V_MIN).abs() < 1e-12);
    }

    /// Scaling the statistics and penalties by a common factor leaves the update unchanged
    /// and scales the gain, up to the extremes of the floating-point range.
    #[test]
    fn update_is_invariant_to_uniform_system_scale() {
        for [s11, s22, s12, alpha, tau] in [
            [2.0, 2.0, -2.0, 0.0, 0.01],
            [2.0, 3.0, -0.4, 0.3, 0.01],
            [1.002001, 1.0, -1.001, 0.0, 0.0],
        ] {
            for rho in [0.0, 0.001, 0.1] {
                let (system, penalties) = ([s11, s22, s12, 0.4, -0.4], [alpha, tau, rho]);
                let reference = solve(system, penalties);
                for scale in [1e-100, 1e100] {
                    let result = solve(system.map(|v| v * scale), penalties.map(|v| v * scale));
                    assert!((result.0 - reference.0).abs() < 1e-10);
                    assert!((result.1 - reference.1).abs() < 1e-10);
                    assert!((result.2 / scale - reference.2).abs() < 1e-10);
                }
            }
        }
    }

    /// Rank-one systems get a bounded descent update with exact positive gain: the
    /// minimum-norm step along the identified range direction, its tilt soft-thresholded by
    /// the L1 penalty, or the zero-tilt step when that penalty outweighs the tilt.
    #[test]
    fn rank_one_systems_get_a_bounded_descent_update_with_exact_gain() {
        // The pure-tilt step of the symmetric system, soft-thresholded by ρ.
        let tilt_step = |rho: f64| {
            let half = 0.5 * (0.4 - rho / 4.0) / (2.0 + 0.01 / 4.0);
            Some((half, -half))
        };
        let zero_tilt_step = Some((0.2, 0.2));
        let p = 1.001;
        // System, penalties and the closed-form update (u₊, u₋) where one is known.
        let cases = [
            (
                [2.0, 2.0, -2.0, 0.4, -0.4],
                [0.0, 0.01, 0.0],
                tilt_step(0.0),
            ),
            (
                [2.0, 2.0, -2.0, 0.4, -0.4],
                [0.0, 0.01, 0.2],
                tilt_step(0.2),
            ),
            ([4.0, 1.0, -2.0, 0.8, -0.4], [0.0; 3], Some((0.16, -0.08))),
            (
                [1.0, 4.0, -2.0, -0.2, 0.4],
                [0.0, 0.0, 10.0],
                zero_tilt_step,
            ),
            ([p * p, 1.0, -p, p, -1.0], [0.0, 0.0, 0.001], None),
            ([1.0, 1.0, -1.0, 0.2, -0.2], [0.0, 5.0, 0.1], None),
            ([1.0, 1.0, -1.0, 20.0, -20.0], [0.0, 5.0, 0.1], None),
            ([1.0, 1.0, -1.0, -20.0, 20.0], [0.0, 5.0, 0.1], None),
        ];
        for (system, penalties, expected) in cases {
            let solution = solve(system, penalties);
            assert!(solution.2 > 0.0, "{system:?} {penalties:?}: {solution:?}");
            assert_exact_gain(system, penalties, solution);
            if let Some((u_plus, u_minus)) = expected {
                assert!((solution.0 - u_plus).abs() < 1e-12, "{solution:?}");
                assert!((solution.1 - u_minus).abs() < 1e-12, "{solution:?}");
            }
        }
        // Candidates are compared at their clamped multipliers, where the range step beats
        // the far zero-tilt step.
        assert!(solve([p * p, 1.0, -p, p, -1.0], [0.0, 0.0, 0.001]).2 > 0.99);
    }

    /// Non-finite or negative statistics, indefinite systems, and penalties or bounds that are
    /// invalid give no update; a system indefinite only by roundoff still gets one.
    #[test]
    fn invalid_and_indefinite_systems_have_no_update() {
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            assert_eq!(solve([bad, 1.0, -0.5, 0.2, -0.2], [0.0; 3]), NO_UPDATE);
            assert_eq!(
                solve([1.0, 1.0, -0.5, 0.2, -0.2], [bad, 0.0, 0.0]),
                NO_UPDATE
            );
        }
        for rho in [0.0, 0.1] {
            for cross in [-2.0, 2.0] {
                assert_eq!(
                    solve([1.0, 1.0, cross, 0.2, -0.2], [0.0, 0.0, rho]),
                    NO_UPDATE
                );
            }
        }
        assert_eq!(
            solve_two_tensor(1.0, 1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 1.5, DEFAULT_V_MAX),
            NO_UPDATE
        );
        assert!(solve([1.0, 1.0, -1.0 - 1e-14, 0.2, -0.2], [0.0; 3]).2 > 0.0);
    }

    /// The L1 tilt penalty soft-thresholds the tilt: the tilt update shrinks as ρ grows and
    /// vanishes beyond a threshold, leaving a pure backbone update, or no update at all when
    /// the target is pure tilt.
    #[test]
    fn l1_tilt_penalty_soft_thresholds_the_tilt_update() {
        let mut previous_tilt = f64::INFINITY;
        for rho in [0.0, 0.1, 0.5, 1.0, 2.0, 10.0] {
            let (u_plus, u_minus, gain) = solve([1.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, rho]);
            let tilt = u_plus - u_minus;
            assert!(gain > 0.0);
            assert!(
                tilt > -1e-12 && tilt <= previous_tilt + 1e-12,
                "rho {rho}: tilt {tilt}"
            );
            previous_tilt = tilt;
        }
        assert!(previous_tilt.abs() < 1e-12);
        assert_eq!(
            solve([2.0, 2.0, -2.0, 0.4, -0.4], [0.0, 0.01, 2.0]),
            NO_UPDATE
        );
    }

    /// The backbone penalty acts only on β = ½ ln(v₊v₋) and the tilt penalties only on
    /// δ = ½ ln(v₊/v₋) at every multiplier scale, so penalizing one coordinate leaves an
    /// update along the other unchanged.
    #[test]
    fn penalties_act_on_separate_log_coordinates() {
        assert!(log_coordinate_penalty(2.0, 0.5, 5.0, 0.0, 0.0).abs() < 1e-12);
        assert!(log_coordinate_penalty(2.0, 2.0, 0.0, 5.0, 1.0).abs() < 1e-12);
        let tilt_penalty = |v_plus, v_minus| log_coordinate_penalty(v_plus, v_minus, 0.0, 3.0, 0.0);
        assert!((tilt_penalty(2.0, 0.5) - tilt_penalty(4.0, 1.0)).abs() < 1e-12);

        for (target, penalties) in [(-0.2, [100.0, 0.0, 0.0]), (0.2, [0.0, 100.0, 0.0])] {
            let system = [1.0, 1.0, 0.0, 0.2, target];
            let (free, penalized) = (solve(system, [0.0; 3]), solve(system, penalties));
            assert!((free.0 - penalized.0).abs() < 1e-12);
            assert!((free.1 - penalized.1).abs() < 1e-12);
        }
    }

    #[test]
    fn multipliers_round_trip_through_backbone_and_tilt() {
        for (v_plus, v_minus) in [(2.0, 0.5), (DEFAULT_V_MIN, DEFAULT_V_MAX), (3.0, 3.0)] {
            let (v_b, delta_d) = convert_multipliers_to_backbone_tilt(v_plus, v_minus);
            assert!((v_b * delta_d.exp() - v_plus).abs() <= 1e-12 * v_plus);
            assert!((v_b * (-delta_d).exp() - v_minus).abs() <= 1e-12 * v_minus);
        }
    }
}
