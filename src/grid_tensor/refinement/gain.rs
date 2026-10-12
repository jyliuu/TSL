//! Split evaluation, local L2 update and gain formulas, Huber weights and merge factors.

use super::prefix::sub_stats;
use super::RefinementStrategy;
use crate::grid_tensor::state::{IntervalStats, PrefixStats};
use crate::grid_tensor::two_tensor_solver::{
    backbone_log_penalty, SolverParams, MAX_BACKBONE, MAX_TILT, MIN_BACKBONE,
};

pub(super) struct SplitEvaluator {
    alpha: f64,
    v_min: f64,
    v_max: f64,
    /// `None` when the configured penalties are invalid, so every two-tensor solve is a no-op.
    solver: Option<SolverParams>,
    is_stage1: bool,
}

impl SplitEvaluator {
    pub(super) fn new(strategy: &RefinementStrategy, is_stage1: bool) -> Self {
        let (v_min, v_max) = strategy.multiplier_bounds();
        Self {
            alpha: strategy.alpha(),
            v_min,
            v_max,
            solver: SolverParams::new(
                strategy.alpha(),
                strategy.tilt_tau(),
                strategy.tilt_rho(),
                v_min,
                v_max,
            ),
            is_stage1,
        }
    }

    /// Two-tensor update `(u_+, u_-)` and penalized gain from one side's solver statistics.
    #[inline]
    pub(super) fn solve(&self, s11: f64, s22: f64, s12: f64, t1: f64, t2: f64) -> (f64, f64, f64) {
        self.solver.map_or((0.0, 0.0, 0.0), |solver| {
            solver.solve(s11, s22, s12, t1, t2)
        })
    }

    #[inline]
    pub(super) fn is_stage1(&self) -> bool {
        self.is_stage1
    }

    /// Clamped stage-1 backbone multiplier `1 + u` for linear term `g` and curvature `h`.
    #[inline]
    pub(super) fn backbone_multiplier(&self, g: f64, h: f64) -> f64 {
        (1.0 + l2_update_unanchored(self.alpha, g, h)).clamp(self.v_min, self.v_max)
    }

    /// Update `(u_+, u_-)` and penalized gain of refitting one side from its solver statistics.
    #[inline]
    pub(super) fn evaluate_side(&self, side: PrefixStats) -> ((f64, f64), f64) {
        if self.is_stage1 {
            let u = self.backbone_multiplier(side.t1, side.s11) - 1.0;
            (
                (u, 0.0),
                backbone_regularized_gain(u, side.t1, side.s11, self.alpha),
            )
        } else {
            let (u_plus, u_minus, gain) =
                self.solve(side.s11, side.s22, side.s12, side.t1, side.t2);
            ((u_plus, u_minus), gain)
        }
    }

    /// Updates of both sides and their gains for splitting `total` after `left`.
    #[inline]
    pub(super) fn evaluate(
        &self,
        left: PrefixStats,
        total: PrefixStats,
    ) -> ((f64, f64), (f64, f64), (f64, f64)) {
        let (left_update, left_gain) = self.evaluate_side(left);
        let (right_update, right_gain) = self.evaluate_side(sub_stats(total, left));
        (left_update, right_update, (left_gain, right_gain))
    }
}

/// Solver statistics of the range from `base` to `value`, whose `s12` and `t2` enter the
/// solver negated.
pub(super) fn solver_stats(value: PrefixStats, base: PrefixStats) -> PrefixStats {
    PrefixStats {
        s11: value.s11 - base.s11,
        s22: value.s22 - base.s22,
        s12: -(value.s12 - base.s12),
        t1: value.t1 - base.t1,
        t2: -(value.t2 - base.t2),
    }
}

/// Solver statistics of one interval, whose `s12` and `t2` enter the solver negated.
pub(super) fn interval_solver_stats(stats: &IntervalStats) -> PrefixStats {
    PrefixStats {
        s11: stats.sum_s11,
        s22: stats.sum_s22,
        s12: -stats.sum_s12,
        t1: stats.sum_t1,
        t2: -stats.sum_t2,
    }
}

#[inline]
pub fn l2_update_unanchored(alpha: f64, s_rb: f64, s_bb: f64) -> f64 {
    let denominator = s_bb + alpha;
    if denominator > 0.0 {
        s_rb / denominator
    } else {
        0.0
    }
}

#[inline]
pub fn l2_gain_raw(update: f64, s_rb: f64, s_bb: f64) -> f64 {
    2.0 * update * s_rb - s_bb * update * update
}

#[inline]
pub(super) fn backbone_regularized_gain(update: f64, s_rb: f64, s_bb: f64, alpha: f64) -> f64 {
    l2_gain_raw(update, s_rb, s_bb) - backbone_log_penalty(1.0 + update, alpha)
}

#[inline]
pub fn huber_weighting(res: f64, c: f64) -> f64 {
    if res.abs() <= c {
        1.0
    } else {
        c / res.abs()
    }
}

pub(super) fn canonical_merge_factors(a_plus: f64, a_minus: f64, is_stage1: bool) -> (f64, f64) {
    if is_stage1 {
        let backbone = a_plus.clamp(MIN_BACKBONE, MAX_BACKBONE);
        return (backbone, backbone);
    }

    let min_axis_factor = MIN_BACKBONE * (-MAX_TILT).exp();
    let max_axis_factor = MAX_BACKBONE * MAX_TILT.exp();
    let log_plus = a_plus.clamp(min_axis_factor, max_axis_factor).ln();
    let log_minus = a_minus.clamp(min_axis_factor, max_axis_factor).ln();
    let log_backbone = (0.5 * (log_plus + log_minus)).clamp(MIN_BACKBONE.ln(), MAX_BACKBONE.ln());
    let tilt = (0.5 * (log_plus - log_minus)).clamp(-MAX_TILT, MAX_TILT);
    let backbone = log_backbone.exp();
    (backbone * tilt.exp(), backbone * (-tilt).exp())
}
