//! Split evaluation, local L2 update and gain formulas, Huber weights and merge factors.

use super::RefinementStrategy;
use crate::grid_tensor::state::PrefixStats;
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
    pub(super) fn evaluate(
        &self,
        left: PrefixStats,
        total: PrefixStats,
    ) -> ((f64, f64), (f64, f64), (f64, f64)) {
        let Self {
            alpha,
            v_min,
            v_max,
            is_stage1,
            ..
        } = *self;
        if is_stage1 {
            let h_l = left.s11;
            let g_l = left.t1;
            let h_r = total.s11 - h_l;
            let g_r = total.t1 - g_l;
            let u_l = l2_update_unanchored(alpha, g_l, h_l);
            let v_b_l = (1.0 + u_l).clamp(v_min, v_max);
            let u_l_clamped = v_b_l - 1.0;
            let gain_l = backbone_regularized_gain(u_l_clamped, g_l, h_l, alpha);
            let u_r = l2_update_unanchored(alpha, g_r, h_r);
            let v_b_r = (1.0 + u_r).clamp(v_min, v_max);
            let u_r_clamped = v_b_r - 1.0;
            let gain_r = backbone_regularized_gain(u_r_clamped, g_r, h_r, alpha);
            ((u_l_clamped, 0.0), (u_r_clamped, 0.0), (gain_l, gain_r))
        } else {
            let s11_r = total.s11 - left.s11;
            let s22_r = total.s22 - left.s22;
            let s12_r = total.s12 - left.s12;
            let t1_r = total.t1 - left.t1;
            let t2_r = total.t2 - left.t2;
            let (u_plus_l, u_minus_l, gain_l) =
                self.solve(left.s11, left.s22, left.s12, left.t1, left.t2);
            let (u_plus_r, u_minus_r, gain_r) = self.solve(s11_r, s22_r, s12_r, t1_r, t2_r);
            (
                (u_plus_l, u_minus_l),
                (u_plus_r, u_minus_r),
                (gain_l, gain_r),
            )
        }
    }
}

pub(super) fn solver_stats(value: PrefixStats, base: PrefixStats) -> PrefixStats {
    PrefixStats {
        s11: value.s11 - base.s11,
        s22: value.s22 - base.s22,
        s12: -(value.s12 - base.s12),
        t1: value.t1 - base.t1,
        t2: -(value.t2 - base.t2),
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
