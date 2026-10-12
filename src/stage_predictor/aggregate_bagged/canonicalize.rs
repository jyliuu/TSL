//! Per-bag gauge fixing and conversions between the `(b, d)` and `a±` factor forms.

use crate::grid_tensor::GridTensor;

use super::{clamped_ln, EPSILON};

pub(super) fn compute_a_factors_from_bd(
    backbone: &[Vec<f64>],
    tilt: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let num_axes = backbone.len();
    let mut a_plus: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
    let mut a_minus: Vec<Vec<f64>> = Vec::with_capacity(num_axes);

    for axis in 0..num_axes {
        let n_bins = backbone[axis].len();
        let mut a_plus_axis = Vec::with_capacity(n_bins);
        let mut a_minus_axis = Vec::with_capacity(n_bins);

        for bin in 0..n_bins {
            let b = backbone[axis][bin].max(EPSILON);
            let d = tilt[axis][bin];
            let exp_d = d.min(50.0).exp();
            let exp_neg_d = (-d).min(50.0).exp();
            a_plus_axis.push(b * exp_d);
            a_minus_axis.push(b * exp_neg_d);
        }
        a_plus.push(a_plus_axis);
        a_minus.push(a_minus_axis);
    }

    (a_plus, a_minus)
}

pub(super) fn convert_a_factors_to_bd(
    a_plus: &[Vec<f64>],
    a_minus: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let num_axes = a_plus.len();
    let mut backbone: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
    let mut tilt: Vec<Vec<f64>> = Vec::with_capacity(num_axes);

    for axis in 0..num_axes {
        let n_bins = a_plus[axis].len();
        let mut backbone_axis = Vec::with_capacity(n_bins);
        let mut tilt_axis = Vec::with_capacity(n_bins);

        for bin in 0..n_bins {
            let a_p = a_plus[axis][bin].max(EPSILON);
            let a_m = a_minus[axis][bin].max(EPSILON);
            let b = (a_p * a_m).sqrt();
            let ratio = (a_p / a_m).max(EPSILON).min(1.0 / EPSILON);
            let d = 0.5 * ratio.ln();
            backbone_axis.push(b);
            tilt_axis.push(d);
        }

        backbone.push(backbone_axis);
        tilt.push(tilt_axis);
    }

    (backbone, tilt)
}

fn center_tilt_per_axis(tilt_values: &mut [Vec<f64>], observation_counts: &[Vec<usize>]) {
    const EPS: f64 = 1e-12;

    for dim in 0..tilt_values.len() {
        let counts = &observation_counts[dim];
        let d = &mut tilt_values[dim];

        if counts.is_empty() || d.is_empty() {
            continue;
        }

        let weights_sum: f64 = counts.iter().map(|&c| c as f64).sum();
        if weights_sum <= EPS {
            continue;
        }

        let mean = d
            .iter()
            .zip(counts.iter())
            .map(|(&x, &c)| x * (c as f64))
            .sum::<f64>()
            / weights_sum;

        for x in d.iter_mut() {
            *x -= mean;
        }
    }
}

fn canonicalize_backbone_log_space(
    backbone_values: &mut [Vec<f64>],
    observation_counts: &[Vec<usize>],
) {
    const EPS: f64 = 1e-12;

    for dim in 0..backbone_values.len() {
        let counts = &observation_counts[dim];
        let b = &mut backbone_values[dim];

        if counts.is_empty() || b.is_empty() {
            continue;
        }

        let weights_sum: f64 = counts.iter().map(|&c| c as f64).sum();
        if weights_sum <= EPS {
            continue;
        }

        let mean_log_b = b
            .iter()
            .zip(counts.iter())
            .map(|(&x, &c)| clamped_ln(x) * (c as f64))
            .sum::<f64>()
            / weights_sum;

        for x in b.iter_mut() {
            *x = (clamped_ln(*x) - mean_log_b).exp();
        }
    }
}

pub(super) fn canonicalize_bag_per_axis(grid: &mut GridTensor, observation_counts: &[Vec<usize>]) {
    canonicalize_backbone_log_space(&mut grid.backbone_values, observation_counts);
    center_tilt_per_axis(&mut grid.tilt_values, observation_counts);
}
