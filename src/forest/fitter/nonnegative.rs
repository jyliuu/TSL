use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, ArrayView1, ArrayView2};

/// Nonnegative least squares with a feasible incumbent and normalized columns.
///
/// Passive-set least squares uses SVD on the design itself. Every accepted
/// incumbent is checked in the original coordinates, including finite products
/// and predictions. The iteration budget returns the best feasible incumbent.
pub(super) fn solve(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior: ArrayView1<f64>,
) -> Option<Array1<f64>> {
    let budget = x.ncols().saturating_mul(x.ncols()).saturating_mul(10);
    solve_with_limit(x, y, prior, budget.clamp(32, 10_000))
}

fn solve_with_limit(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior: ArrayView1<f64>,
    budget: usize,
) -> Option<Array1<f64>> {
    let (n, k) = x.dim();
    if y.len() != n || prior.len() != k || x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
        return None;
    }
    let target_scale = y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    if target_scale == 0.0 || k == 0 {
        return Some(Array1::zeros(k));
    }
    let b = DVector::from_iterator(n, y.iter().map(|v| v / target_scale));
    let mut a = DMatrix::zeros(n, k);
    let mut scales = vec![(0.0, 1.0); k];
    for j in 0..k {
        let largest = x.column(j).iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if largest > 0.0 {
            let norm = x
                .column(j)
                .iter()
                .map(|v| (v / largest).powi(2))
                .sum::<f64>()
                .sqrt();
            scales[j] = (largest, norm);
            for i in 0..n {
                a[(i, j)] = (x[(i, j)] / largest) / norm;
            }
        }
    }

    let mut best = Array1::zeros(k);
    let mut best_loss = b.norm_squared();
    if let Some(loss) = original_loss(x, y, prior, target_scale) {
        if loss <= best_loss {
            best.assign(&prior);
            best_loss = loss;
        }
    }
    let mut z = DVector::zeros(k);
    for j in 0..k {
        if scales[j].0 > 0.0 {
            z[j] = positive_ratio(best[j], scales[j].0, target_scale) * scales[j].1;
        }
    }
    if z.iter().any(|v| !v.is_finite()) {
        z.fill(0.0);
    }
    let mut passive: Vec<bool> = z.iter().map(|v| *v > 0.0).collect();
    let mut needs_solve = passive.iter().any(|v| *v);
    let tolerance = 64.0 * f64::EPSILON * (n.max(k) as f64).sqrt() * b.norm().max(1.0);

    for _ in 0..budget {
        if !needs_solve {
            let residual = &b - &a * &z;
            let mut entering = None;
            let mut largest_gradient = tolerance;
            for j in 0..k {
                if !passive[j] && scales[j].0 > 0.0 {
                    let gradient = a.column(j).dot(&residual);
                    if gradient > largest_gradient {
                        entering = Some(j);
                        largest_gradient = gradient;
                    }
                }
            }
            let Some(j) = entering else { break };
            passive[j] = true;
        }

        let Some(proposal) = solve_passive(&a, &b, &passive) else {
            break;
        };
        let mut alpha = 1.0_f64;
        let mut leaving = None;
        for j in 0..k {
            if passive[j] && proposal[j] <= 0.0 {
                let fraction = if z[j] == 0.0 {
                    0.0
                } else {
                    z[j] / (z[j] - proposal[j])
                };
                if fraction <= alpha {
                    alpha = fraction;
                    leaving = Some(j);
                }
            }
        }
        if let Some(j) = leaving {
            for l in 0..k {
                z[l] = (z[l] + alpha * (proposal[l] - z[l])).max(0.0);
            }
            z[j] = 0.0;
            for l in 0..k {
                if z[l] == 0.0 {
                    passive[l] = false;
                }
            }
            needs_solve = passive.iter().any(|v| *v);
        } else {
            z = proposal;
            needs_solve = false;
        }

        let candidate = Array1::from_shape_fn(k, |j| {
            if scales[j].0 == 0.0 {
                0.0
            } else {
                positive_ratio(z[j] / scales[j].1, target_scale, scales[j].0)
            }
        });
        if let Some(loss) = original_loss(x, y, candidate.view(), target_scale) {
            if loss < best_loss {
                best = candidate;
                best_loss = loss;
            }
        }
    }
    Some(best)
}

fn solve_passive(a: &DMatrix<f64>, b: &DVector<f64>, passive: &[bool]) -> Option<DVector<f64>> {
    let indices: Vec<usize> = passive
        .iter()
        .enumerate()
        .filter_map(|(j, v)| v.then_some(j))
        .collect();
    let block = DMatrix::from_fn(a.nrows(), indices.len(), |i, j| a[(i, indices[j])]);
    let svd = block.svd(true, true);
    let largest = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let cutoff = largest * a.nrows().max(indices.len()) as f64 * f64::EPSILON;
    let solution = svd.solve(b, cutoff).ok()?;
    if solution.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut proposal = DVector::zeros(a.ncols());
    for (j, coefficient) in indices.into_iter().zip(solution.iter()) {
        proposal[j] = *coefficient;
    }
    Some(proposal)
}

fn positive_ratio(value: f64, numerator: f64, denominator: f64) -> f64 {
    if value == 0.0 {
        return 0.0;
    }
    let direct = value * (numerator / denominator);
    if direct.is_finite() && direct > 0.0 {
        direct
    } else {
        (value.ln() + numerator.ln() - denominator.ln()).exp()
    }
}

fn original_loss(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    target_scale: f64,
) -> Option<f64> {
    if weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return None;
    }
    let mut loss = 0.0;
    for (i, row) in x.rows().into_iter().enumerate() {
        let prediction = row
            .iter()
            .zip(weights.iter())
            .map(|(a, w)| a * w)
            .sum::<f64>();
        if !prediction.is_finite() {
            return None;
        }
        loss += (y[i] / target_scale - prediction / target_scale).powi(2);
    }
    loss.is_finite().then_some(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s, Array2};

    #[test]
    fn nonnegative_refits_the_free_coefficients() {
        let x = array![[1.0, 1.0], [0.0, 1.0]];
        let y = array![0.0, 1.0];
        let w = solve(x.view(), y.view(), array![0.0, 0.0].view()).unwrap();
        assert_eq!(w[0], 0.0);
        assert!((w[1] - 0.5).abs() < 1e-14);
        assert!((x.dot(&w) - &y).mapv(|v| v * v).sum() < 0.51);
    }

    #[test]
    fn nonnegative_handles_signed_duplicate_zero_and_rescaled_columns() {
        let x = array![
            [1e100, -1e-100, 2.0, 0.0],
            [2e100, -2e-100, 4.0, 0.0],
            [3e100, -3e-100, 6.0, 0.0]
        ];
        for sign in [-1.0, 1.0] {
            let y = array![sign, 2.0 * sign, 3.0 * sign];
            let w = solve(x.view(), y.view(), Array1::zeros(4).view()).unwrap();
            assert!(w.iter().all(|v| v.is_finite() && *v >= 0.0));
            assert_eq!(w[3], 0.0);
            assert!((x.dot(&w) - y).iter().all(|v| v.abs() < 1e-12));
        }
    }

    #[test]
    fn nonnegative_matches_exhaustive_passive_sets() {
        for seed in 0..12 {
            let storage = Array2::from_shape_fn((18, 10), |(i, j)| {
                (((i + 3) * (j + 7) + seed * (i + j * 3)) as f64).sin()
            });
            let x = storage.slice(s![..;2, ..;2]);
            let y = Array1::from_shape_fn(9, |i| (i as f64 * 0.7 + seed as f64).cos());
            let a = DMatrix::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)]);
            let b = DVector::from_iterator(y.len(), y.iter().copied());
            let mut expected = y.mapv(|v| v * v).sum();
            for mask in 1..(1 << x.ncols()) {
                let passive: Vec<bool> = (0..x.ncols()).map(|j| mask & (1 << j) != 0).collect();
                let candidate = solve_passive(&a, &b, &passive).unwrap();
                if candidate.iter().all(|v| *v >= 0.0) {
                    expected = expected.min((&a * candidate - &b).norm_squared());
                }
            }
            let actual = solve(x, y.view(), Array1::zeros(x.ncols()).view()).unwrap();
            let residual = &y - &x.dot(&actual);
            let loss = residual.mapv(|v| v * v).sum();
            assert!(
                (loss - expected).abs() < 1e-11,
                "seed {seed}: {loss} != {expected}"
            );
            let gradient = x.t().dot(&residual);
            for j in 0..x.ncols() {
                assert!(actual[j] >= 0.0);
                assert!(gradient[j] < 1e-10);
                assert!(actual[j] * gradient[j].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn nonnegative_retains_the_best_feasible_incumbent_at_its_budget() {
        let x = array![[1.0, -0.5], [0.5, -1.0], [2.0, -1.0]];
        let y = array![1.0, 0.2, 1.7];
        let prior = array![0.8, 0.1];
        let zero = Array1::<f64>::zeros(2);
        let baseline = original_loss(x.view(), y.view(), prior.view(), 1.7).unwrap();
        assert_eq!(
            solve_with_limit(x.view(), y.view(), prior.view(), 0).unwrap(),
            prior
        );
        for budget in [1, 2, 20] {
            let w = solve_with_limit(x.view(), y.view(), prior.view(), budget).unwrap();
            assert!(original_loss(x.view(), y.view(), w.view(), 1.7).unwrap() <= baseline);
        }
        let bad_prior = array![1e300, 1e300];
        assert_eq!(
            solve_with_limit(x.view(), y.view(), bad_prior.view(), 0).unwrap(),
            zero
        );
    }

    #[test]
    fn nonnegative_handles_zero_targets_and_rejects_nonfinite_data() {
        let x = Array2::zeros((3, 4));
        let prior = Array1::zeros(4);
        assert_eq!(
            solve(x.view(), array![1.0, -1.0, 2.0].view(), prior.view()).unwrap(),
            prior
        );
        assert_eq!(
            solve(x.view(), Array1::zeros(3).view(), prior.view()).unwrap(),
            prior
        );
        assert!(solve(x.view(), array![f64::NAN, 0.0, 0.0].view(), prior.view()).is_none());
        let invalid_x = array![[f64::INFINITY]];
        assert!(solve(invalid_x.view(), array![1.0].view(), array![0.0].view()).is_none());
    }
}
