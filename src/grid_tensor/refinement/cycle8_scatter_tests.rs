use std::iter::once;

use ndarray::{Array1, Array2};

use super::{cycle5_refresh_tests::assert_statistics_bits, RefinementStrategy};
use crate::grid_tensor::{
    fit::initialize_state_with_input,
    params::GridTensorParamsBuilder,
    state::{BranchConsensus, FittingState},
};

// A single-feature traversal is the arithmetic oracle for independent histogram tiles.
fn one_feature_update<'a>(
    refinement: &RefinementStrategy,
    mut state: FittingState<'a>,
    col: usize,
    start: usize,
    end: usize,
) -> FittingState<'a> {
    state.require_current_statistics();
    assert!(matches!(
        refinement,
        RefinementStrategy::L2Refinement { .. }
    ));
    assert!(state
        .branch_consensus
        .as_ref()
        .is_some_and(|reference| reference.anchors.is_none()));
    assert!(state
        .precomputed_statistics
        .bin_edges
        .iter()
        .all(|edges| !edges.is_empty()));
    let updated_points = &state.precomputed_statistics.sorted_indices[col][start..end];
    if updated_points.is_empty() {
        return state;
    }

    let mut deltas = Vec::with_capacity(updated_points.len());
    for &row in updated_points {
        let point = state.point_curvature_statistics(row, refinement.weight(state.residuals[row]));
        deltas.push([
            point.s11 - state.precomputed_statistics.c_s11[row],
            point.s22 - state.precomputed_statistics.c_s22[row],
            point.s12 - state.precomputed_statistics.c_s12[row],
        ]);
        // Row statistics depend on predictions and the reference, not another row's cache.
        state.precomputed_statistics.c_s11[row] = point.s11;
        state.precomputed_statistics.c_s22[row] = point.s22;
        state.precomputed_statistics.c_s12[row] = point.s12;
    }

    let mut buckets = Vec::new();
    for feature in (0..state.p)
        .filter(|&feature| feature != col)
        .chain(once(col))
    {
        let bins = &state.precomputed_statistics.bin_indices[feature];
        let bin_count = state.precomputed_statistics.binned_prefix_sums[feature].len();
        buckets.clear();
        buckets.resize(bin_count, [0.0; 3]);
        let (mut first, mut last) = (usize::MAX, 0);
        for (&row, delta) in updated_points.iter().zip(&deltas) {
            let bin = bins[row] as usize;
            // Every touched row contributes to the range, including zero deltas.
            first = first.min(bin);
            last = last.max(bin);
            let slot = &mut buckets[bin];
            slot[0] += delta[0];
            slot[1] += delta[1];
            slot[2] += delta[2];
        }

        // Bin membership is monotone in sorted rank. These bounds equal the bins
        // of the minimum and maximum affected ranks, without a separate rank scan.
        let prefix = &mut state.precomputed_statistics.binned_prefix_sums[feature];
        let mut accumulated = [0.0; 3];
        for bin in first..=last {
            accumulated[0] += buckets[bin][0];
            accumulated[1] += buckets[bin][1];
            accumulated[2] += buckets[bin][2];
            prefix[bin].s11 += accumulated[0];
            prefix[bin].s22 += accumulated[1];
            prefix[bin].s12 += accumulated[2];
        }
        for slot in &mut prefix[(last + 1)..] {
            slot.s11 += accumulated[0];
            slot.s22 += accumulated[1];
            slot.s12 += accumulated[2];
        }
    }
    state.linear_statistics_stale = true;
    state
}

fn fixture(p: usize, constant: bool) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((97, p), |(row, col)| {
        let value = (row * [1, 29, 43, 71, 89][col] + 3 * col) % 97;
        if constant && col == p - 1 {
            1.0
        } else {
            match col {
                1 => (value % 2) as f64,
                2 => (value % 5) as f64,
                4 => (value % 11) as f64,
                _ => value as f64 / 11.0,
            }
        }
    });
    let y = Array1::from_shape_fn(97, |row| x[(row, 0)].sin() * (1.0 + x[(row, 0)]) - 1.3);
    (x, y)
}

fn initial<'a>(
    x: &'a Array2<f64>,
    y: &'a Array1<f64>,
    rho: f64,
) -> (FittingState<'a>, RefinementStrategy) {
    let params = GridTensorParamsBuilder::new().max_bins(Some(16)).build();
    let refinement = params.refinement_strategy_params.get_refinement_strategy();
    let state = initialize_state_with_input(x.view(), y.view(), &params, None);
    let reference = BranchConsensus {
        weight: rho,
        plus: Array1::from_shape_fn(state.n, |row| 1.2 + row as f64 / state.n as f64),
        minus: Array1::from_shape_fn(state.n, |row| 0.4 + 0.1 * (row as f64).cos()),
        anchors: None,
    };
    (
        refinement.refresh_consensus_reference(state, reference),
        refinement,
    )
}

fn assert_all_caches(actual: &FittingState<'_>, expected: &FittingState<'_>) {
    assert_statistics_bits(actual, expected);
    for (actual, expected) in [
        (&actual.f_plus, &expected.f_plus),
        (&actual.f_minus, &expected.f_minus),
        (&actual.f, &expected.f),
        (&actual.r_tilde, &expected.r_tilde),
        (&actual.y_hat, &expected.y_hat),
        (&actual.residuals, &expected.residuals),
    ] {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

fn compare(
    state: FittingState<'_>,
    refinement: &RefinementStrategy,
    col: usize,
    start: usize,
    end: usize,
) {
    let expected = one_feature_update(refinement, state.clone(), col, start, end);
    let actual = refinement.update_binned_curvature_statistics(state, col, start, end);
    assert_all_caches(&actual, &expected);
    assert_eq!(actual.linear_statistics_stale, start != end);
    let expected = refinement.refresh_statistics(expected);
    let actual = refinement.refresh_statistics(actual);
    assert_all_caches(&actual, &expected);
}

fn change_rows(
    state: &mut FittingState<'_>,
    col: usize,
    start: usize,
    end: usize,
    scenario: usize,
) {
    let rows = state.precomputed_statistics.sorted_indices[col][start..end].to_vec();
    for (offset, row) in rows.into_iter().enumerate() {
        let (plus, minus) = match scenario {
            0 => (state.f_plus[row], state.f_minus[row]),
            1 => (
                state.f_plus[row] * if offset % 2 == 0 { 1.01 } else { 0.99 },
                state.f_minus[row] * if offset % 3 == 0 { 0.985 } else { 1.015 },
            ),
            2 => {
                let delta = [1e16, 1.0, -1e16, -1.0][offset % 4];
                state.precomputed_statistics.c_s11[row] = -delta;
                state.precomputed_statistics.c_s22[row] = delta * 0.5;
                state.precomputed_statistics.c_s12[row] = -delta * 0.25;
                (0.0, 0.0)
            }
            _ => {
                let magnitude = [f64::from_bits(1), 1e-150, 1e150, 1e-70, 1e70][offset % 5];
                (magnitude, magnitude * 0.5)
            }
        };
        state.f_plus[row] = plus;
        state.f_minus[row] = minus;
        state.f[row] = plus - minus;
        state.y_hat[row] = state.f[row];
        state.residuals[row] = state.labels[row] - state.f[row];
        state.r_tilde[row] = state.residuals[row];
    }
}

#[test]
fn cycle8_scatter_feature_pairs_match_single_feature_oracle() {
    for p in 1..=5 {
        let (x, y) = fixture(p, false);
        let (initial, refinement) = initial(&x, &y, 0.3);
        assert!(initial
            .precomputed_statistics
            .bin_edges
            .iter()
            .all(|edges| !edges.is_empty()));
        if p > 1 {
            assert_ne!(
                initial.precomputed_statistics.binned_prefix_sums[0].len(),
                initial.precomputed_statistics.binned_prefix_sums[1].len()
            );
        }
        let mut columns = vec![0, p / 2, p - 1];
        columns.sort_unstable();
        columns.dedup();
        for col in columns {
            for (start, end) in [
                (0, initial.n),
                (initial.n / 3, 2 * initial.n / 3),
                (initial.n / 2, initial.n / 2 + 1),
                (initial.n - 1, initial.n),
            ] {
                for scenario in 0..4 {
                    let mut state = initial.clone();
                    change_rows(&mut state, col, start, end, scenario);
                    compare(state, &refinement, col, start, end);
                }
            }
        }
    }
}

#[test]
fn cycle8_scatter_zero_deltas_preserve_touched_ranges_and_signed_bits() {
    let (x, y) = fixture(5, false);
    for rho in [0.0, -0.0, 0.3] {
        let (mut state, refinement) = initial(&x, &y, rho);
        state.f_plus.fill(-0.0);
        state.f_minus.fill(0.0);
        state.precomputed_statistics.c_s11.fill(0.0);
        state.precomputed_statistics.c_s22.fill(0.0);
        state.precomputed_statistics.c_s12.fill(-0.0);
        for column in &mut state.precomputed_statistics.binned_prefix_sums {
            for slot in column {
                slot.s11 = -0.0;
                slot.s22 = -0.0;
                slot.s12 = -0.0;
            }
        }
        for col in [0, 2, 4] {
            for (start, end) in [(0, 1), (43, 44), (31, 67)] {
                compare(state.clone(), &refinement, col, start, end);
            }
        }
    }
}

#[test]
fn cycle8_scatter_empty_ranges_and_guards_match_oracle() {
    for p in 1..=5 {
        let (x, y) = fixture(p, false);
        let (state, refinement) = initial(&x, &y, 0.3);
        for rank in [0, state.n / 2, state.n] {
            compare(state.clone(), &refinement, p - 1, rank, rank);
        }
    }
    for (constant, stale) in [(true, false), (false, true)] {
        let (x, y) = fixture(3, constant);
        let (mut state, refinement) = initial(&x, &y, 0.3);
        state.linear_statistics_stale = stale;
        let expected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            one_feature_update(&refinement, state.clone(), 0, 0, 0)
        }));
        let actual = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            refinement.update_binned_curvature_statistics(state, 0, 0, 0)
        }));
        assert!(expected.is_err());
        assert!(actual.is_err());
    }
}
