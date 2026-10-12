//! The tiled linear rebin matches the frozen two-pass refresh bit for bit.

use ndarray::{Array1, Array2};

use super::{
    streamed_refresh_tests::{assert_statistics_bits, two_pass_refresh},
    RefinementStrategy,
};
use crate::grid_tensor::{
    fit::initialize_state_with_input,
    params::GridTensorParamsBuilder,
    state::{BranchConsensus, FittingState},
};

fn fixture(p: usize, constant: bool) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((97, p), |(row, col)| {
        let value = (row * [1, 29, 43, 71, 89, 7, 13, 53, 61][col] + 5 * col) % 97;
        if constant && col == p - 1 {
            2.0
        } else {
            match col % 4 {
                0 => (value % 2) as f64,
                1 => (value % 5) as f64,
                2 => (value % 11) as f64,
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
    bins: Option<u16>,
    rho: f64,
) -> (FittingState<'a>, RefinementStrategy) {
    let params = GridTensorParamsBuilder::new().max_bins(bins).build();
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

fn perturb(state: &mut FittingState<'_>) {
    for row in 0..state.n {
        let (plus, minus) = match row % 5 {
            0 => (state.f_plus[row] * 1.01, state.f_minus[row] * 0.985),
            1 => (state.f_plus[row], state.f_minus[row]),
            2 => ([f64::from_bits(1), 1e-150, 1e150][row % 3], 0.5e-70),
            3 => (-0.0, 0.0),
            _ => (state.f_plus[row] * 0.99, state.f_minus[row] * 1.015),
        };
        state.f_plus[row] = plus;
        state.f_minus[row] = minus;
        state.f[row] = plus - minus;
        state.y_hat[row] = state.f[row];
        state.residuals[row] = state.labels[row] - state.f[row];
        state.r_tilde[row] = state.residuals[row];
    }
}

fn assert_refresh_matches_oracle(state: FittingState<'_>, refinement: &RefinementStrategy) {
    let expected = two_pass_refresh(refinement, state.clone());
    let actual = refinement.refresh_statistics(state);
    assert_statistics_bits(&actual, &expected);
    assert_eq!(
        format!("{:?}", actual.precomputed_statistics),
        format!("{:?}", expected.precomputed_statistics)
    );
}

#[test]
fn tiled_linear_rebin_matches_the_two_pass_oracle() {
    for p in [1usize, 2, 3, 4, 5, 8, 9] {
        for constant in [false, true] {
            if constant && p < 2 {
                continue;
            }
            let (x, y) = fixture(p, constant);
            for bins in [None, Some(3), Some(16), Some(64)] {
                for rho in [0.3, 0.0] {
                    let (mut state, refinement) = initial(&x, &y, bins, rho);
                    perturb(&mut state);
                    assert_refresh_matches_oracle(state, &refinement);
                }
            }
        }
    }
}

#[test]
fn guard_shapes_route_through_the_same_arithmetic() {
    let (x, y) = fixture(5, false);
    let (mut state, refinement) = initial(&x, &y, Some(16), 0.3);
    perturb(&mut state);
    let n = state.n;
    let mut longer = state.clone();
    longer.precomputed_statistics.c_t1.extend([7.0, 8.0]);
    longer.precomputed_statistics.c_t2.extend([9.0, 10.0]);
    assert_refresh_matches_oracle(longer, &refinement);
    let mut extended_prefix = state.clone();
    extended_prefix.precomputed_statistics.binned_prefix_sums[1].push(Default::default());
    assert_refresh_matches_oracle(extended_prefix, &refinement);
    let mut shorter_map = state;
    let map: Vec<u16> = shorter_map.precomputed_statistics.bin_indices[2][..n - 1].to_vec();
    shorter_map.precomputed_statistics.bin_indices[2] = map.into();
    assert_refresh_matches_oracle(shorter_map, &refinement);
}
