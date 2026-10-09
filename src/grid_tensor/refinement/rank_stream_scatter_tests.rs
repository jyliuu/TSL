//! Rank-ordered bin streams scatter the same histograms as the row-indexed fallback.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use super::{cycle5_refresh_tests::assert_statistics_bits, RefinementStrategy};
use crate::grid_tensor::{
    fit::initialize_state_with_input,
    histogram_binning::rank_ordered_bin_streams_within,
    params::GridTensorParamsBuilder,
    prepared_input::PreparedInput,
    state::{BranchConsensus, FittingState},
};

fn fixture(p: usize) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((97, p), |(row, col)| {
        let value = (row * [1, 29, 43, 71, 89][col] + 3 * col) % 97;
        match col {
            1 => (value % 2) as f64,
            2 => (value % 5) as f64,
            4 => (value % 11) as f64,
            _ => value as f64 / 11.0,
        }
    });
    let y = Array1::from_shape_fn(97, |row| x[(row, 0)].sin() * (1.0 + x[(row, 0)]) - 1.3);
    (x, y)
}

fn initial<'a>(
    x: &'a Array2<f64>,
    y: &'a Array1<f64>,
    prepared: Option<&PreparedInput<'_>>,
) -> (FittingState<'a>, RefinementStrategy) {
    let params = GridTensorParamsBuilder::new().max_bins(Some(16)).build();
    let refinement = params.refinement_strategy_params.get_refinement_strategy();
    let state = initialize_state_with_input(x.view(), y.view(), &params, prepared);
    let reference = BranchConsensus {
        weight: 0.3,
        plus: Array1::from_shape_fn(state.n, |row| 1.2 + row as f64 / state.n as f64),
        minus: Array1::from_shape_fn(state.n, |row| 0.4 + 0.1 * (row as f64).cos()),
        anchors: None,
    };
    (
        refinement.refresh_consensus_reference(state, reference),
        refinement,
    )
}

fn perturb(state: &mut FittingState<'_>, col: usize, start: usize, end: usize) {
    let rows = state.precomputed_statistics.sorted_indices[col][start..end].to_vec();
    for (offset, row) in rows.into_iter().enumerate() {
        let (plus, minus) = match offset % 4 {
            0 => (state.f_plus[row] * 1.01, state.f_minus[row] * 0.985),
            1 => (state.f_plus[row], state.f_minus[row]),
            2 => ([f64::from_bits(1), 1e-150, 1e150][offset % 3], 0.5e-70),
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

#[test]
fn rank_streams_match_the_row_indexed_fallback() {
    for p in 1..=5 {
        let (x, y) = fixture(p);
        let prepared = PreparedInput::new(x.view(), Some(16));
        for input in [None, Some(&prepared)] {
            let (state, refinement) = initial(&x, &y, input);
            assert_eq!(state.precomputed_statistics.bin_ranks.len(), p);
            for (col, start, end) in [
                (0, 0, 97),
                (p - 1, 10, 60),
                (0, 5, 5),
                (0, 96, 97),
                (p / 2, 0, 1),
                (p - 1, 40, 97),
            ] {
                let mut streamed = state.clone();
                perturb(&mut streamed, col, start, end);
                let mut fallback = streamed.clone();
                fallback.precomputed_statistics.bin_ranks = Vec::new();
                let streamed =
                    refinement.update_binned_curvature_statistics(streamed, col, start, end);
                let mut fallback =
                    refinement.update_binned_curvature_statistics(fallback, col, start, end);
                assert!(streamed.precomputed_statistics.curvature_deltas.is_empty());
                assert!(fallback.precomputed_statistics.curvature_deltas.is_empty());
                fallback.precomputed_statistics.bin_ranks =
                    streamed.precomputed_statistics.bin_ranks.clone();
                assert_statistics_bits(&streamed, &fallback);
                assert_eq!(
                    streamed.linear_statistics_stale,
                    fallback.linear_statistics_stale
                );
                let streamed = refinement.refresh_statistics(streamed);
                let fallback = refinement.refresh_statistics(fallback);
                assert_statistics_bits(&streamed, &fallback);
                assert_eq!(format!("{streamed:?}"), format!("{fallback:?}"));
            }
        }
    }
}

#[test]
fn rank_streams_follow_each_columns_sorted_order_within_the_budget() {
    let sorted = vec![vec![2, 0, 1], vec![1, 2, 0]];
    let bins: Vec<Arc<[u16]>> = vec![Arc::from([0u16, 1, 2]), Arc::from([5u16, 6, 7])];
    let streams = rank_ordered_bin_streams_within(&sorted, &bins, 3, 12);
    assert_eq!(streams.len(), 2);
    assert_eq!(&*streams[0], &[2, 0, 1, 7, 5, 6]);
    assert_eq!(&*streams[1], &[1, 2, 0, 6, 7, 5]);
    assert!(rank_ordered_bin_streams_within(&sorted, &bins, 3, 11).is_empty());
    assert!(rank_ordered_bin_streams_within(&sorted, &bins, 2, 12).is_empty());
    assert!(rank_ordered_bin_streams_within(&sorted[..1], &bins, 3, 12).is_empty());
    let unbinned: Vec<Arc<[u16]>> = vec![bins[0].clone(), Arc::from([])];
    assert!(rank_ordered_bin_streams_within(&sorted, &unbinned, 3, 12).is_empty());
    assert!(rank_ordered_bin_streams_within(&[], &[], 3, 12).is_empty());
}
