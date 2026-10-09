use ndarray::ArrayView2;

use super::{histogram_binning::quantile_bin_edge_positions, state::BinEdges};

/// Input-dependent ordering and candidate boundaries shared across boosting stages.
pub(crate) struct PreparedInput<'a> {
    x: ArrayView2<'a, f64>,
    max_bins: Option<u16>,
    pub(crate) sorted_indices: Vec<Vec<usize>>,
    pub(crate) sort_order: Vec<Vec<usize>>,
    pub(crate) bin_edges: BinEdges,
    pub(crate) bin_indices: Vec<std::sync::Arc<[u16]>>,
    pub(crate) bin_ranks: Vec<std::sync::Arc<[u16]>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_tensor::{
        fit::{fit_initialized, initialize_state_with_input},
        histogram_binning::apply_histogram_binning_mask,
        params::{GridTensorParamsBuilder, RefinementStrategyParamsBuilder},
        state::FittingState,
    };
    use ndarray::{s, Array1, Array2};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn prepared_and_masked_initialization_preserves_full_candidate_fits() {
        let storage = Array2::from_shape_fn((254, 10), |(row, col)| {
            if col == 8 {
                1.
            } else {
                ((row * (col + 3)) % 23) as f64 / 7.
            }
        });
        let x = storage.slice(s![..;2,..;2]);
        for positive in [false, true] {
            let y = Array1::from_shape_fn(x.nrows(), |i| {
                x[[i, 0]].sin() - x[[i, 1]] + if positive { 5. } else { 0. }
            });
            for huber in [false, true] {
                for bins in [None, Some(1), Some(8), Some(256)] {
                    let refinement = RefinementStrategyParamsBuilder::new()
                        .alpha(0.01)
                        .tilt_tau(0.01);
                    let refinement = if huber {
                        refinement.huber()
                    } else {
                        refinement.l2()
                    }
                    .build();
                    let params = GridTensorParamsBuilder::new()
                        .n_iter(20)
                        .max_bins(bins)
                        .refinement_strategy(refinement)
                        .build();
                    let strategy = params.refinement_strategy_params.get_refinement_strategy();
                    let split = params.split_strategy_params.get_split_strategy();
                    let state = strategy.initialize(FittingState::new(x.view(), y.view()));
                    let mut state = apply_histogram_binning_mask(state, bins);
                    state.current_error = strategy.weighted_squared_error(&state.residuals);
                    let reference = split.initialize(state);
                    let prepared = PreparedInput::new(x.view(), bins);
                    for input in [None, Some(&prepared)] {
                        let actual =
                            initialize_state_with_input(x.view(), y.view(), &params, input);
                        for (a, b) in actual
                            .precomputed_statistics
                            .error_reductions_split
                            .iter()
                            .flatten()
                            .zip(
                                reference
                                    .precomputed_statistics
                                    .error_reductions_split
                                    .iter()
                                    .flatten(),
                            )
                        {
                            assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan());
                        }
                        let mut r1 = StdRng::seed_from_u64(79);
                        let mut r2 = r1.clone();
                        let a = fit_initialized(actual, &params, &mut r1);
                        let b = fit_initialized(reference.clone(), &params, &mut r2);
                        assert_eq!(
                            bincode::serialize(&a.1).unwrap(),
                            bincode::serialize(&b.1).unwrap()
                        );
                        assert_eq!(a.0.y_hat, b.0.y_hat);
                        assert_eq!(a.0.err.to_bits(), b.0.err.to_bits());
                        assert_eq!(r1.gen::<u64>(), r2.gen::<u64>());
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "same feature view")]
    fn prepared_input_checks_strides() {
        let x = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j) as f64);
        let prepared = PreparedInput::new(x.view(), Some(2));
        prepared.assert_matches(x.t(), Some(2));
    }
}

impl<'a> PreparedInput<'a> {
    pub(crate) fn new(x: ArrayView2<'a, f64>, max_bins: Option<u16>) -> Self {
        let mut sorted_indices = Vec::with_capacity(x.ncols());
        let mut sort_order = Vec::with_capacity(x.ncols());
        let mut bin_edges = Vec::with_capacity(x.ncols());
        let mut bin_indices = Vec::with_capacity(x.ncols());
        for col in 0..x.ncols() {
            let mut indices: Vec<_> = (0..x.nrows()).collect();
            indices.sort_by(|&a, &b| x[[a, col]].partial_cmp(&x[[b, col]]).unwrap());
            let mut ranks = vec![0; x.nrows()];
            for (rank, &row) in indices.iter().enumerate() {
                ranks[row] = rank;
            }
            let edges = if let Some(bins) = max_bins.filter(|&bins| bins >= 2) {
                let values: Vec<_> = x.column(col).iter().copied().collect();
                quantile_bin_edge_positions(&indices, &values, bins as usize)
            } else {
                vec![]
            };
            bin_indices.push(super::histogram_binning::row_bin_indices(&indices, &edges));
            sorted_indices.push(indices);
            sort_order.push(ranks);
            bin_edges.push(edges);
        }
        let bin_ranks = super::histogram_binning::rank_ordered_bin_streams(
            &sorted_indices,
            &bin_indices,
            x.nrows(),
        );
        Self {
            x,
            max_bins,
            sorted_indices,
            sort_order,
            bin_edges,
            bin_indices,
            bin_ranks,
        }
    }

    pub(crate) fn assert_matches(&self, x: ArrayView2<f64>, max_bins: Option<u16>) {
        assert!(
            self.x.as_ptr() == x.as_ptr()
                && self.x.shape() == x.shape()
                && self.x.strides() == x.strides()
                && self.max_bins == max_bins,
            "Prepared input must belong to the same feature view and bin configuration"
        );
    }
}
