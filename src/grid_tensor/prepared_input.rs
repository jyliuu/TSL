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
