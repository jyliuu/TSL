//! Branch consensus: the reference branches each grid is pulled toward and their anchor rows.

use ndarray::Array1;
use std::sync::Arc;

/// Fixed training rows on which branch consensus is evaluated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsensusAnchors {
    pub rows: Vec<usize>,
    pub row_to_anchor: Vec<usize>,
}

impl ConsensusAnchors {
    pub fn new(n: usize, rows: Vec<usize>) -> Self {
        assert!(!rows.is_empty(), "consensus anchors must not be empty");
        assert!(
            rows.windows(2).all(|pair| pair[0] < pair[1]),
            "anchor rows must be sorted and unique"
        );
        assert!(
            rows.last().is_some_and(|&row| row < n),
            "anchor row is outside the training data"
        );
        let mut row_to_anchor = vec![usize::MAX; n];
        for (index, &row) in rows.iter().enumerate() {
            row_to_anchor[row] = index;
        }
        Self {
            rows,
            row_to_anchor,
        }
    }
}

pub(crate) fn same_consensus_support(
    left: &Option<Arc<ConsensusAnchors>>,
    right: &Option<Arc<ConsensusAnchors>>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => Arc::ptr_eq(left, right),
        _ => false,
    }
}

/// Reference branches and their per-row penalty weight.
/// Anchored references have one entry per anchor and weight `ρ n / m`.
#[derive(Debug, Clone)]
pub struct BranchConsensus {
    pub weight: f64,
    pub plus: Array1<f64>,
    pub minus: Array1<f64>,
    pub anchors: Option<Arc<ConsensusAnchors>>,
}

impl BranchConsensus {
    pub fn target(&self, row: usize) -> Option<(f64, f64, f64)> {
        let index = match &self.anchors {
            Some(anchors) => {
                let index = anchors.row_to_anchor[row];
                if index == usize::MAX {
                    return None;
                }
                index
            }
            None => row,
        };
        Some((self.weight, self.plus[index], self.minus[index]))
    }
}
