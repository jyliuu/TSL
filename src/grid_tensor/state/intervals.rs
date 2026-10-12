//! Interval lookups on a column's grid and the ranges a structural action affects.

use super::FittingState;

/// The sorted positions of one column whose rows a structural action touched.
#[derive(Debug, Clone, Copy)]
pub struct AffectedRange {
    pub col: usize,
    /// Smallest inclusive range `(lo, hi)` of sorted positions covering every touched row.
    pub point_range: (usize, usize),
}

impl<'a> FittingState<'a> {
    /// Index of the interval on `col` that holds `row`: the number of boundaries at or
    /// below the row's rank, since an interval begins at its boundary rank.
    pub fn interval_index(&self, col: usize, row: usize) -> usize {
        let rank = self.precomputed_statistics.sort_order[col][row];
        self.boundaries[col].partition_point(|&b| b <= rank)
    }

    pub fn interval_range(&self, col: usize, interval_idx: usize) -> (usize, usize) {
        let start = if interval_idx == 0 {
            0
        } else {
            self.boundaries[col][interval_idx - 1]
        };
        let end = self.boundaries[col]
            .get(interval_idx)
            .copied()
            .unwrap_or(self.n);
        (start, end)
    }

    /// Start of interval `interval_idx`, its boundary with the next interval, and that
    /// interval's end.
    pub fn interval_range_left_and_right(
        &self,
        col: usize,
        interval_idx: usize,
    ) -> (usize, usize, usize) {
        let (start, index) = self.interval_range(col, interval_idx);
        let end = self.boundaries[col]
            .get(interval_idx + 1)
            .copied()
            .unwrap_or(self.n);
        (start, index, end)
    }

    /// The affected range of every column whose rows overlap the sorted positions
    /// `interval` of `col`, the acted-on column first.
    pub(crate) fn get_affected_ranges(
        &self,
        col: usize,
        interval: (usize, usize),
    ) -> Vec<AffectedRange> {
        let (start, end) = interval;
        let updated_points = &self.precomputed_statistics.sorted_indices[col][start..end];

        let mut affected_ranges = Vec::with_capacity(self.p);

        affected_ranges.push(AffectedRange {
            col,
            point_range: (start, end - 1),
        });

        for other_col in (0..self.p).filter(|&c| c != col) {
            let mut min_pos = usize::MAX;
            let mut max_pos = 0usize;
            for &pt_idx in updated_points.iter() {
                let pos = self.precomputed_statistics.sort_order[other_col][pt_idx];
                if pos < min_pos {
                    min_pos = pos;
                }
                if pos > max_pos {
                    max_pos = pos;
                }
            }
            if min_pos == usize::MAX {
                // Empty range - skip this column
                continue;
            }

            affected_ranges.push(AffectedRange {
                col: other_col,
                point_range: (min_pos, max_pos),
            });
        }

        affected_ranges
    }

    /// Convert datapoint position range to interval index range
    /// Given boundaries and a range of affected datapoint positions [lo_pos, hi_pos],
    /// returns the range of interval indices [lo_idx, hi_idx] whose data ranges intersect
    /// with the position range.
    /// When boundaries is empty, there is one interval (index 0) covering all points.
    pub(crate) fn compute_boundary_index_range(
        &self,
        col: usize,
        lo_pos: usize,
        hi_pos: usize,
    ) -> (usize, usize) {
        let boundaries = &self.boundaries[col];

        if lo_pos > hi_pos {
            // Empty range - return (1, 0) to indicate no intervals
            return (1, 0);
        }

        // Find the lowest interval index whose range intersects [lo_pos, hi_pos]
        // Interval k covers [boundaries[k-1], boundaries[k]).
        // We want k such that boundaries[k] > lo_pos.
        let lo = boundaries.partition_point(|&b| b <= lo_pos);

        // Find the highest interval index whose range intersects [lo_pos, hi_pos]
        // We want the interval containing hi_pos.
        let hi = boundaries.partition_point(|&b| b <= hi_pos);

        (lo, hi)
    }
}
