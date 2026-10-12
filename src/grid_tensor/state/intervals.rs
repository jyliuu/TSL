//! Interval lookups on a column's grid and the ranges a structural action affects.

use super::FittingState;

/// Represents affected ranges for a column after a transformation
/// Contains both the tight point range (for statistics updates) and
/// the interval range (for cache refreshing)
#[derive(Debug, Clone, Copy)]
pub struct AffectedRange {
    /// Column index
    pub col: usize,
    /// Tight point position range (lo, hi) for statistics updates
    /// This is the minimal range covering all affected data points
    pub point_range: (usize, usize),
    /// Interval index range (blo, bhi) for cache refreshing and filtering
    /// This is the range of interval indices whose data ranges intersect with point_range
    pub interval_range: (usize, usize),
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
            .unwrap_or(self.x.nrows());
        (start, end)
    }

    pub fn interval_range_left_and_right(
        &self,
        col: usize,
        interval_idx: usize,
    ) -> (usize, usize, usize) {
        let start = if interval_idx == 0 {
            0
        } else {
            self.boundaries[col][interval_idx - 1]
        };

        let index = if interval_idx < self.boundaries[col].len() {
            self.boundaries[col][interval_idx]
        } else {
            self.x.nrows()
        };

        let end = if interval_idx + 1 < self.boundaries[col].len() {
            self.boundaries[col][interval_idx + 1]
        } else {
            self.x.nrows()
        };
        (start, index, end)
    }

    /// Get affected ranges for all columns after a transformation
    /// Returns both tight point ranges (for statistics) and interval ranges (for filtering)
    pub(crate) fn get_affected_ranges(
        &self,
        col: usize,
        interval: (usize, usize),
    ) -> Vec<AffectedRange> {
        let (start, end) = interval;
        let updated_points = &self.precomputed_statistics.sorted_indices[col][start..end];

        let mut affected_ranges = Vec::with_capacity(self.x.ncols());

        // For the split column, compute which allowed_intervals intersect with the affected interval
        let point_range = (start, end - 1);
        // Compute the allowed_interval index range that intersects with the affected point range
        let interval_range = self.compute_allowed_interval_index_range(col, start, end - 1);
        affected_ranges.push(AffectedRange {
            col,
            point_range,
            interval_range,
        });

        // For other columns, compute tight point ranges and map to interval ranges
        for other_col in (0..self.x.ncols()).filter(|&c| c != col) {
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

            let point_range = (min_pos, max_pos);

            // Expand the point range to include the full data intervals that were touched
            // This ensures that all allowed intervals within the recomputed data intervals are filtered
            let (d_lo, d_hi) = self.compute_boundary_index_range(other_col, min_pos, max_pos);
            let start_expanded = self.interval_range(other_col, d_lo).0;
            let end_expanded = self.interval_range(other_col, d_hi).1;

            // Compute the allowed_interval index range that intersects with the EXPANDED range
            let interval_range = self.compute_allowed_interval_index_range(
                other_col,
                start_expanded,
                end_expanded.saturating_sub(1),
            );

            affected_ranges.push(AffectedRange {
                col: other_col,
                point_range,
                interval_range,
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

        if boundaries.is_empty() {
            // No boundaries means one interval (index 0) covering all points
            // If the position range is non-empty, it intersects with interval 0
            if lo_pos <= hi_pos {
                return (0, 0);
            } else {
                // Empty range - return (1, 0) to indicate no intervals
                return (1, 0);
            }
        }

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

    /// Convert datapoint position range to allowed_interval index range
    /// Given allowed_intervals and a range of affected datapoint positions [lo_pos, hi_pos],
    /// returns the range of allowed_interval indices [lo_idx, hi_idx] that intersect
    /// with the position range.
    /// Returns (1, 0) to indicate no intervals if the range is empty or no intervals intersect.
    ///
    /// An interval intersects [lo_pos, hi_pos] if: interval.start < hi_pos + 1 && interval.end() > lo_pos
    /// Since allowed_intervals are sorted by start, intersecting intervals are consecutive.
    pub(crate) fn compute_allowed_interval_index_range(
        &self,
        col: usize,
        lo_pos: usize,
        hi_pos: usize,
    ) -> (usize, usize) {
        let allowed_intervals = &self.split_strategy_state.allowed_intervals[col];

        if allowed_intervals.is_empty() || lo_pos > hi_pos {
            return (1, 0);
        }

        // Find the first interval that could intersect: first where interval.end() > lo_pos
        // This is the first interval that ends after lo_pos
        let lo_idx = allowed_intervals.partition_point(|interval| interval.end() <= lo_pos);

        // Find the last interval that could intersect: last where interval.start <= hi_pos
        // This is the rightmost interval that starts at or before hi_pos
        let hi_candidate = allowed_intervals
            .partition_point(|interval| interval.start <= hi_pos)
            .saturating_sub(1);

        // Check if any intervals actually intersect
        if lo_idx > hi_candidate {
            return (1, 0);
        }

        // Verify the last candidate actually intersects (it might start before but end before lo_pos)
        let hi_idx = if allowed_intervals[hi_candidate].end() > lo_pos {
            hi_candidate
        } else {
            // Last candidate doesn't intersect, find the actual last intersecting one
            // We need to search backwards from hi_candidate
            let mut idx = hi_candidate;
            while idx >= lo_idx {
                if allowed_intervals[idx].end() > lo_pos {
                    break;
                }
                if idx == 0 {
                    return (1, 0);
                }
                idx -= 1;
            }
            idx
        };

        (lo_idx, hi_idx)
    }
}
