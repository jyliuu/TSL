//! Interval and structural-action candidate types, and the per-fit split strategy state.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub start: usize,
    pub length: usize,
}

impl Interval {
    pub fn new(start: usize, length: usize) -> Self {
        Self { start, length }
    }

    pub fn end(&self) -> usize {
        self.start + self.length
    }

    pub fn contains(&self, position: usize) -> bool {
        position >= self.start && position < self.end()
    }

    pub fn overlaps(&self, other: &Interval) -> bool {
        self.start < other.end() && other.start < self.end()
    }

    pub fn subtract(&self, forbidden: &Interval) -> Vec<Interval> {
        if !self.overlaps(forbidden) {
            return vec![*self];
        }

        let mut result = Vec::new();

        // Left segment (before forbidden interval)
        if self.start < forbidden.start {
            result.push(Interval::new(self.start, forbidden.start - self.start));
        }

        // Right segment (after forbidden interval)
        if self.end() > forbidden.end() {
            result.push(Interval::new(forbidden.end(), self.end() - forbidden.end()));
        }

        result
    }
}

#[derive(Debug, Clone)]
pub struct SplitCandidate {
    pub col: usize,
    pub error_reduction: f64,
    pub allowed_interval_idx: usize,
    pub index: usize,
    /// Two-tensor updates for left side: (u_plus_L, u_minus_L)
    pub update_left: (f64, f64),
    /// Two-tensor updates for right side: (u_plus_R, u_minus_R)
    pub update_right: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct MergeCandidate {
    pub col: usize,
    pub error_reduction: f64,
    pub interval_idx: usize,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct ResplitCandidate {
    pub col: usize,
    pub error_reduction: f64,
    pub interval_idx: usize,
    pub index: usize,
    /// Two-tensor updates for left side: (u_plus_L, u_minus_L)
    pub update_left: (f64, f64),
    /// Two-tensor updates for right side: (u_plus_R, u_minus_R)
    pub update_right: (f64, f64),
}

/// Split strategy state
#[derive(Debug, Clone, Default)]
pub struct SplitStrategyState {
    /// Valid split intervals per feature [col][n_intervals]
    pub allowed_intervals: Vec<Vec<Interval>>,
    /// Total positions per feature [col]
    pub total_positions: Vec<usize>,
    /// Whether merge statistics participate in unified action selection.
    pub merge_enabled: bool,
    /// Fixed loss-scale cost of adding one boundary to the grid.
    pub boundary_penalty: f64,
}
