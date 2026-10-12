//! Loop-control counters that track the fitting loop's progress.

/// Maximum number of consecutive resplits allowed before forcing a different action
pub const MAX_CONSECUTIVE_RESPLIT: usize = 5;

/// Loop control state for tracking algorithm progress
#[derive(Debug, Clone, Default)]
pub struct LoopState {
    /// Net number of splits performed (splits +1, resplits 0, merges -1)
    pub fineness: usize,
    /// Total number of splits performed
    pub split_count: usize,
    /// Total number of resplits performed
    pub resplit_count: usize,
    /// Total number of merges performed
    pub merge_count: usize,
    /// Number of consecutive resplits (for loop prevention)
    pub consecutive_resplits: usize,
}

impl LoopState {
    pub fn record_split(&mut self) {
        self.split_count += 1;
        self.fineness += 1;
        self.consecutive_resplits = 0;
    }

    pub fn record_resplit(&mut self) {
        self.resplit_count += 1;
        self.consecutive_resplits += 1;
    }

    pub fn record_merge(&mut self) {
        self.merge_count += 1;
        self.fineness = self.fineness.saturating_sub(1);
        self.consecutive_resplits = 0;
    }
}
