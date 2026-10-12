//! Loop-control counters that track the fitting loop's progress.

/// Maximum number of consecutive resplits allowed before forcing a different action
pub const MAX_CONSECUTIVE_RESPLIT: usize = 5;

/// Loop control state for tracking algorithm progress
#[derive(Debug, Clone)]
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
    /// Current iteration number
    pub iteration: usize,
}

impl Default for LoopState {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopState {
    pub fn new() -> Self {
        Self {
            fineness: 0,
            split_count: 0,
            resplit_count: 0,
            merge_count: 0,
            consecutive_resplits: 0,
            iteration: 0,
        }
    }
}
