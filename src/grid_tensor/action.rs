use crate::grid_tensor::splitting::{MergeCandidate, ResplitCandidate, SplitCandidate};

/// A structural action on the grid.
///
/// Split strategies propose an action and the reducer applies it together with every
/// cache and statistics update it implies.
#[derive(Debug, Clone)]
pub enum FittingAction {
    /// Apply a split with all associated updates
    ApplySplit { split: SplitCandidate },
    /// Apply a merge with all associated updates
    ApplyMerge { merge: MergeCandidate },
    /// Apply a resplit with all associated updates
    ApplyResplit { resplit: ResplitCandidate },
}
