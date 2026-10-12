//! Allowed split intervals and split-position sampling within them.

use std::cmp::Ordering;

use rand::{seq::index::sample, Rng};

use super::Interval;

pub(super) fn sample_best_split(
    allowed_intervals: &[Vec<Interval>],
    error_reductions: &[Vec<f64>],
) -> Option<(usize, usize, usize, f64)> {
    let positions = allowed_intervals
        .iter()
        .enumerate()
        .flat_map(|(col, intervals)| {
            intervals
                .iter()
                .enumerate()
                .flat_map(move |(interval_idx, interval)| {
                    (interval.start..interval.end()).map(move |index| (col, index, interval_idx))
                })
        });
    best_scored_position(positions, |col, index| Some(error_reductions[col][index]))
}

pub(super) fn sample_random_split_positions<R: Rng + ?Sized>(
    allowed_intervals: &[Vec<Interval>],
    total_positions: &[usize],
    colsample_bytree: f64,
    split_try: usize,
    rng: &mut R,
    ncols: usize,
) -> Vec<(usize, usize, usize)> {
    let cols_can_be_sampled: Vec<usize> = total_positions
        .iter()
        .enumerate()
        .filter_map(|(idx, &positions)| if positions == 0 { None } else { Some(idx) })
        .collect();

    let ncols_to_sample = (colsample_bytree * ncols as f64) as usize;

    let cols = if cols_can_be_sampled.len() < ncols_to_sample {
        sample(rng, ncols, ncols_to_sample).into_vec()
    } else {
        cols_can_be_sampled
    };

    let mut positions = Vec::new();
    for col in cols {
        // Use cached total positions for this feature
        let total_positions = total_positions[col];

        let abstract_indices = if total_positions < split_try {
            (0..total_positions).collect()
        } else {
            sample(rng, total_positions, split_try).into_vec()
        };

        // Map abstract indices to actual split positions
        for abstract_idx in abstract_indices {
            let (split_position, interval_idx) =
                map_abstract_to_position(allowed_intervals, col, abstract_idx);

            positions.push((col, split_position, interval_idx));
        }
    }
    positions
}

pub(super) fn sample_random_split<R: Rng + ?Sized>(
    allowed_intervals: &[Vec<Interval>],
    total_positions: &[usize],
    colsample_bytree: f64,
    split_try: usize,
    rng: &mut R,
    error_reductions: &[Vec<f64>],
) -> Option<(usize, usize, usize, f64)> {
    let positions = sample_random_split_positions(
        allowed_intervals,
        total_positions,
        colsample_bytree,
        split_try,
        rng,
        error_reductions.len(),
    );
    best_scored_position(positions, |col, index| {
        error_reductions[col].get(index).copied()
    })
}

/// The first position with the strictly highest non-NaN score, as
/// `(col, index, interval_idx, score)`; positions without a score are skipped.
pub(super) fn best_scored_position(
    positions: impl IntoIterator<Item = (usize, usize, usize)>,
    mut score: impl FnMut(usize, usize) -> Option<f64>,
) -> Option<(usize, usize, usize, f64)> {
    let mut best: Option<(usize, usize, usize, f64)> = None;
    positions
        .into_iter()
        .for_each(|(col, index, interval_idx)| {
            if let Some(gain) = score(col, index) {
                if !gain.is_nan() && best.as_ref().is_none_or(|candidate| gain > candidate.3) {
                    best = Some((col, index, interval_idx, gain));
                }
            }
        });
    best
}

pub(super) fn sample_top_k_split<R: Rng + ?Sized>(
    allowed_intervals: &[Vec<Interval>],
    top_k: usize,
    must_fill_all_k: bool,
    rng: &mut R,
    error_reductions: &[Vec<f64>],
) -> Option<(usize, usize, usize, f64)> {
    let mut candidates = Vec::new();

    for (col, intervals) in allowed_intervals.iter().enumerate() {
        for (interval_idx, interval) in intervals.iter().enumerate() {
            for index in interval.start..interval.end() {
                let err_reduction = error_reductions[col][index];
                if !err_reduction.is_nan() {
                    candidates.push((col, index, interval_idx, err_reduction));
                }
            }
        }
    }

    if candidates.is_empty() || (must_fill_all_k && candidates.len() < top_k) {
        return None;
    }

    // Sort by error reduction (descending) and take top k
    candidates.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(Ordering::Equal));
    candidates.truncate(top_k);

    // Randomly select one from the top k
    let random_index = rng.gen_range(0..candidates.len());
    Some(candidates[random_index])
}

fn map_abstract_to_position(
    allowed_intervals: &[Vec<Interval>],
    col: usize,
    abstract_idx: usize,
) -> (usize, usize) {
    let mut remaining = abstract_idx;
    for (interval_idx, interval) in allowed_intervals[col].iter().enumerate() {
        if remaining < interval.length {
            return (interval.start + remaining, interval_idx);
        }
        remaining -= interval.length;
    }
    // This should never happen if abstract_idx is valid
    panic!("Invalid abstract index: {}", abstract_idx);
}

pub(super) fn initialize_intervals(
    error_reductions: &[Vec<f64>],
    min_interval_samples: usize,
) -> (Vec<Vec<Interval>>, Vec<usize>) {
    let mut allowed_intervals = vec![Vec::new(); error_reductions.len()];
    let mut total_positions = vec![0; error_reductions.len()];

    for (col, col_error_reductions) in error_reductions.iter().enumerate() {
        let mut intervals = Vec::new();
        let mut current_start = None;

        for (pos, &err_reduction) in col_error_reductions.iter().enumerate() {
            if !err_reduction.is_nan() {
                if current_start.is_none() {
                    current_start = Some(pos);
                }
            } else if let Some(start) = current_start {
                intervals.push(Interval::new(start, pos - start));
                current_start = None;
            }
        }

        if let Some(start) = current_start {
            intervals.push(Interval::new(start, col_error_reductions.len() - start));
        }

        intervals.retain(|interval| interval.length > 0);

        let n = col_error_reductions.len();
        let margin = min_interval_samples;
        let global_start = margin;
        let global_end = n.saturating_sub(margin - 1);
        let new_intervals: Vec<Interval> = intervals
            .into_iter()
            .filter_map(|interval| {
                let old_end = interval.end();
                let start = interval.start.max(global_start);
                let end = old_end.min(global_end);
                if end > start {
                    Some(Interval::new(start, end - start))
                } else {
                    None
                }
            })
            .collect();

        let total = new_intervals.iter().map(|interval| interval.length).sum();
        allowed_intervals[col] = new_intervals;
        total_positions[col] = total;
    }

    (allowed_intervals, total_positions)
}
