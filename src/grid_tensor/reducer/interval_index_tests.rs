//! The on-demand interval index matches a table maintained by split and merge bookkeeping.

use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, SeedableRng};

use super::{compute_partial_products_for_axis, fitting_reducer};
use crate::grid_tensor::{
    action::FittingAction,
    fit::{fit_step, initialize_state_with_input},
    params::{
        GridTensorParamsBuilder, RefinementStrategyParamsBuilder, SplitStrategyParamsBuilder,
    },
    splitting::{MergeCandidate, SplitCandidate},
    state::FittingState,
};

fn fixture() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((131, 3), |(row, col)| match col {
        0 => ((row * 37) % 131) as f64 / 13.0,
        1 => ((row * 53) % 7) as f64,
        _ => ((row * 71) % 131) as f64 / 29.0 - 2.0,
    });
    let y = Array1::from_shape_fn(131, |row| {
        (x[(row, 0)] * 0.7).sin() + 0.3 * x[(row, 1)] - 0.2 * x[(row, 2)] * x[(row, 2)]
    });
    (x, y)
}

/// Apply the bookkeeping a per-row table would receive for one structural action: a
/// boundary inserted at rank `b` adds one to every row at rank `b` or above, a removed
/// boundary subtracts one from the same rows, and a resplit changes nothing.
fn advance_reference(
    reference: &mut [Vec<usize>],
    before: &[Vec<usize>],
    after: &[Vec<usize>],
    state: &FittingState,
) {
    let mut changed = 0;
    for col in 0..before.len() {
        let (old, new) = (&before[col], &after[col]);
        if old == new {
            continue;
        }
        changed += 1;
        let position = old
            .iter()
            .zip(new)
            .position(|(a, b)| a != b)
            .unwrap_or(old.len().min(new.len()));
        let (boundary, delta): (usize, isize) = if new.len() == old.len() + 1 {
            (new[position], 1)
        } else {
            assert_eq!(old.len(), new.len() + 1);
            (old[position], -1)
        };
        let ranks = &state.precomputed_statistics.sort_order[col];
        for (row, slot) in reference[col].iter_mut().enumerate() {
            if ranks[row] >= boundary {
                *slot = (*slot as isize + delta) as usize;
            }
        }
    }
    assert!(
        changed <= 1,
        "one structural action edits at most one column"
    );
}

fn table_partial_products(
    state: &FittingState,
    col: usize,
    table: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let mut g_plus = Vec::with_capacity(state.n);
    let mut g_minus = Vec::with_capacity(state.n);
    for i in 0..state.n {
        let interval_idx = table[i];
        let b_j = state.backbone_values[col][interval_idx];
        let d_j = state.tilt_values[col][interval_idx];
        let a_plus_j = b_j * d_j.exp();
        let a_minus_j = b_j * (-d_j).exp();
        g_plus.push(state.f_plus[i] / a_plus_j);
        g_minus.push(state.f_minus[i] / a_minus_j);
    }
    (g_plus, g_minus)
}

fn assert_indices_and_partials(state: &FittingState, reference: &[Vec<usize>]) {
    for col in 0..state.p {
        for row in 0..state.n {
            assert_eq!(
                state.interval_index(col, row),
                reference[col][row],
                "col {col} row {row}"
            );
        }
        let (plus, minus) = compute_partial_products_for_axis(col, state);
        let (expected_plus, expected_minus) = table_partial_products(state, col, &reference[col]);
        for (a, e) in plus
            .iter()
            .zip(&expected_plus)
            .chain(minus.iter().zip(&expected_minus))
        {
            assert_eq!(a.to_bits(), e.to_bits());
        }
    }
}

#[test]
fn interval_index_tracks_split_merge_bookkeeping_through_fits() {
    let (x, y) = fixture();
    let mut scripted = 0;
    for huber in [false, true] {
        for bins in [None, Some(8)] {
            for seed in [3u64, 11] {
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
                    .n_iter(14)
                    .max_bins(bins)
                    .refinement_strategy(refinement)
                    .split_strategy(
                        SplitStrategyParamsBuilder::new()
                            .random_split()
                            .split_try(6)
                            .min_interval_samples(2)
                            .complexity_penalty(0.02)
                            .build(),
                    )
                    .build();
                let strategy = params.refinement_strategy_params.get_refinement_strategy();
                let split = params.split_strategy_params.get_split_strategy();
                let mut state = initialize_state_with_input(x.view(), y.view(), &params, None);
                let mut reference = vec![vec![0usize; state.n]; state.p];
                assert_indices_and_partials(&state, &reference);
                let mut rng = StdRng::seed_from_u64(seed);
                let (mut merges, mut splits) = (0, 0);
                for iteration in 0..40 {
                    let before = state.boundaries.clone();
                    let (next, advanced) =
                        fit_step(state, iteration, params.n_iter, &strategy, &split, &mut rng);
                    state = next;
                    let after = state.boundaries.clone();
                    let (old, new): (usize, usize) = (
                        before.iter().map(Vec::len).sum(),
                        after.iter().map(Vec::len).sum(),
                    );
                    splits += usize::from(new > old);
                    merges += usize::from(new < old);
                    advance_reference(&mut reference, &before, &after, &state);
                    assert_indices_and_partials(&state, &reference);
                    if !advanced {
                        break;
                    }
                }
                assert!(splits > 0, "the fit split");
                let _ = merges;
                // Scripted actions cover a split at an existing boundary rank and merges at
                // the first, middle and last boundary of a column.
                if let Some(col) = (0..state.p).find(|&col| state.boundaries[col].len() >= 3) {
                    scripted += 1;
                    let index = state.boundaries[col][1];
                    if let Some(allowed_interval_idx) = state.split_strategy_state.allowed_intervals
                        [col]
                        .iter()
                        .position(|interval| interval.contains(index))
                    {
                        let split_at_boundary = SplitCandidate {
                            col,
                            index,
                            allowed_interval_idx,
                            error_reduction: 0.0,
                            update_left: (0.0, 0.0),
                            update_right: (0.0, 0.0),
                        };
                        let before = state.boundaries.clone();
                        state = fitting_reducer(
                            state,
                            FittingAction::ApplySplit {
                                split: split_at_boundary,
                            },
                            &strategy,
                            &split,
                        );
                        let after = state.boundaries.clone();
                        assert_eq!(after[col].len(), before[col].len() + 1);
                        advance_reference(&mut reference, &before, &after, &state);
                        assert_indices_and_partials(&state, &reference);
                    }
                    for pick in [0usize, 1, 2] {
                        let count = state.boundaries[col].len();
                        if count == 0 {
                            break;
                        }
                        let interval_idx = match pick {
                            0 => 0,
                            1 => count / 2,
                            _ => count - 1,
                        };
                        let merge = MergeCandidate {
                            col,
                            interval_idx,
                            index: state.boundaries[col][interval_idx],
                            error_reduction: state.precomputed_statistics.error_reductions_merge
                                [col][interval_idx],
                        };
                        let before = state.boundaries.clone();
                        state = fitting_reducer(
                            state,
                            FittingAction::ApplyMerge { merge },
                            &strategy,
                            &split,
                        );
                        let after = state.boundaries.clone();
                        assert_eq!(after[col].len() + 1, before[col].len());
                        advance_reference(&mut reference, &before, &after, &state);
                        assert_indices_and_partials(&state, &reference);
                    }
                }
            }
        }
    }
    assert!(scripted > 0, "scripted merges ran");
}
