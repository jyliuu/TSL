//! Chunked member branch totals match the sequential elementwise sum bit for bit.

use super::{accumulate_full_row_totals, Member};
use crate::grid_tensor::state::FittingState;
use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn branch_values(rng: &mut StdRng, n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |_| match rng.gen_range(0..16) {
        0 => 0.0,
        1 => -0.0,
        2 => f64::MIN_POSITIVE / 8.0,
        3 => -1e300,
        4 => 1e-300,
        _ => rng.gen_range(-5.0..5.0) * 10f64.powi(rng.gen_range(-8..8)),
    })
}

fn members<'a>(x: &'a Array2<f64>, y: &'a Array1<f64>, count: usize, seed: u64) -> Vec<Member<'a>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|tree_id| {
            let mut state = FittingState::new(x.view(), y.view());
            state.f_plus = branch_values(&mut rng, x.nrows());
            state.f_minus = branch_values(&mut rng, x.nrows());
            Member {
                tree_id,
                rng: StdRng::seed_from_u64(seed + tree_id as u64),
                state: Some(state),
                next_iteration: 0,
            }
        })
        .collect()
}

fn sequential_totals(members: &[Member<'_>], n: usize) -> (Array1<f64>, Array1<f64>) {
    let (mut plus, mut minus) = (Array1::zeros(n), Array1::zeros(n));
    for member in members {
        let state = member.state.as_ref().unwrap();
        plus += &state.f_plus;
        minus += &state.f_minus;
    }
    (plus, minus)
}

fn assert_bits_equal(actual: &Array1<f64>, expected: &Array1<f64>) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "row {index}: {a:?} != {e:?}");
    }
}

#[test]
fn chunked_totals_match_the_sequential_sum_across_chunk_boundaries() {
    for (n, count) in [
        (1, 3),
        (511, 7),
        (512, 2),
        (513, 5),
        (2049, 11),
        (40_000, 50),
    ] {
        let x = Array2::zeros((n, 2));
        let y = Array1::zeros(n);
        let members = members(&x, &y, count, n as u64);
        let (expected_plus, expected_minus) = sequential_totals(&members, n);
        let (mut plus, mut minus) = (
            Array1::from_elem(n, f64::NAN),
            Array1::from_elem(n, f64::NAN),
        );
        assert!(accumulate_full_row_totals(&members, &mut plus, &mut minus));
        assert_bits_equal(&plus, &expected_plus);
        assert_bits_equal(&minus, &expected_minus);
    }
}

#[test]
fn noncontiguous_or_mismatched_branches_leave_the_totals_untouched() {
    let n = 700;
    let x = Array2::zeros((n, 2));
    let y = Array1::zeros(n);
    let untouched = |members: &[Member<'_>], len: usize| {
        let (mut plus, mut minus) = (Array1::from_elem(len, 7.0), Array1::from_elem(len, -7.0));
        assert!(!accumulate_full_row_totals(members, &mut plus, &mut minus));
        assert!(plus.iter().all(|&v| v == 7.0) && minus.iter().all(|&v| v == -7.0));
    };

    let mut stride_two = members(&x, &y, 3, 9);
    stride_two[1].state.as_mut().unwrap().f_plus =
        Array1::from_shape_fn(2 * n, |i| i as f64).slice_move(ndarray::s![..;2]);
    assert!(stride_two[1]
        .state
        .as_ref()
        .unwrap()
        .f_plus
        .as_slice()
        .is_none());
    untouched(&stride_two, n);

    let mut reversed = members(&x, &y, 3, 10);
    let mut minus = Array1::from_shape_fn(n, |i| i as f64);
    minus.invert_axis(ndarray::Axis(0));
    reversed[2].state.as_mut().unwrap().f_minus = minus;
    assert!(reversed[2]
        .state
        .as_ref()
        .unwrap()
        .f_minus
        .as_slice()
        .is_none());
    untouched(&reversed, n);

    untouched(&members(&x, &y, 2, 11), n + 1);
    untouched(&members(&x, &y, 2, 12), n - 1);
}
