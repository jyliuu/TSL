//! Row caches are contiguous, value-identical to their definitions, and staggered.

use ndarray::{Array1, Array2};

use super::{staggered_zeros, FittingState};

fn offset(array: &Array1<f64>) -> usize {
    array.clone().into_raw_vec_and_offset().1.unwrap_or(0)
}

#[test]
fn row_caches_start_in_distinct_kib_slots_and_keep_their_definitions() {
    for n in [1usize, 7, 20_000] {
        let x = Array2::from_shape_fn((n, 3), |(i, j)| (i * 7 + j) as f64 / 3.0 - 2.0);
        let y = Array1::from_shape_fn(n, |i| ((i * 13) % 11) as f64 - 4.5);
        let state = FittingState::new(x.view(), y.view());
        let caches = [
            (&state.f_plus, 1),
            (&state.f_minus, 2),
            (&state.f, 3),
            (&state.y_hat, 4),
            (&state.residuals, 5),
            (&state.r_tilde, 6),
        ];
        for (cache, slot) in caches {
            assert_eq!(cache.len(), n);
            assert!(cache.as_slice().is_some());
            assert_eq!(offset(cache), slot * 128);
        }
        for i in 0..n {
            assert_eq!(state.f_plus[i].to_bits(), state.lambda_plus.to_bits());
            assert_eq!(state.f_minus[i].to_bits(), state.lambda_minus.to_bits());
            let f = state.lambda_plus - state.lambda_minus;
            assert_eq!(state.f[i].to_bits(), f.to_bits());
            assert_eq!(state.y_hat[i].to_bits(), f.to_bits());
            assert_eq!(state.residuals[i].to_bits(), (y[i] - f).to_bits());
            assert_eq!(state.r_tilde[i].to_bits(), (y[i] - f).to_bits());
        }
        let expected: f64 = state.residuals.iter().map(|r| r * r).sum();
        assert_eq!(state.current_error.to_bits(), expected.to_bits());
        let cloned = state.clone();
        assert_eq!(offset(&cloned.f_plus), 128);
        assert_eq!(offset(&cloned.r_tilde), 6 * 128);
        assert_eq!(cloned.f_plus, state.f_plus);
        assert_eq!(cloned.residuals, state.residuals);
    }
}

#[test]
fn staggered_zeros_are_contiguous_zero_filled_and_padded_in_front() {
    let array = staggered_zeros(5, 3);
    assert_eq!(array.len(), 5);
    assert!(array.iter().all(|&v| v.to_bits() == 0));
    assert_eq!(array.as_slice().map(<[f64]>::len), Some(5));
    let (storage, start) = array.into_raw_vec_and_offset();
    assert_eq!((storage.len(), start), (5 + 3 * 128, Some(3 * 128)));
    assert_eq!(staggered_zeros(0, 0).len(), 0);
}
