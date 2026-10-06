use crate::grid_tensor::GridTensor;

pub(crate) fn refine_grids_to_union_two_tensor(grids: &[GridTensor]) -> Vec<GridTensor> {
    if grids.is_empty() {
        return Vec::new();
    }

    let num_axes = grids[0].intervals.len();
    let mut union_splits: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
    let mut union_intervals: Vec<Vec<(f64, f64)>> = Vec::with_capacity(num_axes);

    for axis in 0..num_axes {
        let mut splits: Vec<f64> = grids
            .iter()
            .flat_map(|grid| grid.splits[axis].iter().copied())
            .collect();

        splits.sort_by(|a, b| a.partial_cmp(b).unwrap());
        splits.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        let mut intervals: Vec<(f64, f64)> = Vec::new();
        if splits.is_empty() {
            intervals.push((f64::NEG_INFINITY, f64::INFINITY));
        } else {
            intervals.push((f64::NEG_INFINITY, splits[0]));
            for i in 0..splits.len() - 1 {
                intervals.push((splits[i], splits[i + 1]));
            }
            intervals.push((splits[splits.len() - 1], f64::INFINITY));
        }

        union_splits.push(splits);
        union_intervals.push(intervals);
    }

    let mut refined_grids = Vec::with_capacity(grids.len());
    for grid in grids {
        let mut refined_backbone: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
        let mut refined_tilt: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
        let mut refined_observation_counts: Vec<Vec<usize>> = Vec::with_capacity(num_axes);

        for axis in 0..num_axes {
            let n_union_bins = union_intervals[axis].len();
            let mut backbone_axis = Vec::with_capacity(n_union_bins);
            let mut tilt_axis = Vec::with_capacity(n_union_bins);
            let counts_axis = vec![0; n_union_bins];

            let original_intervals = &grid.intervals[axis];
            let mut orig_idx = 0;
            for &(union_a, union_b) in &union_intervals[axis] {
                while orig_idx < original_intervals.len()
                    && original_intervals[orig_idx].1 < union_b
                {
                    orig_idx += 1;
                }
                let parent = original_intervals
                    .get(orig_idx)
                    .filter(|&&(orig_a, orig_b)| union_a >= orig_a && union_b <= orig_b);
                if parent.is_some() {
                    backbone_axis.push(grid.backbone_values[axis][orig_idx]);
                    tilt_axis.push(grid.tilt_values[axis][orig_idx]);
                } else {
                    log::warn!(
                        "Could not find parent interval for union interval [{}, {})",
                        union_a,
                        union_b
                    );
                    backbone_axis.push(1.0);
                    tilt_axis.push(0.0);
                }
            }

            refined_backbone.push(backbone_axis);
            refined_tilt.push(tilt_axis);
            refined_observation_counts.push(counts_axis);
        }

        let refined_grid = GridTensor::new_two_tensor(
            union_splits.clone(),
            refined_observation_counts,
            union_intervals.clone(),
            refined_backbone,
            refined_tilt,
            grid.lambda_plus,
            grid.lambda_minus,
        );

        refined_grids.push(refined_grid);
    }

    refined_grids
}

/// Sign-preserving geometric mean for prediction-space aggregation
/// (`Aggregation::GeometricMean`).
pub(super) fn geometric_mean_combiner(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sign = values.iter().map(|v| v.signum()).sum::<f64>().signum();
    let log_sum = values.iter().map(|v| v.abs().ln()).sum::<f64>();
    let geom_mean = (log_sum / values.len() as f64).exp();
    sign * geom_mean
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_grid(splits: Vec<Vec<f64>>, offset: usize) -> GridTensor {
        let intervals = splits
            .iter()
            .map(|axis| {
                std::iter::once(f64::NEG_INFINITY)
                    .chain(axis.iter().copied())
                    .zip(axis.iter().copied().chain(std::iter::once(f64::INFINITY)))
                    .collect()
            })
            .collect();
        let backbone = splits
            .iter()
            .map(|axis| {
                (0..=axis.len())
                    .map(|bin| (offset + bin + 1) as f64)
                    .collect()
            })
            .collect();
        let tilt = splits
            .iter()
            .map(|axis| {
                (0..=axis.len())
                    .map(|bin| 0.01 * (offset + bin) as f64)
                    .collect()
            })
            .collect();
        let counts = splits.iter().map(|axis| vec![1; axis.len() + 1]).collect();
        GridTensor::new_two_tensor(splits, counts, intervals, backbone, tilt, 1.5, 0.75)
    }

    fn assert_aligned_identical(actual: &[GridTensor], expected: &[GridTensor]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.splits, expected.splits);
            assert_eq!(actual.intervals, expected.intervals);
            assert_eq!(actual.observation_counts, expected.observation_counts);
            assert_eq!(actual.backbone_values, expected.backbone_values);
            assert_eq!(actual.tilt_values, expected.tilt_values);
            assert_eq!(actual.lambda_plus, expected.lambda_plus);
            assert_eq!(actual.lambda_minus, expected.lambda_minus);
        }
    }

    #[test]
    fn union_alignment_matches_nested_scan_at_ties_and_near_duplicate_boundaries() {
        let grids = vec![
            fixture_grid(vec![vec![-2.0, 0.0, 3.0], vec![]], 0),
            fixture_grid(vec![vec![-2.0, 1.0, 3.0], vec![0.0]], 10),
            fixture_grid(vec![vec![-2.0 + 5e-13, 2.0, 3.0], vec![-1.0, 0.0]], 20),
            fixture_grid(vec![vec![], vec![0.0, 2.0]], 30),
        ];
        assert_aligned_identical(
            &refine_grids_to_union_two_tensor(&grids),
            &reference_alignment(&grids),
        );
        assert_aligned_identical(&refine_grids_to_union_two_tensor(&[]), &[]);
    }

    #[test]
    fn union_alignment_preserves_grid_predictions() {
        let grids: Vec<_> = (0..25)
            .map(|grid| {
                fixture_grid(
                    vec![
                        (0..9).map(|bin| (bin * 7 + grid) as f64 / 13.0).collect(),
                        (0..6).map(|bin| (bin * 3 + grid) as f64 / 11.0).collect(),
                    ],
                    grid,
                )
            })
            .collect();
        let aligned = refine_grids_to_union_two_tensor(&grids);
        assert_aligned_identical(&aligned, &reference_alignment(&grids));
        for (original, refined) in grids.iter().zip(&aligned) {
            for index in -10..100 {
                let point = [index as f64 / 13.0, index as f64 / 11.0];
                assert_eq!(
                    original.predict_single(&point).to_bits(),
                    refined.predict_single(&point).to_bits()
                );
            }
        }
    }

    #[test]
    #[ignore = "wall-clock comparison of interval alignment algorithms"]
    fn benchmark_union_alignment() {
        use std::hint::black_box;
        use std::time::Instant;
        let grids: Vec<_> = (0..100)
            .map(|grid| {
                fixture_grid(
                    vec![
                        (0..50).map(|bin| (bin * 100 + grid) as f64).collect(),
                        (0..50).map(|bin| (bin * 100 + grid) as f64).collect(),
                    ],
                    grid,
                )
            })
            .collect();
        assert_aligned_identical(
            &refine_grids_to_union_two_tensor(&grids),
            &reference_alignment(&grids),
        );
        let start = Instant::now();
        for _ in 0..4 {
            black_box(reference_alignment(black_box(&grids)));
        }
        let reference = start.elapsed();
        let start = Instant::now();
        for _ in 0..4 {
            black_box(refine_grids_to_union_two_tensor(black_box(&grids)));
        }
        let monotone = start.elapsed();
        println!(
            "union alignment: reference={reference:?}, monotone={monotone:?}, speedup={:.3}",
            reference.as_secs_f64() / monotone.as_secs_f64()
        );
    }

    fn reference_alignment(grids: &[GridTensor]) -> Vec<GridTensor> {
        if grids.is_empty() {
            return Vec::new();
        }

        let num_axes = grids[0].intervals.len();
        let mut union_splits: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
        let mut union_intervals: Vec<Vec<(f64, f64)>> = Vec::with_capacity(num_axes);

        for axis in 0..num_axes {
            let mut splits: Vec<f64> = grids
                .iter()
                .flat_map(|grid| grid.splits[axis].iter().copied())
                .collect();

            splits.sort_by(|a, b| a.partial_cmp(b).unwrap());
            splits.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

            let mut intervals: Vec<(f64, f64)> = Vec::new();
            if splits.is_empty() {
                intervals.push((f64::NEG_INFINITY, f64::INFINITY));
            } else {
                intervals.push((f64::NEG_INFINITY, splits[0]));
                for i in 0..splits.len() - 1 {
                    intervals.push((splits[i], splits[i + 1]));
                }
                intervals.push((splits[splits.len() - 1], f64::INFINITY));
            }

            union_splits.push(splits);
            union_intervals.push(intervals);
        }

        let mut refined_grids = Vec::with_capacity(grids.len());
        for grid in grids {
            let mut refined_backbone: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
            let mut refined_tilt: Vec<Vec<f64>> = Vec::with_capacity(num_axes);
            let mut refined_observation_counts: Vec<Vec<usize>> = Vec::with_capacity(num_axes);

            for axis in 0..num_axes {
                let n_union_bins = union_intervals[axis].len();
                let mut backbone_axis = Vec::with_capacity(n_union_bins);
                let mut tilt_axis = Vec::with_capacity(n_union_bins);
                let counts_axis = vec![0; n_union_bins];

                for &(union_a, union_b) in &union_intervals[axis] {
                    let mut found = false;
                    for (orig_idx, &(orig_a, orig_b)) in grid.intervals[axis].iter().enumerate() {
                        if union_a >= orig_a && union_b <= orig_b {
                            backbone_axis.push(grid.backbone_values[axis][orig_idx]);
                            tilt_axis.push(grid.tilt_values[axis][orig_idx]);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        log::warn!(
                            "Could not find parent interval for union interval [{}, {})",
                            union_a,
                            union_b
                        );
                        backbone_axis.push(1.0);
                        tilt_axis.push(0.0);
                    }
                }

                refined_backbone.push(backbone_axis);
                refined_tilt.push(tilt_axis);
                refined_observation_counts.push(counts_axis);
            }

            let refined_grid = GridTensor::new_two_tensor(
                union_splits.clone(),
                refined_observation_counts,
                union_intervals.clone(),
                refined_backbone,
                refined_tilt,
                grid.lambda_plus,
                grid.lambda_minus,
            );

            refined_grids.push(refined_grid);
        }

        refined_grids
    }
}
