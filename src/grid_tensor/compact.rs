use super::GridTensor;

impl GridTensor {
    /// Merge adjacent bins whose backbone and tilt values are bitwise equal.
    pub(crate) fn coalesce_equal_bins(&mut self) -> usize {
        let mut removed = 0;
        for axis in 0..self.backbone_values.len() {
            let bins = self.backbone_values[axis].len();
            if bins < 2 {
                continue;
            }
            let mut kept = 0;
            for source in 1..bins {
                if self.backbone_values[axis][source].to_bits()
                    == self.backbone_values[axis][kept].to_bits()
                    && self.tilt_values[axis][source].to_bits()
                        == self.tilt_values[axis][kept].to_bits()
                {
                    self.observation_counts[axis][kept] += self.observation_counts[axis][source];
                    self.intervals[axis][kept].1 = self.intervals[axis][source].1;
                    removed += 1;
                } else {
                    self.splits[axis][kept] = self.splits[axis][source - 1];
                    kept += 1;
                    self.backbone_values[axis][kept] = self.backbone_values[axis][source];
                    self.tilt_values[axis][kept] = self.tilt_values[axis][source];
                    self.observation_counts[axis][kept] = self.observation_counts[axis][source];
                    self.intervals[axis][kept] = self.intervals[axis][source];
                }
            }
            self.splits[axis].truncate(kept);
            self.backbone_values[axis].truncate(kept + 1);
            self.tilt_values[axis].truncate(kept + 1);
            self.observation_counts[axis].truncate(kept + 1);
            self.intervals[axis].truncate(kept + 1);
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use crate::stage_predictor::extract_two_tensor_predictions_unscaled;

    #[test]
    fn coalescing_preserves_boundaries_branches_and_observation_mass() {
        let splits = vec![vec![-2.0, 0.0, 1.0, 3.0, 5.0], vec![0.0]];
        let intervals = splits.iter().map(|cuts| {
            let edges: Vec<_> = std::iter::once(f64::NEG_INFINITY).chain(cuts.iter().copied())
                .chain(std::iter::once(f64::INFINITY)).collect();
            edges.windows(2).map(|pair|(pair[0],pair[1])).collect()
        }).collect();
        let mut grid = GridTensor::new_two_tensor(splits,
            vec![vec![1,2,3,4,5,6],vec![10,11]], intervals,
            vec![vec![1.,1.,2.,2.,2.,3.],vec![2.,2.]],
            vec![vec![0.2,0.2,-0.1,-0.1,0.3,0.3],vec![0.5,0.5]],2.5,0.7);
        grid.scaling = 7.0;
        let mut axis_values = vec![-10.0,10.0];
        for &cut in &grid.splits[0] {
            axis_values.extend([cut.next_down(),cut,cut.next_up()]);
        }
        let points = Array2::from_shape_fn((axis_values.len()*3,2),|(i,j)| {
            if j==0 {axis_values[i/3]} else {[-1.,0.,1.][i%3]}
        });
        let expected = extract_two_tensor_predictions_unscaled(&grid,points.view());
        let expected_backbone_tilt = grid.predict_backbone_and_tilt(points.view());
        assert_eq!(grid.coalesce_equal_bins(),3);
        assert_eq!(grid.splits,vec![vec![0.,3.,5.],vec![]]);
        assert_eq!(grid.observation_counts,vec![vec![3,7,5,6],vec![21]]);
        assert_eq!(grid.intervals,vec![
            vec![(f64::NEG_INFINITY,0.),(0.,3.),(3.,5.),(5.,f64::INFINITY)],
            vec![(f64::NEG_INFINITY,f64::INFINITY)]]);
        assert_eq!(extract_two_tensor_predictions_unscaled(&grid,points.view()),expected);
        assert_eq!(grid.predict_backbone_and_tilt(points.view()),expected_backbone_tilt);
        assert_eq!((grid.lambda_plus,grid.lambda_minus,grid.scaling),(2.5,0.7,7.));
        assert_eq!(grid.coalesce_equal_bins(),0);
    }

    #[test]
    fn coalescing_keeps_distinct_floating_point_values() {
        let mut grid = GridTensor::new_two_tensor(vec![vec![0.]],vec![vec![1,1]],
            vec![vec![(f64::NEG_INFINITY,0.),(0.,f64::INFINITY)]],
            vec![vec![1.0,1.0_f64.next_up()]],vec![vec![0.,0.]],1.,1.);
        assert_eq!(grid.coalesce_equal_bins(),0);
        grid.backbone_values[0][1] = 1.;
        grid.tilt_values[0][1] = -0.;
        assert_eq!(grid.coalesce_equal_bins(),0);
    }
}
