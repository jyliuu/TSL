"""Python access to the original fitter's quantile candidate grid."""
import numpy as np
import pytest

from tensorsl import TSL


def data():
    x = np.linspace(-3.0, 3.0, 240).reshape(-1, 1)
    return x, np.sin(5.0 * x[:, 0]) + 0.3 * x[:, 0]


@pytest.mark.parametrize("consensus", [0.0, 0.3])
def test_histogram_candidates_bound_every_bag(consensus):
    x, y = data()
    model, _ = TSL.fit(
        x, y, epochs=2, n_trees=3, n_iter=16, split_try=32,
        colsample_bytree=1.0, consensus=consensus, max_bins=4, verbosity=0,
    )
    for stage in model.stage_predictors:
        for grid in stage.grid_tensors:
            assert len(grid.splits[0]) <= 3
    assert np.isfinite(model.predict(x)).all()


def test_none_preserves_exact_candidate_path():
    x, y = data()
    config = dict(epochs=2, n_trees=3, n_iter=16, verbosity=0)
    default, _ = TSL.fit(x, y, **config)
    exact, _ = TSL.fit(x, y, max_bins=None, **config)
    np.testing.assert_array_equal(default.predict(x), exact.predict(x))
    assert any(len(grid.splits[0]) > 3 for stage in exact.stage_predictors for grid in stage.grid_tensors)


@pytest.mark.parametrize("max_bins", [0, 1])
def test_histogram_bin_count_validation(max_bins):
    x, y = data()
    with pytest.raises(ValueError, match="max_bins must be at least 2"):
        TSL.fit(x, y, epochs=1, max_bins=max_bins)
