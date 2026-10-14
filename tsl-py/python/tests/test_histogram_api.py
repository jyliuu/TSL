"""Python access to the original fitter's quantile candidate grid."""
import numpy as np
import pytest

from tensorsl import TSL, TSLRegressor


def data():
    x = np.linspace(-3.0, 3.0, 240).reshape(-1, 1)
    return x, np.sin(5.0 * x[:, 0]) + 0.3 * x[:, 0]


@pytest.mark.parametrize("consensus", [0.0, 0.3])
def test_histogram_candidates_bound_every_bag(consensus):
    x, y = data()
    model, _ = TSL.fit(
        x, y, epochs=2, n_trees=3, n_iter=16, split_try=32,
        colsample_bysplit=1.0, consensus=consensus, max_bins=4, verbosity=0,
    )
    for stage in model.stage_predictors:
        for grid in stage.grid_tensors:
            assert len(grid.splits[0]) <= 3
    assert np.isfinite(model.predict(x)).all()


def test_default_bins_and_exact_candidate_path():
    x, y = data()
    config = dict(epochs=2, n_trees=3, n_iter=16, verbosity=0)
    default, _ = TSL.fit(x, y, **config)
    binned, _ = TSL.fit(x, y, max_bins=4095, **config)
    np.testing.assert_array_equal(default.predict(x), binned.predict(x))
    exact, _ = TSL.fit(x, y, max_bins="exact", **config)
    unset, _ = TSL.fit(x, y, max_bins=None, **config)
    np.testing.assert_array_equal(exact.predict(x), unset.predict(x))
    assert any(len(grid.splits[0]) > 3 for stage in exact.stage_predictors for grid in stage.grid_tensors)


def test_regressor_passes_max_bins_to_the_fit():
    x, y = data()
    config = dict(epochs=2, n_trees=3, n_iter=16, verbosity=0)
    assert TSLRegressor().get_params()["max_bins"] == 4095
    for max_bins in [4, "exact"]:
        regressor = TSLRegressor(max_bins=max_bins, **config).fit(x, y)
        model, _ = TSL.fit(x, y, max_bins=max_bins, **config)
        np.testing.assert_array_equal(regressor.predict(x), model.predict(x))


@pytest.mark.parametrize(
    "max_bins, error",
    [(0, ValueError), (1, ValueError), ("bins", ValueError), (2.5, TypeError), (True, TypeError)],
)
def test_histogram_bin_count_validation(max_bins, error):
    x, y = data()
    with pytest.raises(error, match="max_bins must be"):
        TSL.fit(x, y, epochs=1, max_bins=max_bins)
