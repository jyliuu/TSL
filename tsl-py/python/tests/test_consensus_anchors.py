"""Training-row support for branch alignment with all-row prediction fitting."""

import numpy as np
import pytest

from tensorsl import TSL


def data():
    rng = np.random.default_rng(724)
    x = rng.normal(size=(256, 3))
    y = np.sin(2.0 * x[:, 0]) + x[:, 1] * x[:, 2]
    return x, y


def test_full_anchor_support_preserves_default_predictions():
    x, y = data()
    params = dict(epochs=2, n_trees=5, n_iter=12, consensus=0.3,
                  consensus_interval=4, max_bins=16, verbosity=0)
    baseline, _ = TSL.fit(x, y, **params)
    for count in [None, len(x), len(x) + 10]:
        model, _ = TSL.fit(x, y, consensus_anchors=count, **params)
        np.testing.assert_array_equal(model.predict(x), baseline.predict(x))


@pytest.mark.parametrize("max_bins", [None, 16])
def test_sampled_anchor_fits_are_finite_and_reproducible(max_bins):
    x, y = data()
    params = dict(epochs=2, n_trees=5, n_iter=12, consensus=0.3,
                  consensus_interval=4, consensus_anchors=32, max_bins=max_bins,
                  nonnegative_scaling=True, verbosity=0)
    model, _ = TSL.fit(x, y, **params)
    repeat, _ = TSL.fit(x, y, **params)
    assert np.isfinite(model.predict(x)).all()
    np.testing.assert_array_equal(model.predict(x), repeat.predict(x))
    for stage in model.stage_predictors:
        assert len(stage.grid_tensors) == 5
        assert stage.scaling_plus >= 0.0
        assert stage.scaling_minus >= 0.0


def test_empty_anchor_support_is_rejected():
    x, y = data()
    with pytest.raises(ValueError, match="consensus_anchors must be positive"):
        TSL.fit(x, y, epochs=1, consensus_anchors=0, verbosity=0)
