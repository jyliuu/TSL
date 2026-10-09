"""Consensus cadence and sign-preserving outer calibration."""

import numpy as np
import pytest

from tensorsl import TSL


def data():
    rng = np.random.default_rng(123)
    x = rng.uniform(-2.0, 2.0, (180, 3))
    y = np.sin(2.0 * x[:, 0]) * np.exp(0.2 * x[:, 1]) + 0.3 * x[:, 2]
    return x, y


@pytest.mark.parametrize("consensus", [0.0, 0.3])
def test_explicit_defaults_preserve_predictions(consensus):
    x, y = data()
    params = dict(epochs=2, n_trees=5, n_iter=10, max_bins=16,
                  consensus=consensus, verbosity=0)
    default, _ = TSL.fit(x, y, **params)
    explicit, _ = TSL.fit(x, y, consensus_interval=1,
                          nonnegative_scaling=False, **params)
    np.testing.assert_array_equal(default.predict(x), explicit.predict(x))


@pytest.mark.parametrize("interval", [1, 4, 64])
def test_nonnegative_branch_coefficients_and_reproducibility(interval):
    x, y = data()
    params = dict(epochs=3, n_trees=5, n_iter=10, max_bins=16,
                  consensus=0.3, consensus_interval=interval,
                  nonnegative_scaling=True, verbosity=0)
    model, _ = TSL.fit(x, y, **params)
    repeat, _ = TSL.fit(x, y, **params)
    prediction = model.predict(x)
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction, repeat.predict(x))
    for stage in model.stage_predictors:
        assert stage.scaling_plus >= 0.0
        assert stage.scaling_minus >= 0.0


def test_zero_interval_is_rejected():
    x, y = data()
    with pytest.raises(ValueError, match="consensus_interval must be at least 1"):
        TSL.fit(x, y, epochs=1, consensus_interval=0, verbosity=0)
