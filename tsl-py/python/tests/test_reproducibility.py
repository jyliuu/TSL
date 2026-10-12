import numpy as np
from tensorsl import TSL
import pytest


def generate_test_data(n_samples=100, n_features=2, random_state=42):
    """Generate synthetic data for testing."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    y = np.sum(X**2, axis=1) + rng.randn(n_samples) * 0.1
    return X, y


def test_tsl_boosted_reproducibility():
    """Test that TSLBoosted with same seed produces identical predictions."""
    X, y = generate_test_data()

    # Train first model
    model1, _ = TSL.fit(
        X, y, epochs=2, n_trees=5, n_iter=10, split_try=5, colsample_bysplit=1.0, seed=42
    )

    # Train second model with same parameters and seed
    model2, _ = TSL.fit(
        X, y, epochs=2, n_trees=5, n_iter=10, split_try=5, colsample_bysplit=1.0, seed=42
    )
    # Generate predictions
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)

    print(pred1)
    print(pred2)
    # Check predictions are identical
    np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different predictions."""
    X, y = generate_test_data()

    # Train models with different seeds
    model1, _ = TSL.fit(
        X, y, epochs=2, n_trees=5, n_iter=10, split_try=5, colsample_bysplit=1.0, seed=42
    )

    model2, _ = TSL.fit(
        X, y, epochs=2, n_trees=5, n_iter=10, split_try=5, colsample_bysplit=1.0, seed=43
    )

    # Generate predictions
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)

    # Check predictions are different
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)


def test_consensus_fit_is_reproducible_and_validated():
    X, y = generate_test_data(n_samples=300)
    kwargs = dict(epochs=2, n_trees=6, n_iter=10, split_try=5, colsample_bysplit=1.0, seed=7)

    model1, _ = TSL.fit(X, y, consensus=1.0, **kwargs)
    model2, _ = TSL.fit(X, y, consensus=1.0, **kwargs)
    np.testing.assert_array_equal(model1.predict(X), model2.predict(X))

    for bad in (-0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="consensus"):
            TSL.fit(X, y, consensus=bad, **kwargs)


def test_sklearn_regressor_forwards_consensus():
    from tensorsl.sklearn import TSLRegressor

    X, y = generate_test_data(n_samples=300)
    reg = TSLRegressor(epochs=2, n_trees=6, n_iter=10, consensus=1.0, verbosity=0)
    assert reg.get_params()["consensus"] == 1.0
    pred = reg.fit(X, y).predict(X)
    assert np.isfinite(pred).all()
