import numpy as np
import pytest
from tsl_py import TSL, GridTensor


def gen_data(n=5000, seed=1):
    np.random.seed(seed)
    X = np.random.uniform(-5, 5, size=(n, 2))
    # y = 3*np.sin(3 * X[:,0])*np.cos(5*X[:,1]) + np.random.normal(scale=0.5, size=n)
    y = (
        np.exp(np.sin(X[:, 0]) * np.cos(X[:, 1]))
        + X[:, 0]
        + np.random.normal(scale=0.5, size=n)
    )
    return X, y


@pytest.fixture(scope="module")
def training_data():
    X, y = gen_data(seed=1)
    return X, y.ravel()


@pytest.fixture(scope="module")
def test_data():
    X_test, y_test = gen_data(seed=2)
    return X_test, y_test.ravel()


def test_grid_tensor_fit(training_data, test_data):
    X, y = training_data
    X_test, y_test = test_data

    # Train the TSL estimator
    tg, fr = GridTensor.fit(X, y, n_iter=100, split_try=15, colsample_bytree=1.0)

    print("Fit result: ", fr)
    # TSL predictions and loss
    y_pred = tg.predict(X_test)
    tg_test_loss = np.mean((y_test - y_pred) ** 2)

    # Baseline: loss of predicting the mean of y_test
    baseline = np.mean(y_test)
    mean_test_loss = np.mean((y_test - baseline) ** 2)

    # Print losses for debugging (optional)
    print(f"Tree grid test loss: {tg_test_loss}")
    print(f"Mean test loss: {mean_test_loss}")

    print(f"Tree grid scaling: {tg.scaling}")

    assert tg_test_loss < mean_test_loss, "Tree grid should beat the mean predictor"


def test_tsl_boosted_fit(training_data, test_data):
    X, y = training_data
    X_test, y_test = test_data

    # Train the TSL estimator
    tsl, fr = TSL.fit(
        X, y, epochs=3, n_trees=37, n_iter=30, split_try=16, colsample_bytree=1.0
    )

    print("Fit result: ", fr)
    # TSL predictions and loss
    y_pred = tsl.predict(X_test)
    mpf_test_loss = np.mean((y_test - y_pred) ** 2)

    # Baseline: loss of predicting the mean of y_test
    baseline = np.mean(y_test)
    mean_test_loss = np.mean((y_test - baseline) ** 2)

    # Print losses for debugging (optional)
    print(f"TSL test loss: {mpf_test_loss}")
    print(f"Mean test loss: {mean_test_loss}")

    assert mpf_test_loss < mean_test_loss, "TSL should beat the mean predictor"


def test_tsl_predict_on_sampled_indices(training_data, test_data):
    """Test that TSL.predict() works with non-contiguous arrays (random sampled indices)."""
    X, y = training_data
    X_test, y_test = test_data

    # Train the TSL estimator
    tsl, fr = TSL.fit(
        X, y, epochs=3, n_trees=37, n_iter=30, split_try=16, colsample_bytree=1.0
    )

    # Get predictions on full test set (baseline)
    y_pred_full = tsl.predict(X_test)

    # Randomly sample indices to create non-contiguous array
    np.random.seed(42)
    n_samples = X_test.shape[0]
    sampled_indices = np.random.choice(n_samples, size=n_samples // 2, replace=False)
    sampled_indices = np.sort(sampled_indices)  # Sort to maintain some order

    # Predict on sampled indices (this creates a non-contiguous view)
    X_test_sampled = X_test[sampled_indices]
    y_test_sampled = y_test[sampled_indices]
    y_pred_sampled = tsl.predict(X_test_sampled)

    # Verify predictions match the corresponding indices from full prediction
    y_pred_expected = y_pred_full[sampled_indices]

    # Check that predictions are identical
    np.testing.assert_array_almost_equal(
        y_pred_sampled,
        y_pred_expected,
        decimal=10,
        err_msg="Predictions on sampled indices should match full predictions",
    )

    # Also test with transposed array (another non-contiguous case)
    # Transpose and then transpose back to get a non-contiguous view
    X_test_T = X_test.T
    X_test_T_back = X_test_T.T
    y_pred_transposed = tsl.predict(X_test_T_back)

    # Verify predictions are still correct
    np.testing.assert_array_almost_equal(
        y_pred_transposed,
        y_pred_full,
        decimal=10,
        err_msg="Predictions on transposed array should match full predictions",
    )

    print(
        "✅ Successfully tested predict on non-contiguous arrays (sampled indices and transposed)"
    )
