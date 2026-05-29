# Python API

`tsl-py/` wraps the Rust core for Python via PyO3/maturin. The package exposes:

```python
from tsl_py import TSL, GridTensor, StagePredictor, FitResult, TSLRegressor
```

- **`TSLRegressor`** (`python/tsl_py/sklearn.py`) — the scikit-learn estimator and **main
  entry point** for most users.
- **`TSL`** — the raw PyO3 binding to the boosted model, with the interpretation methods.
- **`GridTensor`**, **`StagePredictor`**, **`FitResult`** — the lower-level pieces.

Diagnostics/plotting live in `tsl_py.plot`, documented on the separate
[Plotting reference](plotting.md) page. This page documents one section per user-callable
function on the model objects.

!!! note "Array contract"
    The PyO3 methods expect **C-contiguous `float64`** arrays. Wrap inputs with
    `np.ascontiguousarray(...)` if in doubt. `TSLRegressor` handles this for you.

---

## `TSLRegressor`

A standard scikit-learn regressor wrapping `TSL`. Construct it with flat hyperparameters,
then `fit`/`predict`/`score` like any estimator.

### `TSLRegressor(...)` — constructor

```python
TSLRegressor(
    epochs=10, n_trees=10, n_iter=10, decay=1.0,
    split_try=10, colsample_bytree=0.8,
    alpha=0.0, complexity_penalty=0.0, min_split_loss=0.0, min_interval_samples=1,
    refinement_strategy="l2", prior_sample_size=0.0, update_clamp=float("inf"),
    tilt_tau=0.01, tilt_rho=0.0,
    split_strategy="random", top_k=10, must_fill_all_k=True,
    similarity_threshold=0.0, bagged=False,
    seed=42, verbosity=1, visualdb=None,
)
```

Stores the hyperparameters; no fitting happens until `fit`. Every argument is described in
the [Hyperparameters](../guides/hyperparameters.md) reference.

### `fit(X, y)`

Fit the model. Delegates to `TSL.fit`, storing the fitted core model in `core_estimator_`
and the training diagnostics in `fit_result_`.

- **`X`** — `(n_samples, n_features)` array (or DataFrame).
- **`y`** — `(n_samples,)` target array.
- **Returns** `self` (scikit-learn convention).

```python
model = TSLRegressor(epochs=5, n_trees=16, n_iter=30, seed=0).fit(X_train, y_train)
```

### `predict(X)`

Predict targets for `X`. Requires a fitted model.

- **`X`** — `(n_samples, n_features)`.
- **Returns** `(n_samples,)` array of predictions.

### `score(X, y)`

The coefficient of determination $R^2$ of the prediction (via scikit-learn's `r2_score`).

- **Returns** `float` — $R^2$ of `self.predict(X)` against `y`.

### `stage_predictors` (property)

The list of fitted [`StagePredictor`](#stagepredictor) objects from the underlying `TSL`.
Requires a fitted model.

### `save(path)`

Serialize the fitted core estimator to `path` (binary).

### `TSLRegressor.load(path)` (classmethod)

Load a model saved with `save`, returning a fitted `TSLRegressor`. Also reads the legacy
MPF `.bin` format.

---

## `TSL`

The core boosted model. Use it directly when you want the interpretation methods below;
otherwise prefer `TSLRegressor`.

### `TSL.fit(x, y, ...)` (classmethod)

Fit a boosted TSL model. Takes the same flat hyperparameters as `TSLRegressor` (plus the
data) and maps them onto the Rust builders.

```python
model, fit_result = TSL.fit(
    x, y,                         # C-contiguous float64
    epochs=5, decay=1.0, n_trees=16, n_iter=30, split_try=16,
    colsample_bytree=0.8, alpha=0.0, complexity_penalty=0.0,
    min_split_loss=0.0, min_interval_samples=1,
    refinement_strategy="l2", prior_sample_size=0.0, update_clamp=float("inf"),
    tilt_tau=0.01, tilt_rho=0.0,
    split_strategy="random", top_k=10, must_fill_all_k=True,
    similarity_threshold=0.0, bagged=False, seed=0, verbosity=1, visualdb=None,
)
```

- **Returns** `(TSL, FitResult)`.

### `predict(x)`

- **`x`** — `(n_samples, n_features)` C-contiguous `float64`.
- **Returns** `(n_samples,)` array (sum of all stage predictions).

### `save(path)`

Serialize the model to a binary file at `path`.

### `TSL.load(path)` (classmethod)

Load a model from `path`. Reads the native binary format and the legacy MPF `.bin` format.

### `compute_partial_dependence_function(fixed_indices, fixed_values, data_x)`

The model-native partial dependence (see
[Partial dependence](../math/partial-dependence.md)). Marginalizes over the **empirical
joint** of the non-fixed features in `data_x`.

- **`fixed_indices`** — `list[int]`, the feature column(s) held fixed.
- **`fixed_values`** — `(n_points, len(fixed_indices))` array of values to evaluate at.
- **`data_x`** — `(n_samples, n_features)` background data to marginalize over.
- **Returns** a tuple `(constants, pd_values)`: per stage the $(C_+, C_-)$ normalizing
  constants, and an array of the branch curves
  $[f_+^{(0)}, f_-^{(0)}, f_+^{(1)}, f_-^{(1)}, \dots]$ (one $(+)/(-)$ pair per stage).

### `compute_first_order_partial_dependence_functions(values_x, data_x)`

Convenience wrapper computing the 1D partial dependence for **every** feature at once.

- **`values_x`** — `(grid_points, n_features)` evaluation grid (column $j$ supplies the
  $x_j$ values for feature $j$).
- **`data_x`** — background data.
- **Returns** a `list` with one entry per feature, each `(constants_per_stage, pd_values)`
  as in `compute_partial_dependence_function`.

### `compute_ice_curves(observations, feature_index, x_range, data_x)`

Individual Conditional Expectation curves: sweep one feature over a range for each given
observation.

- **`observations`** — `(n_obs, n_features)` rows to trace.
- **`feature_index`** — `int`, the feature to vary.
- **`x_range`** — `(n_range,)` values to sweep.
- **`data_x`** — background data.
- **Returns** a `(n_obs, n_range, 2 * n_stages)` array, scaled by
  `scaling_plus`/`scaling_minus`.

### `compute_per_stage_feature_importance(data_x)`

Per-stage importance of each feature, split into backbone and tilt.

- **Returns** `(backbone_importance, tilt_importance)`, each a `(n_stages, n_features)`
  array — $\mathrm{Var}[\log b_j]$ and $\mathrm{Var}[d_j]$ per stage.

### `compute_aggregated_feature_importance(data_x)`

Roll the per-stage importances up to global, energy-weighted scores.

- **Returns** `(global_backbone, global_tilt, stage_weights)`, each a 1D array
  (`global_*` over features, `stage_weights` over stages).

### `compute_combined_feature_importance(data_x, gamma=1.0)`

A single combined importance $I_j = I_j^b + \gamma\, I_j^d$ per feature.

- **`gamma`** — weight on the tilt component (default `1.0`).
- **Returns** `(combined, backbone, tilt)`, each a 1D array over features.

### `stage_predictors` (property)

The list of [`StagePredictor`](#stagepredictor) objects making up the model.

---

## `GridTensor`

One fitted separable component (see [GridTensor](grid-tensor.md) for the internals).

### `GridTensor.fit(x, y, n_iter, split_try, colsample_bytree, complexity_penalty=0.0, seed=42)` (classmethod)

Fit a single grid tensor (no boosting, no bagging).

- **Returns** `(GridTensor, FitResult)`.

### `predict(x)`

- **`x`** — `(n_samples, n_features)`.
- **Returns** `(n_samples,)` predictions for this component.

### Attributes

Read-only views of the fitted two-tensor: `splits`, `intervals`, `backbone_values`
($b_j$ per interval per axis), `tilt_values` ($d_j$), `lambda_plus`, `lambda_minus`,
`mean_factor` / `grid_values` (the per-interval $\prod_j b_j e^{d_j}$), and the legacy
`scaling` (ignored in two-tensor mode).

---

## `StagePredictor`

One boosting stage (see [StagePredictor](stage-predictor.md)).

### `predict(x)`

- **Returns** the stage's contribution `(n_samples,)`, with OLS scaling applied.

### Attributes

`grid_tensors` (the bag), `combined_grid_tensor` (the aggregated primary grid),
`candidate_indices` (kept after similarity filtering), `scaling_plus`, `scaling_minus`.

---

## `FitResult`

Returned alongside a fitted model. Read-only attributes:

- **`err`** — final training loss.
- **`residuals`** — `(n_samples,)` training residuals.
- **`y_hat`** — `(n_samples,)` training predictions.

---

## Build & install

`tsl-py` is an `extension-module` cdylib built with maturin. Because maturin cannot
auto-detect a Python 3.14, build against a 3.13 venv with `VIRTUAL_ENV` set:

```sh
# from tsl-py/
VIRTUAL_ENV=/Users/jin/Documents/TSL/.venv \
  /Users/jin/Documents/TSL/.venv/bin/maturin develop
```

Tests run through Python (`python -m pytest python/tests/`); the cdylib can't link
libpython for `cargo test` on Linux. See [Getting started](../guides/getting-started.md).
