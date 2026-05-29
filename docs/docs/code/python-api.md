# Python API

`tsl-py/` wraps the Rust core for Python via PyO3/maturin. The package exposes:

```python
from tsl_py import TSL, GridTensor, StagePredictor, FitResult, TSLRegressor
```

- **`TSLRegressor`** — the scikit-learn estimator and **main entry point** for most users.
- **`TSL`** — the raw PyO3 binding to the boosted model, with the interpretation methods.
- **`GridTensor`**, **`StagePredictor`**, **`FitResult`** — the lower-level pieces.

Diagnostics/plotting live in `tsl_py.plot`, documented on the [Plotting reference](plotting.md)
page.

!!! note "Array contract"
    PyO3 methods expect **C-contiguous `float64`** arrays (wrap with
    `np.ascontiguousarray(...)`). `TSLRegressor` handles this for you.

---

## <span class="api-tag api-tag-class">class</span> `TSLRegressor` { #tslregressor }

A scikit-learn–compatible regressor wrapping [`TSL`](#tsl). Constructed with flat
hyperparameters; no fitting happens until [`fit`](#tslregressor-fit). See the
[Hyperparameters](../guides/hyperparameters.md) reference for tuning guidance.

```python
TSLRegressor(epochs=10, n_trees=10, n_iter=10, decay=1.0, split_try=10,
             colsample_bytree=0.8, alpha=0.0, complexity_penalty=0.0,
             min_split_loss=0.0, min_interval_samples=1, refinement_strategy="l2",
             prior_sample_size=0.0, update_clamp=float("inf"), tilt_tau=0.01,
             tilt_rho=0.0, split_strategy="random", top_k=10, must_fill_all_k=True,
             similarity_threshold=0.0, bagged=False, seed=42, verbosity=1, visualdb=None)
```

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `epochs` | `int` | `10` | number of boosting rounds (stages) |
| `n_trees` | `int` | `10` | bagged grid tensors per stage |
| `n_iter` | `int` | `10` | split budget per grid |
| `decay` | `float` | `1.0` | multiply `n_iter` by this after epoch 1 |
| `split_try` | `int` | `10` | candidate split positions per (feature, interval) |
| `colsample_bytree` | `float` | `0.8` | fraction of features sampled per split |
| `alpha` | `float` | `0.0` | ridge regularization on the bin update |
| `complexity_penalty` | `float` | `0.0` | penalty discouraging extra splits |
| `min_split_loss` | `float` | `0.0` | minimum error reduction to accept a split |
| `min_interval_samples` | `int` | `1` | minimum observations either side of a split |
| `refinement_strategy` | `str` | `"l2"` | `"l2"` or `"huber"` |
| `prior_sample_size` | `float` | `0.0` | parent-anchoring strength (advanced; `0.0` = off) |
| `update_clamp` | `float` | `inf` | update-magnitude cap (advanced; `inf` = off) |
| `tilt_tau` | `float` | `0.01` | $\ell_2$ coupling between $u_+$ and $u_-$ |
| `tilt_rho` | `float` | `0.0` | $\ell_1$ coupling on $(u_+ - u_-)$ |
| `split_strategy` | `str` | `"random"` | `"random"`, `"best_split"`, or `"top_k"` |
| `top_k` | `int` | `10` | (for `top_k`) candidate pool size |
| `must_fill_all_k` | `bool` | `True` | (for `top_k`) require all $k$ slots |
| `similarity_threshold` | `float` | `0.0` | bag trim $\xi$ (`0` keeps all) |
| `bagged` | `bool` | `False` | enable the bagged-aggregation path |
| `seed` | `int` | `42` | RNG seed (fits are deterministic) |
| `verbosity` | `int` | `1` | log verbosity |
| `visualdb` | `str \| None` | `None` | evo-logging SQLite path |

### <span class="api-tag api-tag-method">method</span> `fit` { #tslregressor-fit }

```python
TSLRegressor.fit(X, y) -> TSLRegressor
```

Fit the model. Delegates to [`TSL.fit`](#tsl-fit), storing the fitted core model in
`core_estimator_` and training diagnostics in `fit_result_`.

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `X` | `ndarray (n_samples, n_features)` | _required_ | training features (array or DataFrame) |
| `y` | `ndarray (n_samples,)` | _required_ | training targets |

**Returns**

| Type | Description |
|------|-------------|
| `TSLRegressor` | `self`, fitted (scikit-learn convention) |

### <span class="api-tag api-tag-method">method</span> `predict` { #tslregressor-predict }

```python
TSLRegressor.predict(X) -> np.ndarray
```

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `X` | `ndarray (n_samples, n_features)` | _required_ | features to predict |

**Returns**

| Type | Description |
|------|-------------|
| `ndarray (n_samples,)` | predictions |

### <span class="api-tag api-tag-method">method</span> `score` { #tslregressor-score }

```python
TSLRegressor.score(X, y) -> float
```

The coefficient of determination $R^2$ (via scikit-learn's `r2_score`).

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `X` | `ndarray (n_samples, n_features)` | _required_ | features |
| `y` | `ndarray (n_samples,)` | _required_ | true targets |

**Returns**

| Type | Description |
|------|-------------|
| `float` | $R^2$ of `predict(X)` against `y` |

### <span class="api-tag api-tag-method">method</span> `save` { #tslregressor-save }

```python
TSLRegressor.save(path) -> None
```

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `path` | `str` | _required_ | destination file (binary) |

### <span class="api-tag api-tag-classmethod">classmethod</span> `load` { #tslregressor-load }

```python
TSLRegressor.load(path) -> TSLRegressor
```

Load a model saved with [`save`](#tslregressor-save); also reads the legacy MPF `.bin` format.

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `path` | `str` | _required_ | model file |

**Returns**

| Type | Description |
|------|-------------|
| `TSLRegressor` | a fitted estimator |

### <span class="api-tag api-tag-property">property</span> `stage_predictors` { #tslregressor-stage-predictors }

The list of fitted [`StagePredictor`](#stagepredictor) objects. **Type:** `list[StagePredictor]`.

---

## <span class="api-tag api-tag-class">class</span> `TSL` { #tsl }

The core boosted model. Use it directly for the interpretation methods below; otherwise
prefer [`TSLRegressor`](#tslregressor).

### <span class="api-tag api-tag-classmethod">classmethod</span> `fit` { #tsl-fit }

```python
TSL.fit(x, y, epochs, decay, n_trees, n_iter, split_try, colsample_bytree, alpha,
        complexity_penalty, min_split_loss, min_interval_samples, refinement_strategy,
        prior_sample_size, update_clamp, tilt_tau, tilt_rho, split_strategy, top_k,
        must_fill_all_k, similarity_threshold, bagged, seed, verbosity, visualdb=None)
        -> tuple[TSL, FitResult]
```

Fit a boosted TSL model.

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `ndarray (n_samples, n_features)` | _required_ | training features (C-contiguous `float64`) |
| `y` | `ndarray (n_samples,)` | _required_ | training targets |
| _hyperparameters_ | — | — | same names/types as the [`TSLRegressor` constructor](#tslregressor) |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[TSL, FitResult]` | the fitted model and its training diagnostics |

### <span class="api-tag api-tag-method">method</span> `predict` { #tsl-predict }

```python
TSL.predict(x) -> np.ndarray
```

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `ndarray (n_samples, n_features)` | _required_ | features (C-contiguous `float64`) |

**Returns**

| Type | Description |
|------|-------------|
| `ndarray (n_samples,)` | sum of all stage predictions |

### <span class="api-tag api-tag-method">method</span> `save` { #tsl-save }

```python
TSL.save(path) -> None
```

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `path` | `str` | _required_ | destination binary file |

### <span class="api-tag api-tag-classmethod">classmethod</span> `load` { #tsl-load }

```python
TSL.load(path) -> TSL
```

Reads the native binary format and the legacy MPF `.bin` format.

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `path` | `str` | _required_ | model file |

**Returns** — `TSL`.

### <span class="api-tag api-tag-method">method</span> `compute_partial_dependence_function` { #tsl-pd }

```python
TSL.compute_partial_dependence_function(fixed_indices, fixed_values, data_x)
    -> tuple[list, np.ndarray]
```

Model-native partial dependence (see [Partial dependence](../math/partial-dependence.md)),
marginalizing over the **empirical joint** of the non-fixed features.

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `fixed_indices` | `list[int]` | _required_ | feature column(s) held fixed |
| `fixed_values` | `ndarray (n_points, len(fixed_indices))` | _required_ | values to evaluate at |
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | background data |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[list, ndarray]` | per-stage $(C_+, C_-)$ constants, and branch curves $[f_+^{(0)}, f_-^{(0)}, f_+^{(1)}, \dots]$ |

### <span class="api-tag api-tag-method">method</span> `compute_first_order_partial_dependence_functions` { #tsl-pd1 }

```python
TSL.compute_first_order_partial_dependence_functions(values_x, data_x) -> list
```

1D partial dependence for **every** feature at once.

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `values_x` | `ndarray (grid_points, n_features)` | _required_ | evaluation grid (column $j$ supplies $x_j$) |
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | background data |

**Returns**

| Type | Description |
|------|-------------|
| `list` | one `(constants_per_stage, pd_values)` entry per feature |

### <span class="api-tag api-tag-method">method</span> `compute_ice_curves` { #tsl-ice }

```python
TSL.compute_ice_curves(observations, feature_index, x_range, data_x) -> np.ndarray
```

Individual Conditional Expectation curves: sweep one feature for each given observation.

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `observations` | `ndarray (n_obs, n_features)` | _required_ | rows to trace |
| `feature_index` | `int` | _required_ | feature to vary |
| `x_range` | `ndarray (n_range,)` | _required_ | values to sweep |
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | background data |

**Returns**

| Type | Description |
|------|-------------|
| `ndarray (n_obs, n_range, 2·n_stages)` | ICE curves, scaled by `scaling_plus`/`scaling_minus` |

### <span class="api-tag api-tag-method">method</span> `compute_per_stage_feature_importance` { #tsl-fi-stage }

```python
TSL.compute_per_stage_feature_importance(data_x) -> tuple[np.ndarray, np.ndarray]
```

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | data to evaluate over |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[ndarray, ndarray]` | `(backbone, tilt)` importance, each `(n_stages, n_features)` — $\mathrm{Var}[\log b_j]$ and $\mathrm{Var}[d_j]$ |

### <span class="api-tag api-tag-method">method</span> `compute_aggregated_feature_importance` { #tsl-fi-agg }

```python
TSL.compute_aggregated_feature_importance(data_x)
    -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Energy-weighted roll-up of the per-stage importances.

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | data to evaluate over |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[ndarray, ndarray, ndarray]` | `(global_backbone, global_tilt, stage_weights)`, each 1D |

### <span class="api-tag api-tag-method">method</span> `compute_combined_feature_importance` { #tsl-fi-combined }

```python
TSL.compute_combined_feature_importance(data_x, gamma=1.0)
    -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

A single combined importance $I_j = I_j^b + \gamma\, I_j^d$ per feature.

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `data_x` | `ndarray (n_samples, n_features)` | _required_ | data to evaluate over |
| `gamma` | `float` | `1.0` | weight on the tilt component |

**Returns**

| Type | Description |
|------|-------------|
| `tuple[ndarray, ndarray, ndarray]` | `(combined, backbone, tilt)`, each 1D over features |

### <span class="api-tag api-tag-property">property</span> `stage_predictors` { #tsl-stage-predictors }

The stages of the model. **Type:** `list[StagePredictor]`.

---

## <span class="api-tag api-tag-class">class</span> `GridTensor` { #gridtensor }

One fitted separable component (internals: [GridTensor](grid-tensor.md)).

### <span class="api-tag api-tag-classmethod">classmethod</span> `fit` { #gridtensor-fit }

```python
GridTensor.fit(x, y, n_iter, split_try, colsample_bytree,
               complexity_penalty=0.0, seed=42) -> tuple[GridTensor, FitResult]
```

Fit a single grid tensor (no boosting, no bagging).

**Parameters**

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `ndarray (n_samples, n_features)` | _required_ | training features |
| `y` | `ndarray (n_samples,)` | _required_ | training targets |
| `n_iter` | `int` | _required_ | split budget |
| `split_try` | `int` | _required_ | candidate split positions |
| `colsample_bytree` | `float` | _required_ | fraction of features per split |
| `complexity_penalty` | `float` | `0.0` | penalty discouraging extra splits |
| `seed` | `int` | `42` | RNG seed |

**Returns** — `tuple[GridTensor, FitResult]`.

### <span class="api-tag api-tag-method">method</span> `predict` { #gridtensor-predict }

```python
GridTensor.predict(x) -> np.ndarray
```

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `ndarray (n_samples, n_features)` | _required_ | features |

**Returns** — `ndarray (n_samples,)`, this component's prediction.

### <span class="api-tag api-tag-attr">attributes</span> Attributes { #gridtensor-attributes }

| Name | Type | Description |
|------|------|-------------|
| `splits` | `list[list[float]]` | split thresholds per axis |
| `intervals` | `list` | interval count/bounds per axis |
| `backbone_values` | `list[list[float]]` | $b_j \ge 0$ per interval per axis |
| `tilt_values` | `list[list[float]]` | $d_j \in \mathbb{R}$ per interval per axis |
| `lambda_plus`, `lambda_minus` | `float` | non-negative branch scalars |
| `mean_factor` / `grid_values` | `list` | per-interval $\prod_j b_j e^{d_j}$ |
| `scaling` | `float` | legacy; ignored in two-tensor mode |

---

## <span class="api-tag api-tag-class">class</span> `StagePredictor` { #stagepredictor }

One boosting stage ([details](stage-predictor.md)).

### <span class="api-tag api-tag-method">method</span> `predict` { #stagepredictor-predict }

```python
StagePredictor.predict(x) -> np.ndarray
```

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `x` | `ndarray (n_samples, n_features)` | _required_ | features |

**Returns** — `ndarray (n_samples,)`, the stage's contribution **with OLS scaling applied**.

### <span class="api-tag api-tag-attr">attributes</span> Attributes { #stagepredictor-attributes }

| Name | Type | Description |
|------|------|-------------|
| `grid_tensors` | `list[GridTensor]` | the bag of fitted components |
| `combined_grid_tensor` | `GridTensor` | the aggregated primary grid |
| `candidate_indices` | `list[int] \| None` | bags kept after similarity filtering |
| `scaling_plus`, `scaling_minus` | `float \| None` | OLS coefficients for $f_+$ and $-f_-$ |

---

## <span class="api-tag api-tag-class">class</span> `FitResult` { #fitresult }

Training diagnostics, returned alongside a fitted model.

### <span class="api-tag api-tag-attr">attributes</span> Attributes { #fitresult-attributes }

| Name | Type | Description |
|------|------|-------------|
| `err` | `float` | final training loss |
| `residuals` | `ndarray (n_samples,)` | training residuals |
| `y_hat` | `ndarray (n_samples,)` | training predictions |

---

## Build & install

`tsl-py` is an `extension-module` cdylib built with maturin. Because maturin cannot
auto-detect a Python 3.14, build against a 3.13 venv with `VIRTUAL_ENV` set:

```sh
# from tsl-py/
VIRTUAL_ENV=/Users/jin/Documents/TSL/.venv \
  /Users/jin/Documents/TSL/.venv/bin/maturin develop
```

Tests run through Python (`python -m pytest python/tests/`). See
[Getting started](../guides/getting-started.md).
