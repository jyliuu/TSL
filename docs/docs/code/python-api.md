# Python API

`tsl-py/` wraps the Rust core for Python via PyO3/maturin. The package exposes the four core
types and a scikit-learn-compatible regressor:

```python
from tsl_py import TSL, GridTensor, StagePredictor, FitResult, TSLRegressor
```

`TSLRegressor` (in `python/tsl_py/sklearn.py`) is the **main user entry point**; the
lower-level `TSL` is the raw PyO3 binding.

## `TSLRegressor` (scikit-learn API)

A standard estimator with `fit` / `predict` / `score` (R²) and `save` / `load`. The
constructor takes flat hyperparameters (defaults shown):

```python
TSLRegressor(
    epochs=10, n_trees=10, n_iter=10, decay=1.0,
    split_try=10, colsample_bytree=0.8,
    alpha=0.0, complexity_penalty=0.0, min_split_loss=0.0, min_interval_samples=1,
    refinement_strategy="l2",                 # "l2" | "huber"
    prior_sample_size=0.0, update_clamp=float("inf"),
    tilt_tau=0.01, tilt_rho=0.0,
    split_strategy="random",                  # "random" | "best_split" | "top_k"
    top_k=10, must_fill_all_k=True,
    similarity_threshold=0.0,                  # ξ trim threshold
    bagged=False,
    seed=42, verbosity=1, visualdb=None,       # visualdb → evo-logging SQLite path
)
```

`fit(X, y)` stores the fitted core estimator (`core_estimator_`) and the `FitResult`
(`fit_result_`); `stage_predictors` exposes the fitted stages. Every parameter is described
in the [Hyperparameters](../guides/hyperparameters.md) reference.

## The PyO3 types (`tsl-py/src/lib.rs`)

### `TSL`

- **`TSL.fit(x, y, ...)`** — a classmethod taking the flat hyperparameters above and mapping
  them onto the Rust builders; returns `(TSL, FitResult)`.
- **`predict(x)`** → array of predictions.
- **`save(path)` / `TSL.load(path)`** — binary serialization; `load` also reads the legacy
  MPF `.bin` format.
- **`stage_predictors`** — the list of `StagePredictor` objects.

Interpretation methods (all marginalize over the **empirical joint**, not assuming feature
independence — see [Partial dependence](../math/partial-dependence.md)):

| Method | Returns |
|--------|---------|
| `compute_partial_dependence_function(fixed_indices, fixed_values, data_x)` | per-stage $(C_+, C_-)$ constants + PD values |
| `compute_first_order_partial_dependence_functions(values_x, data_x)` | one PD entry per feature |
| `compute_ice_curves(observations, feature_index, x_range, data_x)` | ICE curves, scaled by `scaling_plus`/`scaling_minus` |
| `compute_per_stage_feature_importance(data_x)` | $(\mathrm{Var}[\log b_j],\ \mathrm{Var}[d_j])$ per stage |
| `compute_aggregated_feature_importance(data_x)` | global backbone/tilt importance + stage weights |
| `compute_combined_feature_importance(data_x, gamma=1.0)` | combined importance score |

### `StagePredictor`, `GridTensor`, `FitResult`

- **`StagePredictor`** — `grid_tensors`, `combined_grid_tensor`, `candidate_indices`,
  `scaling_plus`, `scaling_minus`; `predict(x)`.
- **`GridTensor`** — `splits`, `intervals`, `backbone_values`, `tilt_values`,
  `lambda_plus`, `lambda_minus`, `mean_factor`; `GridTensor.fit(x, y, ...)` and `predict(x)`.
- **`FitResult`** — read-only `err`, `residuals`, `y_hat`.

## Plotting (`tsl_py.plot`)

`tsl_py.plot` is lazy-imported (it needs `matplotlib`, installed via the `[plots]` extra).
Every function returns **both** a figure/axes and the raw numerical arrays, so callers can
re-style or save as they like. The public surface:

- **Partial dependence / ICE** — `plot_first_order_pd`, `pd_difference_plot`, `plot_2d_pd`,
  `plot_ice` (with result types `PDDifferenceResult`, `PD2DResult`, `PD2DLinesResult`,
  `ICEResult`, `NormalizedDiagnostics`).
- **Backbone** — `plot_2d_backbone` (`Backbone2DResult`).
- **Tilt** — `plot_tilt_1d`, `plot_2d_tilt`, `plot_tilt_diagnostics` (`Tilt1DResult`,
  `Tilt2DResult`, `TiltDiagnosticsResult`).
- **Local interpretation** — `compute_local_explanation` (`LocalExplanation`) and
  `plot_local_interpretation` (intercept-absorbed backbone + tilt waterfall per observation).
- **Feature importance** — `plot_feature_importance` (`FeatureImportanceResult`).
- **Component plots** — `plot_grid_tensor_components`, `plot_combined_grid_tensors`,
  `plot_epoch_components`.

## Build & install

`tsl-py` is an `extension-module` cdylib built with maturin. Because maturin cannot
auto-detect the machine's Python 3.14, build against the project's 3.13 venv with
`VIRTUAL_ENV` set:

```sh
# from tsl-py/
VIRTUAL_ENV=/Users/jin/Documents/TSL/.venv \
  /Users/jin/Documents/TSL/.venv/bin/maturin develop
```

Tests run through Python (the cdylib can't link libpython for `cargo test` on Linux):
`python -m pytest python/tests/`. See [Getting started](../guides/getting-started.md).
