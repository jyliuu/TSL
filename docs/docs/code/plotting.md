# Plotting (`tsl_py.plot`)

`tsl_py.plot` holds the diagnostic plots. It is **lazy-imported** (it needs `matplotlib`,
installed via the `[plots]` extra) so importing `tsl_py` stays light.

```python
import tsl_py.plot as tplot
```

!!! tip "Figure **and** data"
    Every plotting helper returns a **result object carrying the raw numerical arrays** in
    addition to drawing the figure, so you can re-style, export, or rebuild a custom
    visualization. The result dataclasses (`PDDifferenceResult`, `Backbone2DResult`, …) are
    exported from `tsl_py.plot`.

Common arguments: `model` (a fitted `TSL`/`TSLRegressor.core_estimator_`), `X` (background
data to marginalize over), `features` / `feature_x` / `feature_y` (int indices or names),
`feature_names`, `stages` (which stages to draw), `grid_points`, and `figsize`.

---

## Partial dependence & ICE

### `plot_first_order_pd(model, X, features=None, ...)`

First-order partial dependence — the $f_+$ and $f_-$ branch curves — per stage for the
selected features (one row per stage, one column per feature). Key kwargs: `grid_points=200`,
`stages`, `pd_scale="raw"`, `show_data_density=False`. **Returns** `PDDifferenceResult`.

### `pd_difference_plot(model, X, features=None, ...)`

The signed PD difference $\mathrm{PD}_+ - \mathrm{PD}_-$ with the
$\sqrt{C_+ C_-}\, b_j$ **backbone overlay** (dotted). Extra kwargs: `show_backbone_overlay=True`,
`show_global=False`. **Returns** `PDDifferenceResult`. This is the workhorse 1D
interpretation plot.

### `plot_2d_pd(model, X, feature_x, feature_y, kind="surface", ...)`

Two-feature partial dependence as a `kind="surface"` or `kind="lines"` plot, per stage
(`grid_points=50`, `y_values`, `stages`, `cmap`). **Returns** `PD2DResult` (surface) or
`PD2DLinesResult` (lines).

<figure markdown="span">
  ![Hour × working-day 2D partial dependence](../assets/img/pd_hour_workingday_tsl.png){ width="75%" }
  <figcaption><code>plot_2d_pd(..., kind="lines")</code> on bike-sharing: demand-by-hour conditioned on working day.</figcaption>
</figure>

### `plot_ice(model, X, feature, ...)`

Individual Conditional Expectation curves for one feature (`n_ice=50`, `grid_points=100`,
`seed=0`, `ax`, `figsize=(7, 4)`). **Returns** `ICEResult`.

---

## Backbone & tilt

### `plot_2d_backbone(model, X, feature_x, feature_y, ...)`

The 2D backbone product $b_x\cdot b_y$ and the 2D PD per stage — the generic "spatial
backbone" plot. Returns the meshgrid and per-stage arrays so callers can overlay e.g.
cartopy. Kwargs: `stages`, `grid_points=100`, `cmap_backbone`, `cmap_pd`,
`return_data_only=False`. **Returns** `Backbone2DResult`.

<figure markdown="span">
  ![2D spatial backbone and PD per stage](../assets/img/california_spatial_backbone.png){ width="100%" }
  <figcaption><code>plot_2d_backbone</code> on California latitude × longitude, with a cartopy basemap overlaid by the example script.</figcaption>
</figure>

### `plot_tilt_1d(model, X, features=None, ...)`

The per-feature, per-stage tilt $d_j(x_j)$ as step curves (layout mirrors
`plot_first_order_pd`), with a zero reference line. **Returns** `Tilt1DResult`.

### `plot_2d_tilt(model, X, feature_x, feature_y, ...)`

The 2D tilt product $d_x(x)\cdot d_y(y)$ per stage (`return_data_only=False`). **Returns**
`Tilt2DResult`.

### `plot_tilt_diagnostics(model, X, features=None, ...)`

Exploratory tilt diagnostics — four curves per `(stage, feature)` cell (pure vs.
density-weighted tilt). **Returns** `TiltDiagnosticsResult`.

---

## Feature importance

### `plot_feature_importance(model, X, feature_names=None, gamma=1.0, figsize=(14, 10))`

A six-panel summary: per-stage backbone and tilt importance (heatmaps), global backbone and
tilt importance (bars), the combined $I_j = I_j^b + \gamma\, I_j^d$ (bar), and energy-based
stage weights (bar). **Returns** `FeatureImportanceResult`.

<figure markdown="span">
  ![Feature importance panels](../assets/img/california_feature_importance.png){ width="100%" }
  <figcaption><code>plot_feature_importance</code> on California housing.</figcaption>
</figure>

---

## Local (per-observation) interpretation

### `compute_local_explanation(model, x)`

Per-stage decomposition of a single prediction `x`: the $f_+/f_-$ contributions, per-feature
backbone/tilt values, and the intercept $(b_0, d_0)$ absorbing the OLS scaling. With the
intercept as axis $j=0$, every stage satisfies
$m^{(\ell)}(\mathbf{x}) = 2\,b^{(\ell)}(\mathbf{x})\sinh(d^{(\ell)}(\mathbf{x}))$. **Returns**
`LocalExplanation`.

### `plot_local_interpretation(explanations, points, titles, feature_names, save_path, ...)`

The three-column "Backbone × Tilt" local-interpretation plot — one column per point, rows =
stages sorted by absolute net contribution (`top_k_features=3`). Takes a list of
`LocalExplanation`s (from `compute_local_explanation`). **Returns** the figure.

<figure markdown="span">
  ![Local explanation — coastal](../assets/img/california_local_interp_coastal.png){ width="49%" }
  ![Local explanation — desert](../assets/img/california_local_interp_desert.png){ width="49%" }
  <figcaption><code>plot_local_interpretation</code> for a coastal vs. an inland point.</figcaption>
</figure>

---

## Component plots

### `plot_grid_tensor_components(grid_tensor, individual_plots=False, axis=None)`

Plot a single `GridTensor`'s backbone/tilt component curves.

### `plot_combined_grid_tensors(model, individual_plots=True, axis=None)`

Overlay the combined grid-tensor components across a model's stages.

### `plot_epoch_components(model, epoch)`

Plot the per-feature components for one stage/epoch.

---

## Result dataclasses

Each plotting function returns a small dataclass exposing the underlying arrays:
`PDDifferenceResult`, `NormalizedDiagnostics`, `PD2DResult`, `PD2DLinesResult`, `ICEResult`,
`Backbone2DResult`, `Tilt1DResult`, `Tilt2DResult`, `TiltDiagnosticsResult`,
`LocalExplanation`, `FeatureImportanceResult`. Use these to export the numbers or build a
bespoke figure without recomputing the partial dependence.
