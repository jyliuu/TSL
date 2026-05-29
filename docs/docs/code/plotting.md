# Plotting (`tsl_py.plot`)

`tsl_py.plot` holds the diagnostic plots. It is **lazy-imported** (it needs `matplotlib`,
installed via the `[plots]` extra) so importing `tsl_py` stays light.

```python
import tsl_py.plot as tplot
```

!!! tip "Figure **and** data"
    Every helper returns a **result dataclass carrying the raw arrays** in addition to
    drawing the figure, so you can re-style, export, or rebuild a custom visualization.

### Common parameters

Most plotting functions share these; per-function tables below list only the distinctive
ones.

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `model` | `TSL` | _required_ | a fitted model (`TSLRegressor.core_estimator_`) |
| `X` | `ndarray (n_samples, n_features)` | _required_ | background data to marginalize over |
| `features` | `Iterable[int \| str] \| None` | `None` | features to plot (default: all) |
| `feature_x`, `feature_y` | `int \| str` | _required_ | the two features for 2D plots |
| `feature_names` | `Sequence[str] \| None` | `None` | names for labelling |
| `stages` | `Iterable[int] \| None` | `None` | which stages to draw (default: all) |
| `grid_points` | `int` | `200`/`100`/`50` | evaluation resolution |
| `figsize` | `tuple[float, float] \| None` | `None` | matplotlib figure size |

---

## Partial dependence & ICE

### <span class="api-tag api-tag-function">function</span> `plot_first_order_pd` { #plot-first-order-pd }

```python
plot_first_order_pd(model, X, features=None, feature_names=None, grid_points=200,
                    stages=None, figsize=None, pd_scale="raw",
                    show_data_density=False) -> PDDifferenceResult
```

First-order partial dependence — the $f_+$ and $f_-$ branch curves — per stage for the
selected features (one row per stage, one column per feature).

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `pd_scale` | `"raw" \| ...` | `"raw"` | scaling applied to the PD curves |
| `show_data_density` | `bool` | `False` | overlay a data-density rug |

**Returns** — `PDDifferenceResult`.

### <span class="api-tag api-tag-function">function</span> `pd_difference_plot` { #pd-difference-plot }

```python
pd_difference_plot(model, X, features=None, feature_names=None, grid_points=200,
                   stages=None, show_backbone_overlay=True, show_global=False,
                   figsize=None, pd_scale="raw", show_data_density=False)
                   -> PDDifferenceResult
```

The signed PD difference $\mathrm{PD}_+ - \mathrm{PD}_-$ with the $\sqrt{C_+ C_-}\,b_j$
**backbone overlay** (dotted). The workhorse 1D interpretation plot.

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `show_backbone_overlay` | `bool` | `True` | draw the dotted backbone overlay |
| `show_global` | `bool` | `False` | also draw the summed-over-stages curve |

**Returns** — `PDDifferenceResult`.

### <span class="api-tag api-tag-function">function</span> `plot_2d_pd` { #plot-2d-pd }

```python
plot_2d_pd(model, X, feature_x, feature_y, feature_names=None, grid_points=50,
           kind="surface", y_values=None, stages=None, cmap=None, figsize=None)
```

Two-feature partial dependence per stage.

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `kind` | `str` | `"surface"` | `"surface"` or `"lines"` |
| `y_values` | `Sequence[float] \| None` | `None` | (for `"lines"`) values of `feature_y` to slice at |
| `cmap` | `Colormap \| None` | `None` | colormap |

**Returns** — `PD2DResult` (surface) or `PD2DLinesResult` (lines).

<figure markdown="span">
  ![Hour × working-day 2D partial dependence](../assets/img/pd_hour_workingday_tsl.png){ width="75%" }
  <figcaption><code>plot_2d_pd(..., kind="lines")</code> on bike-sharing.</figcaption>
</figure>

### <span class="api-tag api-tag-function">function</span> `plot_ice` { #plot-ice }

```python
plot_ice(model, X, feature, feature_names=None, n_ice=50, grid_points=100,
         seed=0, ax=None, figsize=(7, 4)) -> ICEResult
```

Individual Conditional Expectation curves for one feature.

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `feature` | `int \| str` | _required_ | feature to vary |
| `n_ice` | `int` | `50` | number of observations sampled |
| `seed` | `int` | `0` | sampling seed |
| `ax` | `Axes \| None` | `None` | draw onto an existing axis |

**Returns** — `ICEResult`.

---

## Backbone & tilt

### <span class="api-tag api-tag-function">function</span> `plot_2d_backbone` { #plot-2d-backbone }

```python
plot_2d_backbone(model, X, feature_x, feature_y, feature_names=None, stages=None,
                 grid_points=100, cmap_backbone=None, cmap_pd=None, figsize=None,
                 return_data_only=False) -> Backbone2DResult
```

The 2D backbone product $b_x\cdot b_y$ and the 2D PD per stage — the generic "spatial
backbone" plot. Returns the meshgrid and per-stage arrays so callers can overlay e.g.
cartopy.

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `cmap_backbone`, `cmap_pd` | `Colormap \| None` | `None` | colormaps for each panel |
| `return_data_only` | `bool` | `False` | skip drawing; return arrays only |

**Returns** — `Backbone2DResult`.

<figure markdown="span">
  ![2D spatial backbone and PD per stage](../assets/img/california_spatial_backbone.png){ width="100%" }
  <figcaption><code>plot_2d_backbone</code> on California latitude × longitude (cartopy basemap added by the example).</figcaption>
</figure>

### <span class="api-tag api-tag-function">function</span> `plot_tilt_1d` { #plot-tilt-1d }

```python
plot_tilt_1d(model, X, features=None, feature_names=None, grid_points=200,
             stages=None, figsize=None, color=None) -> Tilt1DResult
```

The per-feature, per-stage tilt $d_j(x_j)$ as step curves (layout mirrors
`plot_first_order_pd`), with a zero reference line. **Returns** `Tilt1DResult`.

### <span class="api-tag api-tag-function">function</span> `plot_2d_tilt` { #plot-2d-tilt }

```python
plot_2d_tilt(model, X, feature_x, feature_y, feature_names=None, stages=None,
             grid_points=100, cmap=None, figsize=None, return_data_only=False)
             -> Tilt2DResult
```

The 2D tilt product $d_x(x)\cdot d_y(y)$ per stage. **Returns** `Tilt2DResult`.

### <span class="api-tag api-tag-function">function</span> `plot_tilt_diagnostics` { #plot-tilt-diagnostics }

```python
plot_tilt_diagnostics(model, X, features=None, feature_names=None, grid_points=200,
                      stages=None, figsize=None, pure_color=None,
                      weighted_color=None) -> TiltDiagnosticsResult
```

Exploratory tilt diagnostics — four curves per `(stage, feature)` cell (pure vs.
density-weighted tilt). **Returns** `TiltDiagnosticsResult`.

---

## Feature importance

### <span class="api-tag api-tag-function">function</span> `plot_feature_importance` { #plot-feature-importance }

```python
plot_feature_importance(model, X, feature_names=None, gamma=1.0,
                        figsize=(14, 10)) -> FeatureImportanceResult
```

A six-panel summary: per-stage backbone and tilt importance (heatmaps), global backbone and
tilt importance (bars), the combined $I_j = I_j^b + \gamma\, I_j^d$ (bar), and energy-based
stage weights (bar).

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `gamma` | `float` | `1.0` | weight on the tilt component in the combined score |

**Returns** — `FeatureImportanceResult`.

<figure markdown="span">
  ![Feature importance panels](../assets/img/california_feature_importance.png){ width="100%" }
  <figcaption><code>plot_feature_importance</code> on California housing.</figcaption>
</figure>

---

## Local (per-observation) interpretation

### <span class="api-tag api-tag-function">function</span> `compute_local_explanation` { #compute-local-explanation }

```python
compute_local_explanation(model, x) -> LocalExplanation
```

Per-stage decomposition of a single prediction: the $f_+/f_-$ contributions, per-feature
backbone/tilt values, and the intercept $(b_0, d_0)$ absorbing the OLS scaling.

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `model` | `TSL` | _required_ | fitted model |
| `x` | `ndarray (n_features,)` | _required_ | the single point to explain |

**Returns** — `LocalExplanation`.

### <span class="api-tag api-tag-function">function</span> `plot_local_interpretation` { #plot-local-interpretation }

```python
plot_local_interpretation(explanations, points, titles, feature_names, save_path,
                          top_k_features=3, point_value_formatter=None,
                          units_label="Contribution to prediction",
                          prediction_format=<callable>, header=True) -> object
```

The three-column "Backbone × Tilt" local-interpretation plot — one column per point, rows =
stages sorted by absolute net contribution.

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `explanations` | `list[LocalExplanation]` | _required_ | from `compute_local_explanation` |
| `points` | `list[ndarray]` | _required_ | the explained points |
| `titles` | `list[str]` | _required_ | per-column titles |
| `feature_names` | `Sequence[str]` | _required_ | feature labels |
| `save_path` | `Path` | _required_ | output path |
| `top_k_features` | `int` | `3` | features shown per stage row |

**Returns** — the matplotlib figure.

<figure markdown="span">
  ![Local explanation — coastal](../assets/img/california_local_interp_coastal.png){ width="49%" }
  ![Local explanation — desert](../assets/img/california_local_interp_desert.png){ width="49%" }
  <figcaption><code>plot_local_interpretation</code> for a coastal vs. an inland point.</figcaption>
</figure>

---

## Component plots

### <span class="api-tag api-tag-function">function</span> `plot_grid_tensor_components` { #plot-grid-tensor-components }

```python
plot_grid_tensor_components(grid_tensor, individual_plots=False, axis=None)
```

Plot a single `GridTensor`'s backbone/tilt component curves.

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `grid_tensor` | `GridTensor` | _required_ | the component to plot |
| `individual_plots` | `bool` | `False` | one figure per axis vs. a combined grid |
| `axis` | `int \| None` | `None` | restrict to a single feature axis |

### <span class="api-tag api-tag-function">function</span> `plot_combined_grid_tensors` { #plot-combined-grid-tensors }

```python
plot_combined_grid_tensors(model, individual_plots=True, axis=None)
```

Overlay the combined grid-tensor components across a model's stages.

### <span class="api-tag api-tag-function">function</span> `plot_epoch_components` { #plot-epoch-components }

```python
plot_epoch_components(model, epoch) -> None
```

Plot the per-feature components for one stage/epoch.

| Name | Type | Default | Description |
|------|------|:--:|-------------|
| `epoch` | `int` | _required_ | the stage/epoch index |

---

## Result dataclasses

<span class="api-tag api-tag-dataclass">dataclass</span> Each plotting function returns a small dataclass exposing the underlying arrays, so you can export the numbers or build a bespoke figure without recomputing:

`PDDifferenceResult` · `NormalizedDiagnostics` · `PD2DResult` · `PD2DLinesResult` ·
`ICEResult` · `Backbone2DResult` · `Tilt1DResult` · `Tilt2DResult` ·
`TiltDiagnosticsResult` · `LocalExplanation` · `FeatureImportanceResult`.
