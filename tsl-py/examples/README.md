# TSL Python — examples

End-to-end scripts that reproduce **every plot from the TSL paper**, using
the generic plotting helpers in `tsl_py.plot` plus a small amount of
per-dataset custom code (local explanations, paper-style split-panel
exports, EBM/XGBoost comparison plots).

## Running

```bash
# from the repo root
python tsl-py/examples/california.py
python tsl-py/examples/bike_sharing.py
python tsl-py/examples/synthetic.py
```

Each script:
- accepts `--data-root` (raw CSVs; defaults to `reproducibility/data`)
- accepts `--out` (defaults to `/tmp/tsl_examples/<dataset>`)
- defaults its model paths to the pretrained binaries shipped in
  `tsl-py/examples/models/<dataset>/` (`mpf_*.bin` for TSL, `ebm_model.pkl`
  for EBM, `xgb_model*.json` for XGBoost). `TSL.load(...)` reads the legacy
  MPF `.bin` format directly. Pass `--refit` to retrain TSL from scratch,
  or set a path to `""` to skip that model (and any figure that depends on it).

## What each example produces

Filenames are human-readable — no `figure_X_Y_` paper-index prefixes.

### California ([`california.py`](california.py))

| Output file | Source |
|---|---|
| `pd_difference_plot_{blackbox,interpretable}.pdf` | `pd_difference_plot` |
| `spatial_backbone_evolution.pdf` | `plot_2d_backbone` (combined) |
| `spatial_backbone_stage{1,2}.pdf` | per-stage backbone-product panel, re-drawn from `Backbone2DResult` |
| `spatial_pd_stage{1,2}.pdf` | per-stage 2D-PD panel, re-drawn from `Backbone2DResult` |
| `feature_importance_{blackbox,interpretable}.pdf` | `plot_feature_importance` |
| `local_explanations_{blackbox,interpretable}.pdf` | verbatim port of `cali_analysis.py::plot_figure_5_local_explanations` |
| `pd_comparison_{latitude,longitude}.pdf` | 1D PD overlay: TSL (Stage 1) + EBM + XGBoost (blackbox) + XGBoost (interpretable) |

Pass `--variant {blackbox, interpretable}` to switch the TSL model. To
regenerate the paper's full California figure set, run both variants
into the same `--out` directory (the paper uses the *interpretable* TSL
for the spatial-backbone plot, so run that variant last):

```bash
python tsl-py/examples/california.py --variant blackbox      --out tsl-py/examples/figures/california
python tsl-py/examples/california.py --variant interpretable --out tsl-py/examples/figures/california
```

### Bike Sharing ([`bike_sharing.py`](bike_sharing.py))

| Output file | Source |
|---|---|
| `pd_difference_plot.pdf` | `pd_difference_plot` |
| `pd_hour_workingday_tsl.pdf` | `plot_2d_pd` (`kind="lines"`) |
| `pd_hour_workingday_ebm.pdf` | EBM PD via repeated `ebm.predict` (custom) |

### Synthetic PD-cancellation ([`synthetic.py`](synthetic.py))

| Output file | Source |
|---|---|
| `pd_difference_plot.pdf` | `pd_difference_plot` (combined 2×3) |
| `pd_difference_plot_x{1,2,3}.pdf` | per-feature, stage-1 split-out, re-drawn from `PDDifferenceResult` |
| `ice_x1_tsl.pdf` | TSL `plot_ice` |
| `ice_x1_ebm.pdf` | EBM ICE (custom) |
| `ice_x1_xgboost.pdf` | XGBoost ICE (custom) |
| `pd_x1_all_models.pdf` | 1D PD overlay: TSL + EBM + XGBoost |
| `pd_x1_x2.pdf` | TSL `plot_2d_pd` (surface) |

## Pre-rendered output

[`figures/`](figures/) holds the rendered PDFs from a fresh run on the
pretrained models, mirroring the per-dataset paths above. Re-running the
scripts with default arguments will reproduce them.

## Optional dependencies

The comparison plots need `interpret` (for EBM) and `xgboost`:

```bash
pip install interpret 'xgboost==2.1.3'
```
