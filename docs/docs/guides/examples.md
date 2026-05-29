# Examples

The scripts in [`tsl-py/examples/`](https://github.com/jyliuu/TSL/tree/main/tsl-py/examples)
reproduce **every plot from the TSL paper** using the generic
[`tsl_py.plot`](../code/python-api.md#plotting-tsl_pyplot) helpers plus a little per-dataset
custom code (local explanations, paper-style panels, EBM/XGBoost comparisons).

```bash
# from the repo root
python tsl-py/examples/california.py
python tsl-py/examples/bike_sharing.py
python tsl-py/examples/synthetic.py
python tsl-py/examples/synthetic2.py
```

Each script accepts `--data-root` (raw CSVs; defaults to `reproducibility/data`), `--out`
(defaults to `/tmp/tsl_examples/<dataset>`), and defaults its model paths to the pretrained
binaries in `tsl-py/examples/models/<dataset>/` (`.bin` for TSL — `TSL.load` reads the
legacy MPF format — `.pkl` for EBM, `.json` for XGBoost). Pass `--refit` to retrain TSL from
scratch, or set a path to `""` to skip that model and any figure depending on it.

## The scripts

### `california.py` — spatial interaction & 1D faithfulness
California housing (geography + socioeconomics). Produces PD-difference plots, per-stage 2D
**spatial backbone** and PD panels, feature-importance plots, local explanations, and 1D PD
comparisons against EBM/XGBoost. The headline result: TSL's 1D PD curves keep localized
spatial structure (peaks near LA, SF, the Bay Area) that marginalization-based baselines
flatten. Use `--variant {blackbox, interpretable}` to switch the TSL model.

### `bike_sharing.py` — 2D interaction PD
UCI bike sharing. PD-difference plot and a 2D `hour × workingday` PD (`plot_2d_pd` with
`kind="lines"`), alongside an EBM PD comparison.

### `synthetic.py` — the masked interaction
The $Y = x_1^2 x_2 (1+x_3)$ construction where the 1D PD of $x_1$ is identically zero. All
models (TSL included) yield a near-zero 1D PD, but TSL's **signed-branch** backbone recovers
the quadratic effect — the practical payoff of the two-tensor form. See
[Partial dependence → masked interaction](../math/partial-dependence.md#signed-branch-pd-and-the-masked-interaction).

### `synthetic2.py` — bagging diagnostics
A two-feature case illustrating backbone bimodality at epoch 0 and validating the
similarity-filtering step of [aggregation](../math/bagging-aggregation.md): bagged grids
converge to two distinct representations of the same stage, and the reference + trim step
keeps a single canonical branch.

### `sepals_synthetic.py`
A small synthetic example used for factor-value illustrations and SepALS comparison.

## Notes

- Every `tsl_py.plot` helper returns the figure **and** the underlying arrays, so you can
  save directly or rebuild a custom visualization.
- The example outputs are committed under `tsl-py/examples/figures/` for reference.
- See the example folder's own
  [README](https://github.com/jyliuu/TSL/blob/main/tsl-py/examples/README.md) for the full
  per-script output table and CLI flags.
