# Visualizing a fit

The [`tslviz` dashboard](../code/dashboard.md) replays how a TSL model was built — every
split, the backbone/tilt evolution, the $f_+/f_-$ branches, and the per-stage aggregation.
It reads a SQLite database that the core writes during a fit (the
[evo-logging](../code/logging.md) system).

## 1. Fit with a `visualdb` path

Set `visualdb` to a SQLite file when fitting (requires the `evo-logging` feature, on by
default):

```python
from tsl_py.sklearn import TSLRegressor

model = TSLRegressor(epochs=5, n_trees=16, n_iter=30, seed=0, visualdb="run.sqlite")
model.fit(X, y)
```

Every split / resplit / merge, component snapshot, and per-stage scaling is streamed to
`run.sqlite`.

## 2. Launch the dashboard

Install and run `tslviz` (a separate package, `tsl-split-evolution-dashboard/`):

```bash
pip install ./tsl-split-evolution-dashboard   # or from the repo
tslviz --db run.sqlite                          # serves http://localhost:8051
```

Options: `--port` to change the port, `--reload` for autoreload. Alternatively set the
`DATABASE_PATH` environment variable and run `tslviz` with no `--db`.

## 3. Explore

The single-page UI is backed by the read-only JSON API
([endpoint list](../code/dashboard.md#the-json-api)). Typical things to look at:

- **Timeline / learning curves** — how error dropped split by split and epoch by epoch.
- **Tree evolution** — reconstruct a single grid's interval structure up to any iteration.
- **Backbone / tilt evolution** — watch the magnitude gate and signed direction form per
  feature.
- **$f_+ / f_-$ components** — the two positive branches whose difference is the stage.
- **λ scatter / scalings / energy** — diagnose aggregation across the bagged grids
  (the [bimodal-alignment](../math/bagging-aggregation.md#the-bimodal-alignment-example)
  issue shows up here).

The database is opened read-only, so the dashboard never modifies your run. The logging
output it consumes is exercised by `tests/evo_logging.rs` in the core crate.
