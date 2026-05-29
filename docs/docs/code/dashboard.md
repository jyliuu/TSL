# Dashboard (tslviz)

`tsl-split-evolution-dashboard/` is a standalone **FastAPI + D3** app, packaged as `tslviz`,
that reads an [evo-logging](logging.md) SQLite database **read-only** and replays how each
stage of a fit was built — split by split, component by component.

It is independent of the core: you fit with `visualdb=...` set, then point `tslviz` at the
resulting `.sqlite` file.

## Running it

```sh
tslviz --db run.sqlite              # serves on http://localhost:8051
tslviz --db run.sqlite --port 8080 --reload
```

Equivalently, set the `DATABASE_PATH` environment variable and run `tslviz` (or
`uvicorn tslviz.backend.app:app`). The entry point is `tslviz.backend.app:main`
(`pyproject.toml`). Dependencies are `fastapi`, `uvicorn[standard]`, and `orjson`.

## Layout

| Path | Role |
|------|------|
| `backend/app.py` | the entire FastAPI app: DB connection, JSON API, static-file serving |
| `frontend/index.html`, `styles.css` | the single-page D3 front end |
| `pyproject.toml` | packaging + the `tslviz` console script |

On startup the backend opens the DB with read-optimized pragmas (WAL, large cache) and
creates helper indexes; responses use `orjson` (encoding non-finite floats as `null`), GZip,
and permissive CORS.

## What it reads

The backend queries the tables the core's logger writes (see [Logging](logging.md)):

- **`runs`** — run metadata (rows, cols, params).
- **`events`** — split / resplit / merge actions (epoch, tree, iteration, column, split
  value, gain, counts).
- **`component_states`** — per-iteration backbone/tilt per feature (binary blobs).
- **`combined_grids`** — per-stage aggregated grid snapshots.
- **`epoch_scalings`**, **`training_errors`**, **`combination_choices`**,
  **`f_component_stats`**, **`tensor_lambdas`** — scaling, learning curves, aggregation
  choices, and $f_+/f_-$ branch statistics.

## The JSON API

`backend/app.py` exposes a read-only API consumed by the front end; a representative slice:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/runs` | list runs with summary counts |
| `GET /api/run/{id}/timeline` | ordered split events |
| `GET /api/run/{id}/learning`, `/convergence` | per-epoch learning / convergence curves |
| `GET /api/run/{id}/tree_evolution?epoch&tree_id&iteration` | reconstruct a tree's state up to an iteration |
| `GET /api/run/{id}/backbone_tilt_evolution[_all_columns]` | backbone/tilt over iterations |
| `GET /api/run/{id}/f_component_evolution`, `/f_component_per_axis` | $f_+/f_-$ branch evolution |
| `GET /api/run/{id}/identified_components[_all]` | final per-feature components after identification |
| `GET /api/run/{id}/tensor_lambdas`, `/scalings`, `/energy` | $\lambda_\pm$, OLS scalings, stage energy |

Helper routines reconstruct tree geometry by replaying split/resplit/merge events and decode
the binary `f64` component blobs (the same `encode_f64_array`/`decode_f64_array` convention
used by the core).

## Workflow

See [Visualizing a fit](../guides/visualizing.md) for the end-to-end loop: fit with
`visualdb`, launch `tslviz`, and read the stage-evolution views. The logging output the
dashboard depends on is exercised by `tests/evo_logging.rs` in the core crate.
