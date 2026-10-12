# TSL

`src/forest.rs` and `src/forest/` implement the boosted model — a `Vec<StagePredictor>` —
and the stagewise fitting loop.

## The `TSL` struct

```rust
pub struct TSL {
    stage_predictors: Vec<StagePredictor>,
}
```

`TSL::predict(x)` sums the per-stage predictions: $\hat{m}(\mathbf{x}) = \sum_\ell \hat{m}^{(\ell)}(\mathbf{x})$,
each stage carrying its outer branch coefficients.

## The boosting loop (`forest/fitter.rs`)

`fit_boosted(x, y, &TSLBoostedParams) -> (FitResult, TSL)` runs the forward-stagewise loop
(`fit_boosted_with_test_error` is the same but also tracks held-out error per epoch):

1. initialize the outer residuals $R = y$;
2. for each epoch:
    - fit a `StagePredictor` on the current residuals (`fit_ensemble`);
    - extract the per-stage unscaled $\tilde{m}_+, \tilde{m}_-$;
    - **joint coefficient backfit:** refit `scaling_plus`/`scaling_minus` for *all* stages so
      far simultaneously, regressing $y$ on the stacked $[\tilde{m}_+^{(k)}, -\tilde{m}_-^{(k)}]$ columns;
      the default uses unconstrained SVD least squares, while `nonnegative_scaling=true`
      constrains every coefficient to be nonnegative;
    - recompute residuals $R = y - \sum_k \hat{m}^{(k)}$;
    - after the first epoch, decay the split budget: `n_iter ← n_iter * decay`.

This **orthogonal-greedy** refit (each stage fit greedily on residuals, then *all* stage
coefficients re-optimized together) is what makes later stages correct earlier ones rather
than merely stacking. The math is in [Fitting](../math/fitting.md#coefficient-backfitting).

The optional nonnegative solver (`fitter/nonnegative.rs`) normalizes the design columns
and uses SVD for each active-set subproblem. It checks feasible candidates in the
original coordinates and retains the best training-loss incumbent. Earlier stage
coefficients with zeros for the new stage provide a feasible warm start. Its iteration
budget can return before the optimality conditions are met. A coefficient may be zero;
when both effective branch scales are positive, the stage has finite backbone–sinh
coordinates. An inactive branch is a boundary case of the two-branch representation.

For multi-epoch fits with multiple grids per stage and split logging disabled,
`PreparedInput` caches the feature sort order, inverse ranks, quantile boundaries, and row-to-bin mappings
across epochs. Residual-dependent statistics are recomputed for each stage. The cache
belongs to the same feature view and `max_bins` configuration throughout the fit.

If a `visualdb_path` is set (and `evo-logging` is on), per-stage snapshots — grid JSON,
$\tilde{m}_+/\tilde{m}_-$ arrays, scaling — are streamed to SQLite; see [Logging](logging.md).

## Parameters (`forest/params.rs`)

`TSLBoostedParams` is the top-level config, flattened by `TSLBoostedParamsBuilder`:

| Field | Meaning |
|-------|---------|
| `epochs` | number of boosting rounds |
| `decay` | multiplicative `n_iter` decay after epoch 1 (`1.0` = none) |
| `nonnegative_scaling` | constrain every outer branch coefficient to be nonnegative; default `false` |
| `sp_params` | nested `StagePredictorParams` (n_trees, aggregation, …) |
| `seed` | RNG seed for reproducibility |
| `visualdb_path` | optional SQLite path for evo-logging |

The builder nests `StagePredictorParamsBuilder` → `GridTensorParamsBuilder`, so a full
configuration is assembled fluently. The Python `TSL.fit(...)` classmethod maps its flat
keyword arguments onto this tree — see [Python API](python-api.md) and the
[Hyperparameters](../guides/hyperparameters.md) reference.

## Tests

- `tests/forest.rs` — boosted-fit reproducibility, seed consistency, the positivity
  invariants, error computation.
- `tests/openml_tuned.rs` — tuned-hyperparameter fits on OpenML datasets.
- `tests/serialization.rs` — Serde round-trips for `GridTensor`, `StagePredictor`, `TSL`.

Run them with `cargo test -p tsl_rust --release` (the suite is numerically heavy — always
`--release`).
