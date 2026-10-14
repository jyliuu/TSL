# StagePredictor

`src/stage_predictor.rs` and `src/stage_predictor/` implement one **boosting stage**: a bag
of `GridTensor`s aggregated into one representative grid, plus the outer branch scaling that is the
*only* place stage scaling is applied.

## The `StagePredictor` struct

```rust
pub struct StagePredictor {
    pub grid_tensors: Vec<GridTensor>,        // the bag of fitted components (n_trees)
    pub primary_grid_tensor: GridTensor,      // the aggregated representative grid
    pub candidate_indices: Option<Vec<usize>>,// indices kept after similarity filtering
    pub aggregation_method: Aggregation,      // Mean | GeometricMean | Combined
    pub scaling_plus: Option<f64>,            // outer coefficient for m̃₊
    pub scaling_minus: Option<f64>,           // outer coefficient for −m̃₋
    pub energy: Option<f64>,                  // optional unscaled prediction energy
}
```

### Prediction and the scaling invariant

`StagePredictor::predict` is where the outer branch scaling is applied — exactly once:

$$
\text{stage prediction} = \texttt{scaling_plus}\cdot \tilde{m}_+ \;+\; \texttt{scaling_minus}\cdot(-\tilde{m}_-).
$$

The free function `extract_two_tensor_predictions_unscaled(grid, x)` pulls $\tilde{m}_+$ and $\tilde{m}_-$
straight from the two-tensor fields with **no** scaling — this is what the
[backfit](../math/fitting.md#coefficient-backfitting) regresses on, and what enforces the
[scaling invariant](architecture.md#two-critical-invariants). `predict_unscaled` returns the
stage prediction before outer branch scaling.

### The `Aggregation` enum

```rust
pub enum Aggregation { Mean, GeometricMean, Combined }
```

| Variant | How the bag is reduced |
|---------|------------------------|
| `Mean` | arithmetic mean of the unscaled per-grid predictions |
| `GeometricMean` | sign-preserving geometric mean |
| `Combined` | take $\tilde{m}_+, \tilde{m}_-$ from the aggregated two-tensor `primary_grid_tensor` and apply `scaling_plus`/`scaling_minus` |

## Fitting a stage (`fitter.rs`)

`fit_ensemble` builds a stage:

1. derive `n_trees` seeds and fit that many `GridTensor`s independently when
   `consensus=0` — in parallel under the `use-rayon` feature. With `bootstrap` (the
   default), each grid fits its own
   [bootstrap resample](../math/bagging-aggregation.md#bootstrap-resampling): row $i$'s
   weight is multiplied by its draw count in $n$ draws with replacement, from a stream
   derived from the grid's seed;
2. align them to a common grid (`refine_grids_to_union_two_tensor`) and center each
   axis's log-backbone and tilt for factor-shape comparison;
3. similarity-filter and combine them into `primary_grid_tensor`
   (`aggregate_bagged_two_tensor`, gated by `similarity_threshold`);
4. return the `StagePredictor` and a `FitResult`. The boosted forest's
   [coefficient backfit](../math/fitting.md#coefficient-backfitting) sets the outer
   `scaling_plus`/`scaling_minus` values.

`fit_ensemble_weighted(x, y, sample_weight, params, rng)` builds the same stage for the
weighted loss $\sum_i s_i\,(y_i-f(x_i))^2$: every grid in the bag, including a
consensus bag, weighs row $i$'s loss and consensus terms by $s_i$, and the returned
`FitResult.err` is the weighted mean squared residual. It uses the weights as given;
`fit_boosted_weighted` rescales them to mean one before fitting its stages. See
[weighted fits](forest.md#weighted-fits).

For bags whose grids fit all rows, with multiple grids and split logging disabled,
initialization computes the deterministic fitting state once and clones it for each
independently seeded grid. A bootstrap bag shares the rows' sort orders and histogram
bins (`PreparedInput`) and initializes each grid from its own weights. Logging-enabled
fits initialize each grid in its own logging context.

With `consensus` $\rho>0$, the bag is fitted in lockstep (`consensus.rs`). Every
grid starts from the shared initial state. Each round, an eligible grid receives
the mean branches of the other grids as its `BranchConsensus` reference.
Existing full-row references reuse their two buffers when the destinations and all
branch inputs are contiguous and have exactly the training row count. Each target
retains subtraction followed by division using the round's fixed branch totals.
Other shapes, layouts and first references use the array-expression construction.
Full-row consensus uses `RefinementStrategy::refresh_statistics` to rebuild the
reference-dependent statistics, interval summaries, and resplit/merge scores, then the grid takes up to
`consensus_interval` action attempts
under the [branch-consensus objective](../math/bagging-aggregation.md#consensus-fitting).
The default interval is 1. References stay fixed throughout each batch, and the grids
run in parallel under `use-rayon`.

Initialization scores all retained split positions. After a round's reference refresh,
a two-branch `Random` proposal evaluates only its sampled split positions with the same
numeric evaluator. Positive-only states and other strategies use eager split scoring.
Pending scores are materialized before intermediate actions, whose local refreshes
remain eager; this retains the floating-point values cached for untouched intervals.
After the batch's final action, interval sums and candidate scores can wait for the
next round's reference refresh. Full-row L2 fitting keeps curvature row and prefix
statistics current and defers the two linear statistics to that refresh; anchored,
Huber and eager updates maintain all five. Proposals require fully refreshed statistics.
With binned caches for every feature, full-row L2 deferred updates locate the touched
bins during the curvature scatter, retaining row and cumulative addition order.
The unused split solves are avoided when a proposal follows a global refresh and its
action uses this final-action deferral, as with `consensus_interval=1`.
This internal scoring policy adds no public parameter; parity tests compare its actions,
RNG state, logs, and serialized models with eager scoring.
A stalled grid ends its batch and is retried while another grid progresses.
Terminated grids, grids that exhaust their attempt budget, and grids at their split
budget with merging disabled remain fixed and continue contributing their branches.
Each grid has at most $3\,\texttt{n_iter}$ action attempts, with individual attempt
numbers recorded in the split log. Fitting also stops after a whole round without
progress. The reference is dropped before identification.

`consensus_anchors` optionally selects a uniform sample of training rows shared by
all bags in the stage. The data loss uses all rows; the consensus penalty uses the
sample with weight $\rho n/m$ per anchor. Sampling has its own deterministic RNG
stream derived from the stage's first bag seed. `None` and counts at least $n$ use the
full-row path. With binned caches on every feature and merges disabled,
`refresh_consensus_reference` updates only reference-dependent anchor contributions
after initialization; other configurations use a full weighted refresh. Both L2
and Huber data statistics retain all training rows.

The coupling encourages training-branch agreement while preserving a separate RNG
and partition for every grid. It does not guarantee convergence, identical factors,
or latent identifiability; it can also reduce the diversity supplied by independent bags.

`fit_grid_bag(x, y, params, rng)` exposes the ordered `(FitResult, GridTensor)` pairs
before component aggregation. Each result contains the normalized grid's training
predictions. Rust callers implementing prediction-based ensemble rules can reuse those
values and choose their own aggregation.

## Combining grids

- **`combine_grids.rs`** — `refine_grids_to_union_two_tensor` resamples every grid onto the
  union of all split points per axis (so components are comparable interval-by-interval);
  `geometric_mean_combiner` is the sign-preserving geometric mean used internally.
- **`aggregate_bagged.rs`** — `aggregate_bagged_two_tensor` runs the full
  [align → normalize → reference → similarity-filter → log-space average](../math/bagging-aggregation.md)
  pipeline and returns the combined grid. The retained indices are written to the
  combination log; `fit_ensemble` stores all fitted grid indices in `candidate_indices`.

Union alignment walks the ordered source and union boundaries together. The combined
primary grid coalesces adjacent intervals when both backbone and tilt values have equal
bit patterns. This preserves each branch, its predictions, and total observation counts.

## Parameters (`params.rs`)

`StagePredictorParams` carries `n_trees`, the nested `GridTensorParams`,
`similarity_threshold` ($\xi$), `aggregation_method`, `consensus` ($\rho$),
`consensus_interval` (default 1), `consensus_anchors` (default `None`), and `bootstrap`
(default `true`), with a fluent
`StagePredictorParamsBuilder` that delegates to `GridTensorParamsBuilder`. See
[Architecture → builder pattern](architecture.md#the-builder-pattern).
