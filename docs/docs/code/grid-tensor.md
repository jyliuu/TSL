# GridTensor

`src/grid_tensor.rs` and the `src/grid_tensor/` module tree implement one fitted separable
component — the rank-1 product in [two-tensor form](../math/model.md#backbone-and-exponential-tilt).
This is the heart of the codebase.

## The `GridTensor` struct

```rust
pub struct GridTensor {
    pub splits: Vec<Vec<f64>>,              // split thresholds per axis
    pub observation_counts: Vec<Vec<usize>>,// samples per interval per axis (for normalization)
    pub intervals: Vec<Vec<(f64, f64)>>,    // interval bounds per axis (±inf at the ends)
    pub scaling: f64,                       // legacy; ignored in two-tensor mode (= 1.0)
    pub backbone_values: Vec<Vec<f64>>,     // b_j ≥ 0  per interval per axis
    pub tilt_values: Vec<Vec<f64>>,         // d_j ∈ ℝ  per interval per axis
    pub lambda_plus: f64,                   // λ₊ ≥ 0
    pub lambda_minus: f64,                  // λ₋ ≥ 0
}
```

Prediction follows the model directly — the difference of the two non-negative branches,
$\hat{m}_+ - \hat{m}_- = \lambda_+\prod_j b_j e^{d_j} - \lambda_-\prod_j b_j e^{-d_j}$:

| Method | Returns |
|--------|---------|
| `new_two_tensor(...)` | construct from the two-tensor fields (sets `scaling = 1.0`) |
| `predict_single_unscaled(x)` / `predict_unscaled(X)` | **unscaled** $\tilde{m}_+ - \tilde{m}_-$ (one point / batch) |
| `predict_single_backbone_and_tilt(x)` / `predict_backbone_and_tilt(X)` | decompose into $(\prod_j b_j,\ \sum_j d_j)$ |
| `coalesce_equal_bins()` | merge adjacent bins with bitwise-equal backbone and tilt values, summing their observation counts |

Batch prediction caches the two branch factors per bin when there are enough input
values to amortize their construction. Multiplication follows feature order, preserving
the scalar prediction calculation.

!!! warning "Use the unscaled path"
    `predict_unscaled` returns $\tilde{m}_+ - \tilde{m}_-$ **without** $\lambda$ scaling beyond what is
    baked into the grid; the OLS stage scaling lives on `StagePredictor`. See the
    [scaling invariant](architecture.md#two-critical-invariants).

## The fitting loop

`grid_tensor::fit` (`src/grid_tensor/fit.rs`) iterates the action/reducer pattern:

1. build a `FittingState` from `(x, y)`;
2. initialize the `RefinementStrategy` row statistics and prefix sums, then apply the
   optional histogram-binning mask (`max_bins`);
3. score the retained candidate positions and initialize the `SplitStrategy`
   allowed intervals;
4. score eligible split, resplit, and merge actions under one fixed-scale cost-complexity
   objective, then apply the highest positive-gain action through the **reducer**;
5. at the `n_iter` fineness budget, exclude new splits while leaving resplits and merges
   eligible; a merge frees capacity for another split;
6. normalize with [`l2_identify`](#identification) and return the `GridTensor`.

With `complexity_penalty=0.0`, merge statistics are not maintained and fitting retains the
split/resplit-only budget behavior.

### `FittingState` (`state.rs`)

The mutable per-grid context threaded through the loop: the in-progress two-tensor grid
(`backbone_values`, `tilt_values`, `lambda_plus`, `lambda_minus`); per-point caches
(`f_plus`, `f_minus`, `f`, `r_tilde` = within-stage residuals); sorted row/rank mappings;
and a `PrecomputedStatistics` cache of prefix sums and per-interval sufficient statistics
enabling $O(1)$ split-candidate scoring. Holds an optional `LoggingState`.

### Actions (`action.rs`) and splitting (`splitting.rs`)

`FittingAction` enumerates the three structural actions: `ApplySplit`, `ApplyResplit` and
`ApplyMerge`. The `SplitStrategy` enum proposes them:

- **`Random`** — sample up to `split_try` valid positions over `colsample_bytree` of the
  features (the default; usually best for speed);
- **`Best`** — greedily take the highest-gain split (better on small data);
- **`TopK`** — sample from the top-`k` candidates (`must_fill_all_k` controls strictness).

A `SplitCandidate` carries the column, position, parameter-regularized gain, and the
left/right $(u_+, u_-)$ updates. A `MergeCandidate` identifies the removed boundary and its
parameter-regularized gain; the absolute merged factors live in the aligned precomputed cache.
A `ResplitCandidate` refits the factors in the two intervals adjacent to an existing
boundary, keeping its position fixed.
`SplitStrategyState` tracks valid positions and the fixed boundary cost. After a split at
position `idx`, the neighborhood is forbidden from further splitting.

### The reducer (`reducer.rs`)

`fitting_reducer` matches on the action and applies it through the action's submodule
(`reducer/split.rs`, `resplit.rs`, `merge.rs`): it inserts the split into the grid,
updates backbone/tilt on the two new sub-intervals, recomputes the affected per-point caches
and the prefix-sum/interval statistics. With `evo-logging` and an active logging state,
it constructs and emits transformation events. For a merge, it removes one boundary, stores the candidate's absolute factors, and rescales
the rows of the two merged intervals by the ratio of the merged factors to their side's
previous factors. Prediction
updates scatter through contiguous cache slices when available; strided arrays use
indexed access. Each row retains the same branch, prediction and residual arithmetic.

Initialization scores all retained split positions eagerly. Ordinary fitting also
refreshes affected candidate scores after each action. For two-branch `Random` consensus
proposals following a global reference refresh, the split evaluator computes gains and
updates only at the sampled positions. The existing sampling order, eligibility masks,
and action comparison apply; interval statistics and resplit/merge scores are current.
Positive-only states and other split strategies use eager scoring. This execution policy
has no public parameter.

Before an intermediate consensus action, pending split scores are
materialized and local refreshes remain eager. Untouched intervals retain their cached
scores: recomputing them from shifted prefix sums can change floating-point rounding.
The batch's final action defers interval sums and candidate scores until the next
round's reference refresh. With L2 and a full-row consensus reference, it updates the
curvature row and prefix statistics $s_{11},s_{22},s_{12}$ in the same arithmetic order
and defers the linear statistics $t_1,t_2$. The next full reference refresh reconstructs
both linear row arrays and their prefixes before rebuilding interval and candidate
scores. The refresh writes each row directly to its statistic caches. Constant L2
weights with unchanged consensus support retain the curvature caches; other refreshes
write all five statistics. Full-row references with contiguous prediction, within-stage residual and
reference buffers rebuild the two linear fields through slices, preserving each product
and penalty addition. Strided or incomplete buffers use the indexed row evaluation.
An internal stale-state marker rejects proposals and incremental updates until
that reconstruction. Anchored references, Huber fitting and eager updates maintain all
five statistics. When every feature has a binned cache, the full-row L2 deferred update
locates touched bins while scattering curvature deltas in affected-row order. It stores
each row's curvature immediately and retains only the three deltas for the feature
passes. Nonchanged features share the row/delta traversal in pairs, using separate
zero-initialized histograms in a reusable buffer. Each feature preserves its row-ordered
additions and prefix sweep; an unpaired feature and the changed column use individual
traversals. Each prefix update starts at the first touched bin, includes zero-delta rows in
its bounds, and preserves the existing cumulative and tail arithmetic. Exact or constant
columns use the affected-rank scan. Split solves
are saved when the post-reference proposal leads directly to this deferred final action,
as in one-action batches; intermediate actions materialize the pending split cache.
Logging and finalization read the selected action, factors and predictions independently
of the deferred statistics. Eager/on-demand parity tests compare random-number
state, action logs, branch values, and serialized models across loss and binning settings.

For anchored consensus, `BranchConsensus` stores a shared row-to-anchor lookup and
reference values only on the sampled rows. `point_statistics` adds the consensus
term only at anchors, weighted by $\rho n/m$; its supervised contribution uses every
row. Merge scoring applies the same anchor weights in its direct objective scan.

`refresh_consensus_reference` can update reference-dependent $t_1,t_2$ contributions
using the anchors and existing bin prefixes. It retains curvature and data statistics
when the anchor support and weight are unchanged, every feature has a binned cache,
and merges are disabled. The reference update then costs $\mathcal O(p(m+K))$ plus
candidate solves, where $K$ bounds bins per feature. Initialization, changed support
or weights, exact or constant columns, and merge-enabled fits use a full weighted
refresh. Ordinary structural actions retain their all-row data updates.

## The two-tensor solver

`solve_two_tensor` (`src/grid_tensor/two_tensor_solver.rs`) is the load-bearing primitive.
It builds a locally regularized $2\times2$ proposal system from the five sufficient
statistics $S_{11}, S_{22}, S_{12}, t_1, t_2$,

$$
A =
\begin{pmatrix}S_{11}&S_{12}\\S_{12}&S_{22}\end{pmatrix}
+\frac14
\begin{pmatrix}\alpha+\tau&\alpha-\tau\\\alpha-\tau&\alpha+\tau\end{pmatrix},
\qquad t=\begin{pmatrix}t_1\\t_2\end{pmatrix},
$$

solves $\hat{u}=A^{-1}t$ (or the three-case subgradient system when $\rho>0$), and clamps
$v_\pm = \mathrm{clamp}(1+\hat{u}_\pm, v_{\min}, v_{\max})$ to preserve positivity. The
candidate score subtracts the exact log-coordinate cost
$\alpha\beta^2+\tau\delta^2+\rho|\delta|$, where
$\beta=\tfrac12(\log v_++\log v_-)$ and
$\delta=\tfrac12(\log v_+-\log v_-)$. The full derivation is in
[Fitting → the closed-form solver](../math/fitting.md#the-closed-form-2x2-solver).

Numerical rank is determined relative to the system's scale. For a positive-semidefinite
rank-deficient system, the solver proposes a minimum-norm step in its retained range,
clamps it to the multiplier bounds, and backtracks against the exact log-penalized
objective. With a tilt $\ell_1$ penalty, it also considers the signed tilt regions and
the zero-tilt boundary. A finite positive-gain update is required on this path.

### Refinement strategies (`refinement.rs`)

`RefinementStrategy` has two variants that differ only in the weights $w_i$:

- **`L2Refinement`** — $w_i = 1$;
- **`HuberRefinement`** — robust weights $w_i = \min(1, c/|r_i|)$, $c\approx1.345$.

Both expose `alpha` (log-backbone $\ell_2$), `tilt_tau` (tilt $\ell_2$), `tilt_rho` (tilt
$\ell_1$) and `update_clamp`. `initialize` precomputes the penalized
gain of every candidate position.

`update_clamp` accepts nonnegative values and positive infinity. With cap $c$,
branch multipliers lie in $[\max(0.05,e^{-c}),\min(20,e^c)]$; infinity retains the
numerical bounds. Split and boundary-refit updates use the current factors as
their reference. Merge updates use the count-weighted geometric reference of
the neighboring factors. Proposal scoring and action application share these
bounds; see [Fitting](../math/fitting.md).

## Identification

`l2_identify` (`src/grid_tensor/identification.rs`) applies the prediction-preserving
[normalization](../math/model.md#normalization-gauge-fixing) after a fit: it rescales each
axis's backbone to unit weighted $L^2$ (pushing the scale into $\lambda_\pm$) and centers
each tilt by its weighted mean (pushing the offset into $\lambda_+ \mathrel{*}= e^{\sum c_j}$,
$\lambda_- \mathrel{*}= e^{-\sum c_j}$). It does **not** flip orientation (which would change
predictions). The same routine normalizes the combined grid after factor averaging
([Bagging & aggregation](../math/bagging-aggregation.md)).

## Histogram binning

`apply_histogram_binning_mask` (`src/grid_tensor/histogram_binning.rs`) is an optional
prologue enabled by `max_bins >= 2`. Quantile targets snap to the nearest boundary between
distinct values, with lower-rank ties selected deterministically. Features with at most
`max_bins` distinct values retain every distinct-value boundary. Repeated targets share a
boundary, so a feature can have fewer candidates than the requested bin count.

The mask excludes other positions before initial candidate scoring and replaces
per-position prefix sums with per-bin cumulative sums. Each candidate uses constant-time
prefix lookups; statistic updates traverse retained bin boundaries. With `max_bins: None`
the exact candidate path is used. Parity between the two paths is tested in
`tests/histogram_binning.rs`.

Each binned feature stores a `u16` bin index for every training row. Statistic updates
use this mapping to place row deltas into bin buckets. The immutable mappings are shared
across cloned fitting states and reused across stages when `PreparedInput` is available.

## Parameters (`params.rs`)

`GridTensorParams` bundles `n_iter` (the split budget), a `SplitStrategyParams`
(`split_try`, `colsample_bytree`, `min_interval_samples`, `min_split_loss`,
`complexity_penalty`, kind), a
`RefinementStrategyParams` (`alpha`, `tilt_tau`, `tilt_rho`, …, kind), and `max_bins`. All
have fluent `…Builder` types. The Python flat hyperparameters map onto these — see
[Hyperparameters](../guides/hyperparameters.md).
