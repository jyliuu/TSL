# Hyperparameters

All hyperparameters are exposed as flat keyword arguments on `TSLRegressor` (and the
`TSL.fit(...)` classmethod), and mapped onto the nested Rust
[builders](../code/architecture.md#the-builder-pattern):
`TSLBoostedParams` → `StagePredictorParams` → `GridTensorParams`
(`SplitStrategyParams`, `RefinementStrategyParams`). Defaults below are the `TSLRegressor`
constructor defaults.

## Boosting (forest level)

| Param | Default | Meaning |
|-------|:------:|---------|
| `epochs` | `10` | number of boosting rounds (stages $R$); each fits the current residual |
| `decay` | `1.0` | multiply `n_iter` by this after epoch 1 (`<1` makes later stages coarser; `1.0` = off) |
| `seed` | `42` | RNG seed — fits are deterministic given the seed |
| `verbosity` | `1` | log verbosity |
| `visualdb` | `None` | path to an evo-logging SQLite DB; see [Visualization dashboard](visualizing.md) |

## Bag & aggregation (stage level)

| Param | Default | Meaning |
|-------|:------:|---------|
| `n_trees` | `10` | grid tensors fit per stage ($n_{\text{grids}}$); more reduces variance |
| `similarity_threshold` | `0.0` | trim $\xi$ — keep top $\lceil(1-\xi)n_{\text{trees}}\rceil$ bags by similarity (`0` keeps all) |
| `bagged` | `False` | enable the bagged-aggregation path |

The aggregation mode (`Mean` / `GeometricMean` / `Combined`) is set internally; see
[Bagging & aggregation](../math/bagging-aggregation.md).

## Grid refinement (per grid)

| Param | Default | Meaning |
|-------|:------:|---------|
| `n_iter` | `10` | split budget per grid ($T$); larger captures more structure |
| `split_strategy` | `"random"` | `"random"`, `"best_split"`, or `"top_k"` |
| `split_try` | `10` | candidate split positions sampled per (feature, interval) |
| `colsample_bytree` | `0.8` | fraction of features sampled per split proposal |
| `min_interval_samples` | `1` | minimum observations on each side of a split |
| `min_split_loss` | `0.0` | minimum objective reduction to accept any structural action |
| `complexity_penalty` | `0.0` | fixed-scale boundary cost used by split, resplit, and merge (`0.0` disables merge candidates) |
| `top_k` | `10` | (for `top_k`) sample from the top-$k$ candidates |
| `must_fill_all_k` | `True` | (for `top_k`) require all $k$ slots filled |

!!! tip "Choosing a split strategy"
    `random` (the default) is usually best for speed and generalizes well; `best_split`
    can help on small datasets where exhaustively picking the top split matters.

!!! tip "Tuning structural complexity"
    A positive finite `complexity_penalty` assigns each boundary a cost fixed from the
    grid's initial loss scale. Splits pay the cost, resplits leave it unchanged, and merges
    recover it. Start with a small value such as `0.01`, measure validation error and
    boundary count, and increase it when a smaller model is worth the extra bias.

## Refinement solver (per node)

| Param | Default | Meaning |
|-------|:------:|---------|
| `refinement_strategy` | `"l2"` | `"l2"` (weights $w_i=1$) or `"huber"` (robust weights) |
| `alpha` | `0.0` | $\ell_2$ penalty $\alpha\beta^2$ on the log-backbone update $\beta=\log v_b$ |
| `tilt_tau` | `0.01` | $\ell_2$ penalty $\tau\delta^2$ on the tilt update $\delta=\frac12\log(v_+/v_-)$ |
| `tilt_rho` | `0.0` | $\ell_1$ penalty $\rho|\delta|$; positive values can produce exactly zero tilt updates |

These feed the [closed-form $2\times2$ proposal solver](../math/fitting.md#the-closed-form-2x2-solver),
which keeps the backbone and tilt directions separate and scores the clamped proposal with
the exact logarithmic penalties.

## Advanced / experimental

| Param | Default | Meaning |
|-------|:------:|---------|
| `prior_sample_size` | `0.0` | parent-anchoring strength ($\tau_0$); the default `0.0` disables anchoring |
| `update_clamp` | `inf` | cap on update magnitude; the default `inf` imposes no extra cap |

!!! note
    Leave the two advanced parameters at their defaults unless you are experimenting — the
    defaults shown are the no-op values. The positivity clamp $[v_{\min}, v_{\max}]$ in the
    solver is separate and always active.

## A reasonable starting point

```python
TSLRegressor(
    epochs=5,            # a few stages
    n_trees=16,          # moderate bagging
    n_iter=30,           # enough splits to capture structure
    split_try=16,
    colsample_bytree=0.8,
    alpha=0.01,          # light log-backbone shrinkage
    seed=0,
)
```

Tune `epochs`, `n_iter`, and `n_trees` first. Use `alpha` to shrink multiplicative backbone
changes, `tilt_tau`/`tilt_rho` to shrink tilt changes, `complexity_penalty` for structural
cost-complexity selection, and `similarity_threshold` if bagged components disagree.
