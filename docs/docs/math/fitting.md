# Fitting

TSL fits a regression function in stages. Each stage asks: **what separable function best
predicts the residuals left by the current model?**

The answer is built at three levels:

1. one grid repeatedly splits, refits, or merges feature intervals;
2. one stage combines several randomized grids;
3. the full model jointly refits the coefficients of every completed branch.

This page follows those levels in order. The code has the same hierarchy:
[`GridTensor`](../code/grid-tensor.md), [`StagePredictor`](../code/stage-predictor.md),
and [`TSL`](../code/forest.md).

## What the fit targets

Let $(X_i,Y_i)$, $i=1,\ldots,n$, be the training observations, where
$X_i\in\mathbb R^p$ and $Y_i\in\mathbb R$. Under squared-error loss, the population
prediction target is

$$
m^\star(x)=\mathbb E[Y\mid X=x].
$$

This is the mean response among observations with features $x$.

For a model with $R$ stages and fixed hyperparameters, let
$\mathcal F_{\mathrm{TSL}}$ be the set of prediction functions that TSL can represent.
Its members are finite sums

$$
m(x)
=
\sum_{\ell=1}^R
\sum_{s\in\{+,-\}}
\gamma_{\ell,s}h_{\ell,s}(x),
\qquad
\gamma_{\ell,s}\in\mathbb R.
$$

Each $h_{\ell,s}$ is a non-negative separable branch of an aggregated stage grid. The two
branches in a stage share a partition. Their outer coefficients are unconstrained, so
either branch may have either sign in the final model.

The best population prediction available in this family is

$$
m^\star_{\mathrm{TSL}}
\in
\operatorname*{arg\,min}_{m\in\mathcal F_{\mathrm{TSL}}}
\mathbb E\left[(Y-m(X))^2\right].
$$

Fitting is a greedy, randomized estimate of this best-in-family function. The target is
the prediction function itself. Different stages and factors can represent the same
function, so normalization chooses a stable stored representation rather than a unique
scientific decomposition. See
[Identifiability and stability](model.md#identifiability-and-stability).

### What one stage targets

Suppose stages $1,\ldots,\ell-1$ give the current predictor $\hat m_{\ell-1}$. Stage
$\ell$ receives the **stage residuals**

$$
R_i^{(\ell)}=Y_i-\hat m_{\ell-1}(X_i).
$$

The first stage has $\hat m_0=0$, so $R_i^{(1)}=Y_i$. If we hold the current fitted model
fixed, the ideal correction for a fresh observation is

$$
g_\ell^\star(x)
=
\mathbb E\left[Y-\hat m_{\ell-1}(X)\mid X=x\right]
=
m^\star(x)-\hat m_{\ell-1}(x).
$$

One stage greedily approximates this correction. After adding it, TSL jointly refits the
coefficients of all completed branches. The next stage uses the residuals after that
least-squares refit.

## How one grid represents a correction

A grid has positive and negative branches

$$
f_+(x)=\lambda_+\prod_{j=1}^p a_{+,j}(x_j),
\qquad
f_-(x)=\lambda_-\prod_{j=1}^p a_{-,j}(x_j),
$$

and predicts

$$
g(x)=f_+(x)-f_-(x).
$$

The internal scales satisfy $\lambda_+,\lambda_-\geq0$, and the univariate factors
$a_{\pm,j}$ are positive step functions. For feature $j$, let

$$
\mathcal P_j=\{I_{j,1},\ldots,I_{j,L_j}\}
$$

be a partition of its observed range. The factors are constant on these intervals. The
number of boundaries in the full partition
$\mathcal P=(\mathcal P_1,\ldots,\mathcal P_p)$ is

$$
K(\mathcal P)=\sum_{j=1}^p(L_j-1).
$$

A split divides one interval and adds one boundary. Because that feature factor is
multiplied by every other feature factor, the split changes every Cartesian cell
containing the interval while preserving separability.

The implementation stores each factor pair in
[backbone and tilt coordinates](model.md#backbone-and-exponential-tilt):

$$
a_{\pm,j,I}=b_{j,I}e^{\pm d_{j,I}},
\qquad
b_{j,I}>0,
\qquad
d_{j,I}\in\mathbb R.
$$

The backbone $b$ controls common magnitude. The tilt $d$ controls the imbalance between
the two branches.

## Starting one grid

The fit uses either L2 weights or Huber weights. Define

$$
w(e)
=
\begin{cases}
1, & \text{L2},\\
1, & \text{Huber and }|e|\leq\kappa,\\
\kappa/|e|, & \text{Huber and }|e|>\kappa,
\end{cases}
\qquad
\kappa=1.345.
$$

Every feature starts with one interval, $b=1$, and $d=0$, so the initial grid is
constant.

If every incoming stage residual is non-negative, grid fitting uses
**positive-only mode**:

$$
\lambda_+=1,
\qquad
\lambda_-=0,
\qquad
d_{j,I}=0.
$$

Only the positive backbone is updated during fitting. Final normalization gives
$\lambda_-$ a small positive numerical floor, so the returned negative branch is
negligible rather than exactly zero.

For mixed-sign stage residuals, let

$$
w_i^{\mathrm{in}}=w\left(R_i^{(\ell)}\right).
$$

The initial constant scales are

$$
\lambda_+
=
\max\left(
\frac{\sum_i w_i^{\mathrm{in}}[R_i^{(\ell)}]_+}{\sum_i w_i^{\mathrm{in}}},
10^{-10}
\right),
\qquad
\lambda_-
=
\max\left(
\frac{\sum_i w_i^{\mathrm{in}}[-R_i^{(\ell)}]_+}{\sum_i w_i^{\mathrm{in}}},
10^{-10}
\right),
$$

where $[z]_+=\max(z,0)$. The **grid residual** after initialization is

$$
e_i^{(0)}=R_i^{(\ell)}-g_0(X_i).
$$

These residuals set the scale of the boundary penalty introduced below.

## How one grid chooses its next action

At step $t$, let $g_t$ be the current grid and define

$$
e_i^{(t)}=R_i^{(\ell)}-g_t(X_i),
\qquad
w_i^{(t)}=w\left(e_i^{(t)}\right).
$$

TSL holds these weights fixed while comparing all actions at this step. Every candidate
uses the same weighted squared loss:

$$
L_t(g)
=
\sum_{i=1}^n
w_i^{(t)}
\left(R_i^{(\ell)}-g(X_i)\right)^2.
$$

For Huber fitting, this is a fixed-weight quadratic approximation to Huber loss. After
accepting an action, TSL recomputes the residuals and weights before the next comparison.

There are three actions:

| Action | Change | Boundary change $\Delta K_a$ |
|---|---|---:|
| Split | Divide one interval and fit both children | $+1$ |
| Boundary refit | Keep a boundary fixed and refit the factors on both sides | $0$ |
| Merge | Remove a boundary and fit one shared backbone/tilt pair on the union | $-1$ |

Each candidate $a$ receives the score

$$
\operatorname{score}(a)
=
D_a-M_a-C_{\partial}\Delta K_a.
$$

Here $D_a$ is the reduction in fixed-weight loss, $M_a$ is the factor-update penalty,
and $C_{\partial}$ is the penalty for one boundary. We derive these terms next.

## Loss reduction and the factor-update penalty

First consider one region affected by an action: either one child of a split or one side
of a boundary refit. Let $S$ be the indices of the training rows in that region. For this
derivation, write

$$
f_{+,i}=f_+(X_i),
\qquad
f_{-,i}=f_-(X_i),
\qquad
e_i=e_i^{(t)},
\qquad
w_i=w_i^{(t)}.
$$

The candidate multiplies the current branch predictions in $S$ by positive factors
$v_+$ and $v_-$:

$$
g_{\mathrm{new}}(X_i)=v_+f_{+,i}-v_-f_{-,i}.
$$

Let

$$
u_+=v_+-1,
\qquad
u_-=v_--1.
$$

Write the prediction change for row $i$ as

$$
\Delta_i
=
g_{\mathrm{new}}(X_i)-g_t(X_i)
=
u_+f_{+,i}-u_-f_{-,i}.
$$

### Where $D_S$ comes from

$D_S$ is the loss before the update minus the loss after the update, restricted to rows
in $S$. The new residual is

$$
e_{i,\mathrm{new}}
=
e_i-\Delta_i.
$$

Because $e_i^2-(e_i-\Delta_i)^2=2e_i\Delta_i-\Delta_i^2$, the loss reduction is

$$
D_S(u)
=
\sum_{i\in S}w_i
\left(2e_i\Delta_i-\Delta_i^2\right).
$$

Substituting $\Delta_i=u_+f_{+,i}-u_-f_{-,i}$ and expanding gives

$$
\begin{aligned}
D_S(u)
={}&
2u_+\sum_{i\in S}w_ie_if_{+,i}
-
2u_-\sum_{i\in S}w_ie_if_{-,i}\\
&-
u_+^2\sum_{i\in S}w_if_{+,i}^2
-
u_-^2\sum_{i\in S}w_if_{-,i}^2\\
&+
2u_+u_-\sum_{i\in S}w_if_{+,i}f_{-,i}.
\end{aligned}
$$

The code computes this expression from five sums:

$$
\begin{aligned}
S_{++}&=\sum_{i\in S}w_i f_{+,i}^2,
&
S_{--}&=\sum_{i\in S}w_i f_{-,i}^2,
&
S_{+-}&=-\sum_{i\in S}w_i f_{+,i}f_{-,i},\\
t_+&=\sum_{i\in S}w_i e_i f_{+,i},
&
t_-&=-\sum_{i\in S}w_i e_i f_{-,i}.
&&
\end{aligned}
$$

The minus signs in $S_{+-}$ and $t_-$ absorb the minus sign in
$g=f_+-f_-$, which gives the compact form below.

Collecting terms gives

$$
D_S(u)
=
2(t_+u_+ + t_-u_-)
-
\left(
S_{++}u_+^2
+2S_{+-}u_+u_-
+S_{--}u_-^2
\right).
$$

Thus $D_S>0$ means that the proposed multipliers reduce the fixed-weight loss on $S$
before penalties.

Write the multipliers as a backbone update $\beta$ and a tilt update $\delta$:

$$
\beta
=
\frac12(\log v_+ + \log v_-),
\qquad
\delta
=
\frac12(\log v_+ - \log v_-).
$$

Equal multipliers change only the backbone; reciprocal multipliers change only the tilt.
The update penalty is

$$
M_S(v_+,v_-)
=
\alpha\beta^2+\tau\delta^2+\rho|\delta|.
$$

Here $\alpha$, $\tau$, and $\rho$ correspond to `alpha`, `tilt_tau`, and `tilt_rho`.
They shrink common log-magnitude changes, shrink tilt changes, and allow an exactly zero
tilt update, respectively.

For a split, both children are measured from their common parent. For a boundary
refit, each new factor is measured from its current value. In both cases,

$$
D_a=D_L+D_R,
\qquad
M_a=M_L+M_R.
$$

A merge uses a different baseline, described below.

## Why one boundary has a penalty

A boundary makes the grid more flexible. For a positive finite `complexity_penalty`,
TSL assigns every boundary the fixed penalty

$$
C_{\partial}
=
\lambda_{\mathrm{cc}}\,
s_0^2\,
\nu_{\partial}\,
\log(\max(n,2)),
$$

where $\lambda_{\mathrm{cc}}$ is the value of `complexity_penalty` and

$$
w_i^{(0)}=w\left(e_i^{(0)}\right),
\qquad
s_0^2
=
\frac1n
\sum_{i=1}^n
w_i^{(0)}\left(e_i^{(0)}\right)^2.
$$

Although $C_{\partial}$ has several symbols, it is only a product of four parts:

- $\lambda_{\mathrm{cc}}$ is the user-controlled strength;
- $s_0^2$ puts the penalty in the same squared-error units as $D_a$;
- $\nu_{\partial}$ counts the interval parameters added by one boundary;
- $\log(\max(n,2))$ increases the penalty with sample size.

The parameter count is

$$
\nu_{\partial}
=
\begin{cases}
1, & \text{positive-only mode},\\
2, & \text{full two-branch mode}.
\end{cases}
$$

A boundary adds one backbone value in positive-only mode, or one backbone and one tilt
value in the full model.

For Gaussian regression, suppose a boundary adds $\nu_{\partial}$ parameters and reduces
the residual sum of squares ($\mathrm{RSS}$) by $\Delta_{\mathrm{RSS}}$. A first-order
BIC calculation favors the boundary when

$$
\Delta_{\mathrm{RSS}}
\gtrsim
\frac{\mathrm{RSS}}{n}\nu_{\partial}\log n.
$$

TSL replaces $\mathrm{RSS}/n$ with the initialized weighted mean squared residual
$s_0^2$ and multiplies by `complexity_penalty`. This is a BIC-inspired calibration.
Because the initialized loss can contain learnable signal and the parameter count omits
threshold search, shrinkage, normalization, and greedy selection, tune
`complexity_penalty` on validation data.

## How the three actions use the score

The score for each action is

| Action | Score |
|---|---:|
| Split | $D_a-M_a-C_{\partial}$ |
| Boundary refit | $D_a-M_a$ |
| Merge | $D_a-M_a+C_{\partial}$ |

The code calls a boundary refit a `resplit`, although the boundary does not move.
Only the factor values on its two sides are refitted.

The highest-scoring eligible action is accepted only when its score exceeds
`min_split_loss` and a relative numerical tolerance.

The boundary penalty depends only on the resulting boundary count. Because the update
penalty depends on the starting factors, two routes to the same partition can receive
different scores.

## How a merge is fitted

Suppose a boundary on feature $j$ separates intervals $L$ and $R$, with factors

$$
a_{\pm,L}=b_Le^{\pm d_L},
\qquad
a_{\pm,R}=b_Re^{\pm d_R}.
$$

If the intervals contain $n_L$ and $n_R$ rows, the merge baseline is their
sample-count-weighted geometric mean:

$$
a_{\pm,\mathrm{ref}}
=
\exp\left(
\frac{n_L\log a_{\pm,L}+n_R\log a_{\pm,R}}
{n_L+n_R}
\right).
$$

This baseline is symmetric, suits positive factors updated multiplicatively, and equals
the common factor when both intervals agree.

TSL evaluates both branch predictions at this baseline and applies the same
two-multiplier calculation used for the other actions. If the fitted one-interval grid is
$g_{\mathrm{merged}}$, then

$$
D_{\mathrm{merge}}
=
L_t(g_t)-L_t(g_{\mathrm{merged}}).
$$

The update penalty $M_{\mathrm{merge}}$ measures the change from the geometric baseline
to the fitted merged factor. The loss reduction accounts for forcing two intervals to
share one factor. Removing the boundary supplies the separate $+C_{\partial}$ score term.

## How multiplier proposals are computed { #the-closed-form-2x2-solver }

The same two-variable solver proposes multipliers for all three actions. Splits and
boundary refits use the current branch predictions and residuals. A merge uses the
predictions and residuals at its geometric baseline.

The exact update penalty is expressed in $\log v_+$ and $\log v_-$ and is not quadratic
in $u_\pm=v_\pm-1$. Since $v_\pm=1+u_\pm$ and $\log(1+u)\approx u$ near zero, the
solver uses

$$
\beta\approx\frac{u_++u_-}{2},
\qquad
\delta\approx\frac{u_+-u_-}{2}.
$$

Using the five regional sums defined above, maximizing the resulting local approximation
to $D_S-M_S$ when $\rho=0$ gives the linear system

$$
A
\begin{pmatrix}u_+\\u_-\end{pmatrix}
=
\begin{pmatrix}t_+\\t_-\end{pmatrix},
$$

where

$$
A
=
\begin{pmatrix}
S_{++}&S_{+-}\\
S_{+-}&S_{--}
\end{pmatrix}
+
\frac14
\begin{pmatrix}
\alpha+\tau&\alpha-\tau\\
\alpha-\tau&\alpha+\tau
\end{pmatrix}.
$$

When $\rho>0$, the solver checks positive, negative, and exactly zero tilt updates and
keeps the best valid proposal.

The proposed multipliers are limited to

$$
v_+,v_-\in[0.05,20].
$$

The action score uses the exact log-coordinate update penalty at this limited proposal.
When the selected action is applied, the code also enforces

$$
b\in[10^{-10},10^{10}],
\qquad
d\in[-10,10].
$$

These are absolute safety bounds. If one activates, the applied update can differ from
the scored proposal.

### Positive-only proposal

In positive-only mode, $f_{-,i}=0$ and the tilt remains zero. The proposal is

$$
u_+
=
\frac{\sum_{i\in S}w_i e_i f_{+,i}}
{\sum_{i\in S}w_i f_{+,i}^2+\alpha}.
$$

After limiting the multiplier, its score uses the exact update penalty
$\alpha\log^2(1+u_+)$.

## Search, boundary budget, and stopping

For each considered feature, `split_try` bounds the number of currently allowed split
positions examined by the default `Random` strategy. `Best` searches all valid
positions. `TopK` samples from the highest-scoring positions. Every strategy rejects
thresholds that violate `min_interval_samples`.

The code calls a grid's boundary count its `fineness`. Because fitting starts with no
boundaries, it equals $K(\mathcal P)$: a split adds one, a merge removes one, and a
boundary refit leaves it unchanged. The `n_iter` parameter is the boundary budget.

A positive finite `complexity_penalty` enables merges and the boundary penalty. At the
boundary budget, further splits are excluded, but eligible merges and boundary
refits may continue. A merge can free room for a later split.

When `complexity_penalty` is zero, negative, or non-finite, merges are disabled, the
boundary penalty is zero, and reaching the budget stops the grid fit.

Two guards prevent repeated boundary refits from cycling:

- the boundary touched by the preceding split or refit is skipped;
- at most five boundary refits may occur consecutively.

A split or merge resets this count. A separate limit stops the full action loop after
$3\,\texttt{n_iter}$ steps. Fitting also stops when no candidate clears the acceptance
threshold or no valid candidate remains.

Histogram binning changes which split thresholds are examined, not how they are scored.
Setting `max_bins` restricts candidates to quantile-bin edges. Prefix sums let the code
score each threshold in constant time, regardless of the number of rows in the interval.
See [GridTensor: histogram binning](../code/grid-tensor.md#histogram-binning).

## Combining the randomized grids

Once every grid stops, fitting moves from grid level to stage level. Each grid fit
receives its own seed and runs in parallel when Rayon is enabled. `Random` and `TopK`
use the seed during partition search; `Best` searches deterministically. Every grid uses
all training rows, and rows are not resampled.

Aggregation places every grid on the union of their split points and centers the aligned
factor coordinates before comparing grid shapes. When `similarity_threshold` requests
trimming, it computes pairwise component-shape distances, chooses the grid with the
smallest total distance to the others as the reference, and discards grids farthest from
it. Otherwise, every grid is retained without computing those distances. Aggregation
then takes geometric means of the retained positive factors, negative factors, and
internal scales, and normalizes the aggregated factor coordinates. See
[Bagging and aggregation](bagging-aggregation.md).

## Refitting all branch coefficients { #coefficient-backfitting }

After aggregation, fitting moves from stage level back to the full model. For stage
$\ell$, define its two branch columns

$$
\phi_{\ell,+}(X_i)=f_{\ell,+}(X_i),
\qquad
\phi_{\ell,-}(X_i)=-f_{\ell,-}(X_i).
$$

These columns include the internal scales $\lambda_{\ell,+}$ and $\lambda_{\ell,-}$ but
exclude the outer OLS branch coefficients. Collect all branch columns through stage
$\ell$:

$$
\Phi_\ell
=
\begin{pmatrix}
\phi_{1,+}&\phi_{1,-}&\cdots&\phi_{\ell,+}&\phi_{\ell,-}
\end{pmatrix}
\in\mathbb R^{n\times2\ell}.
$$

The full model solves

$$
\hat\gamma_\ell
\in
\operatorname*{arg\,min}_{\gamma\in\mathbb R^{2\ell}}
\|Y-\Phi_\ell\gamma\|_2^2.
$$

The coefficients are unconstrained and may have either sign. There is no added intercept
column. The SVD drops singular directions below its relative tolerance and returns the
minimum-norm solution when branch columns are linearly dependent. This joint
least-squares refit updates every completed branch coefficient before the next stage
residual is formed.

!!! note "Where the two scales live"
    `GridTensor.lambda_plus` and `GridTensor.lambda_minus` are positive internal grid
    scales included in $f_{\ell,+}$ and $f_{\ell,-}$. The unconstrained OLS coefficients
    are stored separately as `StagePredictor.scaling_plus` and
    `StagePredictor.scaling_minus`. Prediction applies these outer scalings exactly once;
    the legacy `GridTensor.scaling` field is ignored in two-tensor mode. See the
    [architecture invariants](../code/architecture.md#two-critical-invariants).

## Computational cost

For $R$ stages, $B$ randomized grids per stage, at most $T$ grid actions, $n$
observations, and $p$ features, a rough upper bound for grid search is

$$
\mathcal O(RBTnp).
$$

Randomized grids run in parallel. Cached cumulative sums make split-threshold scores
cheap to evaluate, and histogram binning replaces scans over rows with scans over
candidate bins. Partition alignment, pairwise grid comparisons during aggregation, and
the repeated SVD coefficient refits add separate costs outside this bound.
