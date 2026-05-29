# The model

## Separation models

A **separation model** is a function representable as a sum of separable rank-1 products:

$$
m(\mathbf{x}) = \sum_{\ell=1}^{r} s_\ell \prod_{j=1}^{p} g_j^{(\ell)}(x_j),
\qquad s_\ell\in\mathbb{R},\ \ g_j^{(\ell)}:\mathbb{R}\to\mathbb{R}.
$$

The number of separable products is the **separation rank**. TSL learns such a model, but
differs from prior separation-learning work in two ways: it estimates the model
**stagewise** (fitting each stage on the residual of the previous ones, rather than a
fixed-rank joint optimization), and it constrains every factor to be **positive**, fitting
an *ordered difference of two positive rank-1 tensors* per stage.

## The TSL estimator

TSL learns $R$ stages:

$$
\hat{m}(\mathbf{x}) = \sum_{\ell=1}^{R}
\Bigl(
  \lambda_{+}^{(\ell)}\prod_{j=1}^{p}\hat{m}_{+,j}^{(\ell)}(x_j)
  \;-\;
  \lambda_{-}^{(\ell)}\prod_{j=1}^{p}\hat{m}_{-,j}^{(\ell)}(x_j)
\Bigr),
$$

subject to **positive univariate components** $\hat{m}_{+,j}^{(\ell)}(x_j) > 0$ and
$\hat{m}_{-,j}^{(\ell)}(x_j) > 0$ for all $j\in[p]$, with non-negative stage scalars
$\lambda_{\pm}^{(\ell)} \ge 0$. For brevity, define the **signed products** and the **stage
predictor**:

$$
\hat{m}_{\pm}^{(\ell)}(\mathbf{x}) \coloneqq \lambda_{\pm}^{(\ell)}\prod_{j=1}^p \hat{m}_{\pm,j}^{(\ell)}(x_j),
\qquad
\hat{m}^{(\ell)}(\mathbf{x}) \coloneqq \hat{m}_{+}^{(\ell)}(\mathbf{x}) - \hat{m}_{-}^{(\ell)}(\mathbf{x}).
$$

Each stage contributes at most two separable products, so the total separation rank is at
most $2R$. A low separation rank aids interpretability: only a handful of products are
needed to represent the model.

!!! note "Implementation note — two-tensor storage"
    A fitted `GridTensor` stores per-feature `backbone_values` ($b_j\ge 0$) and
    `tilt_values` ($d_j\in\mathbb{R}$) plus scalars `lambda_plus`/`lambda_minus`, rather
    than the raw factors $\hat{m}_{\pm,j}$. The two representations are equivalent (see
    [the backbone–tilt map](#backbone-and-exponential-tilt) below). This is the
    "two-tensor form" referred to throughout the codebase. See
    [GridTensor](../code/grid-tensor.md).

## Why positivity

In an unconstrained product $\prod_j h_j(x_j)$ with $h_j\in\mathbb{R}$, the sign of the
whole product is the product of all the component signs — so a "positive $h_2$" need not
mean a positive contribution if $h_1<0$ on the region where the pair is observed. The same
factor then flips its apparent role on different parts of the support.

Constraining $\hat{m}_{\pm,j}^{(\ell)} > 0$ removes this ambiguity: each component
unambiguously **amplifies** (or, near zero, **gates off**) its product, and any signed
behavior is carried by the *difference* of the two branches, not by sign flips inside a
product. Positivity also stabilizes the [aggregation](bagging-aggregation.md) of
independently fitted components — averaging two positive factors cannot cancel them.

!!! warning "Positivity is enforced in the solver"
    The per-node solver clamps updates so that bin values stay in $[v_{\min}, v_{\max}]$
    with $v_{\min}>0$, keeping $b\ge 0$ and $\lambda_\pm\ge 0$. This invariant is tested
    (`tests/forest.rs`, `tests/stage1_positive_only.rs`).

## Backbone and exponential tilt

On a given feature the pair $(\hat{m}_{+,j}^{(\ell)}, \hat{m}_{-,j}^{(\ell)})$ is often
nearly *balanced* (the feature acts as a pure scale knob, factoring out of the difference)
or strongly *imbalanced* (it tilts the stage toward one branch). To make these two roles
explicit, reparametrize each pair with a positive **backbone** $b_j^{(\ell)}(x_j)>0$
(shared magnitude) and a **tilt** $d_j^{(\ell)}(x_j)\in\mathbb{R}$ (signed imbalance):

$$
\hat{m}_{\pm,j}^{(\ell)}(x_j) = b_j^{(\ell)}(x_j)\, e^{\pm d_j^{(\ell)}(x_j)}.
$$

This map is a **bijection**, with closed-form inverse

$$
b_j^{(\ell)} = \sqrt{\hat{m}_{+,j}^{(\ell)}\,\hat{m}_{-,j}^{(\ell)}},
\qquad
d_j^{(\ell)} = \tfrac{1}{2}\log\!\bigl(\hat{m}_{+,j}^{(\ell)}/\hat{m}_{-,j}^{(\ell)}\bigr).
$$

The same parametrization absorbs the stage scalars into a stage-level backbone scale
$b_0^{(\ell)} = \sqrt{\lambda_{+}^{(\ell)}\lambda_{-}^{(\ell)}}$ and tilt intercept
$d_0^{(\ell)} = \tfrac{1}{2}\log(\lambda_{+}^{(\ell)}/\lambda_{-}^{(\ell)})$, so the
per-feature pieces aggregate cleanly at the stage level:

$$
b^{(\ell)}(\mathbf{x}) = b_0^{(\ell)}\prod_{j=1}^p b_j^{(\ell)}(x_j),
\qquad
d^{(\ell)}(\mathbf{x}) = d_0^{(\ell)} + \sum_{j=1}^p d_j^{(\ell)}(x_j).
$$

### The sinh form

Substituting the backbone–tilt map into the stage predictor collapses the ordered
difference into a single clean expression:

$$
\hat{m}(\mathbf{x}) = 2\sum_{\ell=1}^{R} b^{(\ell)}(\mathbf{x})\, \sinh\!\bigl(d^{(\ell)}(\mathbf{x})\bigr).
$$

This cleanly separates **activity** from **direction**:

- The **backbone** $b^{(\ell)}(\mathbf{x})>0$ is an *activity gate*: $b_0^{(\ell)}$ sets the
  stage's overall scale, and a near-zero per-feature backbone $b_j^{(\ell)}(x_j)$ switches
  the stage off in that region.
- The **tilt** $d^{(\ell)}(\mathbf{x})$ sets the *signed direction* through the strictly
  increasing odd function $\sinh$. Holding the backbone fixed, increasing
  $d_j^{(\ell)}(x_j)$ always increases the stage contribution; $d>0$ tilts toward the $(+)$
  product, $d<0$ toward the $(-)$ product, and $d=0$ is local branch balance. So
  $d_j^{(\ell)}$ reads as an additive imbalance score, with $\sinh$ only providing a
  monotone response-scale transformation.

## Normalization (gauge fixing)

The backbone–tilt form is invariant to feature-wise rescalings of $b_j^{(\ell)}$ and
constant shifts of $d_j^{(\ell)}$. TSL fixes the representative that satisfies

$$
\mathbb{E}\!\left[\log b_j^{(\ell)}(X_j)\right] = 0,
\qquad
\mathbb{E}\!\left[d_j^{(\ell)}(X_j)\right] = 0,
\qquad j\in[p],
$$

equivalently $\mathbb{E}[\log \hat{m}_{+,j}^{(\ell)}(X_j)] = \mathbb{E}[\log \hat{m}_{-,j}^{(\ell)}(X_j)] = 0$.
Each univariate branch factor then has **geometric mean one**, and all branch-level scale
is carried by $\lambda_{\pm}^{(\ell)}$. The empirical version of this normalization is
applied in code by `l2_identify` (after a fit) and before averaging bagged grids; see
[GridTensor → identification](../code/grid-tensor.md#identification) and
[Bagging & aggregation](bagging-aggregation.md).

This fixes a *within-stage* representative but does not remove the support-induced
non-identifiability discussed in [Theory](theory.md#identifiability-and-stability).

## Where to go next

- [Fitting](fitting.md) — how a stage (and the whole model) is learned.
- [Partial dependence](partial-dependence.md) — why a 1D PD curve recovers a factor shape exactly.
- [GridTensor](../code/grid-tensor.md) — the struct that stores a fitted stage component.
