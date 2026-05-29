# TSL — Tensor Separation Learning

**TSL** is a glass-box regression model. A fitted model is a sum of *stages*; each stage
is the ordered difference of two non-negative rank-1 products of univariate functions:

$$
\hat{m}(\mathbf{x}) = \sum_{\ell=1}^{R}
\Bigl(
  \lambda_{+}^{(\ell)}\prod_{j=1}^{p}\hat{m}_{+,j}^{(\ell)}(x_j)
  \;-\;
  \lambda_{-}^{(\ell)}\prod_{j=1}^{p}\hat{m}_{-,j}^{(\ell)}(x_j)
\Bigr),
\qquad
\hat{m}_{\pm,j}^{(\ell)} > 0,\quad \lambda_{\pm}^{(\ell)} \ge 0 .
$$

Because each stage is *separable* (a product of one-dimensional factors), its effect on
any single feature can be read off **exactly** from a one-dimensional partial-dependence
curve — without the marginalization artifacts that contaminate additive surrogates. This
is what makes TSL a glass box rather than a model you explain after the fact.

## The two-tensor / backbone–tilt form

Each univariate factor is reparametrized into a **backbone** \(b_j \ge 0\) (a magnitude
gate) and a **tilt** \(d_j \in \mathbb{R}\) (a signed direction):

$$
\hat{m}_{\pm,j}^{(\ell)}(x_j) = b_j^{(\ell)}(x_j)\, e^{\pm d_j^{(\ell)}(x_j)},
\qquad\Longrightarrow\qquad
\hat{m}^{(\ell)}(\mathbf{x}) = 2\, b^{(\ell)}(\mathbf{x})\, \sinh\!\bigl(d^{(\ell)}(\mathbf{x})\bigr).
$$

A near-zero backbone switches the stage off; the summed tilt sets its sign and size. See
[The model](math/model.md) for the full derivation.

## How the codebase is organized

The model is a three-level hierarchy, and the `src/` module tree mirrors it exactly:

| Level | Type | What it is | Docs |
|------|------|------------|------|
| 1 | `GridTensor` | one fitted separable component (backbone/tilt + \(\lambda_\pm\)) | [GridTensor](code/grid-tensor.md) |
| 2 | `StagePredictor` | one boosting stage: a bag of `GridTensor`s + OLS scaling | [StagePredictor](code/stage-predictor.md) |
| 3 | `TSL` | the boosted model: a `Vec<StagePredictor>` summed | [TSL (forest)](code/forest.md) |

The core is the Rust crate `tsl_rust` (library name `tsl`). `tsl-py/` wraps it for Python
with a scikit-learn API ([Python API](code/python-api.md)), and
`tsl-split-evolution-dashboard/` (`tslviz`) visualizes how a fit was built
([Dashboard](code/dashboard.md)).

## Where to start

- **New to TSL?** [Getting started](guides/getting-started.md) — install, fit, predict.
- **Using the model?** The [Python API](code/python-api.md), the
  [Hyperparameters](guides/hyperparameters.md) reference, then [Examples](guides/examples.md).
- **Working on the code?** Start with [Architecture](code/architecture.md) and its two
  critical invariants, then the per-module pages.
- **Want the math?** [Notation](math/index.md) → [The model](math/model.md) →
  [Fitting](math/fitting.md) → [Partial dependence](math/partial-dependence.md) →
  [Theory](math/theory.md).

!!! note "Code-faithful documentation"
    These docs describe **what the implementation actually does**. Where a clean
    mathematical statement is realized by a slightly different but equivalent mechanism in
    code, a neutral *Implementation note* points it out. The mathematics itself is drawn
    from the TSL paper.
