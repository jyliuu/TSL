# Theory

Two theoretical pillars: an **approximation-rate guarantee** (TSL can represent a rich
function class efficiently) and an honest account of **identifiability** (what the fitted
factors can and cannot be read as).

## Universal approximation via OGA

TSL is analyzed through the **Orthogonal Greedy Approximation** (OGA) framework. The target
class is the Sobolev space of *dominant mixed smoothness* $\mathcal{W}_{\text{mix}}^{(1,1)}$:
functions $f:\mathcal{X}\to\mathbb{R}$ whose every mixed partial derivative
$D_{\boldsymbol{\alpha}} f$ with $\boldsymbol{\alpha}\in\{0,1\}^p$ (i.e.
$\|\boldsymbol{\alpha}\|_\infty\le1$) is $L^1$-integrable. For functions anchored at zero
this reduces to requiring only the highest-order mixed derivative $D_{(1,\dots,1)}f \in L^1$.

!!! abstract "Proposition 2 (approximation rate)"
    Let $\mathcal{X} = [0,1]^p$, assume $P_X$ is absolutely continuous w.r.t. Lebesgue
    measure on $\mathcal{X}$, and consider the positive dictionary $\mathcal{D}_p^+$. Let
    $f_r$ be the OGA approximation to $f$ after $r$ greedy steps over $\mathcal{D}_p^+$. For
    any $f\in\mathcal{W}_{\text{mix}}^{(1,1)}(\mathcal{X})$ anchored at zero
    ($f(\mathbf{x})=0$ whenever some $x_j=0$),

    $$
    \|f - f_r\|_{L^2(P_X)} \le \frac{2\,\|D_{(1,\dots,1)}f\|_{L^1(\mathcal{X})}}{\sqrt{r}}.
    $$

**Proof idea.** For any dictionary $\mathcal{D}$ in a Hilbert space, the $r$-term OGA
approximation of a target $f$ in the variation class
$\mathcal{V}_1(\mathcal{D}) = \{\sum_k \lambda_k g_k : g_k\in\mathcal{D},\ \sum_k|\lambda_k|<\infty\}$
satisfies $\|f-f_r\|_{L^2(P_X)} \le 2\|f\|_{\mathcal{V}_1(\mathcal{D})}/\sqrt{r}$. Applying
this with $\mathcal{D} = \mathcal{D}_p^+$ (the normalized non-negative rank-1 products TSL
fits) and showing $\mathcal{W}_{\text{mix}}^{(1,1)}(\mathcal{X})\subset\mathcal{V}_1(\mathcal{D}_p^+)$
— with the variation norm controlled by the $L^1$ norm of the highest mixed derivative —
transfers Barron's rate.

The rate is $O(1/\sqrt{r})$ and **does not depend explicitly on the dimension** $p$. The
catch is that the *target class tightens* with $p$: requiring mixed differentiability up to
order $p$ becomes more restrictive as $p$ grows.

Here $r$ is the separation rank; with $R$ stages contributing at most two products each,
$r \le 2R$.

## Identifiability and stability

Separable models are **not uniquely identified**. There are three distinct sources of
ambiguity, and TSL only resolves some of them — the factors should be read as *one
admissible representation*, not a uniquely recoverable ground truth.

### 1. Classical scaling and permutation

As in classical tensor decomposition, a fully observed array is identifiable only up to
permutation and scaling. TSL inherits these and remedies them by
[normalization](model.md#normalization-gauge-fixing) (geometric-mean-one factors, scale in
$\lambda_\pm$).

### 2. Non-rectangular support

When $\mathcal{X}$ is non-rectangular, sign and scale ambiguities go beyond the global ones.
For $\mathcal{X} = A\cup B$ with $A=[-1,0]^2$, $B=[0,1]^2$ and

$$
m(x_1,x_2) = \mathbbm{1}_A\,a_1(x_1)a_2(x_2) + \mathbbm{1}_B\,b_1(x_1)b_2(x_2),
$$

three rank-1 factorizations with different sign choices on the disconnected pieces all
agree with $m$ on the observed support:

$$
\begin{aligned}
m^{(1)} &= (\mathbbm{1}_{[-1,0]} a_1 + \mathbbm{1}_{[0,1]} b_1)(\mathbbm{1}_{[-1,0]} a_2 + \mathbbm{1}_{[0,1]} b_2),\\
m^{(2)} &= (-\mathbbm{1}_{[-1,0]} a_1 + \mathbbm{1}_{[0,1]} b_1)(-\mathbbm{1}_{[-1,0]} a_2 + \mathbbm{1}_{[0,1]} b_2),\\
m^{(3)} &= (-\mathbbm{1}_{[-1,0]} a_1 - \mathbbm{1}_{[0,1]} b_1)(-\mathbbm{1}_{[-1,0]} a_2 - \mathbbm{1}_{[0,1]} b_2).
\end{aligned}
$$

**Positivity helps but does not fully resolve this.** Imposing positivity prevents the
cancellation that would occur from averaging, say, the $x_1$-component of $m^{(1)}$ with
that of $m^{(3)}$ — this is what stabilizes [aggregation](bagging-aggregation.md). Yet even
with signs fixed, a positive component can still carry region-dependent scalings: on $A$,
replacing $a_1\mapsto a_1/c$ and $a_2\mapsto c\,a_2$ ($c>0$) leaves predictions unchanged on
$\mathcal{X}$. Only evaluating on the full rectangular span $[-1,1]^2$ — where the rescaling
would change predictions off-support — resolves it.

### 3. Noisy observations

The supervised fitting criterion sees only predictive error against noisy $y^{(i)}$, so
distinct separable representations can attain essentially the same error. Consequently the
latent factors are **not** uniquely recoverable; they are one of many admissible
representations selected by the stochastic fit.

A concrete consequence: when fitting $n_{\text{grids}}$ grids in parallel, bagged grids can
converge to two distinct backbone representations of the *same* stage, populating opposite
ends of the $(\lambda_+,\lambda_-)$ spectrum. The
[align-then-filter](bagging-aggregation.md#the-aggregation-pipeline) aggregation resolves
this by selecting a single canonical representative before averaging — the practical reason
the similarity filter exists.

## Takeaways

- TSL achieves a dimension-free $O(1/\sqrt{r})$ rate on mixed-smoothness targets.
- Read backbone/tilt **shapes** and PD curves as faithful (that is the content of
  [Proposition 1](partial-dependence.md#proposition-1-partial-dependence-decomposition));
  do **not** read the raw factor values as a unique ground-truth decomposition.
