# Agentic Reader Result
**Paper ID:** 6JlzhISecd
**Issue File:** 6JlzhISecd_1.md
**Status:** success
**Timestamp:** 2026-01-19T06:36:40.639544
**Model:** openai:gpt-5-mini
**Max Iterations:** 10

**Issue Content:**
## LINE 119-121

SGDA) updates. Take $\alpha > 0 , \beta > 0$ as the step sizes, we have

$$
\begin{array} { r } { \mathbf { x } ^ { t + 1 } = \mathbf { x } ^ { t } - \alpha \nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } ^ { t } , \lambda ^ { t } ; \boldsymbol { \xi } ^ { t } ) , \lambda ^ { t + 1 } = \lambda ^ { t } + \beta \nabla _ { \lambda } \mathcal { L } ( \mathbf { x } ^ { t } , \lambda ^ { t } ; \boldsymbol { \xi } ^ { t } ) . } \end{array}
$$

Taking the variable substitution 119 $\widehat { \lambda } : = \mathbf { A } ^ { \top } \lambda$ yields the following recursion:

FSPDA-SA: for any $t \geq 0$ and any $i \in [ n ]$ ,

$$
\begin{array} { r l } & { \mathbf { x } _ { i } ^ { t + 1 } = \mathbf { x } _ { i } ^ { t } - \alpha \nabla f _ { i } ( \mathbf { x } _ { i } ^ { t } ; \boldsymbol { \xi } _ { i } ^ { t } ) - \eta \widehat { \lambda } _ { i } ^ { t } + \gamma \sum _ { j \in { \mathcal { N } } _ { i } ( \boldsymbol { \xi } _ { a } ^ { t } ) } \mathbf { C } _ { i j } ( \boldsymbol { \xi } _ { a } ^ { t } ) ( \mathbf { x } _ { j } ^ { t } - \mathbf { x } _ { i } ^ { t } ) , } \\ & { \widehat { \lambda } _ { i } ^ { t + 1 } = \widehat { \lambda } _ { i } ^ { t } + \beta \sum _ { j \in { \mathcal { N } } _ { i } ( \boldsymbol { \xi } _ { a } ^ { t } ) } \mathbf { C } _ { i j } ( \boldsymbol { \xi } _ { a } ^ { t } ) ( \mathbf { x } _ { j } ^ { t } - \mathbf { x } _ { i } ^ { t } ) . } \end{array}
$$



Note that 21 $\mathbf { x } ^ { 0 } , \widehat { \lambda } ^ { 0 }$ can be initialized arbitrarily.

FSPDA-STORM Algorithm. The second variant of FSPDA reduces the variance of the stochastic   
gradient term in (5) using the recursive momentum variance reduction technique [Cutkosky and   
Orabona, 2019]. Herein, the key idea is to utilize a control variate in estimating the (primal-dual)   
gradients of $\mathcal { L } ( \mathbf { x } , \lambda )$ . Take $\alpha , \beta > 0$ and $a _ { x } , a _ { \lambda } \in [ 0 , 1 ]$ as the momentum parameters, we have   
$\mathbf { x } ^ { t + 1 } = \mathbf { x } ^ { t } - \alpha \mathbf { m } _ { x } ^ { t } , \lambda ^ { t + 1 } = \lambda ^ { t } + \beta \mathbf { m } _ { \lambda } ^ { t }$ as the primal-dual updates, and

$$
\begin{array} { r l } & { \mathbf { m } _ { x } ^ { t + 1 } = \nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } ^ { t + 1 } , \boldsymbol { \lambda } ^ { t + 1 } ; \boldsymbol { \xi } ^ { t + 1 } ) + ( 1 - a _ { x } ) ( \mathbf { m } _ { x } ^ { t } - \nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } ^ { t } , \boldsymbol { \lambda } ^ { t } ; \boldsymbol { \xi } ^ { t + 1 } ) ) , } \\ & { \mathbf { m } _ { \lambda } ^ { t + 1 } = \nabla _ { \lambda } \mathcal { L } ( \mathbf { x } ^ { t + 1 } , \boldsymbol { \lambda } ^ { t + 1 } ; \boldsymbol { \xi } ^ { t + 1 } ) + ( 1 - a _ { \lambda } ) ( \mathbf { m } _ { \lambda } ^ { t } - \nabla _ { \lambda } \mathcal { L } ( \mathbf { x } ^ { t } , \boldsymbol { \lambda } ^ { t } ; \boldsymbol { \xi } ^ { t + 1 } ) ) . } \end{array}
$$

The aim of $\mathbf { m } _ { r } ^ { t + 1 }$ is to estimate $\nabla _ { \mathbf x } \mathcal L ( \mathbf x ^ { t + 1 } , \lambda ^ { t + 1 } )$ . Now


## Agentic Reader Analysis

Yes — this snippet indicates a mathematical/formula inconsistency in the paper.

What the paper states
- The stochastic gradients are defined (Sec. around positions 14880–15580) as
  ∇_x L(x,λ;ξ) = ∇ f(x;ξ) + \tilde η A^T λ + \tilde γ A^T A(ξ) x,
  ∇_λ L(x,λ;ξ) = \tilde η A(ξ) x.
  (see content at ~positions 14880–15580)

- The SGDA (primal–dual) update is written (around positions ~17566–17724) as
  x^{t+1} = x^t − α ∇_x L(x^t,λ^t;ξ^t),
  λ^{t+1} = λ^t + β ∇_λ L(x^t,λ^t;ξ^t).

- Immediately after (FSPDA-SA recursion, at ~positions 18150–18650) the per-agent updates are given as
  x_i^{t+1} = x_i^t − α ∇ f_i(x_i^t;ξ_i^t) − η \widehat λ_i^t + γ ∑_{j∈N_i(ξ_a^t)} C_{ij}(ξ_a^t) (x_j^t − x_i^t),
  \widehat λ_i^{t+1} = \widehat λ_i^t + β ∑_{j∈N_i(ξ_a^t)} C_{ij}(ξ_a^t) (x_j^t − x_i^t),
  with \widehat λ := A^T λ. (see content at ~positions 18150–18650)

Why this is inconsistent / what the issue is
- Expanding the SGDA update using the defined gradients yields (per agent)
  x_i^{t+1} = x_i^t − α ∇ f_i(x_i^t;ξ_i^t) − α \tilde η \widehat λ_i^t − α \tilde γ ∑_{j} C_{ij}(ξ)(x_j^t − x_i^t),
  and
  \widehat λ^{t+1} = \widehat λ^t + β \tilde η ∑_{j} C_{ij}(ξ)(x_j^t − x_i^t).

  That is, the neighbor-coupling term and the λ-coupling term should carry the multiplicative factors coming from α (and \tilde η, \tilde γ), and the sign of the A^T A(ξ) contribution in x-update should be negative (because x-update is x − α ∇_x L and ∇_x L contains +\tilde γ A^T A x).

- The FSPDA-SA recursion as written uses −η \widehat λ_i^t (instead of −α \tilde η \widehat λ_i^t) and +γ ∑ C_{ij}(x_j − x_i) (instead of −α \tilde γ ∑ C_{ij}(x_j − x_i)), and the \widehat λ-update has +β (instead of +β \tilde η). In other words:
  - factors α and \tilde η / \tilde γ appear to be missing or have been silently redefined (η, γ appear with no mapping shown),
  - the sign of the coupling term in the x-update is flipped (paper shows +γ · sum, expansion predicts −α \tilde γ · sum),
  - the \widehat λ update is missing the \tilde η multiplier (should be β \tilde η ...).

- The paper earlier used the parameters \tilde η, \tilde γ in the gradient definitions but later presents η, γ in the recursion without stating how they relate to the tilded parameters (or to α). That creates an ambiguity/inconsistency: either the authors intended to absorb α and the tildes into new symbols (they must state that explicitly), or the signs/factors shown are simply incorrect.

Conclusion / recommended fix
- This is a formula/notation error: the per-agent recursion (FSPDA-SA) is inconsistent with the SGDA update and the defined gradients. The authors should either:
  1) explicitly state the reparameterization (e.g. define η := α \tilde η and γ := −α \tilde γ and show how β and \tilde η relate), or
  2) correct the recursion to include the correct multiplicative factors and signs:
     x_i^{t+1} = x_i^t − α ∇ f_i(·) − α \tilde η \widehat λ_i^t − α \tilde γ ∑ C_{ij}(x_j − x_i),
     \widehat λ_i^{t+1} = \widehat λ_i^t + β \tilde η ∑ C_{ij}(x_j − x_i).

Citations to paper locations
- Gradient definitions: ~positions 14880–15580 (equations for ∇_x L and ∇_λ L).
- SGDA update (x^{t+1}, λ^{t+1}): ~positions 17566–17724.
- FSPDA-SA recursion (per-agent formulas): ~positions 18150–18650.

If you want, I can produce the corrected per-agent recursion explicitly (using either the original tilded symbols or by defining η,γ in terms of α,\tilde η,\tilde γ) so the paper text can be fixed unambiguously.