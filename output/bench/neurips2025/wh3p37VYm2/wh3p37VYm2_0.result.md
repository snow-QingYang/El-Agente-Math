# Agentic Reader Result
**Paper ID:** wh3p37VYm2
**Issue File:** wh3p37VYm2_0.md
**Status:** success
**Timestamp:** 2026-01-19T07:46:00.969822
**Model:** openai:gpt-5-mini
**Max Iterations:** 10

**Issue Content:**
## LINE 218-222

following expected update:

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { t + 1 } - \mathcal { L } _ { t } ] \leq \nabla _ { \theta _ { t } } \mathcal { L } ^ { T } ( \theta _ { t + 1 } + \theta _ { t } ) } \\ & { - \displaystyle \sum _ { i = 1 } ^ { V } \nabla _ { e _ { i , t } } \mathcal { L } ^ { T } \mathbb { E } ( e _ { i , t + 1 } - e _ { i , t } ) + \frac { \beta } { 2 } \| \Delta \| ^ { 2 } , } \end{array}
$$

Substituting the embedding update based on equation 4 into the smoothness inequality,



$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { t + 1 } - \mathcal { L } _ { t } ] \leq \nabla _ { \theta _ { t } } \mathcal { L } ^ { T } ( \theta _ { t + 1 } - \theta _ { t } ) } \\ & { \quad - \eta \displaystyle \sum _ { i = 1 } ^ { V } \big ( p _ { i } \| \nabla _ { e _ { i , t } } \mathcal { L } \| ^ { 2 } + \lambda e _ { i , t } ^ { T } \nabla _ { e _ { i , t } } \mathcal { L } \big ) + \frac { \beta } { 2 } \| \Delta \| ^ { 2 } , } \end{array}
$$

and noting from the right hand side of the inequality above, $p _ { i }$ plays important role in reduction of the expected loss. However, the dependence on $p _ { i }$ , is coupled with weight decay, which explains why these two parameters are important to study more deeply to draw a conclusion about grokking.

# 4.2 Dataset Splitting Strategies


## Agentic Reader Analysis

Yes — this indicates a formula (typo/sign) issue.

What the reviewer likely noticed
- In the displayed "following expected update" (around "following expected update:"; see document near line 218–222), the first inequality contains the term
  ∇_{θ_t} ℒ^T (θ_{t+1} + θ_t),
  but the smoothness (first-order + quadratic remainder) inequality should have the gradient inner-product with the parameter change Δ = θ_{t+1} − θ_t, i.e. ∇_{θ_t} ℒ^T (θ_{t+1} − θ_t).
- Immediately after, when substituting the embedding update, the paper uses the correct sign:
  ∇_{θ_t} ℒ^T (θ_{t+1} − θ_t)
  (see the substitution displayed a few lines below).

Why this is an issue
- The standard smoothness bound is
  ℒ(θ_{t+1}) − ℒ(θ_t) ≤ ∇ℒ(θ_t)^T(θ_{t+1} − θ_t) + (β/2)‖θ_{t+1} − θ_t‖^2.
  Replacing (θ_{t+1} − θ_t) by (θ_{t+1} + θ_t) is a sign/term error and makes the displayed inequality incorrect or at least inconsistent with the later steps.

Citations (locations in the paper)
- Problematic line: the "following expected update" inequality showing ∇_{θ_t} ℒ^T (θ_{t+1} + θ_t) (around lines ~21840–21890 in the document).
- Correct/consistent use appears in the substitution that follows, where the term is ∇_{θ_t} ℒ^T (θ_{t+1} − θ_t) (the substituted inequality a few lines later, around lines ~22700–22750).

Suggested fix
- Replace (θ_{t+1} + θ_t) by (θ_{t+1} − θ_t) in the first displayed inequality (so it matches the smoothness bound and the subsequent algebra).

If you want, I can produce an exact corrected LaTeX version of the affected display and point to the precise line numbers to quote in a rebuttal.