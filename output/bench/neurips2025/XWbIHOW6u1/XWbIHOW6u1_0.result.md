# Agentic Reader Result
**Paper ID:** XWbIHOW6u1
**Issue File:** XWbIHOW6u1_0.md
**Status:** success
**Timestamp:** 2026-01-19T07:02:21.416586
**Model:** openai:gpt-5-mini
**Max Iterations:** 10

**Issue Content:**
## LINE 174-187

Proposition 3, we can 171 apply the iterative algorithm used in SBM algorithm [45, Algorithm 1] to the multi-marginal setting:

$$
\begin{array} { r } { \mathbb { P } ^ { ( 2 n + 1 ) } : = \mathcal { M } ^ { \mathfrak { m } } ( \mathbb { P } ^ { ( 2 n ) } , \mathcal { T } ) , \ \mathbb { P } ^ { ( 2 n + 2 ) } : = \mathcal { R } ^ { \mathfrak { m } } ( \mathbb { P } ^ { ( 2 n + 1 ) } , \mathcal { T } ) , \quad | \mathcal { T } | > 2 . } \end{array}
$$

72 The convergence guarantees proved for the iteration apply equally well to the multi-marginal case.

Proposition 4 (Convergence). $\mathbb { P } ^ { ( n ) } = \mathbb { P } ^ { m S B P }$ of mSBP as $n \uparrow \infty$ with iterative procedure in (13).

# 3.2 Practical Implementation.

In practice, at each iteration 175 $n$ of (13) we approximate the optimal control $v ^ { \star }$ from (11) by a neural 176 network $v _ { \theta }$ . By Girsanov theorem, $\theta$ are chosen to minimize the following training objective function:

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } , \mathcal { T } , \Pi _ { \mathcal { T } } ) = \int _ { 0 } ^ { T } \mathbb { E } _ { \Pi _ { t , \tau } } [ | | \sigma \nabla \log \mathbb { Q } _ { \beta _ { \mathcal { T } } ( t ) | t } ( \mathbf { X } _ { \beta _ { \mathcal { T } } ( t ) } | \mathbf { X } _ { t } ) - v _ { \boldsymbol { \theta } } ( t , \mathbf { X } _ { t } ) | | ^ { 2 } d t ] , } \end{array}
$$

where $\beta _ { \mathcal { T } } ( t ) = \operatorname* { m i n } _ { u } \{ u > t | t \in \mathcal { T } \} \in [ 0 , T ]$ is the most recent time point in $\tau$ after time $t$ . With   
this notation, the SBM can be generalized to the case of multi-marginal constraints. For example,   
when $\mathcal { T } = \{ 0 , T \}$ then (14) reduces to the objective function described in [45].   
The learned Markov control ${ v } _ { \boldsymbol { \theta } \star } \big ( t , \mathbf { x } _ { t } \big )$ then ensures $\begin{array} { r } { \mathbb { P } _ { t } ^ { \theta ^ { \star } } = \Pi _ { t } } \end{array}$ for all $t \in [ 0 , T ]$ . Moreover, prior   
SBM algorithms interleave forward and backward-time Markov projections to re-anchor the terminal   
distribution and prevent bias between $\mathbb { P } _ { T } ^ { ( n ) }$ and $\Pi _ { T }$ accumulate for each $n \in \mathbb { N }$ . In the multi-marginal   
setting, we again build the backward-time Markov projection as in Proposition 2 by gluing the local   
bridge reversals, so that $\mathbb { P } ^ { \star }$ is governed by both SDEs (10) and the corresponding backward dynamics:

$$
\begin{array} { r l } & { d \mathbf { Y } _ { t } ^ { \star } = [ - f _ { T - t } ( \mathbf { Y } _ { t } ^ { \star } ) + \sigma u ^ { \star } ( t , \mathbf { Y } _ { t } ^ { \star } ) ] d t + \sigma d \mathbf { W } _ { t } , \quad \mathbf { Y } _ { 0 } ^ { \star } \sim \Pi _ { T } , } \\ & { \mathrm { w h e r e ~ } u ^ { \star } ( t , \mathbf { y } ) = \sum _ { i = 1 } ^ { k } \mathbf { 1 } _ { ( t _ { i - 1 } , t _ { i } ] } ( t ) \mathbb { E } _ { \Pi _ { t \mid t _ { i - 1 } } } \left[ \nabla \log \mathbb { Q } _ { t \mid t _ { i - 1 } } ( \mathbf { Y } _ { t } \vert \mathbf { Y } _ { t _ { i - 1 } } ) \vert \mathbf { Y } _ { t } = \mathbf { y } \right] , } \end{array}
$$

where the backward optimal control 185 $u ^ { \star }$ in (16) can be approximated with neural network $u _ { \phi }$ where $\phi$ 186 is chosen to minimize the following training objective function with $\begin{array} { r } { \gamma _ { T } ( t ) = \operatorname* { m a x } _ { u } \{ u < t | t \in T \} } \end{array}$ :

$$
\mathcal { L } ( \phi , \mathcal { T } , \Pi _ { \mathcal { T } } ) = \int _ { 0 } ^ { T } \mathbb { E } _ { \Pi _ { t , \tau } } [ | | \sigma \nabla \log \mathbb { Q } _ { t | \gamma _ { \tau } ( t ) } ( \mathbf { Y } _ { t } | \mathbf { Y } _ { \gamma _ { \tau } ( t ) } ) - u _ { \phi } ( t , \mathbf { Y } _ { t } ) | | ^ { 2 } d t ] .
$$

# 187 4 Multi-Marginal Schrodinger Bridge Matching ¨

A na¨ıve extension of the standard SBM using, multi-marginal projections ${ \mathcal { R } } ^ { \mathrm { m m } }$ and ${ \mathcal { M } } ^ { \mathrm { m m } }$ in Sec 3,   
encounters significant limitations not present in the traditional two-endpoint setting. In such an   
extension, each iteration typically enforces marginal constraints only at the global endpoints $( \rho _ { 0 } , \rho _ { T } )$ .   
The multi-marginal coupling $\Pi _ { T } ^ { ( n ) }$ at each iteration $n$ of (13) is then derived by propagating the   
projected dynamics in (10) or (15) solely from these end points $\rho _ { 0 }$ or $\rho _ { T }$ , respectively.   
This approach leads to critical issues specific to the multi-marginal


## Agentic Reader Analysis

Yes — there is a formula/notation issue in the paper.

What is wrong
- The definitions of the indexing maps β_T(t) and γ_T(t) use the wrong dummy variable / set membership in the set-builder, and the accompanying English is inconsistent (saying "most recent ... after time t" is contradictory). Concretely:

  - Forward loss (paper's displayed forward objective, referred to around (14)). The paper writes
    β_{T}(t) = operatorname*{min}_u { u > t | t ∈ T } ∈ [0,T]
    (see the displayed forward loss and the line immediately after it: "where β_{T}(t) = ... is the most recent time point in τ after time t".)

    Problems:
    - The condition inside the set-builder uses "t ∈ T" instead of "u ∈ T" (the quantified variable must be u). As written it is ill-formed.
    - The English phrase "most recent time point in τ after time t" is ambiguous/contradictory: "most recent" normally means the nearest prior (≤ t), while "after time t" means the next future time (> t). Given the forward loss uses Q_{β_T(t)|t}(X_{β_T(t)} | X_t) (transition from t to β_T(t)), the intended meaning is likely the next time point in T after t, i.e. β_T(t) = min{ u ∈ T : u > t } (or possibly min{ u ∈ T : u ≥ t } depending on convention).

  - Backward loss (displayed backward objective, see where γ is defined near the backward objective). The paper writes
    γ_{T}(t) = operatorname*{max}_u { u < t | t ∈ T }

    Problems:
    - Same syntactic error: the set-builder wrongly tests "t ∈ T" instead of "u ∈ T".
    - The intended meaning is presumably the nearest earlier checkpoint in T, i.e. γ_T(t) = max{ u ∈ T : u < t } (or max{ u ∈ T : u ≤ t } depending on convention).

Where to find it in the manuscript (citations)
- Forward loss and β definition: see the displayed forward training objective and the line immediately after it (paper around the paragraph labeled "3.2 Practical Implementation", equations around (14); captured in the paper text: "where β_{𝒯}(t) = operatorname*{min}_u { u > t | t ∈ 𝒯 } ∈ [0,T] is the most recent time point in τ after time t.").
- Backward loss and γ definition: see the displayed backward training objective and the line introducing γ (near the backward objective; the paper shows "γ_{T}(t) = operatorname*{max}_u { u < t | t ∈ T }").

Suggested correction
- Replace the definitions by well-formed set-builder expressions, for example:
  - β_{T}(t) := min{ u ∈ T : u > t }  (the next time point in T after t)
  - γ_{T}(t) := max{ u ∈ T : u < t }  (the previous time point in T before t)

- Fix the English wording to match the intended direction (e.g., "β_T(t) is the next time point in T after t" or "γ_T(t) is the most recent time point in T before t").

Conclusion
- These are formula/notation errors that should be corrected for mathematical clarity and correctness.