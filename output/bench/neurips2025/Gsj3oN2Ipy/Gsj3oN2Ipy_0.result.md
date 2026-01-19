# Agentic Reader Result
**Paper ID:** Gsj3oN2Ipy
**Issue File:** Gsj3oN2Ipy_0.md
**Status:** success
**Timestamp:** 2026-01-19T06:46:28.832178
**Model:** openai:gpt-5-mini
**Max Iterations:** 10

**Issue Content:**
## LINE 174-175

views $\mathbf { X } _ { [ N ] }$ according to vision-text similarity:

$$
\begin{array} { r } { \hat { y } = \arg \underset { c } { \operatorname* { m a x } } \bar { p } ( y = c \mid \mathbf { X } _ { [ n ] } , \mathbf { p } _ { \mathrm { a d a } } ) , \quad i ^ { * } = \arg \underset { i \in \mathcal { I } } { \operatorname* { m i n } } \mathcal { H } ( p ( \{ \mathbf { X } _ { [ N ] } \} _ { i } , \mathbf { p } _ { \mathrm { a d a } } ) ) , } \\ { \mathrm { w h e r e } \quad \mathcal { I } = \left\{ j : \arg \underset { c } { \operatorname* { m a x } } p ( y = c \mid \{ \mathbf { X } _ { [ N ] } \} _ { j } , \mathbf { p } _ { \mathrm { a d a } } ) = \hat { y } \right\} . } \end{array}
$$

We first obtain a confident prediction $\hat { y }$ by aggregating predictions over the selected subset $\mathbf { X } _ { [ n ] }$   
using the adapted prompt $\mathbf { p } _ { \mathrm { a d a } }$ . Then, from the subset $\mathcal { T }$ of patches whose predicted label matches   
$\hat { y }$ , we select the patch $\mathbf { X } _ { i ^ { * } }$ with the lowest prediction entropy. This avoids directly selecting the   
lowest-entropy patch from the entire set $\mathbf { X } _ { [ N ] }$ , which may include highly confident but irrelevant   
patches. Finally, we insert the selected patch into the corresponding memory slot $\mathcal { M } _ { \hat { y } }$ . If the memory   
is at full capacity, we remove the patch with the highest entropy among the existing entries and the   
current candidate.   
These three steps for each test image constitute a round of mutual promotion between the tunable   
textual prompt and the evolving visual memory. Afterward, we obtain two predictions for the current   
test image: one from the optimized prompt and one from the updated memory $\mathcal { M } ^ { \prime }$ . We combine   
them to produce the final prediction:

$$
P _ { \mathrm { f i n a l } } = P _ { \mathrm { p t } } + P _ { \mathrm { m e m o } } = p ( \mathbf { y } \mid \mathbf { v } , \mathbf { p } _ { \mathrm { a d a } } ) + \mathrm { S o f t m a x } ( \mathbf { M } ^ { \prime ^ { \mathrm { a d a } } \top } \mathbf { v } ) ,
$$

where 175 $P _ { \mathrm { p t } } , P _ { \mathrm { m e m o } } \in \mathbb { R } ^ { C }$ . The prediction $P _ { \mathrm { m e m o } }$ is obtained via similarity-based classification, as in the memory retrieval step, and 176 $\mathbf { M } ^ { \mathrm { \prime } \mathrm { a d a } }$ is computed from the updated memory following Eqs. 3 and 4.

It is worth noting that we perform only a single


## Agentic Reader Analysis

Yes — a formula issue is present.

What the reviewer snippet shows
- The paper defines the final prediction as
  P_final = P_pt + P_memo = p(y | v, p_ada) + Softmax( M′^{ada^T} v ).
  (displayed in the paper in the paragraph that begins “These three steps …” and the displayed equation for P_final.)

Why this is a problem
1. Adding two probability vectors without normalization or explanation:
   - p(y | v, p_ada) is presented as a probability vector (a predictive distribution).
   - Softmax( M′^{ada^T} v ) is also a probability vector (Softmax of similarity scores).
   - The equation adds these two vectors elementwise. The sum of two probability vectors is not a probability distribution (its entries will generally sum to 2), yet no normalization, weighting, or justification is given. Standard practice is to (a) combine logits and then apply Softmax, (b) average/weight the probability vectors and (optionally) renormalize, or (c) explicitly state why an unnormalized sum is used and how it is interpreted. The paper gives none of these clarifications, so the combination is mathematically inconsistent as written.

2. Notational/formatting ambiguity in the Softmax term:
   - The matrix/transposition/ superscript placement is poorly rendered: "M′^{ada^T}" (in the PDF: \mathbf{M}^{\prime^{\mathrm{ada}}\top}\mathbf{v}) is ambiguous. It should be written clearly as, for example, (M'^{ada})^T v or (M'^{\mathrm{ada}})^\top v, to avoid confusion about whether the transpose applies to the prime, to the ada superscript, or to the whole matrix. This matters for interpretation of dimensions and the Softmax input.

Minor readability issue (not fatal but confusing)
- In the earlier patch-selection equation the paper uses notation like {X_{[N]}}_i to denote the i-th patch — this is nonstandard/awkward but interpretable. The main mathematical issue is the unnormalized addition in P_final.

Relevant locations (where the formulas appear)
- Patch-selection display (hat y and i^*): displayed equation block beginning “\hat{y} = …, i^{*} = …” (viewable in the paragraph that starts “We first obtain a confident prediction \hat{y} …”).
- Final prediction equation: the displayed equation
  P_{final} = P_{pt} + P_{memo} = p(y | v, p_{ada}) + Softmax( M'^{ada^T} v )
  (appears immediately after the paragraph describing the three steps and just before “where P_pt, P_memo ∈ R^C …”).

Suggested fixes
- Clarify whether P_pt and P_memo are probabilities or logits. If they are probabilities, either:
  - average/weight them and (if desired) renormalize: P_final = Normalize( w1 * P_pt + w2 * P_memo ), or
  - combine logits (before Softmax) and then apply Softmax: P_final = Softmax( logits_pt + logits_memo ).
- Fix the matrix/transposition notation to an unambiguous form, e.g. Softmax( (M'^{\mathrm{ada}})^\top v ).

If you want, I can produce an exact suggested corrected equation (with weighting or normalization) and a cleaned-up notation version to replace the paper's current line.