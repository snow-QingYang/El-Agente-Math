# Agentic Reader Result
**Paper ID:** Gsj3oN2Ipy
**Issue File:** Gsj3oN2Ipy_0.md
**Status:** success
**Timestamp:** 2026-01-19T23:43:29.440930
**Model:** gpt-5.2
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

Yes — a **mathematical formula/notation issue** is indicated around **Lines 174–175**, and it shows up as **inconsistent or incorrect notation/arguments in the equations**, relative to the surrounding text.

### 1) Mismatch between the set name in text vs. formula (likely typo)
- **Formula defines** the candidate index set as **\(\mathcal{I}\)**:
  \[
  i^{*}=\arg\min_{i\in\mathcal{I}} \mathcal{H}(\cdot),\quad
  \mathcal{I}=\left\{j:\arg\max_{c} p(y=c\mid \{\mathbf{X}_{[N]}\}_{j},\mathbf{p}_{\mathrm{ada}})=\hat{y}\right\}.
  \]
  (Lines **174–175**, patch-selection equation block)

- But the **text immediately after** says: “Then, from the subset **\(\mathcal{T}\)** of patches whose predicted label matches \(\hat{y}\) …” (same location).
  
This is a direct **notation inconsistency**: the subset is called \(\mathcal{I}\) in the formula but \(\mathcal{T}\) in the text.

### 2) Incorrect/missing conditioning in entropy term (formula inconsistency)
In the same equation block (Lines **174–175**), the entropy is written as:
\[
\mathcal{H}\Big(p(\{\mathbf{X}_{[N]}\}_{i},\mathbf{p}_{\mathrm{ada}})\Big).
\]
But elsewhere, probabilities are written as **class probabilities conditioned on the input**, e.g.
\[
p(y=c\mid \{\mathbf{X}_{[N]}\}_{j},\mathbf{p}_{\mathrm{ada}}),
\]
and the text says “lowest prediction entropy,” which normally means entropy of the **class distribution** \(p(y\mid \cdot)\), not \(p(\text{inputs})\).

So the entropy argument should likely be something like:
\[
\mathcal{H}\Big(p(y\mid \{\mathbf{X}_{[N]}\}_{i},\mathbf{p}_{\mathrm{ada}})\Big),
\]
i.e., it is **missing \(y\mid\)** (and/or missing the class dimension), making the formula ambiguous/wrong as written.

### 3) Another likely formula typo in the final prediction (problematic exponent/prime placement)
The final prediction equation (around **Line 175**) is:
\[
P_{\mathrm{final}} = p(\mathbf{y}\mid \mathbf{v},\mathbf{p}_{\mathrm{ada}}) + \mathrm{Softmax}\big(\mathbf{M}^{\prime^{\mathrm{ada}}\top}\mathbf{v}\big).
\]
The notation \(\mathbf{M}^{\prime^{\mathrm{ada}}\top}\) is malformed/unclear (prime applied to an exponent). The text below refers to
\(\mathbf{M}^{\mathrm{\prime}\mathrm{ada}}\),
suggesting the intended form is probably something like \((\mathbf{M}^{\prime \mathrm{ada}})^\top\) or \((\mathbf{M}^{\prime})^{\mathrm{ada}\top}\).

**Cited location:** the patch-selection equation block and the \(P_{\mathrm{final}}\) equation shown in the snippet (Lines **174–176** in the paper).