# Agentic Reader Result
**Paper ID:** InyYuWLWHD
**Issue File:** InyYuWLWHD_0.md
**Status:** success
**Timestamp:** 2026-01-19T06:47:06.152566
**Model:** openai:gpt-5-mini
**Max Iterations:** 10

**Issue Content:**
## LINE 145

Layer-wise Cosine Similarity Score

Inspired by the Layer-Wise MultiKrum defense method (details in Appendix A), and employing   
cosine similarity as the core measurement(Section 3.1), we introduce a novel metric termed the   
Layer-wise Cosine Similarity Score (LCSS). Diverging from traditional approaches that analyze   
client updates at the network level, LCSS enables independent analysis of updates at each individual   
layer.   
Formally, let $\Delta w _ { i } ^ { l }$ denote the update of client $i$ at the $l$ -th layer, where $l \in [ L ]$ indexes the $L$ layers of   
the global model and $i , j \in { 1 , \dots , N }$ represent client indices among the total of $N$ participants. The   
cosine similarity matrix $S ^ { l }$ at layer $l$ is defined as:

$$
S _ { i , j } ^ { l } = \frac { \langle \Delta w _ { i } ^ { l } , \Delta w _ { j } ^ { l } \rangle } { \| \Delta w _ { i } ^ { l } \| _ { 2 } \cdot \| \Delta w _ { j } ^ { l } \| _ { 2 } }
$$

where $\langle \cdot , \cdot \rangle$ denotes the inner product and $\| \cdot \| _ { 2 }$ is the $\ell _ { 2 }$ -norm. This matrix


## Agentic Reader Analysis

Yes — the snippet exposes mathematical/definition issues (notational edge cases and ambiguities) in the formulas as written.

Relevant locations:
- Cosine similarity definition: S^l_{i,j} (see positions ~12720–13540)
  S_{i,j}^l = <Δw_i^l, Δw_j^l> / (||Δw_i^l||_2 · ||Δw_j^l||_2)
- LCSS definition: LCSS_i^l = (1/m) ∑_{j ∈ N_i} S_{i,j}^l (see positions ~13540–13960)

Problems and why they are issues
1) Division-by-zero / zero-update case (mathematical edge case)
   - The cosine formula divides by the product of norms but the text does not specify what happens if ||Δw_i^l||_2 = 0 (or ||Δw_j^l||_2 = 0). Zero updates can occur in practice (e.g., no change at a layer), which makes S_{i,j}^l undefined.
   - Location: S_{i,j}^l formula (positions ~13120–13540).
   - Fix: explicitly define behavior for zero norms (e.g., define S_{i,j}^l = 0 or 1 in that case, or add a small epsilon in denominator).

2) Ambiguity whether the neighbor set N_i includes the client itself (self-similarity bias)
   - LC SS_i^l averages over j ∈ N_i but the paper does not state whether i ∈ N_i. If i is included then S_{i,i}^l = 1 will bias LCSS upward; if excluded the interpretation is different.
   - Location: LCSS formula (positions ~13540–13960).
   - Fix: state explicitly whether N_i excludes i (typical choice) and whether ties or ranking rules are used.

3) Representation/shape ambiguity for Δw_i^l (implicit flattening / inner-product domain)
   - The inner product and norms assume Δw_i^l are vectors of the same dimension. For a given layer, parameters may be tensors; the paper does not state whether these are flattened or how shapes are matched across clients. This should be clarified to make the formula precise.
   - Location: definition of Δw_i^l and S^l (positions ~12720–13540).
   - Fix: state that layer updates are flattened into vectors before computing inner products, or define the tensor inner-product convention used.

4) Unspecified constraints on m and “nearest neighbors in similarity space”
   - m is used without a formal constraint (e.g., 1 ≤ m ≤ N−1). Also “nearest neighbors in similarity space based on S^l” should clarify whether ordering uses descending similarity (largest S) and whether distance vs. similarity ties are broken.
   - Location: sentence immediately after S^l and before LC SS (positions ~13120–13960).
   - Fix: state the allowed range for m, whether i is excluded, and define ordering/tie-breaking.

Summary recommendation
- These are definitional/edge-case mathematical issues, not algebraic errors in the cosine formula itself. The remedies are straightforward: add a small epsilon or a defined rule for zero norms, explicitly state inclusion/exclusion of self in N_i, clarify vectorization/flattening of layer updates, and state m’s allowable range and ordering convention.

If you want, I can draft precise wording to insert into the paper that resolves these issues and a minimal algorithmic pseudocode handling the edge cases.