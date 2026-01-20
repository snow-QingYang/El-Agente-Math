# Agentic Reader Result
**Paper ID:** InyYuWLWHD
**Issue File:** InyYuWLWHD_0.md
**Status:** success
**Timestamp:** 2026-01-19T23:43:13.503132
**Model:** gpt-5.2
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

Yes — a **mathematical formula issue is indicated** (a missing/undefined case in the cosine-similarity formula).

### What the issue is
The paper defines the layer-wise cosine similarity as

\[
S_{i,j}^{l}=\frac{\langle \Delta w_i^{l}, \Delta w_j^{l}\rangle}{\|\Delta w_i^{l}\|_2 \cdot \|\Delta w_j^{l}\|_2}
\]
(Section **3.3 Layer-wise Cosine Similarity Score**, around the provided snippet / “LINE 145”).

This formula **can be undefined when** \(\|\Delta w_i^{l}\|_2 = 0\) or \(\|\Delta w_j^{l}\|_2 = 0\) (i.e., a client has a zero update vector at a layer). In that case the denominator becomes zero, causing a division-by-zero / undefined cosine similarity. The text does not specify any safeguard (e.g., adding \(\epsilon\) to norms, skipping such clients/layers, or defining similarity as 0).

### Where it appears in the paper
- **Section 3.3**, definition of the cosine similarity matrix \(S^{l}\): the equation for \(S_{i,j}^{l}\) shown above.