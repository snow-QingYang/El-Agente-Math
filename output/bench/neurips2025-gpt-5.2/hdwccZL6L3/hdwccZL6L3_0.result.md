# Agentic Reader Result
**Paper ID:** hdwccZL6L3
**Issue File:** hdwccZL6L3_0.md
**Status:** success
**Timestamp:** 2026-01-19T23:43:32.404285
**Model:** gpt-5.2
**Max Iterations:** 10

**Issue Content:**
## LINE 94-95

such as KL divergence or binary cross-entropy.

Contrastive Neighbor Embedding. CNE [11] extends NE into the contrastive learning framework   
by training an encoder $f _ { \theta }$ to map $\mathbf { x } _ { i }$ to $\mathbf { y } _ { i } = f _ { \theta } ( \mathbf { x } _ { i } )$ such that the neighborhood structure from a $k$ -NN   
graph is preserved. CNE uses a distance-aware contrastive loss (see Def A.3 in Appendix), framed   
as a binary similarity matching problem. Let $S ^ { d _ { h } } \in \{ 0 , 1 \} ^ { n \times n }$ denote ground-truth neighborhood   
indicators and $S ^ { d _ { l } }$ denote kernel-based similarities in the embedding space. The loss is a weighted   
binary cross-entropy:

$$
\mathcal { L } ( \mathbf { Y } ) = - \sum _ { i , j } \left[ S _ { i j } ^ { d _ { h } } \log S _ { i j } ^ { d _ { l } } + b ( 1 - S _ { i j } ^ { d _ { h } } ) \log ( 1 - S _ { i j } ^ { d _ { l } } ) \right] .
$$

Key Challenges in Decentralized Settings. (C1) CNE, like NE, relies on a full similarity matrix, which   
is unavailable in privacy-sensitive, decentralized settings. (C2) Conventional distributed learning   
captures only intra-client structure, omitting crucial inter-client neighbor information. (C3) Clients   
lack access to global data, leading to incorrect kNN graphs and biased negative sampling, as true   
neighbors may reside on other clients.   
CO-SNE (for Hyperbolic Data). Hierarchical structures in social, biological, and knowledge   
graphs grow exponentially, making Euclidean embeddings unsuitable due to distortion of tree-like   
geometry. Hyperbolic space, with constant negative curvature, naturally models such growth and   
supports hierarchy-aware learning [19, 36, 40] (see Appendix A.3.1). Standard methods like t-SNE   
assume Euclidean geometry and distort global structure when applied to hyperbolic data, collapsing   
depth and relative positioning. CO-SNE [22] extends t-SNE to hyperbolic space (see Def A.4).   
It preserves both local and global structure using distance-aware kernels in hyperbolic geometry:   
$S _ { i j } ^ { \bar { d } _ { h } } = f ( d _ { \mathbb { B } ^ { n } } ( x _ { i } , x _ { j } ) ) , \quad S _ { i j } ^ { \bar { d } _ { l } } = g ( d _ { \mathbb { B } ^ { 2 } } ( y _ { i } , y _ { j } ) )$ , where $f$ is a hyperbolic normal kernel and $g$ is a   
heavy-tailed hyperbolic Cauchy kernel. A regularization term also aligns global depth via norm   
matching. The full objective is:

$$
\mathcal { L } ( \mathbf { Y } ) = \lambda _ { 1 } \cdot \mathcal { D } ( S ^ { d _ { h } } , S ^ { d _ { l } } ) + \lambda _ { 2 } \sum _ { i } ( \rho ( x _ { i } ) - \rho ( y _ { i } ) ) ^ { 2 } ,
$$

where $\rho ( x ) = \| x \|$ and $\mathcal { D }$ is typically $\mathrm { K L }$ divergence.

# 2.1 Problem Formulation

We consider a decentralized system with $M$ clients $\{ { \mathcal { C } } _ { 1 } , \dotsc , { \mathcal { C } } _ { M } \}$ coordinated


## Agentic Reader Analysis

Yes — a **mathematical formula/notation issue** is indicated around **Lines 94–95** in the snippet, due to **inconsistent notation for the low-dimensional distance/similarity superscript**.

### What’s wrong
Just before the CNE paragraph, the paper defines the generic NE objective as
\[
\mathcal { L } ( \mathbf { Y } ) = \sum _ { i , j } \mathcal { D } ( S _ { i j } ^ { d _ { h } } , S _ { i j } ^ { d _ { \ell } } ) ,
\]
(where \(\mathcal{D}\) can be “KL divergence or binary cross-entropy”). Here the low-dimensional similarity uses **\(d_\ell\)** (ell).  

But immediately after, in the CNE section, the weighted binary cross-entropy loss is written using **\(d_l\)** (letter “l”):
\[
\mathcal { L } ( \mathbf { Y } ) = - \sum _ { i , j } \left[ S _ { i j } ^ { d _ { h } } \log S _ { i j } ^ { d _ { l } } + b ( 1 - S _ { i j } ^ { d _ { h } } ) \log ( 1 - S _ { i j } ^ { d _ { l } } ) \right] .
\]

### Why this is a formula issue
Using \(d_\ell\) in one formula and \(d_l\) in the next is ambiguous (especially in math typesetting where \(\ell\) and \(l\) can look similar) and makes it unclear whether these denote the **same low-dimensional distance/similarity** or different quantities.

### Location / citation from the paper
This inconsistency occurs in the formulas in the NE objective and the CNE weighted BCE loss in the provided excerpt (the part beginning “objective minimizes the divergence…” and then “Contrastive Neighbor Embedding… The loss is a weighted binary cross-entropy:”).