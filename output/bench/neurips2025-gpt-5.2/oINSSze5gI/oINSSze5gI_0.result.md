# Agentic Reader Result
**Paper ID:** oINSSze5gI
**Issue File:** oINSSze5gI_0.md
**Status:** success
**Timestamp:** 2026-01-19T23:44:01.848776
**Model:** gpt-5.2
**Max Iterations:** 10

**Issue Content:**
## LINE 140-141

bound $\mathcal { L } ( \Psi , \theta ; \mathcal { G } )$ can be expressed as follows:

$$
\mathcal { L } ( \Psi , \theta ; \mathcal { G } ) = \mathbb { E } _ { q \ast ( \mathbf { Z } | \mathcal { G } ) } \left[ \log \frac { p _ { \theta } ( \mathbf { Z } , \mathcal { G } ) } { q _ { \Psi } ( \mathbf { Z } | \mathcal { G } ) } \right] = \mathbb { E } _ { q \ast ( \mathbf { Z } | \mathcal { G } ) } \left[ \log p _ { \theta } ( \mathbf { Z } | \mathcal { G } ) \right] - \mathrm { K L } \left( q _ { \Psi } ( \mathbf { Z } | \mathcal { G } ) \| p ( \mathbf { Z } ) \right) .
$$

Variational inference enhances the model’s robustness and generalization capabilities [26, 27]. How  
ever, due to the differing distributions between noisy heterogeneous graph data and standard graph   
data, the obtained distribution tends to align with the noisy distribution, potentially misleading the   
GNN explainer into generating incorrect explanatory subgraphs. Therefore, we introduce a denoising   
module during the process of variational inference. The original encoder part is modified to:

$$
q _ { \Psi } ^ { \prime } ( \mathbf { Z } | \mathcal { G } ) = \int q _ { \Psi } ( \mathcal { G } | \tilde { \mathcal { G } } ) q ( \tilde { \mathcal { G } } | \mathcal { G } ) \mathrm { d } \tilde { \mathcal { G } } ,
$$

where 141 $\Psi$ is the encoder based on $\tilde { \mathcal { G } }$ , and $\begin{array} { r } { q ( \tilde { \mathcal { G } } | \mathcal { G } ) = \prod _ { r \in \mathcal { R } } q ( \tilde { \mathbf { A } } _ { r } | \mathbf { A } _ { r } ) } \end{array}$ . During this process, the 142 evidence lower bound is expressed as:

$$
\mathcal { L } _ { d } = \mathbb { E } _ { q _ { \Psi } ^ { \prime } ( \mathbf { Z } | \mathcal { G } ) } [ \log \frac { p _ { \theta } ( \mathbf { Z } , \mathcal { G } ) } { q _ { \Psi } ^ { \prime } ( \mathbf { Z } | \mathcal { G } ) } ] .
$$

As we need to derive the distribution of the noisy graph


## Agentic Reader Analysis

Yes — a mathematical formula issue is indicated.

### 1) ELBO expectation uses the wrong distribution symbol (`q*` vs `q_Ψ`)
In the ELBO, the expectation is taken under \(q^\*(\mathbf Z\mid \mathcal G)\), but the denominator inside the log uses \(q_{\Psi}(\mathbf Z\mid \mathcal G)\). This is internally inconsistent unless \(q^\* = q_{\Psi}\) is explicitly defined (it is not in the snippet).

Cited location (Lines ~140–141 in the paper text shown):
\[
\mathcal { L } ( \Psi , \theta ; \mathcal { G } ) = \mathbb { E } _ { q \ast ( \mathbf { Z } | \mathcal { G } ) } \left[ \log \frac { p _ { \theta } ( \mathbf { Z } , \mathcal { G } ) } { q _ { \Psi } ( \mathbf { Z } | \mathcal { G } ) } \right]
\]
(from the paragraph beginning “Then, the evidence lower bound \(\mathcal L(\Psi,\theta;\mathcal G)\) can be expressed as follows:”)

Normally, the ELBO is written with a single variational posterior, e.g. \(\mathbb E_{q_{\Psi}(\mathbf Z\mid \mathcal G)}[\cdots]\). Using \(q^\*\) suggests an optimal variational distribution, but then the rest of the expression should match that choice.

### 2) Incorrect ELBO decomposition: uses \(\log p_\theta(\mathbf Z\mid \mathcal G)\) instead of \(\log p_\theta(\mathcal G\mid \mathbf Z)\)
The paper rewrites the ELBO term into an expression containing \(\mathbb E[\log p_\theta(\mathbf Z\mid \mathcal G)]\), which is not the standard ELBO decomposition for a generative model \(p_\theta(\mathbf Z,\mathcal G)=p(\mathbf Z)p_\theta(\mathcal G\mid \mathbf Z)\). The expected reconstruction term should be \(\mathbb E_{q}[\log p_\theta(\mathcal G\mid \mathbf Z)]\), not \(\log p_\theta(\mathbf Z\mid \mathcal G)\).

Cited formula:
\[
\mathcal { L } ( \Psi , \theta ; \mathcal { G } ) = \dots = \mathbb { E } _ { q \ast ( \mathbf { Z } | \mathcal { G } ) } \left[ \log p _ { \theta } ( \mathbf { Z } | \mathcal { G } ) \right] - \mathrm { K L } \left( q _ { \Psi } ( \mathbf { Z } | \mathcal { G } ) \| p ( \mathbf { Z } ) \right) .
\]
(same location)

This looks like a swap of conditional direction in the likelihood term.

### 3) KL term formatting/parentheses error later in the “refined” bound
In the refined bound, the KL term appears malformed: `KL( q_Ψ( Z | \tilde{G} ) ) | | p(Z) ]` (extra parentheses and `| |` instead of `\|`), making the formula syntactically incorrect.

Cited location (later refinement after \(\mathcal L_d\)):
\[
= \mathbb { E } _ { q _ { \Psi } ^ { \prime } ( { \bf Z } | \mathcal { G } ) } [ \log p _ { \theta } ( \mathcal { G } | { \bf Z } ) ] - \mathbb { E } _ { q ( \tilde { \mathcal { G } } | \mathcal { G } ) } [ \mathrm { K L } ( q _ { \Psi } ( { \bf Z } | \tilde { \mathcal { G } } ) ) | | p ( { \bf Z } ) ] .
\]
(from the block beginning “this lower bound can be further refined as:”)

These issues collectively indicate a genuine mathematical/formula problem (inconsistency and likely incorrect ELBO decomposition) in the paper around Lines 140–144.