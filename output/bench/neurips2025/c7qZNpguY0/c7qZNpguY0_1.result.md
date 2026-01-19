# Agentic Reader Result
**Paper ID:** c7qZNpguY0
**Issue File:** c7qZNpguY0_1.md
**Status:** success
**Timestamp:** 2026-01-19T07:08:37.847789
**Model:** openai:gpt-5-mini
**Max Iterations:** 10

**Issue Content:**
## LINE 274

distance.

Proposition 3.6. Denote $P _ { \varepsilon } = ( 1 - \varepsilon ) P + \varepsilon C$ where $C$ is some contamination distribution. We have when Assumption 1 is verified: $| d _ { K T } ( P _ { \varepsilon } , Q ) - d _ { K T } ( P , Q ) | \leq 2 \varepsilon$ .

The proof relies on the triangular inequality (see Appendix $\underline  { \vert \mathbf { A } . 3 . 2 \} }$ . Hence, we see that $d _ { K T }$ is robust while for the Wasserstein distance, a contamination $C$ arbitrarily “far away from the distribution $Q ^ { , , }$ will incur an arbitrarily high distance. The proof of robustness also works for MMD.

# 4 Statistical Properties

# 4.1 Convergence rate

In this section, we consider a measure $\mu$ and its empirical counterpart $\mu _ { n }$ for $n$ independent samples and study the rate of convergence of $d _ { K T } ( \mu , \mu _ { n } )$ . We note $A \lesssim _ { \mu ^ { \otimes n } } b$ where $A$ is r.v., when for any $\delta > 0$ , there exists $c _ { \delta } < \infty$ such that $\mu ^ { \otimes n } ( A \leq c _ { \delta } b ) \geq \delta$ . With the Schatten 1-norm, it is not enough to study only the concentration of one (the maximal) eigenvalue as for the operator norm $\begin{array} { r } { p = \infty , } \end{array}$ ), we need to handle an infinity of eigenvalues (when the RKHS is of infinite dimension), neither can we use the Cauchy-Schwarz trick as for the Hilbert norm $( p = 2 )$ ). However, since the trace of our kernel density operators are bounded by 1, only a few of the eigenvalues will have a significant contribution. Therefore, assuming some decay rate on those eigenvalues, we can focus on the convergence of operators on a subspace of the top eigenvectors, using results from the Kernel PCA literature. We introduce the population and empirical square loss associated with some projector $P$ :

$$
R ( P ) = \mathbb { E } _ { X \sim \mu } | | \phi ( X ) - P \phi ( X ) | | _ { \mathcal { H } } ^ { 2 } , \qquad R _ { n } ( P ) = \sum _ { i = 1 } ^ { n } \frac { 1 } { n } | | \phi ( x _ { i } ) - P \phi ( x _ { i } ) | | _ { \mathcal { H } } ^ { 2 }
$$

where the $( x _ { i } ) _ { i = 1 \ldots n }$ are each drawn independently from $\mu$ . We first make


## Agentic Reader Analysis

No formula issue detected.

Explanation and citations:
- The paper defines d_{KT}(μ, ν) = ||Σ_μ − Σ_ν||_1 (see definition around "d _ { K T } ( μ , ν ) = || Σ _ { μ } − Σ _ { ν } || _ { 1 }", positions ≈13280–13520).
- Assumption 1 is k(x,x)=1 (so Tr Σ_μ = 1) (see "Assumption 1. ∀ x ∈ X, k(x,x)=1" positions ≈10460–10750).
- For P_ε = (1−ε)P + εC we have Σ_{P_ε} = (1−ε)Σ_P + εΣ_C, hence
  Σ_{P_ε} − Σ_Q = (Σ_P − Σ_Q) + ε(Σ_C − Σ_P).
  By the reverse/standard triangle inequality for the Schatten-1 norm,
  | ||Σ_{P_ε} − Σ_Q||_1 − ||Σ_P − Σ_Q||_1 | ≤ ε ||Σ_C − Σ_P||_1 = ε d_{KT}(C,P).
  Since Assumption 1 gives ||Σ_C||_1 = ||Σ_P||_1 = 1, we have d_{KT}(C,P) ≤ 2, so
  | d_{KT}(P_ε, Q) − d_{KT}(P, Q) | ≤ 2 ε.
  This is exactly Proposition 3.6 (see proposition text around positions ≈31400–31750).

The only issues in the reviewer snippet are typographical/formatting (garbled appendix reference "Appendix $\underline{\vert\mathbf{A}.3.2\}$" and the odd "Q^{,,}" text). Those are presentation problems, not a mathematical/formula error.