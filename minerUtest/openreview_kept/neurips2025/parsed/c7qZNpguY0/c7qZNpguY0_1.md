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
