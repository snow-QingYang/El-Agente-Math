# Agentic Reader Result
**Paper ID:** GgD01U3Y0H
**Issue File:** GgD01U3Y0H_1.md
**Status:** success
**Timestamp:** 2026-01-19T06:45:20.108729
**Model:** openai:gpt-5-mini
**Max Iterations:** 10

**Issue Content:**
## LINE 194

Rankings as finite-sample approximations

86 As mentioned above, while we have no access to the CDFs themselves, we have samples from the   
joint distribution over the objectives, i.e., over, $p ( [ \mathsf { y } _ { 1 } , \mathsf { y } _ { 2 } , \ldots , \mathsf { y } _ { K } ] )$ . Namely, we can consider each   
model $h \in \mathcal H$ as a sample from the joint distribution and, by looking at each objective individually,   
as a sample from the marginal distributions.   
Let us now focus on the $k$ -th objective, ${ \tt y } _ { k }$ , and drop the subindex in the following to ease notation. Say   
that we have $| \mathcal { H } | = N$ i.i.d. realizations of the objective, i.e., $\left\{ \mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } , \ldots , \mathbf { y } _ { N } \right\} \overset { \forall \mathrm { i . i . d . } } { \sim } P _ { k }$ . Then, we can   
approximate Eq. 8 for the $i$ -th sample, $\mathbf { u } _ { i } = F ( \mathbf { y } _ { i } )$ , by computing its order statistic, i.e., the random   
variable representing its relative ranking within the population, $\begin{array} { r } { R ( i ) : = \sum _ { j = 1 } ^ { N } [ \mathbf { y } _ { j } < \mathbf { y } _ { i } ] } \end{array}$ , where   
Iverson brackets denote the indicator function, such that $\begin{array} { r } { \mathbf { y } _ { R ( 1 ) } \leq \mathbf { y } _ { R ( 2 ) } \leq \ldots \leq \mathbf { y } _ { R ( N ) } . } \end{array}$ . Specifically,   
since the empirical $C D F$ is the fraction of samples smaller than the input, it is direct to show that

$$
{ \hat { \mathbf { u } } } _ { i } = { \hat { F } } ( i ) : = { \frac { 1 } { N } } \sum _ { j = 1 } ^ { N } [ \mathbf { y } _ { j } < \mathbf { y } _ { i } ] = { \frac { 1 } { N } } R ( i )
$$

enjoys the following properties


## Agentic Reader Analysis

Yes — this snippet indicates a mathematical/formula error.

What the paper states (location ~ lines 21350–21950)
- Equation given:
  hat{u}_i = hat{F}(i) := (1/N) sum_{j=1}^N [ y_j < y_i ] = (1/N) R(i)
  (see the displayed equation around "hat{u}_i = hat{F}(i) ... = (1/N) R(i)").  
- Immediately after (Proposition 3.1) the paper claims:
  "hat{u}_i is an unbiased estimator of the CDF at y_i, u_i, with variance u_i(1-u_i)/N."
  (see Proposition 3.1 and the following proof).

Why this is wrong
- The sum R(i) = sum_{j=1}^N [y_j < y_i] includes the term j = i. For j = i the indicator [y_i < y_i] = 0 deterministically, so R(i) is actually a sum of N−1 independent Bernoulli( u_i ) random variables (the j ≠ i terms), not N such variables.
- Conditioning on y_i, for j ≠ i we have E[[y_j < y_i] | y_i] = u_i. Therefore
  E[R(i) | y_i] = (N−1) u_i, and
  E[hat{u}_i | y_i] = (1/N) E[R(i) | y_i] = (N−1)/N · u_i.
  So hat{u}_i is biased (downward) by the factor (N−1)/N; it is not unbiased for u_i.
- Similarly, Var(R(i) | y_i) = (N−1) u_i(1−u_i), so
  Var(hat{u}_i | y_i) = (1/N^2) (N−1) u_i(1−u_i) = u_i(1−u_i) (N−1)/N^2,
  not u_i(1−u_i)/N as stated.

Corrected expressions (conditional on y_i)
- E[hat{u}_i | y_i] = (N−1)/N · u_i
- Var(hat{u}_i | y_i) = u_i(1−u_i) (N−1)/N^2

(For large N the difference is small, but the exact unbiasedness/variance statements in Proposition 3.1 are incorrect as written given the definition hat{u}_i = (1/N) sum_{j=1}^N [y_j < y_i].)

References in the paper: the displayed equation for hat{u}_i and Proposition 3.1 (and its proof) in the "Rankings as finite-sample approximations" subsection (around the displayed formula and Prop. 3.1).