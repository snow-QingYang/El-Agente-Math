# Agentic Reader Result
**Paper ID:** SjAHFGoUb6
**Issue File:** SjAHFGoUb6_2.md
**Status:** success
**Timestamp:** 2026-01-19T23:43:23.265015
**Model:** gpt-5.2
**Max Iterations:** 10

**Issue Content:**
## LINE 155-156

approximation of the data distribution. We show our modeling   
assumptions in Fig. 2.

# 152 3.1 Bounded CIB

We can consider the upper bound to the concept bottleneck loss (2) in terms of the entropy-based   
definitions of the mutual information. Then, by using a variational approximation of the data   
distribution, we bound it by

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { U B - C I B } } \le H ( Y ) + ( 1 - \beta ) H ( C ) + H \left( p ( y \mid c ) , q ( y \mid c ) \right) + \left( 1 + \beta \right) \underset { p ( z ) } { \mathbb { E } } H \left( p ( c \mid z ) , q ( c \mid z ) \right) , } \\ & { \mathcal { L } _ { \mathrm { U B - C I B } } \le ( 1 - \beta ) H ( C ) + \underset { p ( c ) } { \mathbb { E } } H \left( p ( y \mid c ) , q ( y \mid c ) \right) + ( 1 + \beta ) \underset { p ( z ) } { \mathbb { E } } H \left( p ( c \mid z ) , q ( c \mid z ) \right) . } \end{array}
$$

We detail this derivation in Appendix A. We can maximize the concepts’ information bottleneck by   
minimizing the cross entropies of the predictive variables, $y$ and $c$ , and their corresponding ground   
truths and by adjusting the entropy of the concepts—cf. Fig. 2. The simplified upper bound of the   
concept information bottleneck is

$$
\mathcal { L } _ { \mathrm { S U B - C I B } } = ( 1 - \beta ) H ( C ) + \underset { p ( c ) } { \mathbb { R } } H \left( p ( y \mid c ) , q ( y \mid c ) \right) + ( 1 + \beta ) \underset { p ( z ) } { \mathbb { R } } H \left( p ( c \mid z ) , q ( c \mid z ) \right) .
$$

We denote the models that were trained using this bounded concept


## Agentic Reader Analysis

Yes — a mathematical formula issue is indicated.

### What’s wrong
In the “simplified upper bound” equation for SUB-CIB, the expectation operator is incorrectly typeset as **\(\mathbb{R}\)** instead of the standard expectation symbol **\(\mathbb{E}\)**.

### Where it occurs (in the paper)
In Section **3.1 Bounded CIB** (around the snippet’s lines 155–156), the paper writes:
\[
\mathcal{L}_{\mathrm{SUB-CIB}}=(1-\beta)H(C)+\underset{p(c)}{\mathbb{R}}\,H(p(y\mid c),q(y\mid c))+(1+\beta)\underset{p(z)}{\mathbb{R}}\,H(p(c\mid z),q(c\mid z)).
\]
Here, both \(\underset{p(c)}{\mathbb{R}}\) and \(\underset{p(z)}{\mathbb{R}}\) should be \(\underset{p(c)}{\mathbb{E}}\) and \(\underset{p(z)}{\mathbb{E}}\), respectively, since the preceding bound uses expectations \(\mathbb{E}_{p(c)}\) and \(\mathbb{E}_{p(z)}\).