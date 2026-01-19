# Agentic Reader Result
**Paper ID:** SjAHFGoUb6
**Issue File:** SjAHFGoUb6_2.md
**Status:** success
**Timestamp:** 2026-01-19T06:59:04.466455
**Model:** openai:gpt-5-mini
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

Yes — this indicates a (typesetting/notation) formula issue.

Explanation and citation:
- In Section 3.1 "Bounded CIB" (around lines 152–166) the paper first shows bounds using expectation operators written as \mathbb{E}, e.g.
  - in the displayed array: "... (1 + β) \underset{p(z)}{\mathbb{E}} H( p(c|z), q(c|z) ) ..." (the upper bound display).
- However, the "simplified upper bound" immediately after is written as
  - mathcal{L}_{SUB-CIB} = (1−β) H(C) + \underset{p(c)}{\mathbb{R}} H( p(y|c), q(y|c) ) + (1+β) \underset{p(z)}{\mathbb{R}} H( p(c|z), q(c|z) ).
  - (This appears in the paragraph beginning "The simplified upper bound of the concept information bottleneck is".)

Issue: the operator \mathbb{R} in the simplified bound is almost surely a typo and should be the expectation operator \mathbb{E} (i.e., \underset{p(c)}{\mathbb{E}} and \underset{p(z)}{\mathbb{E}}). This is inconsistent with the prior displayed bound which correctly uses \mathbb{E}. (If left as \mathbb{R} it is unexplained and mathematically incorrect notation in this context.)

Minor remark: the earlier two-line display also shows one line including H(Y) and a second line without H(Y) — that is reasonable if H(Y) is treated as a constant, but it would help the authors to state that explicitly. 

Recommendation: replace \mathbb{R} by \mathbb{E} in the simplified upper bound (and clarify the removal of the H(Y) term if intended).