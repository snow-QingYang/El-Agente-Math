# Agentic Reader Result
**Paper ID:** MYfyMjaoQh
**Issue File:** MYfyMjaoQh_0.md
**Status:** success
**Timestamp:** 2026-01-19T23:43:17.113845
**Model:** gpt-5.2
**Max Iterations:** 10

**Issue Content:**
## LINE 168

Independent Subspace (IS) Loss. We can substitute the orthogonalisation procedure by using an   
auxiliary loss component for maximising the rank of the predicted subspace. For this purpose, we   
chose the squared cosine similarity computed between pairs of predicted vectors. The final expression   
for the independent subspace (IS) loss is

$$
\mathrm { I S } \mathrm { L o s s } = \frac { 1 } { K ^ { 2 } } \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { K } ( { \bf x } _ { k } ^ { T } W { \bf x } _ { l } ) ^ { 2 } - \frac { 1 } { K ^ { 2 } } \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { K } ( { \bf y } _ { k } ^ { T } W ^ { \frac { 1 } { 2 } } { \bf x } _ { l } ) ^ { 2 } ,
$$

where the predictions 176 $\left\{ { \bf x } _ { l } \right\}$ are normalised prior to the loss computation


## Agentic Reader Analysis

Yes — this snippet indicates a **mathematical formula issue** in the paper.

### What the issue is
In the IS Loss equation, the paper uses a term involving **\(W^{1/2}\)**:
\[
\mathrm { I S } \mathrm { L o s s } = \frac { 1 } { K ^ { 2 } } \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { K } ( { \bf x } _ { k } ^ { T } W { \bf x } _ { l } ) ^ { 2 }
- \frac { 1 } { K ^ { 2 } } \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { K } ( { \bf y } _ { k } ^ { T } W ^ { \frac { 1 } { 2 } } { \bf x } _ { l } ) ^ { 2 } .
\]
(Location: **Independent Subspace (IS) Loss**, around **line 168** in the paper text)【paper, IS Loss equation near line 168】.

However, immediately after, the paper states the predictions are normalized such that:
\[
\mathbf{x}_l^T W \mathbf{x}_l = 1
\]
(appears right after the equation)【paper, text following IS Loss equation】.

The formula issue is that the paper introduces **\(W^{1/2}\)** in the second term without defining or justifying:
- what \(W\) is (e.g., whether it is symmetric positive definite),
- whether the matrix square root \(W^{1/2}\) exists/ is unique,
- and why a mixed form \(y_k^T W^{1/2} x_l\) is the correct counterpart to the \(x_k^T W x_l\) similarity.

This is a **mathematical specification/definition gap**: the loss depends on \(W^{1/2}\), but the conditions under which it is valid and how it is computed are not given in the shown section, making the formula ambiguous or potentially incorrect in general.