# Agentic Reader Result
**Paper ID:** vWaMUMrBpF
**Issue File:** vWaMUMrBpF_2.md
**Status:** success
**Timestamp:** 2026-01-19T23:43:56.378082
**Model:** gpt-5.2
**Max Iterations:** 10

**Issue Content:**
## LINE 248

algorithm for IAM-S and provide an analysis of its objective. The algorithm for IAM-D involves a similar inner maximization for $S _ { \rho } ( \theta )$ followed by a standard gradient descent step on $L _ { \mathrm { I A M - D } } ( \theta )$ .

# 5.1 Algorithm for IAM-D and IAM-S

Optimizing $L _ { \mathrm { I A M - S } } ( \theta )$ and $L _ { \mathrm { I A M - S } } ( \theta )$ involves a min-max procedure. The inner maximization to find $\delta ^ { * }$ (i.e., computing $S _ { \rho } ( \theta )$ and the corresponding $\delta ^ { * }$ ) is performed using an Algorithm 1, typically for $K = 1$ step for efficiency. IAM-D simply add the $\beta S _ { \rho } ( \theta )$ with $\delta _ { K }$ to the $L ( \theta )$ , and then update $\theta$ with standard SGD. The outer minimization step of IAM-S updates $\theta$ based on the gradient of the loss $L ( \theta + \delta _ { K } )$ dropping the second-order terms same with SAM: $\nabla _ { \boldsymbol { \theta } } L _ { \mathrm { I A M - S } } ( \boldsymbol { \theta } ) \approx \nabla _ { \boldsymbol { \theta } } \bar { L } ( \boldsymbol { \theta } ) | _ { \boldsymbol { \theta = \theta + \delta _ { K } } }$ . This two-step process is summarized in Algorithm 2 in Appendix D.

# 5.2 Empirical evaluation in supervised learning


## Agentic Reader Analysis

Yes — a **mathematical formula/notation issue** is indicated.

### What the issue is
In Section **5.1 “Algorithm for IAM-D and IAM-S”**, the paper states (emphasis added):

> “Optimizing \(L_{\mathrm{IAM-S}}(\theta)\) and **\(L_{\mathrm{IAM-S}}(\theta)\)** involves a min-max procedure.”  
(around the start of Section 5.1; the snippet corresponds to line ~248 in the reviewer note)

This is almost certainly a **typo in the mathematical expressions**: it repeats \(L_{\mathrm{IAM-S}}(\theta)\) twice, but the surrounding text is discussing **both IAM-D and IAM-S**, and immediately above it also references \(L_{\mathrm{IAM-D}}(\theta)\):

> “…followed by a standard gradient descent step on \(L_{\mathrm{IAM-D}}(\theta)\).”

So the second term in the “Optimizing … and …” phrase should presumably be **\(L_{\mathrm{IAM-D}}(\theta)\)**, not \(L_{\mathrm{IAM-S}}(\theta)\).

### Citation / location
- Section **5.1 Algorithm for IAM-D and IAM-S**, sentence beginning:  
  “Optimizing \(L_{\mathrm{IAM-S}}(\theta)\) and \(L_{\mathrm{IAM-S}}(\theta)\) …” (the duplicated formula occurs there).