# Agentic Reader Result
**Paper ID:** wh3p37VYm2
**Issue File:** wh3p37VYm2_0.md
**Status:** success
**Timestamp:** 2026-01-19T23:43:57.469754
**Model:** gpt-5.2
**Max Iterations:** 10

**Issue Content:**
## LINE 218-222

following expected update:

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { t + 1 } - \mathcal { L } _ { t } ] \leq \nabla _ { \theta _ { t } } \mathcal { L } ^ { T } ( \theta _ { t + 1 } + \theta _ { t } ) } \\ & { - \displaystyle \sum _ { i = 1 } ^ { V } \nabla _ { e _ { i , t } } \mathcal { L } ^ { T } \mathbb { E } ( e _ { i , t + 1 } - e _ { i , t } ) + \frac { \beta } { 2 } \| \Delta \| ^ { 2 } , } \end{array}
$$

Substituting the embedding update based on equation 4 into the smoothness inequality,



$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { t + 1 } - \mathcal { L } _ { t } ] \leq \nabla _ { \theta _ { t } } \mathcal { L } ^ { T } ( \theta _ { t + 1 } - \theta _ { t } ) } \\ & { \quad - \eta \displaystyle \sum _ { i = 1 } ^ { V } \big ( p _ { i } \| \nabla _ { e _ { i , t } } \mathcal { L } \| ^ { 2 } + \lambda e _ { i , t } ^ { T } \nabla _ { e _ { i , t } } \mathcal { L } \big ) + \frac { \beta } { 2 } \| \Delta \| ^ { 2 } , } \end{array}
$$

and noting from the right hand side of the inequality above, $p _ { i }$ plays important role in reduction of the expected loss. However, the dependence on $p _ { i }$ , is coupled with weight decay, which explains why these two parameters are important to study more deeply to draw a conclusion about grokking.

# 4.2 Dataset Splitting Strategies


## Agentic Reader Analysis

Yes—this snippet points to a **mathematical formula issue** in the paper.

### What the issue is
In the first “expected update” inequality, the inner-product term involving the parameter update for non-embedding parameters uses a **plus**:
\[
\mathbb { E } [ \mathcal { L } _ { t + 1 } - \mathcal { L } _ { t } ] \leq \nabla _ { \theta _ { t } } \mathcal { L } ^ { T } ( \theta _ { t + 1 } + \theta _ { t } ) - \sum_{i=1}^V \nabla_{e_{i,t}}\mathcal{L}^T \mathbb{E}(e_{i,t+1}-e_{i,t}) + \frac{\beta}{2}\|\Delta\|^2.
\]
But in the next line (after “Substituting…”), the same kind of term is written with the **minus**:
\[
\ldots \le \nabla_{\theta_t}\mathcal{L}^T(\theta_{t+1}-\theta_t) - \eta\sum_{i=1}^V\Big(p_i\|\nabla_{e_{i,t}}\mathcal{L}\|^2 + \lambda e_{i,t}^T\nabla_{e_{i,t}}\mathcal{L}\Big)+\frac{\beta}{2}\|\Delta\|^2.
\]

Under the standard \(\beta\)-smoothness inequality, the linear term should be an inner product with the **update difference** (i.e., proportional to \(\theta_{t+1}-\theta_t\)), not \(\theta_{t+1}+\theta_t\). So the appearance of \((\theta_{t+1}+\theta_t)\) is almost certainly a **sign/typo error**.

### Where this occurs in the paper (citation/location)
This is in the section deriving the expected loss decrease right before **“# 4.2 Dataset Splitting Strategies”**. The problematic line is the displayed equation beginning:
\[
\mathbb { E } [ \mathcal { L } _ { t + 1 } - \mathcal { L } _ { t } ] \leq \nabla _ { \theta _ { t } } \mathcal { L } ^ { T } ( \theta _ { t + 1 } + \theta _ { t } )
\]
followed immediately by the “Substituting the embedding update…” equation where it becomes \((\theta_{t+1}-\theta_t)\).