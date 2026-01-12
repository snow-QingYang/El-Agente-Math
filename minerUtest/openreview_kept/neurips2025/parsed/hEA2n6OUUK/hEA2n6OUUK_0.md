## LINE 195-199

covariance distributions [10, 14, 18] to approximate the true gradient.

Lemma 2.3 (Robust Mean Estimation [18]). Let $D$ be a distribution on $\mathbb { R } ^ { d }$ with unknown mean $\mu$ and unknown covariance matrix $\Sigma$ where $\Sigma \preceq \sigma ^ { 2 } I$ . Let $\epsilon _ { 3 } > 0$ be a sufficiently small universal constant. Let $0 < \epsilon \le \epsilon _ { 3 }$ and $\tau > 0$ . Given an $\epsilon$ -corrupted set of $n$ samples drawn from $D$ , one can output a vector $ { \widehat { \mu } } \in  { \mathbb { R } } ^ { d }$ in time ${ \widetilde { \cal O } } ( n d \log ( 1 / \tau ) ) $ ) such that, with probability at least $1 - \tau - \exp ( - n \epsilon )$ , we have $\begin{array} { r } { \| \widehat { \mu } - \mu \| _ { 2 } = O \left( \sqrt { \epsilon } + \sqrt { \frac { d } { n \tau } } + \sqrt { \frac { d ( \log d + \log ( 1 / \tau ) ) } { n } } \right) \sigma . } \end{array}$ .

# 3 Outlier-Robust Phase Retrieval
