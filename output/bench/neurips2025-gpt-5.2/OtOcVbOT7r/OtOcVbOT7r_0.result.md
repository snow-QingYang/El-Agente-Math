# Agentic Reader Result
**Paper ID:** OtOcVbOT7r
**Issue File:** OtOcVbOT7r_0.md
**Status:** success
**Timestamp:** 2026-01-19T23:43:19.529540
**Model:** gpt-5.2
**Max Iterations:** 10

**Issue Content:**
## LINE 226-244

Convex problems. We start with the convex case.

6 Theorem 4.1 (Convergence of DP-Clipped-SGD for the convex objectives). Let the integer $K \geq 0$ and $\beta \in ( 0 , 1 ]$ be given. Furthermore, let Assumptions 2.1, 2.2, 2.3, 2.4, hold for $Q = B _ { 2 R } ( x ^ { \star } )$ , $R \geq$ 8 $\| x ^ { 0 } - x ^ { \star } \|$ . Set $\zeta _ { \lambda } : = \operatorname* { m a x } \left\{ 0 , 2 L R - { \textstyle { \frac { \lambda } { 2 } } } \right\}$ , and further assume that the step-size $\gamma$ is selected to 9 satisfy

$$
\begin{array} { r } { \gamma \leq \mathcal { O } \left( \operatorname* { m i n } \left\{ \frac { 1 } { L } , \frac { R } { \lambda ^ { 1 - \alpha / 2 } \sqrt { K \ln \left( \frac { K } { \beta } \right) \left( \sigma ^ { \alpha } + \zeta _ { \lambda } ^ { \alpha } \right) } } , \right. } \\ { \left. \frac { R \lambda ^ { \alpha - 1 } } { K ( \sigma ^ { \alpha } + \zeta _ { \lambda } ^ { \alpha } ) \left( \frac { L R } { \lambda } + \frac { \lambda ^ { \alpha - 1 } \zeta _ { \lambda } } { \sigma ^ { \alpha } + \zeta _ { \lambda } ^ { \alpha } } + \left( \sigma ^ { \alpha } + \zeta _ { \lambda } ^ { \alpha } \right) ^ { \frac { - 1 } { \alpha } } \right) } , \frac { R } { \sigma _ { \omega } \sqrt { d K \ln \left( \frac { K } { \beta } \right) } } \right\} \right) . } \end{array}
$$

Then, after $K$ iterations of DP-Clipped-SGD, the iterates with probability at least $1 - \beta$ satisfy

$$
\operatorname* { m i n } _ { t \in [ 0 , K ] } f ( x ^ { t } ) - f ( x ^ { \star } ) \leq \frac { 4 R ^ { 2 } } { \gamma ( K + 1 ) } + \frac { 6 4 L R ^ { 4 } } { \lambda ^ { 2 } \gamma ^ { 2 } ( K + 1 ) ^ { 2 } } .
$$

The convergence rate and the neighborhood to which the algorithm converges depend on the magnitude of $\lambda$ in a non-trivial way. Table 1 summarizes these relationships for different values of $\lambda$ in the absence of DP noise. In the special case where $\lambda = \mathcal { O } \left( \sigma \left( K / \ln \frac { K } { \beta } \right) ^ { 1 / \alpha } \right)$ , our theorem provides a convergence rate of $\mathcal { O } \left( \left( \left( \ln \frac { K } { \beta } \right) / K \right) ^ { ( \alpha - 1 ) / \alpha } + ( \ln \frac { K } { \beta } ) \big / K \right)$ to the exact solution in the asymptotic regime. This matches the rate previously derived by Sadiev et al. (2023).

In contrast, if $\lambda$ is chosen as a constant, independent of $K$ , the leading term in the convergence rate simplifies to $\mathcal { O } ( \sqrt { ( \ln \frac { K } { \beta } ) / \kappa } )$ , which is faster than the more conservative bound $\mathcal { O } \left( \left( ( \ln \frac { K } { \beta } ) \ / \middle / K \right) ^ { ( \alpha - 1 ) \ / \alpha } \right)$ . However, this faster rate comes at the cost of only guaranteeing convergence to a neighborhood around the optimal solution, determined by the third term in the stepsize condition (8).

To ensure $( \varepsilon , \delta )$ -DP for DP-Clipped-SGD in our setting (i.e., expectation minimization), one can set the noise scale as $\begin{array} { r } { \sigma _ { \omega } = \Theta \left( \frac { \lambda } { \varepsilon } \sqrt { K \ln \left( \frac { K } { \delta } \right) \ln \left( \frac { 1 } { \delta } \right) } \right) } \end{array}$ and apply the advanced composition theorem (Dwork et al., 2014, Theorem 3.22). Given the fourth term in (8), this choice implies that the stepsize decreases as $^ { 1 / K }$ , resulting in convergence to a certain neighborhood. This observation is formalized in the next corollary.

Corollary 4.2 (Convergence of Clipped-SGD for the convex


## Agentic Reader Analysis

Yes — a **mathematical/formula issue is indicated** in this snippet.

### 1) Apparent typo in the bound (“6 4”)
In Theorem 4.1, the final inequality is written as:
\[
\min_{t\in[0,K]} f(x^t)-f(x^\star)\le \frac{4R^2}{\gamma(K+1)}+\frac{6 4 L R^4}{\lambda^2\gamma^2 (K+1)^2}.
\]
The factor **“6 4”** in the numerator is almost surely a formatting/typing error and should likely be **64** (or some other single constant), not two separate numbers. This is a formula correctness/presentation issue in the theorem statement itself (Theorem 4.1, around line 226–244 in your snippet).

### 2) Incomplete / malformed expression “decreases as \(^{1/K}\)”
Later, the text says:
> “...this choice implies that the stepsize decreases as \(^{1/K}\) ...”

The expression **\(^{1/K}\)** is not meaningful on its own (it is missing a base, e.g., something like \(K^{-1/2}\), \(1/\sqrt{K}\), \(1/K\), etc.). This is another formula/notation issue in the same excerpt (right after the discussion of choosing \(\sigma_\omega\) and referencing the fourth term in (8)).

**Cited locations (from the paper text shown):**
- Theorem 4.1 bound: the term \(\frac{6 4 L R^4}{\lambda^2\gamma^2 (K+1)^2}\).
- Stepsize discussion: “the stepsize decreases as \(^{1/K}\)”.