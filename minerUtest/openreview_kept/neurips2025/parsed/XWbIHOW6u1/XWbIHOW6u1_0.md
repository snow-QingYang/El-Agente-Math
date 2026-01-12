## LINE 174-187

Proposition 3, we can 171 apply the iterative algorithm used in SBM algorithm [45, Algorithm 1] to the multi-marginal setting:

$$
\begin{array} { r } { \mathbb { P } ^ { ( 2 n + 1 ) } : = \mathcal { M } ^ { \mathfrak { m } } ( \mathbb { P } ^ { ( 2 n ) } , \mathcal { T } ) , \ \mathbb { P } ^ { ( 2 n + 2 ) } : = \mathcal { R } ^ { \mathfrak { m } } ( \mathbb { P } ^ { ( 2 n + 1 ) } , \mathcal { T } ) , \quad | \mathcal { T } | > 2 . } \end{array}
$$

72 The convergence guarantees proved for the iteration apply equally well to the multi-marginal case.

Proposition 4 (Convergence). $\mathbb { P } ^ { ( n ) } = \mathbb { P } ^ { m S B P }$ of mSBP as $n \uparrow \infty$ with iterative procedure in (13).

# 3.2 Practical Implementation.

In practice, at each iteration 175 $n$ of (13) we approximate the optimal control $v ^ { \star }$ from (11) by a neural 176 network $v _ { \theta }$ . By Girsanov theorem, $\theta$ are chosen to minimize the following training objective function:

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } , \mathcal { T } , \Pi _ { \mathcal { T } } ) = \int _ { 0 } ^ { T } \mathbb { E } _ { \Pi _ { t , \tau } } [ | | \sigma \nabla \log \mathbb { Q } _ { \beta _ { \mathcal { T } } ( t ) | t } ( \mathbf { X } _ { \beta _ { \mathcal { T } } ( t ) } | \mathbf { X } _ { t } ) - v _ { \boldsymbol { \theta } } ( t , \mathbf { X } _ { t } ) | | ^ { 2 } d t ] , } \end{array}
$$

where $\beta _ { \mathcal { T } } ( t ) = \operatorname* { m i n } _ { u } \{ u > t | t \in \mathcal { T } \} \in [ 0 , T ]$ is the most recent time point in $\tau$ after time $t$ . With   
this notation, the SBM can be generalized to the case of multi-marginal constraints. For example,   
when $\mathcal { T } = \{ 0 , T \}$ then (14) reduces to the objective function described in [45].   
The learned Markov control ${ v } _ { \boldsymbol { \theta } \star } \big ( t , \mathbf { x } _ { t } \big )$ then ensures $\begin{array} { r } { \mathbb { P } _ { t } ^ { \theta ^ { \star } } = \Pi _ { t } } \end{array}$ for all $t \in [ 0 , T ]$ . Moreover, prior   
SBM algorithms interleave forward and backward-time Markov projections to re-anchor the terminal   
distribution and prevent bias between $\mathbb { P } _ { T } ^ { ( n ) }$ and $\Pi _ { T }$ accumulate for each $n \in \mathbb { N }$ . In the multi-marginal   
setting, we again build the backward-time Markov projection as in Proposition 2 by gluing the local   
bridge reversals, so that $\mathbb { P } ^ { \star }$ is governed by both SDEs (10) and the corresponding backward dynamics:

$$
\begin{array} { r l } & { d \mathbf { Y } _ { t } ^ { \star } = [ - f _ { T - t } ( \mathbf { Y } _ { t } ^ { \star } ) + \sigma u ^ { \star } ( t , \mathbf { Y } _ { t } ^ { \star } ) ] d t + \sigma d \mathbf { W } _ { t } , \quad \mathbf { Y } _ { 0 } ^ { \star } \sim \Pi _ { T } , } \\ & { \mathrm { w h e r e ~ } u ^ { \star } ( t , \mathbf { y } ) = \sum _ { i = 1 } ^ { k } \mathbf { 1 } _ { ( t _ { i - 1 } , t _ { i } ] } ( t ) \mathbb { E } _ { \Pi _ { t \mid t _ { i - 1 } } } \left[ \nabla \log \mathbb { Q } _ { t \mid t _ { i - 1 } } ( \mathbf { Y } _ { t } \vert \mathbf { Y } _ { t _ { i - 1 } } ) \vert \mathbf { Y } _ { t } = \mathbf { y } \right] , } \end{array}
$$

where the backward optimal control 185 $u ^ { \star }$ in (16) can be approximated with neural network $u _ { \phi }$ where $\phi$ 186 is chosen to minimize the following training objective function with $\begin{array} { r } { \gamma _ { T } ( t ) = \operatorname* { m a x } _ { u } \{ u < t | t \in T \} } \end{array}$ :

$$
\mathcal { L } ( \phi , \mathcal { T } , \Pi _ { \mathcal { T } } ) = \int _ { 0 } ^ { T } \mathbb { E } _ { \Pi _ { t , \tau } } [ | | \sigma \nabla \log \mathbb { Q } _ { t | \gamma _ { \tau } ( t ) } ( \mathbf { Y } _ { t } | \mathbf { Y } _ { \gamma _ { \tau } ( t ) } ) - u _ { \phi } ( t , \mathbf { Y } _ { t } ) | | ^ { 2 } d t ] .
$$

# 187 4 Multi-Marginal Schrodinger Bridge Matching ¨

A na¨ıve extension of the standard SBM using, multi-marginal projections ${ \mathcal { R } } ^ { \mathrm { m m } }$ and ${ \mathcal { M } } ^ { \mathrm { m m } }$ in Sec 3,   
encounters significant limitations not present in the traditional two-endpoint setting. In such an   
extension, each iteration typically enforces marginal constraints only at the global endpoints $( \rho _ { 0 } , \rho _ { T } )$ .   
The multi-marginal coupling $\Pi _ { T } ^ { ( n ) }$ at each iteration $n$ of (13) is then derived by propagating the   
projected dynamics in (10) or (15) solely from these end points $\rho _ { 0 }$ or $\rho _ { T }$ , respectively.   
This approach leads to critical issues specific to the multi-marginal
