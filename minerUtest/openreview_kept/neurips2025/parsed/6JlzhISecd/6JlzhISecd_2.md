## LINE 132-133

Primal Dual Algorithm   
58 (FSPDA) framework that leads to efficient decentralized algorithms tackling (1) in its general form.   
9 The framework features the design of a new stochastic augmented Lagrangian function.   
As pointed out by [Chang et al., 2020], many decentralized algorithms (including gradient tracking)   
can be interpreted as primal-dual algorithms finding a saddle point of the augmented Lagrangian func  
tion. However, its extension to time varying topology is not straightforward due to the inconsistency   
in dual variables updates. To overcome this challenge, we propose a stochastic equality constrained   
reformulation of (1) to model randomness in topology. Then, the latter yields a stochastic augmented   
Lagrangian function. Applying stochastic approximation (SA) to solve the latter leads to the FSPDA   
framework. Our contributions are   
• We propose two new algorithms: (i) FSPDA-SA is derived by vanilla SA that applies primal-dual   
stochastic gradient descent-ascent on the stochastic augmented Lagrangian, (ii) FSPDA-STORM uses   
an additional control variate / momentum term to reduce the drift term’s variance in a recursive   
manner. Both algorithms are fully stochastic as the random time varying topology is treated as   
a part of randomness. Additionally, our framework supports sparsified communication, i.e., the   
agents can choose to communicate a subset of primal coordinates at each iteration.   
We show that after $T$ iterations, FSPDA-SA (resp. FSPDA-STORM) finds in expectation a solution   
whose squared gradient norm is $\mathcal { O } ( 1 / \sqrt { T } )$ (resp. $\mathcal { O } ( 1 / T ^ { 2 / 3 } ) \rangle$ ). The convergence analysis is derived   
from a new Lyapunov function design that involves an unsigned inner product term and incorporates   
a variance condition on the random time varying topologies. Interestingly, we show empirically   
that using momentum in dual updates benefits the consensus error convergence.   
We also demonstrate that both FSPDA-SA and FSPDA-STORM can be implemented in a fully asyn  
chronous manner, i.e., the agents can communicate and compute at different time slots, and supports   
local update as the algorithms allow for arbitrary time varying topology. That said, we remark that   
81 the convergence rates with local updates of FSPDA-SA and FSPDA-STORM are only suboptimal.   
2 We provide numerical experiments to show that FSPDA-SA and FSPDA-STORM outperform existing   
algorithms in terms of iteration and communication complexity.   
Notations. Let $\mathbf { W } \in \mathbb { R } ^ { d \times d }$ be a symmetric (not necessarily positive semidefinite) matrix, the W  
weighted (semi) inner product of vectors a $, \mathbf { b } \in \mathbb { R } ^ { d }$ is denoted as $\langle \mathbf { a } \mid \mathbf { b } \rangle _ { \mathbf { w } } : = \mathbf { a } ^ { \top } \mathbf { W } \mathbf { b }$ . Similarly,   
the W-weighted (semi) norm is denoted by $\| \mathbf { a } \| _ { \mathbf { W } } ^ { 2 } : = \langle \mathbf { a } \mid \mathbf { a } \rangle _ { \mathbf { W } }$ . The subscript notation is omitted   
for I-weighted inner products. For any square matrix $\mathbf { X }$ , $( \mathbf { X } ) ^ { \dagger }$ denotes its pseudo inverse.

# 2 The Fully Stochastic Primal Dual Algorithm (FSPDA) Framework

This section develops the FSPDA framework for tackling (1) and describes two variants of the   
framework leading to decentralized stochastic optimization of (1). Let $\widetilde { \mathbf { A } } \in \{ - 1 , 0 , 1 \} ^ { | \mathcal { E } | \times n }$ be an   
incidence matrix of $\mathcal { G }$ . By defining $\mathbf { A } = \widetilde { \mathbf { A } } \otimes \mathbf { I } _ { d } \in \{ - 1 , 0 , 1 \} ^ { | \mathcal { E } | d \times n d }$ , we observe that the consensus   
constraint in (1) is equivalent to $\mathbf { A x } = \mathbf { 0 }$ .   
Our first step is to model the randomness in the time varying topology using the random variable   
(r.v.) $\xi _ { a } \sim \mathbb { P } _ { a }$ . For each realization $\xi _ { a }$ , we define the random incidence matrix $\mathbf { A } ( \xi _ { a } ) : = \mathbf { I } ( \xi _ { a } ) \mathbf { A } \in$   
$\{ - 1 , 0 , 1 \} ^ { | \mathcal { E } | d \times n d }$ where $\mathbf { I } ( \xi _ { a } ) \in \{ 0 , 1 \} ^ { | \varepsilon | d \times | \mathcal { E } | d }$ is a binary diagonal matrix. In addition to selecting   
each edge of $\mathcal { G }$ randomly, $\mathbf { I } ( \xi _ { a } )$ selects a random subset of $d$ coordinates. As we will see later, this   
allows our approach to simultaneously achieve random sparsification for communication compression.

Assume that 98 $\mathbb { E } _ { \xi _ { a } \sim \mathbb { P } _ { a } } [ \mathbf { I } ( \xi _ { a } ) ]$ is a positive diagonal matrix, (1) is equivalent to:

$$
\begin{array} { r } { \operatorname* { m i n } _ { { \bf x } \in { \mathbb { R } } ^ { n d } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { \mathbb { E } } _ { \xi _ { i } \sim { \mathbb { P } } _ { i } } \left[ f _ { i } ( { \bf x } _ { i } ; \xi _ { i } ) \right] \quad \mathrm { s . t . } \quad { \mathbb { E } } _ { \xi _ { a } \sim { \mathbb { P } } _ { a } } \left[ { \bf A } ( \xi _ { a } ) \right] { \bf x } = { \bf 0 } . } \end{array}
$$

Denote ${ \boldsymbol { \xi } } = ( \xi _ { 1 } , \dots , \xi _ { n } , \xi _ { a } )$ , FSPDA hinges on the following augmented Lagrangian function of (

$$
\begin{array} { r l } & { \mathcal { L } ( \mathbf { x } , \lambda ) : = \mathbb { E } _ { \xi } [ \mathcal { L } ( \mathbf { x } , \lambda ; \xi ) ] } \\ & { \mathrm { w i t h ~ } \mathcal { L } ( \mathbf { x } , \lambda ; \xi ) : = \sum _ { i = 1 } ^ { n } f _ { i } ( \mathbf { x } _ { i } ; \xi _ { i } ) + \tilde { \eta } \left. \lambda \mid \mathbf { A } ( \xi _ { a } ) \mathbf { x } \right. + \frac { \tilde { \gamma } } { 2 } \| \mathbf { A } ( \xi _ { a } ) \mathbf { x } \| ^ { 2 } , } \end{array}
$$

where $\tilde { \eta } > 0 , \tilde { \gamma } > 0$ are penalty parameters. It can be verified that the saddle points of $\mathcal { L } ( \mathbf { x } , \lambda )$   
correspond to the KKT points of (1) [Bertsekas, 2016]. For brevity, in the rest of this paper, we may   
drop the subscript in $\xi$ whenever the notation is clear from the context.

FSPDA is developed from applying stochastic approximation (SA) to seek a saddle point of (4). By recognizing 104 $\mathbf { A } ( \boldsymbol { \dot { \xi } } ) ^ { \top } \mathbf { A } ( \boldsymbol { \xi } ) = \mathbf { \bar { A } } ^ { \top } \mathbf { A } ( \boldsymbol { \xi } )$ , we consider the stochastic gradients:

$$
\nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } , \lambda ; \boldsymbol { \xi } ) : = \nabla \mathbf { f } ( \mathbf { x } ; \boldsymbol { \xi } ) + \tilde { \eta } \mathbf { A } ^ { \top } \lambda + \tilde { \gamma } \mathbf { A } ^ { \top } \mathbf { A } ( \boldsymbol { \xi } ) \mathbf { x } , \nabla _ { \lambda } \mathcal { L } ( \mathbf { x } , \lambda ; \boldsymbol { \xi } ) : = \tilde { \eta } \mathbf { A } ( \boldsymbol { \xi } ) \mathbf { x } ,
$$

where $\nabla \mathbf { f } ( \mathbf { x } ; \boldsymbol { \xi } ) \ = \ [ \nabla f _ { 1 } ( \mathbf { x } _ { 1 } ; \boldsymbol { \xi } _ { 1 } ) ; \ldots ; \nabla f _ { n } ( \mathbf { x } _ { n } ; \boldsymbol { \xi } _ { n } ) ] \ \in \ \mathbb { R } ^ { n d }$ . Notice that to facilitate algorithm   
development, we have taken a deterministic $\mathbf { A }$ for the term in $\nabla _ { \mathbf x } \mathcal L$ related to $\boldsymbol { \lambda }$ . Now observe the ith   
$d$ -dimensional block of $\mathbf { A } ^ { \top } \mathbf { A } ( \xi ) \mathbf { x }$ which can be aggregated within $\mathcal { N } _ { i } ( \boldsymbol { \xi } )$ the neighborhood of the   
ith agent as:

$$
\begin{array} { r } { \left[ \mathbf { A } ^ { \top } \mathbf { A } ( \xi ) \mathbf { x } \right] _ { i } = \sum _ { j \in \mathcal { N } _ { i } ( \xi ) } \mathbf { C } _ { i j } ( \xi ) ( \mathbf { x } _ { j } - \mathbf { x } _ { i } ) , } \end{array}
$$

where 109 $\mathbf { C } _ { i j } ( \xi ) \in \{ 0 , 1 \} ^ { d \times d }$ is diagonal and depends on the selected coordinates for the edge $( i , j )$ 110 under randomness $\xi$ . Eq. (6) only relies on $\mathbf { x } _ { j }$ from neighbor $j$ that is connected on the time varying

topology $\mathcal G ( \xi )$ . For illustration, an example of the above random graph model is given by Figure 3 in   
Appendix A. Importantly, (5) shows that with the stochastic augmented Lagrangian function, the time   
varying topology can be treated implicitly as a part of the randomness in the stochastic primal-dual   
gradients. The framework is thus described as being fully stochastic as in [Bianchi et al., 2021], and   
departs from [Liu et al., 2024, Alghunaim, 2024] that treat the topology as fixed during the derivation   
116 of primal-dual algorithm(s). From (5), (6), we derive two variants of FSPDA.   
17 FSPDA-SA Algorithm. The first variant of FSPDA is derived from a direct application of stochastic   
8 gradient descent-ascent (SGDA) updates. Take $\alpha > 0 , \beta > 0$ as the step sizes, we have

$$
\begin{array} { r } { \mathbf { x } ^ { t + 1 } = \mathbf { x } ^ { t } - \alpha \nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } ^ { t } , \lambda ^ { t } ; \boldsymbol { \xi } ^ { t } ) , \lambda ^ { t + 1 } = \lambda ^ { t } + \beta \nabla _ { \lambda } \mathcal { L } ( \mathbf { x } ^ { t } , \lambda ^ { t } ; \boldsymbol { \xi } ^ { t } ) . } \end{array}
$$

Taking the variable substitution 119 $\widehat { \lambda } : = \mathbf { A } ^ { \top } \lambda$ yields the following recursion:

FSPDA-SA: for any $t \geq 0$ and any $i \in [ n ]$ ,

$$
\begin{array} { r l } & { \mathbf { x } _ { i } ^ { t + 1 } = \mathbf { x } _ { i } ^ { t } - \alpha \nabla f _ { i } ( \mathbf { x } _ { i } ^ { t } ; \boldsymbol { \xi } _ { i } ^ { t } ) - \eta \widehat { \lambda } _ { i } ^ { t } + \gamma \sum _ { j \in { \mathcal { N } } _ { i } ( \boldsymbol { \xi } _ { a } ^ { t } ) } \mathbf { C } _ { i j } ( \boldsymbol { \xi } _ { a } ^ { t } ) ( \mathbf { x } _ { j } ^ { t } - \mathbf { x } _ { i } ^ { t } ) , } \\ & { \widehat { \lambda } _ { i } ^ { t + 1 } = \widehat { \lambda } _ { i } ^ { t } + \beta \sum _ { j \in { \mathcal { N } } _ { i } ( \boldsymbol { \xi } _ { a } ^ { t } ) } \mathbf { C } _ { i j } ( \boldsymbol { \xi } _ { a } ^ { t } ) ( \mathbf { x } _ { j } ^ { t } - \mathbf { x } _ { i } ^ { t } ) . } \end{array}
$$



Note that 21 $\mathbf { x } ^ { 0 } , \widehat { \lambda } ^ { 0 }$ can be initialized arbitrarily.

FSPDA-STORM Algorithm. The second variant of FSPDA reduces the variance of the stochastic   
gradient term in (5) using the recursive momentum variance reduction technique [Cutkosky and   
Orabona, 2019]. Herein, the key idea is to utilize a control variate in estimating the (primal-dual)   
gradients of $\mathcal { L } ( \mathbf { x } , \lambda )$ . Take $\alpha , \beta > 0$ and $a _ { x } , a _ { \lambda } \in [ 0 , 1 ]$ as the momentum parameters, we have   
$\mathbf { x } ^ { t + 1 } = \mathbf { x } ^ { t } - \alpha \mathbf { m } _ { x } ^ { t } , \lambda ^ { t + 1 } = \lambda ^ { t } + \beta \mathbf { m } _ { \lambda } ^ { t }$ as the primal-dual updates, and

$$
\begin{array} { r l } & { \mathbf { m } _ { x } ^ { t + 1 } = \nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } ^ { t + 1 } , \boldsymbol { \lambda } ^ { t + 1 } ; \boldsymbol { \xi } ^ { t + 1 } ) + ( 1 - a _ { x } ) ( \mathbf { m } _ { x } ^ { t } - \nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } ^ { t } , \boldsymbol { \lambda } ^ { t } ; \boldsymbol { \xi } ^ { t + 1 } ) ) , } \\ & { \mathbf { m } _ { \lambda } ^ { t + 1 } = \nabla _ { \lambda } \mathcal { L } ( \mathbf { x } ^ { t + 1 } , \boldsymbol { \lambda } ^ { t + 1 } ; \boldsymbol { \xi } ^ { t + 1 } ) + ( 1 - a _ { \lambda } ) ( \mathbf { m } _ { \lambda } ^ { t } - \nabla _ { \lambda } \mathcal { L } ( \mathbf { x } ^ { t } , \boldsymbol { \lambda } ^ { t } ; \boldsymbol { \xi } ^ { t + 1 } ) ) . } \end{array}
$$

The aim of $\mathbf { m } _ { r } ^ { t + 1 }$ is to estimate $\nabla _ { \mathbf x } \mathcal L ( \mathbf x ^ { t + 1 } , \lambda ^ { t + 1 } )$ . Now, instead of the straightforward estimator   
$\nabla _ { \mathbf x } \mathcal L ( \mathbf x ^ { t + 1 } , \lambda ^ { t + 1 } ; \xi ^ { t + 1 } )$ , we include an extra zero-mean term $\mathbf { m } _ { x } ^ { t } - \nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } ^ { \tilde { t } } , \lambda ^ { t } ; \xi ^ { t + 1 } )$ to reduce   
the variance of the stochastic gradient estimation. The latter is a control variate that is computed   
recursively. Particularly, it has been shown in [Cutkosky and Orabona, 2019] that it can effectively   
reduce variance with a carefully designed parameter $a _ { x }$ , provided that the stochastic gradient map   
satisfies a mean-square Lipschitz condition. We summarize the algorithm as follows.

FSPDA-STORM: for any $t \geq 0$ and any $i \in [ n ]$ ,

$$
\begin{array} { r l } & { \mathbf { x } _ { i } ^ { t + 1 } = \mathbf { x } _ { i } ^ { t } - \alpha \mathbf { m } _ { x , i } ^ { t } , } \\ & { \widehat { \lambda } _ { i } ^ { t + 1 } = \widehat { \lambda } _ { i } ^ { t } + \beta \mathbf { m } _ { \lambda , i } ^ { t } , } \\ & { \mathbf { m } _ { x , i } ^ { t + 1 } = \left( 1 - a _ { x } \right) \left[ \mathbf { m } _ { x , i } ^ { t } + \nabla f _ { i } ( \mathbf { x } _ { i } ^ { t } ; \xi _ { i } ^ { t + 1 } ) - \eta \widehat { \lambda } _ { i } ^ { t } + \gamma \sum _ { j \in { \cal N } _ { i } ( \xi _ { a } ^ { t + 1 } ) } \mathbf { C } _ { i j } ( \xi _ { a } ^ { t + 1 } ) ( \mathbf { x } _ { j } ^ { t } - \mathbf { x } _ { i } ^ { t } ) \right] } \\ & { \quad \quad \quad + \nabla f _ { i } ( \mathbf { x } _ { i } ^ { t + 1 } ; \xi _ { i } ^ { t + 1 } ) - \eta \widehat { \lambda } _ { i } ^ { t + 1 } + \gamma \sum _ { j \in { \cal N } _ { i } ( \xi _ { a } ^ { t + 1 } ) } \mathbf { C } _ { i j } ( \xi _ { a } ^ { t + 1 } ) ( \mathbf { x } _ { j } ^ { t + 1 } - \mathbf { x } _ { i } ^ { t + 1 } ) } \\ & { \mathbf { m } _ { \lambda , i } ^ { t + 1 } = ( 1 - a _ { \lambda } ) \left[ \mathbf { m } _ { \lambda , i } ^ { t } + \sum _ { j \in { \cal N } _ { i } ( \xi _ { a } ^ { t + 1 } ) } \mathbf { C } _ { i j } ( \xi _ { a } ^ { t + 1 } ) ( \mathbf { x } _ { j } ^ { t } - \mathbf { x } _ { i } ^ { t } ) \right] } \\ & { \quad \quad \quad + \sum _ { j \in { \cal N } _ { i } ( \xi _ { a } ^ { t + 1 } ) } \mathbf { C } _ { i j } ( \xi _ { a } ^ { t + 1 } ) ( \mathbf { x } _ { j } ^ { t + 1 } - \mathbf { x } _ { i } ^ { t + 1 } ) } \end{array}
$$

Note that to achieve the theoretical performance (see later in Sec. 3), 134 $\mathbf { x } ^ { 0 } , \widehat { \lambda } ^ { 0 } , \mathbf { m } _ { x } ^ { 0 } , \mathbf { m } _ { \lambda } ^ { 0 }$ shall be 135 initialized as ${ \bf x } _ { i } ^ { 0 } = \bar { \bf x } ^ { 0 }$ , $\widehat { \bf \sf { A } } _ { i } ^ { 0 } = ( \alpha / \eta ) n ^ { - 1 } ( \nabla F ( \bar { \bf { x } } ^ { 0 } ) - \nabla f _ { i } ( \bar { \bf { x } } ^ { 0 } ) )$ , $\mathbf { m } _ { x , i } ^ { 0 } ~ = ~ \nabla F ( \bar { \mathbf { x } } ^ { 0 } )$ , $\mathbf { m } _ { \lambda , i } ^ { 0 } = \mathbf { 0 }$ 136 according to (23). We remark that a simple initialization choice $\widehat { \mathbf { \lambda } } ^ { 0 } = \mathbf { m } _ { x , i } ^ { 0 } = \mathbf { m } _ { \lambda , i } ^ { 0 } = \mathbf { 0 }$ works well 137 in practice.

Both FSPDA-SA and FSPDA-STORM are decentralized
