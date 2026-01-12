# HOTA: Hamiltonian framework for Optimal Transport Advection

Anonymous Author(s)   
Affiliation   
Address   
email

# Abstract

Optimal transport (OT) has become a natural framework for guiding the probability flows. Yet, the majority of recent generative models assume trivial geometry (e.g., Euclidean) and rely on strong density-estimation assumptions, yielding trajectories that do not respect the true principles of optimality in the underlying manifold. We present Hamiltonian Optimal Transport Advection (HOTA), a Hamilton–Jacobi–Bellman based method that tackles the dual dynamical OT problem explicitly through Kantorovich potentials, enabling efficient and scalable trajectory optimization. Our approach effectively evades the need for explicit density modeling, performing even when the cost functionals are non-smooth. Empirically, HOTA outperforms all baselines in standard benchmarks, as well as in custom datasets with non-differentiable costs, both in terms of feasibility and optimality.

# 12 1 Introduction

Static (Monge-Kantorovich) optimal transport was originally considered as the main framework for   
comparing and finding a cost-minimizing coupling between distributions [Villani et al., 2008], while   
optimality was mainly measured through the boundary marginals. Development of efficient and   
scalable OT solvers [Cuturi, 2013, Peyré et al., 2019] popularized OT across different areas, such as   
generative modeling [Makkuva et al., 2020, Korotin et al., 2022, Buzun et al., 2024], computational   
biology [Bunne et al., 2022], graphics [Bonneel and Digne, 2023], high-energy physics [Nathan   
T. Suri, 2024], and reinforcement learning [Klink et al., 2022, Asadulaev et al., 2024, Bobrin et al.,   
2024, Rupf et al., 2025]. However, one crucial limitation of static formulation is its inability to   
produce non-straight paths, which completely ignores the underlying geometry of the manifold of the   
data. In classical OT, the underlying geometric structure is solely determined by the choice of cost   
function (e.g., , Euclidean distance), inherently limiting the capacity for fine-grained control over the   
trajectories. We refer to [Montesuma et al., 2024, Pereira and Amini, 2025] for recent overview of   
practical applications of OT and to Villani et al. [2008], Santambrogio [2015], Peyré et al. [2019] for   
a formal treatment.   
On the other hand, the dynamical optimal transport paradigm, developed by Benamou and Brenier   
[2000], recasts static OT as a continuous-time variational problem on the space of probability paths,   
effectively incorporating time variable and enabling more nuanced control over optimal trajectories   
(e.g., , through velocity, acceleration, length, or energy over the paths). Importantly, such formulation   
enables one to directly operate on manifolds of non-trivial geometry, whenever the underlying space   
contains curvature, obstacles, or is defined through potentials. This formulation is closely connected   
to stochastic optimal control (SOC), where trajectories are stochastic yet must still maintain optimality,   
a problem class known as the generalized Schrödinger bridge (GSB) Liu et al. [2024], Bartosh et al.   
[2024],.   
A common strategy for GSB involves solving the dual formulation via Hamilton-Jacobi-Bellman   
(HJB) equations, which provide a flexible and a theoretically grounded framework for deriving   
optimal trajectories (Liu et al. [2022], Neklyudov et al. [2024]). These methods parameterize the

Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

cost through a Lagrangian, enforcing optimality via the preservation of kinetic energy or using other   
path-based penalties. While HJB-based approaches yield theoretically sound solutions, they suffer   
from critical drawbacks: (1) unstable optimization dynamics, leading to high-variance gradients   
and poor sample efficiency in high dimensions, and (2) the absence of a strict terminal distribution   
matching criterion, resulting in inexact couplings. Additionally, they typically require differentiable   
Lagrangians, restricting applicability to smooth costs only.   
In the current work, we study the Generalized Schrodinger Bridge problem between two mea  
sures, where the underlying geometry is defined through potentials. We propose a new HJB-based   
framework that explicitly solves GSB task, resolves the learning stability problems of the previous   
approaches, and has theoretical guarantees. We conduct extensive empirical evaluations on existing   
low-dimensional physically-inspired benchmarks, as well as in the high-dimensional generative   
setting. In short, our contributions are as follows:

• Hamiltonian dual reformulation of dynamical OT that binds Kantorovich potentials with an HJB value function, yielding a density-free objective and providing the performance gain compared to existing works;   
Proposed approach is robust to complex geometries and works even with non-smooth cost functions as the proposed objective explicitly incorporates the potential term;   
• HOTA attains state-of-the-art empirical results in a diverse set of tasks, demonstrating both better feasibility (exact marginal matching) and optimality (cost along trajectories) compared to current dynamic OT solvers.

# 59 2 Related work

Diffusion Models and Matching Algorithms. Diffusion models have emerged as powerful tools for generative modeling by prescribing the time evolution of marginal distributions. Matching algorithms, such as Action Matching (Neklyudov et al. [2024]) and Flow Matching (Lipman et al. [2023]), learn stochastic differential equations (SDEs) that align with prescribed probability paths [Blessing et al., 2025]. These methods typically assume explicit or implicit intermediate densities of the flow, whereas our approach (HOTA) optimizes a complete stochastic path from source to target distributions.

Generalized Schrödinger Bridge. The GSB problem extends SB by introducing state costs that   
penalize or reward specific trajectories (Chen et al., 2015). Prior methods for solving GSB, such as   
DeepGSB (Liu et al. [2022]), often relax feasibility constraints or rely on Sinkhorn-based approxima  
tions, which can lead to instability or suboptimal solutions.   
A recent approach GSBM [Liu et al., 2024] follows an alternating optimization scheme: in the first   
stage, it learns the drift field $v _ { t }$ while keeping the marginal distributions $\rho _ { t } ( x _ { t } )$ fixed, using a Flow   
Matching-style objective. In the second stage, it updates the marginals conditioned on the boundary  
coupled distribution $\rho _ { t } ( x _ { t } \mid x _ { 0 } , x _ { 1 } )$ , which is defined via the previously learned drift. While GSBM   
demonstrates strong empirical performance, it imposes two critical limitations: 1) it requires the state   
cost function $U ( x _ { t } )$ to be differentiable everywhere, and 2) it assumes that the conditional marginals   
$\rho ( x _ { t } \mid x _ { 0 } , x _ { 1 } )$ are Gaussian. The first constraint restricts the method’s applicability to domains with   
smooth geometries, sometimes mitigated via interpolation [Kapusniak et al., 2024], while the second   
can lead to suboptimal solutions, unless $U _ { t }$ function is not quadratic.   
Stochastic Optimal Control. The connection between GSB and stochastic optimal control (SOC) has   
been explored in prior works (Theodorou et al. [2010]; Levine [2018]). SOC formulations often relax   
hard distributional constraints into soft terminal costs, which can introduce bias or require adversarial   
training (Liu et al. [2022]). Recently introduced Adjoint Matching approach [Domingo-Enrich et al.,   
2024a] and Stochastic Optimal Control matching (SOCM) [Domingo-Enrich et al., 2024b] address   
several existing limitations, but still produce highly unstable variance estimations. Our method   
provides a natural way to preserve the feasibility via Kantorovich potential sum.

# 3 Preliminaries

Consider stochastic process with controlled drift and diffusion:

$$
\mathrm { d } x _ { t } = v ( t , x _ { t } ) \mathrm { d } t + \sigma ( t , x _ { t } ) \mathrm { d } W _ { t }
$$

where 88 $v : [ 0 , 1 ] \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is the drift (control), $\sigma : [ 0 , 1 ] \times \mathbb { R } ^ { d }  \mathbb { R }$ is the diffusion coefficient, $W _ { t }$ 89 is $d$ -dimensional Brownian motion. We solve the OT minimization task with marginal distributions $( \alpha$ ,

$\beta$ ) and dynamic cost functions $c ( x , \mu )$ and stochastic transport mapping $\mu : \mathbb { R } ^ { d }  \mathcal { P } ( \mathbb { R } ^ { d } )$ presented   
in paper Korotin et al. [2022]

$$
c ( x , \mu ) = \operatorname* { i n f } _ { v ( t , x ) : x _ { 0 } = x , x _ { 1 } \sim \mu } \int _ { 0 } ^ { 1 } \mathbb { E } \mathcal { L } ( t , x _ { t } , v _ { t } ) d t , \quad \mathcal { L } ( t , x _ { t } , v _ { t } ) = \frac { \| v _ { t } \| ^ { 2 } } { 2 } + U ( x _ { t } ) .
$$

This problem is also known as generalized Schrödinger bridge (GSB). It is an extension of the classical   
Schrödinger Bridge (SB) problem, which is a distribution-matching task seeking a diffusion model   
that transports an initial distribution $\alpha$ to a target distribution $\beta$ . While the standard SB minimizes the   
kinetic energy ( $L ^ { 2 }$ cost in OT), the GSB introduces additional flexibility by incorporating a state cost   
$U ( x _ { t } )$ , allowing for more general optimality conditions beyond just kinetic energy minimization. The   
standard SB’s reliance on kinetic energy (Euclidean cost) may not be ideal for all applications (e.g.,   
image spaces, where distance may not be meaningful). Many scientific domains (population modeling,   
robotics, molecular dynamics) require richer optimality conditions, which GSB accommodates via   
$U ( x _ { t } )$ . The potential term usually characterizes the geometry of the space. But in addition, we can   
also include some physical properties of the flow, e.g., entropic penalty or “mean-field” interaction   
[Liu et al., 2022]. Thus, the optimal trajectories are curved to avoid regions with high values of   
$U ( x _ { t } )$ .   
Neural networks can effectively solve high-dimensional Optimal Transport (OT) problems by learning   
the Kantorovich potentials, which maximizes the dual objective (Korotin et al. [2022], Buzun et al.   
[2024]). It is shown in Villani et al. [2008] (Theorem 5.10) that OT task is equivalent to the   
maximization of the Kantorovich potentials sum:

$$
\operatorname* { s u p } _ { g \in L _ { 1 } ( \beta ) } \Big [ \mathbb { E } _ { \alpha } \big [ g ^ { c } ( x ) \big ] + \mathbb { E } _ { \beta } [ g ( y ) ] \Big ] ,
$$

where 108 $g ^ { c }$ denotes $c$ -conjugate transform of the potential $g$

$$
g ^ { c } ( x ) = \operatorname* { i n f } _ { \mu ( x ) : \mathbb { R } ^ { d } \to \mathcal { P } ( \mathbb { R } ^ { d } ) } \mathbb { E } _ { y \sim \mu ( x ) } \Big [ c ( x , \mu ) - g ( y ) \Big ] .
$$

Here $\mu ( x )$ is the stochastic transport mapping, and in our notation it is the final distribution of   
the stochastic process $x _ { 1 }$ under condition that $x _ { 0 } = x$ . The marginality requirement of the final   
distribution of $x _ { 1 }$ (which must correspond to $\beta$ ) is ensured by the potential difference $\mathbb { E } _ { \beta } { \boldsymbol { g } } ( { \boldsymbol { y } } )$ and   
$\mathbb { E } _ { \alpha } \mathbb { E } _ { y \sim \mu ( x ) } g ( y )$ , which tends to infinity otherwise.   
But unlike classical OT, we need to minimize the cost throughout the trajectory $x _ { t }$ , $t \in [ 0 , 1 ]$ with the   
following objective

$$
g ^ { c } ( x ) = \operatorname* { i n f } _ { v ( x , t ) } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \left( \frac { \| v ( t , x _ { t } ) \| ^ { 2 } } { 2 } + U ( x _ { t } ) \right) \mathrm { d } t - g ( x _ { 1 } ) \bigg | x _ { 0 } = x \right] .
$$

In the last expression, we have united infimums by $\mu ( x )$ and control $v ( t , x )$ and as a sequence have   
removed the right side condition $x _ { 1 } \sim \mu ( x )$ . Based on dynamic programming approach, define the   
value function. For any $0 \leq t \leq 1$ , the value function satisfies:

$$
s ( t , x ) = \operatorname* { i n f } _ { x _ { t } } \mathbb { E } \left[ \int _ { t } ^ { 1 } \left( \frac { \| v ( t , x _ { t } ) \| ^ { 2 } } { 2 } + U ( x _ { t } ) \right) \mathrm { d } t - g ( x _ { 1 } ) \bigg | x _ { t } = x \right] ,
$$

such that our objective equals $s ( 0 , x )$ and the boundary condition at time point $t = 1$ is

$$
\forall x \in \mathbb { R } ^ { d } : s ( 1 , x ) = - g ( x ) .
$$

Function $s ( t , x )$ solves the Hamilton-Jacobi-Bellman (HJB) differential equation and it in turn allows us to find the conjugate potential 120 $g ^ { c }$ (4).

$$
- \partial _ { t } s ( t , x ) = \operatorname* { i n f } _ { v } \{ v ^ { T } \nabla _ { x } s ( t , x ) + \mathcal { L } ( t , x , v ) \} + \frac { \sigma ^ { 2 } } { 2 } \mathrm { t r } \{ \nabla ^ { 2 } s ( t , x ) \} .
$$

Representation of the Lagrange function as a sum of kinetic and potential energy allows us to find   
the minimum in velocity $( v )$ in explicit form, such that $v _ { t } = - \nabla _ { x } s ( t , x _ { t } )$ . Together with potential   
optimization (3), we obtain the final GSB objective in dual Kantorovich form. We provide a detailed   
proof in Section 6.   
Theorem 1 (Dual GSB problem). Given distributions $\alpha , \beta \in \mathcal { P } ( \mathbb { R } ^ { d } )$ and stochastic dynamics (1)   
with cost functional (2), the dynamic optimal transport problem admits the following formulation:

$$
\operatorname* { m a x } _ { s ( 1 , \cdot ) \in L _ { 1 } ( \beta ) } \left\{ \mathbb { E } _ { \alpha } s ( 1 , x _ { 1 } ) - \mathbb { E } _ { \beta } s ( 1 , y ) \right\}
$$

where 127 $s ( t , x ) \in C ^ { 1 , 2 } ( [ 0 , 1 ] \times  { \mathbb { R } } ^ { d } )$ and satisfies HJB PDE $\forall t \in [ 0 , 1 ]$ and $\forall x \in \mathbb { R } ^ { d }$

$$
- \partial _ { t } s ( t , x ) = - \frac { 1 } { 2 } \| \nabla _ { x } s ( t , x ) \| ^ { 2 } + U ( x ) + \frac { \sigma ^ { 2 } } { 2 } t r \{ \nabla ^ { 2 } s ( t , x ) \} .
$$

The first expression in Theorem 1 plays the role of a discriminator and guarantees matching the   
target distribution $\beta$ , and the second one is responsible for the optimality of trajectories. For the   
HJB equation to have a unique solution (in the viscosity sense), we require coercivity (Theorem 4.1   
[Fleeting and Soner, 2006]) of the Hamiltonian for some constants $C _ { 1 } > 0$ and $C _ { 2 } \geq 0$

$$
\begin{array} { l } { \displaystyle H ( x , \nabla s , \nabla ^ { 2 } s ) = \frac { 1 } { 2 } \| \nabla _ { x } s \| ^ { 2 } - U ( x ) - \frac { \sigma ^ { 2 } } { 2 } \mathrm { t r } \{ \nabla ^ { 2 } s \} } \\ { \displaystyle \qquad \geq C _ { 1 } ( \| \nabla s \| ) - C _ { 2 } ( 1 + \| x \| + \| \nabla ^ { 2 } s \| ) } \end{array}
$$

The term $\| \nabla _ { x } s \| ^ { 2 }$ dominates for large values, so in case $U ( x )$ is bounded and $\sigma > 0$ the solution   
is unique. By means of the optimized function $s ( t , x )$ we can generate the OT trajectories using   
Euler-Maruyama algorithm:

$$
x _ { t + \Delta t } = - \nabla _ { x } s ( t , x _ { t } ) \Delta t + \sigma \Delta W , \quad x _ { 0 } \sim \alpha .
$$

Unlike most other methods, here we do not need to model the intermediate density of the $x _ { t }$ $( t \in ( 0 , 1 ) )$ distribution, which greatly simplifies the learning process, but we need to store the generation history in a replay buffer for more stable HJB optimization in high-dimensional spaces.

# 4 Method

To find a stable and balanced solution $s ( t , x )$ for the given dynamic OT problem (1), we can follow a   
composite approach that combines optimal control (via HJB PDE constraints) and RL techniques   
(policy-based trajectory optimization). We approximate the value function using a parametric model   
$s _ { \theta } ( t , x )$ . We have to maximize the potential matching functional (8) subject to the constraint that   
$s _ { \theta } ( t , x )$ satisfies the HJB PDE. For that divide the time interval $[ 0 , 1 ]$ into $T$ time steps and simulate   
$n$ trajectorie $\{ t _ { 0 } ^ { k } , x _ { 0 } ^ { k } , \ldots , t _ { T } ^ { k } , x _ { T } ^ { k } \} _ { k = 1 } ^ { n }$ using initial $\alpha$ distribution and Euler-Maruyama method (12).   
Sample also $n$ points $y _ { k }$ from the target distribution $\beta$

$$
L _ { \mathrm { p o t } } \left( s _ { \theta } \right) = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } { s _ { \theta } ( 1 , x _ { T } ^ { k } ) } - \frac { 1 } { n } \sum _ { k = 1 } ^ { n } { s _ { \theta } ( 1 , y ^ { k } ) } .
$$

The HJB PDE must hold for all $t \in [ 0 , 1 ]$ and $x \in \mathbb { R } ^ { d }$ , but in practice, for more effective training, the   
training data should be sampled in the region of the flow (trajectories) concentration (according to   
Liu et al. [2022]). We enforce this by linear interpolation between datasets from $\alpha$ and $\beta$ as a rough   
estimation of the flow region and subsequently use the replay buffer $\boldsymbol { B }$ to collect points from the   
previously obtained trajectories. Using data samples $\{ t ^ { k } , \boldsymbol { x ^ { \dot { k } } } \} _ { k = 1 } ^ { \bar { n } }$ from $\boldsymbol { B }$ or the linear interpolation   
we compute HJB residual loss as

$$
\begin{array} { r l r } & { } & { L _ { \mathrm { h i b } } ( \boldsymbol { s } _ { \theta } , \boldsymbol { \overline { { s } } } ) = \displaystyle \frac { 1 } { n } \sum _ { k = 1 } ^ { n } \left( \frac { \partial s _ { \theta } ^ { k } } { \partial t } - \frac { 1 } { 2 } \| \nabla _ { x } \boldsymbol { \overline { { s } } } ^ { k } \| ^ { 2 } + U ( \boldsymbol { x } ^ { k } ) + \frac { \sigma ^ { 2 } } { 2 } \mathrm { t r } \{ \nabla ^ { 2 } \boldsymbol { \overline { { s } } } ^ { k } \} + \lambda _ { a } \| a ^ { k } \| \right) ^ { 2 } } \\ & { } & { \displaystyle + \frac { 1 } { n } \sum _ { k = 1 } ^ { n } \left( \frac { \partial \boldsymbol { \overline { { s } } } ^ { k } } { \partial t } - \frac { 1 } { 2 } \| \nabla _ { x } s _ { \theta } ^ { k } \| ^ { 2 } + U ( \boldsymbol { x } ^ { k } ) + \frac { \sigma ^ { 2 } } { 2 } \mathrm { t r } \{ \nabla ^ { 2 } s _ { \theta } ^ { k } \} + \lambda _ { a } \| a ^ { k } \| \right) ^ { 2 } , } \end{array}
$$

where 152 $s _ { \theta } ^ { k } = s _ { \theta } ( t ^ { k } , x ^ { k } )$ , $\overline { { s } } ^ { k } = \overline { { s } } ( t ^ { k } , x ^ { k } )$ denotes the target model with EMA parameters, $a ^ { k }$ is angular 153 acceleration defined as

$$
a ^ { k } = \frac { d } { d t } \frac { \nabla s _ { \theta } ( t ^ { k } , x ^ { k } ) } { \| \nabla s _ { \theta } ( t ^ { k } , x ^ { k } ) \| } .
$$

The angular acceleration with coefficient $\lambda _ { a }$ forces the straightening of the trajectories (optionally).   
We divide the model into $s _ { \theta }$ and $\overline { { s } }$ as it usually done in RL methods to make the optimization problem   
more similar to regression.   
In the result, our model is trained on two criteria ( $\scriptstyle L _ { \mathrm { p o t } }$ and $L _ { \mathrm { h j b . } }$ ) simultaneously and to balance both   
impacts we scale the gradients of the hjb-loss and sum it with the pot-loss:

$$
\nabla _ { \boldsymbol { \theta } } L _ { \mathrm { p o t } } ( s _ { \boldsymbol { \theta } } ) + \lambda _ { \mathrm { h j b } } \mathrm { E M A } \left( \frac { \| \nabla _ { \boldsymbol { \theta } } L _ { \mathrm { p o t } } ( s _ { \boldsymbol { \theta } } ) \| } { \| \nabla _ { \boldsymbol { \theta } } L _ { \mathrm { h j b } } ( s _ { \boldsymbol { \theta } } , \overline { { s } } ) \| } \right) \nabla _ { \boldsymbol { \theta } } L _ { \mathrm { h j b } } ( s _ { \boldsymbol { \theta } } , \overline { { s } } ) .
$$

The complete method is implemented as shown in Algorithm 1. It effectively combines the theo  
retical guarantees of optimal transport with the flexibility of neural network approximations, while   
maintaining numerical stability through careful gradient management. The adaptive balancing of the   
potential matching and HJB residual losses ensures stable convergence to a solution that satisfies both   
163 the optimality conditions and the boundary constraints.

# Algorithm 1 HOTA: Hamiltonian framework for Optimal Transport Advection

1: Input: value model $s _ { \theta }$ , model optimizer s_opt, distributions $\alpha$ and $\beta$ , potential function $U ( x )$ ,   
diffusion coefficient $\sigma$ .   
: Hyperparameters: train steps $N$ , iterpolation sample steps $N _ { 0 }$ , temporal discretization $T$ , batch   
size $n$ , hjb-loss weight $\lambda _ { \mathrm { h j b } }$ , acceleration coefficient $\lambda _ { a }$ , learning rate lr, gradients scale EMA   
coefficient $\tau$ .   
: Initialize target model $\overline { { s } }$ ; replay buffer $B \gets \emptyset$ ; gradients scale $\alpha \gets 1 . 0$   
: for iteration $i = 1$ to $N$ do   
: Sample train data $\{ x _ { 0 } ^ { k } \} _ { k = 1 } ^ { n } \sim \alpha$ ; $\{ y ^ { k } \} _ { k = 1 } ^ { n } \sim \beta$   
: if $i < N _ { 0 }$ then   
: Sample times k}nk=1 ∼ U (0, 1)   
: For $1 \leq k \leq n$ set $x ^ { k } = x _ { 0 } ^ { k } \cdot \left( 1 - t ^ { k } \right) + y ^ { k } \cdot t ^ { k }$   
: else   
: Sample $\{ t ^ { k } , x ^ { k } \} _ { k = 1 } ^ { n } \sim B$   
: 12: end ifGenerate $n$ trajectories $\{ t _ { 0 } ^ { k } , x _ { 0 } ^ { k } , \ldots , t _ { T } ^ { k } , x _ { T } ^ { k } \} _ { k = 1 } ^ { n }$ using current policy $\boldsymbol { v } _ { t } = - \nabla s ( t , \boldsymbol { x } )$   
: Add the 1-st trajectory $\{ t _ { 0 } ^ { 0 } , x _ { 0 } ^ { 0 } , \ldots , t _ { T } ^ { 0 } , x _ { T } ^ { 0 } \}$ to $\boldsymbol { B }$   
: Compute gradients:   
: $g _ { \mathrm { h j b } } \doteq \nabla _ { \theta } \breve { L } _ { \mathrm { h j b } } \bigl ( s _ { \theta } , \overline { { s } } , \{ t ^ { k } , x ^ { k } \} _ { k = 1 } ^ { n } \bigr )$   
: $g _ { \mathrm { p o t } } = \nabla _ { \boldsymbol { \theta } } L _ { \mathrm { p o t } } ( s _ { \boldsymbol { \theta } } , \{ x _ { T } ^ { k } \} _ { k = 1 } ^ { n } , \{ y ^ { k } \} _ { k = 1 } ^ { n } )$   
: Update Parameters:   
: Compute norms $G _ { \mathrm { h j b } } = \| g _ { \mathrm { h j b } } \| _ { 2 }$ and $G _ { \mathrm { p o t } } = \| g _ { \mathrm { p o t } } \| _ { 2 }$   
: EMA update of gradients scale $\alpha = \tau \dot { G } _ { \mathrm { p o t } } / G _ { \mathrm { h j b } } + ( 1 - \tau ) \alpha$   
: Sum the gradients $g = g _ { \mathrm { p o t } } + \lambda _ { \mathrm { h j b } } \alpha g _ { \mathrm { h j b } }$   
: Update model parameters $\theta$ with $\operatorname { s } \_ { \mathrm { o p t } ( g ) }$   
: EMA update of target model $\overline { { s } }$   
: end for

# 164 5 Experiments

In this section, we evaluate our method on a series of distribution matching tasks with non-trivial geometries. In Section 5.2, we compare HOTA with state-of-the-art baselines, demonstrating its superior performance on both standard benchmarks including datasets with almost non-differentiable potentials. In Section 5.3, we demonstrate the scalability of our approach by showcasing its effectiveness in high-dimensional settings. Finally, in Section 5.4, we ablate key components of our method.

# 5.1 Experimental Setup

Evaluation We assess performance using two metrics: feasibility and optimality. Feasibility reflects   
how well the method matches the target distribution, evaluated via Wasserstein distance with squared   
Euclidean cost $( W _ { 2 } ( T _ { \# } \alpha , \beta ) )$ , where the transport mapping $T$ uses optimized value function $s _ { \theta }$ and   
samples $x _ { 1 }$ by procedure (12). Optimality measures the quality of the resulting mapping, estimated   
through the integral trajectory cost: $\begin{array} { r } { \int _ { 0 } ^ { 1 } \left[ \dot { \mathbb { E } } _ { \rho _ { t } } \frac { \| v _ { t } ( x _ { t } ) \| ^ { 2 } } { 2 } + \dot { U } ( x _ { t } ) \right] d t } \end{array}$ , where $x _ { t }$ follows (1).   
Network In all our experiments, we employ a simple MLP augmented with Fourier feature encoding   
of the time component. For general time embeddings of the form $\operatorname { e m b } ( t ) = \sin ( f \cdot t + \varphi )$ , the time   
derivative is given by $\partial _ { t } \mathbf { e m b } ( t ) = \boldsymbol { f } \cdot \cos ( \boldsymbol { f } \cdot \boldsymbol { t } + \varphi )$ . As the frequency $f$ increases, the magnitude   
of this derivative also grows, potentially leading to numerical instability—especially when the time   
derivative of the network is explicitly involved in the objective. This issue has been previously   
discussed in Lu and Song [2024]. To address this, we restrict the frequency range to [1, 20] and   
183 normalize the resulting Fourier features by dividing by the corresponding frequencies.

Baselines We use source code from GSBM repository for running it in our experiments on BabyMaze, Slit and Box datasets. Other results were taken from the original papers Liu et al. [2024], Pooladian et al. [2024] where dataset were previously introduced.

7 All experiments are conducted on a GeForce RTX 3090 GPU and take less than ten minutes for   
training. Additional experimental details are provided in Appendix A.

# 189 5.2 Comparative Evaluation on Two-Dimensional Data

![](images/26b4fc0f60627f0763caaf2df29a47e4b1e8925e98319d01e8e38ac40bc9f1f2.jpg)  
Figure 1: Evaluation of HOTA method on smooth (top) and non-smooth datasets (bottom): Stunnel, Vneck, GMM, BabyMaze, Slit, Box. Blue regions indicate high values of potential $U ( x )$ . Distributions $\alpha$ (red), $\beta$ (black) and the mapped $T _ { \# } \alpha$ (green).

In this section, we compare our method to previous state-of-the-art approaches on the standard bench  
marks including datasets that feature almost non-differentiable potential functions. Visualizations   
of the datasets are provided in Figure 1. The first three datasets—Stunnel, Vneck, and GMM—are   
adopted from Liu et al. [2024]. These benchmarks incorporate state cost functions $U ( x _ { t } )$ that en  
courage the optimal solution to respect complex geometric constraints. Each dataset is designed to   
highlight specific capabilities of the evaluated algorithms. Stunnel assesses whether a method can   
capture drift fields that undergo rapid and localized changes. Vneck evaluates the ability to model   
drift that compresses and expands the support of marginal distributions. GMM tests whether the   
method can disambiguate closely situated points and assign them to distinct trajectories. The re  
maining datasets—BabyMaze, Slit, and Box (Pooladian et al. [2024])—are constructed using similar   
underlying principles but pose additional difficulties due to the presence of almost non-differentiable   
state cost functions. A summary of the quantitative results across all datasets is provided in Table 1.   
Our method, HOTA, consistently outperforms existing approaches in terms of both feasibility and   
optimality. In particular, HOTA achieves a substantial performance gain on the GMM dataset, which   
may refer to its superior capability in trajectory separation for closely situated points.

# 5.3 Scalability to High-Dimensional Spaces

In this section, we test the scalability of our method, demonstrating its stable performance in higher  
dimensional settings. For this purpose, we use Sphere datasets parameterized by data dimensionality   
$N$ . Specifically, we define an $N$ -dimensional unit sphere as a potential barrier inducing corresponding   
209 state cost function $U ( x _ { t } )$ . The source and target distributions are samples from a standard distribution   
10 located at the poles, projected onto the unit sphere. The three-dimensional case is visualized in   
Figure 2a. The performance of our method across varying data dimensions is shown in Figure 2b   
(right). Notably, HOTA demonstrates robust and stable performance as the dimensionality $N$   
213 increases.

Table 1: Quantitative comparison between recent state-of-the-art methods and our approach, HOTA. Performance is evaluated using two criteria: Feasibility (how well the target distribution is covered) and Optimality (efficiency of the learned mapping). Our method consistently outperforms existing approaches, with significantly better results in certain tasks, such as GMM. N/A cells indicate that original authors of particular method did not include results for those tasks. The mean and the standard deviations of our method are computed across 5 different seeds. Best values are highlighted by bold font (lower is better). Gray values correspond to the method’s divergence.   

<table><tr><td rowspan="2"></td><td colspan="3">Feasibility W2(T#(α), β)</td><td colspan="3">Optimality (integral cost)</td></tr><tr><td>Stunnel</td><td>Vneck</td><td>GMM</td><td>Stunnel</td><td>Vneck</td><td>GMM</td></tr><tr><td>NLSB</td><td>30.54</td><td>0.02</td><td>67.76</td><td>207.06</td><td>147.85</td><td>4202.71</td></tr><tr><td>GSBM</td><td>0.03</td><td>0.01</td><td>4.13</td><td>460.88</td><td>155.53</td><td>229.12</td></tr><tr><td>HOTA</td><td>0.006±0.003</td><td>0.002±0.001</td><td>0.19±0.05</td><td>320.90±12.5</td><td>115.09±8.9</td><td>80.44±2.6</td></tr><tr><td></td><td>BabyMaze</td><td>Slit</td><td>Box</td><td>BabyMaze</td><td>Slit</td><td>Box</td></tr><tr><td>NLSB</td><td>&gt;1</td><td>0.013</td><td>0.024</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>NLOT</td><td>&gt;1</td><td>0.013</td><td>0.016</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>GSBM</td><td>0.01</td><td>0.01</td><td>0.02</td><td>6.5</td><td>4.9</td><td>3.8</td></tr><tr><td>HOTA</td><td>0.004±0.003</td><td>0.0004±0.0001</td><td>0.002±0.001</td><td>4.87±0.14</td><td>3.06±0.09</td><td>2.84±0.11</td></tr></table>

![](images/ba2c7d27462d0852f299f8563b69c40ae657363707f07c19ad7cb73db6d3c83c.jpg)  
Figure 2: (a) Visualization of Sphere dataset for $N = 3$ . (b) Feasibility and Optimality trends with respect to $3 D$ unit sphere smoothness (left) and unit sphere dimensionality (right). Our method maintains robust performance across both non-differentiable potentials and high-dimensional settings.

# 5.4 Ablation study

Table 2 presents comparison of the full HOTA model against variants without the replay buffer $\boldsymbol { B }$ that stores simulation history or the adaptive gradient balancing by means of $\alpha$ (17), evaluating as previously feasibility and optimality metrics across Stunnel, Vneck, and GMM datasets. The full HOTA achieves strong metric scores, while removing the buffer severely degrades feasibility in Vneck and GMM and increases costs in Stunnel. Disabling gradient balancing harms feasibility in Stunnel and GMM. The results highlight the buffer’s critical role in maintaining feasibility and the nuanced trade-offs between gradient balancing and transport efficiency across different scenarios.

22 Additionally we have evaluated the influence of acceleration term $\lambda _ { a } \| a \|$ used in loss $L _ { \mathrm { h j b } }$ depending   
23 on $\lambda _ { a }$ (Figure 3). It performs the function of straightening trajectories by penalizing the change   
4 in angular velocity. It follows from the results that increasing $\lambda _ { a }$ improves the optimality of the   
transportation trajectories while introducing a small bias in the matching of the target distribution   
$\beta$ , which is reflected in the feasibility metric. In the GMM task, due to the specificity of the dataset

and the divergence of trajectories in different directions, a small penalization of acceleration also improves feasibility.

Table 2: Comparison of HOTA method against variants without the replay buffer $\boldsymbol { B }$ and the adaptive gradient balancing. Best values are highlighted by bold font (lower is better). Gray values correspond to the method’s divergence.   

<table><tr><td rowspan="2"></td><td colspan="3">Feasibility W2(T#(α), β)</td><td colspan="3"> Optimality (integral cost)</td></tr><tr><td>Stunnel</td><td>Vneck</td><td>GMM</td><td>Stunnel</td><td>Vneck</td><td>GMM</td></tr><tr><td>HOTA</td><td>0.006</td><td>0.002</td><td>0.19</td><td>320.90</td><td>115.09</td><td>80.44</td></tr><tr><td>HOTA w/o buffer</td><td>0.076</td><td>16.47</td><td>1.248</td><td>706.89</td><td>82.49</td><td>121.6</td></tr><tr><td>HOTA w/o grad. balancing</td><td>3.60</td><td>0.026</td><td>2.64</td><td>325.22</td><td>109.25</td><td>72.77</td></tr></table>

![](images/e2156aa2e6aee37b62b3235622bdefc5fee7f5c50924a14376a45b3ac94d2975.jpg)  
Figure 3: Impact of acceleration coefficient $\lambda _ { a }$ . Left: Stunnel, right: GMM datasets.

# 229 6 Proof of Theorem 1 (Dual Formulation of GSB)

We prove in the first step the equivalence between the GSB (stochastic control formulation) and its   
dual formulation using Kantorovich-style duality. Remind that we consider the stochastic process   
$x _ { t }$ (1) with conditions $x _ { 0 } \sim \alpha$ , $x _ { 1 } \sim \beta$ , control function $v ( t , x _ { t } )$ , and Brownian motion $\sigma ( t , x _ { t } ) d W _ { t }$ .   
The primary problem of GSB optimization is:

$$
\operatorname* { i n f } _ { v ( x , t ) } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \mathcal { L } ( t , x _ { t } , v _ { t } ) d t \right] \quad \mathrm { s . t . } \quad x _ { 0 } \sim \alpha , x _ { 1 } \sim \beta ,
$$

where in the particular case 234 $\mathcal { L } ( t , x , v ) = v ^ { 2 } / 2 + U ( x )$ . Since the stochastic process $x _ { t }$ starts from 235 $x _ { 0 } \sim \alpha$ the primal problem is equivalent to:

$$
\operatorname* { i n f } _ { v ( t , x ) } \left( \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \mathcal { L } ( t , x _ { t } , v _ { t } ) d t \right] + \operatorname* { s u p } _ { g \in L _ { 1 } ( \beta ) } \left( - \mathbb { E } [ g ( x _ { 1 } ) ] + \mathbb { E } _ { \beta } [ g ( y ) ] \right) \right) ,
$$

where the supremum over $g$ enforces the constraint $x _ { 1 } \sim \beta$ (via Lagrange duality). Rewrite the   
Lagrangian problem as

$$
\operatorname* { i n f } _ { v ( t , x ) } \operatorname* { s u p } _ { g \in L _ { 1 } ( \beta ) } \left( \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \mathcal { L } ( t , x _ { t } , v _ { t } ) d t - g ( x _ { 1 } ) \right] + \mathbb { E } _ { \beta } [ g ( y ) ] \right) .
$$

Assuming strong duality holds under mild regularity conditions (e.g., $\mathcal { L }$ convex in $v , \alpha , \beta$ absolutely   
continuous), we swap inf and sup:

$$
\operatorname* { s u p } _ { g \in L _ { 1 } ( \beta ) } \left( \operatorname* { i n f } _ { v ( t , x ) } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \mathcal { L } ( t , x _ { t } , v _ { t } ) d t - g ( x _ { 1 } ) \right] + \mathbb { E } _ { \beta } [ g ( y ) ] \right) .
$$

Note that since the optimal $v ^ { * } ( t , x )$ is Markovian (depends only on current time $t$ and state $x$ ) and   
does not depend on the initial distribution $\alpha$ it holds that

$$
\mathbb { E } \left[ \int _ { 0 } ^ { 1 } \mathcal { L } ( t , x _ { t } , v _ { t } ^ { * } ) d t - g ( x _ { 1 } ) \right] = \mathbb { E } _ { x \sim \alpha } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \mathcal { L } ( t , x _ { t } , v _ { t } ^ { * } ) d t - g ( x _ { 1 } ) \bigg | x _ { 0 } = x \right] .
$$

Buy the definition of $c$ -conjugate transform (5):

$$
\mathbb { E } _ { x \sim \alpha } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \mathcal { L } ( t , x _ { t } , v _ { t } ^ { * } ) d t - g ( x _ { 1 } ) \Big | x _ { 0 } = x \right] = \mathbb { E } _ { x \sim \alpha } g ^ { c } ( x ) .
$$

Thus, the dual problem becomes: 243 $\begin{array} { r } { \operatorname* { s u p } _ { g \in L _ { 1 } ( \beta ) } \left( \mathbb { E } _ { \alpha } [ g ^ { c } ( x ) ] + \mathbb { E } _ { \beta } [ g ( y ) ] \right) . } \end{array}$ In the second step find the 244 optimal control solution $v ^ { * } ( t , x )$ by means of dynamic programming principle. Define the value 245 function $s ( t , x )$ that for any $0 \leq t \leq \tau \leq 1$ satisfies:

$$
s ( t , x ) = \operatorname* { i n f } _ { v ( t , x ) } \mathbb { E } \left[ \int _ { t } ^ { \tau } \mathcal { L } ( z , x _ { z } , v _ { z } ) d z + s ( \tau , x _ { \tau } ) \bigg | x _ { t } = x \right] .
$$

Applying Ito’s formula to $s ( \tau , x _ { \tau } )$ we obtain that

$$
\begin{array} { l } { \displaystyle \mathrm { d } s ( \tau , x _ { \tau } ) = \partial _ { \tau } s \mathrm { d } \tau + \nabla s \cdot \mathrm { d } x _ { \tau } + \frac { 1 } { 2 } \mathrm { t r } ( \sigma ^ { 2 } \nabla ^ { 2 } s ) \mathrm { d } \tau } \\ { \displaystyle = \left( \partial _ { \tau } s + \nabla s ^ { T } v _ { \tau } + \frac { 1 } { 2 } \mathrm { t r } ( \sigma ^ { 2 } \nabla ^ { 2 } s ) \right) \mathrm { d } \tau + \nabla s ^ { T } \sigma \mathrm { d } W _ { s } . } \end{array}
$$

Consider the evolution of the value between times $t$ and $\tau$ :

$$
s ( \tau , x _ { \tau } ) - s ( t , x _ { t } ) = \int _ { t } ^ { \tau } \left( \partial _ { z } s + \nabla s \cdot v _ { z } + \frac { 1 } { 2 } \mathrm { t r } ( \sigma ^ { 2 } \nabla ^ { 2 } s ) \right) \mathrm { d } z + \int _ { t } ^ { \tau } \nabla s ^ { T } \sigma \mathrm { d } W .
$$

Basing on the martingale property of Ito integrals248 $\begin{array} { r } { ( \mathbb { E } [ \int \nabla s \cdot \sigma \mathrm { d } W | x _ { t } = x ] = 0 , } \end{array}$ ) it holds that

$$
\mathbb { E } [ s ( \tau , x _ { \tau } ) | x _ { t } = x ] = s ( t , x ) + \mathbb { E } \left[ \int _ { t } ^ { \tau } \left( \partial _ { z } s + \nabla s \cdot v _ { z } + \frac { 1 } { 2 } \mathrm { t r } ( \sigma ^ { 2 } \nabla ^ { 2 } s ) \right) \mathrm { d } z \right] .
$$

Substitute back into dynamic programming and plug the last expression into the equation (24):

$$
s ( t , x ) = \operatorname* { i n f } _ { v ( t , x ) } \mathbb { E } \left[ \int _ { t } ^ { \tau } \mathcal { L } ( z , x _ { z } , v _ { z } ) \mathrm { d } z + s ( t , x ) + \int _ { t } ^ { \tau } \left( \partial _ { z } s + \nabla s ^ { T } v _ { \tau } + \frac { 1 } { 2 } \mathbf { t r } ( \sigma ^ { 2 } \nabla ^ { 2 } s ) \right) \mathrm { d } z \right] .
$$

250 Cancel $s ( t , x )$ from both sides and divide by $( \tau - t )$ :

$$
0 = \operatorname* { i n f } _ { v ( s , t ) } \frac { 1 } { \tau - t } \mathbb { E } \left[ \int _ { t } ^ { \tau } \left( \mathcal { L } ( z , x _ { z } , v _ { z } ) + \partial _ { z } s + \nabla s ^ { T } v _ { z } + \frac { 1 } { 2 } \mathrm { t r } ( \sigma ^ { 2 } \nabla ^ { 2 } s ) \right) \mathrm { d } z \right] .
$$

251 Take limit $\tau \downarrow t$ to derive the HJB equation for a general Lagrangian $\mathcal { L }$

$$
0 = \operatorname* { i n f } _ { v } \left\{ { \mathcal { L } } ( t , x , v ) + \partial _ { t } s + \nabla s ^ { T } v + { \frac { 1 } { 2 } } \mathrm { t r } ( \sigma ^ { 2 } \nabla ^ { 2 } s ) \right\} .
$$

Identify optimal control for the particular $\mathcal { L } ( t , x , v ) = v ^ { 2 } / 2 + U ( x )$ . The infimum is attained when $\boldsymbol { v } ^ { * } = - \boldsymbol { \nabla } s$ , yielding the final result of Theorem 1.

# 7 Limitations and Future Work

While HOTA exhibits strong and robust performance, we observed sensitivity to certain network design choices—particularly the Fourier feature encoding of time, a commonly used technique in models that estimate ODE drifts. Additionally, because the value function in our framework must simultaneously support optimal control estimation and serve as a Kantorovich potential, it requires a network architecture capable of aggregating rich temporal and spatial information. The use of a simple MLP, while effective, may not be optimal from an optimization standpoint. Incorporating architectures with stronger inductive biases could further enhance performance. These considerations lie beyond the scope of this work, but we believe they offer promising directions for future research.

# 8 Conclusion

In this work, we introduced HOTA, a new OT method based on the Hamilton–Jacobi–Bellman (HJB) framework for solving the Generalized Schrödinger Bridge problem. We demonstrated that HOTA consistently outperforms recent state-of-the-art methods on standard benchmarks and scales effectively to high-dimensional settings. Remarkably, it works for non-smooth potentials and with non-differentiable cost functions, yielding robust performance gain in terms of strictly defined concepts of feasibility and optimality.

References   
Arip Asadulaev, Rostislav Korst, Aleksandr Korotin, Vage Egiazarian, Andrey Filchenkov, and Evgeny Burnaev.   
Rethinking optimal transport in offline reinforcement learning. Advances in Neural Information Processing   
Systems, 37:123592–123607, 2024.   
Grigory Bartosh, Dmitry P Vetrov, and Christian Andersson Naesseth. Neural flow diffusion models: Learnable   
forward process for improved diffusion modelling. Advances in Neural Information Processing Systems, 37:   
73952–73985, 2024.   
Jean-David Benamou and Yann Brenier. A computational fluid mechanics solution to the monge-kantorovich   
mass transfer problem. Numerische Mathematik, 84(3):375–393, 2000.   
Denis Blessing, Julius Berner, Lorenz Richter, and Gerhard Neumann. Underdamped diffusion bridges with   
applications to sampling. In The Thirteenth International Conference on Learning Representations, 2025.   
URL https://openreview.net/forum?id=Q1QTxFm0Is.   
Maksim Bobrin, Nazar Buzun, Dmitrii Krylov, and Dmitry V Dylov. Align your intents: Offline imitation   
learning via optimal transport. arXiv preprint arXiv:2402.13037, 2024.   
Nicolas Bonneel and Julie Digne. A survey of optimal transport for computer graphics and computer vision.   
Computer Graphics Forum, 42(2):439–460, 2023. doi: https://doi.org/10.1111/cgf.14778. URL https:   
//onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14778.   
Charlotte Bunne, Laetitia Papaxanthos, Andreas Krause, and Marco Cuturi. Proximal optimal transport   
modeling of population dynamics. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera, editors,   
Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, volume 151 of   
Proceedings of Machine Learning Research, pages 6511–6528. PMLR, 28–30 Mar 2022. URL https:   
//proceedings.mlr.press/v151/bunne22a.html.   
Nazar Buzun, Maksim Bobrin, and Dmitry V Dylov. Expectile regularization for fast and accurate training of   
neural optimal transport. Advances in Neural Information Processing Systems, 37:119811–119837, 2024.   
Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information   
processing systems, 26, 2013.   
Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky TQ Chen. Adjoint matching: Fine-tuning flow   
and diffusion generative models with memoryless stochastic optimal control. arXiv preprint arXiv:2409.08861,   
2024a.   
Carles Domingo-Enrich, Jiequn Han, Brandon Amos, Joan Bruna, and Ricky T. Q. Chen. Stochastic optimal   
control matching. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024b.   
URL https://openreview.net/forum?id=wfU2CdgmWt.   
Wendell H. Fleeting and H. Mete Soner. Controlled markov processes and viscosity solutions (2nd ed.). Springer,   
2006.   
Kacper Kapusniak, Peter Potaptchik, Teodora Reu, Leo Zhang, Alexander Tong, Michael Bronstein, Joey Bose,   
and Francesco Di Giovanni. Metric flow matching for smooth interpolations on the data manifold. Advances   
in Neural Information Processing Systems, 37:135011–135042, 2024.   
Pascal Klink, Haoyi Yang, Carlo D’Eramo, Jan Peters, and Joni Pajarinen. Curriculum reinforcement learning   
via constrained optimal transport. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari,   
Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning,   
volume 162 of Proceedings of Machine Learning Research, pages 11341–11358. PMLR, 17–23 Jul 2022.   
URL https://proceedings.mlr.press/v162/klink22a.html.   
Alexander Korotin, Daniil Selikhanovych, and Evgeny Burnaev. Neural optimal transport. arXiv preprint   
arXiv:2201.12220, 2022.   
Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. CoRR,   
abs/1805.00909, 2018.   
Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for   
generative modeling. In The Eleventh International Conference on Learning Representations, 2023.   
Guan-Horng Liu, Tianrong Chen, Oswin So, and Evangelos A Theodorou. Deep generalized schrödinger bridge.   
In Advances in Neural Information Processing Systems, 2022.   
Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos Theodorou, and Ricky T. Q.   
Chen. Generalized schrödinger bridge matching. In The Twelfth International Conference on Learning   
Representations, 2024.   
Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. 2024.   
Ashok Makkuva, Amirhossein Taghvaei, Sewoong Oh, and Jason Lee. Optimal transport mapping via input   
convex neural networks. In International Conference on Machine Learning, pages 6672–6681. PMLR, 2020.   
Eduardo Fernandes Montesuma, Fred Maurice Ngole Mboula, and Antoine Souloumiac. Recent advances in   
optimal transport for machine learning. IEEE Transactions on Pattern Analysis and Machine Intelligence,   
.   
Benjamin Nachman Nathan T. Suri, Vinicius Mikuni. Wotan: Weakly-supervised optimal transport attention  
based noise mitigation. NeurIPS 2024, 2024.   
Kirill Neklyudov, Rob Brekelmans, Alexander Tong, Lazar Atanackovic, Qiang Liu, and Alireza Makhzani. A   
computational framework for solving Wasserstein lagrangian flows. In Ruslan Salakhutdinov, Zico Kolter,   
Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings   
of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning   
Research, pages 37461–37485. PMLR, 21–27 Jul 2024.   
Luiz Manella Pereira and M Hadi Amini. A survey on optimal transport for machine learning: Theory and   
applications. IEEE Access, 2025.   
Gabriel Peyré, Marco Cuturi, et al. Computational optimal transport: With applications to data science.   
Foundations and Trends® in Machine Learning, 11(5-6):355–607, 2019.   
Aram-Alexandre Pooladian, Carles Domingo-Enrich, Ricky T. Q. Chen, and Brandon Amos. Neural optimal   
transport with lagrangian costs. In The 40th Conference on Uncertainty in Artificial Intelligence, 2024.   
Thomas Rupf, Marco Bagatella, Nico Gürtler, Jonas Frey, and Georg Martius. Zero-shot offline imitation   
learning via optimal transport, 2025. URL https://openreview.net/forum?id=vDecbmWf6w.   
Filippo Santambrogio. Optimal transport for applied mathematicians, volume 87. Springer, 2015.   
Evangelos Theodorou, Jonas Buchli, and Stefan Schaal. Learning policy improvements with path integrals. In   
Yee Whye Teh and Mike Titterington, editors, Proceedings of the Thirteenth International Conference on   
Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pages 828–835,   
Chia Laguna Resort, Sardinia, Italy, 13–15 May 2010. PMLR.   
Cédric Villani et al. Optimal transport: old and new, volume 338. Springer, 2008.

# A Additional Experimental Details

Hyperparameters Table 3 summarizes the hyperparameters used for each dataset presented in the paper. Note that the Sphere datasets, which are parameterized by data dimensionality, share all hyperparameters except for the potential weight, which may take value 10 for the low dimensions are 30 for high ones.

Table 3: Hyperparameters used for each dataset presented in the paper.   

<table><tr><td>Hyperparameter</td><td>Stunnel Vneck</td><td>GMM</td><td>BabyMaze</td><td>Slit</td><td>Box</td><td>Sphere</td></tr><tr><td>MLP hidden layers Fourier frequencies</td><td colspan="6">[512, 512,512,1]</td></tr><tr><td>optimizer</td><td colspan="6">{1,...,20} Adam with cosine annealing (α = le-2)</td></tr><tr><td>initial learning rate</td><td colspan="6">5×10-4</td></tr><tr><td>Adam [β1,β2]</td><td colspan="6">[0.9,0.99]</td></tr><tr><td># training iterations</td><td colspan="6">70000</td></tr><tr><td>batch size</td><td colspan="6"></td></tr><tr><td>EMA decay, T</td><td colspan="6">1024</td></tr><tr><td></td><td colspan="6">0.9</td></tr><tr><td># control steps</td><td colspan="6">30</td></tr><tr><td>diffusion coef., o</td><td>0.2</td><td>0.1</td><td>0.03</td><td>0.05</td><td>0.03</td><td>0.01</td></tr><tr><td>control weight, 入a</td><td>2.0</td><td>0.7</td><td>0.5</td><td>2.0</td><td>0.3</td><td>0.4</td></tr><tr><td>acc. weight, 入a</td><td>0.0001 0.001</td><td>0.2</td><td>0.05</td><td>0.001</td><td>0.01</td><td>0</td></tr><tr><td>potential weight</td><td>25</td><td>1000 25</td><td>10</td><td>30</td><td>700</td><td>{10,30}</td></tr></table>

# 355 NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer:[Yes]

Justification: We provide theoretical derivation of the dual SGB problem. We made a thorough experiments on established benchmarks and compare proposed method with the other GSB solvers. In all extensive tests, HOTA outperforms the competition both in terms of feasibility and optimality.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the paper.   
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.   
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provided discussions on limitations in Section 6 of main paper.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.   
• The authors are encouraged to create a separate "Limitations" section in their paper.   
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.   
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.   
The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.   
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We provided a theoretical proof of dual GSB task formulation.

Guidelines:

• The answer NA means that the paper does not include theoretical results.   
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.   
• All assumptions should be clearly stated or referenced in the statement of any theorems.   
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.   
• Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Along with detailed hyperparameter specifications in the Appendix, we included easy to follow Jupyter notebook which can be found in supplementary materials, enabling the others to fully reproduce the results in the paper.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.   
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We provided a fully reproducible code, written in JAX framework. Moreover, we provided step-by-step jupyter notebook, showcasing the performance of the proposed algorithm in all discussed tasks.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.   
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.   
• While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).   
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.   
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.   
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).   
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We provided detailed hyperparameters specifications in the Appendix for each of the tested benchmarks.

Guidelines:

• The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All reported results are statistically significant. We include evaluation error (StDev) for each model and each dataset in the study across different runs and seeds.

Guidelines:

• The answer NA means that the paper does not include experiments. • The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).   
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)   
• The assumptions made should be given (e.g., Normally distributed errors).   
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.   
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a $96 \%$ CI, if the hypothesis of Normality of errors is not verified.   
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).   
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We include the exact computer configuration in Appendix and mention GPU model in the main text.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.   
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

# 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We read it and adhered to the ethical guidelines.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.   
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.   
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: Our paper does not address the societal impact as we operate with common datasets and benchmarks for testing GSB solvers.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.

• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.   
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.   
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.   
The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.   
If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: Not applicable to this work.

Guidelines:

• The answer NA means that the paper poses no such risks.   
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We properly refer to the original papers and use the open source codes from official repositories, providing the URLs to them.

Guidelines:

• The answer NA means that the paper does not use existing assets.   
• The authors should cite the original paper that produced the code package or dataset.   
• The authors should state which version of the asset is used and, if possible, include a URL.   
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.   
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.   
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.   
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.   
• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

# 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We include all of the details corresponding to train procedures, datasets used, and citations. Moreover, we provide a readme file for the repository details. The released code is legally approved for the publication; no special documentation is needed.

Guidelines:

• The answer NA means that the paper does not release new assets.   
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
• The paper should discuss whether and how consent was obtained from people whose asset is used.   
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No crowdsourcing was used in this study.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No human studies/IRB was needed for this study.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.   
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

# 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This research does not involve LLMs.

Guidelines:

• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
