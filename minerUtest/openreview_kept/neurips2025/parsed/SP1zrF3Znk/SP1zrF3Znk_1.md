## LINE 166-167

function $f _ { \theta }$ will perform well on unseen data?

$$
\operatorname* { P r } _ { S \sim \mathcal { D } ^ { m } } \big \{ \mathcal { L } ( \pmb { \theta } ) \leq \hat { \mathcal { L } } _ { S } ( \pmb { \theta } ) + \epsilon \big \} \ \geq \ 1 - \delta .
$$

Concrete PAC bounds specify how large $m$ must be (or how large the gap $\epsilon$ can be) in terms of prop  
erties of the hypothesis class—e.g. VC-dimension, Rademacher complexity, stability, compression,   
etc. All of those treat $f _ { \theta }$ as a deterministic output of the algorithm.   
The PAC-Bayesian framework [8, 9, 10, 17, 11] extends the PAC learning paradigm to analyze the   
generalization performance of stochastic learning algorithms. Instead of selecting a single hypothesis,   
this approach considers a distribution over a set of candidate models. Let $\Theta$ denote the set of   
parameters defining a family of prediction functions $\{ f _ { \theta } : \mathcal { X } \to \mathcal { Y } \} _ { \theta \in \Theta }$ . Prior to observing data, a   
prior distribution $\mu \in { \mathcal { P } } ( \Theta )$ is specified over $\Theta$ . Upon receiving a training sample $S \sim \mathcal { D } ^ { m }$ , the   
learning algorithm selects a posterior distribution $\rho \in \mathcal { P } ( \Theta )$ , potentially dependent on $S$ . PAC  
Bayesian theory provides high-probability bounds on the population Gibbs risk $\mathbb { E } _ { f _ { \theta } \sim \rho } [ \mathcal { L } ( \pmb { \theta } ) ]$ in terms   
of the empirical Gibbs risk $\mathbb { E } _ { f _ { \theta } \sim \rho } [ \hat { \mathcal { L } } _ { S } ( \pmb { \theta } ) ]$ and an additional term that measures the dependence of   
the posterior distribution $\rho$ . This additional term involves an information measure—typically the   
Kullback-Leibler divergence $\operatorname { K L } ( \rho \| \mu )$ —between the data-dependent posterior $\rho \in \mathcal { P } ( \Theta )$ and a prior   
$\mu \in { \mathcal { P } } ( \Theta )$ , chosen independently of the data. Formally, for any $\lambda > 0$ and with probability at least   
$1 - \delta$ over the choice of the training sample $S$ , the following inequality holds:

$$
\underset { f _ { \theta } \sim \rho } { \mathbb { E } } [ \mathcal { L } ( \pmb { \theta } ) ] \ \leq \ \underset { f _ { \theta } \sim \rho } { \mathbb { E } } [ \hat { \mathcal { L } } _ { S } ( \pmb { \theta } ) ] \ + \ \frac { 1 } { \lambda } \Big ( \mathrm { K L } ( \rho \| \mu ) + \ln { \frac { 1 } { \delta } } + \Psi _ { \ell , \mu } ( \lambda , n ) \Big )
$$

$$
\Psi _ { \ell , \mu } ( \lambda , m ) = \ln \underset { f _ { \theta } \sim \mu } { \mathbb { E } } \Big [ \mathrm { e x p } \big ( \lambda \big ( \mathcal { L } ( \pmb { \theta } ) - \hat { \mathcal { L } } _ { S } ( \pmb { \theta } ) \big ) \Big ]
$$

Compared with classical PAC guarantees, PAC
