# Linear Bandits with Non-i.i.d. Noise

Anonymous Author(s)   
Affiliation   
Address   
email

# Abstract

We study the linear stochastic bandit problem, relaxing the standard i.i.d. assump  
tion on the observation noise. As an alternative to this restrictive assumption,   
we allow the noise terms across rounds to be sub-Gaussian but interdependent,   
with dependencies that decay over time. To address this setting, we develop new   
confidence sequences using a recently introduced reduction scheme to sequential   
probability assignment, and use these to derive a bandit algorithm based on the   
principle of optimism in the face of uncertainty. We provide regret bounds for   
the resulting algorithm, expressed in terms of the decay rate of the strength of   
dependence between observations. Among other results, we show that our bounds   
recover the standard rates up to a factor of the mixing time for geometrically mixing   
observation noise.

# 12 1 Introduction

The linear bandit problem (Abe and Long, 1999; Auer, 2003) is an instance of a multi-armed bandit   
framework, where the expected reward is linear in the feature vector representing the chosen arm.   
More concretely, it is a sequential decision-making problem, where an agent each round picks an arm   
$X _ { t }$ , and receives a reward $Y _ { t } = \langle \theta ^ { \star } , X _ { t } \rangle + \varepsilon _ { t }$ , with $\theta ^ { \star }$ a fixed parameter unknown to the agent, and   
$\varepsilon _ { t }$ zero-mean random noise. This framework has gained significant attention in the literature as it   
yields analytic tools that can be applied to several concrete applications, such as online advertising   
(Abe et al., 2003), recommendation systems (Li et al., 2010; Korkut and Li, 2021), and dynamic   
pricing (Cohen et al., 2020).   
A popular strategy to tackle linear bandits leverages the principle of optimism in the face of uncertainty,   
via upper confidence bound (UCB) algorithms. The idea of optimism can be traced back to Lai and   
Robbins (1985), and its application to linear bandits was already advanced by Auer (2003). Since   
then, this approach has been improved and analysed by several works (Abbasi-Yadkori et al., 2011;   
Lattimore and Szepesvári, 2020; Flynn et al., 2023). This class of methods requires constructing an   
adaptive sequence of confidence sets that, with high probability, contain the true parameter $\theta ^ { \star }$ . Each   
round, the agent selects the arm maximising the expected reward under the most optimistic parameter   
(in terms of reward) in the current confidence set. UCB-based algorithms have become popular as   
they are often easy to implement and come with tight worst-case regret guarantees.   
For a UCB algorithm to perform well, it is necessary that the confidence sets are tight, which can be   
ensured by taking advantage of the structure of the problem. In this paper, our focus is on studying   
various assumptions on the observation noise. A commonly studied situation is when $( \varepsilon _ { t } ) _ { t \geq 0 }$ consists   
of a sequence of i.i.d. realisations of some bounded or sub-Gaussian random variable (see Lattimore   
and Szepesvári, 2020, Chapter 20). Often, the standard analysis can be extended to the case in which   
the realisation are not independent, but conditionally centred and sub-Gaussian (Abbasi-Yadkori   
et al., 2011). Yet, in real-world settings, this assumption is often unrealistic, as one can expect the   
presence of interdependencies among the noise at different rounds. For instance, in the context   
of advertisement selection, the noise models the ensemble of external factors that influence the   
user’s choice on whether to click or not an ad. The i.i.d. assumption implies that across different   
rounds these external factors are completely independent. In practice, the user choice will be affected   
by temporally correlated events, such as recent browsing history or exposure to similar content.   
Therefore, a more realistic assumption is to allow the dependencies to decay with time, rather than   
being completely absent. This way to model dependencies, often referred to as mixing, is common to   
study concentration for sums of non-i.i.d. random variables, with applications to machine learning   
(Bradley, 2005; Mohri and Rostamizadeh, 2008; Abélès et al., 2025).   
In the present paper we relax the assumption that the noise is conditionally zero-mean in the bandit   
problem, and we allow for the presence of dependencies. Concretely, we replacethe standard   
conditionally sub-Gaussian setting with a more general formulation that accounts for conditional   
dependence of the noise on the past, by introducing a natural notion of mixing sub-Gaussianity. Within   
this context, we introduce a UCB algorithm for which we rigorously establish regret guarantees.   
There are two key challenges for our approach: constructing a valid confidence sequence under   
dependent noise, and deriving a regret upper bound for the UCB algorithm that we propose.   
We derive the confidence sequence by adapting the online-to-confidence-sets technique to accommo  
date temporal dependencies in the noise. This approach, originally introduced by Abbasi-Yadkori   
et al. (2011) and recently extended and improved (Jun et al., 2017; Lee et al., 2024; Clerico et al.,   
2025), involves constructing an abstract online learning game whose regret guarantees can be turned   
into a confidence sequence. To deal with the dependencies in the noise, we modify the standard   
online-to-confidence-sets framework by introducing delays in the feedback received within the ab  
stract online game. This approach is inspired by the recent work of Abélès et al. (2025) on extending   
online-to-PAC conversions to non-i.i.d. mixing data sets in the context of deriving generalisation   
bounds for statistical learning. There, a delayed-feedback trick similar to ours is employed to derive   
statistical guarantees (generalisation bounds) from an abstract online learning game.   
For the regret analysis of the bandit algorithm, we also need to face some challenges due to the   
correlated observation noise. We address these by introducing delays into the decision-making policy   
as well. This makes our approach superficially similar to algorithms used in the rich literature on   
bandits with delayed feedback (see, e.g., Vernade et al., 2020a; Howson et al., 2023). These works   
consider delay as part of the problem statement and not part of the solution concept, and are thus   
orthogonal to our work. In particular, a simple adaptation of results from this literature would not   
suffice for dealing with dependent observations, which we tackle by developing new concentration   
inequalities. Another line of work that is conceptually related to ours is that of non-stationary bandits   
(Garivier and Moulines, 2008; Russac et al., 2019). In that setting, the parameter vector $\theta _ { t } ^ { \star }$ evolves in   
time according to a nonstationary stochastic process, and the observation noise remains i.i.d., once   
again making for a rather different problem with its own challenges. Namely, the main obstacle   
to overcome is that comparing with the optimal sequence of actions becomes impossible unless   
strong assumptions are made about the sequence of parameter vectors. A typical trick to deal with   
these nonstationarities is to discard old observations (which may have been generated by a very   
different reward function), and use only recent rewards for decision-making. This is the polar opposite   
of our approach that is explicitly disallowed to use recent rewards, which clearly highlights how   
different these problems are. That said, there exists an intersection between the worlds of delayed   
and nontationary bandits (Vernade et al., 2020b), and thus we would not discard the possibility of   
eventually building a bridge between bandits with nonstationary reward functions and bandits with   
nonstationary observation noise. For simplicity, we focus on the second of these two components in   
this paper.   
Notation. Throughout the paper, we will often use the following notations. For $u$ and $v$ in $\mathbb { R } ^ { p }$ , we   
let $\langle u , v \rangle$ denote their dot product. $\| u \| _ { 2 } = \sqrt { \langle u , u \rangle }$ is the Euclidean norm, while for a non-negative   
definite $( p \times p )$ -matrix $A$ , $\| u \| _ { A } = \sqrt { \langle u , A u \rangle }$ is a semi-norm (a norm if the matrix is strictly positive   
definite). For $r > 0$ , $\boldsymbol { B } ( \boldsymbol { r } )$ denotes the closed centred Euclidean ball in $\mathbb { R } ^ { p }$ with radius $r$ . Given a   
non-empty set $U \subseteq \mathbb { R } ^ { p }$ , we let $\Delta _ { U }$ denote the space of (Borel) probability measures on $\mathbb { R } ^ { p }$ whose   
support in $U$ . Finally, $( u _ { t } ) _ { t \geq t _ { 0 } }$ denotes a sequence indexed on the integers, with $t _ { 0 }$ its smallest index.

# 90 2 Preliminaries on linear bandits

We consider a version of the classic problem of regret minimisation in stochastic linear bandits, where   
an agent needs to make a sequence of decisions (or pick an arm) from a given contextual decision set   
that may change over the sequence of rounds. We assume that the environment is oblivious to the   
actions of the agent, in the sense that the decision sets are determined in advance, and do not depend   
neither on the realisations of the noise nor on the agent’s arm-selection strategy.   
Concretely, we define the problem as follows. Let $\theta ^ { \star } \in \mathbb { R } ^ { p }$ be a parameter vector that is unknown   
to the learning agent. We assume as known an upper bound $B > 0$ on its euclidean norm (namely,   
$\theta ^ { \star } \in B ( B ) )$ . Fix a sequence of decision sets $( \bar { \mathcal X } _ { t } ) _ { t \ge 1 }$ in $\mathbb { R } ^ { p }$ . We assume that for all $t$ we have   
$\mathcal { X } _ { t } \subseteq B ( 1 )$ . At each round $t$ , the agent is required to pick an arm $X _ { t } \in \mathcal { X } _ { t }$ , and receives the reward   
$Y _ { t } = \langle \theta ^ { \star } , X _ { t } \rangle + \varepsilon _ { t }$ . The sequence $( \varepsilon _ { t } ) _ { t \geq 1 }$ represents the random feedback noise. The noise across   
different rounds is typically assumed to be conditionally centred and to have well behaved tails.   
For instance, a common assumption is to ask that $\mathbb { E } [ \varepsilon _ { t } | \mathcal { F } _ { t - 1 } ]$ is centred and sub-Gaussian, where   
$\mathcal { F } _ { t } = \sigma ( \varepsilon _ { 1 } , \ldots , \varepsilon _ { t } )$ is the $\sigma$ -field generated by the noise.1 This is the assumption this work relaxes.   
The agent aims to find a good strategy to pick arms $X _ { t }$ that lead to a high expected $T$ -round reward   
$\textstyle \sum _ { t = 1 } ^ { T } \langle X _ { t } , \theta ^ { \star } \rangle$ . To compare their performance to that of an agent playing each round the best available   
arm (in expectation), we define the regret after $T$ rounds as

$$
\mathrm { R e g } ( T ) = \sum _ { t = 1 } ^ { t } \operatorname* { s u p } _ { x \in \mathcal { X } _ { t } } \left( \langle x , \theta ^ { \star } \rangle - \langle X _ { t } , \theta ^ { \star } \rangle \right) .
$$

A common approach to tackle the linear bandit problem is to follow an upper confidence bound (UCB) strategy. This involves the following protocol. At each round $t$ , we first derive a confidence set $\mathcal { C } _ { t - 1 }$ , based on the arm-reward pairs $( X _ { s } , Y _ { s } ) _ { s \leq t - 1 }$ . This is a random set (as it depends on the past noise realisations), which must be constructed ensuring that $\theta ^ { \star } \in \mathcal { C } _ { t - 1 }$ with high probability. More precisely, the regret can be effectively controlled if one can ensure that $\theta ^ { \star }$ uniformly belongs to every set $( \mathcal { C } _ { t } ) _ { t \geq 1 } ^ { }$ , with high probability (a property often referred to as anytime validity). Then, for every available arm $x$ , we let

$$
\operatorname { U C B } _ { \mathcal { C } _ { t - 1 } } ( x ) = \operatorname* { m a x } _ { \theta \in \mathcal { C } _ { t - 1 } } \left. x , \theta \right. .
$$

By definition, this is a high-probability upper bound on 107 $\langle x , \theta ^ { \star } \rangle$ , which justifies the name “upper confidence bound”. The idea is then to optimistically pick as 108 $X _ { t } \in \mathcal { X } _ { t }$ the arm maximising $\mathrm { U C B } _ { { c } _ { t - 1 } }$ .

A key technical challenge in designing a UCB algorithm is to construct the anytime valid confidence sequence $( \mathcal { C } _ { t } ) _ { t \geq 1 }$ . Typically, under sub-Gaussian assumptions on the noise, these sets take the form of an ellipsoid, centred on a (regularised) maximum likelihood estimator. Explicitly, we often have

$$
\mathcal { C } _ { t } = \left\{ \theta \in \Theta : \| \theta - \widehat { \theta } _ { t } \| _ { V _ { t } } ^ { 2 } \leq \beta _ { t } ^ { 2 } \right\} ,
$$

where $\widehat { \theta } _ { t }$ is the least-squares estimator of $\theta ^ { \star }$ , $V _ { t }$ is the feature-covariance matrix and $\beta _ { t }$ is a radius   
carefully chosen so that the high-probability coverage requirement is satisfied. In this work, to   
construct the confidence sets we will leverage an online-to-confidence-set-conversion approach, a   
method that reduces the problem of proving statistical concentration bounds to proving existence of   
well-performing algorithms for an associated game of sequential probability assignment. We refer to   
Section 4 for more details on our technique to construct the confidence sequence.

# 115 3 Linear bandits with non-i.i.d. observation noise

We study a variant of the standard linear stochastic bandit problem where the observation-noise   
variables feature dependencies across different rounds. We focus on the case of weakly stationary   
noise, meaning we assume all the $\varepsilon _ { t }$ to have the same marginal distribution. However, the core   
assumption we make is what we call mixing sub-Gaussianity. This provides a way to control how   
dependencies decay as the time between two observations increases. It is defined in terms of a   
121 sequence of mixing coefficients $\phi _ { d }$ , which quantify this decay.   
22 Assumption 1 (Mixing sub-Gaussianity). Fix $\sigma > 0$ and let $\phi = ( \phi _ { d } ) _ { d \ge 0 }$ be a non-negative and   
non-increasing sequence. We say that the random sequence $( \epsilon _ { t } ) _ { t \geq 1 }$ is $( \sigma , \bar { \phi } )$ -mixing sub-Gaussian $i f$

$$
\left| \mathbb { E } \left[ \epsilon _ { t } \left| \mathcal { F } _ { t - d } \right. \right] \right| \leq \phi _ { d }
$$

and

$$
\mathbb { E } [ \exp \lambda ( \epsilon _ { t } - \mathbb { E } [ \epsilon _ { t } | \mathcal { F } _ { t - d } ] ) | \mathcal { F } _ { t - d }  ] \leq e ^ { \frac { \lambda ^ { 2 } \sigma ^ { 2 } } { 2 } } , \qquad \forall \lambda > 0 .
$$

Clearly, the above assumption generalises the standard conditionally sub-Gaussian assumption (that   
can be recovered by setting $\phi _ { d } = 0$ for all $t$ ), sometimes considered in the bandit literature. Although   
this might look like an unusual mixing assumption, it is very natural for our problem at hand, and   
can be weaker than standard mixing hypotheses. For instance, if the noise sequence is $\varphi$ -mixing   
(see Bradley, 2005) and each $\varepsilon _ { t }$ is centred and bounded in $[ - a , b ]$ , it is straightforward to check that   
$| \mathbb { E } [ \varepsilon _ { t } | \mathcal { F } _ { t - d } ] | \leq ( a + b ) \phi _ { d }$ , and so Assumption 1 is satisfied since the boundedness automatically   
implies sub-Gaussianity. In the rest of the paper we assume $\sigma = 1$ for simplicity.   
Under Assumption 1, we can build the confidence sequence needed for our UCB algorithm. We state   
this result below, but defer the explicit derivation to Section 4 (see Corollary 1 there).

Proposition 1. For some given $\phi$ , let the noise satisfy Assumption $^ { l }$ with $\sigma = 1$ . Fi ${ \mathfrak { r } } \delta \in ( 0 , 1 )$ $\lambda > 0$ , and $d \geq 1 ,$ . For $t \geq 1$ let

$$
\begin{array} { r } { \ \cdot _ { t } = \left\{ \theta \in \mathcal { B } ( B ) : \frac { 1 } { 2 } \| \theta - \widehat { \theta } _ { t } \| _ { V _ { t } } ^ { 2 } \leq \frac { d p } { 2 } \log \frac { ( B + 1 ) ^ { 2 } e \operatorname* { m a x } ( d p , t + d ) } { d p } + 2 \lambda B ^ { 2 } + t \phi _ { d } ( B + 1 ) + d \log \frac { d } { \delta } \right\} , } \end{array}
$$

where $\begin{array} { r } { V _ { t } = \sum _ { s = 1 } ^ { t } X _ { t } { X _ { t } ^ { \top } } + \lambda \mathrm { I d } , } \end{array}$ , and $\begin{array} { r } { \widehat { \theta } _ { t } = a r g m i n _ { \theta \in \mathcal { B } ( B ) } \sum _ { s = 1 } ^ { n } ( \langle \theta , X _ { t } \rangle - Y _ { t } ) ^ { 2 } } \end{array}$ . Then, $( \mathcal { C } _ { t } ) _ { t \geq 1 }$ is an anytime valid confidence sequence, in the sense that

$$
\begin{array} { r } { \mathbb { P } \big ( \theta ^ { \star } \in \mathcal { C } _ { t } , \forall t \ge 1 \big ) \ge 1 - \delta . } \end{array}
$$

Leveraging the confidence sequence above, we can define a UCB approach for our problem (Algo  
rithm 1). At a high level, the algorithm operates by taking the confidence sets defined in Proposition   
1, and selecting the arm optimistically, as in the standard UCB. A key point is that a delay $d$ is   
introduced, which at round $t$ restricts the agent to use only the information available from the first   
$t - d$ rounds. Although the actual technical reason behind this restriction will become fully clear only   
with the analysis of the coming sections, one can intuitively think of it as a way to prevent overfitting   
to recent noise, which might be highly correlated. If $d$ is sufficiently large, the noise observed in   
each round $t$ will be sufficiently decorrelated from the previous observations, which allows accurate   
estimation and uncertainty quantification of the true parameter $\theta ^ { \star }$ and the associated rewards.

# Algorithm 1 Mixing-LinUCB

set $d > 0$   
for $i \in \{ 1 , 2 , \ldots d \}$ do play an arbitrary $X _ { i }$ and observe $Y _ { i }$   
end for   
for $t \in \{ d + 1 , \ldots \}$ do $X _ { t } = \arg \operatorname* { m a x } _ { x \in \mathcal { X } _ { t } } \mathrm { U C B } _ { \mathcal { C } _ { t - d } } ( x )$ , where $\mathcal { C } _ { t - d }$ is as in Proposition 1 play $X _ { t }$ and observe reward $Y _ { t }$   
end for

In Section 5 we provide a detailed analysis of the regret of the algorithm that we proposed. For   
instance, assuming that the mixing coefficients decay exponentially as $\phi _ { d } = C e ^ { - d / \tau }$ (geometric   
mixing), we show that the regret can be upper bounded in high probability as

$$
\begin{array} { r } { \mathrm { R e g } ( T ) \leq \mathcal { O } \left( \tau p \sqrt { T } \log ( T ) ^ { 2 } + \tau \log T \sqrt { p T \log T } \right) . } \end{array}
$$

We refer to Theorem 2 and Corollary 2 in Section 5 for more details.

# 148 4 Constructing the confidence sequence

In this section we derive a confidence sequence for linear models with non-i.i.d. noise. First, we   
briefly describe the online-to-confidence-set conversion scheme from Clerico et al. (2025), which   
serves as our starting point. We then extend this technique to handle mixing noise.   
Before proceeding for the analysis of mixing sub-Gaussian noise, which is the focus of this work,   
we start by describing how to derive a confidence sequence when the noise is independent (or   
conditionally) centred and sub-Gaussian across different rounds, as in Clerico et al. (2025). The   
online-to-confidence sets framework that we consider instantiates an abstract game played between   
an online learner and an environment. We define the squared loss $\ell _ { s } ( \theta ) = \bar { \Psi } ( \langle \theta , \dot { X } _ { s } \rangle ^ { - } Y _ { s } ) ^ { 2 }$ . For   
158 each round $s = 1 , \ldots , t$ , the following steps are repeated:   
1. the environment reveals $X _ { s }$ to the learner;   
. the learner plays a distribution $Q _ { s } \in \Delta _ { \mathbb { R } ^ { p } }$ ;   
. the environment reveals $Y _ { s }$ to the learner;   
. the learner suffers the log loss $\begin{array} { r } { \begin{array} { r } { \mathcal { L } _ { s } ( Q _ { s } ) = - \log \int _ { \mathbb { R } ^ { p } } \exp ( - \ell _ { s } ( \theta ) ) \mathrm { d } Q _ { s } ( \theta ) . } \end{array} } \end{array}$   
This game is a special case of a well-studied problem called sequential probability assignment   
(Cesa-Bianchi and Lugosi, 2006). The learner can use any strategy to choose $Q _ { 1 } , \ldots , Q _ { t }$ , as long as   
each $Q _ { s }$ depends only on $X _ { 1 } , Y _ { 1 } , \dots , X _ { s - 1 } , Y _ { s - 1 } , X _ { s }$ . We define the regret of the learner against a   
(possibly data-dependent) comparator $\bar { \theta } \in \mathbb { R } ^ { p }$ as

$$
\mathrm { R e g r e t } _ { t } ( \bar { \theta } ) = \sum _ { s = 1 } ^ { t } \mathcal { L } _ { s } ( Q _ { s } ) - \sum _ { s = 1 } ^ { t } \ell _ { s } ( \bar { \theta } ) .
$$

Clerico et al. (2025) provide a regret bound upper bound (Proposition 3.1 there) for when the learner’s   
strategy is from an exponential weighted average (EWA) forecaster with a centred Gaussian prior   
$Q _ { 1 }$ . However, to account for the presence of dependencies in our analysis, we will need the prior’s   
support to be bounded. We hence state here a regret bound (whose proof is deferred to Appendix   
A.2) for the regret of an EWA forecaster with a uniform prior.   
Proposition 2. Fix $B > 0$ and consider the EWA forecaster with as prior the uniform distribution on   
$B ( \bar { B } + 1 )$ . Then, for all ${ \bar { \theta } } \in B ( B )$ and any $t \geq 1$ ,

$$
\mathrm { R e g r e t } _ { t } ( \bar { \theta } ) \leq \frac { p } { 2 } \log \frac { ( B + 1 ) ^ { 2 } e \operatorname * { m a x } ( p , t ) } { p } .
$$



We remark that, by adding and subtracting the total log loss of the learner, the excess loss of $\theta ^ { \star }$ (relative to 176 $\bar { \theta }$ ) can be rewritten as

$$
\sum _ { s = 1 } ^ { t } \ell _ { s } ( \theta ^ { \star } ) - \sum _ { s = 1 } ^ { t } \ell _ { s } ( \bar { \theta } ) = \mathrm { R e g r e t } _ { t } ( \bar { \theta } ) + \sum _ { s = 1 } ^ { t } \ell _ { s } ( \theta ^ { \star } ) - \sum _ { s = 1 } ^ { t } \mathcal { L } _ { s } ( Q _ { s } ) .
$$

This simple decomposition is the key idea in the online-to-confidence sets scheme.

Since the noise is conditionally sub-Gaussian and the distributions played by the online learner are predictable (super-m $Q _ { s }$ cannot depend onngale (cf. the no-h $Y _ { s }$ ), er $\begin{array} { r } { \sum _ { s = 1 } ^ { t } \ell _ { s } ( \theta ^ { \star } ) - \sum _ { s = 1 } ^ { t } \mathcal { L } _ { s } ( Q _ { s } ) } \end{array}$ is the logarithm of a non-negativenwald, 2007 or Proposition 2.1 in Clerico et al., 2025) with respect to the noise filtration $( \ddot { \mathcal F _ { t } } ) _ { t \geq 1 }$ .2 Henceforth, from Ville’s inequality (a classical anytime valid Markov-like inequality that holds for non-negative super-martingales) one can easily derive that $\theta ^ { \star } \in { \mathcal { C } } _ { t }$ (uniformly for all $t$ ) with probability at least $1 - \delta$ , where

$$
{ \mathcal { C } } _ { t } = \left\{ \theta \in \mathbb { R } ^ { p } : \sum _ { s = 1 } ^ { t } \ell _ { s } ( \theta ) - \sum _ { s = 1 } ^ { t } \ell _ { s } ( { \bar { \theta } } ) \leq \mathrm { R e g r e t } _ { t } ( { \bar { \theta } } ) + \log { \frac { 1 } { \delta } } \right\} .
$$

This result can be relaxed by replacing $\mathrm { R e g r e t } _ { t } ( \bar { \theta } )$ by any known regret upper bound for the online   
algorithm used in the abstract game (e.g., the bound of Proposition 2 for the EWA forecaster).   
The standard online-to-confidence sets scheme relies on the fact that $\begin{array} { r } { \sum _ { s = 1 } ^ { t } \ell _ { s } ( \theta ^ { \star } ) - \sum _ { s = 1 } ^ { t } \mathcal { L } _ { s } ( Q _ { s } ) } \end{array}$ is   
the logarithm of a non-negative super-martingale, whose fluctuations can be controlled uniformly in   
time thanks to Ville’s inequality. However, this property hinges on the fact that the noise is assumed   
to be conditionally centred and sub-Gaussian, which now is not anymore the case. Yet, thanks to   
our mixing assumption, if we restrict our focus on rounds that are sufficiently far apart, the mutual   
dependencies get weaker, and the exponential of the sum behaves almost like a martingale. This   
insight suggests to partition the rounds into blocks, whose elements are mutually far apart, then apply   
concentration results to each block, and finally use a union bound to recover the desired confidence   
sequence spanning all rounds. We remark that this is a classical approach to derive concentration   
results for mixing processes, often referred to as the blocking technique (Yu, 1994).   
In order for the online-to-confidence sets scheme to leverage the blocking strategy outlined above,   
the abstract online game used for the analysis must be designed in a way that is compatible with   
the block structure. To address this point, we adopt an approach inspired by Abélès et al. (2025),   
who introduced delays in the feedbacks received by the online learner in order to address a similar   
challenge. More precisely, we will now consider the following delayed-feedback version of the online   
196 game. Fix a delay $d > 0$ . For each round $s = 1 , \ldots , t$ , the following steps are repeated:   
1. the environment reveals to the learner $X _ { s }$ , which is assumed to be $\mathcal { F } _ { s - d }$ -measurable;   
. the learner plays a distribution $Q _ { s } \in \Delta _ { \mathbb { R } ^ { p } }$ ;   
. if $s > d$ , the environment reveals $Y _ { s - d + 1 }$ to the learner;   
4. the learner suffers the log loss $\begin{array} { r } { \begin{array} { r } { \mathcal { L } _ { s } ( Q _ { s } ) = - \log \int _ { \mathbb { R } ^ { p } } \exp ( - \ell _ { s } ( \theta ) ) \mathrm { d } Q _ { s } ( \theta ) . } \end{array} } \end{array}$

Note that the delay $d$ only applies for the rewards, while $Q _ { s }$ can still depend on $X _ { s }$ . Indeed, the choice of $X _ { s }$ in our mixing UCB algorithm is already “delayed”, as it depends on $\mathcal { C } _ { t - d }$ (see Algorithm 1).

03 Of course, in this setting the decomposition of (3) is still valid. We now want to deal with the 4 concentration of $\begin{array} { r } { \sum _ { s = 1 } ^ { t } \bar { \ell } _ { s } ( \theta ^ { \star } ) - \sum _ { s = 1 } ^ { t ^ { \star } } \mathcal { L } _ { s } ( Q _ { s } ) } \end{array}$ via the blocking technique. For convenience, let 05 $\begin{array} { r } { S _ { k } ^ { ( i ) } = \sum _ { j = 1 } ^ { k } D _ { i + ( j - 1 ) d } } \end{array}$ $D _ { t } = { \ell _ { t } ( \theta ^ { \star } ) } - \mathcal { L } _ { t } ( Q _ { t } )$ . We denote as ey idea is now t $S ^ { ( i ) } = ( S _ { k } ^ { ( i ) } ) _ { k \geq 1 }$ $S ^ { ( i ) }$ subsequence defined asbehaves as the log of a martingale, up to a cumulative remainder that accounts for the conditional mean shift in the mixing sub-Gaussianity assumption. In particular, Ville’s inequality and a union bound yield the following.

Lemma 1. Fix a delay $d > 0$ and $\delta \in ( 0 , 1 )$ . We have that

$$
\mathbb { P } \left( \sum _ { s = 1 } ^ { t } \big ( \ell _ { s } ( \theta ^ { \star } ) - \mathcal { L } _ { s } ( Q _ { s } ) \big ) \leq t \phi _ { d } B + d \log \frac { d } { \delta } , \forall t \geq 1 \right) \geq 1 - \delta .
$$

Now that we have a concentration result to control $S _ { t }$ , we only need to be able to upper bound the   
regret of an algorithm for the “delayed” online game that we are considering. To this purpose, we   
propose the following approach. We run $d$ independent EWA forecaster (with uniform prior), each   
one only making prediction and receiving the feedback once every $d$ rounds. More explicitly, the first   
forecaster acts at rounds 1, $d + 1$ , $2 d + 1 . . .$ , the second at round 2, $d + 2$ , $2 d + 2 . . .$ , and so on. As a   
direct consequence of Proposition 2, by summing the individual regret upper bounds we get a regret   
bound for the joint forecaster, which at each round returns the distribution predicted by the currently   
active forecaster. This technique of partitioning rounds into blocks for the regret analysis of online   
learning is common in the literature (e.g., see Weinberger and Ordentlich, 2002).

Lemma 2. Fix $B > 0$ , $d > 0$ , and consider a strategy with $d$ independent EWA forecasters outlined above, all initialised with the uniform distribution on $B ( B + 1 )$ as prior. For all ${ \bar { \theta } } \in B ( B )$ and $t \geq 1$

$$
\mathrm { R e g r e t } _ { t } ( \bar { \theta } ) \leq \frac { d p } { 2 } \log \frac { ( B + 1 ) ^ { 2 } e \operatorname * { m a x } ( d p , t + d ) } { d p } .
$$

Putting together what we have, we get a confidence sequence suitable for our mixing UCB algorithm. Theorem 1. Consider the setting introduced above. Fix $\delta \in ( 0 , 1 )$ and a delay $d > 0$ . Assume as known that $\theta ^ { \star } \in B ( B )$ . Let $\widehat { \theta } _ { t } = a r g m i n _ { \theta \in { \mathcal { B } } ( B ) } \{ \sum _ { s = 1 } ^ { t } \ell _ { s } ( \theta ) \}$ and $\begin{array} { r } { \Lambda _ { t } = \sum _ { s = 1 } ^ { t } X _ { s } X _ { s } ^ { \top } } \end{array}$ . Define

$$
\begin{array} { r } { { \mathcal C } _ { t } = \left\{ \theta \in { \mathcal B } ( B ) : \frac { 1 } { 2 } \| \theta - \widehat \theta _ { t } \| _ { \Lambda _ { t } } ^ { 2 } \leq \frac { d p } { 2 } \log \frac { ( B + 1 ) ^ { 2 } e \operatorname* { m a x } ( d p , t + d ) } { d p } + t \phi _ { d } ( B + 1 ) + d \log \frac { d } { \delta } \right\} . } \end{array}
$$

Then, $( \mathcal { C } _ { t } ) _ { t \geq 1 }$ is an anytime valid confidence sequence for $\theta ^ { \star }$ , namely

$$
\begin{array} { r } { \mathbb { P } \big ( \theta ^ { \star } \in \mathcal { C } _ { t } , \forall t \ge 1 \big ) \le 1 - \delta . } \end{array}
$$

Proof. The optimality of $\widehat { \theta } _ { t }$ implies $\begin{array} { r } { \sum _ { s = 1 } ^ { t } \langle \theta - \widehat \theta _ { t } , \nabla \ell _ { s } ( \widehat \theta _ { t } ) \rangle \geq 0 } \end{array}$ , for all $\theta \in B ( B )$ . As $\textstyle \sum _ { s = 1 } ^ { t } \ell _ { s }$ is quadratic, it equals its second order Taylor expansion around $\widehat { \theta } _ { t }$ and its Hessian is everywhere $\Lambda _ { t }$ . So,

$$
\frac { 1 } { 2 } \| \theta - \widehat { \theta } _ { t } \| _ { \Lambda _ { t } } ^ { 2 } \leq \frac { 1 } { 2 } \| \theta - \widehat { \theta } _ { t } \| _ { \Lambda _ { t } } ^ { 2 } + \sum _ { s = 1 } ^ { t } \left. \theta - \widehat { \theta } _ { t } , \nabla \ell _ { s } ( \widehat { \theta } _ { t } ) \right. = \sum _ { s = 1 } ^ { t } \left( \ell _ { s } ( \theta ) - \ell _ { s } ( \widehat { \theta } _ { t } ) \right) ,
$$

for any $\theta \in B ( B )$ . This, together with (3), Lemma 1, and Lemma 2, yields the conclusion.

We remark that the confidence sets of Theorem 1 take the form of the intersection between the ball   
$B ( B )$ and the “ellipsoid” $\{ \theta : \| \theta - \widehat { \theta _ { t } } \| _ { \Lambda _ { t } } \leq \beta _ { t } \}$ , for a suitable radius $\beta _ { t }$ . In order to implement and   
analyse the bandit algorithm, it will be more convenient to work with a relaxation of these sets, a   
pure ellipsoid not intersected with $B ( B )$ . We make this explicit in the following corollary.

Corollary 1. Fix $\lambda > 0$ , $d > 0$ , and $\delta \in ( 0 , 1 )$ . For $t \geq 1$ , let $V _ { t } = \Lambda _ { t } + \lambda \mathrm { I d }$ . Assuming that $\theta ^ { \star } \in B ( \bar { B } )$ , the following compact ellipsoids define an anytime valid confidence sequence for $\theta ^ { \star }$ :

$$
\begin{array} { r } { \ \stackrel { , } { \prime } = \left\{ \theta \in \mathcal { B } ( B ) : \frac { 1 } { 2 } \| \theta - \widehat { \theta } _ { t } \| _ { V _ { t } } ^ { 2 } \leq \frac { d p } { 2 } \log \frac { ( B + 1 ) ^ { 2 } e \operatorname* { m a x } ( d p , t + d ) } { d p } + 2 \lambda B ^ { 2 } + t \phi _ { d } ( B + 1 ) + d \log \frac { d } { \delta } \right\} . } \end{array}
$$

Proof. Let 226 $\begin{array} { r } { \beta _ { t } ^ { 2 } = d p \log { \frac { ( B + 1 ) ^ { 2 } e \operatorname* { m a x } ( d p , t + d ) } { d p } } + 2 t \phi _ { d } ( B + 1 ) + 2 d \log { \frac { d } { \delta } } } \end{array}$ . From Theorem 1, with probability at least 227 $1 - \delta$ , uniformly for every $t$ , $\| \theta ^ { \star } - \widehat { \theta } _ { t } \| _ { \Lambda _ { t } } ^ { 2 } \leq \beta _ { t } ^ { 2 }$ . Adding to both sides of this inequality 228 $\frac { \lambda } { 2 } \| \theta ^ { \star } - \widehat { \theta } _ { t } \| _ { 2 } ^ { 2 }$ , and relaxing the RHS using that $\lVert \theta ^ { \star } - \widehat { \theta } _ { t } \rVert _ { 2 } ^ { 2 } \leq 4 B ^ { 2 }$ , we conclude. □

# 5 Regret bounds for Mixing-LinUCB

In this section, we establish worst-case and gap-dependent cumulative regret bounds for mixing UCB algorithm (Mixing Lin-UCB). However, to account for the fact that Mixing-LinUCB selects actions with delays, the standard elliptical potential arguments must be modified. Throughout this section, we let $\tilde { R _ { t } } = \langle \theta ^ { \star } , X _ { t } ^ { \star } - X _ { t } \rangle$ (where $X _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { x \in \mathcal { X } _ { t } } \langle \theta ^ { \star } , x \rangle )$ denote the regret in round $t$ , and (B+1)2e max(dp,t+d) + 4λB2 + 2tϕd(B + 1) + 2d log d denote the squared radius of the ellipsoid $\mathcal { C } _ { t }$ in Corollary 1.

# 5.1 Worst-case regret bounds

First, following the regret analysis in Abbasi-Yadkori et al. (2011) (see also Section 19.3 in Lattimore   
and Szepesvári, 2020), we upper bound the instantaneous regret. From our boundedness assumptions   
$( \theta ^ { \star } \in B ( B )$ and $\mathcal { X } _ { t } \subseteq B ( 1 ) ,$ ), we easily deduce that $R _ { t } \leq 2 B$ . Under the event that our confidence   
sequence contains $\theta ^ { \star }$ at every step $t$ , we have another bound on $R _ { t }$ . If we define $ { \widetilde { \theta } } _ { t - d } \in  { \mathcal { C } } _ { t - d }$ to be   
the point at which $\langle \widetilde { \theta } _ { t - d } , X _ { t } \rangle = \mathrm { U C B } _ { \mathcal { C } _ { t - d } } ( X _ { t } )$ , then from the definition of $X _ { t }$ we have

$$
\langle \theta ^ { \star } , X _ { t } ^ { \star } \rangle \leq \operatorname* { m a x } _ { x \in \mathcal { X } _ { t } } \operatorname* { m a x } _ { \theta \in \mathcal { C } _ { t - d } } \langle \theta , x \rangle = \operatorname* { m a x } _ { x \in \mathcal { X } _ { t } } \mathrm { U C B } _ { \mathcal { C } _ { t - d } } ( x ) = \mathrm { U C B } _ { \mathcal { C } _ { t - d } } ( X _ { t } ) = \langle \widetilde { \theta } _ { t - d } , X _ { t } \rangle .
$$

Recall that, for all $s$ , $V _ { s } = \Lambda _ { s } + \lambda \mathrm { I d }$ , which is invertible as $\lambda > 0$ . Thus, by Cauchy-Schwarz,

$$
\begin{array} { r } { { \cal R } _ { t } \le \langle \widetilde \theta _ { t - d } - \theta ^ { \star } , { \cal X } _ { t } \rangle \le \| \widetilde \theta _ { t - d } - \theta ^ { \star } \| _ { { \cal V } _ { t - d } } \| { \cal X } _ { t } \| _ { { \cal V } _ { t - d } ^ { - 1 } } \le 2 \beta _ { t - d } \| { \cal X } _ { t } \| _ { { \cal V } _ { t - d } ^ { - 1 } } . } \end{array}
$$

This means that the instantaneous regret satisfies the bound

$$
R _ { t } \leq 2 \operatorname* { m a x } ( B , \beta _ { t - d } ) \operatorname* { m i n } ( 1 , \| X _ { t } \| _ { V _ { t - d } ^ { - 1 } } ) .
$$

Next, we separate the regret suffered in the first $d$ rounds and the remaining $T - d$ rounds. We then   
use Cauchy-Schwarz once more, and the fact that $\beta _ { t }$ is increasing in $t$ , to obtain

$$
\begin{array} { r l } & { \mathrm { R e g } ( T ) \leq 2 d B + \sqrt { ( T - d ) \sum _ { t = d + 1 } ^ { T } { R _ { t } ^ { 2 } } } } \\ & { \qquad \leq 2 d B + \sqrt { 4 ( T - d ) \operatorname* { m a x } ( B ^ { 2 } , \beta _ { T - d } ^ { 2 } ) \sum _ { t = d + 1 } ^ { T } \operatorname* { m i n } ( 1 , \| X _ { t } \| _ { V _ { t - d } ^ { - 1 } } ^ { 2 } ) } . } \end{array}
$$

At this point, we must depart from the standard linear UCB analysis (Abbasi-Yadkori et al., 2011; Latti  
more and Szepesvári, 2020). We bound the sum of the elliptical potentials $\begin{array} { r } { \sum _ { t = d + 1 } ^ { T } \operatorname* { m i n } ( 1 , \| X _ { t } \| _ { V _ { t - d } ^ { - 1 } } ^ { 2 } ) } \end{array}$   
using the following variant of the well-known “elliptical potential lemma” (see Appendix), which   
accounts for the fact that the feature covariance matrix $V _ { t - d }$ is updated with a delay of $d$ steps.

Lemma 3. For all $T \geq 1$ ,

$$
\sum _ { \ell = d + 1 } ^ { T } \operatorname* { m i n } ( 1 , \| X _ { t } \| _ { V _ { t - d } ^ { - 1 } } ^ { 2 } ) \leq 2 d p \log ( 1 + \frac { T } { \lambda d p } ) .
$$



We can now state a worst-case regret upper bound for Mixing-LinUCB.

Theorem 2. Fix 53 $\lambda = 1 / B ^ { 2 }$ , $d > 0$ and $\delta \in ( 0 , 1 )$ . With probability at least $1 - \delta$ , for all $T > d$ , the 54 regret of Mixing-LinUCB satisfies

$$
\begin{array} { r } { \mathrm { R e g } ( T ) \le 2 d B + \sqrt { 8 d p T \mathrm { m a x } ( B ^ { 2 } , \beta _ { T } ^ { 2 } ) \log ( 1 + \frac { B ^ { 2 } T } { d p } ) } . } \end{array}
$$



From the definition of $\beta _ { T }$ , we see that this regret bound is of the order

$$
\begin{array} { r } { \mathrm { R e g } ( T ) = \mathcal { O } \left( d B + d p \sqrt { T } \log \frac { T B } { d p } + T \sqrt { B d p \phi _ { d } \log \frac { T B } { d p } } + d \sqrt { p T \log \frac { T B } { p \delta } } \right) . } \end{array}
$$

For any fixed (i.e., not depending on $T$ ) delay $d$ , this regret bound is linear in $T$ . To obtain meaningful   
regret bounds, it is therefore crucial to set $d$ as a function of $T$ and the rate at which the mixing   
coefficients decay to zero3. Under the assumption that the noise variables are either geometrically or   
algebraically mixing, we obtain the following worst-case regret bounds.

Corollary 2. Suppose that(geometric mixing), and set $\begin{array} { r } { d = \lceil \tau \log \frac { B C T } { p } \rceil } \end{array}$ s Assumption. Then, the re $^ { l }$ with et of $\phi _ { d } = C e ^ { - \frac { d } { \tau } }$ for some CB satisfie $C , \tau > 0$

$$
\begin{array} { r } { \mathrm { \lambda e g } ( T ) = \mathcal { O } \left( \tau p \sqrt { T } \left( \log \frac { T B \operatorname* { m a x } ( 1 , C ) } { p } \right) ^ { 2 } + p \sqrt { T \tau } \log \frac { T B \operatorname* { m a x } ( 1 , C ) } { p } + \tau \log \frac { B C T } { p } \sqrt { p T \log \frac { T B } { p \delta } } \right) . } \end{array}
$$

Corollary 3. Suppose that the noise satisfies Assumption $^ { l }$ with $\phi _ { d } = C d ^ { - r }$ for some $C > 0$ and   
$r > 0$ (algebraic mixing), and set $d = \lceil C \dot { T } ^ { 1 / ( 1 + r ) } \rceil$ . Then, the regret of Mixing-LinUCB satisfies

$$
\begin{array} { r } { \mathrm { R e g } ( T ) \leq \mathcal { O } \left( C B T ^ { 1 / ( 1 + r ) } + T ^ { \frac { 3 + r } { 2 ( 1 + r ) } } \left( C p \log \frac { T B } { d p } + C \sqrt { B p \log \frac { T ^ { r / ( 1 + r ) } B } { C p } } + \sqrt { p \log \frac { T B } { p \delta } } \right) \right) . } \end{array}
$$

Up to a factor of $\tau \log T$ , the bound for geometrically mixing noise matches the regret bound for linear UCB with i.i.d. noise. This bound is trivial for $r \leq 1$ , however for $r > 1$ we get sublinear regret, and in particular we recover standard rates up to logarithmic factors in the limit where $r  \infty$ .

# 5.2 Gap-dependent regret bounds

Under the assumption that, each round, the gap between the expected reward of the optimal arm and the expected reward of any other arm is at least $\Delta > 0$ , we get regret bounds with better dependence

on 272 $T$ . More precisely, define the minimum gap $\begin{array} { r } { \Delta = \operatorname* { m i n } _ { t \in [ T ] } \operatorname* { m i n } _ { x \in \mathcal { X } _ { t } : x \neq X _ { t } ^ { \star } } \langle X _ { t } ^ { \star } - x , \theta ^ { \star } \rangle } \end{array}$ , and 273 assume that $\Delta > 0$ . Since we either have $R _ { t } = 0$ or $R _ { t } \geq \Delta > \bar { 0 }$ , it follows that

$$
R _ { t } \leq R _ { t } ^ { 2 } / \Delta .
$$

In our worst-case analysis, we showed that

$$
\sum _ { t = d + 1 } ^ { T } R _ { t } ^ { 2 } \le 8 d p \operatorname* { m a x } ( B ^ { 2 } , \beta _ { T } ^ { 2 } ) \log ( 1 + { \frac { T } { \lambda d p } } ) .
$$

Combined with the previous inequality, we obtain the following gap-dependent regret bound.

Theorem 3. Fix $\lambda = 1 / B ^ { 2 }$ , $d > 0$ , and $\delta \in ( 0 , 1 )$ . With probability at least $1 - \delta$ , for all $T > d$ , the   
regret of Mixing-LinUCB satisfies

$$
\mathrm { R e g } ( T ) \leq 2 d B + \frac { 8 d p } { \Delta } \operatorname* { m a x } ( B ^ { 2 } , \beta _ { T } ^ { 2 } ) \log \left( 1 + \frac { B ^ { 2 } T } { d p } \right) .
$$



Similarly to the worst-case bound in Theorem 2, for any fixed $d > 0$ , this regret bound is linear in $T$ .   
By setting $d$ as a suitable function of $T$ , we obtain the following gap-dependent regret bounds under   
geometrically or algebraically mixing noise.   
Corollary 4. Suppose that the noise variables are geometrically mixing and set $\begin{array} { r } { d = \lceil \tau \log \frac { B C T } { p } \rceil } \end{array}$   
Then the regret of Mixing-LinUCB satisfies

$$
\mathrm { R e g } ( T ) = \mathcal { O } \left( \frac { 8 \tau p } { \Delta } \left( l o g \frac { B C T } { p } \right) ^ { 2 } \log \left( 1 + \frac { B ^ { 2 } T } { p \tau \log \frac { B C T } { p } } \right) \left( \frac { p } { 2 } \log \frac { T } { p \tau } + \log \frac { \tau \log \frac { B C T } { p } } { \delta } \right) \right) .
$$



Corollary 5. Suppose that the noise variables are algebraically mixing and set $d = \lceil C T ^ { 1 / ( 1 + r ) } \rceil$ .   
Then the regret of Mixing-LinUCB satisfies

$$
\mathrm { R e g } ( T ) = \mathcal { O } \left( \frac { 8 C p } { \Delta } T ^ { \frac { 2 } { 1 + r } } \log \left( 1 + \frac { B ^ { 2 } T } { p C T ^ { 1 / ( 1 + r ) } } \right) \left( \frac { p } { 2 } \log \frac { T } { p \tau } + \log \frac { C T ^ { 1 / ( 1 + r ) } } { \delta } \right) \right) .
$$



# 6 Conclusion

We leave several interesting questions open for future research. Some of these are listed below.

An important limitation of our algorithm is that it requires the knowledge of the mixing coefficients (or at least an upper-bound on them). It would be interesting to explore the possibility of relaxing this assumption and to design an algorithm which infers the mixing coefficients while minimizing the regret. We note that the problem of estimating mixing coefficients is already a hard problem on its own right, with tight sample-complexity results only available in special cases such as Markov chains (Hsu et al., 2019; Wolfer, 2020). We also note that in order to recover the standard rate for the regret bound, the delay $d$ introduced in our algorithm need to be chosen as a function of the horizon $T$ . We believe that this could be fixed at little conceptual expense by using time-varying delay in the analysis, but we did not attempt to work out the (potentially non-trivial) details here.

Another limitation is that our analysis assumed throughout that the adversary picking the decision sets   
$\mathcal { X } _ { t }$ is oblivious, which is typically not required in linear bandit problems. For us, this was necessary   
to avoid potential statistical dependence between decision sets and the nonstationary observations.   
We believe that this issue can be handled at least for some classes of adversaries. For instance, it   
is easy to see that our analysis would carry through under the assumption that the decision sets be   
selected based on delayed information only. We leave the investigation of this question under more   
realistic assumptions open for future work.

References   
Naoki Abe and Philip M. Long. Associative reinforcement learning using linear probabilistic concepts. In Proceedings of the Sixteenth International Conference on Machine Learning, 1999.   
Peter Auer. Using confidence bounds for exploitation-exploration trade-offs. J. Mach. Learn. Res., 3: 397–422, 2003.   
Naoki Abe, Alan W. Biermann, and Philip M. Long. Reinforcement learning with immediate rewards and linear hypotheses. Algorithmica, 37(4):263–293, 2003.   
Lihong Li, Wei Chu, John Langford, and Robert E Schapire. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web, pages 661–670, 2010.   
Melda Korkut and Andrew Li. Disposable linear bandits for online recommendations. Proceedings of the AAAI Conference on Artificial Intelligence, 35(5), 2021.   
Maxime C Cohen, Ilan Lobel, and Renato Paes Leme. Feature-based dynamic pricing. Management Science, 66(11):4921–4943, 2020.   
T.L. Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. Advances in Applied Mathematics, 6(1):4–22, 1985.   
Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems, 24, 2011.   
Tor Lattimore and Csaba Szepesvári. Bandit algorithms. Cambridge University Press, 2020.   
Hamish Flynn, David Reeb, Melih Kandemir, and Jan R Peters. Improved algorithms for stochastic linear bandits using tail bounds for martingale mixtures. Advances in Neural Information Processing Systems, 36:45102–45136, 2023. Richard C. Bradley. Basic properties of strong mixing conditions: A survey and some open questions. Probability Surveys, 2:107–144, 2005.   
M. Mohri and A. Rostamizadeh. Rademacher complexity bounds for non-i.i.d. processes. NeurIPS, 2008. Baptiste Abélès, Eugenio Clerico, and Gergely Neu. Generalization bounds for mixing processes via delayed online-to-PAC conversions. In Proceedings of The 36th International Conference on Algorithmic Learning Theory, 2025.   
Kwang-Sung Jun, Aniruddha Bhargava, Robert Nowak, and Rebecca Willett. Scalable generalized linear bandits: Online computation and hashing. In Advances in Neural Information Processing Systems, volume 30, 2017.   
Junghyun Lee, Se-Young Yun, and Kwang-Sung Jun. Improved regret bounds of (multinomial) logistic bandits via regret-to-confidence-set conversion. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, pages 4474–4482, 2024. Eugenio Clerico, Hamish Flynn, Wojciech Kotłowski, and Gergely Neu. Confidence sequences for generalized linear models via regret analysis, 2025. URL https://arxiv.org/abs/2504. 16555. Claire Vernade, Alexandra Carpentier, Tor Lattimore, Giovanni Zappella, Beyza Ermis, and Michael Brueckner. Linear bandits with stochastic delayed feedback. In International Conference on Machine Learning, pages 9712–9721. PMLR, 2020a.   
Benjamin Howson, Ciara Pike-Burke, and Sarah Filippi. Delayed feedback in generalised linear bandits revisited. In International Conference on Artificial Intelligence and Statistics, pages 6095–6119. PMLR, 2023.   
350 Aurélien Garivier and Eric Moulines. On upper-confidence bound policies for non-stationary bandit problems. arXiv preprint arXiv:0805.3415, 2008.   
Yoan Russac, Claire Vernade, and Olivier Cappé. Weighted linear bandits for non-stationary environments. Advances in Neural Information Processing Systems, 32, 2019.   
Claire Vernade, Andras Gyorgy, and Timothy Mann. Non-stationary delayed bandits with intermediate observations. In International Conference on Machine Learning, pages 9722–9732. PMLR, 2020b.   
Nicolò Cesa-Bianchi and Gabor Lugosi. Prediction, Learning, and Games. Cambridge University Press, USA, 2006.   
Peter D. Grünwald. The Minimum Description Length Principle (Adaptive Computation and Machine Learning). The MIT Press, 2007.   
Bin Yu. Rates of convergence for empirical processes of stationary mixing sequences. The Annals of Probability, 22(1):94–116, 1994.   
M.J. Weinberger and E. Ordentlich. On delayed prediction of individual sequences. IEEE Transactions on Information Theory, 48(7), 2002.   
Daniel Hsu, Aryeh Kontorovich, David A Levin, Yuval Peres, Csaba Szepesvári, and Geoffrey Wolfer. Mixing time estimation in reversible markov chains from a single sample path. The Annals of Applied Probability, 29(4):2439–2480, 2019.   
Geoffrey Wolfer. Mixing time estimation in ergodic markov chains from a single trajectory with contraction methods. In Algorithmic Learning Theory, pages 890–905, 2020.

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

75 Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

• You should answer [Yes] , [No] , or [NA] .   
• [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.   
• Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

• Delete this instruction block, but keep the section heading “NeurIPS Paper Checklist", • Keep the checklist subsection headings, questions/answers and guidelines below. • Do not modify the questions and only use the provided macros for your answers.

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: See sections 3, 4,5.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the paper.   
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.   
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See Conclusion.

# Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.   
• The authors are encouraged to create a separate "Limitations" section in their paper.   
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.   
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.   
The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.   
While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Most of the common assumptions concerning linear bandits are presented in Section 2. The main novel assumption is introduced in section 3. All the proofs that are not addressed in the paper are gathered in the Appendix.

Guidelines:

• The answer NA means that the paper does not include theoretical results.   
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.   
• All assumptions should be clearly stated or referenced in the statement of any theorems.   
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.   
Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.   
• Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: Not Applicable.

Guidelines:

• The answer NA means that the paper does not include experiments.

• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.   
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.   
Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.   
While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: Not Applicable.

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

Answer: [NA]

Justification: Not Applicable.

Guidelines:

• The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Not Applicable.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).   
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)   
• The assumptions made should be given (e.g., Normally distributed errors).   
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.   
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a $96 \%$ CI, if the hypothesis of Normality of errors is not verified.   
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).   
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: Not Applicable.

Guidelines:

• The answer NA means that the paper does not include experiments. • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

# 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification:

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.   
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.   
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification:

Guidelines:

• The answer NA means that there is no societal impact of the work performed.   
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.   
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.   
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.   
The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: This article is purely theoretical and addresses a mathematical problem which it attempts to solve.

Guidelines:

• The answer NA means that the paper poses no such risks.   
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification:

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

Answer: [NA]

Justification:

Guidelines:

• The answer NA means that the paper does not release new assets.   
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
• The paper should discuss whether and how consent was obtained from people whose asset is used.   
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification:

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification:

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.   
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

# 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification:

Guidelines:

• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
