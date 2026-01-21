# An Analysis of Causal Effect Estimation using Outcome Invariant Data Augmentation

# Uzair Akbar∗ Georgia Tech

Niki Kilbertus TU Munich Helmholtz AI

Hao Shen TU Munich Fortiss GmbH

Krikamol Muandet Rational Intelligence CISPA

Bo Dai   
Georgia Tech   
Google DeepMind

# Abstract

The technique of data augmentation (DA) is often used in machine learning for regularization purposes to better generalize under i.i.d. settings. In this work, we present a unifying framework with topics in causal inference to make a case for the use of DA beyond just the i.i.d. setting, but for generalization across interventions as well. Specifically, we argue that when the outcome generating mechanism is invariant to our choice of DA, then such augmentations can effectively be thought of as interventions on the treatment generating mechanism itself. This can potentially help to reduce bias in causal effect estimation arising from hidden confounders. In the presence of such unobserved confounding we typically make use of instrumental variables (IVs)—sources of treatment randomization that are conditionally independent of the outcome. However, IVs may not be as readily available as DA for many applications, which is the main motivation behind this work. By appropriately regularizing IV based estimators, we introduce the concept of IV-like (IVL) regression for mitigating confounding bias and improving predictive performance across interventions even when certain IV properties are relaxed. Finally, we cast parameterized DA as an IVL regression problem and show that when used in composition can simulate a worst-case application of such DA, further improving performance on causal estimation and generalization tasks beyond what simple DA may offer. This is shown both theoretically for the population case and via simulation experiments for the finite sample case using a simple linear example. We also present real data experiments to support our case.

# 1 Introduction

A classical problem in machine learning is that of regression—using i.i.d. samples from some fixed, unknown distribution $\mathbb { P } _ { X , Y }$ , we predict outcome $Y$ values for unlabelled treatment $X$ values. The use of regularization techniques is crucial for this task to achieve good generalization from training to test data [1]. Data augmentation $( D A )$ [2, 3] is one such method, where each sample is randomly perturbed multiple times to grow the dataset size. However, these regression models cannot generally be interpreted causally as the statistical relationship between $X$ and $Y$ may arise from shared common causes, known as confounders, rather than from $X$ influencing $Y$ . Removing such confounders requires independently assigning values of $X$ during data generation, known as an intervention [4, 5].

Unfortunately, we seldom have access to the data generation process to be able to intervene on variables. A common workaround is to use auxiliary variables to correct for unobserved confounders [6–8]. One such approach is that of instrumental variables $( I V s )$ that represent certain conditional independences in the system which can be used to identify the causal effect of $X$ on $Y$ [9–11]. Alas, IVs too are generally hard to find in may popular applications such as computer vision and natural language processing, motivating the need for more accessible ways to mitigate unobserved confounding.

Recent work therefore seeks to leverage more commonly available auxiliary variables to reduce confounding-induced bias even when the causal effect itself cannot be identified [12–15]. Collectively referred to as causal regularization, these methods aim to learn predictors that generalize out-ofdistribution $( O O D )$ by discouraging reliance on spurious (i.e., non-causal,) correlations. Since distribution shifts often correspond to interventions on parts of the data-generating process [16, 4], models that fail under such shifts typically do so because they exploit confounded relationships [17]. Tackling this root cause directly, causal regularization offers a principled approach for more robust prediction.

In the same vein, more ambitious works have also explored the use of common regularization techniques, such as $\ell _ { 1 }$ , $\ell _ { 2 }$ [18] and the min-norm interpolator [19], for the same purpose of causal regularization. This is in contrast to the canonical use of such regularizers for estimation variance reduction and i.i.d. prediction generalization [1]. Other popular regularization methods, however, remain understudied in a similar context of un-identifiable causal effect estimation, motivating our work.

Our contributions. To this end, we provide a first analysis of DA for estimating un-identifiable causal effects using only observational data for $( X , Y )$ . Our contributions, summarized in Tab. 1, include: (i) DA as a soft intervention (Sec. 4.1): We show that DA can synthesize treatment interventions when the outcome function is invariant to DA, lowering bias in causal effect estimates when the intervention acts along spurious features. (ii) Introducing IV-like regression (Sec. 3): Relaxing the properties of IVs, we introduce the concept of IV-like (IVL) variables. This generalization renders IV regression ineffective at identifying causal effects, but when regularized appropriately via our proposed IVL regression, may still reduce confounding bias and improve prediction generalization across treatment interventions. (iii) DA parameters as IVL (Sec. 4.2): By casting parameterized DA as IVL, we show that its composition $\mathrm { D A + I V L }$ with IVL regression further reduces confounding bias beyond just simple DA by essentially simulating a worst-case or adversarial application of the DA.

We validate our approach with theoretical results in a linear setting for the infinite-sample case, and simulation and real-data experiments in the finite-sample case.

# 2 Preliminaries

Consider treatment $X$ and outcome $Y$ taking values in ${ \mathcal { X } } \subseteq \mathbb { R } ^ { m }$ and $\mathcal { y } \subseteq \mathbb { R } ^ { l }$ respectively. Given the set of functions $\mathcal { H } : = \{ h : \mathcal { X } \xrightarrow { } \mathcal { Y } \}$ , the canonical setting described in the literature [4, 15, 20] deals with estimating the function $f \in \mathcal H$ in the structural equation model (SEM) $\mathfrak { M }$ of the following form1

$$
X = \tau ( Y , Z , C , N _ { X } ) , \qquad Y = f ( X ) + \epsilon ( C ) + N _ { Y } ,
$$

where $Z , C , N _ { X }$ , $N _ { Y }$ are exogenous (and therefore mutually independent) random variables and the residual $\xi : = Y - f ( X ) = \epsilon ( C ) + N _ { Y } $ is assumed to be zero mean, i.e. $\mathbb { E } ^ { \mathfrak { M } } [ \xi ] = 0$ . Since $\mathfrak { M }$ is potentially cyclic, a priori it may entail several or no distributions at all. However, here we make the assumption that for all $( \mathbf { x } _ { 0 } , \mathbf { y } _ { 0 } ) \in \mathcal { X } \times \mathcal { Y }$ the unique limits

$$
\mathbf { x } : = \operatorname* { l i m } _ { t \to \infty } \mathbf { x } _ { t } = \operatorname* { l i m } _ { t \to \infty } \tau ( \mathbf { y } _ { t - 1 } , \mathbf { z } , \mathbf { c } , \mathbf { n } _ { X } ) , \qquad \mathbf { y } : = \operatorname* { l i m } _ { t \to \infty } \mathbf { y } _ { t } = \operatorname* { l i m } _ { t \to \infty } f ( \mathbf { x } _ { t - 1 } ) + \boldsymbol { \epsilon } ( \mathbf { c } ) + \mathbf { n } _ { Y }
$$

exist for any $( \mathbf { z } , \mathbf { c } , \mathbf { n } _ { X } , \mathbf { n } _ { Y } ) \sim \mathbb { P } _ { Z , C , N _ { X } , N _ { Y } } ^ { \mathfrak { M } }$ PMZ,C,NX,NY , meaning that the unique distribution entailed by M is in this equilibrium state. Of course, if $\mathfrak { M }$ is acyclic, these limits always exist. Note that assuming the existence of such an equilibrium does not violate the classic independent causal mechanism (ICM) principle [4]; we defer interested readers to Appendix B for further details on cyclic SEMs and the ICM.

Given a proper convex loss $\ell : \mathbb { R } ^ { l } \times \mathbb { R } ^ { l }  \mathbb { R } _ { + }$ , empirical risk minimization (ERM) uses a dataset $\mathcal { D } : = \{ ( \bar { \mathbf { x } } _ { i } , \bar { \mathbf { y } } _ { i } ) \} _ { i = 0 } ^ { n }$ of $n$ samples from $\mathfrak { M }$ to minimize an empirical version of the statistical risk

$$
R _ { \mathrm { E R M } } ^ { \mathfrak { M } } ( h ) : = \mathbb { E } ^ { \mathfrak { M } } [ \ell ( Y , h ( X ) ) ] ,
$$

over $h \in \mathcal H$ . However, since the residual $\xi$ in Eq. (1) is generally correlated with $X$ , i.e., $\mathbb { E } ^ { \mathfrak { M } } [ \xi \mid X ] \neq$ 0, the ERM minimizer $\hat { h } _ { \mathrm { E R M } } ^ { \mathfrak { M } }$ typically yields a biased estimate of $f$ [5, 4]. This bias arises due to the exclusion of the (unobserved) common parent $C$ of $X$ and $Y$ , i.e. a confounder, in the ERM objective (hence fittingly called the omitted-variable bias [21]) and/or the model is cyclic (simultaneity bias [20, 22], or reverse causality [5] in the degenerate case). For simplicity we shall refer to either case by saying that $X$ and $Y$ are confounded and the resulting bias as the confounding bias [5].2

Table 1: A picture summary of our contributions. represents composition of operations or transformations, and $\Leftrightarrow$ represents equivalence.   

<table><tr><td></td><td>Type of Data Augmentation</td><td>Topics in Causal Inference</td></tr><tr><td>Teon grinrnrnuiroeees aareste seettteeetet</td><td>None; observational data √ Outcome (i ↑</td><td>Data generating structural model √ Treatment</td></tr><tr><td></td><td>invariantDA √ Worst-case or</td><td>(soft) intervention √ (iii)</td></tr><tr><td></td><td>adversarial DA</td><td>Regularized IV regression (ii)</td></tr></table>

![](images/2312e39a61951ba44f6f0ca4fedfe43a9a923c86f046173629d8c281eb46aa07.jpg)  
Figure 1: Graph of $\mathfrak { M }$ depicting an instrument $Z$ that satisfies treatment relevance, exclusion restriction, un-confoundedness and outcome relevance properties. An intervention on $X$ gives us the graph in $( b )$ . IV regression simulates such an intervention using only observational data.

# 2.1 Intervention for causal effect estimation

We can make $X$ and the residual $\xi$ uncorrelated via an intervention3 $\mathrm { { d o } } ( X : = X ^ { \prime } )$ , where we explicitly set $X$ to some independently sampled $X ^ { \prime }$ in Eq. (1) irrespective of its parents, resulting now in the new SEM induced by this $\mathfrak { M }$ ;o $\mathrm { d o } ( X \mathrel { \mathop : } = X ^ { \prime } )$ or  ca $\mathfrak { M }$ ;d $\operatorname { d o } ( X )$ as a shorthand for when erventional distribution $X ^ { \prime } \sim \mathbb { P } _ { X } ^ { \mathfrak { M } }$ . The dect to ibution) under $\mathfrak { M }$ which the ERM objective from Eq. (2) now defines the following causal risk $( C R )$ [12, 19, 24] as

$$
R _ { \mathrm { C R } } ^ { \mathfrak { M } } ( h ) : = R _ { \mathrm { E R M } } ^ { \mathfrak { M } ; \mathrm { d o } ( X ) } ( h ) = R _ { \mathrm { E R M } } ^ { \mathfrak { M } ; \mathrm { d o } \left( X : = X ^ { \prime } \right) } ( h ) , \qquad \mathrm { s . t . } \qquad X ^ { \prime } \sim \mathbb { P } _ { X } ^ { \mathfrak { M } } .
$$

Minimizing Eq. (3) is meaningful in two important cases where ERM fails: (i) Causal effect estimation: The minimizer $\hat { h } _ { \mathrm { C R } } ^ { \mathfrak { M } }$ of Eq. (3) gives us an unbiased estimate of the average treatment effect (ATE) [6] ${ \mathbb E } ^ { \mathfrak { M } ; \mathrm { d o } ( X : = \mathbf { x } ) } [ Y \mid X = \mathbf { x } ] = f ( \mathbf { x } )$ that measures the causal influence of $X$ on $Y$ . (ii) Robust prediction: ATE based prediction of $Y$ values for unlabelled $X$ values is robust in the sense that it can generalize across arbitrary OOD treatment interventions or shifts in the treatment distribution [25]. Consequently, the causal risk minimizer $\hat { h } _ { \mathrm { C R } } ^ { \mathfrak { M } }$ is also a robust predictor over the support of $\mathbb { P } _ { X } ^ { \mathfrak { M } }$ . Specifically, $\hat { h } _ { \mathrm { C R } } ^ { \mathfrak { M } }$ minimizes the worst-case ERM objective over the set $\mathcal { P }$ of all possible intervention distributions $\mathbb { P } _ { X ^ { \prime } }$ over the support of $\mathbb { P } _ { X } ^ { \mathfrak { M } }$ [25], i.e. for $\mathcal { P } : = \{ \mathbb { P } _ { X ^ { \prime } } \ \vert \ \operatorname { s u p p } ( \mathbb { P } _ { X ^ { \prime } } ) \subseteq \operatorname { s u p p } ( \mathbb { P } _ { X } ^ { \mathfrak { M } } ) \}$ ,

$$
\hat { h } _ { \mathrm { C R } } ^ { \mathfrak { M } } \in \operatorname * { a r g m i n } _ { h \in \mathcal { H } } \operatorname* { m a x } _ { \mathbb { P } _ { X ^ { \prime } } \in \mathcal { P } } R _ { \mathrm { E R M } } ^ { \mathfrak { M } ; \mathrm { d o } \left( X : = X ^ { \prime } \right) } ( h ) .
$$

To better isolate the estimation error due to confounding, we define the causal excess risk (CER) [19]

$$
\mathrm { C E R } _ { \mathfrak { M } } ( h ) : = R _ { \mathrm { C R } } ^ { \mathfrak { M } } ( h ) - R _ { \mathrm { C R } } ^ { \mathfrak { M } } ( f ) .
$$

This removes the irreducible noise from Eq. (3) (see Appendix A) and directly measures how far a hypothesis $h$ deviates from the true causal function $f$ under interventions, so that $\mathrm { C E R } { \mathfrak { M } } ( f ) = 0$ .

Since interventions are often inaccessible for computing the risk in Eq. (3), we usually rely on observational data/ distribution and additional variables to approximate them, as outlined in the next section.

# 2.2 Instrumental variable regression

One way to get an unbiased estimate of $f$ from the observational distribution of $\mathfrak { M }$ is to use socalled instrumental variables $Z$ with the properties [5, 4, 10, 9, 26] of: (i) Treatment Relevance: $Z \not \perp X$ . (ii) Exclusion Restriction: $Z$ enters $Y$ only through $X$ , i.e. $Z \perp \perp Y ^ { \mathfrak { M } ; \mathrm { d o } ( X : = \mathbf { x } ) }$ .4 (iii) Unconfoundedness: $Z \perp \perp \xi$ . (iv) Outcome Relevance: $Z$ carries information about $Y$ , i.e. $Y \not \perp Z$ .

Conditioning Eq. (1) on $Z$ and using $\mathbb { E } [ \xi \mid Z ] = \mathbb { E } [ \xi ] = 0$ from the unconfoundedness property gives

$$
\mathbb { E } ^ { \mathfrak { M } } [ Y \mid Z ] = \mathbb { E } ^ { \mathfrak { M } } [ f ( X ) \mid Z ] .
$$

IV regression therefore entails solving Eq. (4) for $f$ , which can be done by minimizing the risk [26]

$$
R _ { \mathrm { I V } } ^ { \mathfrak { M } } ( h ) : = \mathbb { E } ^ { \mathfrak { M } } \left[ \ell \left( Y , \mathbb { E } ^ { \mathfrak { M } } [ h ( X ) \mid Z ] \right) \right] .
$$

For linear $f ( \cdot ) : = \mathbf { f } ^ { \top } ( \cdot ) , h ( \cdot ) : = \mathbf { h } ^ { \top } ( \cdot )$ with f , $\mathbf { h } \in \mathbb { R } ^ { m }$ and squared loss $\ell ( \mathbf { y } , \mathbf { y } ^ { \prime } ) : = \left\| \mathbf { y } - \mathbf { y } ^ { \prime } \right\| ^ { 2 }$ , this gives the two-stage-least-squares (2SLS) [27] solution where the first stage regresses $X$ from $Z$ , and the second stage regresses $Y$ from predictions $\mathbb { E } [ X \mid Z ]$ of the first stage to get the estimate $\hat { h } _ { \mathrm { I V } } ^ { \mathfrak { M } }$ .

# 2.3 Data augmentation

In this work we restrict ourselves to data augmentation with respect to which $f$ is invariant [3, 28]. The action of a group $\mathcal { G }$ is a mapping $\delta : \mathcal { X } \times \mathcal { G }  \mathcal { X }$ which is compatible with the group operation. For convenience we shall write $\mathbf { g } \mathbf { x } : = \delta ( \mathbf { x } , \mathbf { g } )$ . We say that $f$ is invariant under $\mathcal { G }$ (or $\mathcal { G }$ -invariant) if

$$
f ( \mathbf { g x } ) = f ( \mathbf { x } ) , \qquad \forall \ ( \mathbf { g } , \mathbf { x } ) \in \mathcal { G } \times \mathcal { X } .
$$

Less formally, we say that the map gx, henceforth assumed to be continuous in $\mathbf { x }$ , is a valid outcomeinvariant DA transformation parameterized by the vector $\mathbf { g } \in \mathcal { G }$ . Let $\mathcal { G }$ have a (unique) normalized Haar measure and $\mathbb { P } _ { G }$ be the corresponding distribution defined over it. For some $G \sim \mathbb { P } _ { G }$ , the canonical application of DA seeks to minimize an empirical version of the following risk.

$$
R _ { \mathrm { D A } _ { G } + \mathrm { E R M } } ^ { \mathfrak { M } } ( h ) : = \mathbb { E } ^ { \mathfrak { M } } [ \ell ( Y , h ( G X ) ) ] .
$$

Note that it is sufficient to have some prior information about the symmetries of $f$ in order to be able to construct such a DA. For example, when classifying images of cats and dogs we already know that whatever the true labeling function may be, it would certainly be invariant to rotations on the images. $G$ would then represent the random rotation angle, whereas $G \mathbf { x }$ would be the rotated image $\mathbf { x }$ .

We wish to contrast the use of DA in this work with the canonical setting—to mitigate overfitting, DA is used to grow the sample size by generating multiple augmentations $\left( G \mathbf { x } , \mathbf { y } \right)$ for each data sample $( \mathbf { x } , \mathbf { y } ) \sim \mathbb { P } _ { X , Y } ^ { \mathfrak { M } }$ [3, 28, 29]. Such regularization, overfitting mitigation, estimation variance reduction, or i.i.d. prediction generalization is not the focus of this work and we intentionally provide Eq. (6) along with theoretical results that follow in the population case to emphasize that DA is not being used as a conventional regularizer. Instead, our goal is to improve causal effect estimation and robust prediction by re-purposing DA to mitigate hidden confounding bias in the data.

# 3 Faithfulness and Outcome Relevance in IVs

The distribution $\mathbb { P } _ { X , Y , Z , C } ^ { \mathfrak { M } }$ is faithful to the graph of $\mathfrak { M }$ if it only exhibits independences implied by the graph [4, 30].5 This standard assumption in IV settings renders outcome-relevance implicit and therefore rarely mentioned. In this section we discuss the case where only the first three IV properties are satisfied, i.e. outcome-relevance may not hold. Since such a $Z$ may not be a valid IV, therefore identifiability of ATE is not possible in general as the problem in Eq. (4) can now be misspecified, having multiple, potentially infinitely many solutions when $Y \perp \perp Z$ . Nevertheless, we shall refer to such a $Z$ as $I V .$ -like $( I V L )$ to emphasize that while $Z$ may not be an IV, it may still be ‘instrumental’ for reducing confounding bias when estimating the ATE compared to the standard ERM baseline.

ERM regularized IV regression. Despite problem miss-specification for a IVL $Z$ , the target function $f$ remains a minimizer for the IV risk in Eq. (5). Albeit, potentially not unique—for example, a linear $h$ with squared loss leads to an under-determined problem in Eq. (5). We therefore propose the following regularized version of the IV risk for such an IVL setting,

$$
R _ { \mathrm { I V L } _ { \alpha } } ^ { \mathfrak { M } } ( h ) : = R _ { \mathrm { I V } } ^ { \mathfrak { M } } ( h ) + \alpha R _ { \mathrm { E R M } } ^ { \mathfrak { M } } ( h ) ,
$$

where $\alpha > 0$ is the regularization parameter. The ERM risk as a penalty allows our estimations to have good predictive performance while the IV risk encourages solution search within the subspace where we know $f$ to be present. We refer to minimising the risk in Eq. (7) as $I V L$ regression.

Note that the motivation behind IVL regression is not the identifiability of $f$ , but rather potentially better estimations of $f$ with lower confounding bias. The next section provides a concrete example.

![](images/c961801fac8db92cb61dec132bdd7bf595620779b7691c3cce0f13df04ec98b4.jpg)  
Figure 2: The observational distribution of $( \bar { G } X , Y , G , C )$ and $( X , Y , G , C )$ for graphs $( a )$ and $( b )$ respectively are the same. The former applies DA on $X$ , whereas the later applies a (soft) intervention on $X$ . Furthermore, for the graph in $( b )$ , $G$ is IVL.

![](images/ba94242554fc60e93e7d681a9f091c0d1373655041e4a05716b04738a2f72115.jpg)  
Figure 3: The ground truth function f in Example 2. The DA applied here corresponds to randomly translating the data samples along their level-set by adding random noise sampled from the null-space of f .

Example 1 (a linear Gaussian IVL example). For scalar $\sigma > 0$ , non-zero matrices $\mathbf { r } , \mathbf { T } \in \mathbb { R } ^ { * \times m }$ and vectors $\boldsymbol { \tau } ^ { \top } , \mathbf { f } , \boldsymbol { \epsilon } \in \mathbb { R } ^ { m }$ such that $\mathbf { f } ^ { \intercal } \tau ^ { \intercal } \neq 1$ so that the following SEM $\mathfrak { M }$ is solvable in $( X , Y )$

$$
X = \pmb { \tau } ^ { \top } Y + \pmb { \Gamma } ^ { \top } Z + \pmb { \mathrm { T } } ^ { \top } C + \sigma N _ { X } ,
$$

$$
\begin{array} { r } { Y = \mathbf { f } ^ { \top } X + \epsilon ^ { \top } C + \sigma N _ { Y } , } \end{array}
$$

where $Z , C , N _ { X } , N _ { Y }$ are conformable, centered Gaussian random vectors and $Z$ is IVL w.r.t. $( X , Y )$

Now, the task is to improve our estimation of f compared to standard ERM. We evaluate an estimate $\hat { \mathbf { h } } ^ { \mathcal { D } }$ using the CER, which for a squared loss and covariance $\Sigma _ { X } ^ { \mathfrak { M } }$ in Example 1 simply comes out to be

$$
\mathrm { C E R } _ { \mathfrak { M } } \left( { \hat { \mathbf { h } } } ^ { D } \right) = \left\| { \hat { \mathbf { h } } } ^ { D } - \mathbf { f } \right\| _ { \Sigma _ { X } ^ { \mathfrak { M } } } ^ { 2 } .
$$

Prior works use this form to quantify the error in ATE estimation [19, 12] or measure some notion of strength of confounding [18, 31, 24]. Similarly, we use it to measure confounding bias of population estimates $\hat { \mathbf { h } } ^ { \mathfrak { M } }$ (Appendix A) and estimation error in finite sample experiments. The next results follow. Theorem 1 (robust prediction with IVL regression). For SEM M in Example $I$ , the following holds:

$$
\hat { \mathbf { h } } _ { N L _ { \alpha } } ^ { \mathfrak { M } } \in \operatorname { a r g m i n } _ { \mathbf { h } } \operatorname* { m a x } _ { \zeta \in \mathcal { P } _ { \alpha } } R _ { E R M } ^ { \mathfrak { M } ; \mathrm { d o } \left( \Gamma ^ { \top } ( \cdot ) = \zeta \right) } ( \mathbf { h } ) , \quad s . t . \quad \mathcal { P } _ { \alpha } : = \bigg \{ \zeta \bigg | \zeta \zeta ^ { \top } \prec \bigg ( \frac { 1 } { \alpha } + 1 \bigg ) \mathbf { r } ^ { \top } \Sigma _ { Z } ^ { \mathfrak { M } } \Gamma \bigg \} .
$$

Proof. See Appendix F.3 for the proof.

Theorem 2 (causal estimation with IVL regression). In SEM M of Example $I$ , for $\alpha < \infty$ , we have

$$
\begin{array} { r } { \mathrm { C E R } _ { \mathfrak { s p } } \Big ( \hat { \mathbf { h } } _ { I V L _ { \alpha } } ^ { \mathfrak { M } } \Big ) \leq \mathrm { C E R } _ { \mathfrak { s p } } \Big ( \hat { \mathbf { h } } _ { E R M } ^ { \mathfrak { M } } \Big ) , \qquad e q u a l i t y i f \qquad \mathbb { E } ^ { \mathfrak { M } } [ X \mid Z ] \ \bot _ { \mathrm { a . s . } } \mathbb { E } ^ { \mathfrak { M } } [ X \mid \xi ] . } \end{array}
$$

Proof. See Appendix F.4 for the proof.

Theorem 1 shows that IVL regression achieves optimal predictive performance across treatment interventions within the perturbation set $\mathcal { P } _ { \alpha }$ defined by $\alpha$ . Theorem 2 further states that this strictly reduces confounding bias in ATE estimates iff the perturbations align with spurious features of $X$ , as indicated by the equality condition (also necessary for identifiability in linear IV settings [32, 25]).

# 4 Causal Effect Estimation using Data Augmentation

We dedicate this section to the main topic and point of this work—discussing the potential of data augmentation for improving predictive performance across interventions and reducing confounding bias in ATE estimates. To that effect, for the rest of this work we shall consider the following SEM $\mathfrak { A }$

$$
X = \tau ( Y , C , N _ { X } ) , \qquad Y = f ( X ) + \epsilon ( C ) + N _ { Y } ,
$$

which is assumed to have a unique stationary distribution with exogenous $C , N _ { X } , N _ { Y }$ and the residual $\xi : = Y - f ( X )$ is zero-mean, i.e. $\mathbb { E } [ \xi ] = 0$ . We also have access to DA transformations $G X$ of $X$ parameterized by $G \sim \mathbb { P } _ { G } ^ { \mathfrak { A } }$ such as described in Sec. 2.3. Figure 2a shows the graph of $\mathfrak { A }$ post DA.

Given samples for only $( X , Y )$ and some valid DA parameterised by $G$ , the task is to improve predictive performance across interventions and reduce confounding bias in ATE estimates. We now make two observations in the following sections and state the respective results that follow thereof.

# 4.1 Data augmentation as a soft intervention

Consider a (soft) intervention on $\mathfrak { A }$ where we substitute the mechanism $\tau$ of $X$ with $G \tau$ . With some abuse of notation, we shall represent this SEM by $\mathfrak { A } ; \mathrm { d o } ( \tau : = G \tau )$ the graph of which is shown in Fig. 2b. Note that this SEM also has a unique stationary distribution (proof in Appendix F.2). Comparing the DA mechanism in $\mathfrak { A }$ (Fig. 2a) and the intervention $\mathfrak { A } ; \mathrm { d o } ( \tau : = G \tau )$ (Fig. 2b), we see: Observation 1 (soft intervention with DA). Distributions $\mathbb { P } _ { G X , Y , G , C } ^ { 2 1 }$ and $\mathbb { P } _ { X , Y , G , C } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = G \tau ) }$ are identical.

We can hence treat samples generated from $\mathfrak { A }$ via DA as if they were instead generated from $\mathfrak { A } ; \mathrm { d o } ( \tau : = G \tau )$ by intervening on $X$ . This allows us to re-write the DA+ERM risk from Eq. (6) as,

$$
R _ { \mathrm { D A } _ { G } + \mathrm { E R M } } ^ { \mathfrak { A } } ( h ) = R _ { \mathrm { E R M } } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = G \tau ) } ( h ) ,
$$

to emphasize that DA is equivalent to a (soft) intervention and as such can be used to reduce confounding bias when estimating $f$ , as we will show with the following example.

Example 2 (a linear Gaussian DA example). For scalars $\kappa , \sigma > 0$ , non-zero matrices $\mathbf { r } , \mathbf { T } \in \mathbb { R } ^ { * \times m }$ and vectors $\tau ^ { \top }$ , f , $\epsilon \in \mathbb { R } ^ { m }$ such that $\mathbf { f } ^ { \top } \pm \mp \kappa ^ { - 1 }$ so that the following SEM $\mathfrak { A }$ is solvable in $( X , Y )$

$$
X = \kappa \cdot \tau ^ { \mathsf { T } } Y + \mathbf { T } ^ { \mathsf { T } } C + \sigma N _ { X } , \quad Y = \mathbf { f } ^ { \mathsf { T } } X + \kappa \cdot \epsilon ^ { \mathsf { T } } C + \sigma N _ { Y } , \quad G X : = X + \gamma \cdot \mathbf { T } ^ { \mathsf { T } } G ,
$$

where $G , C , N _ { X } , N _ { Y }$ are conformable, centered Gaussian random vectors, $\kappa$ determines how much $( X , Y )$ are confounded and $\mathrm { r a n g e } ( \Gamma ^ { \top } ) \subseteq \mathrm { n u l l } ( \mathbf { f } ^ { \top } )$ so that $G X$ is a valid outcome invariant DA transformation of $X$ parameterized by $G$ with strength $\gamma > 0$ . This transformation can be viewed as translating $X$ along its level-set as shown in Fig. 3 and represents our prior knowledge about the symmetries of f for the purposes of this example.

Theorem 3 (causal estimation with DA+ERM). For SEM $\mathfrak { A }$ in Example 2, the following holds:

$$
\begin{array} { r } { \mathrm { C E R } _ { \mathfrak { A } } \Big ( \hat { \mathbf { h } } _ { D A _ { G } + E R M } ^ { \mathfrak { A } } \Big ) \leq \mathrm { C E R } _ { \mathfrak { A } } \Big ( \hat { \mathbf { h } } _ { E R M } ^ { \mathfrak { A } } \Big ) , \qquad e q u a l i t y i f f \qquad e ^ { \mathfrak { A } } [ G X \mid G ] \ \lrcorner _ { \mathrm { a . s . } } \mathbb { E } ^ { \mathfrak { A } } [ X \mid \xi ] . } \end{array}
$$

Proof. See Appendix F.5 for the proof.

That is, DA strictly reduces confounding bias in ATE estimate iff the induced intervention perturbes $X$ along spurious features. Importantly, Theorem 3 suggests that lower confounding bias is not a ‘free lunch’ with outcome invariance of DA and practitioners may need domain knowledge to construct DA that targets spurious features. Fortunately however, Theorem 3 also suggests that with outcome invariance, DA should not perform worse than ERM. We say that DA+ERM dominates ERM on causal estimation [33, p. 48]. Practitioners may therefore be advised to generously use such DA, as it achieves regularization in the worst case, and mitigates confounding bias as a ‘bonus’ in the best case.

# 4.2 Worst-case data augmentation with IVL regression

We once again point our attention to the graph of $\mathfrak { A } ; \mathrm { d o } ( \tau : = G \tau )$ from Fig. 2b to observe that: Observation 2 (IV-like DA parameters). In SEM A; $\mathrm { d o } ( \tau : = G \tau )$ , the DA parameters $G$ are IVL. In light of this we can now re-write the IV and IVL risks for $\mathfrak { A }$ ; $\mathrm { d o } ( \tau : = G \tau )$ to respectively read

$$
R _ { \mathrm { D A } _ { G } + \mathrm { I V } } ^ { \mathfrak { A } } ( h ) = R _ { \mathrm { I V } } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = G \tau ) } ( h ) , \qquad \ R _ { \mathrm { D A } _ { G } + \mathrm { I V L } _ { \alpha } } ^ { \mathfrak { A } } ( h ) = R _ { \mathrm { I V L } _ { \alpha } } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = G \tau ) } ( h ) .
$$

Corollary 1 (worst-case DA with $\mathrm { D A + I V L }$ regression). For SEM $\mathfrak { A }$ in Example 2, it holds that

$$
\hat { \mathbf { h } } _ { D A _ { G } + I M _ { \alpha } } ^ { \mathfrak { A } } \in \mathrm { a r g m i n } \operatorname* { m a x } _ { \mathbf { h } } R _ { D A _ { \mathbf { g } } + { \cal G } _ { \alpha } } ^ { \mathfrak { A } } R _ { D A _ { \mathbf { g } } + E R M } ( \mathbf { h } ) , \quad s . t . \quad { \mathcal { G } } _ { \alpha } : = \left\{ \mathbf { g } ~ \bigg | ~ \Gamma ^ { \top } \mathbf { g } \mathbf { g } ^ { \top } \mathbf { T } \preceq \bigg ( \frac { 1 } { \alpha } + 1 \bigg ) \mathbf { T } ^ { \top } \Sigma _ { G } ^ { \mathfrak { A } } \mathbf { T } \right\} .
$$

Proof. The result follows from Observation 1, Observation 2 and Theorem 1.

Corollary 2 (causal estimation with $\mathrm { D A + I V L }$ regression). For $\alpha , \gamma < \infty$ in SEM A from Example 2,

$$
\begin{array} { r } { \mathrm { C E R } _ { \mathfrak { A } } \left( { \hat { \mathbf { h } } } _ { D A _ { G } + I V L _ { \alpha } } ^ { \mathfrak { A } } \right) \leq \mathrm { C E R } _ { \mathfrak { A } } \left( { \hat { \mathbf { h } } } _ { D A _ { G } + E R M } ^ { \mathfrak { A } } \right) , \quad e q u a l i t y i f \quad \mathbb { E } ^ { \mathfrak { A } } [ G X \mid G ] \ \bot _ { \mathrm { a . s . } } \mathbb { E } ^ { \mathfrak { A } } [ X \mid \xi ] . } \end{array}
$$

Proof. The result follows directly from Theorem 2 and Observation 2.

Using DA parameters as IVL therefore simulates a worst-case, or adversarial application of DA within a set of transforms $\mathcal { G } _ { \alpha }$ . Of course Corollary 1 can also be viewed as a predictor that generalizes to treatment interventions encoded by $\mathcal { G } _ { \alpha }$ . As is intuitive, such a worst-case intervention improves our ATE estimation so long as the features of $X$ intervened along include some that are spurious (Corollary 2). DA and IVL regression may therefore be used in composition if the application can benefit from regularization and/ or better prediction generalization across DA-induced interventions, with a ‘bonus’ of lower confounding bias if the DA also augments any spurious features of $X$ .

![](images/e62b75e20f5dcab485816c4bb2da76465db739639a96933d906227f139012bd7.jpg)  
Figure 4: Simulation experiment for a linear Gaussian SEM. $\kappa$ represents the amount of confounding, $\gamma$ is the strength of DA and $\alpha$ is the IVL regularization parameter. Each data-point represents the average nCER over 25 trials with a $9 5 \%$ confidence interval (CI).

# 5 Related Work

Causal regularization is perhaps the most appropriate classification for this work. These methods aim for more robust prediction by mitigating the upstream problem of confounding bias in a more accessible way than is required for full identification. This is done, for example, by relaxing properties of auxiliary variables [12–15], as we have done via our IVL approach. Most relevant, however, are methods that re-purpose common regularizers, canonically used for estimation variance reduction and i.i.d. prediction generalization, for confounding bias mitigation. Of note is [18], where a certain linear modeling assumption allows the estimation of $\left\| \mathbf { f } \right\| ^ { 2 }$ from observational $( X , Y )$ data, which is then used to develop a cross-validation scheme for $\ell _ { 1 } , \ell _ { 2 }$ regularization. [19] conducted a similar theoretical analysis for the min-norm interpolator. To the best of our knowledge, we are the first to study the same for DA—re-purposing yet another ubiquitous regularizer to mitigate confounding bias.

Domain generalization (DG) [34] methods aim for prediction generalization to unseen test domains via robust optimization $( R O )$ [35] over a perturbation set $\mathcal { P }$ of possible test domains $\rho \in \mathcal P$ as

$$
R _ { \mathrm { R O } } ^ { \mathcal { P } } ( h ) : = \operatorname* { m a x } _ { \rho \in \mathcal { P } } R _ { \mathrm { E R M } } ^ { \rho } ( h ) ,
$$

Since generalizing to arbitrary test domains is impossible, the choice of perturbation set encodes one’s assumptions about which test domains might be encountered. Instead of making such assumptions a priori, it is often assumed to have access to data from multiple training domains which can inform one’s choice of perturbation set. This setting is explored in group distributionally robust optimization (DRO) [36]. Variations have been used to mitigate confounding bias and subsequently generalize to treatment interventions when used with interventional data [16, 37], confounder information (i.e. entire graph) [38–40] or some proxy thereof in the form of environments [41–43, 38]. We, however, do not assume access to any of these and instead synthesize interventions via DA.

Counterfactual DA strategies have been the primary lens for causal analyses of DA [44–50]. These aim for prediction robustness to treatment interventions via DA simulated counterfactuals.8 As with counterfactual reasoning more broadly, this requires strong assumptions—such as access to the full SEM [45, 46], auxiliary variables [44, 46, 49, 50], or causal graphs [47, 48]. By contrast, we show that outcome invariance of DA suffices for treatment intervention robustness without invoking counterfactuals. Moreover, prior work has largely overlooked causal effect estimation, often assuming reverse-causal settings where the ATE becomes trivial [44, 46, 45]. Ours is the first framework to study ATE estimation via DA with minimal structural assumptions.

Invariant prediction based methods aim to make predictions based on statistical relationships that remain stable across all domains in $\mathcal { P }$ . A common assumption, for instance, is that $\mathbb { P } _ { Y \mid X }$ is invariant across $\mathcal { P }$ , with only the marginal $\mathbb { P } _ { X }$ being allowed to vary. Invariance is also closely linked to causal discovery—following the classic ICM principle [4], causal mechanisms remain stable under interventions on inputs [25, 17]. This connection has inspired approaches that enforce invariance conditions to recover causal structures [16, 37]. IV regression can also be viewed as one such method, where the goal is to learn predictors whose residuals are invariant to the instruments [10, 9, 26, 51, 7]. More broadly, the principle of invariance, whether motivated by causality or otherwise, has proven useful for improving prediction generalization across heterogeneous settings [15, 41, 52, 14, 53–56, 34].

# 6 Experiments

We began by presenting results in the infinite-sample setting to emphasize that mitigating confounding bias is fundamentally not a sample size issue, i.e., not solvable through traditional regularization alone. In this section, we turn to the finite-sample regime and empirically evaluate the effectiveness of DA in reducing hidden confounding bias. Importantly, we do not use DA for its conventional purpose of augmenting data to improve i.i.d. generalization or reduce estimation variance. Throughout all experiments, we therefore fix the number of samples in the augmented dataset to match that of the original dataset since our focus lies squarely on robust prediction via confounding bias mitigation.

Finding baselines for evaluating our results is however a challenge—the problem of mitigating confounding bias given only observational $( X , Y )$ data and symmetry knowledge via DA is quite underexplored. Nevertheless, for the sake of completeness we make an effort to re-purpose existing methods from domain generalization, invariance learning and causal inference literature to be used as baselines. These methods often require access to additional variables (e.g. IVs, confounders, domains/environments, etc.), and to maintain fairness we will replace these with DA parameters $G$ . Such a comparison is conceptually valid since by virtue of being DG methods, they are essentially solving a robust loss of a similar form as in Corollary 1, giving us meaningful baselines for $\mathrm { D A + I V L }$ .

In addition to standard ERM, DA and IV regression, our baselines include DRO [36], invariant risk minimization (IRM) [41], invariant causal prediction (ICP) [16], regularization with invariance on causal essential set (RICE) [56], variance risk extrapolation (V-REx) and minimax risk extrapolation (MM-REx) [38]. We also include the causal regularization method by Kania and Wit [12] and the $\ell _ { 1 } , \ell _ { 2 }$ approaches by Janzing [18]. We discretise $G$ if the method accepts only discrete variables. For IVL regression, we select the regularization parameter $\alpha$ in a variety of ways, including vanilla cross validation (CV), level-based CV (LCV) and confounder correction (CC) as described in Appendix D. Other implementation details are provided in Appendix E, and the code to reproduce our results is publicly released at https://github.com/uzairakbar/causal-data-augmentation.

To make CER based evaluation more interpretable for our experiments, we propose the normalization

$$
\mathrm { n C E R } _ { \mathfrak { M } } ( h ) : = \frac { \mathrm { C E R } _ { \mathfrak { M } } ( h ) } { \mathrm { C E R } _ { \mathfrak { M } } ( h ) + \mathrm { C E R } _ { \mathfrak { M } } ( h _ { 0 } ) } \in [ 0 , 1 ] , \qquad h _ { 0 } ( \cdot ) : = \mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } ( X ) } [ Y ] ,
$$

where $h _ { 0 }$ represents the null treatment effect, i.e. when $X$ has no causal influence on $Y$ , then $\operatorname { \mathbb { E } } ^ { \mathfrak { M } ; \mathrm { d o } ( X ) } [ \bar { Y } \mid X ] = \operatorname { \mathbb { E } } ^ { \mathfrak { M } ; \mathrm { d o } ( X ) } [ Y ]$ . The normalized CER (nCER) can be considered a generalization of the metrics used by [18, 24, 31] in linear settings and similarly has the interesting property that it is 0 for the ground-truth causal solution $h = f \neq h _ { 0 }$ but 1 if there is pure confounding for $h \neq f = h _ { 0 }$ . Janzing argues in [24, 31] that using an Euclidean norm instead of the weighted norm in Eq. (8) is more relevant for causal settings, which also motivates our choice when evaluating results of the simulation and optical-device experiments described below. Conceptually, this is equivalent to evaluation based on the causal risk in Eq. (3) under the interventional distribution $X ^ { \prime } \sim \bar { \mathcal { N } } ( \mathbf { 0 } _ { m } , \mathbf { I } _ { m } )$ .

# 6.1 Simulation experiment

For the finite sample results of the linear SEM $\mathfrak { A }$ from Example 2, by taking $m = 3 2$ , $k = 3 1$ (dimension of $G$ ), $\sigma = 0 . 1$ and fixing $\pmb { \tau } ^ { \top } = \mathbf { 0 } _ { m }$ ,9 we sample a new $\mathbf { f } , \epsilon$ and $\mathbf { T } \in \mathbb { R } ^ { m \times m }$ from a standard normal distribution for each of the 32 experiments for every combination of $\kappa$ and $\gamma$ . Each time we construct a $\mathbf { T } : = \mathbf { V } _ { 0 }$ with $k$ rows as orthonormal basis of null(f ), such that the SVD of $\mathbf { f }$ is

$$
\mathbf { f } = \left[ \mathbf { u } \quad \mathbf { U } _ { 0 } \right] \left[ \begin{array} { c c } { \lambda } & { \mathbf { 0 } _ { 1 \times ( m - 1 ) } } \\ { \mathbf { 0 } _ { ( m - 1 ) \times 1 } } & { \mathbf { 0 } _ { ( m - 1 ) \times ( m - 1 ) } } \end{array} \right] \left[ \begin{array} { l } { \mathbf { v } ^ { \top } } \\ { \mathbf { V } _ { 0 } ^ { \top } } \end{array} \right] .
$$

Although this construction of $\mathbf { \delta T }$ relies on direct knowledge of $\mathbf { f }$ , which is of course unavailable in practice, we include it here purely for illustrative purposes. We treat access to $\mathbf { \delta T }$ as having prior knowledge about the structural symmetries of $\mathbf { f }$ , noting that this information alone is insufficient to recover f .

![](images/de4fa969986c1a6bdf203893076e076a5f59284a8c12064eb978a559902534d6.jpg)  
Figure 5: Experiment results; common OOD generalisation benchmarks compared against the ERM, $\mathrm { D A } +$ ERM and $\mathrm { D A + I V }$ baselines including $\mathrm { D A + I V L }$ .

We then generate $n = 2 0 4 8$ samples of $( X , Y )$ for each experiment. For ERM we use a closed form linear OLS solution. For ${ \mathrm { D A } } + { \mathrm { I V } } ,$ , we make use of linear 2SLS. Finally, $\mathrm { D A } { + } \mathrm { I V } \mathrm { L } _ { \alpha }$ was implemented using a closed form linear OLS solution between empirical versions (see Proposition 1) of

$$
X ^ { \prime } : = { \sqrt { \alpha } } X + \left( { \sqrt { 1 + \alpha } } - { \sqrt { \alpha } } \right) \mathbb { E } [ X \mid Z ] , \qquad Y ^ { \prime } : = { \sqrt { \alpha } } Y + \left( { \sqrt { 1 + \alpha } } - { \sqrt { \alpha } } \right) \mathbb { E } [ Y \mid Z ] .
$$

Our first experimental result in Fig. 4a compares the different estimation methods across varying levels of confounding $\kappa \in [ 0 , 1 ]$ . As expected, ERM performance degrades with increasing confounding. Applying DA alone already brings us closer to the causal solution, while $\mathrm { D A + I V L }$ achieves even better performance. $\mathrm { D A + I V }$ regression is unstable and generally performs poorly as it is under-determined.

Next, we fix the confounding and DA strengths at $\kappa = \gamma = 1$ , and sweep over the regularization parameter $\alpha \in [ 1 0 ^ { - 5 } , 1 0 ^ { 5 } ]$ for $\mathrm { D A } { + } \mathrm { I V } \mathrm { L } _ { \alpha }$ . Figure 4b shows that optimal performance is achieved for intermediate values of $\alpha$ , confirming that arbitrarily small values of $\alpha$ , while beneficial in the theoretical population setting (as suggested by Eq. (27) in the proof of Theorem 2), are suboptimal for finite samples.10 We also find that both CV and CC strategies effectively select reasonable values of $\alpha$ .

Lastly, Fig. 4c examines sensitivity to the DA strength $\gamma \in [ 1 0 ^ { - 2 . 5 } , 1$ 0], for fixed confounding strength $\kappa = 1$ . As expected, stronger DA results in stronger interventions on $X$ , which improves causal effect estimation. However, we also observe diminishing returns; when the variation induced by DA is either too small or too large, $\mathrm { D A } { + } \mathrm { I V } \mathrm { L } _ { \alpha }$ does not yield significant improvements over the $\mathrm { D A + E R M }$ baseline.

For completeness, we also benchmark our approach against other baseline methods on 16 distinct simulation SEMs with 2048 samples each. Aggregated results are presented in Fig. 5 (left most).

# 6.2 Real data experiments

Optical device dataset. The dataset from [24] consists of $3 \times 3$ pixel images $X$ displayed on a laptop screen that cause voltage readings $Y$ across a photo-diode. A hidden confounder $C$ controls two LEDs; one affects the webcam capturing $X$ , the other affects the photo-diode measuring $Y$ . The ground-truth predictor $\mathbf { f }$ is computed by first regressing $Y$ on $( \phi { \bar { ( } } X ) , C )$ , where $\phi ( X )$ are polynomial features of $X$ with degree $d \in \{ 1 , \cdot \cdot \cdot , 5 \}$ that best explains the data (degree 2 in most cases). The component corresponding to $C$ is then removed to recover f. We add Gaussian noise $G \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { X } / 1 0 )$ for DA and fit the methods from Sec. 6.1 on features $\phi ( G X )$ for $n = 1 0 0 0$ samples across 12 datasets. Note that using the same ground-truth polynomial degree for $\phi$ during evaluation is important here so as to avoid introducing statistical bias from model-miss-specification as our analysis squarely focuses on confounding bias. Figure 5 (middle) shows the results, where DA+ERM improves over ERM, and $\mathrm { D A } { + } \mathrm { I V } \mathrm { L }$ performs even better, outperforming other baselines.

Colored MNIST. We evaluate on the colored MNIST dataset [41], where labels are spuriously correlated with image color during training, but this correlation is flipped at test time. We use the same neural architecture and parameters as [41] across all baselines, training with the IV-based objective described in the Appendix C. DA is implemented via small perturbations to hue, brightness, contrast, saturation, and translation, each parameterized by $G \sim \beta ( 2 , 2 )$ . Although these do not directly manipulate color, the actual spurious feature, they still help reduce confounding. Results in Fig. 5 (rightmost) show that ERM underperforms, $\mathrm { D A + E R M }$ provides substantial gains, and $\mathrm { D A } { + } \mathrm { I V } \mathrm { L } _ { \alpha }$ performs competitively with the best DG baselines, with $\mathrm { D A } { + } \mathrm { I V L } _ { \alpha } ^ { \mathrm { C V } }$ achieving the best overall performance. Interested readers may also visit Appendix E.3, where we clarify the connection of the colored MNIST model with the cyclic SEM from Eq. (9).

# 7 Limitations

Necessity and practicality of prior knowledge. As discussed in Sec. 4, outcome invariance alone does not suffice to lower confounding bias and practitioners may need domain knowledge to construct DA that targets spurious features as well. Alternatively, one can also take a ‘carpet bombing’ approach by exhausting all available outcome invariant DA in hope that some may align with spurious features. Nevertheless, under outcome invariance, our methods should perform no worse than standard ERM.

Fundamentally, causal estimation from purely observational data is impossible without untestable assumptions. For instance, the IV (or IVL) assumptions of un-confoundedness and exclusion restriction are inherently untestable and must be justified through domain knowledge. Moreover, the requirement of alignment with spurious features in Theorem 2 is not an artifact of our IVL relaxation—it is a rephrasing of the exclusion principle that underlies identifiability in IV regression. If an IV does not influence $Y$ through the spurious features of $X$ , the corresponding causal components of $f$ cannot be identified [25]. IVLs, being relaxations of IVs, inherit these same untestable premises.

Viewed through the lens of IVs/IVLs (Observation 2), our assumptions on DA are arguably more modest than they may initially seem, especially since a symmetry-based DA model has well-established precedent in the literature [3, 28, 53, 57–63]. This correspondence can be summarized as follows:

![](images/1854261277da001b4080d5462d4ba8092b97e1728d8a928ba20438e8750aaae9.jpg)

In this light, our framework may in fact be quite practical in domains where valid IVs (or other auxiliary variables) are scarce, but plausible outcome-invariances—i.e., data augmentations—are abundant.

Finally, we recognize the hesitation in committing to strict notions of outcome invariance in practice and leave a more thorough exploration of approximate or even violated invariance to future work.

Choice of $\alpha$ . Selecting the IVL regularization parameter $\alpha$ in finite-sample settings is not straightforward. As outlined in Appendix D, we propose several strategies that work well empirically, though some may appear less principled since $\alpha$ is tuned via cross-validation within the same distribution, even though the task concerns OOD generalization. This challenge is not unique to IVL, but rather a broader limitation common to DG methods [64].

# 8 Conclusion

We conclude that our proposed causal framework for data augmentation (DA) enables re-purposing the widely used i.i.d. generalization tool for OOD generalization across treatment interventions. By interpreting outcome-invariant DA as interventions and IV-like variables, our approach reduces confounding bias and consequently improves both causal effect estimation and robust prediction.

# Acknowledgments

To my co-authors for their patience, to Zulfiqar for being my rubber-duck and saving the OpenReview submissions minutes before the deadline, and to all of ML pyos for lightening the chaos with comedy. Thank you.

This work was supported by the NSF (ECCS-2401391, IIS-2403240), and ONR (N000142512173).

References   
[1] Vladimir Naumovich Vapnik. Statistical learning theory. Wiley, 1998.   
[2] Connor Shorten and Taghi M. Khoshgoftaar. A survey on image data augmentation for deep learning. Journal of Big Data, 6:60, 2019. doi: 10.1186/s40537-019-0197-0.   
[3] Clare Lyle, Mark van der Wilk, Marta Kwiatkowska, Yarin Gal, and Benjamin Bloem-Reddy. On the benefits of invariance in neural networks, 2020. arXiv:2005.00178.   
[4] Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. Elements of causal inference: Foundations and learning algorithms. The MIT Press, 2017.   
[5] Judea Pearl. Causality. Cambridge University Press, 2009.   
[6] Liyuan Xu and Arthur Gretton. A neural mean embedding approach for back-door and front-door adjustment, 2022. arXiv:2210.06610.   
[7] Tongzheng Ren, Haotian Sun, Antoine Moulin, Arthur Gretton, and Bo Dai. Spectral representation for causal estimation with hidden confounders. In International Conference on Artificial Intelligence and Statistics, 2025.   
[8] Arash Mastouri, Yuhang Zhu, Limor Gultchin, Anna Korba, Ricardo Silva, Matt Kusner, Arthur Gretton, and Krikamol Muandet. Proximal causal learning with kernels: Two-stage estimation and moment restriction. In International Conference on Machine Learning, volume 139, 2021.   
[9] Rahul Singh, Maneesh Sahani, and Arthur Gretton. Kernel instrumental variable regression. In Advances in Neural Information Processing Systems, volume 32, 2019.   
[10] Rui Zhang, Masaaki Imaizumi, Bernhard Schölkopf, and Krikamol Muandet. Instrumental variable regression via kernel maximum moment loss. Journal of Causal Inference, 11(1), 2023. doi: 10.1515/ jci-2022-0073.   
[11] Niki Kilbertus, Matt J. Kusner, and Ricardo Silva. A class of algorithms for general instrumental variable models. In Advances in Neural Information Processing Systems, volume 33, 2020.   
[12] Lucas Kania and Ernst Wit. Causal regularization: On the trade-off between in-sample risk and out-ofsample risk guarantees, 2023. arXiv:2205.01593.   
[13] Peter Bühlmann and Dominik Cevid. Deconfounding and causal regularisation for stability and external validity. International Statistical Review, 88(S1):S114–S134, 2020. doi: 10.1111/insr.12383.   
[14] Michael Oberst, Nikolaj Thams, Jonas Peters, and David Sontag. Regularizing towards causal invariance: Linear models with proxies. In International Conference on Machine Learning, volume 139, 2021.   
[15] Dominik Rothenhäusler, Nicolai Meinshausen, Peter Bühlmann, and Jonas Peters. Anchor regression: Heterogeneous data meet causality. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(2):215–246, 2021.   
[16] Jonas Peters, Peter Bühlmann, and Nicolai Meinshausen. Causal inference by using invariant prediction: Identification and confidence intervals. Journal of the Royal Statistical Society Series B: Statistical Methodology, 78(5):947–1012, 2016. doi: 10.1111/rssb.12167.   
[17] Abbavaram Gowtham Reddy, Celia Rubio-Madrigal, Rebekka Burkholz, and Krikamol Muandet. When shift happens - confounding is to blame, 2025. arXiv:2505.21422.   
[18] Dominik Janzing. Causal regularization. In Advances in Neural Information Processing Systems, volume 32, 2019.   
[19] Chennuru Vankadara, Luca Rendsburg, Ulrike von Luxburg, and Debarghya Ghoshdastidar. Interpolation and regularization for causal learning. In Advances in Neural Information Processing Systems, volume 35, 2022.   
[20] William H. Greene. Econometric analysis. Prentice Hall, 2003. ISBN 9780130661890.   
[21] Kevin A. Clarke. The Phantom Menace: Omitted variable bias in econometric research. Conflict Management and Peace Science, 22(4):341–352, 2005. doi: 10.1080/07388940500339183.   
[22] John Fox. Simultaneous equation models and two-stage least squares. Sociological Methodology, 10: 130–150, 1979. doi: 10.2307/270769.   
[23] Michael R. Roberts and Toni M. Whited. Endogeneity in empirical corporate finance. In Handbook of the Economics of Finance, volume 2, chapter 7, pages 493–572. Elsevier, 2013. doi: 10.1016/ B978-0-44-453594-8.00007-0.   
[24] Dominik Janzing and Bernhard Schölkopf. Detecting confounding in multivariate linear models via spectral analysis. Journal of Causal Inference, 6(1), 2018.   
[25] Rune Christiansen, Niklas Pfister, Martin Emil Jakobsen, Nicola Gnecco, and Jonas Peters. A causal framework for distribution generalization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):6614–6630, 2022. doi: 10.1109/TPAMI.2021.3094760.   
[26] K. Muandet, A. Mehrjou, S. K. Lee, and A. Raj. Dual instrumental variable regression. In Advances in Neural Information Processing Systems, volume 33, 2020.   
[27] David A. Belsley. Two-or three-stage least squares? Computer Science in Economics and Management, 1: 21–30, 1988. doi: 10.1007/BF00435200.   
[28] Shuxiao Chen, Edgar Dobriban, and Jane H. Lee. A group-theoretic framework for data augmentation. Journal of Machine Learning Research, 21(245):1–71, 2020.   
[29] Artem Savkin, Thomas Lapotre, Kevin Strauss, Uzair Akbar, and Federico Tombari. Adversarial appearance learning in augmented Cityscapes for pedestrian recognition in autonomous driving. In IEEE International Conference on Robotics and Automation, pages 3305–3311, 2020. doi: 10.1109/ICRA40945.2020. 9197024.   
[30] Daphne Koller and Nir Friedman. Probabilistic graphical models: principles and techniques. MIT press, 2009.   
[31] Dominik Janzing and Bernhard Schölkopf. Detecting non-causal artifacts in multivariate linear regression models. In International Conference on Machine Learning, volume 80, 2018.   
[32] Jefrey M. Wooldridge. Econometric Analysis of Cross Section and Panel Data. The MIT Press, 2010.   
[33] Erich L. Lehmann and George Casella. Theory of Point Estimation. Springer, 2nd edition, 1998.   
[34] Krikamol Muandet, David Balduzzi, and Bernhard Schölkopf. Domain generalization via invariant feature representation. In International Conference on Machine Learning, volume 28, 2013.   
[35] Aharon Ben-Tal, Laurent El Ghaoui, and Arkadi Nemirovski. Robust Optimization. Princeton University Press, 2009. doi: 10.1515/9781400831050.   
[36] Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. Distributionally robust neural networks. In International Conference on Learning Representations, 2020.   
[37] Christina Heinze-Deml, Jonas Peters, and Nicolai Meinshausen. Invariant causal prediction for nonlinear models. Journal of Causal Inference, 6(2), 2018. doi: doi:10.1515/jci-2017-0016.   
[38] David Krueger, Ethan Caballero, Joern-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Remi Le Priol, and Aaron Courville. Out-of-distribution generalization via risk extrapolation (REx). In International Conference on Machine Learning, 2021.   
[39] Chaochao Lu, Yuhuai Wu, José Miguel Hernández-Lobato, and Bernhard Schölkopf. Invariant causal representation learning for out-of-distribution generalization. In International Conference on Learning Representations, 2022.   
[40] Hugh Dance and Benjamin Bloem-Reddy. Counterfactual cocycles: A framework for robust and coherent counterfactual transports, 2025. arXiv:2405.13844.   
[41] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization, 2019. arXiv:1907.02893.   
[42] Ahsan J. Cheema, Katherine L. Marks, Hamzeh Ghasemzadeh, Jarrad H. Van Stan, Robert E. Hillman, and Daryush D. Mehta. Characterizing vocal hyperfunction using ecological momentary assessment of relative fundamental frequency. Journal of Voice, 2024. ISSN 0892-1997. doi: 10.1016/j.jvoice.2024.10.025.   
[43] Seunghyup Han, Osama Waqar Bhatti, Woo-Jin Na, and Madhavan Swaminathan. Reinforcement learning applied to the optimization of power delivery networks with multiple voltage domains. In 2023 IEEE MTT-S International Conference on Numerical Electromagnetic and Multiphysics Modeling and Optimization (NEMO), 2023. doi: 10.1109/NEMO56117.2023.10202224.   
[44] Maximilian Ilse, Jakub M. Tomczak, and Patrick Forré. Selecting data augmentation for simulating interventions. In International Conference on Machine Learning, volume 139, 2021.   
[45] Jianhao Yuan, Francesco Pinto, Adam Davies, and Philip Torr. Not Just Pretty Pictures: Toward interventional data augmentation using text-to-image generators. In International Conference on Machine Learning, 2024.   
[46] Amir Feder, Yoav Wald, Claudia Shi, Suchi Saria, and David Blei. Data augmentations for improved (large) language model generalization. In Advances in Neural Information Processing Systems, volume 36, 2023.   
[47] Silviu Pitis, Elliot Creager, Ajay Mandlekar, and Animesh Garg. MoCoDA: Model-based counterfactual data augmentation. In Advances in Neural Information Processing Systems, volume 35, 2022.   
[48] Núria Armengol Urpí, Marco Bagatella, Marin Vlastelica, and Georg Martius. Causal action influence aware counterfactual data augmentation. In International Conference on Machine Learning, volume 235, 2024.   
[49] Divyat Mahajan, Shruti Tople, and Amit Sharma. Domain generalization using causal matching. In International Conference on Machine Learning, 2021.   
[50] Ahmed Aloui, Juncheng Dong, Cat Phuoc Le, and Vahid Tarokh. CATE estimation with potential outcome imputation from local regression. In Conference on Uncertainty in Artificial Intelligence, volume 286, 2025.   
[51] Liyuan Xu, Yutian Chen, Siddarth Srinivasan, Nando de Freitas, Arnaud Doucet, and Arthur Gretton. Learning deep features in instrumental variable regression. In International Conference on Learning Representations, 2021.   
[52] Bo Dai, Niao He, Yunpeng Pan, Byron Boots, and Le Song. Learning from conditional distributions via dual embeddings. In International Conference on Artificial Intelligence and Statistics, volume 54, 2017.   
[53] O. Montasser et al. Transformation-invariant learning and theoretical guarantees for OOD generalization. In Advances in Neural Information Processing Systems, volume 37, 2024.   
[54] Fanny Yang, Zuowen Wang, and Christina Heinze-Deml. Invariance-inducing regularization using worstcase transformations suffices to boost accuracy and spatial robustness. In Advances in Neural Information Processing Systems, volume 32, 2019.   
[55] Sravan Jayanthi, Letian Chen, Nadya Balabanska, Van Duong, Erik Scarlatescu, Ezra Ameperosa, Zulfiqar Haider Zaidi, Daniel Martin, Taylor Keith Del Matto, Masahiro Ono, and Matthew Gombolay. DROID: Learning from offline heterogeneous demonstrations via reward-policy distillation. In Conference on Robot Learning, volume 229. PMLR, 2023.   
[56] Ruoyu Wang, Mingyang Yi, Zhitang Chen, and Shengyu Zhu. Out-of-distribution generalization with causal invariant transformations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. doi: 10.1109/CVPR52688.2022.00047.   
[57] H. Shao et al. A theory of PAC learnability under transformation invariances. In Advances in Neural Information Processing Systems, 2022.   
[58] A. Fawzi and P. Frossard. Manitest: Are classifiers really invariant? In British Machine Vision Conference, 2015.   
[59] Y. Dubois et al. Lossy compression for lossless prediction. In Advances in Neural Information Processing Systems, 2021.   
[60] M. Petrache and S. Trivedi. Approximation-generalization trade-offs under (approximate) group equivariance. In Advances in Neural Information Processing Systems, 2023.   
[61] D. Romero and S. Lohit. Learning partial equivariances from data. In Advances in Neural Information Processing Systems, 2022.   
[62] S. Zhu et al. Understanding the generalization benefit of model invariance from a data perspective. In Advances in Neural Information Processing Systems, volume 34, 2021.   
[63] S. Wong et al. Understanding data augmentation for classification: When to warp? In Digital Image Computing: Techniques and Applications, 2016.   
[64] Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In International Conference on Learning Representations, 2021.   
[65] Tom Heskes. Bias-variance decompositions: The exclusive privilege of Bregman divergences, 2025. arXiv:2501.18581.   
[66] Steffen L. Lauritzen and Thomas S. Richardson. Chain graph models and their causal interpretations. Journal of the Royal Statistical Society Series B: Statistical Methodology, 64(3):321–348, 2002. doi: 10.1111/1467-9868.00340.   
[67] Gustavo Lacerda, Peter Spirtes, Joseph Ramsey, and Patrik O. Hoyer. Discovering cyclic causal models by independent components analysis. In Conference on Uncertainty in Artificial Intelligence, pages 366–374. AUAI Press, 2008.   
[68] Antti Hyttinen, Frederick Eberhardt, and Patrik O. Hoyer. Learning linear cyclic causal models with latent variables. Journal of Machine Learning Research, 13:3387–3439, 2012.   
[69] Joris M. Mooij, Dominik Janzing, Tom Heskes, and Bernhard Schölkopf. On causal discovery with cyclic additive noise models. In Advances in Neural Information Processing Systems, volume 24, 2011.   
[70] Stephan Bongers, Patrick Forré, Jonas Peters, and Joris M. Mooij. Foundations of structural causal models with cycles and latent variables. The Annals of Statistics, 49(5), 2021. doi: 10.1214/21-AOS2064.   
[71] Carl F. Christ. The Cowles Commission’s contributions to econometrics at Chicago, 1939-1955. Journal of Economic Literature, 32(1):30–59, 1994.   
[72] Mordecai Ezekiel. The Cobweb theorem. The Quarterly Journal of Economics, 52(2), 1938. doi: 10.2307/1881734.   
[73] John F. Muth. Rational expectations and the theory of price movements. Econometrica, 29(3):315–335, 1961.   
[74] Arnold Zellner and H. Theil. Three-stage least squares: Simultaneous estimation of simultaneous equations. Econometrica, 30(1):54–78, 1962. doi: 10.2307/1911287.   
[75] Alastair R. Hall. Generalized method of moments. In A Companion to Theoretical Econometrics, chapter 11, pages 230–255. Wiley, 2003. doi: 10.1002/9780470996249.ch12.   
[76] Andrew Bennett, Nathan Kallus, and Tobias Schnabel. Deep generalized method of moments for instrumental variable analysis. In Advances in Neural Information Processing Systems, volume 32, 2019.   
[77] Greg Lewis and Vasilis Syrgkanis. Adversarial generalized method of moments, 2018. arXiv:1803.07164.   
[78] John Johnston. Econometric Methods. McGraw-Hill, second edition, 1971.   
[79] Roger A. Horn and Charles R. Johnson. Matrix Analysis. Cambridge University Press, 1985.   
[80] Dennis S. Bernstein. Matrix Mathematics: Theory, Facts, and Formulas. Princeton University Press, second edition, 2009.

# Appendix—An Analysis of Causal Effect Estimation using Outcome Invariant Data Augmentation

# Uzair Akbar Georgia Tech

Niki Kilbertus TU Munich Helmholtz AI

Hao Shen TU Munich Fortiss GmbH

Krikamol Muandet Rational Intelligence CISPA

Bo Dai   
Georgia Tech   
Google DeepMind

# Contents

A Confounding Bias . 17   
B Simultaneity as Cyclic Structures in Equilibrium 18   
IV Regression Supplement . 20   
D IVL Regression Supplement. 22   
E Experiment Supplement . 23   
E.1 Simulation experiment 23   
E.2 Optical device experiment 24   
E.3 Colored-MNIST experiment 24

# F Proofs 27

F.1 Proof of Proposition 1—IVL regression closed form solution in the linear case 27   
F.2 Proof of Proposition 2—Existence of an interventional distribution given a DA 28   
F.3 Proof of Theorem 1—Robust prediction with IVL regression 29   
F.4 Proof of Theorem 2—Causal estimation with IVL regression 31   
F.5 Proof of Theorem 3—Causal estimation with DA+ERM 33   
F.6 Miscellaneous supporting lemmas 34

# List of Symbols

The notation is largely borrowed from [4], with some overloading where necessary.

Rn×∗ $n \times *$ Euclidean space; dimension $^ *$ conformal with $\&$ inferred from context. $x$ Scalar. $\mathbf { x }$ Vector. When $\mathbf { x } ^ { \top }$ is described as a vector, it means $\mathbf { x }$ is a flat $1 \times *$ matrix. $\mathbf { X }$ Matrix. $\mathcal { X }$ Set. $X$ Random vector. $\mathfrak { M }$ SEM. $X ^ { { \mathfrak { M } } }$ Random vector $X$ with its SEM $\mathfrak { M }$ specified when unclear from context. $\mathbb { P } _ { X } ^ { \mathfrak { M } }$ Distribution of $X$ entailed by $\mathfrak { M }$ . Superscript dropped if clear from context. $\Sigma _ { X } ^ { \mathfrak { M } }$ Variance–covariance matrix of $X$ under distribution $\mathbb { P } _ { X } ^ { \mathfrak { M } }$ . $\Sigma _ { X , Y } ^ { \mathfrak { M } }$ Cross–covariance matrix of $X$ and $Y$ under distribution $\mathbb { P } _ { X , Y } ^ { \mathfrak { M } }$ . $\mathbb { E } ^ { \mathfrak { M } } [ X ]$ Expected value of $X$ under distribution $\mathbb { P } _ { X } ^ { \mathfrak { M } }$ . $\mathrm { d o } ( X : = { \bf x } )$ Intervention— $X$ is set to $\mathbf { x }$ during data generation. $\operatorname { d o } ( X )$ Shorthand for $\mathrm { d o } ( X : = X ^ { \prime } )$ where $X ^ { \prime } \sim \mathbb { P } _ { X } ^ { \mathfrak { M } }$ is i.i.d. to $X$ . ${ \mathfrak { M } } ; \mathrm { d o } ( X : = \mathbf { x } )$ Intervention SEM. ${ \mathfrak { M } } _ { X = \mathbf { x } }$ SEM with mechanisms of $\mathfrak { M }$ , but exogenous noise distribution $\mathbb { P } _ { N | X = \mathbf { x } } ^ { \mathfrak { M } }$ . MY =y; do(X := x) Counterfatual SEM—intervention SEM of ${ \mathfrak { M } } _ { Y = \mathbf { y } }$ . $X \perp \perp Y$ Random vectors X, Y are statistically independent, i.e. PMY |X $\mathbb { P } _ { Y | X } ^ { \mathfrak { M } } = \mathbb { P } _ { Y } ^ { \mathfrak { M } }$ $\mathbf x \perp \mathbf y$ $\mathbf x , \mathbf y$ are perpendicular, i.e. $\mathbf { x } ^ { \top } \mathbf { y } = 0$ . For random vectors, $X ^ { \top } Y = 0$ a.s. $\hat { h } ^ { \mathfrak { M } }$ Population/ infinite-sample estimate based on distribution $\mathbb { P } ^ { \mathfrak { M } }$ . $\hat { h } ^ { \mathcal { D } }$ Finite-sample estimate based on samples in the dataset $\mathcal { D }$ .

# A Confounding Bias

Statistical vs. causal inference. The target estimand for the statistical risk in Eq. (2) is the Bayes optimal predictor $\mathbb { E } ^ { \mathfrak { M } } [ Y \mid X = \mathbf { x } ]$ . And the target estimand for the causal risk in Eq. (3) is the average treatment effect (ATE) $\mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } ( X : = \mathbf { x } ) } [ Y \mid X = \mathbf { x } ] = f ( \mathbf { x } )$ . As such, statistical inference is concerned with predictions of outcome $Y$ , whereas causal inference is concerned with estimating $f ( \mathbf { x } )$ .

Statistical vs. confounding bias. Both types of inference are subject to bias. Statistical bias arises due to miss-specification of the hypothesis class $\mathcal { H }$ , whereas confounding bias arises due to how the data are generated. The former is therefore a property of the estimator while the later is a property of the data itself. For an estimator $\hat { h } ^ { \mathcal { D } }$ with the expected value $\bar { h } ( \cdot ) = \mathbb { E } _ { \mathcal { D } } ^ { \mathfrak { M } } \left[ \hat { h } ^ { D } ( \cdot ) \right]$ , we define these as

$$
\begin{array} { r l r } & { } & { \mathrm { S t a t i s t i c a l ~ b i a s ~ a t ~ \mathbf { x } : = { \mathbb E } } ^ { \mathfrak { M } } [ Y \mid X = \mathbf { x } ] - \bar { h } ( \mathbf { x } ) , } \\ & { } & { \mathrm { C o n f o u n d i n g ~ b i a s ~ a t ~ \mathbf { x } : = { \mathbb f } } ( \mathbf { x } ) - { \mathbb E } ^ { \mathfrak { M } } [ Y \mid X = \mathbf { x } ] . } \end{array}
$$

Bias-variance decomposition of the causal risk. Because the treatment $X$ and residual $\xi$ are not correlated under $\mathfrak { M }$ ; $\operatorname { d o } ( X )$ in Eq. (1), for any loss function $\ell$ that admits a ‘clean’ or ‘additive’ bias-variance decomposition [65], the causal risk in Eq. (3) also admits a bias-variance decomposition. Using squared loss as an example, we have for some hypothesis $\hat { h } ^ { \mathcal { D } }$ ,

$$
\begin{array} { r l } & { \Rightarrow \bar { H } _ { \mathrm { C R } } ^ { \mathrm { 9 D } } \bigg ( \hat { h } ^ { \mathrm { 9 D } } \bigg ) } \\ & { = \mathbb { E } ^ { \mathrm { 9 D } _ { \mathrm { c } } \mathrm { A o } ( X ) } \bigg [ \bigg \| Y - \hat { h } ^ { P } ( X ) \bigg \| ^ { 2 } \bigg ] , } \\ & { = \mathbb { E } ^ { \mathrm { 9 D } _ { \mathrm { c } } \mathrm { A o } ( X ) } \bigg [ \bigg \| J ( X ) + \xi - \hat { h } ^ { P } ( X ) \bigg \| ^ { 2 } \bigg ] , \quad \quad \quad \quad \quad \quad \quad ( \mathrm { S t u r a c t u r a l ~ e q . ~ o f ~ Y . } ) } \\ & { = \mathbb { E } ^ { \mathrm { 9 D } _ { \mathrm { c } } \mathrm { A o } ( X ) } \bigg [ \| \xi \| ^ { 2 } \bigg ] + \mathbb { E } ^ { \mathrm { 9 D } _ { \mathrm { c } } \mathrm { A o } ( X ) } \bigg [ \bigg \| f ( X ) - \hat { h } ^ { P } ( X ) \bigg \| ^ { 2 } \bigg ] , \quad \quad \mathrm { ( C r o s s ~ t e r m i s ~ o ~ a s ~ \xi \| ~ \mathcal { X } ^ { \mathrm { 9 D } ; \mathrm { d o } ( X ) } - \boldsymbol { 1 } ) } } \\ &  = \underbrace { \mathbb { E } ^ { \mathrm { 9 D } _ { \mathrm { c } } \mathrm { A o } ( X ) } \bigg [ \| \xi \| ^ { 2 } \bigg ] } _ { \mathrm { i n e r a l s h i e n t i e n t ~ \hat { ~ } \omega } } + \underbrace { \mathbb { E } ^ { \mathrm { 9 D } } \bigg [ \bigg \| f ( X ) - \hat { h } ^ { P } ( X ) \bigg \| ^ { 2 } \bigg ] } _ { \mathrm { c s i m a l ~ \hat { ~ } \omega } \mathrm { c r e a p . } ~ \hat { ~ } \mathrm { C F ~ \mathcal { R } ^ { \mathrm { 9 D } ; \mathrm { d o } ( X ) } ~ i d e n t i c a l b y ~ c o n s t u r c t i o n } . \quad } \end{array}
$$

We can show by following standard procedure that

$$
\mathbb { E } _ { \mathcal { D } } ^ { \mathfrak { M } } \left[ \mathrm { C E R } _ { \mathfrak { s p } } \left( \hat { h } ^ { \mathcal { D } } \right) \right] = \underbrace { \mathbb { E } _ { X } ^ { \mathfrak { M } } \left[ \left. f ( X ) - \bar { h } ( X ) \right. ^ { 2 } \right] } _ { \mathrm { ( a v e r a g e ) ~ b i a s ^ { 2 } ~ } } + \underbrace { \mathbb { E } _ { \mathcal { D } } ^ { \mathfrak { M } } \left[ \mathbb { E } _ { X } ^ { \mathfrak { M } } \left[ \left. \bar { h } ( X ) - \hat { h } ^ { \mathcal { D } } ( X ) \right. ^ { 2 } \right] \right] } _ { \mathrm { v a r i a n c e } } .
$$

Since $\hat { h } ^ { \mathfrak { M } } ( X ) = \bar { h } ( X )$ for any population estimate, CER equals the average squared estimation bias

$$
\operatorname { C E R } _ { \mathfrak { M } } \left( { \hat { h } } ^ { { \mathfrak { M } } } \right) = \mathbb { E } _ { X } ^ { { \mathfrak { M } } } \left[ \left. f ( X ) - { \hat { h } } ^ { { \mathfrak { M } } } ( X ) \right. ^ { 2 } \right] = \mathbb { E } _ { X } ^ { { \mathfrak { M } } } \left[ \left. f ( X ) - { \bar { h } } ( X ) \right. ^ { 2 } \right] .
$$

For a rich enough hypothesis class, the ERM estimate coincides with the Bayes optimal predictor $\hat { h } _ { \mathrm { E R M } } ^ { \mathfrak { M } } ( \cdot ) = \mathbb { E } ^ { \mathfrak { M } } [ Y \mid X = \cdot ]$ and the CER exactly equals the (average squared) confounding bias as we define it above. For a general estimate $\hat { h } ^ { \mathcal { D } }$ , however, the CER also contains statistical bias. Nevertheless, our claims of “better causal estimation via reducing confounding bias” rest on the fact that we are essentially manipulating the data via DA and/or using treatment randomization sources in the form of IVLs. And recall that confounding bias is a property of the data.

# B Simultaneity as Cyclic Structures in Equilibrium

# Linear cyclic assignments

SEMs with cyclic structures have been well studied both in the linear case [66–68], as well as the non-linear case [69, 70]. Here we briefly provide a causal interpretation to linear simultaneous equations as SEMs with cyclic assignments.

Consider a square matrix $\mathbf { M } \in \mathbb { R } ^ { d \times d }$ and the SEM

$$
{ \cal W } = { \bf M } { \cal W } + { \cal N } ,
$$

where random noise vector $N$ is exogenous and $\mathbf { M }$ allows for a cyclic structure. We enforce $( \mathbf { I } _ { d } - \mathbf { M } )$ to be invertible so that the above equation has a unique solution $W$ for any given $N$ . Re-writing the structural form in Eq. (10) into a reduced form, the distribution over $W$ is defined by

$$
W = \left( \mathbf { I } _ { d } - \mathbf { M } \right) ^ { - 1 } N .
$$

One way we can present a causal interpretation of the above solution is to view it as a stationary point to the following sequence of random vectors $W _ { t }$

$$
W _ { t } = \mathbf { M } W _ { t - 1 } + N ,
$$

which converges if M has a spectral norm strictly smaller than one so that $\mathbf { M } ^ { t }  0$ as $t \to \infty$ . The structural form Eq. (10) essentially describes the iterative application of this operation. And in the limit the distribution of $\scriptstyle \operatorname* { l i m } _ { t \to \infty } { \dot { W } } ^ { t }$ will be the same as the reduced form Eq. (11). Although equivalent, reduced form of a cyclic SEM (if one exists) obscures the causal relations in the data generation process.

Furthermore, we restrict our models to not have any “self-cycles” (an edge from a vertex to itself). So, e.g., the matrix M in Eq. (10) has all zero diagonal entries. This not only simplifies our analysis by providing a simple and intuitive interpretation for our definition of DA in Sec. 2.3, but it also ensures that non-linear SEMs entail unique, well-defined distributions under mild assumptions [70, 67].

Similarly we can write the example SEM $\mathfrak { M }$ from Example 1 in this (block matrix) form as

$$
\underset { W } { \underbrace { \left[ \overset { X } { Y } \right] } } = \underset { \mathbf { M } } { \underbrace { \left[ \overset { \mathbf { 0 } _ { m \times m } } { \overbrace { \mathbf { f } ^ { \top } } } \quad \overset { \pmb { \tau } ^ { \top } } { \mathbf { 0 } _ { 1 \times 1 } } \right] } } \underset { W } { \underbrace { \left[ \overset { X } { Y } \right] } } + \underset { W } { \underbrace { \left[ \overset { \mathbf { T } ^ { \top } } { \mathbf { 0 } _ { 1 \times k } } \right] \overset { Z } + \left[ \overset { \mathbf { T } ^ { \top } } { \epsilon ^ { \top } } \right] \overset { C } { C } } } + \sigma \cdot \underset { N } { \underbrace { \left[ \overset { N _ { X } } { N _ { Y } } \right] } } ,
$$

For this simple case, $\left( \mathbf { I } _ { \left( m + 1 \right) } - \mathbf { M } \right)$ is always invertible so long as $\mathbf { f } ^ { \intercal } \tau ^ { \intercal } \neq 1$ from Lemma 3. Or we can also restrict $\left| \mathbf { f } ^ { \top } \pmb { \tau } ^ { \top } \right| < 1$ to ensure that the spectral norm of $\mathbf { M }$ is strictly smaller than 1. We sample from this SEM by first sampling all of the exogenous variables $Z , C , N _ { X } , N _ { Y }$ and then solving the above system for each sample of $X , Y$ via the reduced form in Lemma 3.

# A motivating example

Cyclic SEMs were first discussed in the econometrics literature [71] to model various observational phenomena, and often solved via 2SLS based IV regression [22] since it is computationally less costly compared to solving the entire system [27]. A classic example from economics [72, 73] is that of a supply and demand model $\mathfrak { M }$ where the relation of price $P$ of a good with quantity $Q$ of demand can be thought of as a cyclic feed-back loop where producers adjust their price in response to demand of the good and consumers change their demand in response to price of a good. In contrast, a change in consumer tastes or preferences would be an exogenous change on the demand curve and can therefore be used as an $\textsuperscript { I V Z }$ .

$$
\begin{array} { l } { { Q = \tau \cdot P + \gamma \cdot Z + N _ { Q } , } } \\ { { P = f \cdot Q + N _ { P } . } } \end{array}
$$

Where scalars $f , \tau$ are such that $| f \cdot \tau | < 1$ so that the system converges to an equilibrium. We say that the measurements made for $P$ and $Q$ are at the equilibrium state of the market11 with zero mean measurement noise $N _ { P } , N _ { Q }$ respectively.

Mitigating simultaneity bias for causal effect estimation. If we now want to estimate the effect of demand on price $f$ , standard regression will produce a biased estimate $\begin{array} { r } { \hat { f } _ { \mathrm { E R M } } ^ { \mathfrak { M } } = f + \frac { \mathrm { C o v } \left( Q , N _ { P } \right) } { \mathrm { V a r } \left( Q \right) } } \end{array}$ because of the simultaneity causing $Q$ and $N _ { P }$ to be correlated (to see this, substitute model of $P$ into the model of $Q$ ). We can now use IV regression to get an unbiased estimate of the effect of demand on price in the market as ${ \hat { f } } _ { \mathrm { I V } } ^ { \mathfrak { M } } = f$ .

Mitigating spurious correlations for robust prediction. Similarly, if the producer wants to predict the effect on demand if price is changed (i.e. intervened on), naive ERM will not be a good choice because it will also capture the spurious correlation from $Q  P$ . We therefore use three-stage-least→squares (3SLS) [74, 27] (or similar methods) to estimate the ATE $\hat { \tau } _ { 3 \mathrm { S L S } } ^ { \mathfrak { M } } = \mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } ( P : = . ) } [ Q \ | \ { P } = . ]$ where we use the first two stages to estimate $\hat { f } _ { \mathrm { I V } } ^ { \mathfrak { M } }$ , followed by ERM to regress from the residuals $\hat { N } _ { P } : = P - \hat { f } _ { \mathrm { I V } } ^ { \mathfrak { M } } \cdot Q$ to $Q$ in the third stage.

# Implications for independence of causal mechanisms

Here we clarify how the equilibrium assumption/interpretation of cyclic SEMs is not at odds with the classic independent causal mechanism (ICM) principle [4]. Note that our SEM formulation in Eq. (1) is a direct instantiation of the ICM principle as described by Peters et al. [4]. The two equations represent the autonomous mechanisms, and their independence is captured by the mutual independence of the exogenous noise terms $N _ { X } , N _ { Y }$ . The simultaneity in our model is not a violation of ICM, but rather the equilibrium state resulting from the interaction of these two independent mechanisms. Assuming the existence of this equilibrium is a statement about the scope of systems under analysis, and not about the nature of the mechanisms themselves. Indeed, surgically changing $\tau$ to some $\tau ^ { \prime }$ , for example, does not in itself alter $f$ and vice versa. And precisely because of the ICM, this may or may not make the system unstable depending on the nature of $\tau ^ { \prime }$ . Nevertheless, in our setting, Proposition 2 (Appendix F.2) shows that soft interventions induced by outcome-invariant DA are always stable.

# C IV Regression Supplement

Two-stage estimators. Minimizing the risk in Eq. (5) is known as two-stage IV regression. Another two-stage IV regression approach that we use in our theoretical results is to minimize the risk [8, 15]

$$
R _ { \mathrm { I V _ { L B } } } ^ { \mathfrak { M } } ( h ) : = \mathbb { E } ^ { \mathfrak { M } } \Big [ \big \| \mathbb { E } ^ { \mathfrak { M } } [ Y \mid Z ] - \mathbb { E } ^ { \mathfrak { M } } [ h ( X ) \mid Z ] \big \| ^ { 2 } \Big ] .
$$

This can be shown to lower-bound (hence the subscript LB) the risk in Eq. (5) under squared loss [8].

$$
\begin{array} { r l } & {  R _ { 1 \backslash } ^ { \mathrm { B D } } ( h ) = \mathbb { E } \Big [ \| Y - \mathbb { E } [ h ( X ) \ | \ Z ] \| ^ { 2 } \Big ] , } \\ & { = \mathbb { E } \Big [ \| ( Y - \mathbb { E } [ Y \ | \ Z ] ) + ( \mathbb { E } [ Y \ | \ Z ] - \mathbb { E } [ h ( X ) \ | \ Z ] ) \| ^ { 2 } \Big ] , \quad \mathrm { ( A d d i n g ~ a n d ~ s u b r r a c t i n g ~ \mathbb { E } [ Y \ | \ Z ] ) } } \\ & { = \mathbb { E } \Big [ \| Y - \mathbb { E } [ Y \ | \ Z ] \| ^ { 2 } \Big ] + \mathbb { E } \Big [ \| \mathbb { E } [ Y \ | \ Z ] - \mathbb { E } [ h ( X ) \ | \ Z ] \| ^ { 2 } \Big ] \qquad \mathrm { ( E x p a n d ~ s q u a r e d ~ n o m m ) } } \\ & { \qquad + 2 \mathbb { E } \Big [ \big ( Y - \mathbb { E } [ Y \ | \ Z ] \big ) ^ { \top } ( \mathbb { E } [ Y \ | \ Z ] - \mathbb { E } [ h ( X ) \ | \ Z ] \big ) \Big ] , } \\ & { = \mathbb { E } \Big [ \| Y - \mathbb { E } [ Y \ | \ Z ] \| ^ { 2 } \Big ] + \mathbb { E } \Big [ \| \mathbb { E } [ Y \ | \ Z ] - \mathbb { E } [ h ( X ) \ | \ Z ] \Big ] ^ { 2 } \Big ] , \qquad \mathrm { ( 1 2 ) } } \\ & { = \mathbb { E } \Big [ \| \mathbb { E } [ Y \ | \ Z ] - \mathbb { E } [ h ( X ) \ | \ Z ] \| ^ { 2 } \Big ] + \mathbb { E } \Big [ \mathbb { E } \Big [ \big ( Y - \mathbb { E } [ Y \ | \ Z ] \big ) ^ { 2 } \Big ] \ \Big ] , \qquad \mathrm { ( E n v e r ~ r a l a r ~ Y ) } } \\ & { = \mathbb { E } \Big [ \| \mathbb { E } [ Y \ | \ Z ] - \mathbb { E } [ h ( X ) \ | \ Z ] \| ^ { 2 } \Big ] + \mathbb { E } \Big [ \mathbb { E } \Big [ \big ( Y - \mathbb { E } [ Y \ ] \big ) ^ { 2 } \Big ] \Big ] \Big ] - \mathbb { E } [ \mathrm { N a r s ~ \mathrm { c a l a r } ~ \mathrm { c } ~ 1 3 } ] } \\ &  = \mathbb { E } \Big [ \| \mathbb { E } [ Y \ | Z ] - \mathbb { E } [ h ( X ) \ | \ Z ] \| ^ { 2 } \Big ] + \mathbb { E } [ \mathrm { N a r } ( Y \ | \ Z ) ] - \mathbb { E } [ \mathrm  N a r s  \end{array}
$$

where Eq. (13) follows from the definition of conditional variance and we get Eq. (12) by setting the cross term to zero since

$$
\begin{array} { r l } & { \Rightarrow \mathbb { E } \Big [ ( Y - \mathbb { E } [ Y \mid Z ] ) ^ { \top } ( \mathbb { E } [ Y \mid Z ] - \mathbb { E } [ h ( X ) \mid Z ] ) \Big ] } \\ & { = \mathbb { E } \Big [ \mathbb { E } \Big [ \big ( Y - \mathbb { E } [ Y \mid Z ] \big ) ^ { \top } ( \mathbb { E } [ Y \mid Z ] - \mathbb { E } [ h ( X ) \mid Z ] ) \Big \vert \ Z \Big ] \Big ] , } \\ & { = \mathbb { E } \Big [ \mathbb { E } \Big [ ( Y - \mathbb { E } [ Y \mid Z ] ) ^ { \top } \Big \vert \ Z \Big ] ( \mathbb { E } [ Y \mid Z ] - \mathbb { E } [ h ( X ) \mid Z ] ) \Big ] , } \\ & { = \mathbb { E } \Big [ ( \mathbb { E } [ Y \mid Z ] - \mathbb { E } [ Y \mid Z ] ) ^ { \top } ( \mathbb { E } [ Y \mid Z ] - \mathbb { E } [ h ( X ) \mid Z ] ) \Big ] , } \\ & { = \mathbb { E } \big [ \mathbb { 0 } ^ { \top } ( \mathbb { E } [ Y \mid Z ] - \mathbb { E } [ h ( X ) \mid Z ] ) \big ] = 0 , } \end{array}
$$

where Eq. (14) follows from the “taking out what is known” rule, i.e.,

$$
\mathbb { E } [ g ( B ) A \mid B ] = g ( B ) \mathbb { E } [ A \mid B ] .
$$

Generalized method of moments. The IV regression in our colored-MNIST experiment uses the popular generalized methods of moments (GMM) [75–77], or equivalently the conditional moment restriction (CMR) [8] framework which tries to directly solve for the fact that in Eq. (1) with scalar $Y$

$$
{ \mathbb E } ^ { \mathfrak { M } } [ \xi \mid Z ] = { \mathbb E } ^ { \mathfrak { M } } [ Y - f ( X ) \mid Z ] = 0 ,
$$

which holds as a direct consequence of un-confoundedness of $Z$ . For any $q : \mathcal { Z } \to \mathbb { R }$ , it then follows

$$
\mathbb { E } ^ { \mathfrak { M } } \left[ \left( Y - f ( X ) \right) \cdot q ( Z ) \right] = 0 .
$$

The GMM-IV estimate of $f$ therefore tries to enforce this condition [75–77] by minimizing the risk

$$
R _ { \operatorname { I V _ { G M M } } } ^ { \mathfrak { M } } ( h ) : = \sum _ { i = 1 } ^ { \mu } \mathbb { E } ^ { \mathfrak { M } } \left[ \left( Y - h ( X ) \right) \cdot q _ { i } ( Z ) \right] ^ { 2 } = \left\| \mathbb { E } ^ { \mathfrak { M } } [ \left( Y - h ( X ) \right) \cdot \mathbf { q } ( Z ) ] \right\| ^ { 2 } ,
$$

where $\mathbf { q } ( \cdot ) \in \mathbb { R } ^ { \mu }$ represents a vector form of the set of $\mu$ arbitrary real-valued functions $q _ { i }$ . A more general form of the above GMM based IV risk is to weight the norm by some SPD W [78, 75, 76]

$$
R _ { \operatorname { I V } _ { \bf G M M - W } } ^ { \mathfrak { M } } ( h ) : = \left. \mathbb { E } ^ { \mathfrak { M } } [ ( Y - h ( X ) ) \cdot \mathbf { q } ( Z ) ] \right. _ { \mathbf { W } } ^ { 2 } ,
$$

which gives the most statistically efficient estimator, minimizing the asymptotic variance, for $\mathbf { W } =$ $\Sigma _ { Z } ^ { - 1 }$ [78, 75, 76]. We use the same for our colored-MNIST experiments, together with the identity function $\mathbf { q } ( Z ) = Z$ . This gives us the final loss of the form

$$
R _ { \operatorname { I V } _ { \mathbf { G M M - } \pmb { \Sigma } _ { Z } ^ { - 1 } } } ^ { \mathfrak { M } } ( h ) = \left\| \mathbb { E } ^ { \mathfrak { M } } [ Z \cdot ( Y - h ( X ) ) ] \right\| _ { \pmb { \Sigma } _ { Z } ^ { - 1 } } ^ { 2 } .
$$

And the empirical version of which can be written as follows

$$
R _ { \mathrm { I V } _ { \mathrm { G M M } - \Sigma _ { Z } ^ { - 1 } } } ^ { \mathcal { D } } ( h ) : = \Big ( \hat { \mathbf { y } } - \mathbf { h } \Big ( \hat { \mathbf { X } } \Big ) \Big ) ^ { \top } \hat { \mathbf { Z } } \hat { \mathbf { Z } } ^ { \dagger } \Big ( \hat { \mathbf { y } } - \mathbf { h } \Big ( \hat { \mathbf { X } } \Big ) \Big ) ,
$$

where for dataset samples $\left( \mathbf { x } _ { i } , y _ { i } , \mathbf { z } _ { i } \right) \in \mathcal { D }$ , we construct the vector $\hat { \textbf { y } } : = [ y _ { 0 } , \dots , y _ { n } ] ^ { \intercal }$ , matrices $\hat { \mathbf X } : = [ \mathbf x _ { 0 } ^ { \top } , \cdots , \mathbf x _ { n } ^ { \top } ] ^ { \top }$ , $\hat { \textbf { Z } } : = \left[ \mathbf { z } _ { 0 } \quad \cdot \cdot \cdot \quad \mathbf { z } _ { n } \right] ^ { \intercal }$ with pseudo-inverse $\hat { \mathbf { Z } } ^ { \dagger }$ and define ${ \bf h } \big ( \hat { \bf x } \big ) : = \mathbf { \Omega }$ $[ h ( \mathbf { x } _ { 0 } ) , \cdot \cdot \cdot , h ( \mathbf { x } _ { n } ) ] ^ { \top }$ .

# D IVL Regression Supplement

Closed form solution in the linear case. The following result gives us a way to compute a closedform solution to the $\mathrm { I V L } _ { \alpha }$ regression problem in the linear Gaussian case. An empirical version of this is used for our linear experiments.

Proposition 1 $\mathrm { ( I V L } _ { \alpha }$ closed form solution). For SEM M in Example $\cdot$ , $\hat { \mathbf { h } } _ { I V L _ { \alpha } } ^ { \mathfrak { M } }$ is the closed form linear OLS solution between

$$
X ^ { \prime } : = a X + b \mathbb { E } [ X \mid Z ] , \qquad Y ^ { \prime } : = a Y + b \mathbb { E } [ Y \mid Z ] ,
$$

where

$$
a : = \sqrt { \alpha } , \qquad b : = \sqrt { 1 + \alpha } - \sqrt { \alpha } .
$$

Proof. See Appendix F.1 for the proof.

For the empirical version of Proposition 1 we fit a closed-form OLS regressor between

$$
\begin{array} { r l r l } & { X ^ { \prime } : = \sqrt { \alpha } X + \left( \sqrt { 1 + \alpha } - \sqrt { \alpha } \right) \hat { \mathbf { Z } } \hat { \mathbf { Z } } ^ { \dagger } X , } & & { Y ^ { \prime } : = \sqrt { \alpha } Y + \left( \sqrt { 1 + \alpha } - \sqrt { \alpha } \right) \hat { \mathbf { Z } } \hat { \mathbf { Z } } ^ { \dagger } Y , } \end{array}
$$

where $\hat { \mathbf { Z } } , \hat { \mathbf { Z } } ^ { \dagger }$ are as defined in Eq. (16).

Choice of regularization parameter. We try the following approaches to select the parameter $\alpha$ .

Cross validation $( C V )$ , or any variation thereof. We specifically use the following two in our experiments; (i) vanilla CV with $2 0 \%$ samples held-out for validation (ii) level cross validation $( L C V )$ for when $Z$ is discrete, where hold-out data corresponding to $2 0 \%$ of the levels of $Z$ for validation.

Confounder correction $( C C )$ , where in a linear setting we follow an approach similar to [18] by the length of estimating the length of the true solution $\hat { h } _ { \mathrm { D A + I V L } _ { \alpha } } ^ { \mathcal { D } }$ is closest to the estimated length of the ground truth solution. $f$ from the observational data $\mathcal { D }$ . We then chose $\alpha$ such that

![](images/4535a7dbb4cf1b6e14e986259153a9277da4b28616373efab7a871be53245a0b.jpg)  
Figure 6: Simulation of the linear Gaussian SEM of Example 2 with the same setting as Fig. 4, but $\tau ^ { \top }$ , f sampled uniformly over a unit sphere, representing a cyclic structure. Each data-point represents the average nCER over 25 trials with a $9 \hat { 5 } \%$ CI.

![](images/954439d77d60dfaebddb1b436e45d0c52c4e9f062f3adf46eaebc0999e4057e7.jpg)  
Figure 7: Same experiment as Fig. 4, but with $\mathbf { \delta T }$ constructed by randomly selecting each basis of $\mathrm { n u l l } ( \mathbf { f } ^ { \top } )$ with a probability of $2 / 3$ , simulating the effect of knowing only some symmetries of $\mathbf { f }$ . Each data-point represents the average nCER over 25 trials with a $9 5 \%$ CI.

# E Experiment Supplement

For the methods that use stochastic gradient descent (SGD), we use a learning rate of 0.01, batch size of 256 for 16 epochs. For baselines that require a discrete domains/environments, we uniformly discretise each dimension of $G$ into 2 bins. Higher discretisation bins renders most baselines ineffective since each domain/environment rarely has more than 1 sample. To keep the comparison fair, however, we also discretize $G$ for $\mathrm { I V L } _ { \alpha }$ regression when using LCV. For the colored MNIST experiment, all CV implementations including baselines use 5-folds for a random search over an exponentially distributed regularization parameter with rate parameter of 1. Same is the case for simulation and optical device experiments, except that $\mathrm { D A + I V L }$ methods use a log-uniform distributed regularization parameter over $[ \dot { 1 } 0 ^ { - 4 } , 1 ]$ . Since RICE [56] grows the dataset size by augmenting each sample $T$ times, we provide it a $1 / T$ sub-sample of the original data for fair comparison. Similarly, the causal regularization method by Kania and Wit [12] expects two datasets, a perturbed and an un-perturbed one, which we substitute with $1 / 2$ augmented data and $1 / 2$ original data respectively.

# E.1 Simulation experiment

For the parameter sweep experiments of Fig. 4, we generate a treatment of dimension $m = 3 2$ , but for the OOD baseline comparison experiment in Fig. 5 we use $m = 1 6$ . Furthermore, for the OOD baseline comparison experiment in Fig. 5, we randomly pick each basis of null(f) with a probability $1 / 3$ to construct $\mathbf { \delta T }$ (i.e., we know only some, but not all symmetries of f ).

We also provide additional linear simulation experiment results in Figs. 6 and 7—the former simulates a cyclic structure with a non-zero $\tau$ , and the later simulates a case where only some, but not all symmetries of $\mathbf { f }$ are known. The results of both are consistent with our original experiment in Fig. 4.

Table 2: nCER $\pm$ one standard error (SE) across the 12 optical-device datasets for various choices of DA. Bold and italic denote the lowest and second-lowest average nCER, respectively. Superscripts $^ *$ and $\dagger$ indicate a significant improvement over ERM or both ERM and $\mathrm { D A } +$ ERM, respectively, beyond a margin of SE. Lastly, — indicates that the method was too expensive for the value to be computed.   

<table><tr><td>Method</td><td>rotate &gt; hflip &gt; vflip</td><td>random-permutation</td><td>gaussian-noise</td><td>all</td></tr><tr><td>ERM</td><td>0.827 ± 0.079</td><td>0.827 ± 0.079</td><td>0.827 ± 0.079</td><td>0.823 ± 0.083</td></tr><tr><td>DA+ERM</td><td>0.617 ±0.085*</td><td>0.513± 0.082*</td><td>0.707 ± 0.090*</td><td>0.513±0.075*</td></tr><tr><td>DA+IVLgy</td><td>0.623 ± 0.087*</td><td>0.540 ± 0.085*</td><td>0.641±0.092*</td><td>0.533 ± 0.083*</td></tr><tr><td>DA+IVLICV</td><td>0.619 ± 0.087*</td><td>0.534 ± 0.082*</td><td>0.662 ± 0.091*</td><td>0.574 ± 0.087*</td></tr><tr><td>DA+IVLCC</td><td>0.623 ± 0.085*</td><td>0.527 ±0.082*</td><td>0.639 ± 0.076*</td><td>0.509 ± 0.078*</td></tr><tr><td>DA+IV</td><td>0.689 ± 0.065*</td><td>0.973 ± 0.011</td><td>0.955 ± 0.011</td><td>0.640 ± 0.083*</td></tr><tr><td>IRM</td><td>0.972 ± 0.010</td><td>0.960 ± 0.015</td><td>0.970 ± 0.009</td><td>0.953 ± 0.018</td></tr><tr><td>ICP</td><td>0.544 ± 0.019†</td><td>0.527 ±0.012*</td><td>0.646 ± 0.054†</td><td></td></tr><tr><td>DRO</td><td>0.975 ± 0.005</td><td>0.959 ± 0.012</td><td>0.981 ± 0.003</td><td>0.952 ± 0.014</td></tr><tr><td>RICE</td><td>0.966 ± 0.014</td><td>0.960 ± 0.012</td><td>0.974 ± 0.005</td><td>0.959 ± 0.016</td></tr><tr><td>V-REx</td><td>0.962 ± 0.024</td><td>0.957 ± 0.013</td><td>0.979 ± 0.005</td><td>0.925 ± 0.037</td></tr><tr><td>MM-REx</td><td>0.978 ± 0.013</td><td>1.000 ± 0.000</td><td>1.000 ± 0.000</td><td>1.000 ± 0.000</td></tr><tr><td>l1 Janzing‘19</td><td>0.821 ± 0.081</td><td>0.821 ± 0.081</td><td>0.821 ± 0.081</td><td>0.817 ± 0.077</td></tr><tr><td>l2 Janzing‘19</td><td>0.823 ± 0.076</td><td>0.823 ± 0.076</td><td>0.823 ± 0.076</td><td>0.828 ± 0.079</td></tr><tr><td>Kania, Wit ‘23</td><td>0.652 ± 0.084*</td><td>0.559 ± 0.084*</td><td>0.727 ± 0.088*</td><td>0.543 ± 0.080*</td></tr></table>

# E.2 Optical device experiment

In the simulation and optical device experiments, we fit a linear function $h ( . ) : = \mathbf { h } \in \mathbb { R } ^ { m }$ for a squared loss in all of our risk metrics. For $\mathrm { I V L } _ { \alpha }$ regression, we use the closed-form OLS solution from Appendix D. We also use a closed-form solution for ERM, $_ \mathrm { D A + E R M }$ and $\mathrm { D A + I V }$ (2SLS) baselines. The rest of the baselines (other than ICP) use SGD.

In Tab. 2, we report further experiments on the optical device dataset with various DA choices. The findings continue to confirm our main hypothesis: $\mathrm { D A + I V L }$ dominates DA+ERM, which itself dominates ERM. We never observe an opposite trend with statistical significance.

# E.3 Colored-MNIST experiment

In the colored MNIST experiment, we use the same 3-layer neural network (NN) architecture for $h$ across all methods comprising of a fully-connected input layer of input dimension $m$ , hidden layer of input/output dimension 256 and output classification layer with a Sigmoid function. Each layer is separated by an intermediary rectified linear unit activation function. For the IV risk, we use the empirical version of the GMM based risk from Eq. (16).

![](images/ceafeec15f0329e35c464cb0cdd1557fb081d46d854595f20ea08fa786adbf6c.jpg)  
Colored-MNIST as a cyclic SEM—From invariant prediction to estimating causal effects   
Figure 8: The data generation DAG for colored-MNIST as discussed by the original authors [41]. They aim to learn a predictor $h : \mathcal { X } \to \mathcal { Y }$ such that it is invariant to changes in $\mathbb { P } _ { X \mid Y }$ . We argue that this DAG view of colored-MNIST does not make it obvious how the true labeling function $f ( \mathbf { x } )$ is related to the ATE $\mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } ( X : = \mathbf { x } ) } [ Y \mid X = \mathbf { x } ]$ , which we believe is because it is virtually equivalent to the reduced form of our structural form presented in Fig. 9.

colored image colored image

![](images/35cf892699574cd9ab7135fa41889f908f49272233f5419734b0d6b0c994cbf2.jpg)  
(a) Graph for generating colored-MNIST data. (b) Augmented graph—exogenous variables explicitly shown.   
Figure 9: A cyclic SEM perspective of the colored-MNIST data—an MNIST image $N _ { X }$ is assigned color $C$ to produce a colored-MNIST image $X$ . This is then passed through the ground-truth labeling function $f$ to produce the true label $\tilde { Y }$ . We flip this with probability 0.25 to produce the observed label $Y$ , which in turn is flipped with probability $e$ (at train time $e \in \{ 0 . 1 , 0 . 2 \}$ and $e = 0 . 9$ at test time) to produce the color $C$ . These assignments are iteratively applied for any joint sample of the exogenous variables $N _ { X } , N _ { Y } , N _ { C }$ starting at arbitrary values of endogenous variables until convergence to the unique stationary point $X , Y , C$ (and $\tilde { Y }$ ).

In this section we give a cyclic SEM perspective of the colored-MNIST experiment from [41]. The task is binary classification of colored images $X$ from the MNIST dataset into low digits ( $y = 0$ for digits from 0 to 4) and high digits $( y = 1$ for digits from 5 to 9). The difficulty of the task arises from there being a higher spurious correlation between the color $C$ of the images $\boldsymbol { c } = 0$ for blue and $c = 1$ for green) and (noisy) labels $Y$ as compared to the correlation between the digits in the image and the label.

Consider the following cyclic SEM in Fig. 9.

$\begin{array} { r l } & { \mathbf { n } _ { X } \sim \mathbb { P } _ { N _ { X } } , n _ { Y } \sim \mathbb { B } ( 0 . 2 5 ) , n _ { c } \sim \mathbb { B } ( e ) } \\ & { \quad X = \mathsf { c o l o u r } ( C , \mathbf { n } _ { X } ) } \\ & { \quad \tilde { Y } = f ( X ) } \\ & { \quad Y = \mathop { \mathrm { x o r } } \Big ( \tilde { Y } , n _ { Y } \Big ) } \\ & { \quad C = \mathop { \mathrm { x o r } } ( Y , n _ { C } ) } \end{array}$ sample all exogenous variables apply color $C$ to the image generate ground-truth label with true labeling function flip the label with probability 0.25 generate color by flipping $Y$ with probability $e$ ,

where we first randomly sample an un-colored MNIST image $\mathbf { n } _ { X }$ , and some Bernoulli distributed label noise $n _ { Y } \sim \bar { \mathbb { B } } ( 0 . \bar { 2 } 5 )$ and color noise $n _ { C } \sim \mathbb { B } ( e )$ which is different for each environment $e \in \{ 0 . 1 , 0 . 2 \}$ . Then for some initial arbitrary values $\mathbf { x } _ { \mathrm { 0 } }$ , $\tilde { y } _ { 0 } , y _ { 0 }$ and $c _ { 0 }$ respectively for the observed colored image $X$ , the ground-truth label $\tilde { Y }$ , the observed noisy label $Y$ and the image color $C$ , we iteratively apply the following assignments from the SEM

$$
{ \begin{array} { r l r l } & { \mathbf { x } _ { t } = \mathbf { c o l o u r } ( c _ { t - 1 } , \mathbf { n } _ { X } ) } & { { \mathrm { a p p l y ~ c o l o r ~ } } C { \mathrm { ~ t o ~ t h e ~ i m a g e } } } \\ & { { \tilde { y } } _ { t } = f ( \mathbf { x } _ { t - 1 } ) } & { { \mathrm { g e n e r a t e ~ g r o u n d - t r u t h ~ l a b e l ~ w i t h ~ t r u e ~ l a b e l i n g ~ f u n c t i o n } } } \\ & { y _ { t } = \mathbf { x o r } ( { \tilde { y } } _ { t - 1 } , n _ { Y } ) } & { { \mathrm { f i i p ~ t h e ~ l a b e l ~ w i t h ~ p r o b a b i l i t y ~ } } 0 . 2 5 } \\ & { c _ { t } = \mathbf { x o r } ( y _ { t - 1 } , n _ { C } ) } & { { \mathrm { g e n e r a t e ~ c o l o r ~ b y ~ f i i p p i n g ~ } } Y { \mathrm { ~ w i t h ~ p r o b a b i l i t y ~ } } e , } \end{array} }
$$

until they converge while keeping all sampled exogenous variables $\mathbf { n } _ { X }$ $, n _ { Y } , n _ { C }$ fixed. It is straightforward to show that this SEM will converge after a maximum of $t = 5$ iterations12 due to the invariance of $f$ to the color of the image $C$ . Furthermore, this stationary-point will be uniquely determined by our exogenous samples $\mathbf { n } _ { X } , n _ { Y } , n _ { C }$ . And this is how we generate one sample $\left( \mathbf { x } , y \right)$ for our colored-MNIST experiment. We repeat this process to generate a sample $\left( \mathbf { x } , y \right)$ for each of $n$ samples $\mathbf { n } _ { X } , n _ { Y } , n _ { C } .$ .

Note that the ground-truth labeling function $f$ can only correctly predict the labels $7 5 \%$ of the time.   
At test time we flip the correlation between the label $Y$ and the image color $C$ by setting $e = 0 . 9$ .   
Also, the above cyclic SEM for colored-MNIST produces the same distribution for $( X , Y )$ as [41].

The above cyclic SEM perspective of colored-MNIST is interesting because it makes it clear that colored-MNIST is essentially a causal effect estimation task. Specifically, we can estimate the true labeling function $f$ by estimating the ATE $\mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } ( X : = \mathbf { x } ) } [ Y \mid X = \mathbf { x } ]$ since

$$
{ \begin{array} { r l r } { \mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } ( X \mathrel { \mathop { : } } = \mathbf { x } ) } [ Y \mid X = \mathbf { x } ] = \mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } ( X \mathrel { \mathop { : } } = \mathbf { x } ) } [ \mathbf { x o r } ( f ( X ) , N _ { Y } ) \mid X = \mathbf { x } ] , } \\ & { \qquad = \mathbb { E } ^ { \mathfrak { M } } [ \mathbf { x o r } ( f ( \mathbf { x } ) , N _ { Y } ) ] , } & { ( N _ { Y } \mid \mid X ) ^ { \mathfrak { M } ; \mathrm { d o } ( X \mathrel { \mathop { : } } = \mathbf { x } ) } , } \\ & { \qquad = \mathbb { E } ^ { \mathfrak { M } } [ f ( \mathbf { x } ) + N _ { Y } - 2 f ( \mathbf { x } ) N _ { Y } ] , } & { { \mathrm { ( D e f i n i t i o n ~ o f ~ x o r . ) } } } \\ & { \qquad = f ( \mathbf { x } ) + \mathbb { E } ^ { \mathfrak { M } } [ N _ { Y } ] - 2 f ( \mathbf { x } ) \mathbb { E } ^ { \mathfrak { M } } [ N _ { Y } ] , } \\ & { \qquad = \left( 1 - 2 \mathbb { E } ^ { \mathfrak { M } } [ N _ { Y } ] \right) f ( \mathbf { x } ) + \mathbb { E } ^ { \mathfrak { M } } [ N _ { Y } ] , } \\ & { \qquad = 0 . 5 f ( \mathbf { x } ) + 0 . 2 5 . } \end{array} }
$$

Because this is a binary classification task, we have

$$
\operatorname { r o u n d } \left( \operatorname { \mathbb { E } } ^ { \mathfrak { M } ; \operatorname { d o } ( X : = \mathbf { x } ) } [ Y \mid X = \mathbf { x } ] \right) = f ( \mathbf { x } ) .
$$

This is in contrast to the original DAG perspective of colored-MNIST shown in Fig. 8, where the connection to the estimation of the causal mechanism $f$ is not immediately obvious. We argue that this is because the DAG in Fig. 8 is virtually equivalent to the reduced form of our structural form presented in Fig. 9.

# F Proofs

# F.1 Proof of Proposition 1—IVL regression closed form solution in the linear case

Proposition 1 $\mathrm { T V L } _ { \alpha }$ closed form solution). For SEM $\mathfrak { M }$ in Example $I$ , $\hat { \mathbf { h } } _ { I V L _ { \alpha } } ^ { \mathfrak { M } }$ is the closed form linear OLS solution between

$$
X ^ { \prime } : = a X + b \mathbb { E } [ X \mid Z ] , \qquad Y ^ { \prime } : = a Y + b \mathbb { E } [ Y \mid Z ] ,
$$

where

$$
a : = \sqrt { \alpha } , \qquad b : = \sqrt { 1 + \alpha } - \sqrt { \alpha } .
$$

Proof. The OLS solution for $( X ^ { \prime } , Y ^ { \prime } )$ minimizes the following ERM risk

$$
\begin{array} { r l r } {  { \Rightarrow \mathbb { E } [ \| Y ^ { \prime } - \mathbf { h } ^ { \top } X ^ { \prime } \| ^ { 2 } ] } } \\ & { = \mathbb { E } [ \| a Y + b \mathbb { E } \big [ Y \mid Z \big ] - \mathbf { h } ^ { \top } ( a X + b \mathbb { E } \big [ X \mid Z \big ] ) \| ^ { 2 } ] , } & { \mathrm { ( S u b s t i t u t e ~ i n ~ d e f i n i t i o n s ~ o f ~ } X ^ { \prime } , Y ^ { \prime } \Sigma \mathrm { ) } } \\ & { = \mathbb { E } [ \| a ( Y - \mathbf { h } ^ { \top } X ) + b \big ( \mathbb { E } [ Y \mid Z ] - \mathbf { h } ^ { \top } \mathbb { E } \big [ X \mid Z ] \big ) \| ^ { 2 } ] , } & { \mathrm { ( D i s t r i b u t e ~ t h e ~ s u b t r a c t i o n . ) } } \\ & { = a ^ { 2 } \mathbb { E } [ \| Y - \mathbf { h } ^ { \top } X \| ^ { 2 } ] + b ^ { 2 } \mathbb { E } [ \| \mathbb { E } \big [ Y \mid Z \big ] - \mathbf { h } ^ { \top } \mathbb { E } \big [ X \mid Z \big ] \| ^ { 2 } ] } & { \mathrm { ( E x p a n d ~ s q u a r e d ~ n o r m . ) } } \\ & { } & { + 2 a b \mathbb { E } [ ( Y - \mathbf { h } ^ { \top } X ) ^ { \top } ( \mathbb { E } \big [ Y \mid Z \big ] - \mathbf { h } ^ { \top } \mathbb { E } \big [ X \mid Z \big ] - \mathbf { h } ^ { \top } \mathbb { E } \big [ X \mid Z \big ] ) ] . } & { \mathrm { ( 1 7 ) } } \end{array}
$$

First we note that from the definitions of $a , b$ we have

$$
a ^ { 2 } = \sqrt { \alpha } , b ^ { 2 } + 2 a b = \left( \sqrt { 1 + \alpha } - \sqrt { \alpha } \right) ^ { 2 } + 2 \sqrt { \alpha } \left( \sqrt { 1 + \alpha } - \sqrt { \alpha } \right) = 1 .
$$

Now we evaluate the cross term in Eq. (17)

$$
\begin{array} { r l r } & { \Rightarrow \mathbb { E } \left[ \left( Y - \mathbf { h } ^ { \top } X \right) ^ { \top } \left( \mathbb { E } [ Y \mid Z ] - \mathbf { h } ^ { \top } \mathbb { E } [ X \mid Z ] \right) \right] } \\ & { = \mathbb { E } \left[ \mathbb { E } \left[ \left( Y - \mathbf { h } ^ { \top } X \right) ^ { \top } \left( \mathbb { E } [ Y \mid Z ] - \mathbf { h } ^ { \top } \mathbb { E } [ X \mid Z ] \right) \Bigm \rvert Z \right] \right] , } & { \mathrm { ( L a w ~ o f ~ i t e r a t e d ~ e x p e c t a t i o n . ) } } \\ & { = \mathbb { E } \left[ \mathbb { E } \left[ \left( Y - \mathbf { h } ^ { \top } X \right) ^ { \top } \Bigm \rvert Z \right] \left( \mathbb { E } [ Y \mid Z ] - \mathbf { h } ^ { \top } \mathbb { E } [ X \mid Z ] \right) \right] } & { \mathrm { ( T a k i n g ~ o u t ~ w h a t i s ~ k n o w n ; E q . ~ ( 1 5 ) . ) } } \\ & { = \mathbb { E } \left[ \left( \mathbb { E } [ Y \mid Z ] - \mathbf { h } ^ { \top } \mathbb { E } [ X \mid Z ] \right) ^ { \top } \left( \mathbb { E } [ Y \mid Z ] - \mathbf { h } ^ { \top } \mathbb { E } [ X \mid Z ] \right) \right] } \\ & { = \mathbb { E } \left[ \left. \mathbb { E } [ Y \mid Z ] - \mathbf { h } ^ { \top } \mathbb { E } [ X \mid Z ] \right. ^ { 2 } \right] . } & \end{array}
$$

Substituting this back in Eq. (17) we get

$$
\begin{array} { r l r } & { \Rightarrow \mathbb { E } \left[ \left\| Y ^ { \prime } - \mathbf { h } ^ { \top } X ^ { \prime } \right\| ^ { 2 } \right] } \\ & { = a ^ { 2 } \mathbb { E } \left[ \left\| Y - \mathbf { h } ^ { \top } X \right\| ^ { 2 } \right] + ( b ^ { 2 } + 2 a b ) \mathbb { E } \left[ \left\| \mathbb { E } [ Y \mid Z ] - \mathbf { h } ^ { \top } \mathbb { E } [ X \mid Z ] \right\| ^ { 2 } \right] , } & \\ & { = \alpha \mathbb { E } \left[ \left\| Y - \mathbf { h } ^ { \top } X \right\| ^ { 2 } \right] + \mathbb { E } \left[ \left\| \mathbb { E } [ Y \mid Z ] - \mathbf { h } ^ { \top } \mathbb { E } [ X \mid Z ] \right\| ^ { 2 } \right] , } & { \mathrm { ( F r o m ~ E q . ~ ( ~ Y ^ { \prime } ~ ) = ~ Y ^ { \prime } ~ ) } } \\ & { = \alpha R _ { \mathrm { E R M } } ^ { \mathfrak { M } } ( \mathbf { h } ) + R _ { \mathrm { N } } ^ { \mathfrak { M } } ( \mathbf { h } ) - \mathbb { E } [ \mathrm { V a r } ( Y \mid Z ) ] , } & { \mathrm { ( F r o m ~ E q . ~ ( ~ Y ^ { \prime } ~ ) = Y ^ { \prime } ~ ) } } \\ & { = R _ { \mathrm { T U . } _ { \alpha } } ^ { \mathfrak { M } } ( \mathbf { h } ) - \mathbb { E } [ \mathrm { V a r } ( Y \mid Z ) ] . } & \end{array}
$$

# F.2 Proof of Proposition 2—Existence of an interventional distribution given a DA

Proposition 2 (unique stationary interventional distribution). In SEM A from Eq. (9), given any $( \mathbf { g } , \hat { \mathbf { c } } , \mathbf { n } _ { X } , \mathbf { n } _ { Y } ) \sim P _ { G , C , N _ { X } , N _ { Y } } ^ { \hat { \mathfrak { A } } }$ , if for all $( \mathbf { x } _ { 0 } , \mathbf { y } _ { 0 } ) \in \mathcal { X } \times \mathcal { Y }$ the unique limits

$$
\begin{array} { r l } & { \mathbf { x } ^ { \mathfrak { A } } : = \underset { t  \infty } { \operatorname* { l i m } } \mathbf { x } _ { t } ^ { \mathfrak { A } } = \underset { t  \infty } { \operatorname* { l i m } } \tau \big ( \mathbf { y } _ { t - 1 } ^ { \mathfrak { A } } , \mathbf { c } , \mathbf { n } _ { X } \big ) , } \\ & { \mathbf { y } ^ { \mathfrak { A } } : = \underset { t  \infty } { \operatorname* { l i m } } \mathbf { y } _ { t } ^ { \mathfrak { A } } = \underset { t  \infty } { \operatorname* { l i m } } f \big ( \mathbf { x } _ { t - 1 } ^ { \mathfrak { A } } \big ) + \epsilon ( \mathbf { c } ) + \mathbf { n } _ { Y } } \end{array}
$$

exist, then in $\mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau )$ the unique limits

$$
\begin{array} { r l } & { \mathbf { x } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } : = \underset { t  \infty } { \mathrm { l i m } } ~ \mathbf { x } _ { t } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } = \underset { t  \infty } { \mathrm { l i m } } ~ \mathbf { g } \tau \Big ( \mathbf { y } _ { t - 1 } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } , \mathbf { c } , \mathbf { n } _ { X } \Big ) = \mathbf { g } \mathbf { x } ^ { \mathfrak { A } } , } \\ & { \mathbf { y } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } : = \underset { t  \infty } { \mathrm { l i m } } ~ \mathbf { y } _ { t } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } = \underset { t  \infty } { \mathrm { l i m } } ~ f \Big ( \mathbf { x } _ { t - 1 } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } \Big ) + \epsilon ( \mathbf { c } ) + \mathbf { n } _ { Y } = \mathbf { y } ^ { \mathfrak { A } } } \end{array}
$$

also exist.

Proof. First we try to show that

$$
\begin{array} { r } { \mathbf { y } _ { t } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } = \mathbf { y } _ { t } ^ { \mathfrak { A } } . } \end{array}
$$

For the base case, we have by construction

$$
\begin{array} { r } { \mathbf { y } _ { 0 } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } : = \mathbf { y } _ { 0 } = : \mathbf { y } _ { 0 } ^ { \mathfrak { A } } . } \end{array}
$$

For the step case, assuming that $\mathbf { y } _ { t } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } = \mathbf { y } _ { t } ^ { \mathfrak { A } }$ , we have13,

$$
\begin{array} { r l } & { \mathbf { y } _ { t + 2 } ^ { \mathfrak { A } _ { \xi } \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } = f \Big ( \mathbf { x } _ { t + 1 } ^ { \mathfrak { A } _ { \xi } \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } \Big ) + \epsilon ( \mathbf { c } ) + \mathbf { n } _ { Y } , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad ( \mathrm { I n v a r i a n c e \ t \ T o \ T e \ ~ g } ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad ( \mathrm { I n v a r i a n c e \ t \ T e \ u } \ \tau \ \cdot \ \ \frac { \mathfrak { A } _ { \xi } \mathrm { d o } ( \tau \tau \cdot \ \log \tau ) } { t } ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad ( \mathrm { A s s u m p t i o n \ y i \ m s \ t i o n \ y } _ { t } ^ { \mathfrak { A } _ { \xi } \mathrm { d o } ( \tau \cdot \log \tau ) } = \mathbf { y } _ { t } ^ { \mathfrak { A } _ { \xi } } ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & \end{array}
$$

Hence, we have shown that Eq. (19) holds for all even $t$ . For odd $t$ , we simply replace $t = 0$ with $t = 1$ in the base case

$$
\begin{array} { r l } & { \mathbf { y } _ { 1 } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } = f \Bigl ( \mathbf { x } _ { 0 } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } \Bigr ) + \epsilon ( \mathbf { c } ) + \mathbf { n } _ { Y } , } \\ & { \qquad = f \bigl ( \mathbf { x } _ { 0 } ^ { \mathfrak { A } } \bigr ) + \epsilon ( \mathbf { c } ) + \mathbf { n } _ { Y } , \qquad \mathrm { ( D e f n i t i o n s ~ \mathbf { x } _ 0 ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } ~ : = \mathbf { x } _ 0 = : \mathbf { x } _ 0 ^ { \mathfrak { A } } _ { 0 } ~ ) } } \\ & { \qquad = \mathbf { y } _ { 1 } ^ { \mathfrak { A } } , } \end{array}
$$

We have now finally shown that Eq. (19) holds for all $t \geq 0$ .

Next, it is now relatively straightforward to show that for any $t > 0$ , we have

$$
\begin{array} { r l } & { \mathbf { x } _ { t } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } = \mathbf { g } \tau \Big ( \mathbf { y } _ { t - 1 } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } , \mathbf { c } , \mathbf { n } _ { X } \Big ) , } \\ & { \qquad = \mathbf { g } \tau \big ( \mathbf { y } _ { t - 1 } ^ { \mathfrak { A } } , \mathbf { c } , \mathbf { n } _ { X } \big ) , } \\ & { \qquad = \mathbf { g } \mathbf { x } _ { t } ^ { \mathfrak { A } } . } \end{array}
$$

(Follows from Eq. (19).)

Finally, by applying limit as $t \to \infty$ to both sides of Eq. (19) and Eq. (20), we get

$$
\begin{array} { r l r } {  { \mathbf { y } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } = \operatorname* { l i m } _ { t \to \infty } \mathbf { y } _ { t } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } = \operatorname* { l i m } _ { t \to \infty } \mathbf { y } _ { t } ^ { \mathfrak { A } } = \mathbf { y } ^ { \mathfrak { A } } , } } \\ & { } & { \mathbf { x } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } = \operatorname* { l i m } _ { t \to \infty } \mathbf { x } _ { t } ^ { \mathfrak { A } ; \mathrm { d o } ( \tau : = \mathbf { g } \tau ) } = \operatorname* { l i m } _ { t \to \infty } \mathbf { g x } _ { t } ^ { \mathfrak { A } } = \mathbf { g } \operatorname* { l i m } _ { t \to \infty } \mathbf { x } _ { t } ^ { \mathfrak { A } } = \mathbf { g x } ^ { \mathfrak { A } } , } \end{array}
$$

where the limit can be moved past $\mathbf { g }$ in Eq. (21) because $\mathbf { g }$ is assumed continuous in its domain.

# F.3 Proof of Theorem 1—Robust prediction with IVL regression

Theorem 1 (robust prediction with IVL regression). For SEM M in Example $\cdot$ , the following holds:

$$
\hat { \mathbf { h } } _ { N L _ { \alpha } } ^ { \mathfrak { M } } \in \operatorname { a r g m i n } _ { \mathbf { h } } \operatorname* { m a x } _ { \zeta \in \mathcal { P } _ { \alpha } } R _ { E R M } ^ { \mathfrak { M } ; \mathrm { d o } \left( \Gamma ^ { \top } ( \cdot ) = \zeta \right) } ( \mathbf { h } ) , \quad s . t . \quad \mathcal { P } _ { \alpha } : = \bigg \{ \zeta \bigg | \zeta \zeta ^ { \top } \prec \bigg ( \frac { 1 } { \alpha } + 1 \bigg ) \mathbf { r } ^ { \top } \Sigma _ { Z } ^ { \mathfrak { M } } \Gamma \bigg \} .
$$

Proof. Write $X$ in terms of the exogenous variables $C , Z , N _ { X } , N _ { Y }$ using the reduced form from Lemma 3 as

$$
X = \tilde { Z } + \tilde { C } + \tilde { N } ,
$$

where for readability we represent

$$
\begin{array} { r } { \tilde { Z } : = \mathbf { M } _ { m \times m } \mathbf { \Gamma } ^ { \top } Z , \qquad \tilde { C } : = \mathbf { M } \left[ \mathbf { \Gamma } _ { \epsilon } ^ { \mathsf { T } ^ { \top } } \right] C , \qquad \tilde { N } : = \sigma \cdot \mathbf { M } \left[ \mathbf { \Gamma } _ { N _ { Y } } ^ { N _ { X } } \right] , } \end{array}
$$

with

$$
\mathbf { M } : = \left[ \begin{array} { c c c } { \mathbf { M } _ { m \times m } } & { \mathbf { M } _ { m \times 1 } } \\ { \mathbf { M } _ { 1 \times m } } & { \mathbf { M } _ { 1 \times 1 } } \end{array} \right] = \left[ \begin{array} { c c } { \mathbf { I } _ { m } } & { - \pmb { \tau } ^ { \top } } \\ { - \mathbf { f } ^ { \top } } & { 1 } \end{array} \right] ^ { - 1 } .
$$

Now, we start by writing the ERM objective under the intervention $\mathrm { d o } \big ( \mathbf { r } ^ { \top } ( \cdot ) : = \zeta \big )$ as

$$
\begin{array} { r l r l } & { \Rightarrow \mathcal { T } _ { \mathrm { e r f . H . A } } ^ { \mathrm { S y s h . } ( \Gamma ^ { \tau } ( \gamma _ { \tau } ) - \epsilon ) } [  \mathcal { T } - \mathbf { h } ^ { \tau }  ^ { 2 } ] , } & & { } \\ & { = \mathbb { E } ^ { 9 / \Omega _ { \mathrm { e f f . } } ( \Gamma ^ { \tau } ( \gamma ) - \epsilon ) } [  \mathcal { T } - \mathbf { h } ^ { \tau }  ^ { 2 } ] , } & & { } \\ & { = \mathbb { E } ^ { 9 / \Omega _ { \mathrm { e f f . } } ( \Gamma ^ { \tau } ( \gamma ) - \epsilon ) } [   \xi + ( \Gamma - \mathbf { h } ) ^ { \tau } ( \tilde { Z } + \tilde { C } + \tilde { N } )  ^ { 2 } ] , } & & { ( \gamma \mathrm { ~ s u m e n a l ~ t o t o ~ n ~ } \mathbb { E } \in \mathrm { ~ \mathfrak { q } . \gamma ~ } ( \gamma ) ) , } \\ & { = \mathbb { E } ^ { 9 / \Omega _ { \mathrm { e f f . } } ( \Gamma ^ { \tau } ( \gamma ) - \epsilon ) } [   \xi + ( \Gamma - \mathbf { h } ) ^ { \tau } ( \mathbf { M } _ { \mathrm { a x } \times \tau } + \tilde { C } + \tilde { N } )  ^ { 2 } ] , } & & { ( \tilde { Z } \mathrm { ~ \mathrm { E i n i m e r e n t i t i o n o n } ~ d e f i n i t i o n } , ) } \\ & { = \mathbb { E } ^ { 9 / \Omega _ { \mathrm { e f f . } } ( \Gamma ^ { \tau } ( \gamma ) - \epsilon ) } [   \xi + ( \mathbf { f } - \mathbf { h } ) ^ { \tau } ( \tilde { C } + \tilde { N } ) + ( \mathbf { f } - \mathbf { h } ) ^ { \tau } \mathbf { M } _ { \mathrm { a x } \times \tau }  ^ { 2 } ] , } & & { } \\ & { = \mathbb { E } ^ { 9 / \Omega _ { \mathrm { e f f . } } ( \Gamma ^ { \tau } ( \gamma ) - \epsilon ) } [  \xi + ( \mathbf { f } - \mathbf { h } ) ^ { \tau } ( \tilde { C } + \tilde { N } ) + \mathbf { h } ^ { \tau } \mathcal { T }  ^ { 2 } , } & & { } \\ &  = \mathbb { E } ^  9 / \end{array}
$$

(Follows from exogeneity of $\zeta$ under intervention, $\Rightarrow$ cross term zeros-out.)

$$
\begin{array} { r l } & { = \mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } \left( \mathbf { r } ^ { \top } ( \cdot ) : = \mathbf { 0 } _ { m } \right) } \Big [ \big \| Y - \mathbf { h } ^ { \top } X \big \| ^ { 2 } \Big ] + \mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } \left( \mathbf { r } ^ { \top } ( \cdot ) : = \zeta \right) } \Big [ \Big \| \mathbf { h } ^ { \prime \top } \zeta \Big \| ^ { 2 } \Big ] , } \\ & { = \mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } \left( \mathbf { r } ^ { \top } ( \cdot ) : = \mathbf { 0 } _ { m } \right) } \Big [ \big \| Y - \mathbf { h } ^ { \top } X \big \| ^ { 2 } \Big ] + \Big \| \mathbf { h } ^ { \prime \top } \zeta \Big \| ^ { 2 } , } \\ & { = \mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } \left( \mathbf { r } ^ { \top } ( \cdot ) : = \mathbf { 0 } _ { m } \right) } \Big [ \big \| Y - \mathbf { h } ^ { \top } X \big \| ^ { 2 } \Big ] + \mathrm { t r } \Big ( \zeta ^ { \top } \mathbf { h } ^ { \prime \top } \zeta \Big ) , } \\ & { = \mathbb { E } ^ { \mathfrak { M } ; \mathrm { d o } \left( \mathbf { r } ^ { \top } ( \cdot ) : = \mathbf { 0 } _ { m } \right) } \Big [ \big \| Y - \mathbf { h } ^ { \top } X \big \| ^ { 2 } \Big ] + \mathrm { t r } \big ( \mathbf { h } ^ { \prime \top } \zeta \zeta ^ { \top } \mathbf { h } ^ { \prime } \big ) . } \end{array}
$$

Now, note that the maximum of the trace term over $\zeta \in \mathcal { P } _ { \alpha }$ gives

$$
\begin{array} { r l } & { \underset { \xi \in \mathcal { P } _ { \alpha } } { \Rightarrow } \underset { \xi \in \mathcal { P } _ { \alpha } } { \mathrm { m a x } } \mathrm { ~ t r } \big ( \mathbf { h } ^ { \prime \top } \zeta \zeta ^ { \top } \mathbf { h } ^ { \prime } \big ) , } \\ & { = \Big ( \frac { 1 } { \alpha } + 1 \Big ) \mathrm { ~ t r } \Big ( \mathbf { h } ^ { \prime \top } \Big ( \mathbf { T } ^ { \top } \mathbb { E } ^ { \mathfrak { M } } [ Z Z ^ { \top } ] \mathbf { r } \Big ) \mathbf { h } ^ { \prime } \Big ) , } \\ & { = \Big ( \frac { 1 } { \alpha } + 1 \Big ) \mathbb { E } ^ { \mathfrak { M } } \big [ \mathrm { t r } \big ( \mathbf { h } ^ { \prime \top } \mathbf { T } ^ { \top } Z Z ^ { \top } \mathbf { T } \mathbf { h } ^ { \prime } \big ) \big ] , } \\ & { = \Big ( \frac { 1 } { \alpha } + 1 \Big ) \mathbb { E } ^ { \mathfrak { M } } \big [ \mathrm { t r } \big ( Z ^ { \top } \mathbf { T } \mathbf { h } ^ { \prime } \mathbf { h } ^ { \prime \top } \mathbf { T } ^ { \top } Z \big ) \big ] , } \end{array}
$$

$$
\begin{array} { r l } & { = \left( \frac { 1 } { \alpha } + 1 \right) \mathbb { E } ^ { \Re } \Big [ \big \| \mathbf { h } ^ { \prime \top } \mathbf { T } ^ { \top } Z \big \| ^ { 2 } \Big ] , } \\ & { = \left( \frac { 1 } { \alpha } + 1 \right) \mathbb { E } ^ { \Re } \Big [ \Big \| \big ( \mathbf { f } - \mathbf { h } \big ) ^ { \top } \mathbf { M } _ { m \times m } \mathbf { T } ^ { \top } Z \Big \| ^ { 2 } \Big ] , } \\ & { = \left( \frac { 1 } { \alpha } + 1 \right) \mathbb { E } ^ { \Re } \Big [ \Big \| \big ( \mathbf { f } - \mathbf { h } \big ) ^ { \top } \tilde { Z } \Big \| ^ { 2 } \Big ] . } \end{array}
$$

(Substitute in definition of $\mathbf { h } ^ { \prime ^ { \top } }$ .)

(Definition of $\tilde { Z }$ .)

We can now substitute this in while maximizing both sides of Eq. (24) over interventions $\zeta \in \mathcal { P } _ { \alpha }$ as

$$
\begin{array} { r l r } & {  \underset { \zeta \in \mathcal { R } _ { n } } { \operatorname* { m a x } } \frac { R ^ { 2 } \operatorname* { m i n } ( \overline { { \kappa } } ^ { 1 } ( \overline { { \kappa } } ^ { 1 } ( \overline { { \kappa } } ^ { 1 } + \overline { { \kappa } } ^ { 1 } ) ) } { \kappa ^ { 1 0 } \kappa } ( \overline { { \kappa } } ) } \\ & { = } & { \mathrm { E } ^ { \mathcal { R } _ { n } } \frac { R ^ { 2 } \operatorname* { m i n } ( \overline { { \kappa } } ^ { 1 } ( \overline { { \kappa } } ^ { 1 } ) - \overline { { \kappa } } ^ { 1 } ) } { \kappa ^ { 1 0 } \kappa } \Big [ \big \| \gamma - \overline { { \kappa } } ^ { 1 } \big ( \overline { { \kappa } } ^ { 1 } \big ) - \underset { \zeta \in \mathcal { R } _ { n } } { \operatorname* { m a x } } \kappa ( \mathbf { h } ^ { - 1 } \zeta ^ { - } \zeta ^ { - } \mathbf { h } ^ { \zeta } ) \Big ] , \quad \quad \scriptstyle \scriptstyle \atop \scriptstyle \atop \scriptstyle \atop \displaystyle \prod ^ { \prime } \neq \mathcal { R } _ { n } \neq \kappa \operatorname* { m i n } \neq \infty } \mathrm { E r f u e n ~ a n e t e r s i o n ~ a t e v e } \zeta ,  \\ & { = } & { \mathrm { E } ^ { \mathcal { R } _ { n } \mathcal { L } \operatorname* { m i n } ( \mathbf { T } \cdot \zeta ) \omega - \kappa ( \overline { { \kappa } } ^ { 1 } ) } \Bigg [ \Big \| \gamma - \overline { { \kappa } } ^ { 1 } \mathbf { Z } \Big \| \Bigg ] ^ { \frac { \zeta } { \tau } } + \Bigg ( \frac { \overline { { \kappa } } ^ { 1 } } { \kappa } + 1 \Bigg ) \frac { \| \alpha \sqrt { \tau } \big | } { \kappa ^ { 1 0 } \tau } \Bigg [ \Big \| ( \overline { { \kappa } } ^ { 1 } - \overline { { \kappa } } ^ { 1 } ) \overline { { Z } } \Big | \Bigg ] , } \\ & { = } &  \mathrm { E } ^ { \mathcal { R } _ { n } } \Big [ \Big \| \gamma - \mathbf { h } ^ { \mathrm { T } } \mathbf { Z } \Big \| ^ { 2 } \Big ] + \frac { 1 } { \alpha } \frac { \mathbf { U } ^ { \alpha } } { \kappa } \Big [ \Big \| ( \mathbf { I } - \mathbf { h } ^ { \mathrm { T } } \mathbf { Z } ) \Big \| ^ { 2 } \Bigg ] , \end{array}
$$

# F.4 Proof of Theorem 2—Causal estimation with IVL regression

Theorem 2 (causal estimation with IVL regression). In SEM M of Example 1, for $\alpha < \infty$ , we have

$$
\begin{array} { r } { \mathrm { C E R } _ { \mathfrak { s p } } \Big ( \hat { \mathbf { h } } _ { I V L _ { \alpha } } ^ { \mathfrak { M } } \Big ) \leq \mathrm { C E R } _ { \mathfrak { s p } } \Big ( \hat { \mathbf { h } } _ { E R M } ^ { \mathfrak { m } } \Big ) , \qquad e q u a l i t y i f \qquad \mathbb { E } ^ { \mathfrak { M } } [ X \mid Z ] \ \bot _ { \mathrm { a . s . } } \mathbb { E } ^ { \mathfrak { M } } [ X \mid \xi ] . } \end{array}
$$

Proof. For $\hat { \mathbf { h } } _ { \mathrm { I V L } _ { \alpha } } ^ { \mathfrak { M } }$ , we have from Proposition 1

$$
\begin{array} { r } { \left\| \hat { \mathbf { h } } _ { \mathrm { I V L } _ { \alpha } } ^ { \mathfrak { M } } - \mathbf { f } \right\| _ { \mathbf { \Sigma } _ { X } ^ { \mathfrak { M } } } ^ { 2 } = \left\| \mathbb { E } \left[ X ^ { \prime } { X ^ { \prime } } ^ { \top } \right] ^ { - 1 } \mathbb { E } \left[ X ^ { \prime } { Y ^ { \prime } } ^ { \top } \right] - \mathbf { f } \right\| _ { \mathbf { \Sigma } _ { X } ^ { \mathfrak { M } } } ^ { 2 } . } \end{array}
$$

Note that we have

$$
\begin{array} { r l r } & {  \ Y ( x ^ { 5 } + x ^ { 6 } ) } \\ & { = \ Z [ x ^ { 5 } ( u ^ { 5 } + b \xi ( Y ^ { \top } ) Z ) ] ^ { \top } , } \\ & { = \ Z [ x ^ { 5 } ( u ^ { 5 } + b \xi ( Y ^ { \top } + \xi ) ) ^ { \top } ] , } \\ & { = \ Z [ x ^ { 7 } ( a ^ { 5 } + a \xi ( \xi ( X ^ { \top } + \xi ) ) Z ) ] ^ { \top } , } \\ & { = \ Z [ x ^ { 7 } ( a ^ { 5 } + a \xi ( \xi ( \xi ( X ^ { \top } ) ) Z ) ) ^ { \top } ] , } \\ & { - \ Z [ x ^ { 7 } ( a ^ { 7 } X ^ { \top } + a \xi + a ^ { \frac { 1 } { 2 } } \xi ( Y ^ { \top } ) X ) Z ] ^ { \top } ] , } \\ & { = \ Z [ x ^ { 7 } ( \xi ^ { 7 } X ^ { \top } + a \xi ( \xi ( X ^ { \top } ) ) ^ { \top } ) , } & { \mathrm { ( S u b s i a n i n g ~ i n ~ \xi = n \xi + b \xi ( \xi ( Z ^ { \top } ) ) ~ } } \\ & { = \Xi [ x ^ { 7 } X ^ { \top } ] ^ { \top } + a X ^ { \xi } \xi ^ { ( \xi ) } ] , } \\ & { = \ Z [ x ^ { 7 } X ^ { \top } ] ^ { \top } [ + a X ^ { \xi } \xi ^ { ( \xi ) } ] , } \\ & { = \Xi [ X ^ { \prime } X ^ { \top } ] ^ { \top } [ + a \xi [ X \xi ^ { \xi } ] ] , } \\ & { = \ Z [ x ^ { 7 } X ^ { \top } ] ^ { \top } [ + a \xi [ X \xi ^ { \top } ] , } &  \mathrm { ( 2  { \varDelta } \xi , n u c t e r g e : ~ | X \xi | ^ { \xi } ] = a \widetilde { \xi } [ X \xi ^ { \top } ] / \xi , } \\ & { = \xi [ X X ^ { \prime } ] ^ { \top } [ + a \xi [ X \xi ^ { \top } ] , } \end{array}
$$

We also see that

$$
\begin{array} { r l r } & { \Rightarrow \mathbb { E } \left[ X ^ { \prime } X ^ { \prime \prime } \right] } \\ & { = \mathbb { E } \left[ ( a X + b \mathbb { E } \left[ X \mid Z \right] ) ( a X + b \mathbb { E } \left[ X \mid Z \right] ) ^ { \top } \right] , } \\ & { = \mathbb { E } \left[ \bigg ( a X + b \tilde { Z } \bigg ) \bigg ( a X + b \tilde { Z } \bigg ) ^ { \top } \right] , } & { ( \mathrm { S e t ~  { \tilde { Z } } : = \mathbb { E } \left[ X \mid Z \right] \ f o r \ b r e v i t y . } ) } \\ & { = a ^ { 2 } \mathbb { E } \left[ X X ^ { \top } \right] + b ^ { 2 } \mathbb { E } \left[ \tilde { Z } \tilde { Z } ^ { \top } \right] + a b \mathbb { E } \left[ X \tilde { Z } ^ { \top } \right] + a b \mathbb { E } \left[ \tilde { Z } X ^ { \top } \right] , } \\ & { = a ^ { 2 } \mathbb { E } \left[ X X ^ { \top } \right] + \big ( b ^ { 2 } + 2 a b \big ) \Sigma _ { \tilde { Z } } , } & { ( \mathrm { B e c a u s e \  { \mathbb { E } \left[ X \tilde { Z } ^ { \top } \right] = \mathbf { \Sigma } _ { \tilde { Z } } . } ) } } \\ & { = \alpha \mathbb { E } \left[ X X ^ { \top } \right] + \mathbf { \Sigma } _ { \tilde { Z } } , } & { ( 2 6 ) } \end{array}
$$

where we substituted in Eq. (18) in Eq. (26).

Finally, we now have

$$
\begin{array} { r l } & { \Rightarrow \left. \hat { \mathbf { h } } _ { \mathbf { N L } _ { \alpha } } ^ { \mathfrak { M } } - \mathbf { f } \right. _ { \mathbf { C } _ { X } ^ { \overline { { \mathbf { m } } } } } ^ { 2 } } \\ & { = \left. \mathbb { E } \left[ X ^ { \prime } X ^ { \prime \prime } \right] ^ { - 1 } \mathbb { E } \left[ X ^ { \prime } Y ^ { \prime \prime } \right] - \mathbf { f } \right. _ { \mathbf { C } _ { X } ^ { \overline { { \mathbf { m } } } } } ^ { 2 } , } \\ & { = \left. \mathbb { E } \left[ X ^ { \prime } X ^ { \prime \top } \right] ^ { - 1 } \left( \mathbb { E } \left[ X ^ { \prime } X ^ { \prime \top } \right] \mathbf { f } + \alpha \mathbb { E } \left[ X \xi ^ { \top } \right] \right) - \mathbf { f } \right. _ { \mathbf { E } _ { X } ^ { \overline { { \mathbf { m } } } } } ^ { 2 } , } \\ & { = \left. \mathbf { f } + \alpha \mathbb { E } \left[ X ^ { \prime } X ^ { \prime \top } \right] ^ { - 1 } \mathbb { E } \left[ X \xi ^ { \top } \right] - \mathbf { f } \right. _ { \mathbf { Z } _ { X } ^ { \overline { { \mathbf { m } } } } } ^ { 2 } , } \end{array}
$$

(Substituting in Eq. (25).)

$$
\begin{array} { r l } & { - | z | z ^ { \star \star } x ^ { \star } | ^ { 2 } \frac { \mathsf { R } ^ { \star } } { \mathsf { R } ^ { 2 } } | z ^ { \star } | _ { \theta _ { 0 } } ^ { 2 } } \\ & { = | \varphi ( z ^ { \star } x ) \cdot z ^ { \star \star } ( z ) \cdot z ^ { \star \star } ( z ^ { \star } ) | _ { \theta _ { 0 } } ^ { 2 } , } \\ & { - | \varphi ( z ^ { \star } x ) \cdot z ^ { \star \star } ( z ^ { \star } ) | _ { \theta _ { 0 } } ^ { 2 } , } \\ & { - | \varphi ( z ^ { \star } x ) \cdot z ^ { \star \star } ( z ^ { \star } ) | _ { \theta _ { 0 } } ^ { 2 } , } \\ & { = | \varphi ( z ^ { \star } , x ) \cdot z ^ { \star \star } ( z ^ { \star } ) \cdot z ^ { \star \star } ( z ^ { \star } ) | _ { \theta _ { 0 } } ^ { 2 } , } \\ & { \frac { 1 } { \sqrt { | x ^ { \star } | ^ { 2 } } } | z ^ { \star } ( z ^ { \star } , x ^ { \star \star } ) | _ { \theta _ { 0 } } ^ { 2 } } \\ & { \le | \varphi ( z ^ { \star } x ) \cdot z ^ { \star \star } ( z ^ { \star } ) | _ { \theta _ { 0 } } ^ { 2 } , } \\ & { \le | \varphi ( z ^ { \star } x ) \cdot z ^ { \star \star } ( z ^ { \star } ) | _ { \theta _ { 0 } } ^ { 2 } } \\ & { - | \varphi ( z ^ { \star } x ) | _ { \theta _ { 0 } } ^ { 2 } , } \\ & { \le | \varphi ( z ^ { \star } x ) | _ { \theta _ { 0 } } ^ { 2 } } \\ & { \le | \varphi ( z ^ { \star } x ) | _ { \theta _ { 0 } } ^ { 2 } } \\ & { \le | \varphi ( z ^ { \star } x ) | _ { \theta _ { 0 } } ^ { 2 } } \\ & { \le | \varphi ( z ^ { \star } x ) | _ { \theta _ { 0 } } ^ { 2 } } \\ & { - | \varphi ( z ^ { \star } x ) | _ { \theta _ { 0 } } ^ { 2 } } \\ & { - | \varphi ( z ^ { \star } x ) | _ { \theta _ { 0 } } ^ { 2 } } \\ &  - | \varphi ( z ^ { \star } \end{array}
$$

(Substituting in Eq. (26).)

(Using Lemma 2.)

(S is invertible.)

(Switch to $\ell _ { 2 }$ norm.)

(Substituting ${ \bf I } = { \bf S } { \bf S } ^ { - 1 }$ .)

(Back to weighted norm.)

(Adding and subtracting f .)

$$
\operatorname { S u b s t i t u t e \ I } = \mathbb { E } \left[ X X ^ { \top } \right] ^ { - 1 } \mathbb { E } \left[ X X ^ { \top } \right] . ,
$$

where inequality Eq. (27) holds because $\mathbf { D }$ is non-negative diagonal. Furthermore, inequality Eq. (27) only holds with equality iff $\mathbf { S } ^ { - \top } \mathbb { E } \left[ X \xi ^ { \top } \right]$ is in the kernel of $\mathbf { D }$ . Or equivalently, iff $\mathbb { E } \left[ X \xi ^ { \top } \right]$ is in the kernel of $\mathbf { S } ^ { \top } \mathbf { D } \mathbf { S } = \pmb { \Sigma } _ { \tilde { Z } }$ , which from Lemma 1 is true iff

$$
\mathbb { E } ^ { \mathfrak { M } } [ X \mid Z ] \quad \perp \quad \mathbb { E } ^ { \mathfrak { M } } [ X \mid \xi ] \qquad \mathrm { a . s . }
$$

# F.5 Proof of Theorem 3—Causal estimation with DA+ERM

Theorem 3 (causal estimation with DA+ERM). For SEM A in Example 2, the following holds: $\begin{array} { r } { \mathrm { C E R } _ { \mathfrak { A } } \Big ( \hat { \mathbf { h } } _ { D A _ { G } + E R M } ^ { \mathfrak { A } } \Big ) \leq \mathrm { C E R } _ { \mathfrak { A } } \Big ( \hat { \mathbf { h } } _ { E R M } ^ { \mathfrak { A } } \Big ) , \qquad e q u a l i t y i f f \quad \mathbb { E } ^ { \mathfrak { A } } [ G X \ | \ G ] \ \bot _ { \mathrm { a . s . } } \mathbb { E } ^ { \mathfrak { A } } [ X \ | \ \xi ] . } \end{array}$ .

Proof. We have

$$
\begin{array} { r l r } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { = \left[ \left[ \frac { 1 } { \sigma } \left( \sigma { \hat { x } } \cdot \sigma \sigma \sigma \sigma \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \right) \right] \right] \Bigg | _ { \sigma ^ { 1 } } ^ { \sigma } \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times \quad \quad  \\ & & { = \left[ \sigma \cdot \sigma \sigma \sigma \sigma \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \sigma ^ { 1 } \right. } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

(Switch to $\ell _ { 2 }$ norm.)

$$
( \operatorname { U s e } \mathbf { I } _ { m } = \operatorname { \mathbb { E } } \left[ X X ^ { \top } \right] ^ { - 1 } \operatorname { \mathbb { E } } \left[ X X ^ { \top } \right] . )
$$

(ERM closed form solution.)

where inequality Eq. (28) holds because $\mathbf { D }$ is non-negative diagonal. Furthermore, inequality Eq. (28) only holds with equality iff $\mathbf { S } ^ { - \top } \mathbb { E } [ X \xi ^ { \top } ]$ is in the kernel of $\mathbf { D }$ . Or equivalently, iff $\mathbb { E } \left[ X \xi ^ { \top } \right]$ is in the kernel of $\mathbf { S } ^ { \top } \mathbf { D } \mathbf { S } = \Sigma _ { \tilde { G } }$ , which from Lemma 1 is true iff $\mathbb { E } ^ { { \mathfrak { A } } } [ G X \mid G ] ~ \bot ~ \mathbb { E } ^ { { \mathfrak { A } } } [ X \mid \xi ]$ a.s. □

# F.6 Miscellaneous supporting lemmas

Lemma 1 (Gaussian conditional orthogonality lemma). Let $X , Y , Z \in \mathbb { R } ^ { n }$ be zero-mean jointly Gaussian random vectors with covariance matrices $\Sigma _ { X } = \mathbb { E } [ X X ^ { \top } ]$ , $\Sigma _ { Z } = \mathbb { E } [ Z Z ^ { \top } ]$ , and crosscovariance $\Sigma _ { Y , Z } = \mathbb { E } [ Y Z ^ { \top } ]$ . Define the conditional expectation

$$
\operatorname { \mathbb { E } } [ Y \mid Z ] : = \left( \operatorname { \mathbb { E } } \left[ Z Z ^ { \top } \right] ^ { - 1 } \operatorname { \mathbb { E } } \left[ Z Y ^ { \top } \right] \right) ^ { \top } Z = \Sigma _ { Y , Z } \Sigma _ { Z } ^ { - 1 } Z .
$$

Then the following are equivalent:

$$
X \perp \mathbb { E } [ Y \mid Z ] = 0 \quad a . s . \qquad \Longleftrightarrow \qquad \Sigma _ { X } \Sigma _ { Y , Z } = \mathbf { 0 } .
$$

Proof. Since $X , Y , Z$ are jointly Gaussian, $\mathbb { E } [ Y \mid Z ] = \mathbf { M } Z$ with $\mathbf { M } : = \Sigma _ { Y , Z } \Sigma _ { Z } ^ { - 1 }$ . The scalar random variable

$$
S : = X ^ { \top } \mathbb { E } [ Y \mid Z ] = X ^ { \top } \mathbf { M } Z
$$

is Gaussian with mean zero. Hence,

$$
S = 0 \quad { \mathrm { a . s . } } \qquad \iff \qquad \operatorname { V a r } ( S ) = 0 .
$$

Compute the variance:

$$
\operatorname { V a r } ( S ) = \mathbb { E } \left[ S ^ { 2 } \right] = \mathbb { E } \left[ ( X ^ { \top } \mathbf { M } Z ) ^ { 2 } \right] = \mathbb { E } \left[ Z ^ { \top } \mathbf { M } ^ { \top } X X ^ { \top } \mathbf { M } Z \right] .
$$

Using independence and zero-mean assumptions,

$$
\operatorname { V a r } ( S ) = \operatorname { t r } \big ( \mathbf { M } ^ { \top } \Sigma _ { X } \mathbf { M } \Sigma _ { Z } \big ) .
$$

Since covariance matrices are positive semidefinite, $\mathrm { V a r } ( S ) = 0$ iff

$$
\Sigma _ { X } ^ { 1 / 2 } { \bf M } \Sigma _ { Z } ^ { 1 / 2 } = { \bf 0 } \implies \Sigma _ { X } { \bf M } \Sigma _ { Z } = { \bf 0 } .
$$

Substituting $\mathbf { M } = \pmb { \Sigma } _ { Y , Z } \pmb { \Sigma } _ { Z } ^ { - 1 }$ gives

$$
\pmb { \Sigma } _ { X } \pmb { \Sigma } _ { Y , Z } = \mathbf { 0 } ,
$$

completing the proof.

Lemma 2 (SPD and PSD simultaneous denationalization via congruence). For any $n \times n$ matrices $\mathbf { A } \succ \mathbf { 0 } , \mathbf { B } \succeq \mathbf { 0 }$ , there exists an invertible $\mathbf { S } \in \mathbb { R } ^ { n \times n }$ and non-negative diagonal $\mathbf { D } \in \mathbb { R } ^ { n \times n }$ such that

$$
\begin{array} { r } { \mathbf { A } = \mathbf { S } ^ { \top } \mathbf { S } , \qquad \mathbf { B } = \mathbf { S } ^ { \top } \mathbf { D } \mathbf { S } . } \end{array}
$$

Proof. This is similar to Theorem 7.6.4 in [79, p. 465] for two SPD matrices. We proceed similarly; Since A is SPD, it admits a unique SPD square root ${ \bf A } ^ { 1 / 2 }$ . Define

$$
\mathbf { C } : = \mathbf { A } ^ { - 1 / 2 } \mathbf { B } \mathbf { A } ^ { - 1 / 2 } ,
$$

which is SPD. By the spectral theorem, there exists an orthogonal matrix $\mathbf { U }$ such that

$$
\mathbf { C } = \mathbf { U } ^ { \top } \mathbf { D } \mathbf { U } ,
$$

where $\mathbf { D }$ is diagonal with non-negative entries (the eigenvalues of $\mathbf { C }$ ). Set

$$
\mathbf { S } : = \mathbf { U } \mathbf { A } ^ { 1 / 2 } .
$$

Then

$$
\mathbf { S } ^ { \top } \mathbf { S } = \mathbf { A } ^ { 1 / 2 } \mathbf { U } ^ { \top } \mathbf { U } \mathbf { A } ^ { 1 / 2 } = \mathbf { A } ^ { 1 / 2 } \mathbf { I } \mathbf { A } ^ { 1 / 2 } = \mathbf { A } ,
$$

and

$$
\mathbf { S } ^ { \top } \mathbf { D } \mathbf { S } = \mathbf { A } ^ { 1 / 2 } \mathbf { U } ^ { \top } \mathbf { D } \mathbf { U } \mathbf { A } ^ { 1 / 2 } = \mathbf { A } ^ { 1 / 2 } \mathbf { C } \mathbf { A } ^ { 1 / 2 } = \mathbf { B } .
$$

Since ${ \bf A } ^ { 1 / 2 }$ and $\mathbf { U }$ are invertible, S is invertible, completing the proof.

Lemma 3 (solvability of simultaneous SEM). The SEM M in Example 1 is solvable iff $\mathbf { f } ^ { \intercal } \tau ^ { \intercal } \neq 1$ , in which case the following solution defines the reduced form of the SEM.

$$
\begin{array} { r } { \left[ \boldsymbol { X } \right] = \left[ \begin{array} { c c } { \mathbf { I } _ { m } } & { - \boldsymbol { \tau } ^ { \top } } \\ { - \mathbf { f } ^ { \top } } & { 1 } \end{array} \right] ^ { - 1 } \left( \left[ \begin{array} { c } { \boldsymbol { \Gamma } ^ { \top } } \\ { \mathbf { 0 } _ { 1 \times k } } \end{array} \right] \boldsymbol { Z } + \left[ \begin{array} { c } { \boldsymbol { \mathbf { T } } ^ { \top } } \\ { \boldsymbol { \epsilon } ^ { \top } } \end{array} \right] \boldsymbol { C } + \boldsymbol { \sigma } \cdot \left[ \begin{array} { c } { \boldsymbol { N } _ { X } } \\ { \boldsymbol { N } _ { Y } } \end{array} \right] \right) , } \end{array}
$$

Similarly, SEM A in Example 2 solves for f $^ { \cdot \top } \tau ^ { \top } \neq \kappa ^ { - 1 }$ .

Proof. We re-state the SEM $\mathfrak { M }$ in the following block form

$$
\begin{array} { r } { [ X ] = [ \mathbf { 0 } _ { m \times m } \quad \tau ^ { \top } ] [ X ] + [ \mathbf { 0 } _ { 1 \times k } ^ { \top } ] Z + [ \mathbf { T } ^ { \top } ] C + \sigma \cdot [ N _ { X } ] , } \\ { \Rightarrow [ \mathbf { I } _ { m } \quad - \tau ^ { \top } ] \cdot [ X ] = [ \mathbf { I } _ { 1 \times k } ^ { \top } ] Z + [ \mathbf { T } ^ { \top } ] C + \sigma \cdot [ N _ { X } ] } \\ { \quad - \mathbf { f } ^ { \top } \quad \mathbf { \Omega } _ { 1 } ^ { \top } \quad \mathbf { I } ] . } \end{array}
$$

solving for $( X , Y )$ involves inverting the block matrix on the LHS. The result immediately follows from Proposition 2.8.7 in [80, p. 108], via the Schur complement formula for block matrix inversion.

# NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: Yes, the main claims as stated in the abstract are explicitly enumerated in Sec. 1, each referencing the section of the paper that contains the respective contribution.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the paper.   
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.   
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations of our method in Secs. 4 and 7. We also explicitly state assumptions made for theoretical results in Secs. 3 and 4.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.   
• The authors are encouraged to create a separate "Limitations" section in their paper.   
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.   
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.   
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We provide the full set of assumptions and correct proofs for each theoretical result. Observations, assumptions, examples, lemmas, theorems, all are appropriately referenced.

Guidelines:

• The answer NA means that the paper does not include theoretical results.   
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.   
• All assumptions should be clearly stated or referenced in the statement of any theorems.   
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.   
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.   
• Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide the details of our experimental settings, including algorithm used and its implementation details (hyper-parameters, network architecture, etc.) in Appendix E. All baseline methods are referenced appropriately in Sec. 6, and their parameterization (hyper-parameters, network architecture, etc.) is discussed Appendix E. We also provide code for our experiments for reproducibility.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.   
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.

In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Yes, we provide the code including a README.md file with necessary instructions on how to run and reproduce the experiments.

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

Justification: Yes, we provide sufficient details for our experimental settings in Sec. 6 for the readers to understand the results and additional details in Appendix E as well.

Guidelines:

• The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We provide a $9 5 \%$ confidence interval (CI) for our stand-alone simulation experiments in Sec. 6.1 and Appendix E.1, inter-quartile ranges (IQR) in comparative analysis with other DG and causal regularization baselines in Fig. 5 and standard error (SE) with additional optical-device experiments in Appendix E.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).   
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)   
• The assumptions made should be given (e.g., Normally distributed errors).   
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.   
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a $96 \%$ CI, if the hypothesis of Normality of errors is not verified. For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).   
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We briefly mention the hardware used to generate experimental results in the README.md file with the supplemental code. However, the results should be hardwareagnostic.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.   
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

# 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The study involved no human subjects and all data sources used are publicly available.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.   
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.   
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper discusses data-augmentation, which is a fairly ubiquitous and largely application agnostic technique.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.   
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.   
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks

Guidelines:

• The answer NA means that the paper poses no such risks.   
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cite and credit all assets used, including baseline models and the datasets used.

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

Justification: The code to reproduce our experimental results is publicly released at https://github.com/uzairakbar/causal-data-augmentation, along with appropriate documentation.

Guidelines:

• The answer NA means that the paper does not release new assets.   
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
• The paper should discuss whether and how consent was obtained from people whose asset is used.   
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No crowd-sourcing or human subjects were used.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No human subjects were used.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.   
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

# 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.