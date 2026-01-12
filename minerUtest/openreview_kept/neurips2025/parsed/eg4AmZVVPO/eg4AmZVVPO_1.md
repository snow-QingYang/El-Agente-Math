## LINE 128-129

Algorithm 1 CDDS Training

<table><tr><td>Input:Model parameters θ,a pre-trained control u learning rate n</td></tr><tr><td>0&#x27;个θ repeat</td></tr><tr><td>Sample Xo ~ Pprior and n~U{1,N -1}</td></tr><tr><td> Simulate PF ODE of (2) to get Xtn and Xtn+1</td></tr><tr><td>L(0,0&#x27;)←|1fθr(Xtn+1,tn+1),fe(Xtn,tn)l2</td></tr><tr><td>0←0-n∀θL(0,0&#x27;；u)</td></tr><tr><td>0&#x27;← stopgrad(θ)</td></tr><tr><td></td></tr><tr><td>until convergence</td></tr></table>

<table><tr><td>Algorithm2SCDSTraining Input:Model parameters 0,weights</td></tr><tr><td>入s,入sc 0&#x27;←θ repeat Sample Xo ~ Pprior,d,and t Simulate 2 to get Xo:T Compute target Xt+2d from 目</td></tr><tr><td>Compute shortcut Xt+2d from ( 回 Compute sampling loss Ls via Compute consistency loss Lsc via 四</td></tr><tr><td>0 ←Vθ(λsLs+λscLsc)</td></tr><tr><td>0&#x27;← stopgrad0 until convergence</td></tr></table>

training methods $\boxed { \boxplus 6 }$ to learn the function $f$ . However, this approach is expensive as it necessitates   
pre-collecting and storing a large dataset. Moreover, the accumulation of numerical errors arises from   
the numerical solver , resulting in significant global error.   
To solve the problem, we propose to leverage intermediate states of the pretrained model during each   
training iteration. Using these multiple and short intervals among intermediate helps keep the overall   
global error small.   
During model training, we minimize the discrepancy between the outputs of the consistency function   
at the consecutive intermediate states of the probability flow ODE $\lVert \rVert \bigotimes \rVert$ associated with $\textcircled{2}$ :

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C D } } ( \theta , \theta ^ { \prime } ; u ) ( X ) : = \mathbb { E } \Big [ \| f _ { \theta ^ { \prime } } ( \hat { X } _ { t _ { n + 1 } } , t _ { n + 1 } ) , f _ { \theta } ( \hat { X } _ { t _ { n } } , t _ { n } ) \| _ { 2 } \Big ] , } \end{array}
$$

where the expectation is over discrete time indices $n$ and $\theta ^ { \prime } = \mathrm { s t o p g r a d } ( \theta )$ indicates gradient   
stopping on the target term. Notably, unlike standard consistency generative models, the states $\hat { X } _ { t _ { n + 1 } }$   
and $\hat { X } _ { t _ { n } }$ are obtained from partial integrations of the probability flow ODE rather than from real   
data samples. Consequently, training CDDS incurs computational costs similar to training traditional   
diffusion samplers while substantially accelerating inference. The training procedure is summarized   
in Algorithm 1.   
If the loss $\textcircled { 7 }$ is driven to zero, the learned consistency function recovers the true mapping of the   
probability flow ODE, implying that CDDS can achieve arbitrarily accurate single-step sampling in   
the limit of sufficiently small integration steps. We formally state this in Theorem 1.   
Theorem 1. Let $f _ { \theta } ( X _ { t } , t )$ be a consistency function parameterized by $\theta$ , and let $f ( X _ { t } , t ; u )$ denote   
the consistency function of the PF ODE defined by the control u. Assume that $f _ { \theta }$ is $L - .$ Lipschitz   
continuous. Additionally, assume that for each step $\mathsf { i } \in \{ 1 , 2 , \ldots , N - 1 \}$ , the ODE solver called at   
$t _ { n }$ has a local error bounded by $O ( ( t _ { n + 1 } - t _ { n } ) ^ { p + 1 } )$ for some $p \geq 1 .$ . If $\mathcal { L } _ { C D } ( \theta , \theta ^ { \prime } ; u ) = 0$ , then:

$$
\operatorname* { s u p } _ { n , X _ { t _ { n } } } \| f _ { \theta } ( X _ { t _ { n } } , t _ { n } ) - f ( X _ { t _ { n } } , t _ { n } ; u ) \| _ { 2 } = O ( ( \Delta t ) ^ { p } ) ,
$$

where $\begin{array} { r } { \Delta t : = \operatorname* { m a x } _ { n \in \{ 1 , 2 , \ldots , N - 1 \} } | t _ { n + 1 } - t _ { n } | . } \end{array}$ .

A complete proof is provided in the Appendix. This theoretical
