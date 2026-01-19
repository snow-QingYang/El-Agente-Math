# Agentic Reader Result
**Paper ID:** eg4AmZVVPO
**Issue File:** eg4AmZVVPO_1.md
**Status:** success
**Timestamp:** 2026-01-19T07:13:52.511568
**Model:** openai:gpt-5-mini
**Max Iterations:** 10

**Issue Content:**
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


## Agentic Reader Analysis

Yes — this reviewer snippet points to a math/formula typo in the paper.

Issue summary
- The paper's consistency-loss formula is written with a comma between the two terms inside the norm, which is incorrect/ambiguous. The displayed equation reads (paper's text):
  E [ || f_{θ'}(ĤX_{t_{n+1}}, t_{n+1}), f_θ(ĤX_{t_n}, t_n) ||_2 ] 
  (see the displayed L_CD equation in the paper).
- The same corrupted form appears in Algorithm 1 (the algorithm text shows the loss as something like "L(θ,θ') ← |1 fθ'(X_{t_{n+1}},t_{n+1}), f_θ(X_{t_n},t_n) l2" — garbled but clearly using a comma rather than a difference).

Why this is a formula issue
- A consistency loss should measure the discrepancy between the two function outputs, i.e. the norm of their difference. Writing a comma inside the norm is ambiguous (it suggests a 2‑vector concatenation or is simply a typographical error) and does not express the intended distance. If taken literally, ||(a,b)||2 is the norm of the 2‑component vector [a,b], which is not the intended discrepancy and would not vanish when the two terms are equal unless both are zero.
- This ambiguity undermines the definition of \mathcal{L}_{CD} and the subsequent claim “if \mathcal{L}_{CD}=0 then …” (Theorem 1), because that claim relies on \mathcal{L}_{CD} being the norm of a difference.

Where this appears in the paper (citations to locations read)
- Algorithm 1 (garbled loss line): around document positions ~14180–14550 (the "Algorithm 1 CDDS Training" block).
- Displayed definition of \mathcal{L}_{CD} (with the comma error): around positions ~17080–17990 (the displayed equation defining \mathcal{L}_{CD}).

Suggested correction
- Replace the comma with a minus (and clarify whether the loss uses the norm or squared norm). The intended formula is almost certainly:
  \mathcal{L}_{CD}(\theta,\theta';u)(X) := \mathbb{E}\big[ \| f_{\theta'}(\hat X_{t_{n+1}},t_{n+1}) - f_\theta(\hat X_{t_n},t_n)\|_2 \big]
  (or \mathbb{E}[ \| \cdot \|_2^2 ] if the squared norm is intended).

Other minor/formattings
- There are other formatting artifacts (e.g. "L - . Lipschitz" and garbled algorithm text), but the key mathematical issue is the comma/absence of a difference in the loss definition.

Conclusion
- Yes — this is a mathematical/formula issue: the consistency-loss equation (and the same line in Algorithm 1) is written incorrectly/ambiguously (uses a comma instead of the intended difference).