# Agentic Reader Result
**Paper ID:** o03aOm4Vci
**Issue File:** o03aOm4Vci_0.md
**Status:** success
**Timestamp:** 2026-01-19T07:29:15.614540
**Model:** openai:gpt-5-mini
**Max Iterations:** 10

**Issue Content:**
## LINE 84-85

whose marginal distribution is $p ( x _ { t } | x _ { 0 } )$ .   
This is accomplished using the Markovian projection.   
Proposition 1 (Markovian projection SDE [Shi et al., 2024]). Let $p ( x _ { 1 } | x _ { 0 } )$ be a conditional distribu  
tion over target variables given source variables and let $p ( x _ { t } | x _ { 0 } , x _ { 1 } )$ denote the distribution of the   
base SDE $d x _ { t } = b _ { t } ( x _ { t } ) d t + L _ { t } d W _ { t }$ when conditioned to start at $x _ { 0 }$ and end at $x _ { 1 }$ . The “Markovian   
projection $S D E '$ is an $S D E$ whose marginal distribution, denoted by $q ^ { * } ( x _ { t } | x _ { 0 } )$ is equal to $p ( x _ { t } | x _ { 0 } )$   
It is given by:

$$
\begin{array} { r } { d \boldsymbol { x } _ { t } = \left( b _ { t } ( \boldsymbol { x } _ { t } ) + L _ { t } L _ { t } ^ { T } \mathbb { E } _ { p ( x _ { 1 } | x _ { 0 } , x _ { t } ) } \left[ \nabla \log p ( x _ { 1 } | x _ { 0 } , x _ { t } ) \right] \right) d t + L _ { t } d \boldsymbol { W } _ { t } } \end{array}
$$

See Prop 3. of [De Bortoli et al., 2023] for a proof. Proposition 1 is a solution to the paired generative   
modeling problem because $q ^ { * } ( x _ { t = 1 } | x _ { 0 } ) = p ( x _ { 1 } | x _ { 0 } ) : = p ( y _ { 1 } | y _ { 0 } )$ . Given a sample from the source   
distribution, $x _ { 0 } \sim p ( x _ { 0 } )$ , we can simulate the SDE from $t = 0$ to $t = 1$ to generate a sample from the   
target distribution. However, this SDE contains an intractable drift term that depends on the posterior   
distribution of $x _ { 1 }$ given $x _ { 0 }$ and $x _ { t }$ . This is addressed using a matching learning objective. For   
example, in score matching, [Vincent, 2011, Song et al., 2021], one writes the drift in the following   
variational form:

$$
\nabla \log q ^ { * } ( x _ { t } | x _ { 0 } ) = \underset { s _ { t } ( x _ { t } , x _ { 0 } ) } { \mathrm { a r g m i n } } \mathbb { E } _ { p ( x _ { 0 } , x _ { 1 } , x _ { t } ) } \left[ \left\| L _ { t } L _ { t } ^ { T } \nabla \log p ( x _ { 1 } | x _ { 0 } , x _ { t } ) - s _ { t } ( x _ { t } , x _ { 0 } ) \right\| ^ { 2 } \right]
$$

If $s ( x _ { t } , x _ { 0 } ; \theta )$ is parameterized by a neural network, then one can minimize this expectation using   
the standard machine learning toolkit to find the Markovian projection SDE. However, obtaining a   
Monte Carlo estimate of the expectation for stochastic gradient descent requires being able to sample   
from $p ( x _ { 0 } , x _ { 1 } , x _ { t } )$ , which requires simulation of the base SDE. As such, the base SDE is chosen so   
that this distribution is tractable. After training is complete, then the flow-based generative model is   
given by the SDE $d x _ { t } = \bigl ( b _ { t } \bigl ( x _ { t } \bigr ) + L _ { t } L _ { t } ^ { T } s _ { t } \bigl ( \overset { \sim } { x } _ { t } , x _ { 0 } \bigr ) \bigr ) \dot { d t } + L _ { t } d W _ { t }$ . In general, matching algorithms,   
such as score matching, drift matching and bridge matching, are algorithms for learning the Bayes   
estimator of a random variable because of the well known relationship between posterior expectations   
and mean squared error [Jaynes, 2003]:   
Proposition 2 (Bayes estimate of parameter). Let $p ( z , \theta )$ be a joint distribution and let $\theta ^ { * } ( z )$ be   
the Bayes estimate of $\theta$ based on $z$ under the squared error risk. Then the Bayes estimate takes the   
following two forms:

$$
\theta ^ { * } ( z ) = \mathbb { E } _ { p ( \theta | z ) } [ \theta ] = \underset { f ( z ) } { \mathrm { a r g m i n } } ~ \mathbb { E } _ { p ( z , \theta ) } \left[ \| f ( z ) - \theta \| ^ { 2 } \right]
$$

See Appendix C.3 for a derivation. In score matching, one would have $\boldsymbol { z } ~ = ~ ( x _ { 0 } , x _ { t } )$ and $\theta =$   
$\nabla \log p ( x _ { 1 } | x _ { 0 } , x _ { t } )$ , while other matching approaches, such as flow matching [Albergo and Vanden  
Eijnden, 2023, Lipman et al., 2023, Liu et al., 2023] and bridge matching


## Agentic Reader Analysis

Yes — this snippet contains a mathematical/formula inconsistency.

What the paper states
- Proposition 1 (displayed SDE drift):
  d x_t = ( b_t(x_t) + L_t L_t^T E_{p(x_1 | x_0, x_t)}[ ∇ log p(x_1 | x_0, x_t) ] ) dt + L_t dW_t
  (Proposition 1 — see the displayed SDE; around the Proposition 1 text.)

- Immediately after, the paper gives a variational characterization (score-matching style):
  ∇ log q^*(x_t | x_0) = argmin_{s_t(x_t,x_0)} E_{p(x_0,x_1,x_t)} [ ‖ L_t L_t^T ∇ log p(x_1 | x_0, x_t) − s_t(x_t,x_0) ‖^2 ].
  (the displayed variational formula following Proposition 1)

Why this is inconsistent
- From Proposition 1 and the usual relationship between SDE drift and marginal score, one has
  a_t ∇ log q^*(x_t|x_0) = a_t E_{p(x_1|x_0,x_t)}[ ∇ log p(x_1|x_0,x_t) ],
  where a_t := L_t L_t^T. Hence ∇ log q^*(x_t|x_0) = E_{p(x_1|x_0,x_t)}[ ∇ log p(·) ] (after cancelling a_t), not a_t times that expectation.
- The stated variational problem on the RHS has its minimizer
  s_t^*(x_t,x_0) = E[ a_t ∇ log p(x_1|x_0,x_t) | x_t,x_0 ] = a_t E[ ∇ log p | x_t,x_0 ] = a_t ∇ log q^*(x_t|x_0).
  So the argmin returns a_t ∇ log q^*, not ∇ log q^*. Equating the argmin to ∇ log q^* (as the paper does) is therefore incorrect unless a_t = I (i.e. L_t L_t^T = identity).

Corrected statement(s)
- Either state the argmin identity for a_t ∇ log q^*:
  a_t ∇ log q^*(x_t|x_0) = argmin_{s_t} E[ ‖ a_t ∇ log p − s_t ‖^2 ],
  or, equivalently (if a_t is invertible),
  ∇ log q^*(x_t|x_0) = a_t^{-1} argmin_{s_t} E[ ‖ a_t ∇ log p − s_t ‖^2 ].
- Or remove the a_t factor inside the norm and write a suitably weighted (Mahalanobis) norm so the minimizer equals ∇ log q^*.

References to the paper locations
- Proposition 1 (displayed SDE with L_t L_t^T expectation): see the Proposition 1 display (around the Proposition 1 text — the SDE drift equation).
- Variational formula (the argmin equality equating ∇ log q^* to the argmin): see the displayed variational equation immediately following the discussion of intractable drift / score matching.

Recommendation
- Fix the displayed variational equality as noted above (replace L_t L_t^T ∇ log q^* with ∇ log q^*, or explicitly show the argmin gives a_t ∇ log q^* and then invert a_t if that is intended). Also add the assumption (if used) that a_t = L_t L_t^T is invertible (or equal to identity) when replacing s_t by ∇ log q^*.