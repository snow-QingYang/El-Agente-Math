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
