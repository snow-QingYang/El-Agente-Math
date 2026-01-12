## LINE 227-228

Definition 5. Define the minimum support probability as

$$
\mathfrak { p } _ { \wedge } : = \operatorname* { m i n } _ { s , a \in S \times A } \operatorname* { m i n } _ { s ^ { \prime } \in S : \overline { { P } } _ { s , a } ( s ^ { \prime } ) > 0 } \overline { { P } } _ { s , a } ( \mathfrak { s } ^ { \prime } )
$$

As noted in the literature, the use of ${ \mathfrak { p } } _ { \wedge }$ as a complexity metric is well justified. In the KL case,   
the convergence rate of the estimation error can degrade arbitrarily, depending on the specific MDP   
instance, if there is no lower bound on the minimum support probability. In particular, the rate can   
be as slow as $\Omega ( n ^ { - 1 / \beta } )$ for any $\beta \geq 2$ as the sample size $n$ tends to infinity [19]. Similar negative   
results hold in the $f _ { k }$ -divergence setting when the parameter $k$ approaches 1 [3], highlighting the   
233 necessity of such a complexity measure.

# 4.1 The Kullback-Leibler Divergence Uncertainty Set
