## LINE 204-205

expected (true) loss and its empirical counterpart:

$$
\mathcal { L } ( \theta ) = \left\{ \begin{array} { l l } { - \underset { \xi \sim \mathcal { M } ^ { ( T ) } } { \mathbb { E } } [ G ( \xi ) ] } \\ { = \underset { \mathfrak { D } \sim \mathcal { M } ^ { ( T ) } } { \mathbb { E } } [ \hat { \mathcal { L } } _ { \mathfrak { D } } ( \theta ) ] } \end{array} \right. \mathrm { w h e r e } \quad \hat { \mathcal { L } } _ { \mathfrak { D } } ( \theta ) = - \frac { 1 } { T } \sum _ { j = 1 } ^ { T } G ( \xi ^ { ( j ) } )
$$

Prior and posterior over policies. Following the PAC-Bayesian paradigm we endow $\Theta$ with   
a prior distribution $\mu \in { \mathcal { P } } ( \Theta )$ , selected independently of the data, and a posterior distribution   
$\rho \in { \mathcal { P } } ( \Theta )$ , chosen after observing the sample $\mathfrak { D }$ . This PAC-Bayesian formalism allows us to reason   
about the generalization properties of randomized policies drawn from $\rho$ , with theoretical guarantees   
197 based on their divergence from the prior $\mu$ .   
98 A bounded-differences property for the empirical loss. The following lemma shows that chang  
ing one transition in the data results in quantitative bounded effect of the empirical loss defined in   
(5).   
Lemma 3.1 (Bounded differences) Let $\mathfrak { D }$ be a set of trajectories and $\theta \in \Theta$ be fixed policy param  
eters. Suppose we form $\bar { \mathfrak { D } }$ by changing one transition, say the transition at time step $h \in [ H ]$ of   
trajectory $j \in [ T ]$ , where $\xi _ { h } ^ { ( j ) } = ( s , a , r , s ^ { \prime } )$ is replaced with $\bar { \xi } _ { h } ^ { ( j ) } = ( \bar { s } , \bar { a } , \bar { r } , \bar { s } ^ { \prime } )$ . Then, there exists   
$c \in \mathrm { I R } _ { + } ^ { H \times T }$ such that

$$
\left| \hat { \mathcal { L } } _ { \mathfrak { D } } ( \theta ) - \hat { \mathcal { L } } _ { \bar { \mathfrak { D } } } ( \theta ) \right| \ \leq \ \sum _ { h ^ { \prime } = 1 } ^ { H } \sum _ { j ^ { \prime } = 1 } ^ { T } c _ { ( h ^ { \prime } , j ^ { \prime } ) } \mathbb { I } \Big [ \xi _ { h ^ { \prime } } ^ { ( j ^ { \prime } ) } = \bar { \xi } _ { h ^ { \prime } } ^ { ( j ^ { \prime } ) } \Big ]
$$

Intuitively, $c _ { ( h , j ) }$ quantifies the transition-level influence
