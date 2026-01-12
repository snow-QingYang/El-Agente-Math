## LINE 106

common approach to tackle the linear bandit problem is to follow an upper confidence bound (UCB) strategy. This involves the following protocol. At each round $t$ , we first derive a confidence set $\mathcal { C } _ { t - 1 }$ , based on the arm-reward pairs $( X _ { s } , Y _ { s } ) _ { s \leq t - 1 }$ . This is a random set (as it depends on the past noise realisations), which must be constructed ensuring that $\theta ^ { \star } \in \mathcal { C } _ { t - 1 }$ with high probability. More precisely, the regret can be effectively controlled if one can ensure that $\theta ^ { \star }$ uniformly belongs to every set $( \mathcal { C } _ { t } ) _ { t \geq 1 } ^ { }$ , with high probability (a property often referred to as anytime validity). Then, for every available arm $x$ , we let

$$
\operatorname { U C B } _ { \mathcal { C } _ { t - 1 } } ( x ) = \operatorname* { m a x } _ { \theta \in \mathcal { C } _ { t - 1 } } \left. x , \theta \right. .
$$

By definition, this is a high-probability upper bound on 107 $\langle x , \theta ^ { \star } \rangle$ , which justifies the name “upper confidence bound
