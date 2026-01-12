## LINE 117-118

need to minimize the cost throughout the trajectory $x _ { t }$ , $t \in [ 0 , 1 ]$ with the   
following objective

$$
g ^ { c } ( x ) = \operatorname* { i n f } _ { v ( x , t ) } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \left( \frac { \| v ( t , x _ { t } ) \| ^ { 2 } } { 2 } + U ( x _ { t } ) \right) \mathrm { d } t - g ( x _ { 1 } ) \bigg | x _ { 0 } = x \right] .
$$

In the last expression, we have united infimums by $\mu ( x )$ and control $v ( t , x )$ and as a sequence have   
removed the right side condition $x _ { 1 } \sim \mu ( x )$ . Based on dynamic programming approach, define the   
value function. For any $0 \leq t \leq 1$ , the value function satisfies:

$$
s ( t , x ) = \operatorname* { i n f } _ { x _ { t } } \mathbb { E } \left[ \int _ { t } ^ { 1 } \left( \frac { \| v ( t , x _ { t } ) \| ^ { 2 } } { 2 } + U ( x _ { t } ) \right) \mathrm { d } t - g ( x _ { 1 } ) \bigg | x _ { t } = x \right] ,
$$

such that our objective equals $s ( 0 , x )$ and the boundary condition at time point $t = 1$ is

$$
\forall x \in \mathbb { R } ^ { d } : s ( 1 , x ) = - g ( x ) .
$$

Function $s ( t , x )$ solves the Hamilton-Jacobi-Bellman
