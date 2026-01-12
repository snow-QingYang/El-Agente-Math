## LINE 139-140

action value function for estimating the reward   
function $\mathcal { R } _ { s , a }$

$$
\begin{array} { r l } & { Q ^ { \pi } ( s , a ) =  { \mathbb E } [ R _ { t + 1 } | S _ { t } = s , A _ { t } = a ] } \\ & { \qquad =  { \mathbb E } [ r _ { t + 1 } + \gamma Q ^ { \pi } ( S _ { t + 1 } = s ^ { \prime } , A _ { t + 1 } = \pi ( s _ { t + 1 } ) ) | S _ { t } = s , A _ { t } = a ] } \end{array} .
$$

The estimation forms a Bellman equation, which can be solved by temporal difference (TD) [27, 30]   
methods. TD methods approximate the expected return by gradually lowering down the TD error, i.e.,   
the difference of returns between the state-action value $Q ( s , a )$ and the TD-target $r _ { t + 1 } + \gamma V ( s _ { t + 1 } )$ ,   
where $V ( s _ { t + 1 } )$ is the state-value function satisfying $V ( s _ { t + 1 } ) = Q ( s _ { t + 1 } , \pi ( s _ { t + 1 } ) )$ .   
Establishing approximation function to form a mapping from state space to action space $\pi _ { \phi } : S  A$   
and a mapping from state space and action space to a real-value $Q _ { \theta } : \mathcal { S } \times \mathcal { A }  \mathbb { R }$ by deep neural   
network forms deep reinforcement learning. According to [9], the update of critic then can be made   
by minimizing the critic loss function:

$$
J _ { Q } ( \theta ) = \mathbb { E } _ { ( s , a , r , s ^ { \prime } ) \sim B } [ ( Q _ { \theta } ( s , a ) - r + \gamma V _ { \phi } ( s ^ { \prime } ; \theta ^ { \prime } ) ) ^ { 2 } ] ,
$$

subject to

$$
V _ { \phi } ( s ^ { \prime } ; \theta ^ { \prime } ) = Q _ { \theta ^ { \prime } } ( s ^ { \prime } , \pi _ { \phi } ( s ^ { \prime } ) + \epsilon ) ,
$$

where $\theta ^ { \prime }$ is the parameters of critic target with soft update, satisfying $\theta ^ { \prime }  \tau \theta + ( 1 - \tau ) \theta ^ { \prime }$ , and $\epsilon$   
is the policy noise similar to the technique adopted in SARSA learning [28]. The soft update is for   
stabilizing the learning of critic network using a fixed target. Then, the update
