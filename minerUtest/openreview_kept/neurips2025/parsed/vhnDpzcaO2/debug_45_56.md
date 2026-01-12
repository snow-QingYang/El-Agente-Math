Example 1 (Inventory Model). Consider a classical inventory control problem where the inventory   
evolves according to $S _ { t + 1 } = S _ { t } + A _ { t } - D _ { t }$ , with $\{ D _ { t } : t \ge 0 \}$ representing the stochastic   
demand process and $a _ { t }$ denoting the replenishment decision at time $t$ . The reward function is   
$R ( S _ { t } , A _ { t } , S _ { t + 1 } ) = p ( S _ { t } - S _ { t + 1 } + A _ { t } ) + b \operatorname* { m i n } ( S _ { t + 1 } , 0 ) - h \operatorname* { m a x } ( S _ { t + 1 } , 0 ) - c A _ { t }$ , where $p$ is the   
sales price, $c$ is the purchase cost, $h$ is the holding cost, and $b$ is the penalty of backlog. To address   
the uncertainty in demand, distributionally robust reinforcement learning (DR-RL) provides a natural   
framework for enhancing robustness. In this context, it is reasonable to assume that the adversary can   
only modify the distribution of the demand $D _ { t }$ independently of the controller’s action $A _ { t }$ , leading to   
an S-rectangular uncertainty set. By contrast, the SA-rectangular formulation allows the adversary   
to choose different distributions for $D _ { t }$ based on the controller’s action $A _ { t }$ —for example, assigning   
low demand when $A _ { t }$ is large and high demand when $A _ { t }$ is small—granting the adversary excessive   
power and resulting in an unrealistic model