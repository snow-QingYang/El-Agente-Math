## LINE 224

one ran  
domly chosen arm gives constant reward 1 while all other arms give reward 0. Note that this instance   
in $\mathcal { M } _ { 0 }$ can be equivalently represented as a $d _ { 0 } = d / L$ -dimensional linear bandit where actions are   
one-hot vectors $\mathbf { e } _ { i }$ .

Based on these sub-instances, we create a combined linear bandit instance with dimension $d _ { 0 } + d _ { 1 } + \ldots + d _ { L - 1 } = d$ with weight vector $\pmb { \mu } ~ = ~ \left( \pmb { \mu } _ { 0 } , . . . , \pmb { \mu } _ { L - 1 } \right)$ : At the beginning of each round $k$ , if round $k$ belongs to group $\kappa _ { i }$ , then the learner receives the decision set $\mathcal { D } _ { k } ~ =$ $\left\{ \left( \mathbf { 0 } _ { d _ { 0 } } , . . . , \mathbf { 0 } _ { d _ { i - 1 } } , \mathbf { x } , \mathbf { 0 } _ { d _ { i + 1 } } , . . . , \mathbf { 0 } _ { d _ { L - 1 } } \right) : \mathbf { x } \in \mathcal { A } _ { i } \right\}$ , where $\mathbf { 0 } _ { d _ { j } }$ corresponds to a zero vector with dimension $d _ { j }$ and $\mathbf { \mathcal { A } } _ { i }$ is the action set in the bandit instance $\mathcal { M } _ { i }$ . Under this construction, for any round $k \in \mathcal { K } _ { i }$ , the reward in the combined instance coincides with that of sub-instance $\mathcal { M } _ { i }$ . Specifically, after the learner selects action $\mathbf { x }$ , they receive a reward drawn from a scaled Bernoulli distribution with variance upper bounded by $\sigma ^ { 2 } ( i ) = \left( 2 ^ { i - 1 } / K \right) ^ { 2 }$ for $i \neq 0$ , and variance 0 for $i = 0$ . Note that in all groups, the variance is bounded by $\sigma _ { k } ^ { 2 }$ . With this construction in hand, we now proceed to prove the lower bound in Theorem 4.1.

Remark 4.5 (Linear Contextual Bandits vs. Stochastic
