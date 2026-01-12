## LINE 163-164

qth-quantile among the critic targets:

$$
y ( s , a ) = r + \gamma \hat { V } _ { A } ( s ^ { \prime } ) ,
$$

subject to

$$
\begin{array} { r l } & { \hat { V } _ { A } ( s ^ { \prime } ; C ^ { \prime } ) = \operatorname { M e d } ( \{ \hat { V } _ { \phi _ { i } } ( s ^ { \prime } ; C ^ { \prime } ) \} _ { 1 \leq i \leq N _ { A } } ) } \\ & { \hat { V } _ { \phi _ { i } } ( s ^ { \prime } ; C ^ { \prime } ) = \operatorname { Q u a n t i l e } _ { q } ( \{ Q _ { \theta _ { j } ^ { \prime } } ( s ^ { \prime } , \pi _ { \phi _ { i } } ( s ^ { \prime } ) + \epsilon ) \} _ { 1 \leq j \leq N _ { C } } ) . } \end{array}
$$

The critic loss function is therefore defined as

$$
J _ { Q } ( \theta _ { j } ; C ^ { \prime } ) = \mathbb { E } _ { ( s , a , r , s ^ { \prime } ) \sim \mathcal { B } } [ ( Q _ { \theta _ { j } } ( s , a ) - r + \gamma V _ { A } ( s ^ { \prime } ; C ^ { \prime } ) ) ^ { 2 } ] .
$$

All the target critics are soft-updated with parameter $\tau$ after one out of $M$ iterations of training, which   
is capable of sharing information to each target critic from all the other critic targets and bring to the   
next iteration.   
For the learning of actors, the MAMC also sampled $N _ { A }$ sets of mini-batch $\{ B _ { i } \} _ { 1 \le i \le N _ { A } }$ from the   
replay buffer $\mathcal { R }$ as it does in critics learning stage. The training of each actor $\pi _ { \phi _ { i } }$ is in turn guided   
by each critic $Q _ { \theta _ { j } }$ with objective $J _ { \pi } ( \phi _ { i } ; \theta _ { j } )$ (cf. Eq. (4)) on its mini-batch $B _ { i }$ . The idea of updating   
$M$ times within a mini-batch for each actor and critics is similar to sample multiple reuse (SMR)   
proposed in [22], which is able to stabilize the learning sequence.

# 172 4.3 Actor Evaluation, Exploration, and Exploitation
