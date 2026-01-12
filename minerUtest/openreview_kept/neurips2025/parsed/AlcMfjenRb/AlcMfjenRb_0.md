## LINE 169-172

such that the following inequality holds for all $x , y \in \mathbb { R } ^ { d }$ :

$$
\| \nabla f ( x ) - \nabla f ( y ) \| \leq L \| x - y \| .
$$

In the two-point feedback setting, we require the following generalization:

Assumption60 $\mathbf { 1 } ^ { \prime }$ . For al l $Z \in Z$ the function $F ( \cdot , Z )$ is $L$ -smooth on $\mathbb { R } ^ { d }$ .

Note that the uniform161 $1 ^ { \prime }$ implies 1.

Assumption 2. The function $f$ is $\mu$ -strongly convex on $\mathbb { R } ^ { d }$ , i.e., it is continuously dif  
ferentiable and there is a constant $\mu > 0$ such that the following inequality holds for all   
$x , y \in \mathbb { R } ^ { d }$ :

$$
{ \frac { \mu } { 2 } } \| x - y \| ^ { 2 } \leq f ( x ) - f ( y ) - \langle \nabla f ( y ) , x - y \rangle .
$$

We now turn to assumptions on the sequence of noise states $\{ Z _ { i } \} _ { i = 0 } ^ { \infty }$ . Specifically, we   
consider the case where $\{ Z _ { i } \} _ { i = 0 } ^ { \infty }$ forms a time-homogeneous Markov chain. Let Q denote the   
corresponding Markov kernel. We impose the following assumption on Q to characterize its   
mixing properties:   
Assumption 3. $\{ Z _ { i } \} _ { i = 0 } ^ { \infty }$ is a stationary Markov chain on $( \boldsymbol { \mathbb { Z } } , \boldsymbol { \mathcal { Z } } )$ with Markov kernel Q and   
unique invariant distribution $\pi$ . Moreover, $\mathrm { \Delta Q }$ is uniformly geometrically ergodic with mixing   
time $\tau \in \mathbb { N }$ , i.e., for every $k \in \mathbb N$ ,

$$
\Delta ( \boldsymbol { \mathrm { Q } } ^ { k } ) = \operatorname* { s u p } _ { z , z ^ { \prime } \in \boldsymbol { \mathrm { Z } } } ( 1 / 2 ) \| \boldsymbol { \mathrm { Q } } ^ { k } ( z , \cdot ) - \boldsymbol { \mathrm { Q } } ^ { k } ( z ^ { \prime } , \cdot ) \| _ { \mathsf { T V } } \leq ( 1 / 4 ) ^ { \lfloor k / \tau \rfloor } .
$$

Assumption 3 is common in the literature on Markovian stochasticity [14, 12, 13, 5, 52]. It   
includes, for instance, irreducible aperiodic finite Markov chains [18]. The mixing time $\tau$   
reflects how quickly the distribution of the chain approaches stationarity, providing a natural   
measure of the temporal dependence in the data.   
Next, we specify our assumptions on the oracle. As discussed in Section 1.1, these assumptions   
differ based on the type of feedback.

Assumption 4 (for one-point). For all 178 $x \in \mathbb { R } ^ { d }$ it holds that $\mathbb { E } _ { \pi } [ F ( x , Z ) ] = f ( x )$ . Moreover, for all 179 $Z \in Z$ and $x \in \mathbb { R } ^ { d }$ it holds that

$$
| F ( x , Z ) - f ( x ) | ^ { 2 } \leq \sigma _ { 1 } ^ { 2 } ,
$$

Assumption180 $\mathbf { { 4 } ^ { \prime } }$ (for two-point). For all $x \in \mathbb { R } ^ { d }$ it holds that $\mathbb { E } _ { \pi } [ \nabla F ( x , Z ) ] = \nabla f ( x )$ . Moreover, for all 181 $Z \in Z$ and $x \in \mathbb { R } ^ { d }$ it holds
