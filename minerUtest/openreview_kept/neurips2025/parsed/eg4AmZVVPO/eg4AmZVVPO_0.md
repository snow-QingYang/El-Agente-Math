## LINE 107-108

log-variance (LV) divergences are common choices [6, 34, 38]:

$$
D _ { \mathrm { K L } } ( \mathbb { P } \mid \mathbb { Q } ) ( X ) = \mathbb { E } \left[ \log { \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { Q } } } ( X ) \right] + \log Z , \quad D _ { \mathrm { L V } } ( \mathbb { P } \mid \mathbb { Q } ) ( X ) = \mathbb { V } \left[ \log { \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { Q } } } ( X ) \right] .
$$

The likelihood ratio appearing in $\textcircled{5}$ is given explicitly by the Radon-Nikodym derivative:

$$
\log \frac { \mathrm { d } \mathbb { P } ^ { u , \pi } } { \mathrm { d } \mathbb { P } ^ { v , \tau } } = \int _ { 0 } ^ { T } ( u + v ) \cdot \Big ( u _ { \theta } + \frac { v - u } { 2 } + \nabla \cdot ( \sigma v - \mu ) \Big ) \mathrm { d } s + \int _ { 0 } ^ { T } ( u + v ) \mathrm { d } W _ { s } + \log \frac { p _ { \mathrm { p r i o r } } ( X _ { 0 } ^ { \theta } ) } { p _ { \mathrm { t a r g e t } } ( X _ { T } ^ { \theta } ) }
$$

where $X ^ { \theta }$ is the trajectory obtained by simulating the forward SDE $( 2 )$ using the parameterized control   
$u _ { \theta }$ . The log normalization constant from the target density disappears upon taking gradients, making   
110 this a practical objective for training. See $\mathbb { U }$ and Appendix A.2 of $\pmb { \Vert 3 8 \Vert }$ for detailed derivations.   
11 Once trained, the optimized control $u _ { \theta }$ allows generation of samples from $p _ { \mathrm { t a r g e t } }$ through forward   
simulations of $( 2 )$ . In practice, this continuous-time process must be discretized into finite steps   
$0 = t _ { 1 } < t _ { 2 } < \cdots < t _ { N } = T$ , introducing a trade-off between computational cost and accuracy.

# 114 4 Consistency Distilled Diffusion Samplers
