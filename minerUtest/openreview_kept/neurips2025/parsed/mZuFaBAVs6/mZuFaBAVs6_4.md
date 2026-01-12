## LINE 185-186

forecasts of vector-valued random variables [9]:

$$
\mathrm { E S } ( F , { \mathbf z } ) = \underset { { \mathbf x } \sim F } { \mathbb { E } } { \Vert } x - { \mathbf z } { \Vert } ^ { \beta } - \frac { 1 } { 2 } \underset { { \mathbf x } , { \mathbf x ^ { \prime } } \sim F } { \mathbb { E } } { \Vert } x - { \mathbf x ^ { \prime } } { \Vert } ^ { \beta } ,
$$

where $\lVert \cdot \rVert$ denotes the Euclidean norm and $\beta = 1$ is commonly used in the literature [23]. With   
$\beta = 1$ , the ES essentially becomes a multivariate extension of the CRPS and grows linearly with   
respect to the norm, making it less sensitive to outliers compared to the log-score. Since there is no   
simple closed-form expression for Eq. (10), it is often approximated using Monte Carlo methods,   
where multiple samples $\{ { \pmb x } _ { i } \} _ { i = 1 } ^ { n }$ are drawn from the forecast distribution to approximate the expected   
values:

$$
\operatorname { E S } ( F , \mathbf { z } ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \lVert { \pmb x } _ { i } - \mathbf { z } \rVert ^ { \beta } - { \frac { 1 } { 2 n ^ { 2 } } } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \lVert { \pmb x } _ { i } - { \pmb x } _ { j } \rVert ^ { \beta } .
$$

However, a significant disadvantage of using Eq. (11) as the loss function
