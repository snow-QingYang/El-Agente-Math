## LINE 155-156

approximation of the data distribution. We show our modeling   
assumptions in Fig. 2.

# 152 3.1 Bounded CIB

We can consider the upper bound to the concept bottleneck loss (2) in terms of the entropy-based   
definitions of the mutual information. Then, by using a variational approximation of the data   
distribution, we bound it by

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { U B - C I B } } \le H ( Y ) + ( 1 - \beta ) H ( C ) + H \left( p ( y \mid c ) , q ( y \mid c ) \right) + \left( 1 + \beta \right) \underset { p ( z ) } { \mathbb { E } } H \left( p ( c \mid z ) , q ( c \mid z ) \right) , } \\ & { \mathcal { L } _ { \mathrm { U B - C I B } } \le ( 1 - \beta ) H ( C ) + \underset { p ( c ) } { \mathbb { E } } H \left( p ( y \mid c ) , q ( y \mid c ) \right) + ( 1 + \beta ) \underset { p ( z ) } { \mathbb { E } } H \left( p ( c \mid z ) , q ( c \mid z ) \right) . } \end{array}
$$

We detail this derivation in Appendix A. We can maximize the concepts’ information bottleneck by   
minimizing the cross entropies of the predictive variables, $y$ and $c$ , and their corresponding ground   
truths and by adjusting the entropy of the concepts—cf. Fig. 2. The simplified upper bound of the   
concept information bottleneck is

$$
\mathcal { L } _ { \mathrm { S U B - C I B } } = ( 1 - \beta ) H ( C ) + \underset { p ( c ) } { \mathbb { R } } H \left( p ( y \mid c ) , q ( y \mid c ) \right) + ( 1 + \beta ) \underset { p ( z ) } { \mathbb { R } } H \left( p ( c \mid z ) , q ( c \mid z ) \right) .
$$

We denote the models that were trained using this bounded concept
