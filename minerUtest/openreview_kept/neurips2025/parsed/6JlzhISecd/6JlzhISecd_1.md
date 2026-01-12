## LINE 119-121

SGDA) updates. Take $\alpha > 0 , \beta > 0$ as the step sizes, we have

$$
\begin{array} { r } { \mathbf { x } ^ { t + 1 } = \mathbf { x } ^ { t } - \alpha \nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } ^ { t } , \lambda ^ { t } ; \boldsymbol { \xi } ^ { t } ) , \lambda ^ { t + 1 } = \lambda ^ { t } + \beta \nabla _ { \lambda } \mathcal { L } ( \mathbf { x } ^ { t } , \lambda ^ { t } ; \boldsymbol { \xi } ^ { t } ) . } \end{array}
$$

Taking the variable substitution 119 $\widehat { \lambda } : = \mathbf { A } ^ { \top } \lambda$ yields the following recursion:

FSPDA-SA: for any $t \geq 0$ and any $i \in [ n ]$ ,

$$
\begin{array} { r l } & { \mathbf { x } _ { i } ^ { t + 1 } = \mathbf { x } _ { i } ^ { t } - \alpha \nabla f _ { i } ( \mathbf { x } _ { i } ^ { t } ; \boldsymbol { \xi } _ { i } ^ { t } ) - \eta \widehat { \lambda } _ { i } ^ { t } + \gamma \sum _ { j \in { \mathcal { N } } _ { i } ( \boldsymbol { \xi } _ { a } ^ { t } ) } \mathbf { C } _ { i j } ( \boldsymbol { \xi } _ { a } ^ { t } ) ( \mathbf { x } _ { j } ^ { t } - \mathbf { x } _ { i } ^ { t } ) , } \\ & { \widehat { \lambda } _ { i } ^ { t + 1 } = \widehat { \lambda } _ { i } ^ { t } + \beta \sum _ { j \in { \mathcal { N } } _ { i } ( \boldsymbol { \xi } _ { a } ^ { t } ) } \mathbf { C } _ { i j } ( \boldsymbol { \xi } _ { a } ^ { t } ) ( \mathbf { x } _ { j } ^ { t } - \mathbf { x } _ { i } ^ { t } ) . } \end{array}
$$



Note that 21 $\mathbf { x } ^ { 0 } , \widehat { \lambda } ^ { 0 }$ can be initialized arbitrarily.

FSPDA-STORM Algorithm. The second variant of FSPDA reduces the variance of the stochastic   
gradient term in (5) using the recursive momentum variance reduction technique [Cutkosky and   
Orabona, 2019]. Herein, the key idea is to utilize a control variate in estimating the (primal-dual)   
gradients of $\mathcal { L } ( \mathbf { x } , \lambda )$ . Take $\alpha , \beta > 0$ and $a _ { x } , a _ { \lambda } \in [ 0 , 1 ]$ as the momentum parameters, we have   
$\mathbf { x } ^ { t + 1 } = \mathbf { x } ^ { t } - \alpha \mathbf { m } _ { x } ^ { t } , \lambda ^ { t + 1 } = \lambda ^ { t } + \beta \mathbf { m } _ { \lambda } ^ { t }$ as the primal-dual updates, and

$$
\begin{array} { r l } & { \mathbf { m } _ { x } ^ { t + 1 } = \nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } ^ { t + 1 } , \boldsymbol { \lambda } ^ { t + 1 } ; \boldsymbol { \xi } ^ { t + 1 } ) + ( 1 - a _ { x } ) ( \mathbf { m } _ { x } ^ { t } - \nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } ^ { t } , \boldsymbol { \lambda } ^ { t } ; \boldsymbol { \xi } ^ { t + 1 } ) ) , } \\ & { \mathbf { m } _ { \lambda } ^ { t + 1 } = \nabla _ { \lambda } \mathcal { L } ( \mathbf { x } ^ { t + 1 } , \boldsymbol { \lambda } ^ { t + 1 } ; \boldsymbol { \xi } ^ { t + 1 } ) + ( 1 - a _ { \lambda } ) ( \mathbf { m } _ { \lambda } ^ { t } - \nabla _ { \lambda } \mathcal { L } ( \mathbf { x } ^ { t } , \boldsymbol { \lambda } ^ { t } ; \boldsymbol { \xi } ^ { t + 1 } ) ) . } \end{array}
$$

The aim of $\mathbf { m } _ { r } ^ { t + 1 }$ is to estimate $\nabla _ { \mathbf x } \mathcal L ( \mathbf x ^ { t + 1 } , \lambda ^ { t + 1 } )$ . Now
