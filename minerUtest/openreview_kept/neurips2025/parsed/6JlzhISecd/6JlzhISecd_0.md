## LINE 108-109

consider the stochastic gradients:

$$
\nabla _ { \mathbf { x } } \mathcal { L } ( \mathbf { x } , \lambda ; \boldsymbol { \xi } ) : = \nabla \mathbf { f } ( \mathbf { x } ; \boldsymbol { \xi } ) + \tilde { \eta } \mathbf { A } ^ { \top } \lambda + \tilde { \gamma } \mathbf { A } ^ { \top } \mathbf { A } ( \boldsymbol { \xi } ) \mathbf { x } , \nabla _ { \lambda } \mathcal { L } ( \mathbf { x } , \lambda ; \boldsymbol { \xi } ) : = \tilde { \eta } \mathbf { A } ( \boldsymbol { \xi } ) \mathbf { x } ,
$$

where $\nabla \mathbf { f } ( \mathbf { x } ; \boldsymbol { \xi } ) \ = \ [ \nabla f _ { 1 } ( \mathbf { x } _ { 1 } ; \boldsymbol { \xi } _ { 1 } ) ; \ldots ; \nabla f _ { n } ( \mathbf { x } _ { n } ; \boldsymbol { \xi } _ { n } ) ] \ \in \ \mathbb { R } ^ { n d }$ . Notice that to facilitate algorithm   
development, we have taken a deterministic $\mathbf { A }$ for the term in $\nabla _ { \mathbf x } \mathcal L$ related to $\boldsymbol { \lambda }$ . Now observe the ith   
$d$ -dimensional block of $\mathbf { A } ^ { \top } \mathbf { A } ( \xi ) \mathbf { x }$ which can be aggregated within $\mathcal { N } _ { i } ( \boldsymbol { \xi } )$ the neighborhood of the   
ith agent as:

$$
\begin{array} { r } { \left[ \mathbf { A } ^ { \top } \mathbf { A } ( \xi ) \mathbf { x } \right] _ { i } = \sum _ { j \in \mathcal { N } _ { i } ( \xi ) } \mathbf { C } _ { i j } ( \xi ) ( \mathbf { x } _ { j } - \mathbf { x } _ { i } ) , } \end{array}
$$

where 109 $\mathbf { C } _ { i j } ( \xi ) \in \{ 0 , 1 \} ^ { d \times d }$ is diagonal and depends on the selected coordinates for the edge
