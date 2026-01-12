## LINE 140-141

bound $\mathcal { L } ( \Psi , \theta ; \mathcal { G } )$ can be expressed as follows:

$$
\mathcal { L } ( \Psi , \theta ; \mathcal { G } ) = \mathbb { E } _ { q \ast ( \mathbf { Z } | \mathcal { G } ) } \left[ \log \frac { p _ { \theta } ( \mathbf { Z } , \mathcal { G } ) } { q _ { \Psi } ( \mathbf { Z } | \mathcal { G } ) } \right] = \mathbb { E } _ { q \ast ( \mathbf { Z } | \mathcal { G } ) } \left[ \log p _ { \theta } ( \mathbf { Z } | \mathcal { G } ) \right] - \mathrm { K L } \left( q _ { \Psi } ( \mathbf { Z } | \mathcal { G } ) \| p ( \mathbf { Z } ) \right) .
$$

Variational inference enhances the model’s robustness and generalization capabilities [26, 27]. How  
ever, due to the differing distributions between noisy heterogeneous graph data and standard graph   
data, the obtained distribution tends to align with the noisy distribution, potentially misleading the   
GNN explainer into generating incorrect explanatory subgraphs. Therefore, we introduce a denoising   
module during the process of variational inference. The original encoder part is modified to:

$$
q _ { \Psi } ^ { \prime } ( \mathbf { Z } | \mathcal { G } ) = \int q _ { \Psi } ( \mathcal { G } | \tilde { \mathcal { G } } ) q ( \tilde { \mathcal { G } } | \mathcal { G } ) \mathrm { d } \tilde { \mathcal { G } } ,
$$

where 141 $\Psi$ is the encoder based on $\tilde { \mathcal { G } }$ , and $\begin{array} { r } { q ( \tilde { \mathcal { G } } | \mathcal { G } ) = \prod _ { r \in \mathcal { R } } q ( \tilde { \mathbf { A } } _ { r } | \mathbf { A } _ { r } ) } \end{array}$ . During this process, the 142 evidence lower bound is expressed as:

$$
\mathcal { L } _ { d } = \mathbb { E } _ { q _ { \Psi } ^ { \prime } ( \mathbf { Z } | \mathcal { G } ) } [ \log \frac { p _ { \theta } ( \mathbf { Z } , \mathcal { G } ) } { q _ { \Psi } ^ { \prime } ( \mathbf { Z } | \mathcal { G } ) } ] .
$$

As we need to derive the distribution of the noisy graph
