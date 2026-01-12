## LINE 107

forms of learning models and characterize $\pmb q ^ { * }$ and $L ^ { * }$ for these models.

Data Sets and Training Sequences In our analysis, we refer to the training budget $N$ and our   
learning model specifying learning based on $n _ { k }$ examples per component $k$ . We can think of $N$ and   
$\mathbf { \nabla } _ { \mathbf { \pmb { n } } }$ as specifying the number of training examples, in which case the training complexity is a sample   
complexity. Or, we can think of $N$ as indicating the number of training steps, and $n _ { k }$ as indicating   
the number of steps in which an example from component $k$ is used. In this case, training complexity   
is a measure of training time. Either interpretation is valid. But we should emphasize that we only   
study a dependence on how many examples are used from each component, not on the order (as in   
curriculum learning).   
Learnabilities and Mixing Ratios. We model learning as a function of the number of examples   
from each component, but for our analysis, it will useful to introduce the function $\bar { e } _ { N , k } ( { \pmb q } ) =$   
$\mathbb { E } _ { S \sim ( \mathcal { D } _ { q } ) ^ { n } } [ \mathcal { L } _ { k } ( \bar { \mathcal { A } } ( S ) ) ]$ , which captures the expected error on component $k$ with mixing proportions   
$\pmb q$ . We will refer to $\bar { e } _ { k } ( q )$ as the subpopulation error function in terms of the mixture $\pmb q$ . Since   
the per-component counts $\textbf { \em n }$ are multinomial, we have $\bar { e } _ { N } ( \pmb q ) = \mathbb { E } _ { \pmb { n } \sim \mathrm { M u l t } ( \pmb q , N ) } [ e ( \pmb { n } ) ] \in \bar { \mathbb { R } } ^ { K }$ and   
${ \cal L } _ { N } ( { \pmb p } , { \pmb q } ) = \langle { \pmb p } , \bar { e } _ { N } ( { \pmb q } ) \rangle$ . Frequently for large sample size N , $\bar { e } _ { N } ( { \pmb q } )$ will concentrate around $e ( \pmb q N )$ ,   
and we will sometimes exploit this in the analysis, or analyze for $\bar { e } ( q ) \approx e ( q N )$ .

# 108 3 Orthogonal Power Law

Many machine learning tasks can be captured with power law
