## LINE 72

size $N$ , outputs a model $A ( S )$ with test loss $\bar { \mathcal { D } _ { p } } ( \mathcal { A } ( S ) )$

Training Distribution. We consider training on i.i.d. samples $S \sim \mathcal { D } _ { q } ^ { N }$ from mixtures $\mathcal { D } _ { q }$ of the   
same $K$ components, but with potentially different mixing proportions $\pmb q \doteq \Delta _ { K }$ . For training mixing   
proportions $\pmb q$ , we denote $L _ { N } ( \pmb { p } , \pmb q ) = \dot { \mathbb { E } } _ { S \sim \mathcal { D } _ { p } ^ { N } } [ \mathcal { L } _ { p } ( A ( \bar { S } ) ) ]$ the expected test error on $D _ { \mathrm { t e s t } } = \mathcal { D } _ { p }$   
when training with $D _ { \mathrm { t r a i n } } = \mathcal { D } _ { q }$ (we frequently drop the subscript $N$ if its clear from context).   
The “non-shifted” expected test loss is then denoted $L _ { N } ^ { \mathrm { s a m e } } ( p ) = L _ { N } ( p , p )$ . In contrast, we denote   
$\begin{array} { r } { L _ { N } ^ { * } ( p ) = \operatorname* { m i n } _ { \pmb { q } \in \Delta _ { K } } \bar { L } _ { N } ( \pmb { p } , \pmb { q } ) } \end{array}$ the test error with the best mixing ratios, and $\pmb q ^ { * }$ the minimizing ratios.   
When $L ^ { * } < L ^ { \mathrm { s a m e } }$ K  and so ${ \pmb q } ^ { * } \neq { \pmb p }$ , this means we can benefit from mismatched training. Our main   
analysis objective is to charactarize $\pmb q ^ { * }$ , $L ^ { * }$ and the improvement over $L ^ { \mathrm { s a m e } }$ .   
$L _ { N } ^ { \mathrm { r a t i o } } = L _ { N } ^ { * } / L _ { N } ^ { \mathrm { s a m e } }$ mismatch benefit through the improvement i. Or, we can consider the training complexity $N _ { \epsilon } ( \pmb { p } , \pmb { q } ) = \operatorname* { m i n } \ N$ trs.t. $L _ { N } ( p , q ) \stackrel { \textstyle - } { = }$   
$\epsilon$ and the improvement $\begin{array} { r } { N _ { \epsilon } ^ { \mathrm { r a t i o } } : = \frac { N _ { \epsilon } ^ { * } ( p ) } { N _ { \epsilon } ^ { \mathrm { s a m e } } ( p ) } } \end{array}$ .   
Specifying the Learning Model The expected test loss $L _ { N } ( p , q )$ , and so $\pmb q ^ { * }$ and the benefit of   
mismatch, depend on the data distributions and learning behaviour of the algorithm. We capture   
these by modeling the subpoluation error function $e _ { k } ( n )$ , i.e. the error on each component $\mathcal { D } _ { k }$   
when training with $n _ { i }$ examples from each component $\mathcal { D } _ { i }$ . That is, for a vector of sample sizes   
$\pmb { n } = ( n _ { 1 } , \dots , n _ { K } ) \in \mathbb { Z } _ { \ge 0 } ^ { K }$ , denote $\pmb { \mathcal { D } } ^ { n } = ( \mathcal { D } _ { 1 } ) ^ { n _ { 1 } } \tilde { \times } \cdot \cdot \cdot \times ( \mathcal { D } _ { K } ) ^ { n _ { K } }$ the distributions over samples with   
$n _ { i }$ examples from each component $\mathcal { D } _ { i }$ . Then $e _ { k } ( { \pmb n } ) = \mathbb { E } _ { S \sim \pmb { \mathscr { D } } ^ { n } } [ \mathcal { L } _ { \mathcal { D } _ { k } } \left( A ( S ) \right) ]$ . When $e _ { k } ( { \pmb n } ) = g _ { k } ( n _ { k } )$   
depends only on the amount of within-component data, we say the components are orthogonal,   
meaning there is no transfer between them (as in our Chinese and European history example). The   
scalar function $g _ { k } ( n _ { k } )$ then captures the learning curve for each component. But more generally,   
there might also be transfer, with data from one component helping learning on another.

In any case, the learnability function $e : \mathbb { Z } _ { \geq 0 } ^ { K } \to \mathbb { R } ^ { K }$ , captures
