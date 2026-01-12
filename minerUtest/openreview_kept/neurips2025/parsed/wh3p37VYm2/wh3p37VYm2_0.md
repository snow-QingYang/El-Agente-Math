## LINE 218-222

following expected update:

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { t + 1 } - \mathcal { L } _ { t } ] \leq \nabla _ { \theta _ { t } } \mathcal { L } ^ { T } ( \theta _ { t + 1 } + \theta _ { t } ) } \\ & { - \displaystyle \sum _ { i = 1 } ^ { V } \nabla _ { e _ { i , t } } \mathcal { L } ^ { T } \mathbb { E } ( e _ { i , t + 1 } - e _ { i , t } ) + \frac { \beta } { 2 } \| \Delta \| ^ { 2 } , } \end{array}
$$

Substituting the embedding update based on equation 4 into the smoothness inequality,



$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { t + 1 } - \mathcal { L } _ { t } ] \leq \nabla _ { \theta _ { t } } \mathcal { L } ^ { T } ( \theta _ { t + 1 } - \theta _ { t } ) } \\ & { \quad - \eta \displaystyle \sum _ { i = 1 } ^ { V } \big ( p _ { i } \| \nabla _ { e _ { i , t } } \mathcal { L } \| ^ { 2 } + \lambda e _ { i , t } ^ { T } \nabla _ { e _ { i , t } } \mathcal { L } \big ) + \frac { \beta } { 2 } \| \Delta \| ^ { 2 } , } \end{array}
$$

and noting from the right hand side of the inequality above, $p _ { i }$ plays important role in reduction of the expected loss. However, the dependence on $p _ { i }$ , is coupled with weight decay, which explains why these two parameters are important to study more deeply to draw a conclusion about grokking.

# 4.2 Dataset Splitting Strategies
