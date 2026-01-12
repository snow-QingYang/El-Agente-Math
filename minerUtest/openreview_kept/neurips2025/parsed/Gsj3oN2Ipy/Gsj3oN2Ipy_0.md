## LINE 174-175

views $\mathbf { X } _ { [ N ] }$ according to vision-text similarity:

$$
\begin{array} { r } { \hat { y } = \arg \underset { c } { \operatorname* { m a x } } \bar { p } ( y = c \mid \mathbf { X } _ { [ n ] } , \mathbf { p } _ { \mathrm { a d a } } ) , \quad i ^ { * } = \arg \underset { i \in \mathcal { I } } { \operatorname* { m i n } } \mathcal { H } ( p ( \{ \mathbf { X } _ { [ N ] } \} _ { i } , \mathbf { p } _ { \mathrm { a d a } } ) ) , } \\ { \mathrm { w h e r e } \quad \mathcal { I } = \left\{ j : \arg \underset { c } { \operatorname* { m a x } } p ( y = c \mid \{ \mathbf { X } _ { [ N ] } \} _ { j } , \mathbf { p } _ { \mathrm { a d a } } ) = \hat { y } \right\} . } \end{array}
$$

We first obtain a confident prediction $\hat { y }$ by aggregating predictions over the selected subset $\mathbf { X } _ { [ n ] }$   
using the adapted prompt $\mathbf { p } _ { \mathrm { a d a } }$ . Then, from the subset $\mathcal { T }$ of patches whose predicted label matches   
$\hat { y }$ , we select the patch $\mathbf { X } _ { i ^ { * } }$ with the lowest prediction entropy. This avoids directly selecting the   
lowest-entropy patch from the entire set $\mathbf { X } _ { [ N ] }$ , which may include highly confident but irrelevant   
patches. Finally, we insert the selected patch into the corresponding memory slot $\mathcal { M } _ { \hat { y } }$ . If the memory   
is at full capacity, we remove the patch with the highest entropy among the existing entries and the   
current candidate.   
These three steps for each test image constitute a round of mutual promotion between the tunable   
textual prompt and the evolving visual memory. Afterward, we obtain two predictions for the current   
test image: one from the optimized prompt and one from the updated memory $\mathcal { M } ^ { \prime }$ . We combine   
them to produce the final prediction:

$$
P _ { \mathrm { f i n a l } } = P _ { \mathrm { p t } } + P _ { \mathrm { m e m o } } = p ( \mathbf { y } \mid \mathbf { v } , \mathbf { p } _ { \mathrm { a d a } } ) + \mathrm { S o f t m a x } ( \mathbf { M } ^ { \prime ^ { \mathrm { a d a } } \top } \mathbf { v } ) ,
$$

where 175 $P _ { \mathrm { p t } } , P _ { \mathrm { m e m o } } \in \mathbb { R } ^ { C }$ . The prediction $P _ { \mathrm { m e m o } }$ is obtained via similarity-based classification, as in the memory retrieval step, and 176 $\mathbf { M } ^ { \mathrm { \prime } \mathrm { a d a } }$ is computed from the updated memory following Eqs. 3 and 4.

It is worth noting that we perform only a single
