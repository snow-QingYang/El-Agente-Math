## LINE 194

Rankings as finite-sample approximations

86 As mentioned above, while we have no access to the CDFs themselves, we have samples from the   
joint distribution over the objectives, i.e., over, $p ( [ \mathsf { y } _ { 1 } , \mathsf { y } _ { 2 } , \ldots , \mathsf { y } _ { K } ] )$ . Namely, we can consider each   
model $h \in \mathcal H$ as a sample from the joint distribution and, by looking at each objective individually,   
as a sample from the marginal distributions.   
Let us now focus on the $k$ -th objective, ${ \tt y } _ { k }$ , and drop the subindex in the following to ease notation. Say   
that we have $| \mathcal { H } | = N$ i.i.d. realizations of the objective, i.e., $\left\{ \mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } , \ldots , \mathbf { y } _ { N } \right\} \overset { \forall \mathrm { i . i . d . } } { \sim } P _ { k }$ . Then, we can   
approximate Eq. 8 for the $i$ -th sample, $\mathbf { u } _ { i } = F ( \mathbf { y } _ { i } )$ , by computing its order statistic, i.e., the random   
variable representing its relative ranking within the population, $\begin{array} { r } { R ( i ) : = \sum _ { j = 1 } ^ { N } [ \mathbf { y } _ { j } < \mathbf { y } _ { i } ] } \end{array}$ , where   
Iverson brackets denote the indicator function, such that $\begin{array} { r } { \mathbf { y } _ { R ( 1 ) } \leq \mathbf { y } _ { R ( 2 ) } \leq \ldots \leq \mathbf { y } _ { R ( N ) } . } \end{array}$ . Specifically,   
since the empirical $C D F$ is the fraction of samples smaller than the input, it is direct to show that

$$
{ \hat { \mathbf { u } } } _ { i } = { \hat { F } } ( i ) : = { \frac { 1 } { N } } \sum _ { j = 1 } ^ { N } [ \mathbf { y } _ { j } < \mathbf { y } _ { i } ] = { \frac { 1 } { N } } R ( i )
$$

enjoys the following properties
