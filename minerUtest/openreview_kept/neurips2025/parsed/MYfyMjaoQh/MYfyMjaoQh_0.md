## LINE 168

Independent Subspace (IS) Loss. We can substitute the orthogonalisation procedure by using an   
auxiliary loss component for maximising the rank of the predicted subspace. For this purpose, we   
chose the squared cosine similarity computed between pairs of predicted vectors. The final expression   
for the independent subspace (IS) loss is

$$
\mathrm { I S } \mathrm { L o s s } = \frac { 1 } { K ^ { 2 } } \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { K } ( { \bf x } _ { k } ^ { T } W { \bf x } _ { l } ) ^ { 2 } - \frac { 1 } { K ^ { 2 } } \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { K } ( { \bf y } _ { k } ^ { T } W ^ { \frac { 1 } { 2 } } { \bf x } _ { l } ) ^ { 2 } ,
$$

where the predictions 176 $\left\{ { \bf x } _ { l } \right\}$ are normalised prior to the loss computation
