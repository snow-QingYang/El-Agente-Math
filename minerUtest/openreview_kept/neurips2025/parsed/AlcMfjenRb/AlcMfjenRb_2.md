## LINE 206-207

like to estimate the following for any fixed   
direction $e$ :

$$
\mathbb { E } _ { Z } \big [ d \cdot p _ { m b } ( x ) - d \langle \nabla f , e \rangle \big ] ^ { 2 } \approx \frac { d ^ { 2 } } { t ^ { 2 } } \mathbb { E } _ { Z } \Big [ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } F ( x + t e , Z _ { i } ^ { + } ) - f ( x + t e ) \Big ] ^ { 2 } \overset { ( 1 ) } { \approx } \frac { d ^ { 2 } \tau } { n } \frac { \sigma _ { 1 } ^ { 2 } } { t ^ { 2 } } .
$$

With that, we bound the variance:

$$
\mathbb { E } _ { e } \mathbb { E } _ { Z } \big \| \hat { g } _ { m b } - \nabla f \big \| ^ { 2 } \gtrsim \mathbb { E } _ { e } \mathbb { E } _ { Z } \big \| \hat { g } _ { m b } - \mathbb { E } _ { Z } \hat { g } _ { m b } \big \| ^ { 2 } \approx \mathbb { E } _ { e } \mathbb { E } _ { Z } \big \| \hat { g } _ { m b } - d \langle \nabla f , e \rangle \big \| ^ { 2 } \overset { ( 7 ) } { \approx } \frac { d ^ { 2 } \tau \sigma _ { 1 } ^ { 2 } } { n t ^ { 2 } } .
$$

# 207 Can the mini-batching scheme be improved?

This subsection explores an unexpected source of improvement
