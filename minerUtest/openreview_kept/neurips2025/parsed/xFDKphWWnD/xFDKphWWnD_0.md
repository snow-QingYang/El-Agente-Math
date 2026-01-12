## LINE 153-156

avoids orthogonal rotations between segments.

Theorem 1 (On memory loss bounds). Let each memory row be $\ell _ { 2 }$ -normalized, the $\alpha$ -alignment condition hold, and 154 $\begin{array} { r } { A = s o f t m a x \left( \frac { Q K ^ { \top } } { \sqrt { d } } \right) } \end{array}$ be the MRV attention matrix. Then:

$$
\Vert M _ { n + 1 } - M _ { n } \Vert _ { F } \leq \sqrt { 2 \left( 1 - \frac { \alpha } { m } \right) } \cdot \Vert M _ { n } \Vert _ { F } , \Vert M _ { n + 1 } \Vert _ { F } \geq \left( 1 - \sqrt { 2 \left( 1 - \frac { \alpha } { m } \right) } \right) \cdot \Vert M _ { n } \Vert _ { F } .
$$

In words: at least a 155 $\left( 1 - { \sqrt { 2 \left( 1 - { \frac { \alpha } { m } } \right) } } \right)$ fraction of the initial memory is guaranteed to be preserved 156 after a single MRV update (2) (right), and the memory loss is upper bounded by (2) (left).

Proof. Since each row of the attention
