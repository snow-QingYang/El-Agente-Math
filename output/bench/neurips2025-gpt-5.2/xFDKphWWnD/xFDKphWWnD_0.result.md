# Agentic Reader Result
**Paper ID:** xFDKphWWnD
**Issue File:** xFDKphWWnD_0.md
**Status:** success
**Timestamp:** 2026-01-19T23:43:57.509201
**Model:** gpt-5.2
**Max Iterations:** 10

**Issue Content:**
## LINE 153-156

avoids orthogonal rotations between segments.

Theorem 1 (On memory loss bounds). Let each memory row be $\ell _ { 2 }$ -normalized, the $\alpha$ -alignment condition hold, and 154 $\begin{array} { r } { A = s o f t m a x \left( \frac { Q K ^ { \top } } { \sqrt { d } } \right) } \end{array}$ be the MRV attention matrix. Then:

$$
\Vert M _ { n + 1 } - M _ { n } \Vert _ { F } \leq \sqrt { 2 \left( 1 - \frac { \alpha } { m } \right) } \cdot \Vert M _ { n } \Vert _ { F } , \Vert M _ { n + 1 } \Vert _ { F } \geq \left( 1 - \sqrt { 2 \left( 1 - \frac { \alpha } { m } \right) } \right) \cdot \Vert M _ { n } \Vert _ { F } .
$$

In words: at least a 155 $\left( 1 - { \sqrt { 2 \left( 1 - { \frac { \alpha } { m } } \right) } } \right)$ fraction of the initial memory is guaranteed to be preserved 156 after a single MRV update (2) (right), and the memory loss is upper bounded by (2) (left).

Proof. Since each row of the attention


## Agentic Reader Analysis

Yes — a mathematical formula issue is indicated.

### What’s wrong
In the proof text around Theorem 1, the reverse triangle inequality is written incorrectly, with corrupted notation:

- The paper states (in the proof of Theorem 1):  
  “For any matrices \(M_{n+1}, M_n \in \mathbb{R}^{m\times d}\), we have:
  \[
  \Vert M_{n+1}\Vert_F \ge \Vert \bar{M}_n\Vert_F - \Vert M_{n+1}\bar{-M}_n\Vert_F^{-}.
  \]
  ”  
  This expression contains nonsensical terms like \(\bar{M}_n\), \(\bar{-M}_n\), and especially a Frobenius norm raised to a “minus” (the \(\Vert\cdot\Vert_F^{-}\) superscript). This is not a valid statement of the reverse triangle inequality. (Proof section near the derivation of the lower bound “(2) (left)”).【position ~18680–19350】

### What it should be
The intended inequality is almost certainly the standard reverse triangle inequality:
\[
\|M_{n+1}\|_F \ge \|M_n\|_F - \|M_{n+1}-M_n\|_F,
\]
which matches the subsequent substitution step they perform afterward.

### Where in the paper
- Theorem statement and bounds are introduced near “Theorem 1 (On memory loss bounds)”【position ~15480–16520】.
- The malformed reverse triangle inequality appears in the proof when deriving the lower bound (referred to as “(2) (left)”)【position ~18680–19350】.