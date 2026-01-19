# Agentic Reader Result
**Paper ID:** xFDKphWWnD
**Issue File:** xFDKphWWnD_0.md
**Status:** success
**Timestamp:** 2026-01-19T07:46:57.946629
**Model:** openai:gpt-5-mini
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

Yes — there is a (mathematical/interpretation) issue in the paper as quoted.

What the paper states
- Theorem 1 (lines ~153–156) gives the pair of bounds (displayed as (2)):
  ||M_{n+1} - M_n||_F ≤ sqrt{2(1 − α/m)} · ||M_n||_F,
  ||M_{n+1}||_F ≥ (1 − sqrt{2(1 − α/m)}) · ||M_n||_F.
  (See Theorem 1 and the displayed inequality in the snippet.)

Why this is an issue
- The second bound is presented in words as “at least a (1 − sqrt{2(1 − α/m)}) fraction of the initial memory is guaranteed to be preserved” (lines ~155–156). But the factor (1 − sqrt{2(1 − α/m)}) can be negative for typical parameter ranges, so it does not generally give a meaningful “fraction preserved.”
- Concretely: α is an inner product between ℓ2-normalized vectors, so α ≤ 1. For typical m ≥ 3, sqrt{2(1 − α/m)} > 1 (e.g. take α ≤ 1 ⇒ sqrt{2(1 − 1/m)} ≥ sqrt{2(1 − 1/3)} = sqrt{4/3} > 1), hence 1 − sqrt{2(1 − α/m)} < 0. Thus the bound can be negative even though ||M_{n+1}||_F ≥ 0 always. The derivation in the proof (they use ||M_n||_F = sqrt{m} and the triangle inequality to obtain the second displayed bound — see proof lines around the derivation that follows the first inequality) is algebraically correct, but the resulting lower bound is often vacuous (negative) and the wording “fraction preserved” is misleading.

Suggested fixes
- Replace the wording “at least a (1 − sqrt{2(1 − α/m)}) fraction is preserved” with a clarification that the bound may be vacuous (i.e., can be negative) unless additional assumptions (e.g., extremely large α relative to m) hold. Equivalently state the lower bound as
  ||M_{n+1}||_F ≥ max(0, 1 − sqrt{2(1 − α/m)}) · ||M_n||_F,
  or explicitly note that the nontrivial positive guarantee requires α/m ≥ 1/2 (which is practically unattainable for typical m and α ≤ 1), so in general the bound only yields the trivial nonnegativity ||M_{n+1}||_F ≥ 0.
- If the authors intended a nontrivial positive fraction guarantee, they must add an explicit assumption on α (relative to m) and state it in Theorem 1.

Relevant locations in the paper
- Theorem 1 and the displayed inequalities (lines ~153–156 in the reviewer snippet / the statement of Theorem 1).
- Proof and derivation of the bounds (the steps deriving the per-row bound, summing over rows, using ||M_n||_F = sqrt{m}, and the reverse-triangle step producing the lower bound — see the proof following Theorem 1 in the text).