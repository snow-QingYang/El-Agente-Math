# Agentic Reader Result
**Paper ID:** OCxhcqUtDf
**Issue File:** OCxhcqUtDf_1.md
**Status:** success
**Timestamp:** 2026-01-19T06:52:52.324495
**Model:** openai:gpt-5-mini
**Max Iterations:** 10

**Issue Content:**
## LINE 117-118

need to minimize the cost throughout the trajectory $x _ { t }$ , $t \in [ 0 , 1 ]$ with the   
following objective

$$
g ^ { c } ( x ) = \operatorname* { i n f } _ { v ( x , t ) } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \left( \frac { \| v ( t , x _ { t } ) \| ^ { 2 } } { 2 } + U ( x _ { t } ) \right) \mathrm { d } t - g ( x _ { 1 } ) \bigg | x _ { 0 } = x \right] .
$$

In the last expression, we have united infimums by $\mu ( x )$ and control $v ( t , x )$ and as a sequence have   
removed the right side condition $x _ { 1 } \sim \mu ( x )$ . Based on dynamic programming approach, define the   
value function. For any $0 \leq t \leq 1$ , the value function satisfies:

$$
s ( t , x ) = \operatorname* { i n f } _ { x _ { t } } \mathbb { E } \left[ \int _ { t } ^ { 1 } \left( \frac { \| v ( t , x _ { t } ) \| ^ { 2 } } { 2 } + U ( x _ { t } ) \right) \mathrm { d } t - g ( x _ { 1 } ) \bigg | x _ { t } = x \right] ,
$$

such that our objective equals $s ( 0 , x )$ and the boundary condition at time point $t = 1$ is

$$
\forall x \in \mathbb { R } ^ { d } : s ( 1 , x ) = - g ( x ) .
$$

Function $s ( t , x )$ solves the Hamilton-Jacobi-Bellman


## Agentic Reader Analysis

Yes — this snippet indicates a mathematical/formulation issue.

What is wrong (two related problems)
1. Wrong minimization variable in the value function. In the displayed definition of the value function the infimum is written as inf_{x_t} even though the expectation is conditioned on x_t = x (so x_t is fixed) and the control appears inside the integrand. Concretely (lines ~12280–12680):

s(t,x) = operatorname*{inf}_{x_t} E[ ∫_t^1 ( (||v(t,x_t)||^2/2 + U(x_t)) dt - g(x_1) ) | x_t = x ].

This is incorrect: the minimization should be over admissible control processes v (e.g. inf_{v} or inf_{v(·)}) — not over x_t. As written the infimum index is inconsistent and makes the DP/HJB derivation invalid.

2. Unjustified removal of the terminal-distribution constraint. Just above the value function (lines ~11700–12280) the authors state they “united infimums by μ(x) and control v(t,x) and as a sequence have removed the right side condition x_1 ∼ μ(x).” The earlier definition of the c-conjugate used an infimum over μ(x) (see the prior formula for g^c), so dropping the constraint that the terminal law of x_1 under control v equals μ(x) is not harmless unless a proof is given. Removing that coupling without justification changes the optimization problem and invalidates the equivalence used to derive the HJB.

Where to look in the paper
- g^c formula: “g^{c}(x) = operatorname*{inf}_{v(x,t)} E[ ∫_0^1 ((||v(t,x_t)||^2/2 + U(x_t)) dt - g(x_1) | x_0 = x ]” (around LINE 117–121).
- Value function (incorrect infimum): “s(t,x) = operatorname*{inf}_{x_t} E[ ∫_t^1 ((||v(t,x_t)||^2/2 + U(x_t)) dt - g(x_1) | x_t = x ]” (around LINE 122–126).
- The sentence claiming the removal of the constraint x_1 ∼ μ(x) is right before the value-function definition (around LINE 119–122).

Suggested fixes
- Replace inf_{x_t} with inf_{v} (or inf over admissible control trajectories/processes) in the definition of s(t,x).
- Either restore the constraint linking μ(x) and the terminal law of x_1 (i.e. keep x_1 ∼ μ(x)) or supply a clear argument/lemma showing that optimizing jointly over μ and v with that constraint is equivalent to the unconstrained joint optimization they use.

Without these corrections/clarifications the subsequent HJB derivation and the claimed equivalence to the conjugate potential are not justified.