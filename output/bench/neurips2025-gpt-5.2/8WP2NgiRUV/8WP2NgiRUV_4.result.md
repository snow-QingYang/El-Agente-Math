# Agentic Reader Result
**Paper ID:** 8WP2NgiRUV
**Issue File:** 8WP2NgiRUV_4.md
**Status:** success
**Timestamp:** 2026-01-19T23:43:03.867053
**Model:** gpt-5.2
**Max Iterations:** 10

**Issue Content:**
## LINE 211

Directed Additive Noise Models

For continuous variables, we will generate data from the additive noise model (ANM), meaning that all variables are a deterministic function of their parent variables, plus an additive noise term.

33 Definition 5. Additive Noise Model. This may be written as:

$$
X _ { j } = f _ { \mathrm { P a } ( j )  j } ( X _ { \mathrm { P a } ( j ) } ) + \varepsilon _ { j }
$$

Each of the $\varepsilon _ { j }$ are taken to be independent, mean-zero random variables.

Definition 6. Higher-Order Additive Noise Model. We may once again generalize to the higher-order   
additive model through the use of the structure encoded by the directed hypergraph.

$$
X _ { j } = \Big ( \sum _ { S \in \mathrm { H y p P a } ( j ) } f _ { S  j } ( x _ { S } ) \Big ) + \varepsilon _ { j }
$$

Specifically, we endow the generating function $f _ { \operatorname* { P a } ( j )  j }$ with an additive model structure obeying   
the hyperparents of the HDAG. Models like CAM or LinGAM then correspond to using singleton   
hyperparents, whereas the most general ANM corresponds to using the entire block of parents as the   
largest hyperparent, as depicted in Figure 1. We will follow CAM Bühlmann et al. [2014] in assuming   
Gaussian noise for algorithmic purposes via the minimization of mean-squared error corresponding   
to maximizing the log-likelihood; however, surprisingly, we show in Theorem 4 that the our settings   
2 and 3 actually overlap in the case of additive Gaussian noise.

![](images/9193eefe4edb7a137fad170874858ffc674cd7eedbf097314d711742720f0aa3.jpg)  
Figure 1: A depiction of the distinguishing power for hypergraphs corresponding to the same DAG.

# 144 3 Structure Identifiability

# 3.1 Undirected Models

First recall that under our assumptions of fully observed variables and strictly positive density,   
meaning that the density function is identifiable directly from the observed distribution (under mild   
assumptions like continuity for the continuous variable case) Rosenblatt [1956].   
Importantly, then, one may only be concerned in measuring the hypergraph structure as described in   
Section 2.1; however, this proves to be equally straightforward. For the case of graphical models and   
mixed-type variables, Zheng et al. [2023] write the generalized precision matrix as:

$$
\Omega _ { i , j } : = \Big \| \frac { \partial ^ { 2 } } { \partial _ { i } \partial _ { j } } \log p ( \boldsymbol { x } ) \Big \| : = \Big ( \mathbb { E } \Big [ \Big | \frac { \partial ^ { 2 } \log p ( \boldsymbol { x } ) } { \partial _ { i } \partial _ { j } } \Big | ^ { 2 } \Big ] \Big ) ^ { \frac { 1 } { 2 } }
$$

In the case of hypergraphical models and discrete variables, Enouen and Sugiyama [2024] similarly   
write the existence of ‘higher-order information’ for some subset $T \subseteq [ \bar { d } ]$ (where $T \supsetneq \{ i , j \}$ is   
chosen to imply higher-order) if it is the case that:

$$
\Omega _ { T } : = \Big \| \sum _ { S \supseteq T } \theta _ { S } ( x _ { S } ) \Big \| > 0 \qquad \mathrm { w h e r e } \qquad \log p ( x ) = \sum _ { S } \theta _ { S } ( x _ { S } )
$$

A straightforward combination of these approaches are sufficient for recovery of the hyper Markov   
network or undirected hypergraph. Nonetheless, our experiments will instead focus on identification   
of the directed structure as in the following two sections. Thus, for our purposes it is sufficient to say   
that the density and log density functions are identifiable directly from the observed distribution.

# 3.2 Directed Classical Models

Before our main theorem of identifiability extending the result of Verma and Pearl [1990], we must   
first introduce the notion of multi-dependence to extend the typical notion of conditional independence   
which is the workhorse of causal structure learning. We will focus on discrete and finite variables as   
in the classical case[Verma, 1993, Pearl, 2009]; however, most results clearly extend to continuous or   
mixed variables under mild conditions, and we later discuss one such special case in Theorem 4.

Definition 7. Conditional Multi-dependence. We write that $X _ { i }$ and $X _ { j }$ are dependent if the distribution $\log p ( x _ { i } , x _ { j } )$ must be written with a 2D energy term, $\theta _ { i j } ( x _ { i } , \bar { x _ { j } } )$ , rather than the sum of two 1D energy terms, $\theta _ { i } ( x _ { i } ) + \theta _ { j } ( x _ { j } )$ , (corresponding to the product when the log is removed). We will write that $X _ { i }$ , $X _ { j }$ , and $X _ { k }$ are tri-dependent (or generally multidependent), if the distribution $\log p ( x _ { i } , x _ { j } , x _ { k } )$ must be written with a 3D energy term, rather than the sum of three 2D energy terms. It can be seen that this does not have a convenient product formulation like the classical case of dependence and independence because of the "mixing" or "torsion" between the three 2D terms. Nonetheless, we will attempt to prove the usefulness of such a definition in the following theorem. Generalization to conditional tests is straightforward.

Theorem 1. The HDAG is identifiable up to the hyper Markov Equivalence classes (HMECs), consisting of all HDAGs with the same "body" and the same (unshielded) "multi-colliders", paralleling the existing result identifying DAGs up to their skeleton and (unshielded) colliders.

In the same sense that a condi  
tional independence test can never   
eliminate a causal arrow between   
two variables, a conditional multi  
independence test can never sepa  
rate a higher-order causal relation  
ship between a set of three or more   
variables. Removing the arrow  
heads from the DAG returns the   
DAG’s skeleton; similarly, removing the arrowheads from the HDAG returns the HDAG’s body,   
see Table 1 and Figure 2. In some sense, this half of the theorem about the “body identifiability”   
immediately states that the structure we introduced is identifiable.   
For multicolliders, recall that a collider occurs when there is a conditional dependence which is   
broken after marginalizing out the child, or equally a conditional independence which is broken when   
conditioning on the child. The multicollider of an HDAG will occur similarly via a multidependence   
which is broken after marginalizing out the child. Although collisions between two parents will   
already be covered, there are cases of three or more parents which are unshielded and can hence   
be identified from Theorem 1. In particular, there are cases which are not identified in the classical   
setting, see the RHS of Figure 2. This seeming anomaly is in part due to the historical conflation   
over time between what structure is recoverable from the conditional independence tests vs. what   
structure is recoverable from the observed distribution. Indeed, the MEC only describes what is   
distinguishable via the conditional independence conditions, making it unable to detect what can be   
seen via the conditional multi-independence test we introduce.   
Another key consequence of this different perspective will be a statistical one. In particular, for a   
$K$ -dimensional energy term in the body of an HDAG, we know that it requires on the order of $\mathcal { O } ( n ^ { K } )$   
samples to be appropriately learned. Consequently, without access to infinite samples, this places   
further restrictions on the HMEC classes (and hence MEC classes) of ‘distinguishability under finite   
samples’, whereas MECs are only able to easily represent ‘distinguishability under infinite samples’   
as in the asymptotic regime.

Table 1: Notation for hypergraphs   

<table><tr><td>DAG terms</td><td>HDAG terms</td></tr><tr><td>G&#x27;,undirected graph G,directed acyclic graph (DAG) moralized graph of a DAG skeleton of a DAG (unshielded) collider</td><td>H&#x27;,undirected hypergraph H,hyper DAG or HDAG immoralized hypergraph of an HDAG body of an HDAG (unshielded) multicollider</td></tr></table>

![](images/5a88314dacdc52f9a115095c288b5486d38afb3f632670028bcd0f6c3cccd332.jpg)  
Figure 2: A gradual refinement of the DAGs within a Markov Equivalence Class (MEC) to a stronger refinement of HDAGs based on Theorem 1 to a Hyper Markov Equivalence Class (HMEC). There are $d = 4$ variables with a fully-connected DAG structure. The green triangle represents the third-degree hyperedge in the body of an HDAG. Lack of arrows indicate multiple possible orientations for different DAGs/ HDAGs of the same MEC/ HMEC.

# 06 3.3 Directed Additive Noise Models

In this section, we establish identifiability results for recovering the hyper-DAG in the ANM case. For clearer exposition, we first reproduce the arguments of Hoyer et al. [2008] which shows that, in general position, the additive noise model (ANM) is identifiable. We extend their result to a multi-dimensional result which handles the case of multiple parents rather than only the case of one parent node and one child node (slightly different from the extension in Theorem 28 of Peters et al. [2014] because it will more easily generalize to the hypergraph result).

Theorem 2. Let the joint probability densities


## Agentic Reader Analysis

Yes—this snippet reveals a **mathematical formula / notation inconsistency** in the paper’s definitions of the additive noise model.

### What the issue is
In **Definition 6 (Higher-Order Additive Noise Model)**, the structural equation mixes notation for random variables vs. realizations:

- The left-hand side uses the random variable **\(X_j\)**.
- Inside the sum it uses **\(x_S\)** (lowercase), which typically denotes a *realized value* of \(X_S\), not the random vector itself.

This is inconsistent with **Definition 5**, which correctly uses random-variable notation \(X_{\mathrm{Pa}(j)}\) inside the function.

### Where it occurs (citation)
- **Definition 5**:  
  \[
  X _ { j } = f _ { \mathrm { P a } ( j )  j } ( X _ { \mathrm { P a } ( j ) } ) + \varepsilon _ { j }
  \]
- **Definition 6 (problematic part)**:  
  \[
  X _ { j } = \Big ( \sum _ { S \in \mathrm { H y p P a } ( j ) } f _ { S  j } ( x _ { S } ) \Big ) + \varepsilon _ { j }
  \]
  Here, **\(f_{Sj}(x_S)\)** should be **\(f_{Sj}(X_S)\)** for consistency with a structural equation over random variables.

So the formula issue is a **notation error (uppercase/lowercase mismatch)** in Definition 6.