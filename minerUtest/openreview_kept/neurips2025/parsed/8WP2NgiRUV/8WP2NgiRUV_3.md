## LINE 106-108

Directed hypergraphical models for discrete variables (classical regime)   
. Directed hypergraphical models for continuous variables (additive noise model)   
We will first introduce the ‘hyper Markov property’ which will be respected by distributions which   
are ‘Markov’ with respect to a given hypergraph, rather than Markov with respect to a given graph.   
We emphasize that since hypergraphs are a strict generalization of existing graphical models, we can   
see this hyper DAG or HDAG structure as an intermediate level of structure between the DAG and   
the SEM (structural equation model). In that sense, we write:

$$
\mathrm { D A G s } \precsim \mathrm { H D A G s } \prec \mathrm { S E M s }
$$

In what follows, we will demonstrate that this more fine-grained structure is not only identifiable   
directly from data, but also that this perspective allows for greater insights into the identifiability of   
different hypergraphs (and hence graphs) using finite observations rather than the population limit.

# 72 2.1 Undirected Models

Let us write $X \in \mathbb { R } ^ { d }$ for some number of dimensions $d \in \mathbb { N }$ . We will later choose to restrict to   
discrete, continuous, or mixed $X$ as appropriate. We write an undirected graph as $\mathcal { G } ^ { \prime } = ( V , E ^ { \prime } )$ and   
undirected hypergraph as $\mathcal { H } ^ { \prime } = ( V , H ^ { \prime } )$ , where we take the vertices as $V = [ d ] : = \{ 1 , \dots , d \}$ , the   
undirected edges as $E ^ { \prime } \subseteq \{ ( i , j ) : i \neq j \in V \}$ , and the undirected hyperedges as $\bar { H ^ { \prime } } \subseteq \{ S \subseteq V \}$   
We will sometimes abuse notation and write $( i , j ) \in \mathcal { G } ^ { \prime }$ to mean $( i , \bar { j } ) \in \bar { E ^ { \prime } }$ and similarly for $\mathcal { H } ^ { \prime }$   
(Note that we are reserving the unprimed versions for the directed versions.)   
We will assume throughout this work that we are in the case of fully observed variables. Moreover,   
we will assume that the density is strictly positive to ensure (a) that there is no confusion caused by   
switching between the pairwise, local, and global properties; and (b) that the score-based definitions   
we introduce on the log-probability face no ambiguities in regions of zero density.   
Definition 1. Undirected Markov Property. Let us take $N ( i )$ to denote the neighbors of $i \in V$ . We   
may say that some distribution $p _ { X } ( \pmb { x } )$ is (locally) Markov with respect to some undirected graph $\mathcal { G } ^ { \prime }$   
if it holds for any $i$ that “ $\cdot X _ { i } \perp X _ { V - N ( i ) - \{ i \} } \mid X _ { N ( i ) } \cdots$ , where – denotes set minus. Preparing for   
our focus on additive models of the log probability, this can equally be required as:

$$
\begin{array} { r } { p _ { X } ( \pmb { x } ) = p _ { N ( i ) } ( \pmb { x } _ { N ( i ) } ) \cdot p _ { i } ( x _ { i } | \pmb { x } _ { N ( i ) } ) \cdot p _ { V - N ( i ) - \{ i \} } ( \pmb { x } _ { V - N ( i ) - \{ i \} } | \pmb { x } _ { N ( i ) } ) } \\ { \xi _ { X } ( \pmb { x } ) : = \log p _ { X } ( \pmb { x } ) = \xi _ { N ( i ) } ( \pmb { x } _ { N ( i ) } ) + \xi _ { i } ( x _ { i } | \pmb { x } _ { N ( i ) } ) + \xi _ { V - N ( i ) - \{ i \} } ( \pmb { x } _ { V - N ( i ) - \{ i \} } | \pmb { x } _ { N ( i ) } ) } \end{array}
$$

where there exists some conditional probabilities $p _ { i }$ and $\underline { p } _ { V - N ( i ) \underline { { - } } \{ i \} }$ or some conditional log   
probabilities $\xi _ { i }$ and $\xi _ { V - N ( i ) - \{ i \} }$ such that these equations hold true. This can be additionally written   
in terms of the clique representation, when we write all cliques of the graph as $C l ( { \mathcal { G } } ^ { \prime } ) = \{ S \subseteq V :$   
$S$ is a clique in $\mathcal { G } ^ { \prime } \}$ , as follows:

$$
\log p _ { X } ( { \pmb x } ) = : \xi _ { X } ( { \pmb x } ) = \sum _ { S \in C l ( { \mathcal G } ^ { \prime } ) } \xi _ { S } ( x _ { S } )
$$

Definition 2. Undirected Hyper-Markov Property. It is now straightforward to generalize this   
property to hypergraphs as follows:

$$
\log p _ { X } ( \pmb { x } ) = : \xi _ { X } ( \pmb { x } ) = \sum _ { S \in \mathcal { H } ^ { \prime } } \xi _ { S } ( x _ { S } )
$$

That is, we write the hypergraph edges as specifically representing the energy terms in the log  
probability function. It is straightforward to verify that this is strictly more general than hypergraphs   
which can be created as a result of the maximal clique structure of a typical graph. Nonetheless, in   
what follows we hope to focus on the identifiability as well as the usefulness of this finer-grained   
structure for graphical models.

# 98 2.2 Directed Classical Models

We will write a directed graph as $\mathcal { G } = ( V , E )$ and a directed hypergraph as $\mathcal { H } = ( V , H )$ where   
the directed edges are $E \subseteq \{ ( k , j ) : k \neq j \in V \}$ and the directed hyperedges are $H \subseteq \{ ( S , j ) :$   
$j \in V , S \subseteq ( V ^ { \mathbf { \bar { \mathbf { \alpha } } } } - j ) \}$ . That is, we are assuming that each hyperedge has only one "out arrow" and   
up to $| S |$ "in arrows". It is hoped the purpose for this is relatively clear in the context of a causal   
diagram which must use several parents to generate a single child. We write the ‘parents of $j$ in $\vec { \mathcal { G } } ^ { \bullet }$ as   
$\bar { \mathrm { P a } _ { \mathcal { G } } } ( j ) = \{ k \in [ d ] : ( k , j ) \in \mathcal { G } \}$ and the ‘hyperparents of $j$ in $\mathcal { H }$ as $\mathrm { H y p P a } _ { \mathcal { H } } ( j ) \overset { - } { = } \{ S : ( S , j ) \in \mathcal { H } \}$ ,   
where the dependence on $\mathcal { G }$ and $\mathcal { H }$ will be dropped when obvious.   
Definition 3. Directed Markov Property. Here, we may once again recall the classical Markov   
property with respect to a DAG to be written as:

$$
\log p ( \pmb { x } ) = \sum _ { i = 1 } ^ { d } \log p ( x _ { i } | \pmb { x } _ { P a ( i ) } ) = \sum _ { i = 1 } ^ { d } \theta ( x _ { i } | \pmb { x } _ { P a ( i ) } )
$$

It is very easy to see that we may rewrite this using extraneous functions as:

$$
\log p ( \pmb { x } ) = \sum _ { i = 1 } ^ { d } \quad \sum _ { S \subseteq \mathrm { P a } ( i ) } \theta ( x _ { i } ; \pmb { x } _ { S } ) - \mathcal { Z } ( \pmb { x } _ { \mathrm { P a } ( i ) } )
$$

where it is now the case that we do not have the $\theta$ energy terms explicitly representing a conditional   
distribution, but are instead arbitrary functions which are then set to the proper normalization via   
the $\mathcal { Z }$ function. It can be seen that the $\mathcal { Z }$ function does not explicitly
