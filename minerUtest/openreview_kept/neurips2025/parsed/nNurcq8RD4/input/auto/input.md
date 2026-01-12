# Grouped Satisficing Paths in Pure Strategy Games: a Topological Perspective

Anonymous Author(s)   
Affiliation   
Address   
email

# Abstract

In game theory and multi-agent reinforcement learning (MARL), each agent selects   
a strategy, interacts with the environment and other agents, and subsequently   
updates its strategy based on the received payoff. This process generates a sequence   
of joint strategies $( s ^ { t } ) _ { t \geq 0 }$ , where $s ^ { t }$ represents the strategy profile of all agents   
at time step $t$ . A widely adopted principle in MARL algorithms is "win-stay,   
lose-shift", which dictates that an agent retains its current strategy if it achieves the   
best response. This principle exhibits a fixed-point property when the joint strategy   
has become an equilibrium. The sequence of joint strategies under this principle is   
referred to as a satisficing path, a concept first introduced in [40] and explored in   
the context of $N$ -player games in [39]. A fundamental question arises regarding   
this principle: Under what conditions does every initial joint strategy $s$ admit a   
finite-length satisficing path $( s ^ { t } ) _ { 0 \leq t \leq T }$ where $s ^ { \flat } = s$ and $s ^ { T }$ is an equilibrium?   
This paper establishes a sufficient condition for such a property, and demonstrates   
that any finite-state Markov game, as well as any $N$ -player game, guarantees the   
existence of a finite-length satisficing path from an arbitrary initial strategy to some   
equilibrium. These results provide a stronger theoretical foundation for the design   
of MARL algorithms.

# 18 1 Introduction

Game theory provides a formal framework for analyzing strategic interactions among rational   
decision-makers. It examines how individuals optimize their decisions to maximize their own payoff   
while accounting for the actions of others. In most settings, agents act in self-interest, leading to the   
concept of an equilibrium [24], a strategy profile where no agent can unilaterally deviate to achieve   
a higher payoff. Due to its generality, game theory has become a cornerstone in machine learning,   
particularly in modeling competitive and cooperative behaviors, such as multi-agent systems [19],   
multi-objective reinforcement learning [32], and adversarial learning [18].   
Multi-agent reinforcement learning (MARL) extends traditional reinforcement learning frameworks   
to the situation where multiple autonomous agents interact and make decisions concurrently [26].   
In multi-agent systems, each agent learns optimal behavior through an iterative process, that is   
interacting with both the environment and other agents, receiving the reward based on its actions, and   
dynamically adapting its policy to maximize long-term returns. This paradigm captures the complex   
inter-dependencies that emerge when multiple learning agents co-evolve within a shared environment.   
From the perspective of game theory, MARL can be modeled as a repeated game [38], where agents   
iteratively select strategies based on current information, receive rewards from the environment,   
and update their strategies accordingly. This process generates a strategy path $( s ^ { t } ) _ { t = 0 } ^ { T }$ , where $s ^ { t }$   
represents the joint strategy profile at time step $t$ . The concept of equilibrium is particularly crucial   
in MARL [23, 12, 14], as it represents a stable case where no agent can improve its payoff through   
unilateral deviation.   
A fundamental question in MARL algorithm design is whether decentralized strategy updates can   
lead the joint strategy to converge to an equilibrium. While this problem has been extensively studied   
[36, 22, 27], it remains incompletely resolved [42]. Formally, each agent $i$ updates its strategy   
according to a revision function: $s _ { i } ^ { t + \bar { 1 } } = f _ { i } ( s ^ { u } ) _ { 0 \leq u \leq t }$ , where $f _ { i }$ maps the history of joint strategies   
$( s ^ { u } ) _ { 0 \leq u \leq t }$ to a new strategy for agent $i$ [5, 28]. A widely adopted principle in MARL algorithms is   
"win-stay, lose-shift" [9, 41, 16, 29]: If the agent’s current strategy is a best response to other agents,   
it will maintain this strategy, otherwise it will switch to a substitute strategy. The path generated   
under this principle is termed a satisficing path [40].

This paper studies the satisficing path from a topological perspective, aiming to provide new insights into the following core question:

Question 1. Under what conditions does the game possess the property that for any initial joint strategy s, there exists a finite-length satisficing path $( s ^ { t } ) _ { 0 \leq t \leq T }$ where $s ^ { 0 } = s$ and $s ^ { T }$ is an equilibrium?

This paper adopts a general theoretical perspective to study the existence conditions of the satisficing   
path, rather than analyzing specific revision methods or update functions $f _ { i }$ . The primary objective   
is to establish fundamental theory of the satisficing path, thereby providing essential theoretical   
supplements to the existing research on convergence problems in MARL.   
Contributions. 1) The novel concept of the grouped satisficing path is introduced and formalized   
in this paper, extending the existing concept of the satisficing path. 2) The structural property of   
the local minimum in the grouped satisficing path is studied from a topological perspective. 3) The   
sufficient conditions for the existence of the grouped satisficing path are established, thus solving   
Question 1. In brief, if the pure strategy game satisfies that $a$ ) the strategy set is convex and compact,   
$b$ ) the payoff function is continuous and partially analytic, $c$ ) and any sub-game has an equilibrium,   
then for any initial joint strategy $s$ , there exists a finite-length grouped satisficing path $\bar { ( } s ^ { t } ) _ { 0 \leq t \leq T }$   
where $s ^ { 0 } = s$ and $s ^ { T }$ is an equilibrium. More details can be found in Theorem 2.   
This paper is organized as follows. In Section 2, the theoretical framework is established by formaliz  
ing key concepts, and the novel concept of the grouped satisficing path is introduced. In Section 3,   
the topological properties of the grouped satisficing path are studied, and the conditions for reducing   
the original game to a sub-game are proposed in Theorem 1. In Section 4, the sufficient conditions   
for the existence of the grouped satisficing path are established in Theorem 2, and this theorem is   
applied in various situations, such as continuous games, $N$ -player games and finite-state Markov   
games. In Section 5, the theoretical implications of these results are examined, some open problems   
are discussed for future study, and finally there comes to the conclusion of this paper.   
Related work. MARL literature contains numerous algorithms employing the iterative strategy   
adjustment mechanism. Among these, the fictitious play algorithm [7] stands as a paradigmatic   
example that has profoundly influenced development in this field. While extensive research has   
investigated the convergence properties of the fictitious play and its variants [25, 33], existing   
theoretical guarantees remain limited to the specific game structures [3, 34, 35]. The fundamental   
connection between the fictitious play and the best-response dynamics deserves particular attention.   
Agents select optimal strategies based on historical joint strategy profiles, and iteratively update their   
strategies according to the payoff they get. This formulation suggests that the analytical tools from   
dynamical systems [4, 11, 37] may offer valuable insights for analyzing the convergence of the game   
and establishing broader convergence guarantees.   
The dynamics of the agent strategies in discrete time settings can be formally characterized by the   
update equation $\boldsymbol { s } _ { i } ^ { t + 1 } = f _ { i } ( \boldsymbol { s } ^ { t } )$ , where each agent $i$ ’s strategy selection at time step $t + 1$ is determined   
by an update rule $f _ { i }$ that depends on the previous joint strategy profile $s ^ { t }$ . The seminal work in [20]   
established a crucial theoretical limitation: for continuous strategy dynamics, if any update function   
satisfies the regularity conditions and the uncoupled-ness property, the game will be not convergent   
to an equilibrium generically. Subsequent research [2, 30] has further generalized these impossibility   
results. However, notable positive convergence results emerge when introducing stochastic elements.   
For instance, the agent has the probability to update the strategy randomly when it deviates from the   
best response [21, 15]. The conditional update mechanism has the similar results as the stochastic   
one [10, 8, 9].   
The concept of the satisficing path, originally introduced in [40], enforces a key behavior constraint:   
agents achieving best response must maintain their current strategies, while allowing unconstrained   
adaptation for other agents. This flexible formulation naturally encompasses a broad spectrum of   
learning algorithms [1, 6]. Prior research has progressively established existence guarantees for the   
satisficing path: [16] proved the path existence in two-player games; [40] extended these results to   
$N$ -player symmetric Markov games; [39] demonstrated universal existence in all $N$ -player games.   
While these works successfully addressed the existence problem in specific game classes, they could   
not extend the result in a more general form directly. So it is necessary to establish the theory of the   
satisficing path. Since this paper studies the topological properties of the satisficing path, the existence   
results can be successfully extended in a more general form, thus solving Question 1 mentioned   
above.

# 2 Formalization

# 2.1 Pure strategy game

A pure strategy game is defined by the tuple $G = ( I , ( S _ { i } ) _ { i \in I } , ( g _ { i } ) _ { i \in I } )$ , where $I$ represents the set   
of players, $S _ { i }$ denotes the pure strategy set of player $i$ , and $\begin{array} { r } { g _ { i } : \prod _ { i \in I } S _ { i } \to \mathbb { R } } \end{array}$ is player $i$ ’s payoff   
function. In this paper, only pure strategy games with finite players $\left| I \right| < \infty )$ are discussed.   
Each player $i \in I$ independently chooses a strategy $s _ { i } \in S _ { i }$ from their respective strategy set. The   
resulting payoff for player $i$ is determined by the joint strategy profile $( s _ { 1 } , s _ { 2 } , \ldots , s _ { | I | } )$ through the   
payoff function $g _ { i }$ . Each rational player aims to maximize their individual payoff. However, since the   
payoff functions depend on all players’ simultaneous strategy choices and the players cannot directly   
control others’ strategy selections, the optimal strategic choice reduces to selecting a best response to   
111 the current strategies of other players.

12 Definition 1. In a pure strategy game $G$ , the $\epsilon$ -best response correspondence for player $i$ given opponents’ strategy profile 13 $s _ { - i }$ is

$$
B R _ { \epsilon } ( s _ { - i } ) = \left\{ s \in S _ { i } \mid g _ { i } ( s , s _ { - i } ) \geq \operatorname* { s u p } _ { t \in S _ { i } } g _ { i } ( t , s _ { - i } ) - \epsilon \right\} .
$$

where 114 $\begin{array} { r } { s _ { - i } \in \prod _ { j \neq i , j \in I } S _ { j } } \end{array}$ denotes the joint strategy of all other players. In particular, the notation 115 $B R _ { \epsilon }$ will be simplified to $B R$ when $\epsilon = 0$ .

In the absence of additional constraints, the best response set $B R ( s _ { - i } )$ may be empty, as the existence   
of a maximum point in $S _ { i }$ is not guaranteed a priori. When $S _ { i }$ is closed and compact, and $g _ { i }$ is   
continuous, then $B R ( s _ { - i } )$ is non-empty for all $s _ { - i } \in S _ { - i }$ . For $\epsilon > 0$ , the existence condition can be   
relaxed to the boundedness of $g _ { i }$ .

Definition 2. In a pure strategy game 120 $G$ , a joint strategy $s \in \Pi _ { i \in I } S _ { i }$ is called an ϵ-equilibrium if

$$
\forall i \in I , \quad s _ { i } \in B R _ { \epsilon } ( s _ { - i } ) .
$$

A mixed strategy game can also be defined by the tuple $G = ( I , ( S _ { i } ) _ { i \in I } , ( g _ { i } ) _ { i \in I } )$ . However, each player $i \in I$ can choose a Borel probability measure over $S _ { i }$ as a mixed strategy. Construct a pure strategy game $\tilde { G } = ( I , ( \Delta S _ { i } ) _ { i \in I } , ( \tilde { g } _ { i } ) _ { i \in I } )$ , where $\Delta S _ { i }$ denotes the set of all Borel probability measures over $S _ { i }$ , and

$$
\tilde { g } _ { i } : \prod _ { i \in I } \Delta S _ { i } \to \mathbb { R } , \quad \prod _ { i \in I } \sigma _ { i } \mapsto \int _ { s \in \prod _ { i \in I } S _ { i } } g _ { i } ( s ) \prod _ { i \in I } d \sigma _ { i } ( s _ { i } ) .
$$

Then the equilibrium of the pure strategy game $\tilde { G }$ is equivalent to the Nash equilibrium of the original   
mixed strategy game $G$ . This fundamental equivalence justifies referring to $\tilde { G }$ as the extended game   
of $G$ .   
Remark 1. The class of pure strategy games exhibits strict representational generality over mixed   
strategy games. For any mixed strategy game, there exists an equivalent pure strategy game which   
possesses the same properties of the original game. However, a pure strategy game admits a mixed   
strategy representation only if each strategy set can be embedded into the set of Borel probability   
measures over some base space.

# 129 2.2 Grouped satisficing path

In a pure strategy game $G$ , the repeated play generates a path $( s ^ { t } ) _ { t \geq 0 }$ where $s ^ { t }$ represents the joint strategy profile at time step $t$ . In this paper, only discrete time strategy dynamics $( t \in \mathbb { N } )$ is discussed.

Definition 3. In a pure strategy game 32 $G$ , the strategy path $( s ^ { t } ) _ { t \geq 0 }$ is called an $\epsilon$ -satisficing path if

$$
\forall t \geq 0 , \quad \forall i \in I , \quad s _ { i } ^ { t } \in B R _ { \epsilon } ( s _ { - i } ^ { t } ) \quad \Rightarrow \quad s _ { i } ^ { t + 1 } = s _ { i } ^ { t } .
$$

The definition of the satisficing path is an accurate description of the "win-stay, lose-shift" principle.   
If the player’s strategy is a current best response, it will maintain this strategy in the next time step,   
otherwise it will explore an alternative strategy to replace it. This principle is fundamental in the   
136 MARL algorithm design, since it ensures the stability at the equilibrium.

37 Definition 4. In a pure strategy game $G$ , the group set $P$ is a partition of the player set $I$ satisfying:

$\begin{array} { r } { I . \bigcup _ { p \in P } p = I . } \end{array}$ 2. For any $p , q \in P$ , if $\dot { p } \neq q$ , then $p \cap q = \emptyset$ .

A strategy path 140 $( s ^ { t } ) _ { t \geq 0 }$ is called a grouped $\epsilon$ -satisficing path with respect to the group set $P$ if

$$
\forall t \geq 0 , \quad \forall p \in P , \quad ( \forall i \in p , s _ { i } ^ { t } \in B R _ { \epsilon } ( s _ { - i } ^ { t } ) ) \quad \Rightarrow \quad ( \forall i \in p , s _ { i } ^ { t + 1 } = s _ { i } ^ { t } ) .
$$

The definition of the grouped satisficing path extends the "win-stay, lose-shift" principle to player   
groups. The basic decision unit is a group of players rather than individual players. If all players   
in a group achieve best responses, all of them will maintain their strategies. When each group   
contains exactly one player $\mathbf { \bar { \mathcal { P } } } = \{ \{ i \} | i \in I \} $ ), the grouped satisficing path reduces to the classical   
145 satisficing path.

Remark 2. The grouped satisficing path constitutes a proper generalization of the classical satisficing path. Obviously, any satisficing path can be viewed as a grouped satisficing path, while the converse proposition is not always true. The grouped satisficing path serves as an essential technical tool for proving the existence of the satisficing path in finite-state Markov games.

50 Definition 5. In a pure strategy game $G$ with the group set $P$ , for a joint strategy $s$ , the ϵ-best   
1 response group count with respect to s is

$$
N _ { \epsilon } ( s ) = | \{ p \in P \mid \forall i \in p , s _ { i } \in B R _ { \epsilon } ( s _ { - i } ) \} | .
$$

In particular, the notation $N _ { \epsilon }$ will be simplified to $N$ when $\epsilon = 0$ .

The $\epsilon$ -best response group count constitutes an important characteristic of joint strategy profiles. It quantifies the number of player groups where all members simultaneously achieve $\epsilon$ -best responses.

155 Definition 6. In a pure strategy game $G$ with the group set $P$ , for a joint strategy $s$ , the admissible   
56 subsequent joint strategy set with respect to s is

$$
T _ { \epsilon } ( s ) = \left\{ s ^ { \prime } \in \prod _ { i \in I } S _ { i } \mid ( s , s ^ { \prime } ) s a t i s f i e s t h e g r o u p e d \epsilon - s a t i s f i c i n g d y n a m i c s \right\} .
$$

$N _ { \epsilon } ( s )$ is called a local minimum if

$$
N _ { \epsilon } ( s ) \leq \operatorname* { i n f } _ { t \in T _ { \epsilon } ( s ) } N _ { \epsilon } ( t ) .
$$

$N _ { \epsilon } ( s )$ is called a local maximum if

$$
N _ { \epsilon } ( s ) \geq \operatorname* { s u p } _ { t \in T _ { \epsilon } ( s ) } N _ { \epsilon } ( t ) .
$$

In particular, the notation $T _ { \epsilon }$ will be simplified to $T$ when $\epsilon = 0$ .

The admissible subsequent joint strategy set $T _ { \epsilon } ( s )$ is the set of all possible joint strategies following   
$s$ without violating the property of grouped $\epsilon$ -satisficing path. So, a local minimum means that any   
admissible joint strategy cannot decrease the $\epsilon$ -best response group count, while maximum means   
cannot increase.

Definition 7. $\{ X _ { i } \} _ { i \in I }$ is a family of sets, where each $X _ { i }$ is a convex set in a topological vector space. A function $\begin{array} { r } { f : \prod _ { i \in I } X _ { i } \to \mathbb { R } } \end{array}$ is called a partially analytic function if for any $i \in I$ , $p _ { i } , q _ { i } \in X _ { i }$ , $\begin{array} { r } { x _ { - i } \in \prod _ { j \neq i , j \in I } X _ { j } , } \end{array}$ , the function

$$
g : [ 0 , 1 ] \to \mathbb { R } , \quad a \mapsto f ( a p _ { i } + ( 1 - a ) q _ { i } , x _ { - i } ) .
$$

is real analytic with respect to $a$ .

Definition 8. In a pure strategy game $G$ with the group set $P$ , it is called that any sub-game has an equilibrium, if for any group subset $Q \subset P$ and any joint strategy $s$ , the sub-game $\tilde { G } =$ $( J , ( S _ { i } ) _ { i \in J } , ( \tilde { g } _ { i } ) _ { i \in J } )$ admits at least one equilibrium, in which $J = \textstyle \bigcup _ { p \in Q } p$ and

$$
\tilde { g } _ { i } : \prod _ { j \in J } S _ { j } \to \mathbb { R } , \quad w \mapsto g _ { i } \big ( s _ { - J } , w \big ) .
$$

166 where $s _ { - J } = ( s _ { k } ) _ { k \in I \backslash J }$ denotes the strategy profile of all players outside $J$ .

67 The two definitions above represent the two conditions stated in Theorem 2.

# 3 Topological properties

Two fundamental lemmas are established first and form the basis for proving the main theorem of the topological properties. Lemma 1 provides the necessary conditions for a joint strategy $s$ to be a local minimum of the $\epsilon$ -best response group count $N _ { \epsilon } ( s )$ . Lemma 2 characterizes the local behavior of the best response when a neighborhood of the strategy satisfies optimality conditions. Theorem 1 provides the sufficient conditions under which the original game can be reduced to a well-defined sub-game. Due to space limitation, the detailed proofs of Lemma 1 and Lemma 2 appear in the Appendix.

Lemma 1. In a pure strategy game $G$ with the group set $P$ , suppose that each $S _ { i }$ is compact and Hausdorff, and each $g _ { i }$ is continuous. For a joint strategy $s$ , if $N _ { \epsilon } ( s )$ is a local minimum, then

1. either for any joint strategy $t \in T _ { \epsilon } ( s )$ in the admissible subsequent joint strategy set, the group which has achieved $\epsilon$ -best response in s will still achieve $\epsilon$ -best response in $t$ ,

. or there exists a non-empty open set $O \subset T _ { \epsilon } ( s )$ in the admissible subsequent joint strategy set satisfying:

(a) For any joint strategy $t \in O$ , there exists a group which has achieved ϵ-best response in s will not achieve $\epsilon$ -best response in $t$ .   
(b) There exists a group which has not achieved $\epsilon$ -best response in s will achieve ϵ-best response in any joint strategy $t \in O$ .

Lemma 1 is proved in Appendix A.1. In brief, the main idea is to construct a finite family of closed   
sets covering a given Baire space, and then to discuss different cases about this Baire space. The   
188 property of the Baire space will be used in this proof.

Lemma 2. In a pure strategy game $G$ , suppose that each $S _ { i }$ is a convex set in a topological vector space, and each $g _ { i }$ is partially analytic. If there exists a non-empty open set $\textstyle U \subset \prod _ { i \in I } S _ { i }$ satisfying that the player $k$ achieves best response in any joint strategy $s \in U$ , then the player $k$ will achieve best response in any joint strategy $\textstyle s \in \prod _ { i \in I } S _ { i }$ .

93 Lemma 2 is proved in Appendix A.2. In brief, the main idea is to construct a function to describe the   
property of the best response, and then to use the analytic extendability to expand the best response   
set step by step.

Theorem 1. In a pure strategy game $G$ with the group set $P$ , suppose that each $S _ { i }$ is a convex compact set in a topological vector space, and each $g _ { i }$ is continuous and partially analytic. Then, for a joint strategy s, the best response group count $N ( s )$ is a local minimum, if and only if, for any joint strategy $t \in T ( s )$ in the admissible subsequent joint strategy set, the group which has achieved best response in s still achieves best response in $t$ .

Proof. Sufficiency. Suppose for any joint strategy $t \in T ( s )$ in the admissible subsequent joint strategy set, the group which has achieved best response in $s$ still achieves best response in $t$ . This means that for any joint strategy $t \in T ( s )$ , there exists $N ( s ) \leq N ( t )$ . As a result

$$
N ( s ) \leq \operatorname* { i n f } _ { t \in T ( s ) } N ( t ) .
$$

201 According to Definition 6, $N ( s )$ is a local minimum.

Necessity. Suppose the best response group count 2 $N ( s )$ is a local minimum. Since $S _ { i }$ is the set in a topological vector space, 3 $S _ { i }$ is Hausdorff. According to Lemma 1, there are two cases:

04 1) Either any joint strategy in $T ( s )$ makes the group which has achieved best response in $s$ will still   
achieve,   
2) or there exists a non-empty open set $O \subset T ( s )$ satisfying: $a$ ) There exists one group $p \in P$ which   
has not achieved best response in $s$ will achieve in any joint strategy in the set $O , b$ ) and for any joint   
strategy in the set $O$ , there exists one group which has achieved best response in $s$ will not achieve.

Suppose case 2) is valid. Assume group $p _ { 1 } , \ldots , p _ { n }$ do not achieve best response in $s$ . Obviously, $p \in \{ p _ { 1 } , \ldots , p _ { n } \}$ . Let $\textstyle Q = \bigcup _ { i = 1 } ^ { n } p _ { i }$ . Construct a pure strategy game $\tilde { G } = ( Q , ( S _ { i } ) _ { i \in Q } , ( \tilde { g } _ { i } ) _ { i \in Q } ) . \tilde { g }$ $\tilde { g } _ { i }$ is defined as

$$
\tilde { g } _ { i } : \prod _ { j \in Q } S _ { j } \to \mathbb { R } , \quad w \mapsto g _ { i } ( s _ { - Q } , w ) .
$$

where $s _ { - Q } = ( s _ { k } ) _ { k \in I \backslash Q }$ denotes the strategy profile of all players outside $Q$ .

Let the projection of the non-empty open set $O$ onto $\Pi _ { j \in Q } S _ { j }$ be $O _ { p }$ . Obviously, $O _ { p }$ is a non-empty   
open set, and each player in $p$ achieves best response in any joint strategy in $O _ { p }$ . Each $\tilde { g } _ { i }$ is actually a   
constrained function of $g _ { i }$ . Since $g _ { i }$ is continuous and partially analytic, $\tilde { g } _ { i }$ is also continuous and   
partially analytic. According to Lemma 2, for any joint strategy $\textstyle w \in \prod _ { j \in Q } S _ { j }$ , each player in $p$ will   
achieve best response in $w$ . In turn, for any joint strategy $t \in T ( s )$ , each player in $p$ will achieve best   
response in $t$ in the game $G$ . However, $s \in T ( s )$ . This is a contradiction with the fact that group $p$   
216 does not achieve best response in $s$ . So case 2) is invalid.   
17 As a result, for any joint strategy $t \in T ( s )$ , the group which has achieved best response in $s$ will still   
8 achieve best response in $t$ . □

Remark 3. This theorem actually establishes a formal reduction principle for games under grouped satisficing dynamics. When there exists a group where all players fix their strategies as their strategies are always best responses in all possible grouped satisficing paths, it can be formally treated as part of the environmental dynamics, reducing the original game to a sub-game.

# 4 Existence of paths

In this section, a fundamental existence theorem for the grouped satisficing path in pure strategy games is established. Theorem 2 provides the sufficient conditions for the existence, thus answering Question 1. This theorem yields four corollaries, each addressing distinct game-theoretic scenarios. Due to space limitation, the proofs of these corollaries appear in the Appendix.

Proposition 1. In a pure strategy game $G$ with the group set $P$ , for any infinite-length path $( s ^ { t } ) _ { t \geq 0 }$ , there exists an index v satisfying $N _ { \epsilon } ( s ^ { v } ) = \operatorname* { i n f } _ { t \geq 0 } N _ { \epsilon } ( s ^ { t } )$ .

Proposition 1 is proved in Appendix B.

Theorem 2. In a pure strategy game $G$ with the group set $P$ , suppose that each $S _ { i }$ is a convex compact set in a topological vector space, each $g _ { i }$ is continuous and partially analytic, and any sub-game has an equilibrium. Then for any initial joint strategy $s$ , there exists a finite-length grouped satisficing path $( s ^ { t } ) _ { t = 0 } ^ { T }$ where $s ^ { 0 } = s$ and $s ^ { T }$ is an equilibrium.

Proof. Any finite-length grouped satisficing path $( s ^ { t } ) _ { t = 0 } ^ { T }$ where $s ^ { T }$ is an equilibrium can be extended to an infinite-length one. Construct a path

$$
a ^ { t } = \left\{ \begin{array} { l l } { s ^ { t } , } & { 0 \leq t \leq T , } \\ { s ^ { T } , } & { T < t . } \end{array} \right.
$$

Since $a ^ { T }$ is an equilibrium, any player will not change their strategies any longer. So $( a ^ { t } ) _ { t \geq 0 }$ is a   
grouped satisficing path.

Construct a set

$A = \{ ( s ^ { t } ) _ { t \geq 0 } ~ | ~ ( s ^ { t } ) _ { t \geq 0 }$ is an infinite-length grouped satisficing path, $s ^ { 0 } = s \}$ .

According to Proposition 1, each path $( s ^ { t } ) _ { t \geq 0 }$ has a minimum of $N ( s ^ { t } )$ . Construct $| P | + 1$ sets

$$
A _ { i } = \left\{ ( s ^ { t } ) _ { t \geq 0 } \in A \ | \ \operatorname* { i n f } _ { t \geq 0 } N ( s ^ { t } ) = i \right\} , \quad i = 0 , \ldots , | P | .
$$

There exists a minimum $k$ satisfying $A _ { k } \ \ne \ \mathcal { O }$ . Choose a path $( s ^ { t } ) _ { t \geq 0 } \in A _ { k }$ . According to Proposition 1, there exists 238 $s ^ { x } \in ( s ^ { \bar { t } } ) _ { t \geq 0 }$ satisfying $N ( s ^ { x } ) = k$ .

Reduction to absurdity needs to be used here.

Assume on the contrary that $N ( s ^ { x } )$ is not local minimum. By Definition 6, there exists a joint strategy $u$ following $s ^ { x }$ without violating the property of grouped satisficing path, and satisfying $N ( u ) \leq N ( s ^ { x } ) - 1$ . Construct a path

$$
a ^ { t } = \left\{ \begin{array} { l l } { s ^ { t } , } & { 0 \leq t \leq x , } \\ { u , } & { x < t . } \end{array} \right.
$$

It is sure that $( a ^ { t } ) _ { t = 0 } ^ { x + 1 }$ is a grouped satisficing path. For any $t > x$ , any player does not change their strategies, which means that the group who has achieved best response will still achieve, and the group who not will still not. So $( a ^ { t } ) _ { t \geq 0 }$ is actually a grouped satisficing path. As a result, $( a ^ { t } ) _ { t \geq 0 } \in \bar { A }$ , and

$$
\operatorname* { i n f } _ { t \geq 0 } N ( a ^ { t } ) \leq N ( u ) \leq N ( s ^ { x } ) - 1 = k - 1 .
$$

So $( a ^ { t } ) _ { t \geq 0 }$ belongs to one of $A _ { 0 } , \ldots , A _ { k - 1 }$ , which is a contradiction with the fact that $A _ { k }$ is the   
non-empty set with the minimal index.

As a result, 242 $N ( s ^ { x } )$ is local minimum.

According to Theorem 1, any possible subsequent joint strategy following $s ^ { x }$ without violating the property of grouped satisficing path, makes the group which has achieved best response in $s ^ { x }$ will still achieve. Assume group $p _ { 1 } , \ldots , p _ { n }$ do not achieve best response in $s ^ { x }$ . Let $\textstyle J = \bigcup _ { i = 1 } ^ { n } p _ { i }$ Construct a pure strategy game $\tilde { G } = ( J , ( S _ { i } ) _ { i \in J } , ( \tilde { g } _ { i } ) _ { i \in J } )$ . $\tilde { g } _ { i }$ is defined as

$$
\tilde { g } _ { i } : \prod _ { j \in J } S _ { j } \to \mathbb { R } , \quad w \mapsto g _ { i } \big ( s _ { - J } ^ { x } , w \big ) .
$$

where 243 $s _ { - J } ^ { x } = ( s _ { k } ^ { x } ) _ { k \in I \setminus J }$ denotes the strategy profile of all players outside $J$ .

By assumption that any sub-game has an equilibrium, there exists an equilibrium 244 $w$ in $\tilde { G }$ . Let

$$
u = ( s _ { - J } ^ { x } , w ) \in \prod _ { i \in I } S _ { i } .
$$

Construct a finite path

$$
a ^ { t } = { \left\{ \begin{array} { l l } { s ^ { t } , } & { 0 \leq t \leq x , } \\ { u , } & { t = x + 1 . } \end{array} \right. }
$$

Obviously, $\{ a ^ { t } \} _ { t = 0 } ^ { x + 1 }$ is a grouped satisficing path, and $\boldsymbol { a } ^ { 0 } \ = \ s$ . According to heorem and   
Equation 9, the group which has achieved best response in $s ^ { x }$ will still achieve in $u$ $w$   
equilibrium in the sub-game $\tilde { G }$ , the group which has not achieved best response in $s ^ { x }$ will achieve in   
$u$ . As a result, $a ^ { x + 1 }$ is an equilibrium in $G$ .

$\{ a ^ { t } \} _ { t = 0 } ^ { x + 1 }$ is a finite-length grouped satisficing path where $a ^ { 0 } = s$ and $a ^ { x + 1 }$ is an equilibrium.



This theorem establishes fundamental guarantees for the existence of the grouped satisficing path.   
While the technical requirements may appear stringent, they are in fact satisfied in numerous practical   
scenarios.

Corollary 1. In a pure strategy game $G$ , suppose that each $S _ { i }$ is a convex compact set in a topological vector space and each $g _ { i }$ is continuous and partially analytic. If for any $g _ { i }$ and any $\begin{array} { r } { s _ { - i } \in \prod _ { j \neq i , j \in I } S _ { j } } \end{array}$ , the function $g _ { i } ( s _ { i } , s _ { - i } )$ is quasi-convex with respect to $s _ { i }$ , then for any initial joint strategy s, there exists a finite-length satisficing path $( s ^ { t } ) _ { t = 0 } ^ { T }$ where $s ^ { 0 } = s$ and ${ \bf \Pi } _ { { s } } T$ is an equilibrium.

Corollary 1 is proved in Appendix B.1. The assumption of this corollary is almost the same as Theorem 2, so only need to check whether any sub-game has an equilibrium.

Corollary 2. In a mixed strategy game $G$ , suppose that each $S _ { i }$ is a finite set. Then for any initial joint mixed strategy $\sigma$ , there exists a finite-length satisficing path $( \sigma ^ { t } ) _ { t = 0 } ^ { \cal T }$ where $\sigma ^ { 0 } = \sigma$ and $\sigma ^ { T }$ is $a$ mixed equilibrium.

Corollary 2 is proved in Appendix B.2. This corollary is essentially the main result of [39], where   
the authors directly construct a satisficing path and prove that the equilibrium is a limit point under   
certain conditions. In contrast, the existence of such a path in $N$ -player games is established as a   
corollary of Theorem 2 in this paper. Thus, the proof reduces to verifying that the conditions of   
Theorem 2 hold in this setting.   
A stationary mixed strategy stochastic game is defined as $G = ( I , ( S _ { i } ) _ { i \in I } , X , P , ( g _ { i } ) _ { i \in I } , ( \gamma _ { i } ) _ { i \in I } ) .$ $I$   
is the set of players. $S _ { i }$ is the strategy set of player $i$ . $X$ is the state set. $\begin{array} { r } { P : X \times \prod _ { i \in I } S _ { i } \to \Delta X } \end{array}$   
is the transition probability function, mapping the current state and joint strategy to a probability   
distribution over the next states. $g _ { i } : X \times \bar { \prod _ { i \in I } S _ { i } }  \mathbb { R }$ is the payoff function for player $i$ . $\gamma _ { i } \in [ 0 , 1 )$   
is the discount factor for player $i$ . In this paper, only stationary mixed strategy stochastic games with   
finite players and finite states $( | I | < \infty , | X | < \infty )$ are discussed.   
Each player $i$ selects a strategy from $S _ { i }$ according to the probability distribution $\pi _ { i } ( x )$ . The mapping   
$\pi _ { i } : X \to \Delta S _ { i }$ is called a stationary mixed strategy, as the player $i$ ’s strategy depends solely on the   
current state $x \in X$ . Given a joint strategy profile $s$ and the current state $x$ , two events occur: 1)   
Each player $i$ receives an immediate payoff $g _ { i } ( x , s ) , 2 )$ and the game transits to a new state $x ^ { \prime } \in X$   
with probability $P ( x , s ) ( x ^ { \prime } )$ . As the game progresses, the player $i$ obtains a sequence of payoffs   
$\{ g _ { i } ( \bar { x ^ { t } } , s ^ { t } ) \} _ { t \geq 0 }$ and aims to maximize the discounted average payoff $\begin{array} { r } { \sum _ { t \geq 0 } \gamma _ { i } ^ { t } g _ { i } ( \bar { x ^ { t } } , s ^ { t } ) } \end{array}$ .

A joint mixed strategy profile $( \pi _ { i } ) _ { i \in I }$ constitutes a mixed equilibrium in a stationary mixed strategy stochastic game, if for any initial state $x \in X$ and any player $i \in I$ , there exists

$$
\mathbb { E } _ { ( \pi _ { j } ) _ { j \in I } } \left[ \sum _ { t \ge 0 } \gamma _ { i } ^ { t } g _ { i } ( x ^ { t } , s ^ { t } ) \mid x _ { 0 } = x \right] = \operatorname* { s u p } _ { \sigma _ { i } \in ( \Delta S _ { i } ) ^ { X } } \mathbb { E } _ { \sigma _ { i } , ( \pi _ { j } ) _ { j \neq i , j \in I } } \left[ \sum _ { t \ge 0 } \gamma _ { i } ^ { t } g _ { i } ( x ^ { t } , s ^ { t } ) \mid x _ { 0 } = x \right] .
$$

Corollary 3. In a stationary mixed strategy stochastic game $G$ , suppose that each $S _ { i }$ is a finite set.   
Then for any initial stationary joint mixed strategy $\sigma$ , there exists a finite-length satisficing path   
$( \sigma ^ { t } ) _ { t = 0 } ^ { T }$ where $\sigma ^ { 0 } = \sigma$ and $\sigma ^ { T }$ is a mixed equilibrium.   
Corollary 3 is proved in Appendix B.3. The proof of this corollary is a bit technically demanding.   
First, it is necessary to transform the stationary mixed strategy stochastic game with the satisficing   
path into a pure strategy game with the grouped satisficing path. Since it contains infinite summation,   
the validity of the operation must be verified. Then, check whether each condition listed in Theorem 2   
is satisfied. Contraction mapping theorem will be used in this proof.   
Similar to the definition of the stationary mixed strategy stochastic game, a $k$ -step mixed strategy   
stochastic game can also be defined as $\bar { G } = ( I , ( S _ { i } ) _ { i \in I } , X , P , ( g _ { i } ) _ { i \in I } , ( \gamma _ { i } ) _ { i \in I } )$ , where all compo  
nents are defined analogously to the stationary mixed strategy case. However, each player $i$ selects a   
strategy from $S _ { i }$ according to the probability distribution $\overleftrightarrow { \pi } _ { i } ( x , s ^ { - 1 } , \ldots , s ^ { - k } )$ , where $s ^ { - t }$ denotes   
the joint strategy profile from $t$ steps before the current time step. So $\begin{array} { r } { \pi _ { i } : X \times ( \prod _ { i \in I } S _ { i } ) ^ { k } \to \Delta S _ { i } } \end{array}$   
Similarly, only the case with finite players and finite states $( | I | < \infty , | X | < \infty )$ ) is discussed in this   
paper.

Corollary 4. In a $k$ -step mixed strategy stochastic game $G$ , suppose that each $S _ { i }$ is a finite set. Then for any initial stationary joint mixed strategy $\sigma$ , there exists a finite-length satisficing path $( \sigma ^ { t } ) _ { t = 0 } ^ { T }$ where $\sigma ^ { 0 } = \sigma$ and $\sigma ^ { T }$ is a mixed equilibrium.

Corollary 4 is proved in Appendix B.4. The main idea is to find a bijection between the paths in the   
$k$ -step mixed strategy stochastic game and the paths in a stationary mixed strategy stochastic game,   
300 and then to use Corollary 3 to come to the conclusion.

Remark 4. Usually, the game in Corollary 1 is called a continuous game, where the payoff function is continuous. The game in Corollary 2 is termed an $N$ -player game, while the games in Corollary 3 and Corollary 4 are referred to as finite-state Markov games. Corollary 2 establishes the existence of the satisficing path in a general $N$ -player game, Corollary 3 proves its existence in a standard reinforcement learning setting, and Corollary 4 extends the result to the reinforcement learning with historical records.

# 5 Discussion

Theorem 1 establishes a framework in which the grouped satisficing paths of the original game can be derived from those of the sub-game. Consequently, the original game can be effectively reduced to its sub-game. Theorem 2 presents sufficient conditions for the existence of the grouped satisficing paths connecting any initial joint strategy to an equilibrium. Thus, Theorem 2 provides a solution to Question 1, or more precisely, it identifies sufficient conditions for the question’s resolution.

3 From a practical standpoint, the corollaries of Theorem 2 may be more valuable. Specifically, these results demonstrate that: 1) In any $N$ -player game, there exists satisficing paths from any initial joint mixed strategy to a mixed equilibrium. 2) For any finite-state Markov game, whether stationary or finite-step, such satisficing paths always exist from arbitrary initial joint mixed strategy to a mixed equilibrium. This implies that in reinforcement learning with finite states, satisficing paths are guaranteed to exist regardless of whether agents select strategies based on current states or finite history records.

These results remain consistent with the findings in [30, 20, 2], as no restriction is imposed on the update functions in this paper, unlike the regularity or uncoupled-ness conditions required in those studies. Conversely, these results provide theoretical support for stochastic update approaches [21, 15]. The existence of finite-length satisficing paths from any initial joint strategy to an equilibrium implies that: it is possible for stochastic algorithms to start from arbitrary initial joint strategy and to stop at some equilibrium, while keeping the player who has achieved best response not change its strategy.

Open problems. Theorem 2 establishes a sufficient condition for the existence of grouped satisficing paths connecting any initial joint strategy to an equilibrium. This naturally leads to a question: What constitutes a necessary condition for Question 1? More fundamentally, what is a complete necessary and sufficient condition to characterize the existence of such grouped satisficing paths?

Theorem 1 and Theorem 2 are established under the condition of $\epsilon = 0$ . This restriction arises because the application of the analytic extendability in Lemma 2 requires the use of certain equations to characterize best responses. So, does Theorem 2 remain valid when $\epsilon > 0 2$

Grouped satisficing paths do not impose any neighborhood restriction on strategy selection for the player who has not achieved best response. This raises a question: does Theorem 2 remain valid when non-best-responding players are restricted to select strategies only from a neighborhood of their current strategies? More fundamentally, what restriction can be imposed on non-best-responding players while still preserving the existence of grouped satisficing paths?

Conclusion. In brief, if the pure strategy game satisfies that $a$ ) the strategy set is convex and compact, $b$ ) the payoff function is continuous and partially analytic, $c$ ) and any sub-game has an equilibrium, then for any initial joint strategy $s$ , there exists a finite-length grouped satisficing path $\bar { ( } s ^ { t } ) _ { 0 \leq t \leq T }$ where $s ^ { 0 } = s$ and $s ^ { T }$ is an equilibrium. In particular, any $N$ -player game and any finite-state Markov game have the finite-length satisficing paths from any initial joint mixed strategy to some mixed equilibrium.

# References

[1] ARSLAN, G., AND YUKSEL, S. Decentralized q-learning for stochastic teams and games. IEEE Transactions on Automatic Control, 4 (2017), 1545–1558.   
[2] BABICHENKO, Y. Completely uncoupled dynamics and nash equilibria. Games and Economic Behavior 76, 1 (2012), 1–14.   
[3] BAUDIN, L., AND LARAKI, R. Best-response dynamics and fictitious play in identicalinterest and zero-sum stochastic games. International Conference on Machine Learning (2022), 1664–1690.

[4] BENAIM, M., HOFBAUER, J., AND SORIN, S. Stochastic approximations and differential   
inclusions. SIAM Journal on Control and Optimization, 1 (2005), 328–348.   
[5] BLUME, L. E. The statistical mechanics of strategic interaction. Games and Economic Behavior   
, 3 (1993), 387–424.   
[6] BRIAN, S., CEYHUN, E., SOUMMYA, K., AND ALEJANDRO, R. Distributed inertial best  
response dynamics. IEEE Transactions on Automatic Control 63, 12 (2018), 4294–4300.   
[7] BROWN, G. W. Iterative solution of games by fictitious play. Activity Analysis of Production   
and Allocation, 1 (1951), 374.   
[8] CANDOGAN, O., OZDAGLAR, A., AND PARRILO, P. A. Near-potential games: Geometry and   
dynamics. ACM Transactions on Economics and Computation, 2 (2013), 1–32.   
[9] CHASPARIS, G. C., ARAPOSTATHIS, A., AND SHAMMA, J. S. Aspiration learning in   
coordination games. SIAM Journal on Control and Optimization, 1 (2013), 465–490.   
[10] CHIEN, S., AND SINCLAIR, A. Convergence to approximate nash equilibria in congestion   
games. Games and Economic Behavior 71, 2 (2011), 315–327.   
[11] COLLINS, D. S., AND LESLIE, E. J. Individual q-learning in normal form games. SIAM   
Journal on Control and Optimization, 2 (2005), 495–514.   
[12] DASKALAKIS, C., FRONGILLO, R., PAPADIMITRIOU, C. H., PIERRAKOS, G., AND VALIANT,   
G. On learning algorithms for nash equilibria. In Algorithmic Game Theory - Third International   
Symposium (2010), p. 114–125.   
[13] FINK, A. Equilibrium in a stochastic $n$ -person game. Journal of Science of the Hiroshima   
University 28 (1964), 89–93.   
[14] FLOKAS, L., VLATAKIS-GKARAGKOUNIS, E. V., LIANEAS, T., MERTIKOPOULOS, P., AND   
PILIOURAS, G. No-regreet learning and mixed nash equilibria: They do not mix. Advances in   
Neural Information Processing Systems (2020), 1380–1391.   
[15] FOSTER, D., AND YOUNG, H. P. Regret testing: Learning to play nash equilibrium without   
knowing you have an opponent. Theoretical Economics 1, 3 (2006), 341–367.   
[16] GERMANO, F., AND LUGOSI, G. Global nash convergence of foster and young’s regret testing.   
Games and Economic Behavior 60, 1 (2007), 135–154.   
[17] GLICKSBERG, I. L., BURGESS, D. C. J., AND GOCHBERG, I. C. A further generalization of   
the kakutani fixed point theorem, with application to nash equilibrium points. Proceedings of   
the American Mathematical Society 3, 1 (1952).   
[18] GOODFELLOW, I. J., POUGET-ABADIE, J., MIRZA, M., XU, B., WARDE-FARLEY, D.,   
OZAIR, S., COURVILLE, A., AND BENGIO, Y. Generative adversarial nets. arXiv preprint   
arXiv:1406.2661 (2014).   
[19] GRONAUER, S., AND DIEPOLD, K. Multi-agent deep reinforcement learning: a survey.   
Artificial Intelligence Review (2021), 895–943.   
[20] HART, S., AND MAS-COLELL, A. Uncoupled dynamics do not lead to nash equilibrium. The   
American economic review 93, 5 (2003), 1830–1836.   
[21] HART, S., AND MAS-COLELL, A. Stochastic uncoupled dynamics and nash equilibrium.   
Games and Economic Behavior 57, 2 (2006), 286–303.   
[22] HSIEH, Y. G., ANTONAKOPOULOS, K., AND MERTIKOPOULOS, P. Adaptive learning in   
continuous games: Optimal regret bounds and convergence to nash equilibrium. Conference on   
Learning Theory (2021), 2388–2422.   
[23] JAFARI, A., GREENWALD, A., GONDEK, D., AND ERCAL, G. On no-regret learning, fictitious   
play, and nash equilibrium. International Conference on Machine Learning (2001), 226–233.   
[24] LARAKI, R., RENAULT, J., AND SORIN, S. Mathematical Foundations of Game Theory.   
Springer Nature Switzerland AG, Gewerbestrasse 11, 6330 Cham, Switzerland, 2019.   
[25] LESLIE, D. S., AND COLLINS, E. J. Generalised weakened fictitious play. Games and   
Economic Behavior, 2 (2006), 285–298.   
[26] LI, S. E. Reinforcement Learning for Sequential Decision and Optimal Control. Springer Nature   
Singapore Pte Ltd, 152 Beach Road, 21-01/04 Gateway East, Singapore 189721, Singapore,   
.   
[27] LU, Y. Two-scale gradient descent ascent dynamics finds mixed nash equilibria of continuous   
games: A mean-field perspective. International Conference on Machine Learning (2023),   
22790–22811.   
[28] MARDEN, J. R., AND SHAMMA, J. S. Revisiting log-linear learning: Asynchrony, com  
pleteness and payoff-based implementation. Games and Economic Behavior 75, 2 (2012),   
788–808.   
[29] MARDEN, J. R., YOUNG, H. P., ARSLAN, G., AND SHAMMA, J. S. Payoff-based dynamics   
for multiplayer weakly acyclic games. SIAM Journal on Control and Optimization 48, 1 (2009),   
373–396.   
[30] MILIONIS, J., PAPADIMITRIOU, C., PILIOURAS, G., AND SPENDLOVE, K. An   
impossibility theorem in game dynamics. Proceedings of the National Academy of Sciences   
120, 41 (2023).   
[31] NASH, J. F. The bargaining problem. Econometrica 18, 2 (1950), 155–162.   
[32] ROIJERS, D. M., VAMPLEW, P., WHITESON, S., AND DAZELEY, R. A survey of multi  
objective sequential decision-making. Journal of Artificial Intelligence Research 48, 1 (2013),   
67–113.   
[33] SAYIN, M. O. On the global convergence of stochastic fictitious play in stochastic games with   
turn-based controllers. arXiv e-prints (2022).   
[34] SAYIN, M. O., PARISE, F., AND OZDAGLAR, A. Best-response dynamics and fictitious play in   
identical-interest and zero-sum stochastic games. SIAM Journal on Control and Optimization, 4   
(2022), 2095–2114.   
[35] SAYIN, M. O., PARISE, F., AND OZDAGLAR, A. Fictitious play in markov games with single   
controller. Proceedings of the 23rd ACM Conference on Economics and Computation (2022),   
919–936.   
[36] SINGH, S., KEARNS, M., AND MANSOUR, Y. Nash convergence of gradient dynamics in   
general-sum games. Uncertainty in artificial intelligence: Sixteenth conference on uncertainty   
in artificial intelligence (2000), 541–548.   
[37] SWENSON, B., MURRAY, R., AND KAR, S. On best-response dynamics in potential games.   
SIAM Journal on Control and Optimization 56, 4 (2018), 2734–2767.   
[38] TUYLS, K., AND NOWÉ, A. Evolutionary game theory and multi-agent reinforcement learning.   
Knowledge Engineering Review 20 (2005), 63–90.   
[39] YONGACOGLU, B., ARSLAN, G., PAVEL, L., AND YÜKSEL, S. Paths to equilibrium in games.   
38th Conference on Neural Information Processing Systems (2024).   
[40] YONGACOGLU, B., ARSLAN, G., AND YÜKSEL, S. Satisficing paths and independent   
multiagent reinforcement learning in stochastic games. SIAM Journal on Mathematics of Data   
Science, 3 (2023), 745–773.   
[41] YOUNG, H. P., AND PAO, M. L. Y. Achieving pareto optimality through distributed learning.   
No. 5, p. 2753–2770.   
[42] ZHANG, K., YANG, Z., AND BAAR, T. Multi-agent reinforcement learning: A selective   
overview of theories and algorithms. Handbook of Reinforcement Learning and Control (2021).

# 444 Appendix

# 445 A Some proofs in Section 3

Definition 9. In a pure strategy game $G$ , for a joint strategy $s$ , a subset of players $J \subset I$ , and $a$   
player $i \in I \backslash J$ , the dual $\epsilon$ -best response set of player $i$ over $J$ is

$$
B R D _ { \epsilon , i } ( s , J ) = \left\{ x \in \prod _ { j \in J } S _ { j } \mid s _ { i } \in B R _ { \epsilon } ( s _ { - J \cup \{ i \} } , x ) \right\} .
$$

where $s _ { - J \cup \{ i \} } = ( s _ { k } ) _ { k \in I \setminus ( J \cup \{ i \} ) }$ denotes the strategy profile of all players outside $J \cup \{ i \}$ . In   
particular, the notation $B R D _ { \epsilon , i }$ will be simplified to $B R D _ { i }$ when $\epsilon = 0$ .

The dual $\epsilon$ -best response set $B R D _ { \epsilon , i } ( s , J )$ characterizes the strategic interdependence between player $i$ and the coalition $J$ . If all players in $J$ adopt strategies $x \bar { \in } \ B R { \cal D } _ { \epsilon , i } \bar { ( s , J ) }$ while other players maintain their current strategies $s _ { - J \cup \{ i \} }$ , then player $i$ ’s current strategy $s _ { i }$ becomes an $\epsilon$ -best response to the resulting strategy profile.

Proposition 2. In a pure strategy game 454 $G$ , suppose that each $g _ { i }$ is continuous. Then for a joint 455 strategy $s$ , a subset of players $J \subset I$ , and a player $i \in I \backslash J _ { i }$ , the set $B R D _ { \epsilon , i } ( s , J )$ is closed.

Proof. By Definition 1 and Definition 9

$$
B R D _ { \epsilon , i } ( s , J ) = \left\{ x \in \prod _ { j \in J } S _ { j } \mid \forall t \in S _ { i } , g _ { i } ( s _ { i } , s _ { - J \cup \{ i \} } , x ) \geq g _ { i } ( t , s _ { - J \cup \{ i \} } , x ) - \epsilon \right\} .
$$

Choose a limit point $x$ of $B R D _ { \epsilon , i } ( s , J )$ , choose any $t \in S _ { i }$ , and construct a function

$$
h _ { t } : \prod _ { j \in J } S _ { j } \to \mathbb { R } , \quad y \mapsto g _ { i } \big ( s _ { i } , s _ { - J \cup \{ i \} } , y \big ) - g _ { i } \big ( t , s _ { - J \cup \{ i \} } , y \big ) + \epsilon .
$$

Since $g _ { i }$ is continuous, $h _ { t }$ is also continuous. Assume on the contrary that $h _ { t } ( x ) \ < \ 0$ , then $h _ { t } ^ { - 1 } ( - \infty , 0 )$ is an open neighborhood of $x$ . Since $x$ is a limit point of $B R D _ { \epsilon , i } ( s , J )$ , there exists a point $y \in h _ { t } ^ { - 1 } ( - \infty , 0 ) \cap B R D _ { \epsilon , i } ( s , J )$ . However

$$
\begin{array} { r } { g _ { i } \big ( s _ { i } , s _ { - J \cup \{ i \} } , y \big ) - g _ { i } \big ( t , s _ { - J \cup \{ i \} } , y \big ) + \epsilon \ge 0 . } \end{array}
$$

is a contradiction. As a result, $h _ { t } ( x ) \geq 0$ . Since $t$ is arbitrary

$$
\forall t \in S _ { i } , \quad g _ { i } ( s _ { i } , s _ { - J \cup \{ i \} } , x ) - g _ { i } ( t , s _ { - J \cup \{ i \} } , x ) + \epsilon \geq 0 .
$$

456 So $x \in B R D _ { \epsilon , i } ( s , J )$ which means $B R D _ { \epsilon , i } ( s , J )$ is a closed set.

Proposition 3. In a pure strategy game $G$ , suppose that each $g _ { i }$ is continuous. Then for any $j \in I$ , the set

$$
A _ { j } = \left\{ s \in \prod _ { i \in I } S _ { i } \mid s _ { j } \in B R _ { \epsilon } ( s _ { - j } ) \right\} .
$$

is closed.

Proof. This proof is similar to the proof of Proposition 2. Since $g _ { j }$ is continuous, any limit point of $A _ { j }$ belongs to $A _ { j }$ , which means $A _ { j }$ is closed. □

# A.1 Proof of Lemma 1

Lemma 1. In a pure strategy game $G$ with the group set $P$ , suppose that each $S _ { i }$ is compact and Hausdorff, and each $g _ { i }$ is continuous. For a joint strategy s, if $N _ { \epsilon } ( s )$ is a local minimum, then

1. either for any joint strategy $t \in T _ { \epsilon } ( s )$ in the admissible subsequent joint strategy set, the group which has achieved $\epsilon$ -best response in s will still achieve ϵ-best response in $t$ ,

. or there exists a non-empty open set $O \subset T _ { \epsilon } ( s )$ in the admissible subsequent joint strategy set satisfying:

(a) For any joint strategy $t \in O$ , there exists a group which has achieved ϵ-best response in s will not achieve $\epsilon$ -best response in $t$ .   
(b) There exists a group which has not achieved $\epsilon$ -best response in s will achieve ϵ-best response in any joint strategy $t \in O$ .

Proof. Without loss of generality, let $| P | = n + m$ , group $p _ { 1 } , \ldots , p _ { n }$ achieve $\epsilon$ -best response in the joint strategy $s$ , while group $p _ { n + 1 } , . . . , p _ { n + m }$ not. Let

$$
J = \bigcup _ { x = 1 } ^ { n } p _ { x } , \quad K = \bigcup _ { x = n + 1 } ^ { n + m } p _ { x } .
$$

By assumption, all players in $J$ achieve $\epsilon$ -best response in $s$ . Construct a set

$$
A = \bigcap _ { j \in J } B R D _ { \epsilon , j } ( s , K ) .
$$

Since each $g _ { j }$ is continuous, $B R D _ { \epsilon , j } ( s , K )$ is closed by Proposition 2. $A$ is the intersection of finite   
closed sets, so $A$ is closed.

Let

$$
V = \prod _ { k \in K } S _ { k } .
$$

Since each $S _ { k }$ is compact and Hausdorff, the product space $V$ is also compact and Hausdorff.   
According to Baire category theorem, $V$ is a Baire space. The set $V \backslash A$ is the complement of $A$ in   
$V$ , so $V \backslash A$ is open. The open set in a Baire space is also a Baire space when it is non-empty, so the   
topological subspace $V \backslash A$ is a Baire space when it is non-empty.

Actually, any possible joint strategy following $s$ without violating the property of grouped $\epsilon$ -satisficing path, has the same component of $s$ with index set $J$ , and the entirety of their components with index set $K$ is $V$ , namely

$$
T _ { \epsilon } ( s ) = s _ { J } \times V .
$$

As a result, $s _ { J } \times A \subset s _ { J } \times V$ is all admissible subsequent joint strategies which keep players   
in $J$ achieving $\epsilon$ -best response, in other words, keep group $p _ { 1 } , \ldots , p _ { n }$ achieving $\epsilon$ -best response.   
Since $N _ { \epsilon } ( s )$ is a local minimum, any joint strategy $t \in s _ { J } \times ( V \backslash A )$ must let at least one group $p \in$   
$\{ p _ { n + 1 } , \ldots , p _ { n + m } \}$ achieve $\epsilon$ -best response. Otherwise $N _ { \epsilon } ( t ) \le N _ { \epsilon } ( s ) - 1$ , which is a contradiction   
with Definition 6.

Construct $m$ sets

$$
B _ { n + i } = \{ v \in V \mid w = ( s _ { J } , v ) , \forall x \in p _ { n + i } , w _ { x } \in B R _ { \epsilon } ( w _ { - x } ) \} , \quad i = 1 , \ldots , m .
$$

which means any joint strategy in $s _ { J } \times B _ { n + i }$ lets group $p _ { n + i }$ achieve $\epsilon$ -best response. Obviously

$$
B _ { n + i } = \bigcap _ { x \in p _ { n + i } } \{ v \in V \mid w = ( s _ { J } , v ) , w _ { x } \in B R _ { \epsilon } ( w _ { - x } ) \} , \quad i = 1 , \dots , m .
$$

According to Proposition 3, each set in the right-hand-side is closed, so $B _ { n + i }$ is closed. So $B _ { n + i } \cap$   
$( V \backslash A )$ is closed in the topological subspace $V \backslash A$ , and

$$
\bigcup _ { i = 1 } ^ { m } B _ { n + i } \cap ( V \backslash A ) = V \backslash A .
$$

487 Two cases need to be discussed here.

88 Case 1. $V \backslash A = \emptyset$ . Then $A = V$ which means any possible joint strategy following $s$ without   
violating the property of grouped $\epsilon$ -satisficing path, keeps group $p _ { 1 } , \ldots , p _ { n }$ achieve $\epsilon$ -best response.

Case 2. $V \backslash A \neq \emptyset$ . According to Equation 14, $B _ { n + i } \cap ( V \backslash A )$ is closed and $V \backslash A$ is a Baire space, there exists one $B _ { n + i } \cap ( V \backslash A )$ has a non-empty interior. So there exists a non-empty open set $C$ satisfying

$$
C \subset B _ { n + i } \cap ( V \backslash A ) .
$$

By Equation 13, any joint strategy in $s _ { J } \times C$ lets group $p _ { n + i }$ achieve $\epsilon$ -best response. By Equation 11, Equation 12 and $C \subset V \backslash A$ , any joint strategy in $s _ { J } \times C$ makes at least one group $p \in \{ p _ { 1 } , . . . , p _ { n } \}$ no longer achieve $\epsilon$ -best response. Obviously, $s _ { J } \times C$ is an open set in $T _ { \epsilon } ( s )$ .

Remark 5. This lemma is very critical for Theorem 1. Usually, one point is defined as a closed set, so a closed set does not necessarily have a non-empty interior. In this lemma, Baire space and Baire theorem provide a guarantee to find a non-empty interior in finite closed sets when they satisfy some conditions.

# A.2 Proof of Lemma 2

Lemma 2. In a pure strategy game $G$ , suppose that each $S _ { i }$ is a convex set in a topological vector space, and each $g _ { i }$ is partially analytic. If there exists a non-empty open set $\textstyle U \subset \prod _ { i \in I } S _ { i }$ satisfying that the player $k$ achieves best response in any joint strategy $s \in U$ , then the player $k$ will achieve best response in any joint strategy $s \in \Pi _ { i \in I } S _ { i }$ .

Proof. Without loss of generality, let $I = \{ 1 , \ldots , n \}$ , and $k = 1$ . Since $U$ is open and non-empty,   
504 there exists a non-empty open set $O$

$$
O = \prod _ { i \in I } O _ { i } \subset U .
$$

where $O _ { i }$ is open in $S _ { i }$ . Choose a joint strategy

$$
s = ( s _ { i } ) _ { i \in I } , \quad \forall i \in I , \quad s _ { i } \in O _ { i } .
$$

05 Then $s \in O$ which means the joint strategy $s$ makes player $k$ achieve best response.

Induction needs to be used here.

Head. For any $t _ { k } \in S _ { k }$ , construct a function

$$
f : [ 0 , 1 ] \to \mathbb { R } , \quad x \mapsto g _ { k } \big ( ( 1 - x ) s _ { k } + x t _ { k } , s _ { - k } \big ) - g _ { k } \big ( s _ { k } , s _ { - k } \big ) .
$$

Obviously, $f$ is an analytic function with respect to $x$ according to the assumption of $g _ { k }$ and Definition 7. Consider

$$
t : [ 0 , 1 ] \to \prod _ { i \in I } S _ { i } , \quad x \mapsto ( ( 1 - x ) s _ { k } + x t _ { k } , s _ { - k } ) .
$$

Since each $S _ { i }$ is a convex set in a topological vector space, the finite product space $\Pi _ { i \in I } S _ { i }$ is a subspace in some topological vector space also. Because $t ( 0 ) = s \in O , O$ is open and $t ( x )$ is a linear mapping, there exists an open neighborhood $( a , b ) \subset \mathbb { R }$ of 0 which satisfies

$$
\forall x \in [ 0 , b ) , \quad t ( x ) \in O .
$$

According to Equation 15 and the assumption of $U$

$$
\forall x \in [ 0 , b ) , \quad ( 1 - x ) s _ { k } + x t _ { k } \in B R ( s _ { - k } ) .
$$

As a result

$$
\forall x \in [ 0 , b ) , \quad g _ { k } ( ( 1 - x ) s _ { k } + x t _ { k } , s _ { - k } ) = g _ { k } ( s _ { k } , s _ { - k } ) \geq \operatorname* { s u p } _ { r _ { k } \in S _ { k } } g _ { k } ( r _ { k } , s _ { - k } ) .
$$

So for any $x \in [ 0 , b )$ , $f ( x ) = 0$ . Since $f$ is analytic, then

$$
\forall x \in [ 0 , 1 ] , \quad f ( x ) = 0 .
$$

So $t _ { k } \in B R ( s _ { - k } )$ . Due to arbitrary $t _ { k }$ , $S _ { k } = B R ( s _ { - k } )$ . Construct a set

$$
V _ { 1 } = S _ { k } \times \prod _ { i = 2 } ^ { n } O _ { i } .
$$

So each joint strategy $s \in V$ makes player $k$ achieve best response.

Recursion. Assume

$$
V _ { m } = S _ { k } \times \prod _ { i = 2 } ^ { m } S _ { i } \times \prod _ { i = m + 1 } ^ { n } O _ { i } .
$$

Each joint strategy $s \in V _ { m }$ makes player $k$ achieve best response.

For any 510 $\in S _ { k } , t _ { i } \in S _ { i } , i = 2 , \dots , m + 1$ , construct a function

$$
\begin{array} { r l } & { h : [ 0 , 1 ] \to \mathbb { R } , } \\ & { \quad x \mapsto g _ { k } ( s _ { k } , t _ { 2 } , \ldots , t _ { m } , ( 1 - x ) s _ { m + 1 } + x t _ { m + 1 } , s _ { m + 2 } , \ldots , s _ { n } ) } \\ & { \quad - g _ { k } ( t _ { k } , t _ { 2 } , \ldots , t _ { m } , ( 1 - x ) s _ { m + 1 } + x t _ { m + 1 } , s _ { m + 2 } , \ldots , s _ { n } ) . } \end{array}
$$

Since each part of $h$ is analytic, $h$ is analytic. Since $O _ { m + 1 }$ is open, there exists an open neighborhood $( c , d ) \subset \mathbb { R }$ of 0 which satisfies

$$
\forall x \in [ 0 , d ) , \quad ( 1 - x ) s _ { m + 1 } + x t _ { m + 1 } \in O _ { m + 1 } .
$$

So for any $x \in [ 0 , d )$ , $h ( x ) = 0$ according to the assumption of recursion. Due to analytic property, for any $x \in [ 0 , 1 ]$ , $h ( x ) = 0$ . So

$$
\forall t _ { k } \in S _ { k } , \quad g _ { k } ( t _ { k } , t _ { 2 } , \dots , t _ { m } , t _ { m + 1 } , s _ { m + 2 } , \dots , s _ { n } ) = g _ { k } ( s _ { k } , t _ { 2 } , \dots , t _ { m } , t _ { m + 1 } , s _ { m + 2 } , \dots , s _ { n } ) .
$$

So

$$
S _ { k } \subset B R ( t _ { 2 } , \ldots , t _ { m } , t _ { m + 1 } , s _ { m + 2 } , \ldots , s _ { n } ) .
$$

Construct a set

$$
V _ { m + 1 } = S _ { k } \times \prod _ { i = 2 } ^ { m + 1 } S _ { i } \times \prod _ { i = m + 2 } ^ { n } O _ { i } .
$$

Each joint strategy $s \in V _ { m + 1 }$ makes player $k$ achieve best response.

By induction, each joint strategy 12 $s \in \Pi _ { i \in I } S _ { i }$ makes player $k$ achieve best response.



Remark 6. The function $f$ in Equation $1 6$ is not similar to $h$ in Equation 17. $f$ is used to extend the best response property along $S _ { k }$ , while $h$ to extend the gap between different strategies in $S _ { k }$ along $S _ { m + 1 }$ . The continuity of scalar multiplication in topological vector spaces and the extendability of analytic functions play a important role in the proof of this lemma.

# 18 B Some proofs in Section 4

Proposition 1. In a pure strategy game $G$ with the group set $P$ , for any infinite-length path $( s ^ { t } ) _ { t \geq 0 }$ , there exists an index v satisfying $N _ { \epsilon } ( s ^ { v } ) = \operatorname* { i n f } _ { t \geq 0 } N _ { \epsilon } ( s ^ { t } )$ .

Proof. Construct $| P | + 1$ sets

$$
A _ { i } = \{ t \mid N _ { \epsilon } ( s ^ { t } ) = i \} , \quad i = 0 , \ldots , | P | .
$$

There exists a minimum $k$ making $| A _ { k } | \neq \emptyset$ . Choose an element $t \in A _ { k }$ . For finite joint strategies   
$s _ { 0 } , s _ { 1 } , \ldots , s _ { t }$ , there exists the first index $v$ making $N _ { \epsilon } ( s ^ { v } ) = k$ . So $\begin{array} { r } { N _ { \epsilon } ( s ^ { v } ) = \operatorname* { i n f } _ { i \geq 0 } \bar { N } _ { \epsilon } ( s ^ { i } ) } \end{array}$ .

# B.1 Proof of Corollary 1

Corollary 1. In a pure strategy game $G$ , suppose that each $S _ { i }$ is a convex compact set in a topological vector space and each $g _ { i }$ is continuous and partially analytic. If for any $g _ { i }$ and any $\begin{array} { r } { s _ { - i } \in \prod _ { j \neq i , j \in I } S _ { j } } \end{array}$ the function $g _ { i } ( s _ { i } , s _ { - i } )$ is quasi-convex with respect to $s _ { i }$ , then for any initial joint strategy s, there exists a finite-length satisficing path $( s ^ { t } ) _ { t = 0 } ^ { T }$ where $s ^ { 0 } = s$ and ${ \bf \bar { \mathbf { \Lambda } } } _ { S } T$ is an equilibrium.

Proof. There exists a theorem with respect to continuous games [17]. In a pure strategy game   
$G = ( I , ( S _ { i } ) _ { i \in I } , ( g _ { i } ) _ { i \in I } )$ , suppose that each $S _ { i }$ is a convex compact set in a topological vector space,   
and each $g _ { i }$ is continuous. If for each $g _ { i }$ and any $\begin{array} { r } { s _ { - i } \in \prod _ { j \neq i , j \in I } S _ { j } } \end{array}$ , the function $g _ { i } ( s _ { i } , s _ { - i } )$ is   
quasi-convex with respect to $s _ { i }$ , then $G$ admits at least one equilibrium.

As a result, for any subset $J \subset I$ and any joint strategy $\textstyle s \in \prod _ { i \in I } S _ { i }$ , construct a sub-game $\tilde { G } = ( J , ( S _ { i } ) _ { i \in J } , ( \tilde { g } _ { i } ) _ { i \in J } )$ in which

$$
\tilde { g } _ { i } : \prod _ { i \in J } S _ { i } \to \mathbb { R } , \quad w \mapsto g _ { i } ( s _ { - J } , w ) .
$$

where $s _ { - J } = ( s _ { i } ) _ { i \in I \backslash J }$ denotes the strategy profile of all players outside $J$ .

Function $\tilde { g } _ { i }$ is also continuous, and for any $\begin{array} { r } { s _ { - i } \in \prod _ { j \neq i , j \in J } S _ { j } } \end{array}$ , the function $g _ { i } ( s _ { i } , s _ { - i } )$ is quasiconvex with respect to $s _ { i }$ as well. So $\tilde { G }$ has an equilibrium according to the theorem with respect to continuous games.

So all conditions of Theorem 2 are satisfied. As a result, for any initial joint strategy $s$ in $G$ , there exists a finite-length satisficing path $( s ^ { t } ) _ { t = 0 } ^ { T }$ where $s ^ { 0 } = s$ and $s ^ { \check { T } }$ is some equilibrium. □

# 538 B.2 Proof of Corollary 2

Proposition 4. Suppose that real numbers $a _ { 1 } , \ldots , a _ { n } , b _ { 1 } , \ldots , b _ { n }$ belong to $[ 0 , 1 ]$ , then

$$
\left| \prod _ { i = 1 } ^ { n } a _ { i } - \prod _ { i = 1 } ^ { n } b _ { i } \right| \leq \sum _ { i = 1 } ^ { n } | a _ { i } - b _ { i } | .
$$

Proof. Obviously, the proposition is true when $n = 1$ . Assume the proposition is true when $n = k$ ,   
then

$$
\begin{array} { r l } { \displaystyle \left| \prod _ { i = 1 } ^ { k + 1 } a _ { i } - \prod _ { i = 1 } ^ { k + 1 } b _ { i } \right| = \left| \prod _ { i = 1 } ^ { k + 1 } a _ { i } - b _ { 1 } \prod _ { i = 2 } ^ { k + 1 } a _ { i } + b _ { 1 } \prod _ { i = 2 } ^ { k + 1 } a _ { i } - \prod _ { i = 1 } ^ { k + 1 } b _ { i } \right| } & { } \\ { \leq \left| a _ { 1 } - b _ { 1 } \right| \left| \prod _ { i = 2 } ^ { k + 1 } a _ { i } \right| + \left| b _ { 1 } \right| \left| \prod _ { i = 2 } ^ { k + 1 } a _ { i } - \prod _ { i = 2 } ^ { k + 1 } b _ { i } \right| } & { } \\ { \leq \left| a _ { 1 } - b _ { 1 } \right| + \left| \prod _ { i = 2 } ^ { k + 1 } a _ { i } - \prod _ { i = 2 } ^ { k + 1 } b _ { i } \right| . } & { } \end{array}
$$

By assumption

$$
\left| \prod _ { i = 1 } ^ { k + 1 } a _ { i } - \prod _ { i = 1 } ^ { k + 1 } b _ { i } \right| \leq | a _ { 1 } - b _ { 1 } | + \sum _ { i = 2 } ^ { k + 1 } | a _ { i } - b _ { i } | .
$$

So the proposition is true when $n = k + 1$

Corollary 2. In a mixed strategy game $G$ , suppose that each $S _ { i }$ is a finite set. Then for any initial   
joint mixed strategy $\sigma$ , there exists a finite-length satisficing path $( \sigma ^ { t } ) _ { t = 0 } ^ { \tilde { T } }$ where $\sigma ^ { 0 } = \sigma$ and $\sigma ^ { T }$ is $a$   
544 mixed equilibrium.

Proof. Construct a pure strategy game $\tilde { G } = ( I , ( \Delta S _ { i } ) _ { i \in I } , ( \tilde { g } _ { i } ) _ { i \in I } )$ where $\Delta { \cal { S } } _ { i }$ is the set of all Borel probability measures over $S _ { i }$ , and

$$
\widetilde { g } _ { i } : \prod _ { i \in I } \Delta S _ { i } \to \mathbb { R } , \quad \prod _ { i \in I } \sigma _ { i } \mapsto \sum _ { s \in \prod _ { i \in I } S _ { i } } g _ { i } ( s ) \prod _ { i \in I } \sigma _ { i } ( s _ { i } ) .
$$

45 Since $S _ { i }$ is a finite set, $\Delta S _ { i }$ is actually a $( | S _ { i } | - 1 )$ -dimensional simplex in $\mathbb { R } ^ { | S _ { i } | }$ . So $\Delta S _ { i }$ is a convex   
compact set in R|Si|.

# Prove that 547 $\tilde { g } _ { i }$ is continuous.

$g _ { i }$ is bounded by

$$
M = \operatorname* { m a x } _ { s \in \prod _ { j \in I } S _ { j } } | g _ { i } ( s ) | .
$$

The space $\Pi _ { i \in I } \Delta S _ { i }$ can be endowed with a metric topology.

$$
| \sigma - \eta | = \sum _ { i \in I } | \sigma _ { i } - \eta _ { i } | = \sum _ { i \in I } \sum _ { s \in S _ { i } } | \sigma _ { i } ( s ) - \eta _ { i } ( s ) | , \quad \sigma , \eta \in \prod _ { i \in I } \Delta S _ { i } .
$$

Since the topology constructed by metric $\begin{array} { r } { \sum _ { s \in S _ { i } } | \sigma _ { i } ( s ) - \eta _ { i } ( s ) | } \end{array}$ is the same as the topology in $\mathbb { R } ^ { | S _ { i } | }$ ,   
the definition above is reasonable.

For any $\epsilon > 0$ , there exists $\delta = \epsilon / M$ such that for any $| \sigma - \eta | < \delta$

$$
\left| \tilde { g } _ { i } ( \sigma ) - \tilde { g } _ { i } ( \eta ) \right| \leq \sum _ { s \in \prod _ { i \in I } S _ { i } } M \left| \prod _ { i \in I } \sigma _ { i } ( s _ { i } ) - \prod _ { i \in I } \eta _ { i } ( s _ { i } ) \right| .
$$

According to proposition 4

$$
| \tilde { g } _ { i } ( \sigma ) - \tilde { g } _ { i } ( \eta ) | \leq M \sum _ { s \in \prod _ { i \in I } S _ { i } } \sum _ { i \in I } | \sigma _ { i } ( s _ { i } ) - \eta _ { i } ( s _ { i } ) | = M | \sigma - \eta | < \epsilon .
$$

So 550 $\tilde { g } _ { i }$ is continuous.

Prove that 551 $\tilde { g } _ { i }$ is partially analytic.

For any j ∈ I, σj , ηj ∈ ∆Sj , $\begin{array} { r } { \sigma _ { - j } \in \prod _ { l \neq j , l \in I } \Delta S _ { l } } \end{array}$ , then

$$
\tilde { g } _ { i } ( x \sigma _ { j } + ( 1 - x ) \eta _ { j } , \sigma _ { - j } ) = \sum _ { s \in \prod _ { i \in I } S _ { i } } g _ { i } ( s ) ( x \sigma _ { j } ( s _ { j } ) + ( 1 - x ) \eta _ { j } ( s _ { j } ) ) \sigma _ { - j } ( s _ { - j } ) .
$$

Obviously, $\tilde { g } _ { i } ( x \sigma _ { j } + ( 1 - x ) \eta _ { j } , \sigma _ { - j } )$ is a linear polynomial with respect to $x$ , which means an analytic   
function with respect to $x$ .

# 554 Prove that any sub-game has an equilibrium.

For any subset $J ~ \subset ~ I$ and any joint strategy $\begin{array} { r } { \sigma ~ \in ~ \prod _ { i \in I } \Delta S _ { i } } \end{array}$ , construct a sub-game $\tilde { \cal H } \ =$ $( J , ( \Delta S _ { i } ) _ { i \in J } , ( \tilde { h } _ { i } ) _ { i \in J } )$ where

$$
\tilde { h } _ { i } : \prod _ { i \in J } \Delta S _ { i } \to \mathbb { R } , \quad \eta \mapsto \tilde { g } _ { i } ( \sigma _ { - J } , \eta ) .
$$

Construct functions

$$
h _ { i } : \prod _ { j \in J } S _ { j } \to \mathbb { R } , \quad w \mapsto \sum _ { v \in \prod _ { j \in I \setminus J } S _ { j } } g _ { i } ( w , v ) \sigma _ { - J } ( v ) .
$$

So

$$
\tilde { h } _ { i } ( \eta ) = \sum _ { s \in \prod _ { i \in J } S _ { i } } h _ { i } ( s ) \prod _ { i \in J } \eta _ { i } ( s _ { i } ) .
$$

which means that 555 $\tilde { H }$ is the extended pure strategy game of a mixed strategy game $\cal H \_ =$ 556 $( J , ( S _ { i } ) _ { i \in J } , ( h _ { i } ) _ { i \in J } )$ .

There exists a theorem with respect to mixed strategy games [31]. Each $N$ -player game has a mixed equilibrium.

As a result, $H$ has a mixed equilibrium and $\tilde { H }$ has an equilibrium.

So all conditions of Theorem 2 are satisfied. As a result, for any initial joint mixed strategy $\sigma$ in $G$ , there exists a finite-length satisficing path $( \sigma ^ { t } ) _ { t = 0 } ^ { T }$ where $\sigma ^ { 0 } = { \bar { \sigma } }$ and $\sigma ^ { \check { T } }$ is a mixed equilibrium.

# 563 B.3 Proof of Corollary 3

Proposition 5. For the real number $c \in ( 0 , 1 )$ and the positive integer $p _ { : }$ , there exists

$$
\sum _ { q \geq p , q \in \mathbb { N } } c ^ { q } { \binom { q } { p } } \leq \frac { 2 \sqrt { 2 \pi } p ^ { 1 / 2 } } { ( - \ln c ) ^ { p } } .
$$

when p goes to infinity.

Proof. Since ${ \binom { q } { p } } = q ( q - 1 ) \cdots ( q - p + 1 ) / p ! \leq q ^ { p } / p !$

$$
\sum _ { q \geq p , q \in \mathbb { N } } c ^ { q } { \binom { q } { p } } \leq \frac { 1 } { p ! } \sum _ { q \geq p , q \in \mathbb { N } } c ^ { q } q ^ { p } .
$$

The function $f ( q ) = c ^ { q } q ^ { p }$ has

$$
{ \frac { d } { d q } } f = c ^ { q } q ^ { p - 1 } ( q \ln c + p ) .
$$

Since $f ( 0 ^ { + } ) = f ( + \infty ) = 0$ and $f ( q ) \geq 0 .$ , $f$ has only one maximal point $- p / \ln c . \ d f / d q > 0$ when $0 < q < - p / \ln c$ , and ${ d f } / { d q } < 0$ when $- p / \ln c < q$ . So

$$
\sum _ { q \geq p , q \in \mathbb { N } } c ^ { q } q ^ { p } \leq \int _ { 0 } ^ { \infty } f ( q ) d q + 2 f \left( - { \frac { p } { \ln c } } \right) .
$$

The form of the first term in the right-hand-side is similar to $\Gamma$ function.

$$
\Gamma ( n + 1 ) = \int _ { 0 } ^ { \infty } t ^ { n } e ^ { - t } d t = n ! .
$$

Actually

$$
\int _ { 0 } ^ { \infty } f ( q ) d q = \left( - { \frac { 1 } { \ln c } } \right) ^ { p + 1 } \int _ { 0 } ^ { \infty } e ^ { - t } t ^ { p } d q = \left( - { \frac { 1 } { \ln c } } \right) ^ { p + 1 } p ! .
$$

So

$$
\sum _ { q \geq p , q \in \mathbb { N } } c ^ { q } { \binom { q } { p } } \leq \left( - { \frac { 1 } { \ln c } } \right) ^ { p + 1 } + 2 \left( - { \frac { 1 } { \ln c } } \right) ^ { p } { \frac { e ^ { - p } p ^ { p } } { p ! } } = \left( - { \frac { 1 } { \ln c } } \right) ^ { p } \left( - { \frac { 1 } { \ln c } } + 2 { \frac { e ^ { - p } p ^ { p } } { p ! } } \right) .
$$

According to Stirling’s formula $p ! \sim { \sqrt { 2 \pi p } } p ^ { p } e ^ { - p }$ , there exists $e ^ { - p } p ^ { p } / p ! \sim \sqrt { 2 \pi p }$ . So

$$
\sum _ { q \geq p , q \in \mathbb { N } } c ^ { q } { \binom { q } { p } } \leq \left( - { \frac { 1 } { \ln c } } \right) ^ { p } 2 { \sqrt { 2 \pi } } p ^ { 1 / 2 } , \quad p \to \infty .
$$



Corollary 3. In a stationary mixed strategy stochastic game $G$ , suppose that each $S _ { i }$ is a finite set.   
Then for any initial stationary joint mixed strategy $\sigma$ , there exists a finite-length satisficing path   
$( \sigma ^ { t } ) _ { t = 0 } ^ { T }$ where $\sigma ^ { 0 } = \sigma$ and $\sigma ^ { T }$ is a mixed equilibrium.

Proof. According to the definition of $g _ { i }$ and the finiteness of $\textstyle X \times \prod _ { i \in I } S _ { i } ,$ , $g _ { i }$ is bounded by

$$
M = \operatorname* { m a x } _ { x \in X , s \in \Pi _ { i \in I } } | g _ { i } ( x , s ) | .
$$

Construct a pure strategy game $H = \left( ( I , X ) , ( T _ { i , x } ) _ { ( i , x ) \in ( I , X ) } , \left( h _ { i , x } \right) _ { ( i , x ) \in ( I , X ) } \right)$ . $( I , X )$ means a double index set, namely $( I , X ) = \{ ( i , x ) \mid i \in I , x \in X \}$ . Since $| I | < \infty$ and $| X | < \infty$ , $( I , X )$ is finite.

$$
T _ { i , x } = \Delta S _ { i } .
$$

Since $S _ { i }$ is finite, $T _ { i , x }$ is a simplex in some finite-dimensional Euclidean space. So $T _ { i , x }$ is convex and compact.

$$
h _ { i , x } : \prod _ { ( j , y ) \in ( I , X ) } T _ { j , y } \to \mathbb { R } , \quad \prod _ { ( j , y ) \in ( I , X ) } \pi _ { j } ( y ) \mapsto \mathbb { E } _ { ( \pi _ { j } ) _ { j \in I } } \left[ \sum _ { t \geq 0 } \gamma _ { i } ^ { t } g _ { i } ( x ^ { t } , s ^ { t } ) \mid x _ { 0 } = x \right] .
$$

So

$$
h _ { i , x } ( \pi ) = \mathbb { E } _ { s \sim \pi } [ g _ { i } ( x , s ) ] + \gamma _ { i } \mathbb { E } _ { s \sim \pi , y \sim P ( x , s ) } [ h _ { i , y } ( \pi ) ] .
$$

If $\begin{array} { r } { \pi \in \prod _ { ( j , y ) \in ( I , X ) } T _ { j , y } } \end{array}$ is considered as a constant, $h _ { i , x } ( \pi )$ will be viewed as a function with respect to $x$ , denoted as

$$
h _ { i } ( \pi ) : X \to \mathbb { R } , \quad x \mapsto h _ { i , x } ( \pi ) .
$$

The function space $\mathbb { R } ^ { X }$ is endowed with uniform metric topology, using

$$
| f | = \operatorname* { s u p } _ { x \in X } | f ( x ) | , \quad f \in \mathbb { R } ^ { X } .
$$

as a norm in 569 $\mathbb { R } ^ { X }$ . Since $\mathbb { R }$ is complete, the uniform metric topology in $\mathbb { R } ^ { X }$ is complete.

Construct an operator

$$
\begin{array} { r } { o p _ { \pi } : \mathbb { R } ^ { X }  \mathbb { R } ^ { X } , \quad f \mapsto \{ ( x , \mathbb { E } _ { s \sim \pi } [ g _ { i } ( x , s ) ] + \gamma _ { i } \mathbb { E } _ { s \sim \pi , y \sim P ( x , s ) } [ f ( y ) ] ) ~ | ~ x \in X \} . } \end{array}
$$

For any $x \in X , p , q \in \mathbb { R } ^ { X }$ , there exists

$$
| o p _ { \pi } ( p ) ( x ) - o p _ { \pi } ( q ) ( x ) | = \gamma _ { i } | \mathbb { E } _ { s \sim \pi , y \sim P ( x , s ) } [ p ( y ) - q ( y ) ] | \leq \gamma _ { i } \operatorname* { s u p } _ { y \in X } | p ( y ) - q ( y ) | = \gamma _ { i } | p - q | .
$$

So

$$
| o p _ { \pi } ( p ) - o p _ { \pi } ( q ) | = \operatorname* { s u p } _ { x \in X } | o p _ { \pi } ( p ) ( x ) - o p _ { \pi } ( q ) ( x ) | \leq \gamma _ { i } | p - q | .
$$

Since $0 \leq \gamma _ { i } < 1$ , $o p _ { \pi }$ is a contraction mapping.

According to contraction mapping theorem, there exists a unique fixed point. That is

$$
h _ { i } ( \pi ) = o p _ { \pi } ( h _ { i } ( \pi ) ) .
$$

So $h _ { i } ( \pi )$ exists and is unique. As a result, $h _ { i , x }$ is a well-defined function.

Obviously, $h _ { i , x } ( \pi )$ is bounded by

$$
| h _ { i , x } ( \pi ) | \leq \sum _ { t \geq 0 } \gamma _ { i } ^ { t } M \leq \frac { M } { 1 - \gamma _ { i } } .
$$

This conclusion is always true no matter what 573 $x$ or $\pi$ is. So for any $\begin{array} { r } { \pi \in \prod _ { ( j , y ) \in ( I , X ) } T _ { j , y } } \end{array}$

$$
| h _ { i } ( \pi ) | \leq \frac { M } { 1 - \gamma _ { i } } .
$$

Prove that $h _ { i , x }$ is continuous.

Since each $T _ { i , x }$ is a metric topological space, the finite product space $\textstyle \prod _ { ( i , x ) \in ( I , X ) } T _ { i , x }$ is also a metric topological space. The metric in this space is denoted as $| \pi - \sigma |$ where $\begin{array} { r } { \pi , \boldsymbol { \dot { \sigma } } \in \prod _ { ( i , x ) \in ( I , X ) } T _ { i , x } } \end{array}$ .

For any577 $\begin{array} { r } { x \in X , p , q \in \mathbb { R } ^ { X } , \pi , \sigma \in \prod _ { ( i , x ) \in ( I , X ) } T _ { i , x } } \end{array}$ , there exists

$$
\begin{array} { r l } & { | o p _ { \pi } ( p ) ( x ) - o p _ { \sigma } ( q ) ( x ) | \le | \mathbb { E } _ { s \sim \pi } [ g _ { i } ( x , s ) ] - \mathbb { E } _ { s \sim \sigma } [ g _ { i } ( x , s ) ] | } \\ & { \phantom { | o p _ { \pi } ( p ) ( x ) - o p _ { \sigma } ( q ) ( x ) | } + \gamma _ { i } | \mathbb { E } _ { s \sim \pi , y \sim P ( x , s ) } [ p ( y ) ] - \mathbb { E } _ { s \sim \sigma , y \sim P ( x , s ) } [ q ( y ) ] | . } \end{array}
$$

Since

$$
\begin{array} { r l } {  { \| \mathbb { E } _ { s \sim \pi } [ g _ { i } ( x , s ) ] - \mathbb { E } _ { s \sim \sigma } [ g _ { i } ( x , s ) ] \| = \Bigg | \displaystyle \sum _ { s \in \prod _ { j \in I } S _ { j } } g _ { i } ( x , s ) ( \pi ( x ) ( s ) - \sigma ( x ) ( s ) ) \Bigg | } } \\ & { \leq \displaystyle \sum _ { s \in \prod _ { j \in I } S _ { j } } M | \prod _ { k \in I } \pi _ { k } ( x ) ( s _ { k } ) - \prod _ { k \in I } \sigma _ { k } ( x ) ( s _ { k } ) | } \\ & { \leq M \displaystyle \sum _ { s \in \prod _ { j \in I } S _ { j } } \sum _ { k \in I } \lvert \pi _ { k } ( x ) ( s _ { k } ) - \sigma _ { k } ( x ) ( s _ { k } ) \rvert } \\ & { \leq M C \lVert \pi - \sigma \rVert . } \end{array}
$$

where $C$ is an independent constant. The second $\leq$ inequality is guaranteed by Proposition 4.

$$
\begin{array} { r l } & { \| \mathbb { E } _ { s \sim \pi , y \sim P ( x , s ) } [ p ( y ) ] - \mathbb { E } _ { s \sim \sigma , y \sim P ( x , s ) } [ q ( y ) ] \| } \\ & { = \left| \displaystyle \sum _ { s \in \prod _ { j \in I } S _ { j } , y \in X } \big ( p ( y ) \pi ( x ) ( s ) P ( x , s ) ( y ) - q ( y ) \sigma ( x ) ( s ) P ( x , s ) ( y ) \big ) \right| } \\ & { \leq \displaystyle \sum _ { s \in \prod _ { j \in I } S _ { j } , y \in X } \left| \pi ( x ) ( s ) - \sigma ( x ) ( s ) \right| | p ( y ) | P ( x , s ) ( y ) + | p ( y ) - q ( y ) | \sigma ( x ) ( s ) P ( x , s ) ( y ) } \\ & { \leq \displaystyle \sum _ { s \in \prod _ { j \in I } S _ { j } } \left( \displaystyle \sum _ { s \in \mathbb { Z } } \mid \pi ( x ) ( s ) - \sigma ( x ) ( s ) \| p \| \right) + | p - q | \displaystyle \sum _ { s \in \prod _ { j \in I } S _ { j } , y \in X } \sigma ( x ) ( s ) P ( x , s ) ( y ) } \\ & { \leq \displaystyle \sum _ { y \in X } \left( \displaystyle \sum _ { s \in \prod _ { j \in I } S _ { j } } | \pi ( x ) ( s ) - \sigma ( x ) ( s ) | \| p | \right) + | p - q | \displaystyle \sum _ { s \in \prod _ { j \in I } S _ { j } , y \in X } \sigma ( x ) ( s ) P ( x , s ) ( y ) } \\ & { \leq C | \pi - \sigma | p | \| X | + | p - q | . } \end{array}
$$

So

$$
| o p _ { \pi } ( p ) ( x ) - o p _ { \sigma } ( q ) ( x ) | \leq ( M C + \gamma _ { i } C | p | | X | ) | \pi - \sigma | + \gamma _ { i } | p - q | .
$$

Here consider $| h _ { i } ( \pi ) - h _ { i } ( \sigma ) |$ . According to Equation 18, Equation 19 and Equation 20

$$
\begin{array} { l } { \displaystyle | h _ { i } ( \pi ) - h _ { i } ( \sigma ) | = | o p _ { \pi } ( h _ { i } ( \pi ) ) - o p _ { \sigma } ( h _ { i } ( \sigma ) ) | } \\ { \displaystyle \qquad \leq \left( M C + \gamma _ { i } C \frac { M } { 1 - \gamma _ { i } } | X | \right) | \pi - \sigma | + \gamma _ { i } | h _ { i } ( \pi ) - h _ { i } ( \sigma ) | . } \end{array}
$$

So

$$
\left| h _ { i } ( \pi ) - h _ { i } ( \sigma ) \right| \le \frac { M C } { 1 - \gamma _ { i } } \left( 1 + \frac { \gamma _ { i } } { 1 - \gamma _ { i } } | X | \right) | \pi - \sigma | .
$$

Since each element in the coefficient of the right-hand-side is independent of 82 $\pi$ and $\sigma , h _ { i }$ is continuous 83 about $\pi$ . As a result, $h _ { i , x }$ is continuous about $\pi$ .

Prove that $h _ { i , x }$ is partially analytic.

For any 585 $\begin{array} { r } { ( j , y ) \in ( I , X ) , \pi _ { j } ( y ) , \sigma _ { j } ( y ) \in T _ { j , y } , \eta \in \prod _ { ( k , z ) \neq ( j , y ) , ( k , z ) \in ( I , X ) } T _ { k , z } , \operatorname { l e t } \omega _ { ( k , z ) \neq ( j , y ) , ( k , z ) \in \mathcal { ( \nu } , X ) } } \end{array}$ $\theta _ { a } = ( a \pi _ { j } ( y ) +$ 586 $( 1 - a ) \sigma _ { j } ( y ) , \eta )$ , then

$$
\begin{array} { l } { { \displaystyle h _ { i , x } ( \theta _ { a } ) = { \mathbb E } _ { \theta _ { a } } \left[ \sum _ { t \geq 0 } \gamma _ { i } ^ { t } g _ { i } ( x ^ { t } , s ^ { t } ) \mid x _ { 0 } = x \right] } } \\ { { \displaystyle \ = \sum _ { t \geq 0 } \gamma _ { i } ^ { t } { \mathbb E } _ { x ^ { 0 } = x , \dots , x ^ { u } \sim P ( x ^ { u - 1 } , s ^ { u - 1 } ) , s ^ { u } \sim \theta _ { a } ( x ^ { u } ) , \dots , s ^ { t } \sim \theta _ { a } ( x _ { t } ) } g _ { i } ( x ^ { t } , s ^ { t } ) } . } \end{array}
$$

Considering the boundary of the $k$ -th term $c _ { k } a ^ { k }$ in the polynomial with respect to $a$

$$
| c _ { k } a ^ { k } | \leq M \sum _ { t \geq k } \gamma _ { i } ^ { t - 1 } { \binom { t } { k } } ( 2 a ) ^ { k } = R H S .
$$

Attention, the right-hand-side is the boundary of absolute sum of all possible coefficients of 587 $a ^ { k }$ 588 appeared in $h _ { i , x } ( \theta _ { a } )$ .

When $\gamma _ { i } = 0$ , $h _ { i , x }$ is obviously analytic with respect to $a$

When $0 < \gamma _ { i } < 1$ , according to Proposition 5

$$
R H S \leq M ( 2 a ) ^ { k } \gamma _ { i } ^ { - 1 } \frac { 2 \sqrt { 2 \pi } k ^ { 1 / 2 } } { ( - \ln \gamma _ { i } ) ^ { k } } = \left( \frac { 2 a ( 2 M ) ^ { 1 / k } \gamma _ { i } ^ { - 1 / k } ( 2 \pi k ) ^ { 1 / ( 2 k ) } } { - \ln \gamma _ { i } } \right) ^ { k } , \quad k \to \infty .
$$

Obviously, $( 2 M ) ^ { 1 / k } \gamma _ { i } ^ { - 1 / k } ( 2 \pi k ) ^ { 1 / ( 2 k ) } \to 1$ when $k \to \infty$ . So the radius of convergence is $- \ln \gamma _ { i } / 2$ ,   
which is independent of $\pi _ { j } ( y )$ and $\sigma _ { j } ( y )$ . This means that when $| a | < - \ln \gamma _ { i } \mathrm { / 2 } , h _ { i , x } ( \theta _ { a } )$ is an   
analytic function with respect to $a$ . Since the overall analyticity can be decomposed into the analyticity   
of segments, $h _ { i , x }$ is analytic in the whole $[ 0 , 1 ]$ .

# 594 Prove that any sub-game has an equilibrium.

Construct a group set

$$
A _ { i } = \{ ( i , x ) \mid \forall x \in X \} , \quad A = \{ A _ { i } \mid \forall i \in I \} .
$$

So 595 $A _ { i }$ is actually a real player in the original stochastic game who need choose a probability 596 distribution over $S _ { i }$ for each state in $X$ .

For any joint strategy $\begin{array} { r } { \pi \in \prod _ { ( i , x ) \in ( I , X ) } T _ { i , x } } \end{array}$ , a subset $\Omega \subset A$ , there exists $J \subset I$ satisfying

$$
\bigcup _ { \omega \in \Omega } \omega = ( J , X ) .
$$

Construct a sub-game $U = \left( ( J , X ) , ( T _ { i , x } ) _ { ( i , x ) \in ( J , X ) } , \left( u _ { i , x } \right) _ { ( i , x ) \in ( J , X ) } \right)$ where

$$
u _ { i , x } : ( T _ { i , x } ) _ { ( i , x ) \in ( J , X ) } \to \mathbb { R } , \quad \sigma \mapsto h _ { i , x } \big ( \pi _ { - ( J , X ) } , \sigma \big ) .
$$

So

$$
u _ { i , x } ( \sigma ) = \sum _ { t \geq 0 } \gamma _ { i } ^ { t } \mathbb { E } _ { x ^ { 0 } = x , \ldots , x ^ { u } \sim P ( x ^ { u - 1 } , s ^ { u - 1 } ) , s ^ { u } \sim \left( \pi _ { ( T \setminus J , X ) } , \sigma \right) } ( x ^ { u } ) , \ldots , s ^ { t } \sim \left( \pi _ { ( T \setminus J , X ) } , \sigma \right) ( x _ { t } ) g _ { i } ( x ^ { t } , s ^ { t } ) .
$$

Construct a stationary mixed strategy stochastic game $V = ( J , ( S _ { i } ) _ { i \in J } , X , Q , ( v _ { i } ) _ { i \in I } , ( \gamma _ { i } ) _ { i \in I } ) .$ .

$$
\forall s \in \prod _ { i \in J } S _ { i } , \quad \forall x \in X , \quad \forall y \in X , \quad Q ( x , s ) ( y ) = \sum _ { w \in \prod _ { i \in I \setminus J } S _ { i } } \pi _ { ( I \setminus J , X ) } ( x ) ( w ) P ( x , s , w ) ( y ) .
$$

$$
\forall i \in J , \quad \forall s \in \prod _ { i \in J } S _ { i } , \quad \forall x \in X , \quad v _ { i } ( x , s ) = \sum _ { w \in \prod _ { i \in I \setminus J } S _ { i } } \pi _ { ( I \setminus J , X ) } ( x ) ( w ) g _ { i } ( x , s , w ) .
$$

Induction needs to be used here.

Head. Choose σ ∈ (Ti,x)(i,x)∈(J,X).

$$
\begin{array} { l } { \mathbb { E } _ { { x ^ { 0 } } = x , s ^ { 0 } \sim \sigma ( x ^ { 0 } ) } v _ { i } ( x ^ { 0 } , s ^ { 0 } ) = \displaystyle \sum _ { s ^ { 0 } \in \prod _ { i \in J } S _ { i } } \sigma ( x ^ { 0 } ) ( s ^ { 0 } ) v _ { i } ( x ^ { 0 } , s ^ { 0 } ) } \\ { = \displaystyle \sum _ { s ^ { 0 } \in \prod _ { i \in J } S _ { i } } \sigma ( x ^ { 0 } ) ( s ^ { 0 } ) \sum _ { w \in \prod _ { i \in I \setminus J } S _ { i } } \pi _ { ( I \setminus J , X ) } ( x ^ { 0 } ) ( w ) g _ { i } ( x ^ { 0 } , s ^ { 0 } , w ) } \\ { = \mathbb { E } _ { { x ^ { 0 } } = x , r ^ { 0 } \sim ( \pi _ { ( I \setminus J , X ) } , \sigma ) ( x ^ { 0 } ) } g _ { i } ( x ^ { 0 } , r ^ { 0 } ) } \end{array}
$$

where 599 $r ^ { 0 } = ( w , s ^ { 0 } )$

Recursion. Assume for any $x ^ { 0 } \in X$ , there exists

$$
\begin{array} { r } { \mathbb { E } _ { . . . , x ^ { u } \sim Q Q ( x ^ { u - 1 } , s ^ { u - 1 } ) , s ^ { u } \sim \sigma ( x ^ { u } ) } v _ { i } \big ( x ^ { u } , s ^ { u } \big ) = \mathbb { E } _ { . . . , x ^ { u } \sim P ( x ^ { u - 1 } , r ^ { u - 1 } ) , r ^ { u } \sim \big ( \pi _ { ( I \setminus J , X ) } , \sigma \big ) ( x ^ { u } ) } g _ { i } \big ( x ^ { u } , r ^ { u } \big ) . } \end{array}
$$

Consider the case $u + 1$ . To simplify notations, let

$$
\begin{array} { r l } & { \xi ^ { u } ( x ) = \mathbb { E } _ { x ^ { 0 } = x , \ldots , x ^ { u } \sim Q ( x ^ { u - 1 } , s ^ { u - 1 } ) , s ^ { u } \sim \sigma ( x ^ { u } ) } v _ { i } ( x ^ { u } , s ^ { u } ) . } \\ & { \zeta ^ { u } ( x ) = \mathbb { E } _ { x ^ { 0 } = x , \ldots , x ^ { u } \sim P ( x ^ { u - 1 } , r ^ { u - 1 } ) , r ^ { u } \sim \left( \pi _ { ( I \setminus J , X ) } , \sigma \right) ( x ^ { u } ) } g _ { i } ( x ^ { u } , r ^ { u } ) . } \end{array}
$$

So

$$
\begin{array} { r l } & { \xi ^ { u + 1 } ( x ) = \displaystyle \sum _ { s ^ { 0 } \in \prod _ { i \in J } S _ { i } } \sigma ( x ) ( s ^ { 0 } ) \sum _ { x ^ { 1 } \in X } Q ( x , s ^ { 0 } ) ( x ^ { 1 } ) \xi ^ { u } ( x ^ { 1 } ) } \\ & { \qquad = \displaystyle \sum _ { s ^ { 0 } \in \prod _ { i \in J } S _ { i } } \sigma ( x ) ( s ^ { 0 } ) \sum _ { x ^ { 1 } \in X } \xi ^ { u } ( x ^ { 1 } ) \sum _ { w \in \prod _ { i \in I \setminus J } S _ { i } } \pi _ { ( I \setminus J , X ) } ( x ) ( w ) P ( x , s ^ { 0 } , w ) ( x ^ { 1 } ) } \\ & { \qquad = \displaystyle \sum _ { s ^ { 0 } \in \prod _ { i \in J } S _ { i } } \sum _ { w \in \prod _ { i \in I \setminus J } S _ { i } } \sum _ { x ^ { 1 } \in X } \sigma ( x ) ( s ^ { 0 } ) \pi _ { ( I \setminus J , X ) } ( x ) ( w ) P ( x , s ^ { 0 } , w ) ( x ^ { 1 } ) \zeta ^ { u } ( x ^ { 1 } ) } \\ & { \qquad = \displaystyle \zeta ^ { u + 1 } ( x ) . } \end{array}
$$

So the case 601 $u + 1$ is valid no matter what $x ^ { 0 } = x $ is.

As a result

$$
u _ { i , x } ( \sigma ) = \sum _ { t \geq 0 } \gamma _ { i } ^ { t } \mathbb { E } _ { x ^ { 0 } = x , \ldots , x ^ { u } \sim Q ( x ^ { u - 1 } , s ^ { u - 1 } ) , s ^ { u } \sim \sigma ( x ^ { u } ) , \ldots , s ^ { t } \sim \sigma ( x _ { t } ) } v _ { i } ( x ^ { t } , s ^ { t } ) .
$$

602 This means that the game $U$ is actually the extended form of the game $V$ .

03 There exists a theorem with respect to stationary mixed strategy stochastic games [13]. Each finite  
604 state Markov game has a mixed equilibrium.   
5 As a result, the game $V$ has a mixed equilibrium, which means that the game $U$ also has an   
equilibrium.

To sum up, all conditions of Theorem 2 are satisfied. So in the game $H$ , for any initial strategy $\pi$ , there exists a finite-length grouped satisficing path $( \sigma ^ { t } ) _ { t = 0 } ^ { T }$ where $\sigma ^ { 0 } = \pi$ and $\sigma ^ { \dot { T } }$ is an equilibrium.

Since the group set $A$ actually views a player in different state as a group, if a group achieves best response, the corresponding player will achieve best response in any state. Otherwise there exists a state where this player does not achieve best response.

As a result, for any initial stationary joint mixed strategy $\sigma$ in $G$ , there exists a finite-length satisficing path $( \sigma ^ { t } ) _ { t = 0 } ^ { T }$ where $\sigma ^ { 0 } = \sigma$ and $\sigma ^ { T }$ is a mixed equilibrium.

# B.4 Proof of Corollary 4

Corollary 4. In a $k$ -step mixed strategy stochastic game $G$ , suppose that each $S _ { i }$ is a finite set. Then for any initial stationary joint mixed strategy 617 $\sigma$ , there exists a finite-length satisficing path $( \sigma ^ { t } ) _ { t = 0 } ^ { T }$ where 618 $\sigma ^ { 0 } = \sigma$ and $\sigma ^ { T }$ is a mixed equilibrium.

Proof. Construct a stationary mixed strategy stochastic game $H$ .

$$
\begin{array} { c } { { H = ( I , ( S _ { i } ) _ { i \in I } , Y , Q , ( h _ { i } ) _ { i \in I } , ( \gamma _ { i } ) _ { i \in I } ) . } } \\ { { { } } } \\ { { Y = X \times \displaystyle \left( \prod _ { i \in I } S _ { i } \right) ^ { k } . } } \end{array}
$$

Obviously, $Y$ is finite, $| Y | < \infty$

$$
\begin{array} { r l r } {  { Q : Y \times \prod _ { i \in I } S _ { i } \to \Delta Y , } } \\ & { } & \\ & { } & { ( x , ( s ^ { - t } ) _ { t = 1 } ^ { k } ) \times s } \\ & { } & { \mapsto \{ ( ( y , ( u ^ { - t } ) _ { t = 1 } ^ { k } ) , P ( x , s ) ( y ) I [ u ^ { - 1 } = s ] \prod _ { t = 2 } ^ { k } I [ u ^ { - t } = s ^ { - t + 1 } ] ) \mid ( y , ( u ^ { - t } ) _ { t = 1 } ^ { k } ) \in Y \} . } \end{array}
$$

where $I [ \cdot ]$ equals 1 if $[ \cdot ]$ is true else 0.

$$
h _ { i } : Y \times \prod _ { i \in I } S _ { i } \to \mathbb { R } , \quad \left( x , ( s ^ { - t } ) _ { t = 1 } ^ { k } \right) \times s \mapsto g _ { i } ( x , s ) .
$$

So the game $H$ is the extended form of the game $G$ . When the runtime of the game $G$ exceeds $k$ time   
steps and the first $k$ steps of any path in $G$ are not be considered, each path in $G$ can be project onto a   
certain and unique path in $H$ , and this projection is actually bijective.   
According to Corollary 3, for any initial stationary joint mixed strategy $\sigma$ in $H$ , there exists a finite  
length satisficing path $( \sigma ^ { t } ) _ { t = 0 } ^ { T }$ where $\sigma ^ { 0 } = \sigma$ and $\sigma ^ { T }$ is a mixed equilibrium in $H$ . In turn, this   
conclusion is also true in $G$ .

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: Primary findings are established in Theorem 1 and Theorem 2. Four corollaries provide complementary perspectives for applying these main theorems in different contexts.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the paper.   
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.   
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In Section 5, the theoretical implications and limitations of the main findings are analyzed, and their connections to existing literature are discussed.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.   
• The authors are encouraged to create a separate "Limitations" section in their paper.   
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.   
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.   
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The proofs of Theorem 1 and Theorem 2 appear in the main text, while those of lemmas and corollaries are deferred to the appendix.

Guidelines:

• The answer NA means that the paper does not include theoretical results.   
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.   
• All assumptions should be clearly stated or referenced in the statement of any theorems.   
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.   
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.   
• Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: This paper does not contain experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.   
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.   
Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.   
While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: This paper does not contain experiments.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.   
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.   
• While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).   
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.   
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.   
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).   
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: This paper does not contain experiments.

Guidelines:

• The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: This paper does not contain experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.   
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).   
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)   
• The assumptions made should be given (e.g., Normally distributed errors).   
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.   
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a $96 \%$ CI, if the hypothesis of Normality of errors is not verified.   
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).   
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: This paper does not contain experiments.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.   
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

# 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: This paper strictly adheres to the ethical guidelines established by NeurIPS in all aspects.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.   
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.   
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: While this paper establishes a theoretical foundation for MARL algorithm design, its broader societal implications remain beyond the scope of current assessment.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.   
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.   
Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.   
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.   
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.   
If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: This paper does not pose such risks.

Guidelines:

• The answer NA means that the paper poses no such risks.   
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This paper does not use existing assets.

Guidelines:

• The answer NA means that the paper does not use existing assets.   
• The authors should cite the original paper that produced the code package or dataset.   
• The authors should state which version of the asset is used and, if possible, include a URL.   
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.   
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.   
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.   
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

# 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: This paper does not release new assets.

Guidelines:

• The answer NA means that the paper does not release new assets.   
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
• The paper should discuss whether and how consent was obtained from people whose asset is used.   
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.   
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

# 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this paper does not involve LLMs as any important, original, or non-standard components.

Guidelines:

• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
