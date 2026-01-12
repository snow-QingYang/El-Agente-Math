## LINE 81

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

This paper adopts a general theoretical perspective
