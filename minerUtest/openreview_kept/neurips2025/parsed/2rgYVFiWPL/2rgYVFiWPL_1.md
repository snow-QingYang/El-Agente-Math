## LINE 12

Sample complexity of Schrödinger potential estimation

Anonymous Author(s)   
Affiliation   
Address   
email

# Abstract

We address the problem of Schrödinger potential estimation, which plays a crucial   
role in modern generative modelling approaches based on Schrodinger bridges and   
stochastic optimal control for SDEs. Given a simple prior diffusion process, these   
methods search for a path between two given distributions $\rho _ { 0 }$ and $\rho _ { T }$ requiring min  
imal efforts. The optimal drift in this case can be expressed through a Schrödinger   
potential. In the present paper, we study generalization ability of an empirical   
Kullback-Leibler (KL) risk minimizer over a class of admissible log-potentials   
aimed at fitting the marginal distribution at time $T$ . Under reasonable assumptions   
on the target distribution $\rho _ { T }$ and the prior process, we derive a non-asymptotic   
high-probability upper bound on the KL-divergence between $\rho _ { T }$ and the terminal   
density corresponding to the estimated log-potential. In particular, we show that   
the excess KL-risk may decrease as fast as $\mathcal { O } ( \log n / n )$ when the sample size $n$   
tends to infinity even if both $\rho _ { 0 }$ and $\rho _ { T }$ have unbounded supports.

# 14 1 Introduction

The Schrödinger Bridge problem (SBP) originates from a question posed by Erwin Schrödinger in   
1932 [Schrödinger, 1932], seeking the most likely evolution of a probability distribution between   
two given endpoint distributions while minimizing
