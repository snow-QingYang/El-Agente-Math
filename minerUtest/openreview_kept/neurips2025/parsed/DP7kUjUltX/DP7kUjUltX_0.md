## LINE 97

high compression with minimal performance loss.

# 88 2 Background

Let $\mathbb { K }$ be a set of quantized numbers, and let $n$ denote the bitwidth of $\mathbb { K }$ , that is, the number of bits   
required to represent a number in ${ \mathbb K }$ . The bitwidth $n$ is written in unary; this is motivated by the fact   
that $n$ is small and that we would in any case need to allocate $n$ -bit consecutive memory for storing a   
number. Formally, we consider a sequence $\mathbb { K } _ { 1 } , \mathbb { K } _ { 2 } , \ldots$ corresponding to bitwidths 1, 2, etc., but we   
retain the notation ${ \mathbb K }$ for simplicity. We suppose that ${ \mathbb K }$ saturates: e.g., if $x \geq 0$ , $y \geq 0$ , $x + y \ge 0$   
(i.e., no modulo behavior like in int in C for instance). We suppose that $1 \in \mathbb { K }$ .   
We consider Aggregate-Combine Graph Neural Networks with global Readout (ACR-GNNs), a   
standard class of message-passing GNNs [4, 16]. An ACR-GNN layer is defined by a triple   
$( c o m b , a g g , a g g _ { g } )$ , where comb : $\mathbf { \mathbb { K } } ^ { 3 m } \ \to \ \mathbb { K } ^ { n }$ is a combination function, and agg, $a g g _ { g }$ are lo  
cal and global aggregation functions that map multisets of vectors in $\mathbb { K } ^ { m }$ to a single vector in   
$\mathbb { K } ^ { m }$ .

An ACR-GNN is composed of a sequence of such
