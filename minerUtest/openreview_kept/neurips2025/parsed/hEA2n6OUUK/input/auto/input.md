# Outlier-Robust Phase Retrieval in Nearly-Linear Time

Anonymous Author(s)   
Affiliation   
Address   
email

# Abstract

Phase retrieval is a fundamental problem in signal processing, where the goal is to   
recover a (complex-valued) signal from phaseless intensity measurements. In this   
paper, we propose and study the (real-valued) outlier-robust phase retrieval problem.   
Specifically, we seek to recover a vector $\boldsymbol { x } \in \mathbb { R } ^ { d }$ from $n$ intensity measurements   
$\bar { y _ { i } } = ( a _ { i } ^ { \top } x ) ^ { 2 }$ , where a small fraction of the $( a _ { i } , y _ { i } )$ pairs are adversarially corrupted.   
Our main result is a near-sample-optimal and nearly-linear-time algorithm that   
provably recovers the ground-truth vector. Our algorithm first solves a lightweight   
convex program to find an initial point close to the ground truth, and then runs   
a robust version of gradient descent to achieve exact recovery. Our approach is   
conceptually simple and provides a framework for developing robust algorithms   
for other non-convex optimization problems.

# 12 1 Introduction

Phase retrieval is a fundamental problem in signal processing with applications in various fields, in  
cluding electron microscopy [35], crystallography [36, 39], astronomy [13], and optical imaging [40].   
In these applications, one often has access to only the magnitudes of the Fourier transforms of a   
complex signal. This is because measuring magnitude (e.g., by aggregating energy over time) is   
much easier than measuring phase (which requires detecting rapid changes). We refer the reader to   
the survey articles [40, 29] for more details about the theory and applications of phase retrieval.   
In this paper, we focus on the (real-valued) generalized phase retrieval problem, where we are given   
intensity measurements of an arbitrary linear operator.

Definition 1.1 (Phase Retrieval). Let 21 $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be the ground-truth vector. Let $a _ { 1 } , \dots , a _ { n } \in \mathbb { R } ^ { d } b e$ 22 n sampling vectors and let $y _ { i } = \left. a _ { i } , x \right. ^ { 2 } \in \mathbb { R }$ be the corresponding intensity measurements. Given 23 $( a _ { i } , y _ { i } ) _ { i = 1 } ^ { n }$ as input, the task is to recover $x$ .

Note that it is impossible to distinguish between $x$ and $- x$ , thus recovering either is sufficient. This   
problem has been extensively studied by the machine learning community. Under certain assumptions   
on the distribution of the sampling vectors, such as when the $a _ { i }$ ’s are independent and Gaussian   
distributed, $\Theta ( d )$ input pairs $( a _ { i } , y _ { i } )$ are necessary and sufficient for exact recovery [8]. Additionally,   
it has been shown that the problem can be solved in linear time with respect to the size of the input   
[8]. This was first achieved using semidefinite programming (SDP) relaxations [6]. In practice, the   
problem is often solved by applying first-order optimization methods (e.g., gradient descent) to a   
suitable objective function such as

$$
\operatorname* { m i n } _ { z \in \mathbb { R } ^ { d } } \quad f ( z ) = \sum _ { i = 1 } ^ { n } ( y _ { i } - \left. a _ { i } , z \right. ^ { 2 } ) ^ { 2 } .
$$

Even though many natural formulations of the phase retrieval objective are nonconvex, including   
the one in (1), prior work has shown that, depending on the input distribution, they may not have   
spurious local optima and thus can be solved using first-order optimization methods [37, 4, 43].   
However, these analyses of the objective landscape rely on strong assumptions, such as the sampling   
vectors $a _ { i }$ being i.i.d. Gaussian. Our work is motivated by the following questions: Can we relax   
the assumptions used to prove landscape results for tractable nonconvex problems? In the context of   
phase retrieval, can we recover the ground-truth vector $x$ when a small subset of the $( a _ { i } , y _ { i } )$ pairs are   
adversarially corrupted?   
Our focus on this adversarial setting, where an $\epsilon$ -fraction of the input data is corrupted, is inspired   
by recent advances in high-dimensional robust statistics. An example problem in robust statistics   
is to estimate the mean of a $d$ dimensional spherical Gaussian when $\epsilon$ -fraction of the samples are   
arbitrarily corrupted. The goal of high-dimensional statistics is often to design efficient algorithms   
that can achieve dimension-independent error guarantees. Early work in robust statistics [45, 26, 28]   
provided sample-efficient estimators for various tasks, but with runtimes exponential in the dimension.   
A recent line of work initiated by [16] and [31] has developed computationally efficient robust   
algorithms for many fundamental statistical and learning tasks. Significant progress has been made in   
the algorithmic aspects of robust high-dimensional statistics (see, e.g., [15]).

We now formally define the $\epsilon$ -corruption model we study in this paper. For clarity, we define it directly in the context of phase retrieval.

Definition 1.2 $\epsilon$ -Corrupted Set of Samples). Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be the ground-truth vector. Let $\epsilon > 0$ . First,   
the algorithm specifies the sample complexity $n$ . Then, n sampling vectors $( a _ { 1 } , \ldots , a _ { n } )$ are drawn   
from some known distribution $D$ and the corresponding intensity measurements $y _ { i } = \stackrel { \cdot } { \left. { a _ { i } , x } \right. } ^ { 2 }$ are   
calculated. The adversary is allowed to replace ϵn pairs $( a _ { i } , y _ { i } )$ with arbitrary data. We call a set of   
$( a _ { i } , y _ { i } )$ pairs $\epsilon$ -corrupted if it is generated by this process.   
In this paper, we say samples to refer to $( a _ { i } , y _ { i } )$ pairs. Note that we allow corruption in both the   
sampling vectors $a _ { i } \in \mathbb { R } ^ { d }$ and the intensity measurements $y _ { i } \in \mathbb { R }$ , as long as the fraction of corruption   
is at most $\epsilon$ . We now formally define outlier-robust phase retrieval, the main problem we study in this   
paper.   
Problem 1.3 (Outlier-Robust Phase Retrieval). Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be the ground-truth with $\| x \| _ { 2 } = 1$ . Let   
$\epsilon > 0$ . Let $( a _ { 1 } , \ldots , a _ { n } )$ be n sampling vectors drawn i.i.d. from $\mathcal { N } ( 0 , I ) \in \mathbb { R } ^ { d }$ , and let $y _ { i } = \langle a _ { i } , x \rangle ^ { 2 }$   
be the corresponding intensity measurements. An adversary arbitrarily corrupts an $\epsilon$ -fraction of these   
$( a _ { i } , y _ { i } )$ pairs, and gives the ϵ-corrupted set of samples as input to the algorithm. The task is to find $a$   
vector $z \in \mathbb { R } ^ { d }$ such that $\operatorname* { m i n } \{ \left\| z - x \right\| _ { 2 } , \left\| z + x \right\| _ { 2 } \} \leq \Delta$ for some precision parameter $\Delta > 0$ .   
A simpler adversary setting where corruption is restricted to the intensity measurements $y _ { i }$ has   
been studied previously [27, 51, 19]. In our work, we study a more comprehensive setting where   
we also allow corruption in the measurement vectors. This models realistic scenarios where the   
measurement process is affected by hardware noise, miscalibration, or adversarial tampering, leading   
to perturbations in the sampling vectors $a _ { i }$ as well as in $y _ { i }$ . Concurrent work [3] studies this same   
problem. However, their algorithm uses a black-box subroutine for robust covariance estimation and   
thus requires at least $\Omega ( d ^ { 2 } ) $ time, making it impractical in high dimensions.   
We are interested in designing a scalable and provably robust algorithm for Problem 1.3. We would   
like to resolve the following question:

Can we design a provably robust algorithm for the outlier-robust phase retrieval problem (Problem 1.3) that has near-optimal sample complexity and runs in nearly-linear time?

# 77 1.1 Our Results

We answer the question outlined in the previous subsection affirmatively.

Theorem 1.4 (Main). Consider the outlier-robust phase retrieval problem as defined in Problem 1.3, where $\boldsymbol { x } \in \mathbb { R } ^ { d }$ is the ground-truth vector. Let $0 < \epsilon \le \epsilon _ { 0 }$ for sufficiently small universal constant $\epsilon _ { \mathrm { 0 } }$ . Let $\Delta > 0$ be the desired precision. Given an $\epsilon$ -corrupted set of $n = \widetilde { \Omega } ( d \log ^ { 2 } ( 1 / \Delta ) )$ samples, we can compute $z \in \mathbb { R } ^ { d }$ in time $\widetilde O ( n d )$ such that $\operatorname* { m i n } ( \left\| z - x \right\| _ { 2 } , \left\| z + x \right\| _ { 2 } ) \leq \Delta$ with probability at least 0.95. 1

A few remarks are in order regarding Theorem 1.4. First, even without corruption, phase retrieval   
requires $\Omega ( d )$ samples [8]. When 1poly(d) , Theorem 1.4 requires Oe(d) samples, runs in nearly   
linear time (since the input size is $O ( n d ) {  }$ ), and achieves exact recovery. Therefore, our algorithm   
simultaneously achieves the best possible error, sample complexity, and runtime (up to logarithmic   
factors).   
Second, the success probability in Theorem 1.4 can be boosted to $1 - \tau$ for any $\tau > 0$ by incurring   
an additional $O ( \log ( 1 / \tau ) )$ factor in the sample complexity and runtime. This can be achieved, for   
example, by partitioning the input, repeating the algorithm, letting candidate solutions vote for those   
within distance $2 \Delta$ , and finally selecting the solution with the most votes.   
Lastly, $\epsilon _ { \mathrm { 0 } }$ in Theorem 1.4 is an absolute constant that is independent of $n$ or $d$ , and our algorithm   
works for any corruption level $0 \leq \epsilon \leq \epsilon _ { 0 }$ . An important observation is that the optimal sample   
complexity is $\Theta ( d )$ , which is independent of $\epsilon$ . This follows from the fact that exact recovery is   
possible as long as the clean samples provide enough constraints to fully determine $x$ , which has   
$d - 1$ degrees of freedom. 2 This is in contrast to problems in robust high-dimensional statistics, such   
as robust mean estimation, where exact recovery is impossible with a finite number of samples (even   
without corruption).   
Since our algorithm guarantees exact recovery (to arbitrary precision $\Delta$ ) for any corruption level $\epsilon$ as   
long as $\epsilon < \epsilon _ { 0 }$ , any input with corruption level $\epsilon < \epsilon _ { 0 }$ can be treated as if it were corrupted at a fixed   
level $\epsilon _ { \mathrm { 0 } }$ . This explains why neither the sample complexity nor the runtime of our algorithm depends   
on $\epsilon$ . For simplicity, we refer to the corruption level $\epsilon$ as a sufficiently small universal constant   
throughout the remainder of the paper unless otherwise noted.

# 05 1.2 Our Approach and Techniques

106 When there are infinitely many samples and no corruption, the objective function $f ( z )$ simplifies to

$$
f ( z ) = \underset { a \sim \mathcal { N } ( 0 , I _ { d } ) } { \mathbb { E } } \Big [ ( \langle a _ { i } , z \rangle ^ { 2 } - y _ { i } ) ^ { 2 } \Big ] = 3 \| x \| _ { 2 } ^ { 4 } + 3 \| z \| _ { 2 } ^ { 4 } - 2 \| x \| _ { 2 } ^ { 2 } \| z \| _ { 2 } ^ { 2 } - 4 \langle x , z \rangle ^ { 2 } .
$$

07 Despite being nonconvex, it is known that $f$ has no spurious local optima [37, 4, 43].

Our approach follows a two-step structure used in many local convergence results for nonconvex   
problems (e.g., Candès et al. [4]), where the goal is to first initialize into a region free of saddle points   
and then perform gradient descent. Both steps are vulnerable to adversarial attacks and we develop   
provably robust and nearly-linear time algorithms for both steps. In the first step, we use spectral   
techniques to obtain an initial guess that is sufficiently close to the ground truth. In the second step,   
we run a robust gradient descent algorithm to refine this guess and converge to the final solution.   
Step 1: Robust Spectral Initialization. Consider the following matrix 114 $\begin{array} { r } { Y = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } y _ { i } a _ { i } a _ { i } ^ { \top } } \end{array}$   
where $y _ { i } = \left. a _ { i } , x \right. ^ { 2 }$ . When there is no corruption and the $a _ { i }$ ’s are drawn i.i.d. from $\mathcal { N } ( 0 , I )$ ,   
we have $\mathbb { E } [ Y ] = I + 2 x x ^ { \top }$ . Hence, when there are enough samples and no corruption, we can   
obtain a good estimate of the ground truth $x$ (or $- x )$ by computing the largest eigenvector of $Y$   
However, the corrupted $( a _ { i } , y _ { i } )$ pairs can arbitrarily change the largest eigenvector of $Y$ . One natural   
approach, which was explored in concurrent work [3], is to apply known robust covariance estimation   
algorithms [11, 1] to estimate $Y$ . While this can recover the top eigenvector, the runtime is at least   
$\bar { \Omega ( d ^ { 2 } ) }$ , which is too slow for our goal of designing a nearly linear time algorithm.

One of the main technical insights of our work is that it is not necessary to robustly estimate the entire matrix sample $Y$ , we only need to recover its few largest eigeuch that the weighted intensity-based matrix n a weight  is close to $w _ { i }$ $\begin{array} { r } { Y _ { w } ^ { ' } = \sum _ { i = 1 } ^ { n } w _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } } \end{array}$ $I + 2 x x ^ { \top }$ vector $x$ is unknown, and explicitly computing $Y _ { w }$ via fast matrix multiplication takes $\bar { \Omega } ( d ^ { 2 } )$ time.

A key observation is that the corrupted samples can only add directions to $Y _ { w }$ but cannot remove directions, because each individual term $y _ { i } a _ { i } \bar { a } _ { i } ^ { \top }$ is a PSD matrix. (We assume w.l.o.g. that $y _ { i } \geq 0$ for

all $i$ , because any input with $y _ { i } < 0$ must be corrupted.) Consequently, if we can compute a weight   
vector $w$ that minimizes the sum of the top two eigenvalues of $Y _ { w }$ (which is a convex optimization   
problem), we can recover a matrix that is close to the unknown unbiased expectation $I + 2 x x ^ { \top }$ .   
We show that this optimization problem can be solved in $\widetilde O ( n d )$ time by leveraring algorithmic   
techniques developed for list-decodable mean estimation [12].   
Step 2: Robust Gradient Descent. Starting with the initial guess $z$ given by the robust spectral   
initialization, we want to run gradient descent to recover the ground truth $x$ . Without corruption,   
if the initialization is close enough to $x$ , each iteration will bring $z$ closer to $x$ by a constant factor.   
Intuitively, approximating the gradient at a specific point amounts to a robust mean estimation   
problem (for the underlying distribution of the gradients). When the input data is $\epsilon$ -corrupted, the   
gradients of the $n$ samples can be viewed as an $\epsilon$ -corrupted set of vectors.   
We can approximate the true gradient by running robust mean estimation on this $\epsilon$ -corrupted set of $n$   
gradients. To show convergence, the estimation of the gradient needs to be more accurate as we get   
closer to the optimal solution, and we show that this is possible because the variance of the gradient   
on clean samples decreases as the solution gets closer to the optimal solution.

# 144 1.3 Related and Prior Works

Phase Retrieval. Phase retrieval arises in various fields of science and engineering [13, 36]. Early research introduced error-reduction algorithms [25, 20, 21]. Convex and nonconvex optimization with various objective functions were later proposed and achieved exact recovery [47, 4–6, 48, 49, 41].

Outlier-Robust Phase Retrieval. Robust phase retrieval has been explored in the literature [51, 30, 8, 7, 34]. A simpler setting where corruption is restricted to the intensity measurements $y _ { i }$ has been studied previously [27, 51, 19]. In Appendix C, we show that methods developed for this setting do not work in ours. Concurrent work [3] is the only one that studies the general corruption model that we consider, allowing corruption in both the sampling vectors $a _ { i }$ and the intensity measurements $y _ { i }$ . The algorithm in [3] achieves near-optimal sample complexity, but relies on robust covariance estimation in a black-box manner, resulting in a slower runtime compared to ours. We emphasize that algorithm runs in nearly-linear time and achieves near-optimal sample complexity, which demonstrates that allowing corruption in both $a _ { i }$ and $y _ { i }$ incurs almost no penalty (asymptotically) in terms of statistical or computational complexity.

Nonconvex Optimization. Besides phase retrieval, it is known that all local optima are globally   
optimal for natural nonconvex formulations of various learning problems, such as matrix completion   
[24], matrix sensing [2], dictionary learning [42], and tensor decomposition [23] (we refer the   
interested reader to Chapter 7 of the book by [50]). A recent line of work explored the robustness of   
such landscape results: [33] studied matrix sensing in the $\epsilon$ -corruption model, [9] and [22] studied   
semi-random matrix completion and matrix sensing.   
High-Dimensional Robust Statistics. Recent works developed nearly-linear time algorithms for   
robust mean estimation [10, 18, 32]. The robust gradient descent algorithm we use is closely related   
to algorithms proposed in previous works for finding first-order stationary points in robust stochastic   
optimization [38, 17].

# 168 2 Preliminary and Background

Notation. Let $[ n ] = \{ 1 , 2 , \dots , n \}$ . For a vector $x$ , we denote its $i ^ { t h }$ coordinate by $x _ { i }$ . We use $\lVert x \rVert _ { 1 }$ , $\left. { x } \right. _ { 2 }$ , and $\| { \boldsymbol { x } } \| _ { \infty }$ to denote the $\ell _ { 1 } , \ell _ { 2 }$ , and $\ell _ { \infty }$ norm of $x$ , respectively. For two vectors $x$ and $y$ , we use $\langle x , y \rangle = x ^ { \top } y$ to denote their inner product.

We write $I$ for the identity matrix. For a matrix $A$ , we use $\| A \| _ { 2 }$ to denote its spectral norm. A   
symmetric matrix $A$ is positive semidefinite (PSD) if $x ^ { \top } A x \geq 0$ for all vectors $x$ . For two symmetric   
matrices $A$ and $B$ , we write $A \preceq B$ if $B - A$ is PSD. We write $\lambda _ { k } ( A )$ as the $k ^ { t h }$ largest eigenvalue   
of $A$ , and $\overline { { \lambda } } _ { k } ( A )$ as the sum of the $k$ largest eigenvalues of $A$ . The Ky Fan $k$ -norm of a matrix $A$ is   
the sum of its $k$ largest singular values, which is equal to $\overline { { \lambda } } _ { k } ( A )$ when $A$ is PSD.

Ky Fan Norm Packing SDP. In our robust spectral initialization step, we solve a Ky Fan norm 178 packing semidefinite program (SDP) of the following form:

$$
\operatorname* { m a x } _ { w \in \mathbb { R } _ { \geq 0 } ^ { n } } \quad \| w \| _ { 1 } \quad \mathrm { s u b j e c t t o } \quad \quad \sum _ { i = 1 } ^ { n } w _ { i } A _ { i } \preceq I , \quad \overline { { \lambda } } _ { k } \left( \sum _ { i = 1 } ^ { n } w _ { i } B _ { i } \right) \leq k
$$

179 We use the nearly-linear time Ky Fan norm SDP solver from [12].

Lemma 2.1 (Ky Fan Norm SDP Solver [12]). Given an $S D P \left( ^ { * * } \right)$ with positive semidefinite matrices $A _ { i } \in \mathbb { R } ^ { d _ { 1 } \times d _ { 1 } }$ and $B _ { i } \in \mathbb { R } ^ { d _ { 2 } \times d _ { 2 } }$ with $\bar { A _ { i } } = \bar { C _ { i } } \bar { C _ { i } ^ { \top } }$ and $B _ { i } = \dot { D _ { i } } D _ { i } ^ { \top }$ for all $i \in [ n ]$ , integer $k > 0$ , error tolerance $\epsilon _ { 1 } \geq 1 / n ^ { 2 }$ , and failure probability $\tau > 0$ , one can in time $\widetilde O ( ( t _ { C } + t _ { D } + d _ { 1 } +$ $d _ { 2 } ) \mathrm { p o l y } ( 1 / \epsilon _ { 1 } , \mathrm { l o g } ( 1 / \tau ) ) ;$ ) output $w ^ { \prime } \in \mathbb { R } _ { \geq 0 } ^ { n }$ such that $\| w ^ { \prime } \| _ { 1 } \geq ( 1 - \epsilon _ { 1 } ) 0 \mathsf { P T }$ with probability at least $1 - \tau$ . Here OPT is the optimal value of $( ^ { * * } )$ , $t _ { C _ { i } }$ and $t _ { D _ { i } }$ are the time taken to perform $a$ matrix-vector product with $C _ { i }$ and $D _ { i }$ respectively, and $\textstyle t _ { C } = \sum _ { i = 1 } ^ { n } t _ { C _ { i } }$ and $\textstyle t _ { D } = \sum _ { i = 1 } ^ { n } { \bar { t } } _ { D _ { i } }$ .

Top Eigenvector Computation. We use the power method to compute an approximate top eigenvector. We refer to the analysis of the power method for PSD matrices by Trevisan [44].

Lemma 2.2 (Top Eigenvector via Power Method [44]). Let $A \in \mathbb { R } ^ { d \times d }$ be a PSD matrix. Let $\lambda _ { 1 }$ be the largest eigenvalue of $A$ . For any $\epsilon _ { 2 } > 0$ , one can compute a unit vector $\boldsymbol { x } \in \mathbb { R } ^ { d }$ in time $O ( ( t _ { A } + \bar { d } ) \log ( \bar { d } ) / \epsilon _ { 2 } )$ such that $x ^ { \top } A x \geq ( 1 - \epsilon _ { 2 } ) \lambda _ { 1 }$ with probability at least 0.99, where $t _ { A }$ is the time taken to perform a matrix-vector multiplication with $A$ .

Robust Mean Estimation. In the robust gradient descent step, we use nearly-linear time robust mean estimation algorithms for bounded-covariance distributions [10, 14, 18] to approximate the true gradient.

Lemma 2.3 (Robust Mean Estimation [18]). Let $D$ be a distribution on $\mathbb { R } ^ { d }$ with unknown mean $\mu$ and unknown covariance matrix $\Sigma$ where $\Sigma \preceq \sigma ^ { 2 } I$ . Let $\epsilon _ { 3 } > 0$ be a sufficiently small universal constant. Let $0 < \epsilon \le \epsilon _ { 3 }$ and $\tau > 0$ . Given an $\epsilon$ -corrupted set of $n$ samples drawn from $D$ , one can output a vector $ { \widehat { \mu } } \in  { \mathbb { R } } ^ { d }$ in time ${ \widetilde { \cal O } } ( n d \log ( 1 / \tau ) ) $ ) such that, with probability at least $1 - \tau - \exp ( - n \epsilon )$ , we have $\begin{array} { r } { \| \widehat { \mu } - \mu \| _ { 2 } = O \left( \sqrt { \epsilon } + \sqrt { \frac { d } { n \tau } } + \sqrt { \frac { d ( \log d + \log ( 1 / \tau ) ) } { n } } \right) \sigma . } \end{array}$ .

# 3 Outlier-Robust Phase Retrieval

In this section, we present key technical lemmas for the two stages of our algorithm: robust spectral initialization (Lemma 3.1) and robust gradient descent (Lemma 3.2). We then use these lemmas to prove our main result (Theorem 1.4).

Lemma 3.1 shows that we can compute an initial guess close to the ground truth.

Lemma 3.1 (Robust Spectral Initialization). Consider the setting of Problem 1.3, where $\boldsymbol { x } \in \mathbb { R } ^ { d }$ is the ground-truth. Let ϵ be a sufficiently small universal constant. Given an $\epsilon$ -corrupted set of $n = \widetilde { \Omega } ( d )$ samples, we can compute $z _ { 1 } \in \mathbb { R } ^ { d }$ in time $\widetilde O ( n d )$ such that $\operatorname* { m i n } \{ \left\| z _ { 1 } - x \right\| _ { 2 } , \left\| z _ { 1 } + x \right\| _ { 2 } \} \leq \frac { 1 } { 8 }$ with probability at least 0.97.

Lemma 3.2 shows that after the initialization, a robust gradient descent algorithm can recover the ground-truth vector to arbitrary precision.

Lemma 3.2 (Robust Gradient Descent). Consider the setting of Problem 1.3, where $\boldsymbol { x } \in \mathbb { R } ^ { d }$ is the ground-truth vector. Let $\Delta > 0$ be the desired precision. Let $\epsilon$ be a sufficiently small universal constant. Given an ϵ-corrupted set of $n = \widetilde \Omega ( d \mathrm { l o g } ^ { 2 } ( 1 / \Delta ) )$ samples and an initial guess $z _ { 1 }$ with $\begin{array} { r } { \| z _ { 1 } - x \| _ { 2 } \leq \frac { 1 } { 8 } } \end{array}$ , we can compute a vector $z \in \mathbb { R } ^ { d }$ in time ${ \widetilde { O } } ( n d )$ such that $\| z - x \| _ { 2 } \leq \Delta$ with probability at least 0.98.

Lemma 3.1 is proved in Section 4. Lemma 3.2 is proved in Section 5. We first use these lemmas to prove Theorem 1.4.

Theorem 1.4. Let $\epsilon _ { \mathrm { 0 } }$ be the minimum of the two universal constants of Lemma 3.1 and Lemma 3.2. $0 \le \epsilon \le \epsilon _ { 0 }$ . We assume that we have access to two separate $\epsilon$ -corrupted sets of samples, and use

20 one set for Lemma 3.1 and the other for Lemma 3.2. Formally, one could randomly partition the   
input samples into two sets and apply Chernoff bounds to show that both sets are $( 2 \epsilon )$ -corrupted with   
high probability.

By Lemma 3.1, we can compute a vector $z _ { 1 } \in \mathbb { R } ^ { d }$ such that $\operatorname* { m i n } \{ \left\| z _ { 1 } - x \right\| _ { 2 } , \left\| z _ { 1 } + x \right\| _ { 2 } \} \leq \frac { 1 } { 8 }$ for the ground-truth vector $x$ . Given $z _ { 1 }$ , by Lemma 3.2, we can output a vector $z ~ \in ~ \mathbb { R } ^ { d }$ such that $\operatorname* { m i n } \bar { \{ \| z - x \| _ { 2 } , \| z + x \| _ { 2 } \} } \leq \Delta$ for the desired precision parameter $\Delta > 0$ .

Let $n _ { 1 } = \widetilde { \Omega } ( d )$ and $n _ { 2 } = \widetilde \Omega ( d \log ^ { 2 } ( 1 / \Delta ) )$ denote the number of samples used in Lemma 3.1 and   
Lemma 3.2, respectively. The overall sample complexity is therefore $n = n _ { 1 } + n _ { 2 } = \widetilde \Omega ( d \log ^ { 2 } ( 1 / \Delta ) )$ .   
The overall runtime is $\widetilde { O } ( n _ { 1 } d ) + \widetilde { O } ( n _ { 2 } d ) = \widetilde { O } ( n d )$ . The overall success probability is at least 0.95   
229 by a union bound over Lemma 3.1 and Lemma 3.2. □

# 230 4 Robust Spectral Initialization

In this section, we prove Lemma 3.1: given an $\epsilon$ -corrupted set of samples $\{ ( a _ { i } , y _ { i } ) \} _ { i \in [ n ] }$ , we can compute an initial guess $z _ { 1 } \in \mathbb { R } ^ { d }$ that is close to the ground truth $x$ or $- x$ .

Consider the matrix $\begin{array} { r } { Y = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } y _ { i } a _ { i } a _ { i } ^ { \top } } \end{array}$ . When there is no corruption, where $a _ { i } \sim \mathcal { N } ( 0 , I )$ and $y _ { i } = \left. a _ { i } , x \right. ^ { 2 }$ , we have $\mathbb { E } [ Y ] = I + 2 x x ^ { \top }$ . However, the corrupted $( a _ { i } , y _ { i } )$ ’s can change $Y$ arbitrarily. To address this, we propose a nearly-linear time initialization step (Algorithm 1) that computes a nonnegative weight vector $w \in \mathbb { R } ^ { n }$ such that the weighted sum $\begin{array} { r } { \bar { Y _ { w } } = \mathbf { \bar { \sum } } _ { i = 1 } ^ { n } w _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } } \end{array}$ is close to $I + 2 x x ^ { \top }$ . Consequently, we can show that the largest eigenvector of $Y _ { w }$ is close to $\pm x$ .

38 Let $G \subset [ n ]$ be the set of indices of the remaining good samples. An ideal approach would be to 39 assign weight $\begin{array} { r } { \frac { 1 } { | G | } = \frac { 1 } { ( 1 - \epsilon ) n } } \end{array}$ to every sample in $G$ , and weight 0 to the corrupted samples. Formally, we consider weight vectors 40 $w$ in the set $\Delta _ { n , \epsilon } \ : = \ : \Big \{ w \in \mathbb { R } _ { \geq 0 } ^ { n } : \| w \| _ { 1 } = 1$ and $\begin{array} { r } { \| w \| _ { \infty } \leq \frac { 1 } { ( 1 - \epsilon ) n } \biggr \} } \end{array}$ . 41 Algorithm 1 computes a near-optimal solution $\widehat { w }$ to the following optimization problem:

$$
\operatorname* { m i n } _ { w \in \Delta _ { n , \epsilon } } \lambda _ { 1 } ( Y _ { w } ) + \lambda _ { 2 } ( Y _ { w } ) ,
$$

242 and then returns the largest eigenvector of $Y _ { \widehat { w } }$ .

# Algorithm 1: Robust Spectral Initialization

Input: $\epsilon$ -corrupted set of $n$ samples $\{ ( a _ { i } , y _ { i } ) \} _ { i \in [ n ] }$

Output: An initial guess $z _ { 1 } \in \mathbb { R } ^ { d }$ of the ground-truth $x$ s.t. $\operatorname* { m i n } \{ \left\| z _ { 1 } - x \right\| _ { 2 } , \left\| z _ { 1 } + x \right\| _ { 2 } \} \leq \frac { 1 } { 8 }$

1 $\widehat { w } \gets \mathtt { a }$ near-feasible, near-optimal solution to: $\begin{array} { r } { \operatorname* { m i n } _ { w \in \Delta _ { n , \epsilon } } [ \lambda _ { 1 } ( Y _ { w } ) + \lambda _ { 2 } ( Y _ { w } ) ] } \end{array}$ using Lemma A.2, bwhere $\begin{array} { r } { Y _ { w } = \sum _ { i = 1 } ^ { n } w _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } } \end{array}$ ;

$z _ { 1 } \gets$ an approximate top eigenvector of $Y _ { \widehat { w } }$ using the power method (Lemma 2.2);

return $z _ { 1 }$

To prove that the largest eigenvector of $Y _ { \widehat { w } }$ is close to $x$ , we will show that $x ^ { \top } Y _ { \widehat { w } } x$ is large, and there b244 is a gap between the first and second largest eigenvalues of $Y _ { \widehat { w } }$ .

Lemma 4.1. Consider the setting of Problem 1.3, where $\boldsymbol { x } \in \mathbb { R } ^ { d }$ is the ground-truth vector. Fix   
$\delta > 0$ . There exists constants $\epsilon ( \delta )$ and $c ( \delta )$ such that $i f$ we are given an $\epsilon ( \delta )$ -corrupted set of   
$n = \widetilde \Omega ( c ( \delta ) d )$ samples $( a _ { i } , y _ { i } ) _ { i \in [ n ] }$ , Algorithm $^ { l }$ outputs $\widehat { w } \in \mathbb R ^ { n }$ in time $\widetilde { \cal O } ( n d \mathrm { p o l y } ( 1 / \epsilon ( \delta ) ) )$ such   
bthat, with probability at least 0.98, the following conditions hold:

$$
x ^ { \top } Y _ { \widehat { w } } x \geq 3 - O ( \delta ) , \quad | \lambda _ { 1 } ( Y _ { \widehat { w } } ) - 3 | \leq O ( \delta ) , a n d \quad | \lambda _ { 2 } ( Y _ { \widehat { w } } ) - 1 | \leq O ( \delta ) .
$$

where 249 $\begin{array} { r } { Y _ { \widehat { w } } = \sum _ { i = 1 } ^ { n } \widehat { w } _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } } \end{array}$

We defer the proof of Lemma 4.1 to Appendix A.1 and first use it to prove the correctness and runtime   
of Algorithm 1 (Lemma 3.1).

.253 Proof of Lemma 3.1. Let 252 $\ldots \geq \lambda _ { d }$ . Let $\begin{array} { r } { x = \sum _ { i = 1 } ^ { d } \alpha _ { i } v _ { i } } \end{array}$ $\begin{array} { r } { Y _ { \widehat { w } } = \sum _ { i = 1 } ^ { d } \lambda _ { i } v _ { i } v _ { i } ^ { \top } } \end{array}$ i=1 . Note that $\begin{array} { r } { \sum _ { i = 1 } ^ { d } \alpha _ { i } ^ { 2 } = \left. \boldsymbol { x } \right. _ { 2 } ^ { 2 } = 1 } \end{array}$ be the eigendecomposition of and $| \alpha _ { i } | \le 1$ $Y _ { \widehat { w } }$ , where . If the event of $\lambda _ { 1 } \geq \lambda _ { 2 } \geq$ 254 Lemma 4.1 is true, with $\epsilon < \epsilon ( \delta )$ and $m = \widetilde \Omega ( c ( \delta ) d )$ , then

$$
\begin{array} { r l } & { 3 - O ( \delta ) \leq x ^ { \top } Y _ { \widehat { w } } x = \sum _ { i = 1 } ^ { d } \lambda _ { i } \alpha _ { i } ^ { 2 } \leq \lambda _ { 1 } \alpha _ { 1 } ^ { 2 } + \lambda _ { 2 } ( 1 - \alpha _ { 1 } ^ { 2 } ) } \\ & { \qquad \leq ( 3 + O ( \delta ) ) \alpha _ { 1 } ^ { 2 } + ( 1 + O ( \delta ) ) ( 1 - \alpha _ { 1 } ^ { 2 } ) \leq 1 + O ( \delta ) + 2 \alpha _ { 1 } ^ { 2 } . } \end{array}
$$

This implies 255 $| \alpha _ { 1 } | \geq \alpha _ { 1 } ^ { 2 } \geq 1 - O ( \delta )$ . Consequently,

$$
\begin{array} { r l } & { \operatorname* { m i n } \{ \| v _ { 1 } - x \| _ { 2 } ^ { 2 } , \| v _ { 1 } + x \| _ { 2 } ^ { 2 } \} = \operatorname* { m i n } \{ ( 1 - \alpha _ { 1 } ) ^ { 2 } , ( 1 + \alpha _ { 1 } ) ^ { 2 } \} + \sum _ { i = 2 } ^ { d } \alpha _ { i } ^ { 2 } } \\ & { \qquad = \operatorname* { m i n } \{ 2 - 2 \alpha _ { 1 } , 2 + 2 \alpha _ { 1 } \} = 2 - 2 | \alpha _ { 1 } | \le O ( \delta ) . } \end{array}
$$

We choose 256 $\delta$ as a sufficiently small constant so that $\begin{array} { r } { \operatorname* { m i n } \{ \| v _ { 1 } - x \| _ { 2 } , \| v _ { 1 } + x \| _ { 2 } \} \leq O ( \sqrt { \delta } ) \leq \frac { 1 } { 1 6 } } \end{array}$ 257 Note that since it is sufficient to choose as a small universal constant, $c ( \delta )$ and $\epsilon ( \delta )$ can also be 258 treated as universal constants.

We use the power method to approximate the largest eigenvector $z _ { 1 }$ of $Y _ { \widehat { w } }$ . By Lemma 2.2, $\| z _ { 1 } \| _ { 2 } = 1$   
and $z _ { 1 } ^ { \top } Y _ { \widehat { w } } z _ { 1 } \geq ( 1 - \epsilon _ { 2 } ) \lambda _ { 1 }$ . Choosing $\epsilon _ { 2 } = O ( \delta )$ b and using Lemma 4.1, we have $z _ { 1 } ^ { \top } Y _ { \widehat { w } } z _ { 1 } \geq 3 - \mathbf { \bar { O } } ( \delta )$ .   
bBy the same arguments, we can show that $\begin{array} { r } { \operatorname* { m i n } \lbrace \| v _ { 1 } - \check { z } _ { 1 } \| _ { 2 } , \| v _ { 1 } + z _ { 1 } \| _ { 2 } \rbrace \leq \frac { 1 } { 1 6 } } \end{array}$ b, and then by the   
262 triangle inequality, we conclude that $\operatorname* { m i n } \{ \left\| z _ { 1 } - x \right\| _ { 2 } , \left\| z _ { 1 } + x \right\| _ { 2 } \} \leq \frac { 1 } { 8 }$ .

63 The success probability of Algorithm 1 is at least 0.97, as $\widehat { w }$ satisfies Lemma 4.1 with probability at b64 least 0.98, and the power method succeeds with probability at least 0.99 by Lemma 2.2.

The required number of samples is $n = \widetilde { \Omega } ( d )$ by Lemma 4.1. Algorithm 1 runs in time $\widetilde O ( n d )$ : It   
takes time $\widetilde { \cal O } ( n d \mathrm { p o l y } ( 1 / \epsilon ( \delta ) ) ) = \widetilde { \cal O } ( n d )$ to compute $\widehat { w }$ by Lemma 4.1. The power method can   
approximate the largest eigenvector of $Y _ { \widehat { w } }$ in time $O ( n d \log ( d ) / \delta ) = \widetilde { O } ( n d )$ by Lemma 2.2, since   
the matrix-vector product $\begin{array} { r } { Y _ { \widehat { w } } v = \sum _ { i = 1 } ^ { n } \widehat { w } _ { i } y _ { i } \left. a _ { i } , v \right. a _ { i } } \end{array}$ can be computed in time $O ( n d )$ for any   
$\boldsymbol { v } \in \mathbb { R } ^ { d }$ . □

# 5 Robust Gradient Descent

After the robust initialization in Section 4, we have an initial guess $z _ { 1 } \in \mathbb { R } ^ { d }$ that is close to the ground truth $x$ or $- x$ . We can assume without loss of generality that $z _ { 1 }$ is closer to $x$ than to $- x$ .

In this section, we prove Lemma 3.2: Given an initial guess 273 $z _ { 1 }$ with $\begin{array} { r } { \| z _ { 1 } - x \| _ { 2 } \leq \frac { 1 } { 8 } } \end{array}$ , we can use a 274 robust gradient descent algorithm (Algorithm 2) to recover $x$ to any desired precision $\Delta > 0$ . We 275 will show that Algorithm 2 converges geometrically even when the input is $\epsilon$ -corrupted.

Consider the following nonconvex optimization problem:

$$
\operatorname* { m i n } _ { z \in \mathbb { R } ^ { d } } \sum _ { i = 1 } ^ { n } f _ { i } ( z ) \quad { \mathrm { w h e r e } } \quad f _ { i } ( z ) = \left( \langle a _ { i } , z \rangle ^ { 2 } - y _ { i } \right) ^ { 2 } ~ .
$$

Let $g _ { i }$ denote the gradient of $f _ { i }$ with respect to $z$ . Let $\mathcal { D } _ { z }$ denote the distribution of $g _ { i } ( z )$ when there   
is no corruption. Formally, $g ( z ) \sim \mathcal { D } _ { z }$ is distributed as

$$
g ( z ) = \frac { \partial } { \partial z } \left[ \left( \left. a , z \right. ^ { 2 } - \left. a , x \right. ^ { 2 } \right) ^ { 2 } \right] = 4 \left( \left. a , z \right. ^ { 2 } - \left. a , x \right. ^ { 2 } \right) \left. a , z \right. a \mathrm { ~ w h e r e ~ } a \sim \mathcal { N } ( 0 , I ) .
$$

To perform gradient descent, we want to approximate the expected true gradient

$$
\begin{array} { r } { \mu _ { z } = \underset { g ( z ) \sim \mathcal { D } _ { z } } { \mathbb { E } } \left[ g ( z ) \right] = \left( 1 2 \left. z \right. _ { 2 } ^ { 2 } - 4 \left. x \right. _ { 2 } ^ { 2 } \right) z - 8 \left. x , z \right. x . } \end{array}
$$

However, the input $\{ ( a _ { i } , y _ { i } ) \} _ { i \in [ n ] }$ is $\epsilon$ -corrupted, so the corresponding gradients $\{ g _ { i } ( z ) \} _ { i \in [ n ] }$ are an   
$\epsilon$ -corrupted set of vectors drawn from $\mathcal { D } _ { z }$ . We will run nearly-linear time robust mean estimation   
algorithms (e.g., [18]) on $\{ g _ { i } ( z ) \} _ { i \in [ n ] }$ to approximate the true gradient $\mu _ { z }$ .   
The accuracy of robust mean estimation algorithms depends on the covariance matrix $\Sigma _ { z }$ of the   
284 distribution $\mathcal { D } _ { z }$ . The following lemma upper bounds the spectral norm of $\Sigma _ { z }$ .

# Algorithm 2: Robust Gradient Descent

Input: An $\epsilon$ -corrupted set of $n$ samples $\{ ( a _ { i } , y _ { i } ) \} _ { i \in [ n ] }$ , an initial guess $z _ { 1 }$ with $\begin{array} { r } { \left\| z _ { 1 } - x \right\| \leq \frac { 1 } { 8 } } \end{array}$ , and desired precision $\Delta > 0$ .

Output: $z \in \mathbb { R } ^ { d }$ such that $\begin{array} { r } { \| z - x \| _ { 2 } \leq \Delta } \end{array}$ , where $x$ is the ground-truth vector.

1 $T \gets O ( \log ( 1 / \Delta ) )$ , $\begin{array} { r } { \eta  \frac { 1 } { 3 0 0 } } \end{array}$ ;   
$\{ N _ { 1 } , \dots , N _ { T } \}  \mathrm { a }$ random disjoint partition of $[ n ]$ such that $\begin{array} { r } { | N _ { t } | = \frac { n } { T } } \end{array}$ for all $t \in [ T ]$ ;

for $t = 1 , 2 , \dots , T$ do

end

return $z _ { T + 1 }$ ;

Lemma 5.1. Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be the ground-truth vector. Let $\mathcal { D } _ { z }$ be the distribution of gradients at $z$ as defined in Equation (2). For any 286 $z$ with $\left\| z - x \right\| _ { 2 } \leq 1$ , the covariance matrix $\Sigma _ { z }$ of $\mathcal { D } _ { z }$ satisfies

$$
\Sigma _ { z } \preceq O \left( \| z - x \| _ { 2 } ^ { 2 } \right) I \ .
$$

We defer the proof of Lemma 5.1 to Appendix B. For technical reasons, we randomly partition the   
input $\{ ( a _ { i } , y _ { i } ) \} _ { i \in [ n ] }$ into $T$ subsets and use one subset in each iteration. With high probability, each   
partition has at most (2ϵ)-fraction of corrupted samples. The next lemma shows that, given the   
covariance bound in Lemma 5.1, we can approximate the true gradient $\mu _ { z }$ from a $( 2 \epsilon )$ -corrupted set   
of gradients with a small error.

Lemma 5.2. Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be the ground-truth vector. Consider any $z \in \mathbb { R } ^ { d }$ with $\| z - x \| _ { 2 } \leq 1$ . Let $\mathcal { D } _ { z }$ be the distribution defined in Equation (2) and let $\mu _ { z }$ be the mean of $\mathcal { D } _ { z }$ . Let $c > 0$ , and let $\tau \in ( 0 , 1 / 4 )$ be a small constant. There exists a constant $\epsilon ( c )$ such that given a $2 \epsilon ( c )$ -corrupted set of $m = \Omega ( d \log ( d ) / ( c ^ { 2 } \tau ) )$ vectors drawn from $\mathcal { D } _ { z }$ , we can compute $ { \widehat { \mu } } _ { z } \in  { \mathbb { R } } ^ { d }$ in time ${ \widetilde { \cal O } } ( m d \log ( 1 / \tau ) )$ such that $\| \widehat { \mu } _ { z } - \mu _ { z } \| _ { 2 } \leq c \| z - x \| _ { 2 }$ with probability at least $1 - 2 \tau$ .

Proof of Lemma 5.2. We constraint $\epsilon ( c ) \leq \epsilon _ { 3 } / 2$ , where $\epsilon _ { 3 }$ is the universal constant in the robust   
mean estimation algorithm stated in Lemma 2.3. By Lemma 5.1, the covariance matrix $\Sigma _ { z }$ of $\mathcal { D } _ { z }$   
satisfies $\Sigma _ { z } \preceq O \left( \left| \left| z - x \right| \right| _ { 2 } ^ { 2 } \right) I .$ . Lemma 2.3 guarantees that robust mean estimation returns a vector   
$ { \widehat { \mu } } _ { z } \in  { \mathbb { R } } ^ { d }$ such that

$$
\begin{array} { r } { \| \widehat { \mu } _ { z } - \mu _ { z } \| _ { 2 } \leq O \bigg ( \sqrt { 2 \epsilon ( c ) } + \sqrt { \frac { d } { m \tau } } + \sqrt { \frac { d ( \log d + \log ( 1 / \tau ) ) } { m } } \bigg ) \sqrt { \| \Sigma _ { z } \| _ { 2 } } \leq c \| z - x \| _ { 2 } ~ . } \end{array}
$$

The last inequality follows by properly choosing the constant 301 $\epsilon ( c ) = O ( c ^ { 2 } )$ . By Lemma 2.3, the 302 runtime is ${ \widetilde { \cal O } } ( m d \log ( 1 / \tau ) )$ and the success probability is at least $\begin{array} { r } { 1 - \tau - \exp ( - \epsilon _ { 0 } m ) \geq 1 - 2 \tau } \end{array}$ .

The next lemma shows that the approximate gradient from Lemma 5.2 is sufficient for gradient   
descent to converge, reducing the distance to the ground truth $x$ by a constant factor in each iteration.   
We provide a proof sketch for Lemma 5.3 and defer the full proof to Appendix B.   
Lemma 5.3. Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be the ground-truth vector. Suppose at iteration t of Algorithm 2, the   
current solution $z _ { t }$ satisfies $\begin{array} { r } { \| \boldsymbol { z } _ { t } - \boldsymbol { \bar { x } } \| _ { 2 } \le \frac { 1 } { 8 } } \end{array}$ . Let $\mu _ { z _ { t } }$ denote the expected true gradient at $z _ { t }$ defined   
in Equation (3). Suppose the estimated gradient $\widehat { \mu } _ { z _ { t } } \in \mathbb { R } ^ { d }$ satisfies $\| \widehat { \mu } _ { z _ { t } } - \mu _ { z _ { t } } \| _ { 2 } \leq c \| z _ { t } - x \| _ { 2 } f o$ r   
$c = 4$ . Then, we have

$$
\left\| z _ { t + 1 } - x \right\| _ { 2 } ^ { 2 } \leq 0 . 9 9 \left\| z _ { t } - x \right\| _ { 2 } ^ { 2 } .
$$

Proof Sketch of Lemma 5.3. Even though the objective function is nonconvex, it is known that gradi  
ent descent is well-behaved when initialized close enough to a global optimum [37]. More specifically,

for any 312 $z$ with $\begin{array} { r } { \| z - x \| _ { 2 } \leq \frac { 1 } { 8 } } \end{array}$ , we can show that the expected true gradient at $z$ aligns with the direction 313 $( z - x )$ :

$$
\left. \mu _ { z } , z - x \right. \geq 7 . 5 \left. z - x \right. _ { 2 } ^ { 2 } \quad { \mathrm { ~ a n d ~ } } \quad \left. \mu _ { z } \right. _ { 2 } \leq 2 9 \left. z - x \right. _ { 2 } ,
$$

314 which is sufficient for proving geometric convergence.

Note that this analysis is robust to small error in 315 $\mu _ { z }$ . When $\left\| \widehat { \mu } _ { z } - \mu _ { z } \right\| _ { 2 } \leq c \left\| z - x \right\| _ { 2 }$ , we have

$$
\begin{array} { r } { \langle \widehat { \mu } _ { z } , z - x \rangle \geq \left( 7 . 5 - c \right) \left. z - x \right. _ { 2 } ^ { 2 } \quad \mathrm { ~ a n d ~ } \quad \left. \widehat { \mu } _ { z } \right. _ { 2 } \leq \left( 2 9 + c \right) \left. z - x \right. _ { 2 } . } \end{array}
$$

16 When $c < 7 . 5$ , we can choose an appropriate step size $\eta$ such that the distance between $z$ and $x$   
decreases by a constant factor in each iteration. □

We are now ready to prove Lemma 3.2, which states the correctness and runtime of Algorithm 2.

Lemma 3.2. First, we analyze the success probability of Algorithm 2. Algorithm 2 can fail in two   
ways: $( i )$ some $N _ { t }$ has more than $( 2 \epsilon )$ -fraction of corrupted samples, or $( i i )$ robust gradient estimation   
fails in some iteration $t$ . The probability of event $( i )$ is at most 0.01 for our choice of $n$ , which   
follows from a standard application of Hoeffding’s inequality and a union bound. For event $( i i )$ , we   
choose $\tau \leq 0 . 0 0 5 / T$ in Lemma 5.2, so each robust gradient estimation fails with probability at most   
$2 \tau = 0 . 0 1 / T$ . By a union bound over $T$ iterations, the probability of event $( i i )$ is at most 0.01. For   
325 the rest of the proof, we assume these bad events do not happen.

Next, we prove the correctness of Algorithm 2. Since $\begin{array} { r } { \| z _ { 1 } - x \| _ { 2 } \leq \frac { 1 } { 8 } } \end{array}$ , we can use Lemma 5.2 to obtain an approximation $\widehat { \mu } _ { z _ { 1 } }$ of the true gradient $\mu _ { z _ { 1 } }$ such that $\| \widehat { \mu } _ { z _ { 1 } } ^ { \cup } - \mu _ { z _ { 1 } } \| _ { 2 } \leq c \| z _ { 1 } - x \| _ { 2 }$ with $c = 4$ b. Then, by Lemma 5.3, we have $\left\| z _ { 2 } - x \right\| _ { 2 } \leq 0 . 9 9 \left\| z _ { 1 } - x \right\| _ { 2 }$ after one iteration of gradient descent. Applying these two lemmas repeatedly, after $T = O ( \log ( 1 / \Delta ) )$ iterations, we have $\bar { \| } z _ { T + 1 } - x \| _ { 2 } \leq \Delta$ .

Finally, we analyze the sample complexity and runtime of Algorithm 2. The algorithm requires in total $n = m \bar { T } = \Omega ( d \log \bar { d } \log ^ { 2 } ( \bar { 1 / \Delta } ) )$ samples. A random partition can be computed in $O ( n )$ time via random shuffling. In each iteration, the $m$ gradients in $N _ { t }$ can be computed in time $O ( m d )$ using Equation (2). By Lemma 5.2, the true gradient can be robustly estimated in time $\widetilde { \cal O } ( m d \log ( 1 / \tau ) ) = \widetilde { \cal O } ( m d \log T ) = \widetilde { \cal O } ( m d \log \log ( 1 / \Delta ) )$ , and $z _ { t }$ can be updated in time $O ( d )$ . The overall runtime of Algorithm 2 is $\widetilde { \cal O } ( n + T m d \log \log ( 1 / \Delta ) ) = \widetilde { \cal O } ( n d )$ . □

7 Remark. There are two technical details worth noting. First, robust mean estimation algorithms   
(Lemma 2.3) require a known upper bound $\sigma ^ { 2 }$ on the spectral norm of the covariance matrix of $\mathcal { D } _ { z }$ .   
By Lemma 5.1, a known upper bound on $\lVert z - x \rVert _ { 2 }$ suffices. We can indeed maintain such an upper   
bound, which starts at $\frac { 1 } { 8 }$ and decreases geometrically, as shown in Lemma 5.3. Second, at each   
iteration $t$ , to apply Lemma 5.2 to robustly estimate the gradient at $z _ { t }$ , we need a $( 2 \epsilon )$ -corrupted set of   
gradients drawn from $\mathcal { D } _ { z _ { t } }$ . This is why we use a set of fresh samples $N _ { t }$ . By the principle of deferred   
decisions, we can view $( a _ { i } , y _ { i } ) _ { i \in N _ { t } }$ as being generated and corrupted after $z _ { t }$ is chosen.

# 344 6 Conclusions and Future Directions

In this paper, we propose and study the problem of outlier-robust phase retrieval, where a small fraction of the input data is corrupted. Importantly, we allow adversarial corruption in both the sampling vectors $\boldsymbol { a } _ { i } \in \mathbb { R } ^ { d }$ and the intensity measurements $y _ { i } \in \mathbb { R }$ . We present a near-sampleoptimal and nearly-linear-time algorithm for this problem with provable guarantees. One conceptual contributions of our work is that phase retrieval can be solved using a robust first-order methods even when the input is slightly misspecified or corrupted. Our algorithmic framework provides a general approach for developing robust algorithms for a wide range of tractable nonconvex problems, by first robustly initializing into a region free of saddle points and then using robust gradient descent to converge to a global optimum.

An immediate technical question is whether our sample complexity can be tightened by removing the   
$\log ( 1 / \Delta )$ factors. One potential approach is to examine the stability conditions required by robust   
mean estimation algorithms and see if these conditions can be proved without using fresh samples in   
each iteration.

References [1] P. Abdalla and N. Zhivotovskiy. Covariance estimation: Optimal dimension-free guarantees for adversarial corruption and heavy tails. Journal of the European Mathematical Society, 2024. [2] S. Bhojanapalli, B. Neyshabur, and N. Srebro. Global optimality of local search for low rank matrix recovery. Advances in Neural Information Processing Systems, 29, 2016. [3] A. Buna and P. Rebeschini. Robust gradient descent for phase retrieval. In The 28th International Conference on Artificial Intelligence and Statistics. [4] E. J. Candès, X. Li, and M. Soltanolkotabi. Phase retrieval via wirtinger flow: Theory and algorithms. IEEE Trans. Inf. Theory, 61(4):1985–2007, 2015a. [5] E. J. Candès, X. Li, and M. Soltanolkotabi. Phase retrieval from coded diffraction patterns. Applied and Computational Harmonic Analysis, 39(2):277–299, Sept. 2015b. ISSN 1063-5203. [6] E. J. Candès, Y. C. Eldar, T. Strohmer, and V. Voroninski. Phase retrieval via matrix completion. SIAM Rev., 57(2):225–251, 2015c. [7] J. Chen, L. Wang, X. Zhang, and Q. Gu. Robust Wirtinger Flow for Phase Retrieval with Arbitrary Corruption, Jan. 2018. [8] Y. Chen and E. Candes. Solving Random Quadratic Systems of Equations Is Nearly as Easy as Solving Linear Systems. In Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. [9] Y. Cheng and R. Ge. Non-convex matrix completion against a semi-random adversary. In Proc. 31st Conference on Learning Theory (COLT), volume 75, pages 1362–1394, 2018. [10] Y. Cheng, I. Diakonikolas, and R. Ge. High-dimensional robust mean estimation in nearly-linear time. In Proc. 30th ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 2755–2771, 2019.   
[11] Y. Cheng, I. Diakonikolas, R. Ge, and D. P. Woodruff. Faster algorithms for high-dimensional robust covariance estimation. In A. Beygelzimer and D. Hsu, editors, Proceedings of the Thirty-Second Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 727–757. PMLR, 25–28 Jun 2019.   
[12] Y. Cherapanamjeri, S. Mohanty, and M. Yau. List decodable mean estimation in nearly linear time. In Proc. 61st IEEE Symposium on Foundations of Computer Science (FOCS), pages 141–148, 2020.   
[13] C. Dainty and J. R. Fienup. Phase retrieval and image reconstruction for astronomy. Image recovery: theory and application, 231:275, 1987. [14] J. Depersin and G. Lecué. Robust sub-Gaussian estimation of a mean vector in nearly linear time. The Annals of Statistics, 50(1):511–536, 2022.   
[15] I. Diakonikolas and D. M. Kane. Algorithmic High-Dimensional Robust Statistics. Cambridge University Press, 2023. [16] I. Diakonikolas, G. Kamath, D. M. Kane, J. Li, A. Moitra, and A. Stewart. Robust estimators in high dimensions without the computational intractability. In 57th Annual IEEE Symposium on Foundations of Computer Science—FOCS 2016, pages 655–664. IEEE Computer Soc., Los Alamitos, CA, 2016. [17] I. Diakonikolas, G. Kamath, D. Kane, J. Li, J. Steinhardt, and A. Stewart. Sever: A robust meta-algorithm for stochastic optimization. In K. Chaudhuri and R. Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 1596–1606. PMLR, 09–15 Jun 2019.   
[18] Y. Dong, S. B. Hopkins, and J. Li. Quantum entropy scoring for fast robust mean estimation and improved outlier detection. In Proc. 33rd Advances in Neural Information Processing Systems (NeurIPS), 2019.   
[19] . Duchi and F. Ruan. Solving (most) of a set of quadratic equalities: Composite optimization for robust phase retrieval. Information and Inference: A Journal of the IMA, 8(3):471–529, 2019.   
[20] J. R. Fienup. Reconstruction of an object from the modulus of its Fourier transform. Optics Letters, 3(1):27–29, July 1978. ISSN 1539-4794.   
[21] J. R. Fienup. Phase retrieval algorithms: A comparison. Applied Optics, 21(15):2758–2769, Aug. 1982. ISSN 2155-3165.   
[22] X. Gao and Y. Cheng. Robust matrix sensing in the semi-random model. Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS), 2023.   
[23] R. Ge, F. Huang, C. Jin, and Y. Yuan. Escaping from saddle points—online stochastic gradient for tensor decomposition. In Conference on Learning Theory, pages 797–842, 2015.   
[24] R. Ge, J. D. Lee, and T. Ma. Matrix completion has no spurious local minimum. In Advances in Neural Information Processing Systems, pages 2973–2981, 2016.   
[25] R. W. Gerchberg. A practical algorithm for the determination of phase from image and diffraction plane pictures. Optik, Jan. 1972.   
[26] F. R. Hampel, E. M. Ronchetti, P. J. Rousseeuw, and W. A. Stahel. Robust statistics. The approach based on influence functions. Wiley New York, 1986.   
[27] P. Hand and V. Voroninski. Corruption robust phase retrieval via linear programming. CoRR, abs/1612.03547, 2016.   
[28] P. J. Huber and E. M. Ronchetti. Robust statistics. Wiley New York, 2009.   
[29] K. Jaganathan, Y. C. Eldar, and B. Hassibi. Phase retrieval: An overview of recent developments. Optical Compressive Imaging, pages 279–312, 2016.   
[30] R. Kolte and A. Özgür. Phase Retrieval via Incremental Truncated Wirtinger Flow, June 2016.   
[31] K. A. Lai, A. B. Rao, and S. Vempala. Agnostic estimation of mean and covariance. In focs2016, pages 665–674, 2016.   
[32] G. Lecué, M. Lerasle, and T. Mathieu. Robust classification via MOM minimization. Mach. Learn., 109(8):1635–1665, 2020.   
[33] S. Li, Y. Cheng, I. Diakonikolas, J. Diakonikolas, R. Ge, and S. Wright. Robust second-order nonconvex optimization and its application to low rank matrix sensing. In Proc. 37th Advances in Neural Information Processing Systems (NeurIPS), 2023.   
[34] J.-W. Liu, Z.-J. Cao, J. Liu, X.-L. Luo, W.-M. Li, N. Ito, and L.-C. Guo. Phase Retrieval via Wirtinger Flow Algorithm and Its Variants. In 2019 International Conference on Machine Learning and Cybernetics (ICMLC), pages 1–9, July 2019.   
[35] J. Miao, T. Ishikawa, B. Johnson, E. H. Anderson, B. Lai, and K. O. Hodgson. High resolution 3D X-ray diffraction microscopy. Physical review letters, 89(8):088303, 2002.   
[36] R. P. Millane. Phase retrieval in crystallography and optics. JOSA A, 7(3):394–411, 1990.   
[37] P. Netrapalli, P. Jain, and S. Sanghavi. Phase retrieval using alternating minimization. In Proc. 27th Advances in Neural Information Processing Systems (NeurIPS), pages 2796–2804, 2013.   
[38] A. Prasad, A. S. Suggala, S. Balakrishnan, and P. Ravikumar. Robust estimation via robust gradient estimation. Journal of the Royal Statistical Society. Series B. Statistical Methodology, 82(3):601–627, 2020.   
[39] W. H. Robert. Phase problem in crystallography. JOSA a, 10(5):1046–1055, 1993.   
[40] Y. Shechtman, Y. C. Eldar, O. Cohen, H. N. Chapman, J. Miao, and M. Segev. Phase retrieval with application to optical imaging: a contemporary overview. IEEE signal processing magazine, 32(3):87–109, 2015.

[41] M. Soltanolkotabi. Structured signal recovery from quadratic measurements: Breaking sample   
complexity barriers via nonconvex optimization. IEEE Trans. Inf. Theory, 65(4):2374–2400,   
2019.   
[42] J. Sun, Q. Qu, and J. Wright. Complete dictionary recovery over the sphere i: Overview and the   
geometric picture. IEEE Trans. Inf. Theor., 63(2):853–884, 2 2017.   
[43] J. Sun, Q. Qu, and J. Wright. A geometric analysis of phase retrieval. Found. Comput. Math.,   
18(5):1131–1198, 2018.   
[44] L. Trevisan. Lecture notes on graph partitioning, expanders and spectral methods. University of   
California, Berkeley, https://people. eecs. berkeley. edu/luca/books/expanders-2016. pdf, 2017.   
[45] J. Tukey. Mathematics and picturing of data. In Proceedings of ICM, volume 6, pages 523–531,   
.   
[46] R. Vershynin. High-dimensional probability, volume 47 of Cambridge Series in Statistical and   
Probabilistic Mathematics. Cambridge University Press, Cambridge, 2018.   
[47] V. Voroninski. PhaseLift: A Novel Methodology for Phase Retrieval. PhD thesis, UC Berkeley,   
2013.   
[48] G. Wang, G. Giannakis, Y. Saad, and J. Chen. Solving most systems of random quadratic   
equations. Advances in Neural Information Processing Systems, 30, 2017.   
[49] G. Wang, G. B. Giannakis, Y. Saad, and J. Chen. Phase retrieval via reweighted amplitude flow.   
IEEE Transactions on Signal Processing, 66(11):2818–2833, 2018.   
[50] J. Wright and Y. Ma. High-dimensional data analysis with low-dimensional models: Principles,   
computation, and applications. Cambridge University Press, 2022.   
[51] H. Zhang, Y. Chi, and Y. Liang. Provable non-convex phase retrieval with outliers: Median   
truncated wirtinger flow. In International conference on machine learning, pages 1022–1031.   
473 PMLR, 2016.

# 474 A Omitted Proofs in Section 4

# A.1 Proof of Lemma 4.1

In this section, we prove Lemma 4.1: We can compute $\widehat { w } \in \mathbb R ^ { n }$ in nearly-linear time such that $x ^ { \top } Y _ { \widehat { w } } x$ is large and the two largest eigenvalues of $Y _ { \widehat { w } }$ bare approximately 3 and 1.

78 The following lemma shows that, for any $w \in \Delta _ { n , 2 \epsilon }$ , the contribution of the remaining good samples to 79 $Y _ { w }$ is close to $I + 2 x x ^ { \top }$ . This allows us to lower bound $\lambda _ { 1 } ( Y _ { \widehat { w } } )$ , $\lambda _ { 2 } ( Y _ { \widehat { w } } )$ , and $x ^ { \top } Y _ { \widehat { w } } x$ .

Lemma A.1. Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be the ground-truth vector. $F i x \delta > 0$ . There exists constants $\overline { { \epsilon } } ( \delta )$ and $c ( \delta )$ such that if we let $0 < \epsilon \le \overline { { \epsilon } } ( \delta )$ , and we are given an $\epsilon$ -corrupted set of $n = \widetilde \Omega ( c ( \delta ) d )$ samples $( a _ { i } , y _ { i } ) _ { i \in [ n ] }$ , with probability at least 0.99, for all $w \in \Delta _ { n , 2 \epsilon }$ ,

$$
\begin{array} { r } { \left. Y _ { G , w } - ( I + 2 x x ^ { \top } ) \right. _ { 2 } \leq \delta \ , } \end{array}
$$

where 83 $G$ is the set of indices of the remaining good samples and $\begin{array} { r } { Y _ { G , w } = \sum _ { i \in G } w _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } } \end{array}$

Intuitively, Lemma A.1 holds because the moments of Gaussian distributions are stable when a small   
fraction of the samples are removed. We defer the proof of Lemma A.1 to Appendix A.   
The next lemma shows that, assuming Lemma A.1 holds, we can compute a near-optimal $\widehat { w }$ that   
minimizes $\lambda _ { 1 } ( Y _ { \widehat { w } } ) + \lambda _ { 2 } ( Y _ { \widehat { w } } )$ in nearly-linear time.

Lemma A.2. Let $\delta \in ( 0 , 1 / 2 )$ . There exists a constant $\epsilon ( \delta )$ such that if we are given an $\epsilon ( \delta )$ - corrupted set of n samples $( a _ { i } , y _ { i } ) _ { i \in [ n ] }$ that satisfies Lemma A.1, we can compute $\widehat { w } \in \Delta _ { n , 2 \epsilon ( \delta ) } i n$ time $\widetilde { \cal O } ( n d \mathrm { p o l y } ( 1 / \epsilon ( \delta ) )$ such that with probability at least 0.99,

$$
\lambda _ { 1 } ( Y _ { \widehat w } ) + \lambda _ { 2 } ( Y _ { \widehat w } ) \leq 4 + O ( \delta ) ,
$$

where d is the dimension of 491 $a _ { i }$ and $\begin{array} { r } { Y _ { \widehat { w } } = \sum _ { i = 1 } ^ { n } \widehat { w } _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } } \end{array}$

Proof. We reduce the optimization problem $\mathrm { m i n } _ { w \in \Delta _ { n , \epsilon ( \delta ) } } \left[ \lambda _ { 1 } ( Y _ { \widehat { w } } ) + \lambda _ { 2 } ( Y _ { \widehat { w } } ) \right]$ to the following Ky   
Fan norm packing semidefinite program (SDP):

$$
\operatorname* { m a x } _ { w \in \mathbb { R } _ { \geq 0 } ^ { n } } \quad \| w \| _ { 1 } \quad \mathrm { s u b j e c t t o } \quad \ \sum _ { i = 1 } ^ { n } w _ { i } A _ { i } \preceq I , \quad \overline { { \lambda } } _ { 2 } \left( \sum _ { i = 1 } ^ { n } w _ { i } B _ { i } \right) \leq 4 + O ( \delta )
$$

in which 494 $A _ { i } = ( 1 - \epsilon ( \delta ) ) n e _ { i } e _ { i } ^ { \top }$ where $e _ { i } \in \mathbb { R } ^ { n }$ is the $i ^ { t h }$ basis vector, $B _ { i } = y _ { i } a _ { i } a _ { i } ^ { \top }$ , and $\overline { { \lambda } } _ { 2 }$ is the 495 sum of the two largest eigenvalues.

Let 496 $G$ be the set of indices of the remaining good samples. Consider the weight vector $w ^ { \star } \in \mathbb { R } ^ { n }$ that 497 is uniform on $G$ :

$$
w _ { i } ^ { \ast } = \left\{ \begin{array} { l l } { \frac { 1 } { | G | } = \frac { 1 } { ( 1 - \epsilon ) n } \quad } & { i \in G , } \\ { 0 \quad } & { i \notin G . } \end{array} \right.
$$

Let $\epsilon ( \delta ) = \operatorname* { m i n } \{ \delta , \overline { { \epsilon } } ( \delta ) \}$ , where $\overline { { \epsilon } } ( \delta )$ is the constant of Lemma A.1. By Lemma A.1, $Y _ { w ^ { \star } } \preceq$ 499 $( 1 + \delta ) I + 2 x x ^ { \top }$ , which implies $\overline { { { \lambda } } } _ { 2 } ( Y _ { w ^ { \star } } ) \leq 4 + O ( \delta )$ . Since $w ^ { \star }$ is feasible, the optimal value OPT of500 $( ^ { \ast } )$ must be at least $\| w ^ { \star } \| _ { 1 } = 1$ .

We invoke Lemma 2.1 to solve $( ^ { \ast } )$ with failure probability $\tau = 0 . 0 1$ and error tolerance parameter   
$\epsilon _ { 1 } = \epsilon ( \delta )$ . The resulting solution $w ^ { \prime } \in \mathbb { R } _ { \geq 0 } ^ { n }$ satisfies $\| w ^ { \prime } \| _ { 1 } \geq ( 1 - \epsilon _ { 1 } ) \mathrm { O P T } \geq 1 - \epsilon ( \delta )$ . Define   
$\begin{array} { r } { \widehat { w } = \frac { w ^ { \prime } } { \| w ^ { \prime } \| _ { 1 } } } \end{array}$ so that $\| \widehat { w } \| _ { 1 } = 1$ . The constraint with $A _ { i }$ guarantees that $\begin{array} { r } { \| w ^ { \prime } \| _ { \infty } \leq \frac { 1 } { ( 1 - \epsilon ( \delta ) ) n } } \end{array}$ , so   
$\begin{array} { r } { \| \widehat { w } \| _ { \infty } \leq \frac { \| w ^ { \prime } \| _ { \infty } } { 1 - \epsilon ( \delta ) } \leq \frac { 1 } { ( 1 - 2 \epsilon ( \delta ) ) n } } \end{array}$ and thus $w \in \Delta _ { n , 2 \epsilon ( \delta ) }$ . The constraint with $B _ { i }$ implies that $\overline { { \lambda } } _ { 2 } ( Y _ { w ^ { \prime } } ) \leq$   
$4 + O ( \delta )$ , and after scaling we have $\begin{array} { r } { \overline { { \lambda } } _ { 2 } ( Y _ { \widehat { w } } ) \leq \frac { 4 + O ( \delta ) } { 1 - \epsilon ( \delta ) } \leq 4 + O ( \delta ) } \end{array}$ since $\begin{array} { r } { \epsilon ( \delta ) \leq \delta < \frac { 1 } { 2 } } \end{array}$ . The success   
probability is at least $1 - \tau = 0 . 9 9$ .

We can write 07 $A _ { i } = C _ { i } C _ { i } ^ { \top }$ with $C _ { i } = \sqrt { ( 1 - \epsilon ) n } e _ { i }$ , and $B _ { i } = D _ { i } D _ { i } ^ { \top }$ with $D _ { i } = \sqrt { y _ { i } } a _ { i }$ . It takes 8 $O ( 1 )$ time to perform a matrix-vector product with $C _ { i }$ , and $O ( d )$ time with $D _ { i }$ . Therefore, $t _ { C } + t _ { D }$ 9 in Lemma 2.1 is $O ( n d )$ , and the runtime of Lemma 2.1 is ${ \widetilde O } ( n d \mathrm { p o l y } ( 1 / \epsilon ) )$ for $\tau = 0 . 0 1$ . □

We now proceed to prove Lemma 4.1.

Proof of Lemma 4.1. Without loss of generality, we can assume that $y _ { i } \geq 0$ for all $i \in [ n ]$ . Because $y _ { i }$   
should be $\left. a _ { i } , x \right. ^ { 2 }$ , any sample with $y _ { i } < 0$ must be corrupted and can be discarded. By Lemma A.2,   
we can compute $\widehat { w } \in \Delta _ { n , 2 \epsilon ( \delta ) }$ such that $\lambda _ { 1 } ( Y _ { \widehat { w } } ) + \lambda _ { 2 } ( Y _ { \widehat { w } } ) \leq 4 + O ( \delta )$ .

By Lemma A.1 and $y _ { i } \geq 0$ , we have

$$
Y _ { \widehat { w } } = \sum _ { i \in S } { \widehat w } _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } \succeq \sum _ { i \in G } { \widehat w } _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } = Y _ { G , { \widehat w } } \succeq ( 1 - \delta ) I + 2 x x ^ { \top } ,
$$

which gives a lower bound 515 $x ^ { \top } Y _ { \widehat { w } } x \geq 3 - O ( \delta )$ , as well as lower bounds on the eigenvalues of $Y _ { \widehat { w } }$

$$
\lambda _ { 1 } ( Y _ { \widehat { w } } ) \geq 3 - O ( \delta ) \quad \mathrm { a n d } \quad \lambda _ { 2 } ( Y _ { \widehat { w } } ) \geq 1 - O ( \delta ) .
$$

Putting the upper and lower bounds together, we obtain

$$
\begin{array} { r l } & { \lambda _ { 1 } ( Y _ { \widehat w } ) = \overline { { \lambda } } _ { 2 } ( Y _ { \widehat w } ) - \lambda _ { 2 } ( Y _ { \widehat w } ) \leq 4 + O ( \delta ) - ( 1 - O ( \delta ) ) \leq 3 + O ( \delta ) ~ , ~ \mathrm { a t } } \\ & { \lambda _ { 2 } ( Y _ { \widehat w } ) = \overline { { \lambda } } _ { 2 } ( Y _ { \widehat w } ) - \lambda _ { 1 } ( Y _ { \widehat w } ) \leq 4 + O ( \delta ) - ( 3 - O ( \delta ) ) \leq 1 + O ( \delta ) ~ . } \end{array}
$$

Lemma A.1 holds with probability at least 0.99. Assuming Lemma A.1 holds, Lemma 2.1 succeeds   
518 with probability at least 0.99, so the success probability of Lemma 4.1 is at least 0.98.   
19 For the initialization step, it suffices to use Lemma A.2 to find a $( 1 - \epsilon ( \delta ) )$ -optimal solution $\widehat { w } \in$   
$\Delta _ { n , 2 \epsilon ( \delta ) }$ to the SDP $( ^ { * } )$ , and the runtime to compute such $\widehat { w }$ is $\widetilde { \cal O } ( n d \mathrm { p o l y } ( 1 / \epsilon ( \delta ) ) )$ . □

# A.2 Proof of Lemma A.1

This section is devoted to the proof of Lemma A.1. We will use the following concentration results.

Lemma A.3 ([4], Section A.4.2). Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . For any $\delta > 0$ , there exists a constant $C ( \delta ) > 0$ such that when $n > C ( \delta ) \cdot d \log d$ and we are given a set of $n$ samples $\{ ( a _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ with $a _ { i } \sim \mathcal { N } ( 0 , I )$ independently and $y _ { i } = \left. a _ { i } , x \right. ^ { 2 }$ for all $i \in [ n ]$ , then with probability at least 0.99, it holds

$$
\left\| { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } y _ { i } a _ { i } a _ { i } ^ { \top } - ( I + 2 x x ^ { \top } ) \right\| _ { 2 } \leq \delta \ .
$$

Proposition A.4. Let $\alpha \in ( 0 , 2 / e )$ . Let $X _ { 1 } , \ldots , X _ { m }$ be $m$ random variables drawn i.i.d. from   
$\mathcal { N } ( 0 , 1 )$ . Define

$$
H = \left\{ i \in [ m ] : | X _ { i } | \geq 4 \ln ^ { 2 } ( 2 / \alpha ) \right\} .
$$

With probability at least 528 $1 - 1 0 ^ { - 3 }$ , the following are all true:

$$
\begin{array} { r l } { ( a ) } & { | H | \leq O ( m \alpha ) \ , } \\ { ( b ) } & { \displaystyle \sum _ { i \in H } X _ { i } ^ { 4 } = O \big ( m \alpha \log ^ { 8 } ( 2 / \alpha ) \big ) \ , } \\ { ( c ) } & { \displaystyle \underset { i \in H } { \operatorname* { m a x } } X _ { i } ^ { 2 } = O ( \log m ) \ . } \end{array}
$$

Proof. For 529 $X \ \sim \ { \mathcal { N } } ( 0 , 1 )$ and $t ~ > ~ 0$ , it holds that $\mathbf { P r } [ | X | \geq t ] \ \leq \ 2 \exp ( - t ^ { 2 } / 2 )$ . By setting 530 $t = 4 \ln ^ { 2 } ( 2 / \alpha )$ , we have $\mathbf { P r } \big [ X ^ { 4 } \geq t \big ] \ \leq \ \alpha$ . Let ${ Y _ { i } } \in \{ 0 , 1 \}$ be the indicator random variable for the event531 $^ { \ast } i \in H ^ { \ast }$ , so that $\textstyle | { \dot { \boldsymbol { H } } } | = \sum _ { i \in { \boldsymbol { H } } } Y _ { i }$ .

Because 532 $\mathbb { E } \big [ \sum _ { i \in H } Y _ { i } \big ] \le m \alpha$ , by Markov’s inequality, we have $| H | = O ( m \alpha )$ with probability at least 533 $1 - 1 0 ^ { - 3 } / 3$ . We assume this holds for the rest of the proof.

By the principle of deferred decisions, an equivalent way to draw $X _ { 1 } , \ldots , X _ { m }$ from $\mathcal { N } ( 0 , 1 )$ is to   
first draw $Y _ { i }$ , and then draw $X _ { i }$ conditioned on the value of $Y _ { i }$ . Note that $H$ is fixed after drawing   
$Y _ { 1 } , \dots , Y _ { m }$ .

$$
\mathbb { E } [ \sum _ { i \in { \cal H } } X _ { i } ^ { 4 } ] = \sum _ { i \in { \cal H } } \mathbb { E } [ X _ { i } ^ { 4 } \mid \mid X _ { i } ] \geq 4 \ln ^ { 2 } ( 2 / \alpha ) ] \leq O ( m \alpha ) \mathbb { E } [ X _ { i } ^ { 4 } \mid \mid X _ { i } ] \geq 4 \ln ^ { 2 } ( 2 / \alpha ) ] ~ .
$$

For any 537 $t \geq 4$ , by the definition of conditional expectation, and using the fact that $X _ { i }$ is normally 538 distributed:

$$
\begin{array} { r } { \mathbb { E } \big [ X _ { i } ^ { 4 } \bigm | | X _ { i } | \ge t \big ] = \frac { \frac { 2 } { \sqrt { 2 \pi } } \int _ { t } ^ { \infty } x ^ { 4 } e ^ { - x ^ { 2 } / 2 } d x } { \frac { 2 } { \sqrt { 2 \pi } } \int _ { t } ^ { \infty } e ^ { - x ^ { 2 } / 2 } d x } = \frac { \int _ { t } ^ { \infty } x ^ { 4 } e ^ { - x ^ { 2 } / 2 } d x } { \int _ { t } ^ { \infty } e ^ { - x ^ { 2 } / 2 } d x } \le 2 t ^ { 4 } . } \end{array}
$$

We use Inequality (5) to upper bound (4):

$$
\mathbb { E } \left[ \sum _ { i \in H } X _ { i } ^ { 4 } \right] \le { \cal O } ( m \alpha ) \left( 2 \cdot ( 4 \ln ^ { 2 } ( 2 / \alpha ) ) ^ { 4 } \right) = { \cal O } \left( m \alpha \log ^ { 8 } ( 2 / \alpha ) \right) \ .
$$

By Markov’s inequality, with probability at least 540 $1 \mathrm { ~ - ~ } 1 0 ^ { - 3 } / 3$ , we have $\begin{array} { r l } { \mathbb { E } \left[ \sum _ { i \in H } X _ { i } ^ { 4 } \right] } & { { } = } \end{array}$ 541 $O \big ( m \alpha \log ^ { 8 } ( 2 / \alpha ) \big )$ . Finally, since $X _ { 1 } , \ldots , X _ { m }$ are drawn i.i.d. from $\mathcal { N } ( 0 , 1 )$ , we have that 542 $\mathrm { m a x } _ { i \in [ m ] } | X _ { i } | = O ( \sqrt { \log m } )$ with probability at least $1 - 1 0 ^ { - 3 } / 3$ . □

Proposition A.5. Let $K _ { 1 } \geq 0$ and $K _ { 2 } \geq 0$ . Let $a _ { 1 } , \ldots , a _ { m } \geq 0$ such that $\mathrm { m a x } _ { i \in [ m ] } a _ { i } ^ { 2 } \leq K _ { 1 }$ and $\textstyle \sum _ { i \in [ m ] } a _ { i } ^ { 4 } \leq K _ { 2 }$ . Let $X _ { 1 } , \ldots , X _ { m }$ be $m$ random variables drawn i.i.d. from $\mathcal { N } ( 0 , 1 )$ . Then, with probability at least 1 − 10−312−d,

$$
\sum _ { i \in [ m ] } a _ { i } ^ { 2 } X _ { i } ^ { 2 } = O \Bigl ( \sqrt { d K _ { 2 } } + d K _ { 1 } \Bigr ) \ .
$$

Proof. Since 546 $X _ { i } \sim \mathcal { N } ( 0 , 1 )$ , the random variable $X _ { i } ^ { 2 }$ is sub-exponential. Applying Bernstein’s 547 inequality for sub-exponential random variable [46, Theorem 2.8.2], for every $t \geq 0$ ,

$$
\mathbf { P r } \left[ \sum _ { i \in [ m ] } a _ { i } X _ { i } ^ { 2 } \geq t \right] \leq \exp \left( - c \operatorname* { m i n } \left( { \frac { t ^ { 2 } } { \sum _ { i \in [ m ] } a _ { i } ^ { 4 } } } , { \frac { t } { \operatorname* { m a x } _ { i } a _ { i } ^ { 2 } } } \right) \right) ~ ,
$$

where $c > 0$ is a universal constant.

We need to choose a value of $t$ such that the right-hand side of (6) is upper bounded by $1 0 ^ { - 3 } 1 2 ^ { - d }$ .   
Given our assumptions on $a _ { 1 } , \ldots , a _ { m }$ , it is sufficient to choose $t$ such that $t = \Omega \left( \sqrt { d K _ { 2 } } \right)$ and   
$t = \Omega \left( d K _ { 1 } \right)$ . □

Proposition A.6. Let $\alpha \in ( 0 , 1 )$ . Let $X _ { 1 } , \ldots , X _ { m }$ be $m$ random variables drawn i.i.d. from $\mathcal { N } ( 0 , 1 )$ . With probability at least 553 $1 - 1 0 ^ { - 3 } 1 2 ^ { - d }$ , it holds that:

$$
\operatorname* { m a x } _ { L \subseteq [ m ] : | L | = \alpha m } \sum _ { i \in L } X _ { i } ^ { 2 } = O ( m \alpha \log ( 1 / \alpha ) + d ) .
$$

Proof. We define the threshold function

$$
h _ { r } ( z ) = \left\{ { 0 , \atop z , } \right. \ z \leq r
$$

with $r = 8 \ln ( 1 / \alpha )$ . Since $z \leq r + h _ { r } ( z )$ for all $z > 0$ ,

$$
\operatorname* { m a x } _ { L \subseteq [ m ] , | L | = \alpha m } \sum _ { i \in L } X _ { i } ^ { 2 } \leq \operatorname* { m a x } _ { L } \sum _ { i \in L } r + \operatorname* { m a x } _ { L } \sum _ { i \in L } h _ { r } ( X _ { i } ^ { 2 } ) \leq m \alpha \cdot r + \sum _ { i \in [ m ] } h _ { r } ( X _ { i } ^ { 2 } ) \ .
$$

We will use Chernoff-bound like arguments to obtain a high-probability upper bound on   
$\textstyle \sum _ { i \in [ m ] } h _ { r } ( X _ { i } ^ { 2 } )$ . For any $c > 0$ and $t > 0$ , we have:

$$
\begin{array} { r l } {  { \mathbf { P r } [ \sum _ { i \in [ m ] } h _ { r } ( X _ { i } ^ { 2 } ) \geq t ] = \mathbf { P r } [ \exp ( c \sum _ { i \in [ m ] } h _ { r } ( X _ { i } ^ { 2 } ) ) \geq \exp ( c \cdot t ) ] } } \\ & { \leq e ^ { - c t }  { \mathbb { E } } [ \exp ( c \sum _ { i \in [ m ] } h _ { r } ( X _ { i } ^ { 2 } ) ) ] } \\ & { = e ^ { - c t } \displaystyle \prod _ { i \in [ m ] }  { \mathbb { E } } [ \exp ( c \cdot h _ { r } ( X _ { i } ^ { 2 } ) ) ] \ , } \end{array}
$$

where the inequality follows from Markov’s inequality.

Thus, it is sufficient to upper bound 559 $\mathbb { E } \big [ \exp ( c \cdot h _ { r } ( X _ { i } ^ { 2 } ) ) \big ]$ . For any $c < 1 / 2$ , we have

$$
\begin{array} { l } { \displaystyle \mathbb { E } \big [ \exp \big ( c \cdot h _ { r } ( X _ { i } ^ { 2 } ) \big ) \big ] = 1 \cdot \mathbf { P r } \big [ h _ { r } ( X _ { i } ^ { 2 } ) = 0 \big ] + \frac { 1 } { \sqrt { 2 \pi } } \int _ { \sqrt { r } } ^ { \infty } e ^ { c x ^ { 2 } } e ^ { - x ^ { 2 } / 2 } d x } \\ { \displaystyle \qquad \leq 1 + \frac { 1 } { \sqrt { 2 \pi } \sqrt { 1 - 2 c } } \int _ { \sqrt { r ( 1 - 2 c ) } } ^ { \infty } e ^ { - y ^ { 2 } / 2 } d y } \\ { \displaystyle \qquad = 1 + \frac { 1 } { \sqrt { 1 - 2 c } } \mathbf { P r } \Big [ X _ { i } \geq \sqrt { r ( 1 - 2 c ) } \Big ] } \\ { \displaystyle \qquad \leq 1 + \frac { 1 } { \sqrt { 1 - 2 c } } \exp \big ( - r ( \frac { 1 } { 2 } - c ) \big ) ~ , } \end{array}
$$

where the second inequality is obtained by substituting 560 $x { \sqrt { 1 - 2 c } } = y$ in the integral, and the last inequality uses the Gaussian tail bound that 561 $\mathbf { P r } [ X _ { i } \geq z ] \le e ^ { - z ^ { 2 } / 2 }$ for all $z > 0$ . Recall that 562 $r = 8 \bar { \mathrm { l n ( 1 / } \alpha ) }$ . We set $c = 1 / 4$ , so that $\exp ( - r / 4 ) \stackrel { \cdot \cdot } { = } \alpha ^ { 2 }$ . Thus, we have that:

$$
\mathbb { E } \bigg [ \exp \bigg ( \frac { 1 } { 4 } \cdot h _ { r } ( X _ { i } ^ { 2 } ) \bigg ) \bigg ] \leq 1 + \sqrt { 2 } \alpha ^ { 2 } \leq e ^ { \sqrt { 2 } \alpha ^ { 2 } }
$$

We substitute the upper bound (8) into (7) and obtain that

$$
\mathbf { P r } \left[ \sum _ { i \in [ m ] } h _ { r } ( X _ { i } ^ { 2 } ) \geq t \right] \leq \exp \left( \frac { - t } { 4 } + \sqrt { 2 } m \alpha ^ { 2 } \right) \ .
$$

We can choose 564 $t = 4 \sqrt { 2 } m \alpha ^ { 2 } + \Omega ( d )$ , and then conclude that with probability at least $1 - 1 0 ^ { - 3 } 1 2 ^ { - d }$ ,

$$
\operatorname* { m a x } _ { L \subseteq [ m ] : | L | = \alpha m } \leq m \alpha r + \sum _ { i \in [ m ] } h _ { r } ( X _ { i } ^ { 2 } ) = O ( m \alpha r + m \alpha ^ { 2 } + d ) = O ( m \alpha \log ( 1 / \alpha ) + d ) \ .
$$



Lemma A.1. Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be the ground-truth vector. $F i x \delta > 0$ . There exists constants $\overline { { \epsilon } } ( \delta )$ and   
$c ( \delta )$ such that if we let $0 < \epsilon \le \overline { { \epsilon } } ( \delta )$ , and we are given an $\epsilon$ -corrupted set of $n = \widetilde \Omega ( c ( \delta ) d )$ samples   
$( a _ { i } , y _ { i } ) _ { i \in [ n ] }$ , with probability at least 0.99, for all $w \in \Delta _ { n , 2 \epsilon }$ ,

$$
\begin{array} { r } { \left. Y _ { G , w } - ( I + 2 x x ^ { \top } ) \right. _ { 2 } \leq \delta \ , } \end{array}
$$

where 569 $G$ is the set of indices of the remaining good samples and $\begin{array} { r } { Y _ { G , w } = \sum _ { i \in G } w _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } } \end{array}$

Proof of Lemma A.1. We recall the definition of $\begin{array} { r } { Y _ { G , w } = \sum _ { i \in G } w _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } } \end{array}$ . Let $\ell \leq \epsilon \cdot n$ and let   
$\{ ( a _ { n + i } , y _ { n + i } ) \} _ { i = 1 } ^ { \ell }$ be the set of samples that were removed by the $\epsilon$ -corruption adversary. Let   
$G ^ { \prime } = G \cup \{ n + 1 , \ldots , n + \ell \}$ , $n ^ { \prime } = n + \ell .$ and $\epsilon ^ { \prime } = \epsilon / ( 1 + \epsilon )$ . Note that without loss of generality,   
we can assume that $| G | = 1 - \epsilon ) n$ and $| G ^ { \prime } | = ( 1 - \epsilon ^ { \prime } ) \dot { n } ^ { \prime } = \dot { n }$ .

We define a mapping 574 $\sigma : \Delta _ { n , 2 \epsilon }  \Delta _ { n ^ { \prime } , 3 \epsilon ^ { \prime } }$ such that

$$
\sigma ( w ) _ { i } = \left\{ { w _ { i } \qquad i \in [ n ] } \atop 0 \right. \qquad \mathrm { o t h e r w i s e } \  .
$$

In other words, all the weights are the same for the samples with index in the set $[ n ]$ , and are equal to   
0 for the samples removed by the adversary. We can verify that $\sigma ( w ) \in \Delta _ { n ^ { \prime } , 3 \epsilon ^ { \prime } }$ for all $w \in \Delta _ { n , 2 \epsilon }$   
since $\sigma ( w ) _ { i } \ : \overset { \cdot } { \leq } \ : w _ { i } \leq 1 / ( 1 - 2 \epsilon ) n = 1 / ( \mathrm { 1 } - 3 \epsilon ^ { \prime } ) n ^ { \prime }$ for all $i \in [ n ^ { \prime } ]$ , and $\left\| \sigma ( w ) \right\| _ { 1 } = \left\| w \right\| _ { 1 } = 1$ .   
Furthermore, we have $Y _ { G , w } = Y _ { G ^ { \prime } , \sigma ( w ) }$ for all $w \in \Delta _ { n , 2 \epsilon }$ . We denote with $w ^ { * } \in \Delta _ { n ^ { \prime } , 3 \epsilon ^ { \prime } }$ the desired   
uniform weighting of the samples with index in $G ^ { \prime }$ , i.e., $\begin{array} { r } { w _ { i } ^ { * } = \frac { 1 } { ( 1 - \epsilon ^ { \prime } ) n ^ { \prime } } \mathbb { 1 } _ { i \in G ^ { \prime } } } \end{array}$ .

By triangle inequality, for any $w \in \Delta _ { n , 2 \epsilon }$ , it holds

$$
\begin{array} { r } { \left\| Y _ { G ^ { \prime } , \sigma ( w ) } - ( I + 2 x x ^ { \top } ) \right\| _ { 2 } \leq \left\| Y _ { G ^ { \prime } , w ^ { * } } - ( I + 2 x x ^ { \top } ) \right\| _ { 2 } + \left\| Y _ { G ^ { \prime } , \sigma ( w ) - w ^ { * } } \right\| _ { 2 } , } \end{array}
$$

Thus, it suffices to show both581 $\left\| Y _ { G ^ { \prime } , w ^ { * } } - ( I + 2 x x ^ { \top } ) \right\| _ { 2 } \leq \delta / 2$ and $\left\| Y _ { G ^ { \prime } , \sigma ( w ) - w ^ { * } } \right\| _ { 2 } \leq \delta / 2$ . We upper bound the first term. By using the definition of 582 $w ^ { \ast }$ , note that

$$
\begin{array} { r } { \big \| Y _ { G ^ { \prime } , \sigma ( w ) } - ( I + 2 x x ^ { \top } ) \big \| _ { 2 } = \bigg \| \displaystyle \sum _ { i \in G ^ { \prime } } w _ { i } ^ { * } y _ { i } a _ { i } a _ { i } ^ { \top } - ( I + 2 x x ^ { \top } ) \bigg \| _ { 2 } } \\ { = \bigg \| \displaystyle \sum _ { i \in G ^ { \prime } } \frac { 1 } { | G ^ { \prime } | } y _ { i } a _ { i } a _ { i } ^ { \top } - ( I + 2 x x ^ { \top } ) \bigg \| _ { 2 } . } \end{array}
$$

Since 583 $\mathbb { E } \left[ y _ { i } a _ { i } a _ { i } ^ { \top } \right] = I + 2 x x ^ { \top }$ for any $i \in G ^ { \prime }$ , we can use a concentration inequality to upper bound 584 this term. By Lemma A.3, as long as $n \geq C ( \delta / 2 ) \cdot d \log d .$ , with probability at least 0.995, we have

$$
\left\| Y _ { G ^ { \prime } , w ^ { * } } - ( I + 2 x x ^ { \top } ) \right\| _ { 2 } \leq \delta / 2 .
$$

It remains to show a high-probability upper bound to the second term $\left. Y _ { G ^ { \prime } , w ^ { * } - \sigma ( w ) } \right. _ { 2 } \leq \delta / 2$ that   
holds for any $w \in \Delta _ { w , 2 \epsilon }$ . To achieve this goal, we will provide a high-probability upper bound to the   
following quantity:

$$
J = \operatorname* { s u p } _ { w \in \Delta _ { n , 2 \epsilon } } \left\| Y _ { G ^ { \prime } , w ^ { * } - \sigma ( w ) } \right\| _ { 2 } = \operatorname* { s u p } _ { w \in \Delta _ { n , 2 \epsilon } } \left\| \sum _ { i \in G ^ { \prime } } ( w _ { i } ^ { * } - \sigma ( w ) _ { i } ) y _ { i } a _ { i } a _ { i } ^ { \top } \right\| _ { 2 } .
$$

Note that for every 588 $i$ , the matrix $y _ { i } a _ { i } a _ { i } ^ { \top }$ is positive semidefinite since $y _ { i } \geq 0$ . Thus, it holds that:

$$
J \leq \operatorname* { s u p } _ { w \in \Delta _ { n , 2 \epsilon } } \left\| \sum _ { i \in G ^ { \prime } } | w _ { i } ^ { * } - \sigma ( w ) _ { i } | y _ { i } a _ { i } a _ { i } ^ { \top } \right\| _ { 2 } .
$$

For any $w \in \Delta _ { n , 2 \epsilon }$ and any $i \in [ n ^ { \prime } ]$ , it is easy to see that $\begin{array} { r } { 0 \leq | w _ { i } ^ { * } - \sigma ( w ) _ { i } | \leq \frac { 1 } { ( 1 - 2 \epsilon ) n } } \end{array}$ . Additionally   
the weighting $w ^ { * }$ and $\sigma ( w )$ cannot be too different. In particular, we can show the following upper   
bound:

$$
\sum _ { i \in G ^ { \prime } } \left| w _ { i } ^ { * } - \sigma ( w ) _ { i } \right| \leq \sum _ { i = 1 } ^ { n ^ { \prime } } \left| w _ { i } ^ { * } - \sigma ( w ) _ { i } \right| \leq \operatorname* { s u p } _ { w , w ^ { \prime } \in \Delta _ { n ^ { \prime } , 3 \epsilon ^ { \prime } } } \sum _ { i = 1 } ^ { n ^ { \prime } } \left| w _ { i } - w _ { i } ^ { \prime } \right| .
$$

We observe that $\Delta _ { n ^ { \prime } , 3 \epsilon ^ { \prime } }$ can be seen as the convex combination of all possible uniform weighting   
over subsets of $n ^ { \prime } ( 1 - 3 \epsilon ^ { \prime } )$ samples. Thus, the maximum distance will be between two points of the   
convex hull, and we can upper bound (12) as:

$$
\sum _ { i \in G ^ { \prime } } | w _ { i } ^ { * } - \sigma ( w ) _ { i } | \leq \operatorname* { s u p } _ { w , w ^ { \prime } \in \Delta _ { n ^ { \prime } , 3 \epsilon ^ { \prime } } } \sum _ { i = 1 } ^ { n ^ { \prime } } | w _ { i } - w _ { i } ^ { \prime } | \leq \frac { 6 \epsilon ^ { \prime } n } { n ^ { \prime } ( 1 - 3 \epsilon ^ { \prime } ) } \leq 6 \epsilon .
$$

Consider the family of weights defined as 595 $\Gamma = \Big \{ \beta \in \mathbb { R } ^ { n ^ { \prime } } : \textstyle \sum _ { i } \beta _ { i } \le 6 \epsilon$ and $\begin{array} { r } { 0 \leq \beta _ { i } \leq \frac { 1 } { ( 1 - 2 \epsilon ) n } \biggr \} } \end{array}$ . B y 596 the discussion above, we have that

$$
J \leq \operatorname* { s u p } _ { w \in \Delta _ { n , 2 \epsilon } } \left\| \sum _ { i \in G ^ { \prime } } | w _ { i } ^ { * } - \sigma ( w ) _ { i } | y _ { i } a _ { i } a _ { i } ^ { \top } \right\| _ { 2 } \leq \operatorname* { s u p } _ { \beta \in \Gamma } \left\| \sum _ { i \in G ^ { \prime } } \beta _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } \right\| _ { 2 } .
$$

Since the map $\begin{array} { r } { \beta \mapsto \left. \sum _ { i \in G ^ { \prime } } \beta _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } \right. _ { 2 } } \end{array}$ is convex with respect to $\beta$ , and $\Gamma$ is a convex set, the   
supremum over $\beta \in \Gamma$ of (14) is achieved at one of the extreme points of $\Gamma$ . Thus, it holds:

$$
J \leq \operatorname* { s u p } _ { \beta \in \Gamma } \left\| \sum _ { i \in G ^ { \prime } } \beta _ { i } y _ { i } a _ { i } a _ { i } ^ { \top } \right\| _ { 2 } \leq \frac { 1 } { ( 1 - 2 \epsilon ) n } \operatorname* { m a x } _ { \substack { L \subseteq G ^ { \prime } , | L | = 6 \epsilon n } } \left\| \sum _ { i \in L } y _ { i } a _ { i } a _ { i } ^ { \top } \right\| _ { 2 } .
$$

For any vector 599 $v \in \mathbb { S } ^ { d - 1 }$ in the unit sphere, let

$$
J ( v ) = \operatorname* { m a x } _ { L \subseteq G ^ { \prime } , | L | = 6 \epsilon n } \sum _ { i \in L } y _ { i } ( v _ { i } ^ { \top } a _ { i } ) ^ { 2 } ,
$$

and note that 600 $\begin{array} { r } { J \leq \frac { 1 } { ( 1 - 2 \epsilon ) n } \operatorname* { s u p } _ { v \in { \mathbb S } ^ { d - 1 } } J ( v ) } \end{array}$

Without loss of generality, assume that $x = e _ { 1 }$ , where $e _ { 1 } \in \mathbb { R } ^ { d }$ is the first canonical vector. Given   
any vector $u \in \bar { \mathbb { R } } ^ { d }$ , we will denote with $u _ { 1 }$ the first coordinate, and with $\widetilde { u } \in \mathbb { R } ^ { d - 1 }$ the remaining   
$d - 1$ coordinates, i.e., $\boldsymbol { u } = ( u _ { 1 } , \widetilde { u } )$ . Assuming $x = e _ { 1 }$ , we have that $y _ { i } = ( x ^ { \top } a _ { i } ) ^ { 2 } = a _ { i , 1 } ^ { 2 }$ for any   
$i \in G ^ { \prime }$ . Let $H = \{ i \in G ^ { \prime } : | a _ { i , 1 } | \geq 4 \ln ^ { 2 } ( 2 / \epsilon ) \}$ . We consider a set $L \subseteq G ^ { \prime }$ that always contains $H$   
and then picks 6ϵn additional elements. In particular, it holds:

$$
J ( v ) = \operatorname* { m a x } _ { L \subseteq G ^ { \prime } , | L | = 6 \epsilon n } \sum _ { i \in L } y _ { i } ( a _ { i } ^ { \top } v _ { i } ) ^ { 2 } \leq \left[ \sum _ { i \in H } y _ { i } ( a _ { i } ^ { \top } v _ { i } ) ^ { 2 } + \operatorname* { m a x } _ { L \subseteq G ^ { \prime } \backslash H , | L | = 6 \epsilon n } \sum _ { i \in L } y _ { i } ( a _ { i } ^ { \top } v _ { i } ) ^ { 2 } \right] .
$$

The first term of the right-hand side of (15) can be rewritten as follows:

$$
\sum _ { i \in { \cal H } } y _ { i } ( a _ { i } ^ { \top } v ) ^ { 2 } = \sum _ { i \in { \cal H } } a _ { i , 1 } ^ { 2 } \left( \sum _ { j = 1 } ^ { d } a _ { i , j } v _ { j } \right) ^ { 2 } \leq 2 \sum _ { i \in { \cal H } } \left[ a _ { i , 1 } ^ { 4 } v _ { i , 1 } ^ { 2 } + a _ { i , 1 } ^ { 2 } ( \widetilde { a } _ { i } ^ { \top } \widetilde { v } ) ^ { 2 } \right] .
$$

For the second term of the right-hand side of (15), note that for any 607 $i \in G ^ { \prime } \backslash H$ , it holds that 608 $y _ { i } = a _ { i , 1 } ^ { 2 } < 1 6 \ln ^ { 4 } ( 2 / \epsilon )$ due to the definition of $H$ . Thus, we have that:

$$
\begin{array} { r l r } {  { \operatorname* { m a x } _ { L \subseteq G ^ { \prime } \setminus H , | L | = 6 \epsilon n } \sum _ { i \in L } y _ { i } ( a _ { i } ^ { \top } v _ { i } ) ^ { 2 } \le 1 6 \ln ^ { 4 } ( 2 / \epsilon ) \operatorname* { m a x } _ { L \subseteq G ^ { \prime } \setminus H , | L | = 6 \epsilon n } \sum _ { i \in L } ( a _ { i } ^ { \top } v _ { i } ) ^ { 2 } } } \\ & { } & { \le 3 2 \ln ^ { 4 } ( 2 / \epsilon ) \operatorname* { m a x } _ { L \subseteq G ^ { \prime } \setminus H , | L | = 6 \epsilon n } \sum _ { i \in L } [ a _ { i } ^ { 2 } v _ { 1 } ^ { 2 } + ( \widetilde { v } ^ { \top } \widetilde { a } _ { i } ) ^ { 2 } ] \ . } \end{array}
$$

Also, using the definition of 609 $H$ , note that for any $L \subseteq G ^ { \prime } \setminus H$ with $| L | = 6 \epsilon n$ , we have that :

$$
\sum _ { i \in L } a _ { i , 1 } ^ { 2 } v _ { 1 } ^ { 2 } \leq \sum _ { i \in L } a _ { i , 1 } ^ { 2 } = O \big ( n \epsilon \ln ^ { 4 } ( 2 / \epsilon ) \big ) .
$$

By combining (16), (17), and (19) with (15), we obtain that:

$$
J ( v ) = O \left( n \epsilon \log ^ { 8 } ( 1 / \epsilon ) + \sum _ { i \in H } a _ { i , 1 } ^ { 4 } + \sum _ { i \in H } a _ { i , 1 } ^ { 2 } ( \widetilde { a } _ { i } ^ { \top } \widetilde { v } ) ^ { 2 } + \operatorname* { m a x } _ { L \subseteq G ^ { \prime } \backslash H , \mid L \mid = 6 \epsilon n } \sum _ { i \in L } ( \widetilde { v } ^ { \top } \widetilde { a } _ { i } ) ^ { 2 } \right) \ .
$$

Let $E _ { 1 }$ be the event of Proposition A.4 with $\alpha = \epsilon$ and $n = m$ . That is, with probability at least   
$1 - 1 0 ^ { - 3 }$ , we have that $| H | = O ( \epsilon n )$ , $\textstyle \sum _ { i \in \underline { { H } } } a _ { i , 1 } ^ { 4 } = O \big ( n \epsilon \log ^ { 8 } ( 1 / \epsilon ) \big )$ and $\operatorname* { m a x } _ { i } a _ { i , 1 } ^ { 2 } = O ( \log n )$   
For the remaining of this proof, assume that $\mathrm { E } _ { 1 }$ is true, and thus:

$$
J ( v ) = O \left( n \epsilon \log ^ { 8 } ( 1 / \epsilon ) + \sum _ { i \in H } a _ { i , 1 } ^ { 2 } ( \widetilde { a } _ { i } ^ { \top } \widetilde { v } ) ^ { 2 } + \operatorname* { m a x } _ { L \subseteq G ^ { \prime } \backslash H , | L | = 6 \epsilon n } \sum _ { i \in L } ( \widetilde { v } ^ { \top } \widetilde { a } _ { i } ) ^ { 2 } \right) \ .
$$

Denote with $\begin{array} { r } { \overline { { J } } ( v ) = \sum _ { i \in H } ( \widetilde { a } _ { i } ^ { \top } \widetilde { v } ) ^ { 2 } + \operatorname* { m a x } _ { L \subseteq G ^ { \prime } \backslash H , | L | = 6 \epsilon n } \sum _ { i \in L } ( \widetilde { v } ^ { \top } \widetilde { a } _ { i } ) ^ { 2 } } \end{array}$ the last two terms of the   
right-hand side of (20). We will upper bound each term of $\overline { J }$ individually. First, observe that the   
random variables $\bar { Z } _ { i } = \widetilde { a } _ { i } ^ { \top } \widetilde { v } / \| \widetilde { v } \| _ { 2 }$ for $i \in G ^ { \prime }$ are independent standard normal random variables.   
Let $E _ { 2 } ^ { v }$ e e ebe the event of Proposition A.5 for the random variables $\{ Z _ { i } : i \in H \}$ and weights   
$\{ a _ { i , 1 } ^ { 2 } : i \in H \}$ . That is, with probability at least $1 - 1 0 ^ { - 3 } 1 2 ^ { - d }$ , it holds that $\textstyle \sum _ { i \in H } a _ { i , 1 } ^ { 2 } Z _ { i } ^ { 2 } =$   
$O \Big ( \epsilon ^ { 1 / 2 } \sqrt { d n } \log ^ { 4 } ( 1 / \epsilon ) + d \log n \Big )$ . For the second term of $\overline { J }$ , we can invoke Proposition A.6 over   
the random variables $\{ Z _ { i } : i \in \dot { G } ^ { \prime } \setminus H \}$ with $\alpha = 1 2 \epsilon$ and $m = | G ^ { \prime } \backslash H | \geq n / 2$ (if $\epsilon < 1 / 2$ ). Let   
$E _ { 3 } ^ { v }$ be the event of this proposition, that is, with probability at least $1 - 1 0 ^ { - 3 } 1 2 ^ { - d }$ , it holds that:

$$
\operatorname* { m a x } _ { L \subseteq G ^ { \prime } \setminus H , | L | = 6 \epsilon n } \sum _ { i \in L } ( \widetilde { v } ^ { \top } \widetilde { a } _ { i } ) ^ { 2 } = O ( \epsilon n \log ( 1 / \epsilon ) + d ) .
$$

By taking a union bound of the events 622 $E _ { 2 } ^ { v }$ and $E _ { 3 } ^ { v }$ , we have that:

$$
\begin{array} { r } { \overline { { J } } ( v ) = O \Big ( \epsilon n \log ( 1 / \epsilon ) + \epsilon ^ { 1 / 2 } \sqrt { d n } \log ^ { 4 } ( 1 / \epsilon ) + d \log n \Big ) . } \end{array}
$$

Consider a $1 / 4$ -net $\mathcal { N }$ of $\mathbb { S } ^ { d - 1 }$ , where $| { \mathcal N } | = O ( 1 2 ^ { d } )$ . Note that for any $v \in \mathbb { S } ^ { d - 1 }$ , it holds 624 $\begin{array} { r } { \operatorname* { s u p } _ { v \in \mathbb { S } ^ { d - 1 } } \overline { { J } } ( v ) \leq 2 \operatorname* { s u p } _ { v \in \mathcal { N } } \overline { { J } } ( v ) } \end{array}$ . Thus, by taking a union bound over the event described in (21) for all 625 $v \in \mathcal N$ , we obtain that with probability at least $1 - 2 \cdot 1 0 ^ { - 3 }$ , we have:

$$
\operatorname* { s u p } _ { v \in S ^ { d - 1 } } \overline { { J } } ( v ) = O \Big ( \epsilon n \log ( 1 / \epsilon ) + d \log n + \sqrt { d n \epsilon } \log ^ { 2 } ( 1 / \epsilon ) \Big )
$$

We finally combine (22) and (20) to conclude that with probability at least $1 - 0 . 9 9 5$ , it holds that

$$
J \leq O \left( \frac { 1 } { n } \left[ d \log n + \epsilon ^ { 1 / 2 } \sqrt { d n } \log ^ { 4 } ( 1 / \epsilon ) + n \sqrt { \epsilon } \log ^ { 8 } ( 1 / \epsilon ) \right] \right) \ .
$$

We can pick a sufficiently small $\epsilon$ (depending only on $\delta$ ) and $n \geq c ( \delta ) d \log d$ for a sufficiently large   
constant $c ( \delta )$ so that with probability at least $1 - 0 . 9 9 5$ , it holds that $J \le \delta / 2$ . Utilizing this result   
along with (11) to upper bound (10) yields the desired statement. □

# 30 B Omitted Proofs in Section 5

Lemma 5.1. Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be the ground-truth vector. Let $\mathcal { D } _ { z }$ be the distribution of gradients at $z$ as defined in Equation (2). For any $z$ with $\| z - x \| _ { 2 } \leq 1$ , the covariance matrix $\Sigma _ { z }$ of $\mathcal { D } _ { z }$ satisfies

$$
\Sigma _ { z } \preceq O \left( \| z - x \| _ { 2 } ^ { 2 } \right) I \ .
$$

Proof of Lemma 5.1. Recall that $g \sim \mathcal { D } _ { z }$ is distributed as

$$
g = \frac { \partial } { \partial z } \left[ \left( \left. a , z \right. ^ { 2 } - \left. a , x \right. ^ { 2 } \right) ^ { 2 } \right] = 4 \left( \left. a , z \right. ^ { 2 } - \left. a , x \right. ^ { 2 } \right) \left. a , z \right. a \quad \mathrm { w h e r e } \quad a \sim \mathcal { N } ( 0 , I ) .
$$

Let 634 $\mu _ { z } = \mathbb { E } _ { g \sim \mathcal { D } _ { z } } [ g ]$ . We have

$$
\begin{array} { r } { 0 \preceq \Sigma _ { z } = \underset { g \sim \mathcal { D } _ { z } } { \mathbb { E } } \left[ g g ^ { \top } \right] - \mu _ { z } \mu _ { z } ^ { \top } \preceq \underset { g \sim \mathcal { D } _ { z } } { \mathbb { E } } \left[ g g ^ { \top } \right] \ . } \end{array}
$$

Consequently, it suffices to upper bound the spectral norm of 635 $\mathbb { E } _ { g \sim \mathcal { D } _ { z } } \left[ g g ^ { \top } \right]$ . Let $h = z - x$

$$
\begin{array} { r l } { | \mathbb { E } _ { \lambda , \tau } [ \int _ { 0 } ^ { \tau } \log | ( \frac { \tau } { \lambda } ) | ] _ { \tau } | } & { = } \\ & { - \frac { \lambda } { \mathrm { b } \tau ^ { 2 } \lambda } \sum _ { i = 1 } ^ { N } \tau _ { i } ^ { \lambda } \int _ { 0 } ^ { \tau } \log | \tau | } \\ & { = \frac { \lambda } { \mathrm { b } \tau ^ { 2 } \lambda } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } | \partial _ { i } \phi ^ { i } | ^ { 2 } } \\ & { = \frac { \lambda } { \mathrm { b } \tau ^ { 2 } \lambda } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } | \partial _ { i } \phi ^ { i } | ^ { 2 } } \\ & { - \frac { \lambda } { \mathrm { b } \tau ^ { 3 } \lambda } \sum _ { i = 1 } ^ { N } \exp ( \sum ( \Delta \omega ^ { i } - \Delta \phi ^ { i } ) ^ { i } | \partial _ { i } \omega ^ { i } \xi ^ { i } | ) | \partial _ { i } \xi ^ { i } | ^ { 2 } | } \\ & { - \frac { \lambda } { \mathrm { b } \tau ^ { 3 } \lambda } \sum _ { i = 1 } ^ { N } \frac { \lambda } { \mathrm { b } \tau ^ { 3 } \lambda } \sum _ { j = 1 } ^ { N } [ ( | \partial _ { i } \Delta \xi ^ { i } | ^ { 2 } + 2 ( \Delta \xi ^ { i } \Delta ^ { i } ) ^ { i } | \partial _ { i } \xi ^ { j } | ) \partial _ { i } \xi ^ { i } | ^ { 2 } ] } \\ & { - \frac { \lambda } { \mathrm { b } \tau ^ { 3 } \lambda } \sum _ { i = 1 } ^ { N } \frac { \lambda } { \mathrm { b } \tau ^ { 3 } \lambda } \sum _ { j = 1 } ^ { N } [ \partial _ { i } \Delta ^ { i } ] ^ { 2 } | \partial _ { i } \Delta ^ { i } | ^ { 2 } | \partial _ { i } \xi ^ { i } | ^ { 2 } | } \\ &  \leq \ln \frac { \lambda } { \mathrm { b } \tau ^ { 3 } \lambda } ( \int | \partial _ { i } \Delta \xi ^ { i } | ^ { 2 } | \partial _ { i } \xi ^ { i } | ^ { 2 } | \end{array}
$$

The last inequality follows from the Cauchy-Schwarz inequality. The last step uses the fact that   
$\| 2 x + h \| _ { 2 } = { \cal { O } } ( \bar { 1 } )$ and $\| x + h \| _ { 2 } = O ( 1 )$ , which follows from $\| \dot { { \boldsymbol x } } \| _ { 2 } = 1$ and $\| \bar { h } \| _ { 2 } \leq 1$ . □   
Lemma 5.3. Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be the ground-truth vector. Suppose at iteration t of Algorithm 2, the   
current solution $z _ { t }$ satisfies $\begin{array} { r } { \| z _ { t } - \check { x } \| _ { 2 } \leq \frac { 1 } { 8 } } \end{array}$ . Let $\mu _ { z _ { t } }$ denote the expected true gradient at $z _ { t }$ defined   
in Equation (3). Suppose the estimated gradient $\widehat { \mu } _ { z _ { t } } \in \mathbb { R } ^ { d }$ satisfies $\| \widehat { \mu } _ { z _ { t } } - \mu _ { z _ { t } } \| _ { 2 } \leq c \| z _ { t } - x \| _ { 2 }$ for   
$c = 4$ . Then, we have

$$
\left\| z _ { t + 1 } - x \right\| _ { 2 } ^ { 2 } \leq 0 . 9 9 \left\| z _ { t } - x \right\| _ { 2 } ^ { 2 } .
$$

Proof of Lemma 5.3. Recall that $g \sim \mathcal { D } _ { z }$ is distributed as

$$
g = \frac { \partial } { \partial z } \left[ \left( \left. a , z \right. ^ { 2 } - \left. a , x \right. ^ { 2 } \right) ^ { 2 } \right] = 4 \left( \left. a , z \right. ^ { 2 } - \left. a , x \right. ^ { 2 } \right) \left. a , z \right. a \quad \mathrm { w h e r e } \quad a \sim \mathcal { N } ( 0 , I ) .
$$

We can compute the mean $\mu _ { z }$ of $\mathcal { D } _ { z }$ using moments of Gaussian:

$$
\begin{array} { r } { \mu _ { z } = \underset { g \sim \mathcal { D } _ { z } } { \mathbb { E } } [ g ] = \left( 1 2 \left. z \right. _ { 2 } ^ { 2 } - 4 \left. x \right. _ { 2 } ^ { 2 } \right) z - 8 \left. x , z \right. x . } \end{array}
$$

Consider one step of gradient descent in Algorithm 2: $z _ { t + 1 } = z _ { t } - \eta \widehat { \mu } _ { z _ { t } }$ , where $\widehat { \mu } _ { z _ { t } }$ is close to $\mu _ { z }$ .   
We have

$$
\left\| z _ { t + 1 } - x \right\| _ { 2 } ^ { 2 } = \left\| z _ { t } - \eta \widehat { \mu } _ { z _ { t } } - x \right\| _ { 2 } ^ { 2 } = \left\| z _ { t } - x \right\| _ { 2 } ^ { 2 } - 2 \eta \left. \widehat { \mu } _ { z _ { t } } , z _ { t } - x \right. + \eta ^ { 2 } \left. \widehat { \mu } _ { z _ { t } } , \widehat { \mu } _ { z _ { t } } \right.
$$

To prove convergence, we need to lower bound 646 $\langle \widehat { \mu } _ { z _ { t } } , z _ { t } - x \rangle$ and upper bound $\langle \widehat { \mu } _ { z _ { t } } , \widehat { \mu } _ { z _ { t } } \rangle$

Let $z = z _ { t }$ and $h = z - x$ . Substituting $z = x + h$ in the expression for $\mu _ { z }$ , we get:

$$
\begin{array} { r l } & { \mu _ { z } = \left( 1 2 \left\| x + h \right\| _ { 2 } ^ { 2 } - 4 \left\| x \right\| _ { 2 } ^ { 2 } \right) \left( x + h \right) - 8 \left. x , x + h \right. x } \\ & { \quad = \left( 1 6 \left. x , h \right. + 1 2 \left\| h \right\| _ { 2 } ^ { 2 } \right) x + \left( 8 \left\| x \right\| _ { 2 } ^ { 2 } + 2 4 \left. x , h \right. + 1 2 \left\| h \right\| _ { 2 } ^ { 2 } \right) h . } \end{array}
$$

We will be using the assumptions of this lemma: 648 $\| x \| _ { 2 } = 1$ , $\begin{array} { r } { \| h \| _ { 2 } \leq \frac { 1 } { 8 } } \end{array}$ , and $\left\| \widehat { \mu } _ { z } - \mu _ { z } \right\| _ { 2 } \leq c \left\| h \right\| _ { 2 }$ .   
First we lower bound $\langle \widehat { \mu } _ { z } , h \rangle$ .

$$
\begin{array} { r l } & { \langle \widehat { \mu } _ { z } , h \rangle = \langle \mu _ { z } , h \rangle + \langle \widehat { \mu } _ { z } - \mu _ { z } , h \rangle } \\ & { \qquad = 1 6 \left. x , h \right. ^ { 2 } + 3 6 \left. x , h \right. \left\| h \right\| _ { 2 } ^ { 2 } + 8 \left\| x \right\| _ { 2 } ^ { 2 } \left\| h \right\| _ { 2 } ^ { 2 } + 1 2 \left\| h \right\| _ { 2 } ^ { 4 } + \langle \widehat { \mu } _ { z } - \mu _ { z } , h \rangle } \\ & { \qquad \geq - \frac { 8 1 } { 4 } \left\| h \right\| _ { 2 } ^ { 4 } + 8 \left\| x \right\| _ { 2 } ^ { 2 } \left\| h \right\| _ { 2 } ^ { 2 } + 1 2 \left\| h \right\| _ { 2 } ^ { 4 } - c \left\| h \right\| _ { 2 } ^ { 2 } } \\ & { \qquad \geq \left( - \frac { 8 1 } { 2 5 6 } + 8 + \frac { 1 2 } { 6 4 } - c \right) \left\| h \right\| _ { 2 } ^ { 2 } } \\ & { \qquad \geq \left( 7 . 5 - c \right) \left\| h \right\| _ { 2 } ^ { 2 } . } \end{array}
$$

The first inequality uses the fact that 650 $1 6 \left. x , h \right. ^ { 2 } + 3 6 \left. x , h \right. \left\| h \right\| _ { 2 } ^ { 2 }$ is a quadratic function of $\langle x , h \rangle$ , which has minimum value 651 $- { \frac { 8 1 } { 4 } } \left\| h \right\| _ { 2 } ^ { 4 }$ for all $\langle x , h \rangle \in \mathbb { R }$ .

Next we upper bound 652 $\| \widehat { \mu } _ { z } \| _ { 2 }$ using the triangle inequality.

$$
\begin{array} { r l } & { \| \widehat { \mu } _ { z } \| _ { 2 } \leq \| \mu _ { z } \| _ { 2 } + \| \widehat { \mu } _ { z } - \mu _ { z } \| _ { 2 } } \\ & { \qquad \leq \left( 1 6 \left. x , h \right. + 1 2 \left\| h \right\| _ { 2 } ^ { 2 } \right) \left\| x \right\| _ { 2 } + \left( 8 \left\| x \right\| _ { 2 } ^ { 2 } + 2 4 \left. x , h \right. + 1 2 \left\| h \right\| _ { 2 } ^ { 2 } \right) \left\| h \right\| _ { 2 } + c \left\| h \right\| _ { 2 } } \\ & { \qquad \leq \left( 1 6 + \frac { 1 2 } { 8 } + 8 + \frac { 2 4 } { 8 } + \frac { 1 2 } { 6 4 } + c \right) \left\| h \right\| _ { 2 } } \\ & { \qquad \leq \left( 2 9 + c \right) \left\| h \right\| _ { 2 } ~ . } \end{array}
$$

Putting everything together, we have

$$
\begin{array} { r } { \left\| z _ { t + 1 } - x \right\| _ { 2 } ^ { 2 } = \left\| z _ { t } - x \right\| _ { 2 } ^ { 2 } - 2 \eta \left. \widehat { \mu } _ { z _ { t } } , z _ { t } - x \right. + \eta ^ { 2 } \left. \widehat { \mu } _ { z _ { t } } , \widehat { \mu } _ { z _ { t } } \right. } \\ { \leq \left[ 1 - 2 ( 7 . 5 - c ) \eta + ( 2 9 + c ) ^ { 2 } \eta ^ { 2 } \right] \left\| z _ { t } - x \right\| _ { 2 } ^ { 2 } . } \end{array}
$$

Choosing 654 $c = 4$ and $\eta = 1 / 3 0 0$ gives that $\left\| z _ { t + 1 } - x \right\| _ { 2 } ^ { 2 } \leq 0 . 9 9 \left\| z _ { t } - x \right\| _ { 2 } ^ { 2 }$ .

In this work, we consider a general setting that allows adversarial corruption in both the measurement   
vectors $a _ { i }$ ’s and the intensity measurements $y _ { i }$ ’s. Prior work in robust phase retrieval has addressed a   
special case where the adversarial corruption is restricted to the $y _ { i }$ ’s, still assuming that the measuring   
vectors $a _ { i }$ ’s are independently sampled from the Gaussian distribution [27, 51]. In this section, we   
construct a counter-example demonstrating the failure of algorithms developed for the restricted   
corruption setting when applied to the more general setting considered in our paper.   
The Median Truncated Wirtinger Flow Algorithm [51] is an algorithm to address the robust phase   
retrieval problem with adversarial corruption limited to the $y _ { i }$ ’s. The algorithm first initializes $z ^ { ( 0 ) }$   
using the spectral method. Let $\alpha \geq 3$ . In particular, $z ^ { ( 0 ) }$ is computed as the top eigenvector of   
the empirical matrix $\begin{array} { r } { Y : = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } y _ { i } a _ { i } a _ { i } ^ { \top } \mathbb { 1 } _ { | y _ { i } | \leq \alpha ^ { 2 } \mathrm { m e d } ( \{ y _ { i } \} _ { i = 1 } ^ { m } ) } } \end{array}$ that only uses a truncated set of   
samples, where the threshold is determined by $\mathrm { m e d } ( \{ y _ { i } \} _ { i = 1 } ^ { m } )$ , the median over all $y _ { i }$ ’s. The analysis   
of the algorithm relies on the fact that as long as the fraction of outliers is not too large and the sample   
complexity is large enough, the initialization is guaranteed to be within a small neighborhood of the   
ground truth.   
We show that this initialization can fail to remove the distortion introduced by the adversarial   
if corruption is allowed for both $a _ { i }$ ’s and $y _ { i }$ ’s. Let $x \in \mathbb { S } ^ { d - 1 }$ be the ground truth unit vector.   
We construct an $\epsilon$ -corruption adversary that can manipulate the top eigenvector of the empirical   
covariance matrix $\begin{array} { r } { Y = \dot { \sum } _ { i = 1 } ^ { n } y _ { i } a _ { i } a _ { i } ^ { \top } } \end{array}$ , even when all $y _ { i }$ ’s are accurately calculated as $y _ { i } = ( a _ { i } ^ { \top } x ) ^ { 2 }$ .   
Let $u \in \mathbb { S } ^ { d - 1 }$ be a unit vector such that $x ^ { \top } u = 0$ . Suppose the adversary changes $1 \%$ of the   
$a _ { i }$ ’s to $a _ { i } = \sqrt { d - 1 / 2 5 } \cdot u + ( 1 / 5 ) \cdot x ,$ , and suppose that all the $y _ { i }$ ’s are accurate. In particular,   
the length of the corrupted $a _ { i }$ ’s is comparable to the length of a random Gaussian vector, and   
the corresponding intensity measurements satisfy $y _ { i } = ( \breve { a } _ { i } ^ { \top } x ) ^ { 2 } = 1 / 2 5$ . Let $z = ( a ^ { \top } x ) ^ { 2 }$ for a   
random vector $\bar { a ^ { \mathrm { ~ \scriptsize ~ \cdot ~ } } } \sim \mathcal { N } ( 0 , \bar { I } )$ . By direct computation, note that $\mathbf { P r } [ z \geq 0 . 2 ] \ \geq \ 0 . 6$ . Thus, with   
high-constant probability, the median-truncated initialization in [51] is not able to filter out any of   
those samples. However, after the adversarial corruption, the top eigenvector of √ $\mathbb { E } \big [ \sum _ { i = 1 } ^ { n } y _ { i } a _ { i } a _ { i } ^ { \top } \big ] \approx$   
$O ( d ) u u ^ { \top } + O ( \sqrt { d } ) ( u x ^ { \top } + x u ^ { \top } ) + O ( 1 ) ( I + 2 x x ^ { \top } )$ will be manipulated to $u$ , which is far from   
the ground truth $x$ .   
The checklist is designed to encourage best practices for responsible machine learning research,   
addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove   
the checklist: The papers not including the checklist will be desk rejected. The checklist should   
follow the references and follow the (optional) supplemental material. The checklist does NOT count   
towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

• You should answer [Yes] , [No] , or [NA] .   
• [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.   
• Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

• Delete this instruction block, but keep the section heading “NeurIPS Paper Checklist", • Keep the checklist subsection headings, questions/answers and guidelines below. • Do not modify the questions and only use the provided macros for your answers.

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: We provide formal statements on the guarantee of our algorithm, and state our main theorem. The rest of the paper is devoted in proving this theorem.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the paper.   
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.   
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations and questions for future work are discussed in the conclusion.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.   
• The authors are encouraged to create a separate "Limitations" section in their paper.   
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.   
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Sections 3-5 and the first two sections of the appendix are devoted to the proof of our main result.

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

Justification: The paper does not contain experimental results.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.   
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: The paper does not include experiments requiring code.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.   
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.   
• While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).   
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.   
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

• The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

Guidelines:

• The answer NA means that the paper does not include experiments. • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

# 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research follows the NeurIPS Code of Ethics.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.   
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.   
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.   
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.   
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.   
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.   
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.   
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper is theoretical and poses no such risks.

Guidelines:

• The answer NA means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper is theoretical and does not use existing assets.

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

Justification: The paper does not release new assets.

Guidelines:

• The answer NA means that the paper does not release new assets.   
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
• The paper should discuss whether and how consent was obtained from people whose asset is used.   
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing experiments.

# Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.   
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

# 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
