## 1. Manipulation Inversion by Adversarial Learning on Latent Statistical Manifold
- PDF: https://openreview.net/pdf?id=qyGurHI4As
- Review ID: `qZgRzWyHFc`
- Author Reply ID: `Di9JLoNTWT`
- **Reviewer comment (formula issue)**:

> Thank you for your response. I appreciate the clarifications, but I still have a few questions:

1. In the updated Algorithm 1, there are some square symbols—could you clarify what they represent?

2. In #1qCg [Q3], it is mentioned that the ClipDiff score is defined. However, the results are not presented as they are in Algorithm 1 and Tables 1, 2, and 3. Could you provide more details or clarify this?

3. Would it be possible to include an example of hubness initialization in the final version? It sounds like a significant improvement, and I would appreciate a concrete demonstration.

But I'm happy to improve the score.
- **Author acknowledgement**:

> We thank the reviewer very much for the valuable comments and insightful suggestions. We are glad that we've addressed your questions!

**[Q1 Clarifying Algorithm 1]**  
The square symbols in [Algorithm 1](https://anonymous.4open.science/r/icml2025_14926/alg.pdf) indicate squared $l_2$-norms. More specifically, the perturbation loss $\psi(\mathbf{w}, \mathbf{v}) = ||f(g(\mathbf{w} + \beta \frac{\mathbf{v}}{||\mathbf{v}||_2})) - \beta \frac{\mathbf{v}}{||\mathbf{v}||_2} - \mathbf{w}||_2^2$ measures the squared $l_2$ distance between the perturbed reconstruction $f(g(\mathbf{w} + \beta \frac{\mathbf{v}}{||\mathbf{v}||_2})) - \beta \frac{\mathbf{v}}{||\mathbf{v}||_2}$ and the original latent code $\mathbf{w}$, where the subscript $||\cdot||_2$ denotes the $l_2$ norm and the superscript $||\cdot||^2$ is the squaring operation. Similarly, the manipulation inversion loss $\psi(\mathbf{w}, \mathbf{v^*})$ also employs the squared $l_2$-norm to quantify reconstruction error from the encoder output.

**[Q2 Clarifying ClipDiff Score]**  
Yes, we introduce ClipDiff to evaluate editing performance on the Church dataset, where ground-truth attribute directions are not available. We report the ClipDiff in [Table 1](https://anonymous.4open.science/r/icml2025_14926/table_edit.pdf), under the “Church Editing” section. This is due to unlike human faces with predefined semantic attributes (e.g., eyeglasses), Church dataset lacks ground-truth annotated attributes, in which the edit directions are obtained via GANSpace in an unsupervised style. Therefore, we cannot rely on traditional metrics such as ID and CLIP scores in [Table 2](https://anonymous.4open.science/r/icml2025_14926/table_id.pdf), in which ground-truth labels are required. On the other hand, [Table 3](https://anonymous.4open.science/r/icml2025_14926/tab_reconstruction_quatitative_results.pdf) reports the reconstruction accuracy, in which the ClipDiff score may not be suitable.

To address this, we develop ClipDiff to evaluate editing effectiveness, which is defined as the cosine distance between CLIP image embeddings from the input and edited images. A larger ClipDiff score indicates a distinct semantic shift, capturing editing content. We also use ClipIQA as complementary to assess the perceptual quality of the edited images. This ensures that the editing performance is evaluated from being both semantically meaningful and visually coherent.

We therefore report both ClipDiff and ClipIQA scores for the Church dataset in [Table 1](https://anonymous.4open.science/r/icml2025_14926/table_edit.pdf), under the “Church Editing” section. From this table, our method achieves the highest ClipDiff ($\uparrow0.4273$) and ClipIQA ($\uparrow0.5104$), indicating that our edits are both semantically distinct and perceptually superior quality than all the baselines. We shall further clarify our comparisons based on ClipDiff score in our revised version, together with comprehensive evaluations for editing and comparing methods.

**[Q3 A Concrete Demonstration of Hubness Initialization]**  
We appreciate the insightful feedback! Following the suggestion, we conducted additional experiments on hubness latent features. Indeed, the input of our method is the real-world images, and we added new experiments on our inverted latent features from real-world images, acting as the initialization for the StyleGAN generator. 

We then calculated the portion of falling into the high-density regions, i.e., belonging to the hubness latent feature that deteriorates the inversion. More specifically, since the threshold $t$ determines the minimum number of $k$-nearest points for the current latent feature to be regarded as the hubness, we inverted the latent features from $10k$ test images and evaluated the numbers of hubness latent features under varying $t$  thresholds. We report the results in [Figure 3](https://anonymous.4open.science/r/icml2025_14926/fig_hubness.pdf) and [Table 5](https://anonymous.4open.science/r/icml2025_14926/table_hubness.pdf), in which the default setting of $t$ is 50 in the suggested Ref. [5]. As can be seen from this figure and table, our approach consistently results in the smallest numbers of hubness latent features across $10k$ samples and different $t$ thresholds, whereas baseline methods such as FSE, E2Style and pSp still exhibit considerable concentration in these problematic areas. For the default setting of $t=50$, our method achieves non-hubness latent features for all $10k$ samples. This statistically demonstrates that our encoder-driven mapping effectively avoids high-density regions—commonly referred to as hubness—in the $W$-space of StyleGAN, which negatively impact inversion quality as pointed out by the reviewer. 

In our revised version, we will further clarify this in our final version, emphasizing that the observed advantage is an natural property of our pretrained encoder initialization and manifold-assisted optimization on the latent space of StyleGAN.

## 2. The Expressivity of Fixed-Precision Transformers without Positional Encoding
- PDF: https://openreview.net/pdf?id=3TGUvHmZ2v
- Review ID: `1T82djHBUv`
- Author Reply ID: `mlErywrGVw`
- **Reviewer comment (formula issue)**:

> Thank you for the helpful response! I will consider this further along with the other reviews (and their responses) to settle on a final recommendation. 

This is minor, but the fact that $\sigma \in \Sigma$ wasn't the confusing part. I was confused by how the transformer input could be represented as $\sigma$ concatenated with just the prior timestep's output. I would think it would have to be concatenated with all prior outputs.
- **Author acknowledgement**:

> Thank you. Now I understand what you say. It's true that my definition doesn't allow reference to past outputs. I'll improve it later.

## 3. Graph Transformers Get the GIST: Graph Invariant Structural Trait for Refined Graph Encoding
- PDF: https://openreview.net/pdf?id=Ck6WljG6ZM
- Review ID: `PDsKbBQgGx`
- Author Reply ID: `Ic6zKMFTTz`
- **Reviewer comment (formula issue)**:

> Thanks for the efforts. So, in Fig.1, $I(2, 2)$ means $I_{2, 2}(u, v)$ as introduced in Sec.3.1, authors should keep the consistency of the expression and avoid unnecessary omissions. Anyway, additional information has addressed most of my concerns. As for the efficiency problem, I am more concerned about the prediction time comparison with baselines, rather than the precomputing and training time alone. To sum up, I am willing to improve the rating from 2 to 3.
- **Author acknowledgement**:

> ### **`W1 - GIST inference efficiency (prediction time):` Sure, here we provide comparison of inference time between GIST and baselines.**

We thank the reviewer for acknowledging the merit of our method and raising the score from 2 to 3. We also appreciate the reviewer’s clarification on inference efficiency. **Table 1** below reports the inference time of **GIST and other baselines** across four datasets, where inference time is **a one-time structural encoding pre-processing time + model prediction time for a single batch of size 32 (zinc) or of size 16 (petides-struct, func)**. For any given graph—whether in training or testing—GIST features require only a one-time precomputation. Our results show that **GIST’s inference time is on par with other graph transformers**, demonstrating its efficiency in real-world applications.  

**Table 1**: Inference Time (in seconds)
| Datasets | ZINC | ZINC-full | Peptides-struct | Peptides-func |
|-|:-:|:-:|:-:|:-:|
| GIST | 0.3 | 0.3 | 0.7 | 0.7 |
| GRIT | 0.03 | 0.03 | 0.1 | 0.1 | 
| HDSE | 0.42 | 0.42 | 1.1 | 8.6 |

We will ensure that all these thoughtful discussions and suggestions from the reviewer (e.g., notation consistency) are incorporated into the later version. While the reviewer has already improved the rating—which we sure appreciate—we shamelessly venture to ask for a further improvement if inference efficiency is the only remaining concern; as we believe it should be well-addressed with the inference results above.

---

Last, we want to take this opportunity to highlight **someconcerns raised by multiple reviewers that have already been acknowledged by reviewer `e5VJ`**, such as:  
- **W1 on "one-time computation overhead" and "inference efficiency"**: Both `L3nR` and `e5VJ` raised similar concerns about GIST’s efficiency. We thank `e5VJ` for recognizing the effectiveness of our estimation algorithm in mitigating GIST's pre-computation overhead. We hope that our not supplied inference time results further clarify GIST’s **practical viability**, given its minimal additional overhead and efficient end-to-end performance.

## 4. IO-LVM: Inverse Optimization Latent Variable Models with Graph-based Planning Applications
- PDF: https://openreview.net/pdf?id=k8wsUSPGgZ
- Review ID: `ypoCgT70yX`
- Author Reply ID: `QEjwVylD2R`
- **Reviewer comment (formula issue)**:

> Q1) Thank you for answering this question, I acknowledge that this was not aware to me.

Q2) I still do not understand, from your rebuttal, why is x_eps treated to be independent of y or theta, when infact 
$x_{\epsilon} = argmin E_{\epsilon} [ <y_{\theta} + \epsilon, x> ]$. Does some mathematical trick allow you to do this?


Q3) Thank you for answering this question, I acknowledge that this was not aware to me.

Q4) While I agree with you that IO-VLM seems like it can make better use of the data using a low dimensional latent space, VAEs might be able to learn better given a higher dimensional latent space. I do not see any result with dimensions greater than 10 in the rebuttal to Xit2. More generally, I acknowledge that I do not know how informative it is to show data efficiency of these seemingly toy problem.

Q6) Thank you for reporting this.

I will increase my score to weak accept given the rebuttal to my reviews and other reviews combined with the fact that I am not well versed with this general area of research.
- **Author acknowledgement**:

> First,we would like to thatnk the reviewer to engage with the rebuttal. 
The reviewer has two further remain requests of clarifications that we address below:
* Q2. Why x_eps is intependent of y when taking the gradients.
* Q4. VAEs could potentially be better with more dimensions.


Yes, it is a mathematical trick. While a formal mathematical treatment can be found in [Learning with Fenchel-Young Losses, 2019] and [Learning with Differentiable Perturbed Optimizers, 2020], here we provide an intuitive explanation to illustrate why this is true with an example:

.

* Answer for Q2.

**What do we want to show?**

We aim to show that the partial derivative of the following expression:

$\langle y^{\theta} + \epsilon, \hat{x}_{\epsilon}^{\theta} \rangle$

with respect to $y^{\theta}$  is simply $\hat{x}_{\epsilon}^{\theta}$.

The derivative of $y^{\theta}$  with respect to $\theta$  itself is not relevant here, as this derivative is accounted in a separate term by the neural network's gradient as shown in Eq.6 in the paper (i.e., $\partial{g_{\theta}}$..). Here, we focus exclusively on why the derivative with respect to the argument $y^{\theta}$ of the inner product is exactly  $\hat{x}_{\epsilon}^{\theta}$.

**Simplifying the notation**

By using change of variable $y' = y^{\theta} + \epsilon$, our original problem simplifies to showing that $\frac{\partial}{\partial y'} \langle y', \hat{x} \rangle$ is exactly $\hat{x}$ and doesn't depend on $y'$, where as you already mentioned, $\hat{x} = \arg\min_{x \in \mathcal{X}} \langle y', x \rangle$.

**An example for intuition**

Let us fix $y' = [8, 3, 5, 9, 15]$, and that we have a linear funtion defined by $\langle y', x \rangle$. In a minimization problem to pick the smallest value in a feasible set (e.g., pick the path with lowest cost), the result using our example is $\hat{x} = [0, 1, 0, 0, 0]$, because the second element is the smallest. And this leads to the minimum value of the function to be $\langle y', \hat{x} \rangle = 3$

**The partial derivative**

Now we can evaluate $\frac{\partial}{\partial y'} \langle y', \hat{x} \rangle$ is exactly $\hat{x}$ by slightly changing the elements of $y'$. 

If we slightly perturb the first element (8 → 8.001), the minimum value of $\langle y', \hat{x} \rangle$ remains 3, because it still chooses the second element.

If we slightly perturb the second element (3 → 3.001), the minimum immediately changes from 3 to 3.001.

Perturbing other elements (5, 9, 15) does not affect the minimum since they are not chosen.

**What does it mean?**

The value of $\langle y', \hat{x} \rangle$ changes only if we perturb the coordinate of $y'$ associated with the optimal solution (the chosen element). This change is linear due to the linearity of the inner product. Therefore, the derivative vector (the gradient) is exactly [0,1,0,0,0], which is equal to $\hat{x}$


.

* Answer for Q4.

We believe this question is essential to address and add value to the paper. We go further and try to answer the following: How many latent dimensions, and how many samples in the data the VAE needs to be able to capture the problem feasibility? We decided to quickly run an additional experiment only for the VAE with latent dimension = 100 and considering two different training data sizes: 1000 samples and 10000 samples. Below you find the table results together with the results of Table 3 in the rebuttal for reviewer Xit2.

| Methods    | Latent dims | Train size = 1000 | Train size = 10000 |
| -------- | ------- | ------- | ------- |
| VAE  | 10    | 0.8 \%  | 11.8 \% |
| VAE | 100    | 1.3 \% | 30.2 \% |
| IO-LVM  | 10    | 20.8 \%   | 58.3 \% |

We see that it might happen that, with a lot of training data, the VAE might be able to "memorize" the reconstruction. However, limited amount of training data keeps the VAE results extremely low in terms of reconstruction. This is because the VAE is not reconstructing the solver input, but the solver output, which is a much harder task with a limited amount of samples.

Thank you again for the additional questions.

## 5. PAC-Bayes Bounds for Multivariate Linear Regression and Linear Autoencoders
- PDF: https://openreview.net/pdf?id=1ueDWPv7j9
- Review ID: `slQHC3jkjj`
- Author Reply ID: `wcX3Wxfdfc`
- **Reviewer comment (formula issue)**:

> The authors mention that since Alquier's original bound is a stronger form than their formulation written at the beginning of section 2, than no modification of their statement is necessary. I disagree with this since the prior, in that case, does not play any role and this implies the the KL divergence should be absent in their formulation of the bound. This also means that the KL divergence should also be absent in every risk bound written in their paper since, in all their statements the "\forall \rho" is outside of the probability (of the random draws of the training set). Adding a positive quantity like the KL divergence in the risk bound still makes it valid but it also makes it much looser and irrelevant if the empirical risk of the returned posterior (from the training set) is evaluated on the test set! Admittedly in their rebuttal, this is exactly what they have done in their Algorithm 1 to compute the bound. I was not sure about this (while I was reading their paper) as I would have never though that they would make this error. But this is what they did! Note that Shalaeva's et al. (2020) risk bounds have also been incorrectly written but without anymore implications (except of just correcting risk the bound statements) for the correctness of their paper. This is in sharp contrast with the current paper which evaluates a training set bound bound on a test set. Consequently, I lower my evaluation score accordingly.
- **Author acknowledgement**:

> Thanks for your clarification. We apologize for the lack of clarity in our previous answer due to a 5000-word limit.

**On the "methodological flaw"**:

We believe that our computing method for the PAC-Bayes bound does not contain a "methodological flaw". The reason is as follows:

Consider four different statements of PAC-Bayes bounds:
\begin{align*}
    &\text{S1}: \\; P(\forall \rho, \textsf{E}(\rho)) > 1 - \delta \\\\
    &\text{S2}: \\; \forall \rho, P(\textsf{E}(\rho)) > 1 - \delta \\\\
    &\text{S3}: \\; \rho \text{ is a Gibbs posterior}, P(\textsf{E}(\rho)) > 1 - \delta \\\\
    &\text{S4}: \\; \rho \text{ is a Guassian posterior}, P(\textsf{E}(\rho)) > 1 - \delta
\end{align*}

It is easy to see that $\text{S1} \Longrightarrow \text{S2}, \text{S2} \Longrightarrow \text{S3}, \text{S2} \Longrightarrow \text{S4}$. $\text{S1} \Longrightarrow \text{S2}$ is because $P(\forall \rho, \textsf{E}(\rho)) = P(\cap\_\{\rho\} \textsf{E}(\rho)) \le P(\textsf{E}(\rho\_0))$ holds for any single $\rho\_0$, as $\cap\_\{\rho\} \textsf{E}(\rho) \subset \textsf{E}(\rho\_0)$. The bounds in our paper are of the form S2 in Section 2, 3, 4, and of the form S4 in Section 5.

We understand that optimizing $\rho$ on S3 -- searching for the Gibbs posterior $\rho$ -- is likely what you consider the methodologically correct approach. The Gibbs posterior is known to minimize the right hand side of Alquier's bound, making it optimal. However, it is typically non-computable, and approximating it requires sampling-based methods. Sampling can be computationally inefficient on large datasets. For example, on the MSD dataset used in our experiments, even with a high performance GPU, calculating a single sample takes around 10 minutes.

To reduce computational costs, one practical alternative is to use S4 instead of S3, as the bound is in general easier to compute with a Gaussian posterior than with a Gibbs posterior. This approach was used by Dziugaite and Roy [1] for neural network models, and our computational method adapts their approach to LAE models.

Our Algorithm 1 computes the tightest RH of Eq (14) using S4, not using S3. That is, we search for the optimal $\rho$ within the space of Gaussian distributions. One key advantage of Algorithm 1 is high efficiency, thanks to the closed form solution for $\rho$ provided by Theorem 5.2, which eliminates the need for sampling. As demonstrated in our experiments, Algorithm 1 can efficiently compute a sub-optimal RH on large real-world datasets. This sub-optimal result is sufficient for our purposes, as it successfully validates the non-vacuousness of the bound.

**On converting the statement from $\forall \rho, P(...)$ to $P(\forall \rho, ...)$**:

We agree to adopt this conversion for all bounds in Section 2, 3, 4, as we realized that they can be generalized from S2 to S1. We also acknowledge that S1 is a more standard statement in the PAC-Bayes literature [2]. After careful review, we confirm that this change is *independent of* the proofs presented in our paper. Therefore, implementing the change does not affect the correctness or integrity of our theoretical framework, but reveals its more fundamental form.

However, our bound in Section 5 is not applicable to this conversion, since it is of the S4 form where we explicitly take $\rho$ to be Gaussian.

**Other issues related to the reviewer's comments**:

1. " the KL divergence should be absent in their formulation of the bound... since in all their statements the $\forall \rho$ is outside of the probability". The meaning of "absent" here is ambiguous to us. Additionally, we are unclear why "$\forall \rho$ is outside of the probability" would imply that "KL-Divergence is absent from the bound", as the KL-Divergence is explicitly present in the bound regardless of the position of $\forall \rho$.

2. "Adding a positive quantity like the KL divergence in the risk bound still makes it valid but it also makes it much looser". We respectfully disagree with this. Our experimental results show that the KL-Divergence does not make our bound much looser: as shown in the Table 3 of our paper, their values are trivial.

3. "I would have never though that they would make this error. But this is what they did!" As mentioned earlier, we agree that replacing $\forall \rho, P(...)$ with $P(\forall \rho, ...)$  is reasonable and will implement this in the final version. We appreciate you pointing out this issue and helping to improve the paper. However, we respectfully disagree referring this issue as an "error", since it does not involve any logical mistake in our proofs. Our theoretical framework remains correct with the revised statement based on your suggestion.

**References**:

[1] Dziugaite and Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. arXiv, 2017.

[2]  Pierre Alquier. User-friendly introduction to pac-bayes bounds. arXiv, 2021.

## 6. Fisher Divergence for Attribution through Stochastic Differential Equations
- PDF: https://openreview.net/pdf?id=eHIc7SL0sr
- Review ID: `K8OTV73c9B`
- Author Reply ID: `tHygfMPh6c`
- **Reviewer comment (formula issue)**:

> I thank the authors for engaging in discussions.

Can the authors comment on how good a proxy -t is to the mutual information loss? I assume some approximation is employed here.

Furthermore, in the provided algorithm, \theta is updated using \nabla_{\btheta} L, but \theta does not appear in the computational graph of L_{CE}, so it is only trained using L_{MI}. How does this relate to the algorithm provided in response to reviewer 38Uj, in which case the score network seems to be trained using standard diffusion loss?
- **Author acknowledgement**:

> We thank the reviewer for the insightful comments and valuable feedback. Below are our responses to your questions:

**Q1:** Can the authors comment on how good a proxy -t is to the mutual information loss? I assume some approximation is employed here.

**A1:** Since increasing $t$ monotonically decreases the mutual information $I(X_t; X_0)$, we can use $-t$ as a proxy for the mutual information loss term, thereby avoiding the costly integral-based computation during training. Once the optimization is complete and the final $t$-tensor is obtained, we can still compute the exact mutual information using Equations (49) or (50) to produce the final attribution map. This two-stage approach—using $-t$ as a surrogate loss and then performing an exact mutual information calculation—strikes a balance between computational efficiency and theoretical rigor.

**Q2:** Furthermore, in the provided algorithm, \theta is updated using \nabla_{\btheta} L, but \theta does not appear in the computational graph of L_{CE}, so it is only trained using L_{MI}.

**A2:** In our algorithm, although $\theta$ does not appear directly in the expression for $L_{CE}$, the loss $L_{CE}$ is computed on the classifier output $y$ obtained from the perturbed input $z$, and $z$ is generated based on $t$, which in turn is produced by the network parameterized by $\theta$. Thus, $L_{CE}$ indirectly influences $\theta$ via the chain of dependencies $ \theta \rightarrow t \rightarrow z \rightarrow y $. Consequently, both $L_{MI}$ and $L_{CE}$ contribute gradient signals to update $\theta$ in our training process.

**Q3:** How does this relate to the algorithm provided in response to reviewer 38Uj, in which case the score network seems to be trained using standard diffusion loss?

**A3:** In our approach using the Variance Exploding (VE) noise addition method, we require a score network to compute the mutual information with Equations (50) (second stage as explained in A1). If a pretrained score-based diffusion model suitable for our research scenario is unavailable, we train the score network ourselves. Furthermore, when performing attribution on a small number of images, it is unnecessary to employ a diffusion model trained on a large-scale dataset like ImageNet. In our response to reviewer 38Uj, we provided an algorithm for training a score network on a single image—a method that is similar to that presented in Song et al., 2020.

## 7. Fixing Value Function Decomposition for Multi-Agent Reinforcement Learning
- PDF: https://openreview.net/pdf?id=qUtxbtsfwp
- Review ID: `4RCZ3t2mOH`
- Author Reply ID: `XfOAlRSje1`
- **Reviewer comment (formula issue)**:

> Thank you, Authors, for your rebuttal. Stating that you will include the citations for state-history value functions, you made me more optimistic about this work. However, the most crucial problems of mine with this paper have not been addressed.  
> 1. I would have to see any preliminary results (possibly under an anonymous link) before I lean positively towards this paper. Crucially, an ablation which compares some (doesn't have to be all) variants of QFix to non-QFix methods, with bigger networks, is a pre-requisite.  
> 2. It is not true that Definition 3.1 prescribes a unique maximum. Generally, $argmax$ is a set and IGM says that the argmax set of the global value function is the product of argmax sets of local value functions. For example, in case of one-dimensional actions and $N$ agents, if $Q(s,a)=-(a_1 - 1)^2 (a_1 + 1)^2 \dots (a_N - 1)^2 (a_N + 1)^2$, the IGM from Definition 3.1 holds and there are $2^N$ maxima. My original comment remains unchanged.  
> 3. Regarding Lemma 4.2 and Theorem 4.3 and 4.4: you claim the ease with which your results come to be your strength. If your theoretical results are meant to make a contribution of your paper, the formulated problems shouldn't be too obvious to make proofs about. If something follows too easily, it should be acknowledged. For example, I can now define a function class, let's say $Q(h,a) = b(h) - w(h,a)(u_1\cdot \dots \cdot u_N)^2$ where $w(h,a) > 0$ satisfies your conditions. Of course you can prove the same results about it and it is even simpler than your proposed class. This brings me to another point: words like "minimal" should be used with caution - minimality means something very specific.   
> 4. Regarding your use of universal function approximation and citation of QPLEX (which I am familiar with): the fact that QPLEX paper does something does not make it correct. QPLEX, just like you in this rebuttal, refers to universal function approximation theorem. But even that theorem (including the version cited by QPLEX (Csaji, 2021)) assumes that the approximated function is continuous. You do not, and I gave you an example when your theorem breaks. Before I consider raising my score, the theoretical limitations of this paper should be addressed, and its rigor improved. 

**References**  
Csaji, 2021. Approximation with Artificial Neural Networks.
- **Author acknowledgement**:

> We thank the reviewer for the further feedback.
Please see auxiliary figures at [1], which includes:
- Fig 1: Updated results (now 5-6 runs per model per task, also shown as interquartile mean (IQM) [2]), of interest to rev. `5gNj`.
- Fig 2: Probability of improvement [2], of interest to rev. `5gNj`.
- Tab 1, Fig 3: Model size ablation, of interest to revs. `zepP`, `Cnks`.

1. We run additional experiments for QMIX-big (QMIX with bigger size) and Q+FIX-mono-small (Q+FIX-mono with smaller size) on all the 5v5 maps. [1, Table 1]. shows all mixer sizes. In terms of size, QMIX-big is comparable to QPLEX and Q+FIX-mono, while Q+FIX-mono-small is comparable to QMIX. [1, Fig 3] contains the ablation results; to avoid clutter, only QMIX, QMIX-big, Q+FIX-mono and Q+FIX-mono-small are shown (other methods are shown in [1, Fig 1]). These results reaffirm that Q+FIX-mono performs well not because of model size, but often in spite of smaller models, and due to our mixing structure. For the final version of the paper, we will extend this ablation to 10v10 and 20v20.
2. We understand better now that $\argmax$ can itself describe a set of solutions, and does not intrinsically assume a unique maximal element; we agree completely with the reviewer, and will fix the definitions accordingly. We note that this does not affect QIGM or QFIX.
3.
    - We are unable to concretely understand the reviewer’s concern here; if they are saying that the lemmas/theorems are so obvious they need not be stated or proven, then we strongly disagree, and expect other reviewers would have requested formal proof. If they are saying that they do not meet a threshold of importance to be called theorems, but should, e.g., be reformulated as lesser results like propositions, then that is agreeable and we can make these minor changes. If the concern lies elsewhere, we would appreciate further clarification.
    - The example provided is not just another function class; it is very specifically a special case of QIGM for one specific function $f(u_1, \ldots, u_N) = - \sum_i u_i^2$; This is a perfectly valid special case of QIGM, but our result remains strictly speaking a generalization of any of the provided examples. The importance of proving the more general case is that it is necessary for QFIX. Without the general case over a general class of $f$ functions, we would not be able to define QFIX by having a fixee advantage take the role of $f$. Without proving lemma 4.2 and theorem 4.3 for a general class of functions $f$, QFIX would not exist.
    - We agree that our use of the term "minimal" is not formal and can cause confusion; we will happily replace all mention of "minimality" into something less formal like "simplicity".
4. We understand the concern better now; universal function approximation theorems (UATs) come in many forms, but not all UATs are exclusively formulated to approximate continuous functions, at least partially because not all refer to the same notion of approximation. The most well known UAT by Cybenko is formulated in terms of uniform convergence, a very strong notion of approximation. However, there are other forms of UAT that use weaker notions of approximation and that are applicable to approximate non-continuous functions. E.g., Hornik’s Theorem [3, Theorem 1] is another popular UAT, and is based on $p$-norm convergence (with $p<\infty$) that proves approximation to $L^p$ functions; this includes large classes of non-continuous functions. In the same document, Hornik informally formulates a corollary that implies another form of approximation for functions that are merely measurable; this is a version of UAT we can employ while making minimal assumptions on $Q$ and $Q_\text{fixee}$.

    We are happy to clarify these assumptions and conclusions more explicitly; Thm 4.3 is not a statement related to NNs, so it needs no adjustment (it fundamentally states that eqs (38, 39) are the values of $w, b$ sufficient to guarantee QIGM=Q for arbitrary Q). Thm 4.4 does need to be reformulated. We need to assume IGM values that are measurable, and fixees that are also measurable. We must also assume that the fixee’s preimage $A^{-1}_\text{fixee}(0)$ is a measurable set. All of these are fairly mild assumptions. Then, eq (39) is trivially measurable, and eq (38) is measurable as a whole, as it is a piecewise construction based on measurable functions on measurable partitions. Since eqs (38, 39) are measurable, we can apply the corollary informally stated in the discussion section of [3], to justify using neural networks to learn these functions, with the corresponding approximation guarantees on compact subsets of the input space.

**References**

1. https://anonymous.4open.science/r/qfix-icml-rebuttal-5C81/icml-2025-rebuttal.pdf 
2. Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice", NeurIPS 2021.
3. Hornik, "Approximation Capabilities of Multilayer Feedforward Networks", Neural Networks 4, 1991.

## 8. $Q\sharp$: Provably Optimal Distributional RL for LLM Post-Training
- PDF: https://openreview.net/pdf?id=1J1Kju4rto
- Review ID: `cAXPVCOq7p`
- Author Reply ID: `sb15j9MrHF`
- **Reviewer comment (formula issue)**:

> Thanks for the rebuttal. After re-evaluating the manuscript and reading other comments, I still have the following concerns/questions:

1. I am not convinced why the proposed method is superior to the existing policy-based method in the domains beyond the "star-graph" cases. 

2. After finding the optimal parameters of $Z$, how to instantiate the learning policy in practice?

3. Can the authors shed more light on the meaning of "shortcuts"? Why the proposed method can alleviate it?

4. Can you add more details on the derivation from Eq. (3) to Eq. (4)? Should not $V^{*,\eta}_{h}$ correspond to the optimal policy?
- **Author acknowledgement**:

> Dear Reviewer GoFf,

Thank you for your valuable feedback and questions. We truly appreciate the opportunity to answer your remaining questions:

**Q1.**

Thanks for the question. We only intend to claim that Q♯ outperforms policy-based methods on the star-graph task. We apologize for the confusion and will revise the text to clarify this positioning.

We chose star-graph to highlight a known failure case of next-token prediction loss [1]. As shown in [2], it embeds the "sparse parity" problem — determining whether the sum of a binary string is even or odd — which is known to be difficult for gradient-based optimizers and is widely studied in learning theory and optimization [3,4,5,6]. Though the task is simple, it captures an important failure case for policy-based methods that struggle to generalize to unseen graphs due to learning shortcuts (please see our response to Q3 for discussion on shortcuts). In contrast, we show that value-based methods like Q♯ do not learn shortcuts and generalizes well at test time.

Beyond star-graph, a practical benefit of Q♯ is that it allows training a small guiding model while keeping the reference policy frozen. As shown in Table 2, a 1B Q♯ model can effectively steer a 70B reference model. Post-training this 70B model is far more expensive using policy-based methods like REINFORCE or DPO and is infeasible with the compute resources available in our academic setting.

**Q2.**

After estimating the parameters of $Z$ — the conditional distribution of cumulative rewards under $\pi_{ref}$ — we instantiate the policy via $\pi(y\mid x) \propto \pi\_{ref}(y\mid x) \cdot \mathbb{E}_{z\sim Z(x,y)}[\exp(z/\eta)]$, as described in Eq. (5) and implemented in Line 4 of Algorithm 1. This is motivated by the fact that, if we recover the true distribution $Z^\star$, this recovers the optimal KL-regularized policy $\pi^{\star,\eta}$ defined in Eq. (2). Moreover, Theorem 4.4 provides a bound on the regret incurred from using an approximate $Z$ learned from data, showing that the induced policy converges to $\pi^{\star,\eta}$. We’ll include an explicit definition of the induced policy within the algorithm box to make this clearer in the revision.

**Q3.**

By “shortcuts,” we refer to spurious behaviors that arise during pre-training with the auto-regressive next-token prediction. This is also called the Clever Hans Trick by [1], because the model simply learns to follow the next edge as opposed to solving the actual star-graph task. The consequence of this shortcut is that models achieve low training loss by memorizing the first token in the training set and following the Clever Hans Trick, but struggle at test time when the first token is not available. Thus, the model trained with next-token prediction loss fails to generalize on test graphs and achieves poor accuracy. Q♯ avoids this failure mode by **not relying on next-token supervision**. It learns to predict the future cumulative rewards (i.e., reward-to-go) under the reference policy and uses that to guide generation.

**Q4.**

Thank you for the helpful question! We’re happy to clarify this derivation.

Eq. (3) expresses the soft value function under the optimal policy as: $\exp(\eta^{-1}V^{\star,\eta}\_h(x))=\mathbb{E}\_{y\sim\pi\_{ref}(x)}[\exp(\eta^{-1}r\_h(x,y)+\eta^{-1}V^{\star,\eta}\_{h+1}(x’))]$, where $x'=P(x, y)$ is the next state. For LLMs, $x’$ is the concatenation of $x$ and $y$. This follows from the log-sum-exp form of soft value iteration (Eq. (2)), where the optimal policy is an exponential tilting of $\pi\_{ref}$.

We recursively apply this to $V^{\star,\eta}\_{h+1}(x’)$ and so on, unrolling until the final step. Each stage adds a new reward term inside the exponent, and since exponentials multiply, they combine into an exponent of the reward sum — yielding Eq. (4).

Although $V^{\star,\eta}\_{h}(x’)$ corresponds to the soft value of the optimal policy, its *recursion* is expressed via expectations over $\pi\_{ref}$, since the optimal policy is a softmax over $\pi\_{ref}$ weighted by exponentiated cumulative rewards (Eq. (2)). We’ll include this clarification in the revision.

Thanks again for your valuable comments. We hope we have addressed your concerns and we will certainly include all of the above discussion and new experimental results into the final version!

**Citations**

[1] Bachmann and Nagarajan, “The pitfalls of next-token prediction”, ICML 2024.

[2] Hu et al, “The Belief State Transformer”, ICLR 2025.

[3] Shalev-Shwartz et al, “Failures of Gradient-Based Deep Learning”, ICML 2017.

[4] Barak et al, “Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit”, NeurIPS 2022.

[5] Kou et al, “Matching the Statistical Query Lower Bound for k-Sparse Parity Problems with Sign Stochastic Gradient Descent”, NeurIPS 2024.

[6] Abbe and Sandon, “Polynomial-time universality and limitations of deep learning”, Communications on Pure and Applied Mathematics 2023.

## 9. The Minimal Search Space for Conditional Causal Bandits
- PDF: https://openreview.net/pdf?id=GOWRex7nOA
- Review ID: `ujjll3Vd1R`
- Author Reply ID: `ZQ2R6TjSrh`
- **Reviewer comment (formula issue)**:

> Thanks for the detailed response. However, I’m still a bit confused by the statement *"We do not impose any restriction on the policy $g$"*.

For example, the function $g$ could be defined independently of the inputs $\mathbf{Z}_{X}$ so that $do(X= g(\mathbf{Z}_X))=c$, where $c$ can be *any value* within the range. Here I have  some questions.
- Is the intervention means two aspects? The target of intervention and the selection of $g$ function
- How is the mean reward $\mu_a$ defined, is it a function of $g$ (as it seems need to be a function of $c$)?
- How is $g$ selected in C4 algorithm?
- **Author acknowledgement**:

> Thank you for your comment.

Indeed, we can choose $g$ to be a constant function equal to $c$, where $c$ can be any value in the range of $X$.
This is the same as performing the atomic intervention $do(X = c)$. (Are we right to understand that there was a typo in your comment when you wrote $do(X = g(\mathbf{Z}_X)) = c$? We believe you wanted to write $do(X = g(\mathbf{Z}_X) = c)$, which indeed is possible and is just the atomic intervention $do(X = c)$).

We now address each of your questions:

1. Yes, to specify a conditional intervention one needs to select which node $X$ to intervene on, and to choose a policy $g$.
2. The mean reward $\mu_a$, being the expected value of $Y$ if arm $a$ is selected, depends on the conditional intervention (arm) $a$ that is chosen. Since $a$ is characterized by a node $X$ and a policy $g$ (as described in point 1. above), indeed $\mu_a$ depends on $g$.
   Explicitly, we can write: $\mu_a = \mathbb{E}[Y \mid do(X = g(\mathbf{Z}_X))] = \sum_y p(y\mid do(X = g(\mathbf{Z}_X)) \cdot y$.
   Notice that in the case of an atomic intervention $do(X = g(\mathbf{Z}_X) = c)$ this reduces to the atomic intervention mean reward $\mathbb{E}[Y \mid do(X = c)] = \sum_y p(y\mid do(X = c) \cdot y$.
   We can clarify this in the preliminaries section.
3. Please note that the C4 algorithm is an algorithm that finds the minimal set of nodes guaranteed to contain the optimal node on which to perform a conditional intervention (that is, the mGISS of the causal graph). It is not the job of the C4 algorithm to select $g$. Instead, $g$ can be learned using a bandits algorithm for conditional interventions. This bandit algorithm can be restricted to only consider conditional interventions on the nodes found by the C4 algorithm. In Figure 4, you can see the impact of restricting a bandits algorithm search to the nodes found by the C4 algorithm on the cumulative regret curves.

## 10. Scale Invariance of Graph Neural Network for Node Classification
- PDF: https://openreview.net/pdf?id=0aS8nvlxpD
- Review ID: `VPEXHnFlw9`
- Author Reply ID: `iCZyVAqM2i`
- **Reviewer comment (formula issue)**:

> I want to clarify that my comments on the linear model refer to the theoretical results in Section 4. In particular, Theorems 4.1, 4.2, and 4.3 all use linear models with no non-linear activation functions. None of your proofs in Appendix C.3 handle non-linear layers.  

Your notations can be significantly improved by defining directed adjacency matrices $\mathbf{A}$ for in and out edges. You have not explained what does $\sum$ in my first question.

My other concerns listed in the review have also not been adequately addressed.

--------------
Second round:

(1) It is unclear what you mean by "layer-wise propagation". The normal interpretation looking at your theorems and proofs is that you don't include normalization and non-linear activations. These are totally missing in your proofs! Theorems and proofs should be mathematically rigorous. It is not acceptable to over-simplify the results "purely for ease of expression".

(2) Please check your understanding of the UAT. Equation (a) is clearly incorrect. Please see how the UAT is applied in e.g., [4]. You can use MLPs to approximate continuous functions, but this does not imply (a)!

Eq. (b) is equality, not an approximation! It is not acceptable for a prestigious conference like ICML to have a paper that does not follow basic mathematical rigor.

(3) I am saying that your notations can be defined much better. I understand what you have defined in the paper, which is suboptimal, and easy to misinterpret.
- **Author acknowledgement**:

> Thank you for your feedback.

(1) Not a linear model
In Theorems 4.1, we explicitly mentioned "layer-wise propagation." As noted in our rebuttal, a GNN layer can include propagation, normalization, and non-linearity. For simplicity, we focused on the linear components in the subsequent discussions.

We appreciate your feedback on Claim 3 regarding the universal approximation theorem (UAT). To clarify, our model does include non-linear activation functions. In our UAT proof, we stated:
``For a 2-layer GCN, omitting the non-linear activation for simplicity, the propagation outputs are..."
This omission was for ease of expression in the derivation, not to suggest a lack of non-linearity. We will revise the text to explicitly state "for simplicity of expression" to avoid any misunderstanding.

(2) UAT

Thanks to the UAT, if we remove the inner non-linearities and retain only the final activation, the revised model is equivalent in approximation power to the original one.
should be:

$\sigma \left( A \left( \sigma \left( AXW_1 \right) W_2 \right) \right) \approx \sigma \left( AAXW_1W_2 \right)$
    (a)

As we stated "omitting the non-linear activation for simplicity", it correctly becomes:

$A(AXW_1)W_2  \approx AAXW_1W_2  $            (b)

*If you don't agree with equation(a), here is the explanation:

This step follows directly from standard function approximation principles under UAT, which states that a sufficiently expressive neural network can approximate any continuous function. The simplification of nested activations into a single activation over a transformed input is a well-established result in the literature [1–3].

A similar application of UAT in Graph Neural Networks (GNNs) has been explored in prior work [4], further supporting our approach. Additionally, empirical studies have confirmed the effectiveness of removing nested non-linearities in GNN architectures [5].

References
[1] Hanin, B., & Sellke, M. (2017). Approximating continuous functions by ReLU nets of minimal width. arXiv preprint arXiv:1710.11278.
[2] Lu, Z., Pu, H., Wang, F., Hu, Z., & Wang, L. (2017). The expressive power of neural networks: A view from the width. NeurIPS.
[3] Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. Neural Networks, 2(5), 359–366.
[4] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph Neural Networks? ICLR.
[5] Wu, F., Souza, A., Zhang, T., Fifty, C., Yu, T., & Weinberger, K. Q. (2019). Simplifying Graph Convolutional Networks. ICML.

*If you agree with Equation (a) but not Equation (b), we acknowledge that our expression may have been overly simplified, which might have made the transition appear abrupt. We apologize for this and will revise the text to improve clarity and ensure a smoother logical flow.

(3) Notation misunderstanding

In our rebuttal, we explicitly stated:  

    ``$X_I$ consists of all nodes that can be reached from $X$ within 1 step of scaled-edge $I$ (in-edge)."

Thus, $\sum X_I$ represents the sum of features of all 1-step in-neighbors of $X$. Your suggestion of using $AX$ corresponds to all 1-step out-neighbors of $X$, which should instead be denoted as $\sum X_O$.  

Additionally, we defined the adjacency matrix as:  

   ``An adjacency matrix $A \in \{0,1\}^{n \times n}$, where $A_{ij} = 1$ indicates the presence of a directed edge from node $i$ to node $j$."
This definition is clearer than explicitly defining separate adjacency matrices for in- and out-edges, which might be like:  

    ``$A_{ij} = 1$ means an out-edge for node $i$ and an in-edge for node $j$."

(4) Baseline and large graph

The Citeseer dataset in our Table 4 is a directed graph. The 70% accuracy you mentioned refers to the undirected version, where even MLP achieves 71% and GCN reaches 74%. 
The directed Citeseer graph consists of 3312 nodes, while the undirected version has 3327 nodes. They differ in several aspects, including node count, edges and their train/validation/test splits, which results in a variation in performance.
You can verify this using our code: use "citeseer/" for the directed graph and "CiteSeer" for the undirected one.

About scalability, please refer our rebuttal to Reviewer YrK7(first point).

(5) Related work
Thank you for bringing up "Universal Invariant and Equivariant Graph Neural Networks." As discussed in Related Work in Appendix A, our review of existing research on graph invariance has primarily focused on permutation invariance, and your reference aligns with this discussion. However, we appreciate your suggestion and will be happy to include your reference in this category.

Finally, thank you very much for taking the time to read our paper and rebuttal. This is a complex work, and condensing multiple ideas into a limited space is inherently challenging, so we understand that misunderstandings can arise. We truly appreciate your time and consideration.  


Best Regards,
Authors
