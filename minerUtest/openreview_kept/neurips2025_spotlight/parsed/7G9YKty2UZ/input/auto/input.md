# Learning Robust Vision-Language Models from Natural Latent Spaces

Zhangyun Wang∗ School of Computer Science University of Auckland zwna875@aucklanduni.ac.nz

Ni Ding School of Computer Science University of Auckland ni.ding@auckland.ac.nz

# Aniket Mahanti

School of Computer Science University of Auckland a.mahanti@auckland.ac.nz

# Abstract

Pre-trained vision-language models (VLMs) exhibit significant vulnerability to imperceptible adversarial perturbations. Current advanced defense strategies typically employ adversarial prompt tuning to improve the adversarial robustness of VLMs, which struggle to simultaneously maintain generalization across both natural and adversarial examples under different benchmarks and downstream tasks. We propose a collaborative adversarial prompt tuning (CoAPT) approach from pre-trained VLMs to target robust VLMs. Inspired by the image mask modeling, we adopt an improved real-time total variation algorithm to suppress and eliminate high-frequency details from images while preserving edge structures, thereby disrupting the adversarial perturbation space. Subsequently, guided by the high-level image and text representations in the latent space of the pre-trained VLMs, the corrupted natural features are restored while inheriting the superior generalization capability. Experiments on four benchmarks demonstrate that CoAPT achieves an excellent trade-off among natural generalization, adversarial robustness, and task-specific adaptation compared to state-of-the-art methods.

# 1 Introduction

Vision-language models (VLMs) such as CLIP[1] and ALBEF[2] have shown significant potential for application in multiple industry ecosystems in recent years. However, recent studies [3, 4] have revealed that VLMs exhibit a range of concerning vulnerabilities in real-world deployment. When confronted with distributional biases, adversarial samples, or semantic ambiguities, they often display reasoning biases that deviate from human cognition. As an increasing number of downstream applications built upon VLMs as foundational models emerge, the chain reactions triggered by the vulnerability of VLMs pose serious threats to the security and reliability of multimodal downstream tasks. In this paper, we holistically investigate the vulnerabilities of VLMs and their adversarial robustness, with a particular focus on the typical base model CLIP.

Current adversarial robustness strategies for VLMs primarily include model fine-tuning and adversarial prompt tuning. During adversarial training, model fine-tuning [5, 6] relearns the entire set of model parameters to adapt to adversarial examples. This process disrupts the natural data distribution captured by the pre-trained model, leading to a contradiction between robustness and generalization.

Adversarial prompt tuning [7, 8, 9] improves the robust adaptability of VLMs by guiding the pretrained models to efficiently adapt to adversarial data distributions, without altering the pre-trained model parameters. Textual adversarial prompt tuning [10, 11] employs learnable prompts in the language branch to match and counteract adversarial attacks from the visual branch. In contrast, visual adversarial prompt tuning C-AVP[12] directly recognize and refine the adversarial images to allow the pre-trained models to make more accurate predictions. More promising multimodal adversarial prompt methods [7, 13, 14, 8] simultaneously introduce deep learnable prompts into the visual and language branches to achieve more comprehensive adversarial robustness. Although adversarial prompt tuning preserves the generalized feature representations of pre-trained VLMs, excessive reliance on in-distribution adversarial samples causes degradation of their natural generalization distribution during the adaptation process. Out-of-distribution (OOD) or unseen tasks further challenge the natural generalization and robustness of prompt-tuned VLMs[10].

Pre-trained VLMs retain generalizable knowledge for unseen tasks, while adversarial prompts can guide the shift of natural distribution toward adversarial-robust distributions or downstream taskspecific distributions [15]. Therefore, we propose to leverage adversarial prompt tuning to identify a shared latent distribution that effectively balances natural generalization, adversarial robustness, and task-specific adaptation. Due to the inherent discrepancies among different distributions, directly training models with a mixture of natural and adversarial samples to fit the latent distribution leads to suboptimal solutions. Recent findings [16, 17] indicate that masked image modeling (MIM) enables models to learn more generalizable and robust representations, which significantly enhances their capacity to adapt to input distribution variations and improve fine-tuning performance in downstream vision tasks. The success of MIM is due to masked image input and image-level reconstruction objectives. However, this paradigm directs the model to pay more attention to high-frequency (HF) components where adversarial perturbations are concentrated, thus failing to effectively improve adversarial robustness [9]. We propose a collaborative adversarial prompt tuning (CoAPT) in which pre-trained CLIP collaborates with a target robust CLIP to address this issue. We convert the patch-level image masking from MIM to pixel-level image corruption for model inputs. An improved real-time total variation (TV) regularization method is employed to suppress the adversarial perturbation space by drastically smoothing the high-frequency details of the input images while preserving the image edge structures. To mitigate the cost of sacrificing natural high-frequency features, we shift the reconstruction objective from the pixel space of the target robust CLIP to the latent representation space of the natural CLIP. The corrupted natural detail features are restored under the guidance of high-level features of natural CLIP images and texts, thereby inheriting their excellent generalization ability. Overall, the fine-tuned adversarial prompts work in synergy with the frozen weights of the original pre-trained CLIP to support the target robust CLIP. They achieve a good balance between (a) improving adversarial robustness while maintaining natural performance on indistribution tasks, and (b) maintaining natural generalization while enhancing the robust adaptability of the original VLMs on OOD or unseen tasks. Our contributions are threefold:

• We propose a novel paradigm for adversarial prompt tuning that learns robust CLIP from the latent space of natural CLIP. CoAPT weakens high-frequency details of input images to suppress the adversarial perturbation space. Guided by natural CLIP, corrupted generalization features are restored in the latent space. We introduce Rényi divergence to minimize the discrepancy between the similarity distributions of adversarial and natural examples. • We design a real-time adaptive TV regularization method to efficiently suppress the perturbation space. It addresses the slow convergence and residual adversarial perturbations of traditional TV regularization by combining a spatially adaptive regularization strategy based on edge strength response and an accelerated gradient method with adaptive restart. • An optimal trade-off among natural generalizability, adversarial robustness, and task-specific adaptation is achieved. Without benchmark-specific or dataset-specific hyperparameter tuning, we improve natural and adversarial robustness performance on 15 datasets across four benchmarks by an average of $9 . 8 3 \%$ and $2 4 . 1 6 \%$ , respectively.

# 2 Related Work

Adversarial attacks on VLMs. Adversarial attacks induce incorrect decisions in VLMs by applying elaborate and imperceptible perturbations to the input texts or images[18, 19, 20, 21, 22]. Textbased attacks [23, 24, 25, 26] mislead models into generating incorrect outputs through synonym substitution, rewriting, or character-level perturbations. FGSM [27], PGD [28], AutoAttack [29], and C&W [30] are classical image-based white-box attacks that construct adversarial images by accessing model parameters and gradient information. In terms of multimodal attacks, Co-Attack [31] is a white-box attack method designed for VLMs, while more works focus on building transferable adversarial black-box attack frameworks [32, 33, 34, 35, 36, 37].

General adversarial robustness. Researchers have proposed multiple robustness strategies to enhance the reliability of models in adversarial settings [38, 39]. Detector-based approaches [40, 41] defend against adversarial attacks by detecting and filtering anomalous patterns within input samples. Purification methods [42, 43, 44] utilize techniques such as image transformations [45, 46] and denoising filters [47] to disrupt or remove potential adversarial perturbations from the input, yet they run the risk of weakening normal sample characteristics. Certified robustness approaches [48, 49, 50] provide theoretical and verifiable guarantees for model robustness, though they are typically applicable only to simple threat models with small certified radii. Adversarial training [51, 52, 53, 54] addresses model vulnerabilities by mining potential adversarial examples in the dataset and adapting the model to withstand adversarial attacks during the training process.

Adversarial robustness of VLMs. Numerous studies have explored the robustness of VLMs under adversarial attacks, mainly including defense strategies based on model fine-tuning and adversarial prompt tuning. TeCoA [5] and LAAT [6] enhance zero-shot adversarial robustness by leveraging the semantic consistency of the text encoder to guide fine-tuning of the image encoder. PMG-AFT [55] and FARE [56] leverage the generalization features of the original pre-trained model to improve the adversarial robustness of the CLIP visual encoder on downstream tasks while preserving natural generalizability. Prompt tuning serves as a lightweight adaptation approach that facilitates the efficient transfer of pretrained models toward the target task distribution [57, 15, 58, 59]. Recent studies [7, 8, 9] have shown that adversarial prompt tuning can efficiently enhance the robust adaptability of VLMs. APT [10] and AdvPT [11] approaches improve model robustness by introducing learnable textual prompts into the language branch of CLIP to align with adversarial image embeddings. Correspondingly, C-AVP [12] and TeCoA [5] incorporate learnable visual prompts to defend against adversarial attacks. Recent multimodal adversarial prompt methods [7, 13, 14, 8] enhance the consistency between visual and language features of adversarial examples under the guidance of pre-trained CLIP, thereby balancing natural generalization and robust adaptation.

# 3 Proposed Method

Although prompt learning preserves the general representations of pre-trained VLMs, the adapted prompts lead to overfitting on specific supervised tasks. We propose architectural refinements to enhance VLMs for achieving robustness in both in-distribution and OOD scenarios. Figure 1 provides an overview of our proposed approach, with further details presented in the following sections.

# 3.1 Preliminaries

CLIP recap. Let $\mathcal { V } _ { \boldsymbol { \theta } _ { v } } ( \cdot )$ and $\tau _ { \theta _ { t } } ( \cdot )$ denote the image encoder and text encoder of CLIP, respectively, where $\theta _ { v }$ and $\theta _ { t }$ represent the corresponding pre-trained weights. Given a natural image $v$ , the input sequence for the visual branch is constructed as $\tilde { v } = \{ v _ { \mathrm { c l s } } , v _ { 1 : M } \}$ , where $v _ { 1 : M }$ are the patch-level linearly projections of the image, and $v _ { \mathrm { c l s } }$ is a learnable vector aggregating global features. Given a manually designed fixed text template $t$ , the input sequence for the language branch is constructed as $\tilde { t } = \{ t _ { \mathrm { s o s } } , t _ { 1 : N } , t _ { c } , t _ { \mathrm { e o s } } \}$ , where $t _ { 1 : N }$ and $t _ { c }$ represent the word embeddings of the template text and the class label, respectively. $t _ { \mathrm { s o s } }$ and $t _ { \mathrm { e o s } }$ are non-parametric start and end tokens. The input sequences from the visual and language branches are encoded by CLIP in the latent space into image embeddings $\mathcal { V } _ { \boldsymbol { \theta } _ { v } } ( \tilde { v } )$ and text embeddings $\mathcal { T } _ { \theta _ { t } } ( \tilde { t } )$ , respectively. During zero-shot inference, the similarity between $\mathcal { V } _ { \boldsymbol { \theta } _ { v } } ( \tilde { v } )$ and the text embeddings of all candidate categories $\{ \mathcal { T } _ { \theta _ { t } } ( \tilde { t } _ { c } ) \} _ { c = 1 } ^ { C }$ is computed as exp(sim(Vθv (˜v),Tθt (t˜))/ϑ)PC exp(sim(Vθv (˜v),Tθ (t˜c))/ϑ) , where sim(·, ·) denotes the cosine similarity function, $\vartheta$ is the temperature parameter, and $C$ is the total number of classes.

Adversarial attacks against CLIP. Given a natural image $v$ with ground-truth label $y$ , adversaries construct a perceptually imperceptible adversarial example $v _ { \mathrm { a d v } } = v + \delta$ by optimizing the perturbation $\delta$ within a $q$ -norm ball of radius $\epsilon$ . A successful attack must satisfy the following criteria:

![](images/1c87de8566409b858e4d9bc3229961976739c1c02d52c8afc8f9c78d21a2caf3.jpg)  
Figure 1: An overview of CoAPT. Natural CLIP processes natural images and extended descriptive text inputs. Robust CLIP takes as input the images subjected to HF suppression via the real-time Adaptive-FGP algorithm and restores the corrupted natural generalization features under the guidance of Natural CLIP in the latent space. The outputs of Robust CLIP are collaboratively regulated by the frozen CLIP weights $\theta$ , the trainable deep multimodal adversarial prompts $\phi$ , and the low-rank residual modules $\varphi$ . The Rényi branch explicitly regulates the discrepancy between natural and adversarial distributions by calculating the divergence between their similarity scores.

$$
\arg \operatorname* { m a x } _ { c \in \{ 1 , \dots , C \} } \ \mathrm { s i m } ( \mathcal { V } _ { \theta _ { v } } ( \tilde { v } _ { \mathrm { a d v } } ) , \mathcal { T } _ { \theta _ { t } } ( \tilde { t } _ { c } ) ) \neq y , \quad \mathrm { s . t . } \quad \| v _ { \mathrm { a d v } } - v \| _ { q } \leq \epsilon .
$$

Adversarial prompt tuning. APT enhances the adversarial adaptability of pre-trained VLMs for specific or novel downstream tasks by optimizing visual or textual prompts through adversarial training. Given the prompts $\phi = \{ \phi _ { v } ^ { 1 : V } , \phi _ { t } ^ { 1 : T } \}$ to be optimized during adversarial training, where $V$ and $T$ represent the number of trainable tokens within the visual and textual prompts, respectively. Adversarial visual-only and text-only prompting [10, 11, 12] typically employs shallow prompting, where prompts are inserted solely into the input sequences. Specifically, the visual and textual input sequences are updated as $\tilde { v } = \{ v _ { \mathrm { c l s } } , \phi _ { v } ^ { 1 : V } , v _ { 1 : M } \}$ and $\tilde { t } = \{ t _ { \mathrm { s o s } } , ~ \phi _ { t } ^ { 1 : T } , ~ \dot { t } _ { c } , ~ t _ { \mathrm { e o s } } \}$ . Building upon shallow prompting, both independent and joint vision-language adversarial prompting [7, 8] incorporate deep prompts into multiple layers within the visual and language transformer architectures.

We aim to develop joint vision-language adversarial prompts that learn adversarial transformationinvariant features during training, strengthening the adversarial robustness of the CLIP visual branch. We still denote the adversarial deep prompts as $\phi$ . Given a downstream dataset $\mathcal { D }$ , $\phi$ is optimized jointly with the frozen parameters $\theta$ on adversarial examples. Focusing on the $\ell _ { \infty }$ threat model, the adversarial optimization process for obtaining the optimal parameters of robust prompts $\phi ^ { * }$ can be formalized as:

$$
\phi ^ { * } = \arg \operatorname* { m i n } _ { \phi } \mathbb { E } _ { ( v , y ) \sim \mathcal { D } } \left[ \operatorname* { m a x } _ { \| v _ { \mathrm { a d v } } - v \| _ { \infty } \leq \epsilon } \mathcal { L } ( \mathcal { V } _ { \theta _ { v } , \phi _ { v } } ( \tilde { v } _ { \mathrm { a d v } } ) , \mathcal { T } _ { \theta _ { t } , \phi _ { t } } ( \tilde { t } _ { c } ) ) \right] .
$$

# 3.2 Real-Time Total Variation Regularization for High-Frequency Suppression

Background on total variation. Total variation regularization is implemented in the continuous and discrete settings by solving an unconstrained convex optimization problem in its penalized form:

$$
\operatorname* { m i n } _ { u \in U } \frac { 1 } { 2 \lambda } \| u - v _ { \mathrm { ( a d v ) } } \| ^ { 2 } + \| u \| _ { \mathrm { T V } } ,
$$

where $u \in U = \mathbb { R } ^ { m \times n }$ denotes the image to be restored, $v _ { \mathrm { ( a d v ) } } \in U$ represents either a natural or adversarial image. For simplicity, $v$ is used uniformly in this section. $\lVert \cdot \rVert _ { \mathrm { T V } }$ represents the discrete total variation of the image gradient, and $\lambda > 0$ balances the fidelity and regularization terms. Chambolle[60] transforms Eq. (3) into a nonlinear projection problem on a constrained space via dual formulation. However, this method lacks real-time capability and is prone to over-smoothing image details and residual adversarial perturbations. We design adaptive-FGP, a fast gradient projection (FGP) method with an adaptive restart mechanism and a spatially adaptive regularization strategy.

Accelerated gradient method with adaptive restart mechanism. We obtain the optimal solution from a norm-constrained dual vector field, thereby recovering $v$ in the form:

$$
\operatorname* { m i n } _ { \mathbf { w } \in \mathcal { W } } \left\{ f ( \mathbf { w } ^ { k } ) : = \left. v - \gamma ( v ) \cdot \mathrm { d i v } ( \mathbf { w } ^ { k } ) \right. ^ { 2 } \right\} ,
$$

where denotes the current time step, and $\mathcal { W } \subseteq \mathbb { R } ^ { ( m - 1 ) \times n } \times \mathbb { R } ^ { m \times ( n - 1 ) }$ is the unit-ball constraint set for the gradient dual components $\mathbf { w } _ { i , j } ^ { k } = ( p _ { i , j } ^ { k , x } , ~ p _ { i , j } ^ { k , y } ) ^ { \top }$ . If the gradient vector is defined in both horizontal and vertical directions, it satisfies $\| \mathbf { w } _ { i , j } ^ { k } \| \leq 1$ . Otherwise, only the single-direction constraint remains, satisfying $\| p _ { i , n } ^ { k , x } \| _ { \infty } \leq 1$ and $\| p _ { m , j } ^ { k , y } \| _ { \infty } \leq 1$ . $\mathrm { d i v } ( \cdot )$ denotes the discrete divergence operator, which maps the dual variables w from the vector field $\mathcal { W }$ to the image domain $U$ . The gradient of $f ( \mathbf { w } ^ { k } )$ can be computed as $\nabla _ { \mathbf { w } ^ { k } } f ( \mathbf { w } ^ { k } ) = - 2 \cdot \gamma ( v ) \cdot \mathrm { d i v } ^ { * } \left( v - \gamma ( v ) \mathbf { \bar { \cdot } } \mathrm { d i v } ( \mathbf { w } ^ { k } ) \right)$ . Using a step size of $1 / L$ , where $L$ denotes the Lipschitz constant of $f ( \mathbf { w } ^ { k } )$ with its upper bound derived as $1 6 \gamma ^ { 2 } ( v )$ in the Appendix B. The dual variable update rule can be expressed as:

$$
\mathbf { w } ^ { k } = \Pi _ { \mathcal { W } } \left( \bar { \mathbf { w } } ^ { k } - \frac { \nabla ( v - \gamma ( v ) \cdot \mathrm { d i v } ( \bar { \mathbf { w } } ^ { k } ) ) } { 8 \cdot \gamma ( v ) } \right) ,
$$

where $\Pi _ { \mathcal { W } }$ represents the projection operator. The update of $\bar { \bf w }$ is performed as follows:

$$
\bar { \mathbf { w } } ^ { k + 1 } = \left. \begin{array} { l l } { \mathbf { w } ^ { k } + ( \tau _ { k } - 1 ) \cdot \left( \mathbf { w } ^ { k } - \mathbf { w } ^ { k - 1 } \right) / \tau _ { k + 1 } , } & { \mathrm { i f ~ } \theta ^ { k } < \theta _ { \mathrm { t h } } , } \\ { \mathbf { w } ^ { k } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

when $\theta ^ { k }$ meets the predefined threshold $\theta _ { \mathrm { t h } }$ , the Nesterov [61] time-scale variable is updated with $\tau _ { k + 1 } = \left( 1 + \sqrt { 1 + 4 \tau _ { k } ^ { 2 } } \right) / 2$ ; otherwise, it is reset to 1.0. The solution to the objective function is denoted as $u ^ { k } = v - \gamma ( v ) \cdot { \mathrm { d i v } } ( { \bar { \mathbf { w } } } ^ { k - 1 } )$ . The solution increments at two consecutive time steps are defined as $\sigma _ { k } = u ^ { k } - u ^ { k - 1 }$ and $\sigma _ { k - 1 } = u ^ { k - 1 } - u ^ { k - 2 }$ . Whether the current momentum accumulation benefits the variable update is determined utilizing a cosine similarity-based adaptive restart criterion:

$$
\cos ( \theta _ { k } ) = { \frac { \langle \sigma _ { k } , \sigma _ { k - 1 } \rangle } { \left\| \sigma _ { k } \right\| \cdot \left\| \sigma _ { k - 1 } \right\| + \zeta } } ,
$$

where $\zeta$ is a numerical stabilization term. When the angle between directions exceeds $9 0 °$ , it signals a sharp deviation or reversal between momentum and update, indicating trajectory discontinuity. We then reset the temporal scaling and disable momentum to avoid overshooting.

Spatially adaptive regularization strategy. The regularization map $\gamma ( v ) \in \mathbb { R } _ { + } ^ { m \times n }$ is given by:

$$
\gamma ( v ) = \mu _ { \mathrm { b a s e } } \cdot ( 1 + \mu _ { \mathrm { g a i n } } \cdot \Phi ( v ) ) ,
$$

where $\mu _ { \mathrm { b a s e } }$ ${ \bf \Psi } _ { \mathrm { { e } } } , \mu _ { \mathrm { { g a i n } } } \in \mathbb { R } ^ { + }$ represent the base regularization strength and the sensitivity of the adjustment factor, respectively. The edge magnitude response function $\Phi ( v ) \in \mathbb { R } _ { + } ^ { m \times n }$ is estimated using Sobel convolution kernels as $\sqrt { ( v * K _ { x } ) ^ { 2 } + ( v * K _ { y } ) ^ { 2 } }$ , where $K _ { x }$ and $K _ { y }$ denote the horizontal and vertical Sobel operators respectively. This adaptive regularization strategy automatically reduces the regularization strength in edge regions while enhancing it in flat regions, thereby preserving structural image details and effectively suppressing adversarial perturbations.

Convergence criterion. The relative change in update is measured through the Frobenius norm:

$$
\operatorname* { m a x } _ { i \in \{ k , k - 1 , . . . , k - s \} } \frac { \| \sigma _ { i } \| _ { F } } { \| u ^ { i } \| _ { F } + \zeta } < \xi .
$$

If the convergence tolerance threshold $\xi > 0$ is satisfied for $s$ consecutive iterations, the projection optimization problem is considered to have converged. Based on the optimal solution $\mathbf { w } ^ { \hat { k } \star } ( v )$ , the optimal image estimate for the original problem can be recovered as $\rho ( v ) = v - \gamma ( v ) \cdot \mathrm { d i v } ( \mathbf { w } ^ { k \star } ( v ) )$ .

# 3.3 Natural-Latent-Guided Adversarial Prompt Learning

Reconstruction of natural generalization representations. CoAPT employs deep contextual multimodal prompts and refines visual prompts through linear projection onto language prompts to foster synergy between visual-language prompts. As illustrated in Figure 1, we efficiently learn generalizable knowledge from the natural CLIP by aligning its clean vision-language embeddings with adversarial embeddings from the robust CLIP in the latent space. Notably, Vanilla CLIP employs fixed text templates, which limit its ability to capture the semantic diversity required for generalization effectively during fine-tuning. A Gaussian radial basis function (RBF) is used to measure the embedding similarity between the natural CLIP and the robust CLIP in the latent space. Compared to cosine similarity, which primarily captures angular differences of vectors, Gaussian RBF highlights feature shifts caused by small-scale perturbations, allowing more sensitive detection of subtle distributional changes. In particular, we align both the visual and language branches:

$$
\mathcal { L } _ { \mathrm { r e c o n } } = 2 - \exp \left( - \beta \left( \| \gamma _ { \theta _ { v } , \phi _ { v } , \varphi _ { v } } ( \rho ( \tilde { v } _ { \mathrm { a d v } } ) ) - \gamma _ { \theta _ { v } } ( \tilde { v } ) \| _ { 2 } ^ { 2 } + \| \mathcal { T } _ { \theta _ { t } , \phi _ { t } , \varphi _ { t } } ( \tilde { t } ) - \mathcal { T } _ { \theta _ { t } } ( \tilde { t } ) \| _ { 2 } ^ { 2 } \right) \right) ,
$$

where the parameter $\beta = ( 2 \sigma ^ { 2 } ) ^ { - 1 }$ controls the sensitivity of distance variation to similarity. $\varphi _ { v }$ and $\varphi _ { t }$ are the low-rank residual modules introduced next. The learnable prompts in both the language and visual branches can adapt the data distribution of Vanilla CLIP to that of specific downstream adversarial tasks, while preserving and enhancing generalization and robustness to OOD tasks.

Low-rank residual module. Directly imposing consistency constraints in the latent space is equivalent to introducing a strong supervisory signal, which lacks the flexibility to adapt to taskspecific requirements and interpretable deviations. Inspired by LoRA [62], we introduce two low-rank matrices as an intermediate learnable bottleneck structure. This design allows the model to preserve the backbone features while selectively capturing fine-grained task-specific shifts within a compact subspace. Specifically, we incorporate an additional update term through low-rank reparameterization:

$$
\mathcal { V } _ { \theta , \phi , \varphi } = ( I + \eta \cdot B A ) \mathcal { V } _ { \theta , \phi } ,
$$

where $\eta$ is the scaling factor, $B \in \mathbb { R } ^ { d \times r }$ , $A \in \mathbb { R } ^ { r \times d }$ , and $r \ll d$ . The initial parameter perturbation is controlled by initializing the matrices as $A \sim { \mathcal { N } } ( 0 , 1 / r )$ and $B \sim \delta ( 0 )$ .

Rényi regularization. Let $P$ and $Q$ denote the predicted probability distributions of natural and adversarial samples in the vision-language space of robust CLIP, respectively. Since adversarial samples are derived from minor perturbations of natural samples, $P$ is considered absolutely continuous with respect to $Q$ . We introduce a regularization loss based on the $\alpha$ -order Rényi divergence [63] to reduce the discrepancy between the natural and adversarial predictive distributions in robust CLIP:

$$
\mathcal { L } _ { \mathrm { r \ ' e n y i } } = \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { P } \left[ \left( \frac { d P } { d Q } \right) ^ { \alpha - 1 } \right] , \alpha \in [ 0 , \infty ) ,
$$

where $\textstyle { \frac { d P } { d Q } }$ is the Radon–Nikodym derivative of $P$ with respect to $Q$ . $\alpha$ explicitly controls the sensitivity to distributional differences. Higher orders $( \alpha > 1 )$ ) enhance the ability of the model to suppress spurious correlations. This mechanism corrects potential discriminative boundary ambiguities and reduces overfitting risks by preserving task-beneficial generalized features. Correspondingly, the supervised loss for downstream classification tasks can be expressed with the Rényi cross-entropy [64]:

$$
{ \mathcal { L } } _ { \mathrm { r c e } } = { \frac { \alpha } { 1 - \alpha } } \log \sum _ { i } P ( i ) \cdot Q ( i ) ^ { \frac { \alpha - 1 } { \alpha } } , \quad \alpha \in [ 0 , \infty ) .
$$

Note that the Rényi cross entropy degenerates into Shannon cross entropy when the dataset labels are represented in one-hot coding. The overall training objective of CoAPT can be expressed as follows:

$$
\mathcal { L } _ { \mathrm { c o a p t } } = \kappa _ { 1 } \mathcal { L } _ { \mathrm { r e c o n } } + \kappa _ { 2 } \mathcal { L } _ { \mathrm { r e n y i } } + \kappa _ { 3 } \mathcal { L } _ { \mathrm { r e e } } ,
$$

$\kappa _ { 1 } , \kappa _ { 2 } , \kappa _ { 3 }$ are hyperparameters weighting contributions of individual losses to the overall objective.

Overview of proposed method. Algorithm 1 illustrates the adversarial prompt optimization procedure adopted by CoAPT. In each training iteration, a batch of image-label pairs $( v , y )$ is sampled from the downstream dataset $\mathcal { D }$ . Subsequently, the visual and textual sequences are constructed and accompanied by trainable deep prompts. Perceptually invisible adversarial examples $v _ { \mathrm { a d v } }$ are crafted under $\ell _ { \infty }$ norm constraints to induce erroneous model predictions (Lines $2 { \sim } 4$ ). These sequences are then fed into the natural CLIP and the robust CLIP equipped with low-rank residual modules $\varphi _ { v }$ and $\varphi _ { t }$ to obtain the corresponding visual and language representations (Lines $5 { \sim } 9$ ). CoAPT integrates

Input: Dataset $\mathcal { D }$ , frozen CLIP encoders $\nu _ { \theta _ { v } } , \tau _ { \theta _ { t } }$ , prompt parameters $\phi = \{ \phi _ { v } , \phi _ { t } \}$ , low-rank modules $\varphi = \{ \varphi _ { v } , \varphi _ { t } \}$ , loss weights $\kappa _ { 1 } , \kappa _ { 2 } , \kappa _ { 3 }$ , adversarial budget $\epsilon$

Output: Optimized robust prompts $\phi ^ { \star }$   
1: for each minibatch $( v , y ) \sim \mathcal { D }$ do   
2: Set the real-time total variation regularization parameters   
3: Construct input sequences $\tilde { v } , \tilde { t }$ and deep prompts $\phi$   
4: Generate adversarial example $v _ { \mathrm { a d v } }$ under $\ell _ { \infty }$ constraint: $\| v _ { \mathrm { a d v } } - v \| _ { \infty } \leq \epsilon$   
5: Generate visual and textual representations for natural CLIP and robust CLIP:   
6: $\begin{array} { r l } & { \mathcal { V } _ { \mathrm { n a t } }  \mathcal { V } _ { \theta _ { v } } ( \tilde { v } ) } \\ & { \mathcal { T } _ { \mathrm { n a t } }  \mathcal { T } _ { \theta _ { t } } ( \tilde { t } ) } \\ & { \mathcal { V } _ { \mathrm { a d v } }  \mathcal { V } _ { \theta _ { v } , \phi _ { v } , \varphi _ { v } } ( \rho ( \tilde { v } _ { \mathrm { a d v } } ) ) } \\ & { \mathcal { T } _ { \mathrm { a d v } }  \mathcal { T } _ { \theta _ { t } , \phi _ { t } , \varphi _ { t } } ( \tilde { t } ) } \end{array}$   
7:   
8:   
9:   
10: Compute reconstruction loss $\mathcal { L } _ { \mathrm { r e c o n } }  2 - \exp ( - \beta ( \| \mathcal { V } _ { \mathrm { a d v } } - \mathcal { V } _ { \mathrm { n a t } } \| _ { 2 } ^ { 2 } + \| \mathcal { T } _ { \mathrm { a d v } } - \mathcal { T } _ { \mathrm { n a t } } \| _ { 2 } ^ { 2 } ) )$   
11: Compute visual-textual representation similarity $P = s c a l e \cdot \mathcal { V } _ { \mathrm { n a t } } \cdot \mathcal { T } _ { \mathrm { n a t } } ^ { \top }$ , $Q = s c a l e \cdot \mathcal { V } _ { \mathrm { a d v } } \cdot \mathcal { T } _ { \mathrm { a d v } } ^ { \top }$   
12: Compute Rényi divergence loss $\begin{array} { r } { \mathcal { L } _ { \mathrm { r e n y i } } \gets \frac { 1 } { \alpha - 1 } \log \mathbb { E } _ { P } [ ( \frac { d P } { d Q } ) ^ { \alpha - 1 } ] } \end{array}$   
13: Compute Rényi cross-entropy loss: $\begin{array} { r } { \mathcal { L } _ { \mathrm { r c e } }  \frac { \alpha } { 1 - \alpha } \log \sum _ { i } P ( i ) \cdot Q ( i ) ^ { \frac { \alpha - 1 } { \alpha } } } \end{array}$   
14: Take gradient step on $\nabla _ { \phi , \varphi } ( \kappa _ { 1 } \mathcal { L } _ { \mathrm { r e c o n } } + \kappa _ { 2 } \mathcal { L } _ { \mathrm { r e n y i } } + \kappa _ { 3 } \mathcal { L } _ { \mathrm { r e e } } )$   
15: ϕ, $\varphi \gets$ Backward $( \nabla _ { \phi , \varphi } )$   
16: end for

three losses, including a reconstruction loss for recovering generalization, a Rényi divergence loss to quantify prediction discrepancies between natural and adversarial samples, and a cross-entropy loss for classification (Lines $1 0 { \sim } 1 3$ ). Finally, only the prompt parameters $\phi$ and the low-rank module parameters $\varphi$ are updated via gradient descent. Adversarial prompt learning significantly improves the robust generalization of the model under image perturbations and distributional shifts, and exhibits strong cross-task transferability (Lines $1 4 \sim 1 5$ ).

# 4 Experiments

# 4.1 Evaluation Settings

Datasets and benchmark settings. We conduct a comprehensive evaluation of the proposed CoAPT method across four benchmark settings on 15 datasets spanning diverse vision tasks. For the evaluation of few-shot learning, base-to-novel class generalization, and zero-shot benchmarks, we adopt 11 image classification datasets, including EuroSAT [65] for satellite imagery, UCF101 [66] for action recognition, DTD [67] for texture classification, SUN397 [68] for scene recognition, Caltech101 [69] and ImageNet [70] for general object recognition, and FGVC Aircraft [71], Flowers102 [72], OxfordPets [73], Food101 [74], and StanfordCars [75] for fine-grained classification tasks. For the OOD benchmark, we select four variants of ImageNet, ImageNet-A [76], ImageNet-R [77], ImageNet-Sketch [78], and ImageNetV2 [79], as the domain generalization test sets. Notably, both zero-shot and OOD utilize the training set of ImageNet as the source dataset.

Adversarial training and evaluation. The attack settings of baseline methods TeCoA [5] and FAP [7] are adopted to ensure fair comparison. During adversarial training, we adopt a two-step PGD attack with a maximum perturbation magnitude $\bar { \ell } _ { \infty } = 1 / 2 5 5$ and step size $\alpha = 1 / 2 5 5$ . For robustness evaluation, we employ a 100-step PGD attack under the same constraints to thoroughly assess the defense capability of the model under strong attacks.

Implementation details. Our method is built upon the ViT-B/32 architecture of Vanilla CLIP. Each experiment is conducted three times with different random seeds, and the average results are reported. The convergence tolerance threshold in Adaptive-FGP is set to $\xi = 1 e ^ { - 3 }$ , $s = 3$ , and the maximum number of iterations is 30. The parameters of the regularization factor map $\gamma ( v )$ are set to $\mu _ { \mathrm { b a s e } } = 0 . 1$ and $\mu _ { \mathrm { g a i n } } = 1 . 2$ . We employed 2.5-order Rényi divergence regularization, with $\mathcal { L } _ { \mathrm { c o a p t } }$ coefficients set to $\kappa _ { 1 } \stackrel { \cdot } { = } 8 , \kappa _ { 2 } = 1 , \kappa _ { 3 } = 1 .$ . Adversarial prompts with a length of 4 and a depth of 9 are applied to both the visual and textual branches. The RAdam optimizer with an initial learning rate of 0.00735 is adopted, and the batch size is set to 64. In contrast to the existing research work, we do not set proprietary hyperparameters for any of the benchmarks and datasets, in order to prove the generality of the proposed CoAPT. Under few-shot settings we compare with FAP and baselines from its paper.

![](images/4832b12f6cd5b04cea82f0fe42479e873cf921529be283a7892e60d5e64ec37c.jpg)  
Figure 2: The few-shot performance across 11 benchmark datasets under varying numbers of shots.

# 4.2 Adversarial Few-Shot Learning

The robust generalization capability of each model to specific tasks is evaluated under the condition of only a few identically distributed samples. As shown in Figure 2, CoAPT demonstrates consistently superior performance compared to all baseline methods. CoAPT exhibits robust learning ability with near-linear steady improvement in natural and adversarial accuracy as the number of shots increases. In contrast, the baseline methods show significant performance fluctuations across different shot counts. Furthermore, our approach achieves superior control over the trade-off between natural accuracy and adversarial robustness. In most of the datasets, CoAPT is able to match the natural accuracy of Vanilla CLIP with only 1-shot learning. On six datasets, including Caltech101, our robust accuracy is even higher than the natural accuracy of the baseline method. The robust accuracy of CoAPT on five datasets, including DTD, can be improved to higher than the natural accuracy of Vanilla CLIP by few-shot learning.

# 4.3 Adversarial Base-to-New Generalization

We assess the ability of the models to balance robust adaptation to specific class distributions and robust generalization to unseen class distributions. Specifically, the models are trained on base classes with a 16-shot setting and jointly evaluated on the base classes and the novel unseen classes. As shown in Table 1, our method outperforms state-of-the-art approaches on all datasets. While improving the average harmonic mean (HM) of robustness by $3 2 . 3 9 \%$ , the natural generalization performance of the model also achieves an average gain of $1 3 . 0 9 \%$ . Notably, the harmonic mean of robustness for novel classes reaches a maximum of $5 1 . 5 7 \%$ on the OxfordPets dataset. These results demonstrate that the robust prompts learned by CoAPT not only adapt to category-specific distributional shifts and distributional discrepancies between natural and adversarial examples but also effectively preserve the natural generalization capability of the original pretrained model.

Table 1: Comparison with state-of-the-art methods on base-to-novel generalization. Gain denotes the absolute performance improvement.   

<table><tr><td></td><td colspan="5">(a) Average</td><td colspan="5">(b) ImageNet</td><td colspan="5">(c) Caltech101</td><td colspan="5">(d)DTD</td></tr><tr><td></td><td>Acc.</td><td>FAP</td><td>CoAPT|</td><td>Gain↑</td><td></td><td>Acc.</td><td>FAP</td><td>CoAPT|</td><td>|Gain↑</td><td></td><td>Acc.</td><td>FAP</td><td>CoAPT|</td><td>|Gain↑</td><td></td><td>Acc.</td><td>FAP</td><td>CoAPT</td><td>|Gain↑</td></tr><tr><td rowspan="2"></td><td>Base</td><td>70.52</td><td>78.47</td><td>7.95</td><td></td><td>Base</td><td>58.10</td><td>66.15</td><td>8.05</td><td></td><td>Base</td><td>94.07</td><td>97.25</td><td>3.18</td><td></td><td>Base</td><td>69.17</td><td>76.08</td><td>6.91</td></tr><tr><td>Novel</td><td>49.58</td><td>65.35</td><td>15.77</td><td>M</td><td>Novel</td><td>47.83</td><td>55.41</td><td>7.58</td><td>M</td><td>Novel</td><td>76.53</td><td>92.72</td><td>16.19</td><td>N</td><td>Novel</td><td>35.17</td><td>54.03</td><td>18.86</td></tr><tr><td></td><td>HM</td><td>58.22</td><td>71.31</td><td>13.09</td><td></td><td>HM</td><td>52.47</td><td>60.30</td><td>7.84</td><td></td><td>HM</td><td>84.40</td><td>94.93</td><td>10.53</td><td></td><td>HM</td><td>46.63</td><td>63.17</td><td>16.54</td></tr><tr><td></td><td>Base</td><td>38.05</td><td>67.70</td><td>29.65</td><td></td><td>Base</td><td>25.83</td><td>52.65</td><td>26.82</td><td></td><td>Base</td><td>74.20</td><td>94.38</td><td>20.18</td><td></td><td>Base</td><td>41.63</td><td>67.98</td><td>26.35</td></tr><tr><td>3</td><td>Novel</td><td>21.86</td><td>54.13</td><td>32.27</td><td>3</td><td>Novel</td><td>21.57</td><td>45.07</td><td>23.50</td><td>3</td><td>Novel</td><td>50.00</td><td>88.03</td><td>38.03</td><td>R</td><td>Novel</td><td>19.77</td><td>43.88</td><td>24.11</td></tr><tr><td></td><td>HM</td><td>27.77</td><td>60.16</td><td>32.39</td><td></td><td>HM</td><td>23.51</td><td>48.57</td><td>25.06</td><td></td><td>HM</td><td>59.74</td><td>91.09</td><td>31.35</td><td></td><td>HM</td><td>26.81</td><td>53.31</td><td>26.50</td></tr><tr><td></td><td colspan="5">(e)EuroSAT</td><td colspan="5">(f)FGVCAircraft</td><td colspan="5">(g)Food101</td><td colspan="5">(h)Flowers102</td></tr><tr><td></td><td>Acc.</td><td>FAP</td><td>CoAPT</td><td>Gain↑</td><td></td><td>Acc.</td><td>FAP</td><td>CoAPT</td><td>Gain↑</td><td></td><td>Acc.</td><td>FAP</td><td>CoAPT</td><td>|Gain↑</td><td></td><td>Acc.</td><td>FAP</td><td>CoAPT</td><td>Gain↑</td></tr><tr><td></td><td>Base</td><td>87.70</td><td>91.61</td><td>3.91</td><td></td><td>Base</td><td>24.83</td><td>35.37</td><td>10.54</td><td></td><td>Base</td><td>72.37</td><td>78.20</td><td>5.83</td><td></td><td>Base</td><td>89.30</td><td>94.94</td><td>5.64</td></tr><tr><td></td><td>Novel</td><td>32.80</td><td>56.11</td><td>23.31</td><td>2</td><td>Novel</td><td>15.83</td><td>25.41</td><td>9.58</td><td>M</td><td>Novel</td><td>68.20</td><td>79.47</td><td>11.27</td><td>2</td><td>Novel</td><td>45.67</td><td>63.07</td><td>17.40</td></tr><tr><td></td><td>HM</td><td>47.74</td><td>69.33</td><td>21.59</td><td></td><td>HM</td><td>19.33</td><td>29.58</td><td>10.24</td><td></td><td>HM</td><td>70.22</td><td>78.83</td><td>8.60</td><td></td><td>HM</td><td>60.43</td><td>75.79</td><td>15.36</td></tr><tr><td></td><td>Base</td><td>51.80</td><td>84.67</td><td>32.87</td><td></td><td>Base</td><td>8.00</td><td>25.37</td><td>17.37</td><td></td><td>Base</td><td>27.57</td><td>62.03</td><td>34.46</td><td></td><td>Base</td><td>65.50</td><td>88.57</td><td>23.07</td></tr><tr><td></td><td>Novel</td><td>13.40</td><td>47.40</td><td>34.00</td><td>R</td><td>Novel</td><td>4.23</td><td>16.68</td><td>12.45</td><td>R</td><td>Novel</td><td>24.20</td><td>62.86</td><td>38.66</td><td>3</td><td>Novel</td><td>18.10</td><td>51.89</td><td>33.79</td></tr><tr><td></td><td>HM</td><td>21.29</td><td>60.55</td><td>39.25</td><td></td><td>HM</td><td>5.53</td><td>20.12</td><td>14.59</td><td></td><td>HM</td><td>25.78</td><td>62.44</td><td>36.66</td><td></td><td>HM</td><td>28.36</td><td>65.42</td><td>37.06</td></tr><tr><td colspan="5">(i) OxfordPets</td><td colspan="5">(j) StanfordCars</td><td colspan="5">(k) SUN397</td><td colspan="5">(l UCF101</td></tr><tr><td rowspan="2"></td><td>Acc.</td><td>FAP</td><td>CoAPT|</td><td>|Gain↑</td><td>Acc.</td><td></td><td>FAP</td><td>CoAPT|</td><td>|Gain↑</td><td></td><td>Acc.</td><td>FAP CoAPT</td><td>|Gain↑</td><td></td><td>Acc.</td><td>FAP</td><td>CoAPT</td><td></td><td>|Gain↑</td></tr><tr><td>Base</td><td>87.37</td><td>90.55</td><td>3.18</td><td></td><td>Base</td><td>53.97</td><td>73.34</td><td>19.37</td><td></td><td>Base</td><td>68.47</td><td>76.69</td><td>8.22</td><td></td><td>Base</td><td>70.37</td><td>82.95</td><td>12.58</td></tr><tr><td>M</td><td>Novel</td><td>72.13</td><td>94.50</td><td>22.37</td><td>M</td><td>Novel</td><td>42.67</td><td>59.20</td><td>16.53</td><td>M</td><td>Novel</td><td>61.47</td><td>70.46</td><td>8.99</td><td>Novel</td><td>47.10</td><td></td><td>68.45</td><td>21.35</td></tr><tr><td></td><td>HM</td><td>79.02</td><td>92.48</td><td>13.46</td><td></td><td>HM</td><td>47.66</td><td>65.51</td><td>17.85</td><td></td><td>HM</td><td>64.78</td><td>73.44</td><td>8.66</td><td></td><td>HM</td><td>56.43</td><td>75.00</td><td>18.57</td></tr><tr><td rowspan="2">3</td><td>Base</td><td>34.13</td><td>78.72</td><td>44.59</td><td></td><td>Base</td><td>18.60</td><td>54.20</td><td>35.60</td><td></td><td>Base</td><td>34.63</td><td>64.50</td><td>29.87</td><td></td><td>Base</td><td>36.63</td><td>71.65</td><td>35.02</td></tr><tr><td>Novel HM 29.56</td><td>26.07</td><td>83.71 81.13</td><td>57.64 51.57</td><td></td><td>Novel HM</td><td>14.10</td><td>40.95</td><td>26.85 30.61</td><td>B</td><td>Novel</td><td>30.77 32.59</td><td>58.50 61.35</td><td>27.73 28.76</td><td>3</td><td>Novel 18.30</td></table>

# 4.4 Zero-Shot Performance

The generalization ability of the models across datasets is explored. CoAPT is trained on ImageNet as the source dataset and then evaluated on ten different types of downstream target datasets. The evaluation for each dataset and the corresponding statistical results are presented in Figure 3 and Table 2, respectively. Compared to the FAP method, our approach achieves significant

Table 2: CoAPT performance on source dataset and average results across 10 target datasets.   

<table><tr><td rowspan="3">Method</td><td colspan="2">ImageNet</td><td colspan="2">Average</td></tr><tr><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td></tr><tr><td>CLIP</td><td>62.10</td><td>1.57</td><td>61.89</td><td>4.53</td></tr><tr><td>FAP</td><td>50.80</td><td>21.60</td><td>45.72</td><td>23.89</td></tr><tr><td>CoAPT</td><td>63.421.32↑</td><td>51.1829.58↑</td><td>54.067.83↓</td><td>43.9020.01↑</td></tr></table>

improvements across all metrics on all datasets, particularly in adversarial robustness. We attain a better trade-off between natural and adversarial generalization. Relative to Vanilla CLIP, we sacrifice only $7 . 8 3 \%$ in natural generalization accuracy while achieving absolute gains of $4 9 . 6 1 \%$ and $3 9 . 3 7 \%$ in robustness on the source and target datasets.

![](images/ae8b35c8003f0fad5997e8aa16056dafd7496bee78f66949ba7a61a65ee83a75.jpg)  
Figure 3: Zero-shot robust and natural accuracies on the source and 10 target datasets.

# 4.5 Out-of-Distribution Performance

We test the natural generalization and adversarial robustness of the model under domain distribution shift. While maintaining ImageNet as the source dataset, we conduct direct evaluations on four representative variant datasets that share the same set of categories. As shown in Table 3, our method achieves superior natural generalization and robust adaptation across all target datasets compared to the comparison methods.

Table 3: Comparison of OOD generalization performance.   

<table><tr><td rowspan="2">Method|</td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-Sketch</td><td colspan="2">ImageNet-V2</td><td colspan="2">Average</td></tr><tr><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td></tr><tr><td>FAP CoAPT</td><td>9.40 16.997.59↑</td><td>1.20 9.728.52↑</td><td>51.60 60.358.75↑</td><td>28.20 50.7122.51个</td><td>28.40 35.767.36↑</td><td>16.30 29.1312.83↑</td><td>42.80 54.3511.55个</td><td>16.60 42.2325.63↑</td><td>33.05 41.868.81↑</td><td>15.575 32.9517.37个</td></tr></table>

# 4.6 Ablation Analysis

As shown in Table 4, we progressively ablate CoAPT components to evaluate their generalizability and importance across the four benchmarks. CoAPT with all components achieves the best performance on all benchmarks. We first remove the adaptive restart mechanism. Most metrics exhibited varying degrees of degradation, with 16-shot and OOD robust accuracy declining by $1 . 8 5 \%$ and $1 . 6 9 \%$ , respectively. This mechanism restores optimal convergence without prior knowledge of function parameters and enhances stability near the optimum. We replace the spatially adaptive regularization strategy with a fixed global regularization factor. The ablated model ignores the diversity of image spatial structures, leading to structural blurring and loss of details, with an average drop of $6 . 0 0 \%$ in clean accuracy across the four benchmarks. We subsequently remove the entire adaptive-FGP method, thereby eliminating the adversarial space compression. During high-level feature recovery in the natural CLIP latent space, the model places greater emphasis on high-frequency components where adversarial perturbations are concentrated, resulting in a degradation in adversarial robustness. However, even with full natural images, the ablated model yields lower natural accuracy than full CoAPT across all benchmarks. Removing the low-rank residual module leads to drops in few-shot-16 robustness and base-to-novel accuracy. As it is sensitive to dataset-specific hyperparameters and was not fine-tuned, its effectiveness is limited. However, due to its potential on certain datasets, the module is retained. When we remove Rényi regularization, the overall performance of the model decreases. Rényi regularization facilitates early detection and correction of boundary ambiguities, and mitigates overfitting by preserving task-relevant generalizable features. CoAPT reduces to a TeCoA-like approach when the final reconstruction loss is removed. The performance drop on unseen tasks is due to the reconstruction loss guiding prompts toward task-irrelevant generalization.

Table 4: Ablation study of CoAPT components on 15 datasets across four benchmarks.   

<table><tr><td rowspan="2">Ablation term</td><td colspan="2">Few-shot-16</td><td colspan="6">Base-to-novel</td><td colspan="2">Zero-shot</td><td colspan="2">OOD</td></tr><tr><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td><td>HM</td><td>Nat.</td><td>Rob.</td><td>HM</td><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td></tr><tr><td>No ablation</td><td>74.96</td><td>62.98</td><td>78.47</td><td>67.70</td><td>72.69</td><td>65.35</td><td>54.13</td><td>59.21</td><td>54.91</td><td>44.57</td><td>41.86</td><td>32.95</td></tr><tr><td>Adp. rst.</td><td>74.82</td><td>61.13</td><td>78.31</td><td>66.73</td><td>72.06</td><td>65.56</td><td>53.40</td><td>58.86</td><td>54.38</td><td>43.06</td><td>40.91</td><td>31.26</td></tr><tr><td>Adp.reg.</td><td>68.86</td><td>63.34</td><td>73.33</td><td>68.03</td><td>70.58</td><td>57.74</td><td>52.87</td><td>55.20</td><td>49.16</td><td>44.15</td><td>34.86</td><td>31.70</td></tr><tr><td>Adp.FGP</td><td>74.34</td><td>31.64</td><td>78.15</td><td>35.92</td><td>49.21</td><td>64.36</td><td>24.41</td><td>35.40</td><td>53.00</td><td>18.65</td><td>38.52</td><td>13.21</td></tr><tr><td>Res.mod.</td><td>74.64</td><td>31.41</td><td>78.33</td><td>35.99</td><td>49.32</td><td>64.18</td><td>26.00</td><td>37.01</td><td>55.07</td><td>19.74</td><td>41.04</td><td>14.31</td></tr><tr><td>Renyi</td><td>73.09</td><td>30.73</td><td>78.18</td><td>33.66</td><td>47.06</td><td>63.57</td><td>24.97</td><td>35.86</td><td>55.30</td><td>19.84</td><td>41.05</td><td>14.55</td></tr><tr><td>Recon.loss</td><td>71.82</td><td>31.47</td><td>76.66</td><td>34.52</td><td>47.60</td><td>58.71</td><td>22.25</td><td>32.27</td><td>51.85</td><td>19.85</td><td>38.66</td><td>13.91</td></tr></table>

# 5 Conclusion

We focus on the adversarial robustness of VLMs and propose a novel adversarial prompt tuning paradigm in which pre-trained VLMs collaborate with target robust VLMs. CoAPT begins with a proposed real-time adaptive TV regularization algorithm to attenuate high-frequency details of the input images to compress the perturbation space of the adversarial samples. Subsequently, under the guidance of natural CLIP, CoAPT restores the natural generalization features disrupted by adversarial perturbations in the latent representation space. CoAPT achieves an effective trade-off among natural generalization, adversarial robustness, and task-specific adaptation. The overall performance of CoAPT significantly surpasses that of current state-of-the-art methods on 15 datasets across the benchmarks of few-shot, base-to-novel, zero-shot, and out-of-distribution generalization.

References   
[1] Radford, A., J. W. Kim, C. Hallacy, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), pages 8748– 8763. 2021.   
[2] Li, J., R. Selvaraju, A. Gotmare, et al. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705, 2021.   
[3] Fang, Z., R. Wang, T. Huang, et al. Strong transferable adversarial attacks via ensembled asymptotically normal distribution learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24841–24850. 2024.   
[4] Wu, H., G. Ou, W. Wu, et al. Improving transferable targeted adversarial attacks with model self-enhancement. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24615–24624. 2024.   
[5] Mao, C., S. Geng, J. Yang, et al. Understanding zero-shot adversarial robustness for large-scale models. In International Conference on Learning Representations (ICLR). OpenReview.net, 2023.   
[6] Li, X., W. Zhang, Y. Liu, et al. Language-driven anchors for zero-shot adversarial robustness. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24686– 24695. 2024.   
[7] Zhou, Y., X. Xia, Z. Lin, et al. Few-shot adversarial prompt learning on vision-language models. Advances in Neural Information Processing Systems, 37:3122–3156, 2024.   
[8] Wang, X., K. Chen, J. Zhang, et al. Tapt: Test-time adversarial prompt tuning for robust inference in vision-language models. arXiv preprint arXiv:2411.13136, 2024.   
[9] Huang, Q., X. Dong, D. Chen, et al. Improving adversarial robustness of masked autoencoders via test-time frequency-domain prompting. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 1600–1610. 2023.   
[10] Li, L., H. Guan, J. Qiu, et al. One prompt word is enough to boost adversarial robustness for pre-trained vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24408–24419. 2024.   
[11] Zhang, J., X. Ma, X. Wang, et al. Adversarial prompt tuning for vision-language models. In European Conference on Computer Vision (ECCV), vol. 15103, pages 56–72. 2024.   
[12] Chen, A., P. Lorenz, Y. Yao, et al. Visual prompting for adversarial robustness. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.   
[13] Luo, L., X. Wang, B. Zi, et al. Adversarial prompt distillation for vision-language models. arXiv preprint arXiv:2411.15244, 2024.   
[14] Yang, F., M. Xia, S. Xia, et al. Revisiting the robust generalization of adversarial prompt tuning. arXiv preprint arXiv:2405.11154, 2024.   
[15] Khattak, M. U., S. T. Wasim, M. Naseer, et al. Self-regulating prompts: Foundational model adaptation without forgetting. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 15190–15200. 2023.   
[16] He, K., X. Chen, S. Xie, et al. Masked autoencoders are scalable vision learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16000–16009. 2022.   
[17] Lee, J., E. J. Hwang, S. Cho, et al. Pilamim: Toward richer visual representations by integrating pixel and latent masked image modeling. arXiv preprint arXiv:2501.03005, 2025.   
[18] Szegedy, C., W. Zaremba, I. Sutskever, et al. Intriguing properties of neural networks. In International Conference on Learning Representations (ICLR). 2014.   
[19] Moosavi-Dezfooli, S.-M., A. Fawzi, P. Frossard. Deepfool: a simple and accurate method to fool deep neural networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2574–2582. 2016.   
[20] Goodfellow, I. J., J. Shlens, C. Szegedy. Explaining and harnessing adversarial examples. In International Conference on Learning Representations (ICLR). 2015.   
[21] Jia, X., Y. Zhang, B. Wu, et al. Las-at: adversarial training with learnable attack strategy. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13398– 13408. 2022.   
[22] Hsiung, L., Y.-Y. Tsai, P.-Y. Chen, et al. Towards compositional adversarial robustness: Generalizing adversarial training to composite semantic perturbations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24658–24667. 2023.   
[23] Jin, D., Z. Jin, J. T. Zhou, et al. Is bert really robust? a strong baseline for natural language attack on text classification and entailment. In AAAI Conference on Artificial Intelligence, vol. 34, pages 8018–8025. 2020.   
[24] Gao, J., J. Lanchantin, M. L. Soffa, et al. Black-box generation of adversarial text sequences to evade deep learning classifiers. In IEEE Security and Privacy Workshops (SPW), pages 50–56. IEEE, 2018.   
[25] Ren, S., Y. Deng, K. He, et al. Generating natural language adversarial examples through probability weighted word saliency. In Association for Computational Linguistics (ACL), pages 1085–1097. 2019.   
[26] Li, L., R. Ma, Q. Guo, et al. BERT-ATTACK: adversarial attack against BERT using BERT. In Empirical Methods in Natural Language Processing (EMNLP), pages 6193–6202. 2020.   
[27] Goodfellow, I. J., J. Shlens, C. Szegedy. Explaining and harnessing adversarial examples. In Y. Bengio, Y. LeCun, eds., International Conference on Learning Representations (ICLR). 2015.   
[28] Madry, A., A. Makelov, L. Schmidt, et al. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations (ICLR). OpenReview.net, 2018.   
[29] Croce, F., M. Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International Conference on Machine Learning (ICML), pages 2206–2216. PMLR, 2020.   
[30] Carlini, N., D. Wagner. Towards evaluating the robustness of neural networks. In IEEE Symposium on Security and Privacy (SP), pages 39–57. Ieee, 2017.   
[31] Zhang, J., Q. Yi, J. Sang. Towards adversarial attack on vision-language pre-training models. In ACM International Conference on Multimedia, pages 5005–5013. 2022.   
[32] Yin, Z., M. Ye, T. Zhang, et al. Vlattack: Multimodal adversarial attacks on vision-language tasks via pre-trained models. Advances in Neural Information Processing Systems, 36:52936– 52956, 2023.   
[33] Han, D., X. Jia, Y. Bai, et al. Ot-attack: Enhancing adversarial transferability of vision-language models via optimal transport optimization. arXiv preprint arXiv:2312.04403, 2023.   
[34] Wang, H., K. Dong, Z. Zhu, et al. Transferable multimodal attack on vision-language pretraining models. In IEEE Symposium on Security and Privacy $( S P )$ , pages 1722–1740. IEEE, 2024.   
[35] He, B., X. Jia, S. Liang, et al. Sa-attack: Improving adversarial transferability of vision-language pre-training models via self-augmentation. arXiv preprint arXiv:2312.04913, 2023.   
[36] Zhao, Y., T. Pang, C. Du, et al. On evaluating adversarial robustness of large vision-language models. Advances in Neural Information Processing Systems, 36:54111–54138, 2023.   
[37] Lu, D., Z. Wang, T. Wang, et al. Set-level guidance attack: Boosting adversarial transferability of vision-language pre-training models. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 102–111. 2023.   
[38] Kuang, H., H. Liu, Y. Wu, et al. Semantically consistent visual representation for adversarial robustness. IEEE transactions on information forensics and security, 18:5608–5622, 2023.   
[39] Naseer, M., S. Khan, M. Hayat, et al. Stylized adversarial defense. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):6403–6414, 2022.   
[40] Deng, Z., X. Yang, S. Xu, et al. Libre: A practical bayesian approach to adversarial detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 972–982. 2021.   
[41] Liu, H., Y. Wu, Z. Yu, et al. Slowlidar: Increasing the latency of lidar-based detection using adversarial examples. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5146–5155. 2023.   
[42] Xiao, C., Z. Chen, K. Jin, et al. Densepure: Understanding diffusion models for adversarial robustness. In International Conference on Learning Representations (ICLR). 2023.   
[43] Nie, W., B. Guo, Y. Huang, et al. Diffusion models for adversarial purification. In International Conference on Machine Learning (ICML), vol. 162, pages 16805–16827. 2022.   
[44] Ho, C.-H., N. Vasconcelos. Disco: Adversarial defense with local implicit functions. Advances in neural information processing systems, 35:23818–23837, 2022.   
[45] Wang, H., C. Xiao, J. Kossaifi, et al. Augmax: Adversarial composition of random augmentations for robust training. Advances in neural information processing systems, 34:237–250, 2021.   
[46] Chen, C., D. Ye, Y. He, et al. Improving adversarial robustness with adversarial augmentations. IEEE Internet of Things Journal, 11(3):5105–5117, 2023.   
[47] Guo, C., M. Rana, M. Cissé, et al. Countering adversarial images using input transformations. In International Conference on Learning Representations (ICLR). OpenReview.net, 2018.   
[48] Wang, Y., H. Fu, W. Zou, et al. Mmcert: Provable defense against adversarial attacks to multi-modal models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24655–24664. 2024.   
[49] Carlini, N., F. Tramèr, K. D. Dvijotham, et al. (certified!!) adversarial robustness for free! In International Conference on Learning Representations (ICLR). 2023.   
[50] Xu, Y., Y. Sun, M. Goldblum, et al. Exploring and exploiting decision boundary dynamics for adversarial robustness. In International Conference on Learning Representations (ICLR). OpenReview.net, 2023.   
[51] Zhang, J., F. Liu, D. Zhou, et al. Improving accuracy-robustness trade-off via pixel reweighted adversarial training. In International Conference on Machine Learning (ICML). 2024.   
[52] Pang, T., X. Yang, Y. Dong, et al. Boosting adversarial training with hypersphere embedding. Advances in Neural Information Processing Systems, 33:7779–7792, 2020.   
[53] Hou, P., J. Han, X. Li. Improving adversarial robustness with self-paced hard-class pair reweighting. In AAAI Conference on Artificial Intelligence, vol. 37, pages 14883–14891. 2023.   
[54] Yuan, Z., J. Zhang, S. Shan. Fulllora-at: Efficiently boosting the robustness of pretrained vision transformers. arXiv preprint arXiv:2401.01752, 2024.   
[55] Wang, S., J. Zhang, Z. Yuan, et al. Pre-trained model guided fine-tuning for zero-shot adversarial robustness. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24502–24511. 2024.   
[56] Schlarmann, C., N. D. Singh, F. Croce, et al. Robust CLIP: unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models. In International Conference on Machine Learning (ICML). 2024.   
[57] Khattak, M. U., H. Rasheed, M. Maaz, et al. Maple: Multi-modal prompt learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19113–19122. 2023.   
[58] Roy, S., A. Etemad. Consistency-guided prompt learning for vision-language models. In International Conference on Learning Representations (ICLR). OpenReview.net, 2024.   
[59] Zhou, K., J. Yang, C. C. Loy, et al. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337–2348, 2022.   
[60] Chambolle, A. An algorithm for total variation minimization and applications. Journal of Mathematical imaging and vision, 20:89–97, 2004.   
[61] Nesterov, Y. A method for solving the convex programming problem with convergence rate o (1/k2). In Dokl akad nauk Sssr, vol. 269, page 543. 1983.   
[62] Hu, E. J., Y. Shen, P. Wallis, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR). 2022.   
[63] Ding, N., F. Farokhi, T. Guo, et al. $\alpha$ -leakage interpretation of sibson mutual information and rényi capacity. In IEEE Information Theory Workshop (ITW). 2025.   
[64] Ding, N., M. A. Zarrabian, P. Sadeghi. A cross entropy interpretation of renyi entropy for $\alpha$ -leakage. In 2024 IEEE International Symposium on Information Theory (ISIT), pages 2760– 2765. 2024.   
[65] Helber, P., B. Bischke, A. Dengel, et al. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.   
[66] Soomro, K., A. R. Zamir, M. Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.   
[67] Cimpoi, M., S. Maji, I. Kokkinos, et al. Describing textures in the wild. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3606–3613. 2014.   
[68] Xiao, J., J. Hays, K. A. Ehinger, et al. Sun database: Large-scale scene recognition from abbey to zoo. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3485–3492. IEEE, 2010.   
[69] Fei-Fei, L., R. Fergus, P. Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 178–178. 2004.   
[70] Deng, J., W. Dong, R. Socher, et al. Imagenet: A large-scale hierarchical image database. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 248–255. Ieee, 2009.   
[71] Maji, S., E. Rahtu, J. Kannala, et al. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.   
[72] Nilsback, M.-E., A. Zisserman. Automated flower classification over a large number of classes. In Indian conference on computer vision, graphics & image processing, pages 722–729. 2008.   
[73] Parkhi, O. M., A. Vedaldi, A. Zisserman, et al. Cats and dogs. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3498–3505. 2012.   
[74] Bossard, L., M. Guillaumin, L. Van Gool. Food-101–mining discriminative components with random forests. In European Conference on Computer Vision (ECCV), pages 446–461. 2014.   
[75] Krause, J., M. Stark, J. Deng, et al. 3d object representations for fine-grained categorization. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 554–561. 2013.   
[76] Hendrycks, D., K. Zhao, S. Basart, et al. Natural adversarial examples. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15262–15271. 2021.   
[77] Hendrycks, D., S. Basart, N. Mu, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 8340–8349. 2021.   
[78] Wang, H., S. Ge, Z. Lipton, et al. Learning robust global representations by penalizing local predictive power. Advances in neural information processing systems, 32, 2019.   
[79] Recht, B., R. Roelofs, L. Schmidt, et al. Do imagenet classifiers generalize to imagenet? In International Conference on Machine Learning (ICML), pages 5389–5400. PMLR, 2019.

# NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

• You should answer [Yes] , [No] , or [NA] .   
• [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.   
• Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

• Delete this instruction block, but keep the section heading “NeurIPS Paper Checklist", • Keep the checklist subsection headings, questions/answers and guidelines below. • Do not modify the questions and only use the provided macros for your answers.

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly explain the scope and importance of the work, and the main contributions are summarized at the end of the introduction.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the paper.   
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.   
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations of this paper are discussed in the appendix.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.   
• The authors are encouraged to create a separate "Limitations" section in their paper.   
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.   
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.   
The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [NA]

Justification: Only a small portion of this work requires theoretical justification, which has been rigorously proven. The remaining contributions focus on improving adversarial prompt learning from an empirical perspective.

Guidelines:

• The answer NA means that the paper does not include theoretical results.   
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.   
• All assumptions should be clearly stated or referenced in the statement of any theorems.   
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.   
Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.   
• Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide detailed implementation information in Section 4.1 and the appendix to support the reproduction of our experimental results. The corresponding code will also be included in the supplemental material.

# Guidelines:

• The answer NA means that the paper does not include experiments.   
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: For datasets, we only use open-source datasets that are publicly available. For codes, we list the original paper of baseline methods in the appendix with access to their respective code repositories.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.   
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.   
While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).   
The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.   
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: In the experimental section, we give all details concerning the experiment settings, parameter values, optimizer, etc.

Guidelines:

• The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report the average performance and standard deviations across multiple runs in the experimental results section and the appendix.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).   
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)   
• The assumptions made should be given (e.g., Normally distributed errors).   
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.   
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a $96 \%$ CI, if the hypothesis of Normality of errors is not verified. For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).   
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Details on compute resources are provided in the appendix.

Guidelines:

• The answer NA means that the paper does not include experiments.   
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.   
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

# 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We have carefully read the NeurIPS Code of Ethics and checked the anonymity of our submission.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.   
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.   
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the boarder impact of our paper in Appendix.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.   
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.   
Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.   
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.   
The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.   
If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: Our paper does not include generative models and typically uses open-source datasets for training and evaluation.

Guidelines:

• The answer NA means that the paper poses no such risks.   
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The creator of assets used in our paper states the license in their repository (MIT License).

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

Justification: Although we will submit the code in the supplementary materials, we will continue to improve the codebase and make it publicly available after the paper is officially accepted. Currently, we have not released any new assets.

Guidelines:

• The answer NA means that the paper does not release new assets.   
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: There are no crowdsourcing experiments and research with human subjects under adversarial prompt learning settings.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: There are no crowdsourcing experiments and research with human subjects under adversarial prompt learning settings.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.   
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

# 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This paper does not involve LLM.

Guidelines:

• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.

# Appendix

# A Pipelines of Adaptive-FGP Algorithm

Algorithm 2 presents the proposed adaptive fast gradient projection (adaptive-FGP) method for real-time total variation regularization. It is designed to disrupt the perturbation space of adversarial examples while maximally preserving the structural integrity of image content.

<table><tr><td></td></tr><tr><td>Input: Image U(adu), base coefficient μbase, gain μgain, convergence tolerance £</td></tr><tr><td>Output: Recovered image u*</td></tr><tr><td></td></tr><tr><td>1: Compute γ(u) = μbase · (1 + μgain · Φ(v)) on image v using Sobel operator</td></tr><tr><td>2: Initialize w° = O,w° = O, To = 1</td></tr><tr><td>3: for k = 1 to Maximum iterations do</td></tr><tr><td>4: Compute uk = v - γ(v) : div(w𝑘-1)</td></tr><tr><td>5: Compute gradient Vf(wk) = -2· γ(u) · div*(uk)</td></tr><tr><td>6: Update wk = IIw (wk -∀(uk)/8· γ(u))</td></tr><tr><td>Compute σk =uk -uk-1,σk-1= uk-1- uk-2</td></tr><tr><td>7: 8:</td></tr><tr><td>Compute cos(0k)= (σk,σk-1&gt;/(llσkll: lσk-1l + S)</td></tr><tr><td>9: if cos(0k)&gt;cos(0th) then</td></tr><tr><td>10: Tk+1=(1+√1+4²)/2</td></tr><tr><td>w+(Tk-1).(wk−wk-1)/Tk+1 11:</td></tr><tr><td>12: else Tk+1 = 1,wk+1 = wk</td></tr><tr><td>13: 14: end if</td></tr><tr><td>15: if maxie{k,k-s} |loill F/(lui|lF + S) &lt; ε then</td></tr><tr><td>16: break</td></tr><tr><td></td></tr><tr><td>17: end if</td></tr><tr><td>18: end for</td></tr><tr><td>return u* = uk</td></tr><tr><td>19:</td></tr></table>

Initialization phase. The algorithm first constructs a spatially adaptive regularization map $\gamma ( v )$ based on the input image $v$ . This regularization term is governed by a baseline intensity coefficient $\mu _ { \mathrm { b a s e } }$ and an edge sensitivity coefficient $\mu _ { \mathrm { g a i n } }$ , with the edge response $\Phi ( v )$ is estimated via the Sobel convolution operator. This strategy automatically reduces the regularization strength in edge regions to preserve structural details, while enhancing regularization intensity in the flat areas to effectively suppress adversarial perturbations. Subsequently, the dual variable $\mathbf { w } ^ { 0 }$ and its accelerated counterpart $\bar { \mathbf { w } } ^ { \hat { 0 } }$ , along with the temporal scaling factor $\tau _ { 0 }$ are initialized (Lines $1 { \sim } 2$ ).

Gradient projection update in the dual space. First, the dual variable field from the previous iteration is transformed into a scalar field via the divergence operator, which is utilized to construct the current estimate of the primal variable image $u ^ { k }$ . Subsequently, the dual variable $\bar { \mathbf { w } } ^ { k }$ at the current iteration is updated and projected onto the dual constraint set $\mathcal { W }$ to ensure that the gradient field satisfies the unit ball constraint (Lines $4 { \sim } 6$ ).

Momentum acceleration with adaptive restart mechanism. We measure whether the direction of the angle between two consecutive step increments $\sigma _ { k }$ and $\sigma _ { k - 1 }$ is reversed to determine whether a restart has occurred. If no deviation in direction is detected, the Nesterov momentum acceleration mechanism is applied to enhance convergence speed. Otherwise, if the angle between directions exceeds a predefined threshold, the momentum accumulation is reset to prevent overshooting caused by trajectory discontinuity, thereby improving the stability of the algorithm. The adaptive restart mechanism originates from an analysis of oscillatory behavior inherent in Nesterov-type momentum schemes, which is particularly important in the context of spatially weighted total variation with non-uniform regularization terms (Lines $7 \sim 1 4$ ).

Convergence criterion. The algorithm is deemed to have converged when the relative change in updates, measured by the Frobenius norm, remains below the threshold $\xi$ for $s$ consecutive iterations. This condition ensures the stability of the solution across multiple time steps in the output image while effectively avoiding redundant iterations. Upon completion of the iterations, the optimal solution $u ^ { \star }$ of the output image is obtained. The adaptive-FGP method exhibits strong parallelizability and efficient acceleration mechanisms, significantly enhancing model robustness while keeping the computational overhead below $10 \%$ (Lines $1 5 { \sim } 1 7$ ).

# B Upper Bound Analysis of the Lipschitz Constant

Since the gradient $\nabla f ( \pmb { w } )$ of the objective function $f ( w )$ is Lipschitz continuous, there exists a constant $L > 0$ such that for any $w _ { 1 } , w _ { 2 }$ , the following inequality holds:

$$
\begin{array} { r } { \| \nabla f ( \pmb { w } _ { 1 } ) - \nabla f ( \pmb { w } _ { 2 } ) \| \leq L \| \pmb { w } _ { 1 } - \pmb { w } _ { 2 } \| . } \end{array}
$$

The gradient difference can be computed as:

$$
\begin{array} { r l } & { \nabla f ( w _ { 1 } ) - \nabla f ( w _ { 2 } ) = - 2 \gamma ( v ) \cdot \nabla \operatorname { d i v } ^ { * } \left[ ( v - \gamma ( v ) \cdot \operatorname { d i v } ( w _ { 1 } ) ) - ( v - \gamma ( v ) \cdot \operatorname { d i v } ( w _ { 2 } ) ) \right] } \\ & { \qquad = 2 \gamma ( v ) ^ { 2 } \cdot \nabla \operatorname { d i v } ^ { * } \left[ \operatorname { d i v } ( w _ { 1 } ) - \operatorname { d i v } ( w _ { 2 } ) \right] } \\ & { \qquad = 2 \gamma ( v ) ^ { 2 } \cdot \nabla \operatorname { d i v } ^ { * } \cdot \operatorname { d i v } ( w _ { 1 } - w _ { 2 } ) . } \end{array}
$$

Thus, the norm is bounded by:

$$
\begin{array} { r l } & { \| \nabla f ( { \pmb w } _ { 1 } ) - \nabla f ( { \pmb w } _ { 2 } ) \| \le 2 \gamma ( v ) ^ { 2 } \cdot \| \nabla \operatorname { d i v } ^ { T } \cdot \operatorname { d i v } \| \cdot \| { \pmb w } _ { 1 } - { \pmb w } _ { 2 } \| } \\ & { \qquad \le 2 \gamma ( v ) ^ { 2 } \cdot \| \operatorname { d i v } \| ^ { 2 } \cdot \| { \pmb w } _ { 1 } - { \pmb w } _ { 2 } \| . } \end{array}
$$

Analogous to the spectral norm bound of the discrete gradient operator in the TV regularization term,√ if the operator norm of the discrete divergence operator satisfies $| \operatorname { d i v } | \leq { \sqrt { 8 } }$ , we obtain:

$$
\begin{array} { r } { \| \nabla f ( \pmb { w } _ { 1 } ) - \nabla f ( \pmb { w } _ { 2 } ) \| \le 1 6 \gamma ( \pmb { v } ) ^ { 2 } \cdot \| \pmb { w } _ { 1 } - \pmb { w } _ { 2 } \| . } \end{array}
$$

Therefore, the upper bound of the Lipschitz constant $L ( f )$ for the objective function $f ( w )$ is given by:

$$
L ( f ) \leq 1 6 \gamma ( v ) ^ { 2 } .
$$

# C Additional Experimental Results

# C.1 Sensitivity Analysis of PGD Attack Hyperparameters

Table 5 systematically evaluates the impact of different configurations on natural and robust accuracy across five datasets (Caltech101 [69], DTD [67], EuroSAT [65], FGVC-Aircraft [71], OxfordPets [73]) under the 16-shot setting and varying perturbation budgets $\epsilon = \{ 1 / 2 5 5 , 2 / 2 5 5 , 4 / 2 5 5 \}$ . Specifically, it assesses the sensitivity to different numbers of attack iterations $\iota = \{ 2 , 4 , 8 \}$ and step sizes $\varsigma = \dot { \{ \epsilon / \iota , 2 \epsilon / \iota , 4 \epsilon / \iota \} }$ . During the robustness evaluation phase, a 100-step PGD attack with the same perturbation budget and step size as in the training phase is employed to fully examine the defense capability of the model under strong attacks. We aim to determine the optimal combination of hyperparameters to more efficiently perform the next adversarial robustness tests under stronger attacks.

As evidenced in Table 5, employing larger attack step counts and step sizes during training $\zeta = 8 , \zeta =$ $4 \epsilon / \iota \dot$ ) does not enhance adversarial robustness during evaluation. Adversarial examples generated by PGD-8 tend to deviate significantly from the true data distribution, potentially causing the model to overfit the distribution of adversarial samples encountered during training rather than learning generalizable robust features. The model achieves higher natural accuracy when trained with a larger number of attack steps and a smaller step size $( \iota = 8 , \varsigma = \epsilon / \iota )$ , as the resulting adversarial examples remain in close proximity to the original data manifold. The model demonstrates the capability to learn robust features while preserving discriminative power for natural samples. Across all perturbation budget settings, the combination of two attack iterations with a step size of $4 \epsilon / \iota$ consistently achieves optimal robust accuracy and high clean accuracy. Therefore, we adopt this hyperparameter configuration for subsequent experiments involving varying perturbation budgets and different adversarial attack methods.

Table 5: Impact of perturbation budgets, attack iteration steps, and attack step sizes on natural and robust accuracy across 5 datasets under the 16-shot benchmark. Bold values highlight the best average results per perturbation budget.   

<table><tr><td rowspan="2">Pert. budg. e</td><td rowspan="2">Iter. steps</td><td rowspan="2">Step size s</td><td colspan="2">Caltech101</td><td colspan="2">DTD</td><td colspan="2">EuroSAT</td><td colspan="2">FGVCAircraft</td><td colspan="2">OxfordPets</td><td colspan="2">Average</td></tr><tr><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td></tr><tr><td rowspan="3">1/255</td><td>2</td><td>e/ 2/ 4e/</td><td>94.16 94.04 94.20</td><td>89.57 89.78 90.55</td><td>65.37 65.84 65.19</td><td>54.20 56.03 57.03</td><td>86.42 84.77 86.35</td><td>71.67 70.60 73.52</td><td>38.94 39.39 39.39</td><td>25.47 27.51 28.74</td><td>87.05 86.29 87.33</td><td>70.84 71.85 74.35</td><td>74.39 74.07 74.49</td><td>62.35 63.15 64.84</td></tr><tr><td>4</td><td>/ 2/ 4ε/</td><td>94.36 94.08 94.04</td><td>89.86 89.78 90.14</td><td>65.43 65.13 65.66</td><td>55.38 55.38 56.21</td><td>86.16 85.62 86.47</td><td>71.20 73.37 74.75</td><td>39.18 39.24 39.72</td><td>26.52 27.24 27.30</td><td>87.44 86.37 86.86</td><td>70.43 71.74 72.47</td><td>74.51 74.09 74.55</td><td>62.68 63.50 64.18</td></tr><tr><td>8</td><td>/ 2/ 4/</td><td>94.20 94.04 94.24</td><td>89.74 89.98 89.90</td><td>66.31 65.72 65.19</td><td>54.85 55.56 55.50</td><td>86.52 86.49 86.65</td><td>69.32 73.10 74.09</td><td>39.30 38.94 38.88</td><td>25.59 28.05 27.18</td><td>87.11 86.07 86.59</td><td>69.80 71.41 71.95</td><td>74.69 74.25 74.31</td><td>61.86 63.62 63.72</td></tr><tr><td rowspan="3">2/255</td><td>2</td><td>e/ 2/ 4/</td><td>93.96 93.67 93.91</td><td>87.10 86.82 88.32</td><td>64.24 63.12 64.30</td><td>49.23 50.00 52.36</td><td>83.86 81.70 84.07</td><td>67.12 64.26 66.90</td><td>38.58 37.11 36.09</td><td>21.90 21.90 23.52</td><td>84.93 83.21 84.36</td><td>58.63 59.23 63.07</td><td>73.11 71.76 72.55</td><td>56.80 56.44 58.83</td></tr><tr><td>4</td><td>/ 2/ 4/</td><td>94.00 93.71 93.71</td><td>86.09 86.77 86.73</td><td>65.07 63.48 63.42</td><td>49.11 49.88 49.70</td><td>83.98 83.90 83.77</td><td>68.99 70.63 68.53</td><td>37.95 36.72 36.72</td><td>22.11 22.95 24.36</td><td>84.71 83.05 83.89</td><td>58.74 58.63 59.42</td><td>73.14 72.17 72.30</td><td>57.01 57.77 57.75</td></tr><tr><td>8</td><td>e/ 2/ 4e/</td><td>93.91 93.67 93.71</td><td>86.21 86.73 86.94</td><td>64.30 62.83 63.00</td><td>48.58 48.88 49.59</td><td>84.10 80.64 83.74</td><td>67.96 67.58 67.54</td><td>37.68 37.11 37.44</td><td>22.02 23.76 23.64</td><td>84.87 82.75 83.05</td><td>57.10 57.78 58.54</td><td>72.97 71.40 72.19</td><td>56.37 56.95 57.25</td></tr><tr><td rowspan="3">4/255</td><td>2</td><td>e/ 2/ 4/</td><td>92.41 92.01 92.58</td><td>82.15 80.41 83.20</td><td>60.87 58.92 59.63</td><td>41.08 38.36 45.27</td><td>79.53 79.54 79.49</td><td>67.17 57.63 58.51</td><td>33.39 32.67 33.33</td><td>17.28 18.24 19.80</td><td>79.26 74.79 79.45</td><td>40.88 39.60 49.03</td><td>69.09 67.59 68.90</td><td>49.71 46.85 51.16</td></tr><tr><td>4</td><td>/ 2/ 4/</td><td>92.74 92.01 92.33</td><td>81.30 80.20 80.45</td><td>61.82 59.69 59.04</td><td>40.07 40.31 38.42</td><td>80.12 78.70 79.57</td><td>64.15 63.54 59.01</td><td>33.57 31.53 33.03</td><td>17.76 17.85 18.54</td><td>79.50 75.61 76.04</td><td>41.40 39.55 39.11</td><td>69.55 67.51 68.00</td><td>48.94 48.29 47.11</td></tr><tr><td>8</td><td>/ 2/ 4/</td><td>93.10 91.60 91.85</td><td>81.99 79.63 80.81</td><td>61.76 59.46 58.92</td><td>40.07 40.19 40.60</td><td>82.06 78.99 78.51</td><td>60.85 61.49 63.32</td><td>35.07 32.37 32.82</td><td>18.15 18.60 19.50</td><td>78.30 75.28 76.02</td><td>40.47 38.35 40.94</td><td>70.06 67.54 67.62</td><td>48.31 47.65 49.03</td></tr></table>

# C.2 Impact of Perturbation Budget on Model Performance

We document the performance of CoAPT under four benchmark settings with three perturbation budgets $\epsilon = \{ 1 / 2 5 5 , 2 / 2 5 5 , 4 / 2 5 5 \}$ . The case of $\epsilon = 1 / 2 5 5$ corresponds to the results presented in the main text of the paper. As shown in Table 6 under the base-to-novel benchmark, the robust HM metrics decrease by $5 . 7 2 \%$ and $9 . 7 3 \%$ as the perturbation budgets increase, remaining within acceptable thresholds overall. The natural HM metrics decrease by only $2 . 2 7 \%$ and $4 . 5 9 \%$ , respectively, demonstrating the effectiveness of CoAPT in preserving natural generalization.

Table 6: Performance of CoAPT under varying perturbation budgets on the base-to-novel benchmark across 11 datasets.   

<table><tr><td>E</td><td>Metric</td><td>Caltech101</td><td></td><td>DTD EuroSAT</td><td>FGVCAircraft</td><td>Food101</td><td>ImageNet</td><td>Flowers101</td><td></td><td>OxfordPets StanfordCars</td><td>SUN397</td><td>UCF101</td><td>Average</td></tr><tr><td rowspan="7">6</td><td>Base M</td><td>97.25</td><td>76.08</td><td>91.61</td><td>35.37</td><td>78.20</td><td>66.15</td><td>94.94</td><td>90.55</td><td>73.34</td><td>76.69</td><td>82.95</td><td>78.47</td></tr><tr><td>Novel</td><td>92.72 94.93</td><td>54.03</td><td>56.11</td><td>25.41</td><td>79.47</td><td>55.41</td><td>63.07</td><td>94.50</td><td>59.20</td><td>70.46</td><td>68.45</td><td>65.35</td></tr><tr><td>HM</td><td></td><td>63.18</td><td>69.60</td><td>29.58</td><td>78.83</td><td>60.30</td><td>75.79</td><td>92.49</td><td>65.51</td><td>73.44</td><td>75.01</td><td>71.31</td></tr><tr><td>Base </td><td>94.38</td><td>67.98</td><td>84.67</td><td>25.37</td><td>62.03</td><td>52.65</td><td>88.57</td><td>78.72</td><td>54.20</td><td>64.50</td><td>71.65</td><td>67.70</td></tr><tr><td>Novel</td><td>88.03</td><td>43.88</td><td>47.40</td><td>16.68</td><td>62.86</td><td>45.07</td><td>51.89</td><td>83.71</td><td>40.95</td><td>58.50</td><td>56.50</td><td>54.13</td></tr><tr><td>HM</td><td>91.09</td><td>53.33</td><td>60.78</td><td>20.12</td><td>62.44</td><td>48.57</td><td>65.44</td><td>81.13</td><td>46.65</td><td>61.35</td><td>63.18</td><td>60.16</td></tr><tr><td>M</td><td>Base</td><td>96.90</td><td>75.46</td><td>89.00</td><td>34.45</td><td>71.77</td><td>63.68</td><td></td><td></td><td>69.34</td><td>74.99 81.08</td><td>76.40</td></tr><tr><td rowspan="6">6 3</td><td>Novel</td><td>90.39</td><td>52.29</td><td>62.72</td><td>25.19</td><td>73.29</td><td>53.45</td><td>94.87 57.23</td><td>88.89 91.16</td><td>55.06</td><td>67.65</td><td>64.31</td><td>62.98</td></tr><tr><td>HM</td><td>93.53</td><td>61.78</td><td>73.58</td><td>29.11</td><td>72.52</td><td>58.12</td><td>71.40</td><td>90.01</td><td>61.38</td><td>71.13</td><td>71.72</td><td>69.04</td></tr><tr><td>Base</td><td>93.35</td><td>63.77</td><td>79.00</td><td>20.11</td><td>50.12</td><td>48.76</td><td>85.94</td><td></td><td>44.70</td><td>60.50</td><td>67.79</td><td>62.15</td></tr><tr><td>Novel</td><td>84.83</td><td>40.34</td><td>53.36</td><td>14.04</td><td>50.20</td><td>40.93</td><td>42.48</td><td>69.59 73.21</td><td>32.83</td><td>52.94</td><td>47.54</td><td>48.43</td></tr><tr><td>HM</td><td>88.88</td><td>49.42</td><td>63.70</td><td>16.53</td><td>50.16</td><td>44.50</td><td>56.86</td><td>71.35</td><td>37.86</td><td>56.47</td><td>55.89</td><td>54.44</td></tr><tr><td rowspan="5">M 8</td><td>Base</td><td>95.22</td><td>71.99</td><td>89.24</td><td>29.65</td><td>64.67</td><td>58.48</td><td>91.17</td><td>84.26</td><td>62.12</td><td>71.27</td><td>77.40</td><td>72.32</td></tr><tr><td>Novel</td><td>86.24</td><td>48.31</td><td>65.10</td><td>22.14</td><td>64.41</td><td>48.42</td><td>49.08</td><td>85.51</td><td>47.78</td><td>63.70</td><td>58.73</td><td>58.13</td></tr><tr><td>HM</td><td>90.51</td><td>57.82</td><td>75.28</td><td>25.35</td><td>64.54</td><td>52.98</td><td>63.81</td><td>84.88</td><td>54.02</td><td>67.27</td><td>66.79</td><td>64.45</td></tr><tr><td>Base</td><td>88.32</td><td>54.75</td><td>74.52</td><td>15.97</td><td>34.56</td><td>39.71</td><td>78.63</td><td>54.12</td><td>32.03</td><td>51.19</td><td>57.08</td><td>52.81</td></tr><tr><td>Novel HM</td><td>75.11</td><td>32.97 81.18</td><td>50.56 41.16 60.25</td><td>10.62 12.75</td><td>31.62</td><td>33.04</td><td>30.92 44.39</td><td>57.10</td><td>23.32</td><td>43.99</td><td>37.21</td><td>38.77 44.71</td></tr></table>

Table 7 reports the natural and robust accuracy of CoAPT under the 16-shot setting across different perturbation budgets. Compared to the base-to-novel setup, the few-shot scenario provides more training samples, enabling the model to exhibit greater stability when confronted with increased perturbations. Specifically, as the perturbation budgets increase, the robust accuracy declines by

$5 . 4 2 \%$ and $8 . 7 3 \%$ , while the natural accuracy drops by only $2 . 2 6 \%$ and $4 . 4 2 \%$ , indicating a more moderate performance degradation trend.

Table 7: Performance of CoAPT under varying perturbation budgets on the few-shot benchmark across 11 datasets.   

<table><tr><td>E</td><td>Metric</td><td>Caltech101</td><td>DTD</td><td>EuroSAT</td><td>FGVCAircraft</td><td>Food101</td><td>ImageNet</td><td>Flowers101</td><td></td><td>OxfordPets StanfordCars</td><td>SUN397</td><td>UCF101</td><td>Average</td></tr><tr><td></td><td>Nat.</td><td>94.51</td><td>65.50</td><td>85.42</td><td>38.60</td><td>73.00</td><td>62.96</td><td>93.86</td><td>86.82</td><td>73.94</td><td>70.84</td><td>79.09</td><td>74.96</td></tr><tr><td></td><td>Rob.</td><td>90.03</td><td>56.09</td><td>71.43</td><td>27.39</td><td>56.72</td><td>51.32</td><td>85.36</td><td>72.28</td><td>55.91</td><td>58.33</td><td>67.91</td><td>62.98</td></tr><tr><td></td><td>Nat.</td><td>93.91</td><td>64.30</td><td>84.07</td><td>36.09</td><td>68.08</td><td>61.38</td><td>91.64</td><td>84.36</td><td>70.25</td><td>69.42</td><td>76.24</td><td>72.70</td></tr><tr><td></td><td>Rob.</td><td>88.48</td><td>52.36</td><td>66.90</td><td>23.28</td><td>46.99</td><td>47.79</td><td>81.20</td><td>63.07</td><td>47.05</td><td>54.52</td><td>61.56</td><td>57.56</td></tr><tr><td></td><td>Nat.</td><td>92.58</td><td>59.63</td><td>79.49</td><td>33.33</td><td>60.69</td><td>56.98</td><td>87.74</td><td>79.45</td><td>63.50</td><td>65.70</td><td>72.03</td><td>68.28</td></tr><tr><td></td><td>Rob.</td><td>83.20</td><td>45.27</td><td>58.51</td><td>20.10</td><td>32.19</td><td>40.14</td><td>72.55</td><td>49.03</td><td>35.43</td><td>47.11</td><td>53.56</td><td>48.83</td></tr></table>

As shown in the evaluation results under the zero-shot settings in Table 8, our model consistently demonstrates strong natural generalization, adversarial robustness, and stability across different perturbation budgets. Specifically, under the zero-shot scenario, the average robust accuracy decreases by $3 . 9 3 \%$ and $7 . 3 5 \%$ with increasing perturbation budgets, while the average natural accuracy declines by only $1 . 6 3 \%$ and $4 . 2 1 \%$ . The results indicate that the model maintains strong perturbation resistance even under extreme generalization conditions. The evaluation results under the out-of-distribution settings in Table 9 exhibit a similar trend.

Table 8: Performance of CoAPT under varying perturbation budgets on the zero-shot benchmark across 11 datasets.   

<table><tr><td>E</td><td>Metric</td><td>ImageNet</td><td>Caltech101</td><td>DTD</td><td>EuroSAT</td><td>FGVCAircraft</td><td>Food101</td><td>Flowers101</td><td>OxfordPets StanfordCars</td><td></td><td>SUN397</td><td>UCF101</td><td>Average</td></tr><tr><td></td><td>Nat.</td><td>63.42</td><td>89.10</td><td>36.66</td><td>28.37</td><td>13.84</td><td>69.85</td><td>56.30</td><td>82.98</td><td>47.99</td><td>58.10</td><td>57.34</td><td>54.91</td></tr><tr><td></td><td>Rob.</td><td>51.18</td><td>83.29</td><td>30.59</td><td>20.13</td><td>9.43</td><td>55.99</td><td>45.34</td><td>70.03</td><td>30.24</td><td>46.64</td><td>47.36</td><td>44.57</td></tr><tr><td></td><td>Nat.</td><td>61.33</td><td>88.84</td><td>36.52</td><td>26.94</td><td>11.64</td><td>66.13</td><td>55.42</td><td>82.07</td><td>46.14</td><td>56.52</td><td>54.51</td><td>53.28</td></tr><tr><td></td><td>Rob.</td><td>46.98</td><td>80.24</td><td>29.31</td><td>19.06</td><td>7.17</td><td>50.28</td><td>41.29</td><td>64.54</td><td>23.96</td><td>42.11</td><td>42.11</td><td>40.64</td></tr><tr><td></td><td>Nat.</td><td>56.77</td><td>87.42</td><td>33.75</td><td>22.72</td><td>12.39</td><td>57.69</td><td>48.40</td><td>77.13</td><td>41.01</td><td>52.60</td><td>49.85</td><td>49.07</td></tr><tr><td></td><td>Rob.</td><td>38.62</td><td>75.66</td><td>25.24</td><td>14.17</td><td>6.51</td><td>36.61</td><td>33.82</td><td>52.03</td><td>15.00</td><td>34.87</td><td>33.65</td><td>33.29</td></tr></table>

Table 9: Performance of CoAPT under varying perturbation budgets on the out-of-distribution benchmark across 11 datasets.   

<table><tr><td rowspan="2">E</td><td colspan="2">ImageNet-A</td><td colspan="2">ImageNet-R</td><td colspan="2">ImageNet-Sketch</td><td colspan="2">ImageNet-V2</td><td colspan="2">Average</td></tr><tr><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td><td>Nat.</td><td>Rob.</td></tr><tr><td>1/255</td><td>16.99</td><td>9.72</td><td>60.35</td><td>50.71</td><td>35.76</td><td>29.13</td><td>54.35</td><td>42.23</td><td>41.86</td><td>32.95</td></tr><tr><td>2/255</td><td>14.27</td><td>7.49</td><td>57.81</td><td>45.92</td><td>34.26</td><td>25.81</td><td>52.54</td><td>38.50</td><td>39.72</td><td>29.43</td></tr><tr><td>4/255</td><td>10.35</td><td>4.23</td><td>54.28</td><td>38.80</td><td>32.17</td><td>21.10</td><td>47.63</td><td>30.67</td><td>36.11</td><td>23.70</td></tr></table>

# C.3 Robustness Evaluation under Varying Attacks

We evaluate our method using attack types based on different perturbation mechanisms. The CW attack is an optimization-based method designed to generate adversarial perturbations that are minimal in magnitude yet highly effective in misleading the model. It has demonstrated strong attack performance across various tasks. The TPGD attack is a targeted variant of the PGD attack that misdirects samples toward specific target classes. AutoAttack is an ensemble-based, parameterfree robustness evaluation framework that integrates multiple strong attack algorithms to provide reliable adversarial assessment results. Specifically, we evaluate CW, TPGD, and AutoAttack attacks under the zero-shot benchmark, while only CW and TPGD are evaluated under the base-to-novel benchmark. We adopt PGD attack with the hyperparameter configuration $\epsilon = 4 / 2 5 5$ , $\iota = 2$ , $\varsigma = 4 \epsilon / \iota$ for adversarial training. During the robustness evaluation phase, both CW and TPGD attacks are applied with the same perturbation budget and step size, while the number of attack steps is uniformly set to 100. For AutoAttack, we use the same perturbation budget $\epsilon = 4 / 2 5 5 )$ ), and its attack process does not rely on hyperparameters such as step size or the number of steps. Overall, the robustness advantage of our method is not a result of overfitting to any specific attack.

Table 10: Performance of CoAPT against various attack methods under the base-to-novel benchmark.   

<table><tr><td rowspan="2">Type</td><td rowspan="2">Metric</td><td rowspan="2">Base</td><td rowspan="2">Caltech101 95.22</td><td rowspan="2">DTD</td><td rowspan="2">EuroSAT</td><td rowspan="2">FGVCAircraft Food101</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">ImageNet Flowers101 OxfordPets StanfordCars SUN397</td><td rowspan="2"></td><td rowspan="2">UCF101 Average 77.40</td></tr><tr><td>91.17</td></tr><tr><td rowspan="6">U</td><td rowspan="2">Nat.</td><td>Novel</td><td>86.24</td><td>71.99 48.31</td><td>89.24 65.10</td><td>29.65 22.14</td><td>64.67 64.41</td><td>58.48 48.42</td><td>49.08</td><td>84.26 85.51</td><td>62.12 47.78</td><td>71.27 63.70</td><td>58.73</td><td>72.32 58.13</td></tr><tr><td>HM</td><td>90.51</td><td>57.82</td><td>75.28</td><td></td><td>64.54</td><td></td><td>63.81</td><td></td><td></td><td></td><td>66.79</td><td>64.45</td></tr><tr><td rowspan="2"></td><td></td><td></td><td></td><td></td><td>25.35</td><td></td><td>52.98</td><td></td><td>84.88</td><td>54.02</td><td>67.27</td><td></td><td></td></tr><tr><td>Base Novel</td><td>86.38</td><td>59.38</td><td>81.64</td><td>21.67</td><td>58.12</td><td>51.32</td><td>86.32</td><td>67.68</td><td>44.50</td><td>60.27</td><td>67.68</td><td>62.27</td></tr><tr><td>Rob.</td><td></td><td>75.00</td><td>38.41</td><td>51.64</td><td>17.34</td><td>56.55</td><td>40.39</td><td>42.48</td><td>69.35</td><td>35.31</td><td>52.56</td><td>46.73</td><td>47.80</td></tr><tr><td rowspan="2"></td><td>HM</td><td>80.29</td><td>46.64</td><td>63.27</td><td>19.26</td><td>57.33</td><td>45.20</td><td>56.94</td><td>68.50</td><td>39.37</td><td>56.15</td><td>55.29</td><td>54.08</td></tr><tr><td>Base Novel</td><td>95.16</td><td>71.99</td><td>89.21</td><td>29.83</td><td>64.68</td><td>58.46</td><td>91.17</td><td>84.26</td><td>62.12</td><td>71.21</td><td>77.40</td><td>72.32</td></tr><tr><td rowspan="4">0</td><td>Nat.</td><td></td><td>86.24 48.31</td><td>65.05</td><td>22.08</td><td>64.39</td><td>48.44</td><td>49.01</td><td>85.51</td><td>47.81</td><td>63.66</td><td>58.73</td><td></td><td>58.11</td></tr><tr><td></td><td>HM</td><td>90.48</td><td>57.82 75.24</td><td>25.37</td><td>64.53</td><td>52.98</td><td>63.75</td><td>84.88</td><td></td><td>54.03 67.22</td><td>66.79</td><td></td><td>64.44</td></tr><tr><td></td><td>Base</td><td>93.74</td><td>68.29 90.62</td><td>29.59</td><td></td><td>62.93</td><td>55.85</td><td>90.50</td><td>79.59</td><td>58.12</td><td>68.74</td><td>75.85</td><td>70.35</td></tr><tr><td>Rob.</td><td>Novel</td><td>84.39</td><td>42.39 62.41</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>55.64</td></tr><tr><td></td><td></td><td>HM</td><td>88.82</td><td>52.31</td><td>73.91</td><td>21.60 24.97</td><td>62.67 62.80</td><td>46.47 50.73</td><td>48.01 62.74</td><td>80.48 80.03</td><td>43.85 49.99</td><td>61.42 64.88</td><td>58.36 65.96</td><td>62.14</td></tr></table>

As can be seen from the experimental results under the base-to-novel benchmark in Table 10, our approach exhibits strong robust generalization capabilities when confronted with different types of adversarial attacks. Overall, the CW attack is more destructive. Although it induces significant accuracy degradation on novel classes, the performance remains within acceptable range. In contrast, under the TPGD attack, the model maintains relatively high natural and robust accuracy, further validating the stable performance of CoAPT across different types of adversarial attacks.

Figure 4 presents the robust accuracy of the model under CW, AutoAttack, and TPGD attacks across 11 datasets in the zero-shot benchmark. In terms of overall trends, the model demonstrates the strongest robustness under TPGD attacks, achieving the highest robust accuracy across nearly all datasets. In contrast, CW attacks are more destructive, particularly showing stronger attack effectiveness on complex datasets such as ImageNet and StanfordCars. AutoAttack, as an ensemble-based evaluation framework, displays intermediate attack strength between CW and TPGD. Moreover, significant robustness variations exist across different datasets. The model maintains relatively high robust accuracy on Caltech101, Flowers102, and OxfordPets, while showing noticeably lower performance on FGVCAircraft and EuroSAT.

![](images/24ca2b3a1ff1187ebaed7286ff45de88b732ab29692efccf8d8e2c687bb6ceb5.jpg)  
Figure 4: Comparison of robust accuracy under different attack methods on zero-shot benchmarks.

To evaluate the impact of $\ell _ { 2 }$ -norm adversarial attacks on robust VLMs, we designed and conducted an experiment based on $\ell _ { 2 }$ -norm perturbations. The training weights were derived from the $\ell _ { \infty }$ -based PGD attack, and the evaluation settings remained consistent. Table 11 presents the experimental results of our approach across five datasets under varying perturbation budgets. It can be observed that as the perturbation budget increases, the model’s classification accuracy experiences a moderate decline. Nevertheless, our approach significantly improves the model’s robustness against $\ell _ { 2 }$ -norm attacks, even under the $\ell _ { \infty }$ -norm threat model.

Table 11: Robust accuracy under $\ell _ { 2 }$ -norm PGD attacks on the base-to-novel benchmark.   

<table><tr><td rowspan="2">E</td><td colspan="2">Caltech101</td><td colspan="2">DTD</td><td colspan="2">EuroSAT</td><td colspan="2">FGVCAircraft</td><td colspan="2">OxfordPets</td></tr><tr><td>Base</td><td>Novel</td><td>Base</td><td>Novel</td><td>Base</td><td>Novel</td><td>Base</td><td>Novel</td><td>Base</td><td>Novel</td></tr><tr><td>1/255</td><td>94.25</td><td>87.99</td><td>68.29</td><td>44.44</td><td>85.45</td><td>55.23</td><td>25.87</td><td>16.86</td><td>80.75</td><td>83.95</td></tr><tr><td>2/255</td><td>92.32</td><td>84.06</td><td>62.38</td><td>37.80</td><td>79.64</td><td>54.49</td><td>20.05</td><td>12.96</td><td>73.52</td><td>75.17</td></tr><tr><td>4/255</td><td>90.70</td><td>81.00</td><td>61.00</td><td>36.96</td><td>81.00</td><td>54.38</td><td>18.91</td><td>13.92</td><td>70.28</td><td>70.86</td></tr></table>

# C.4 Sensitivity Analysis of Prompt Length and Depth in Multimodal Prompting

Prompt depth and prompt length. We conduct ablation studies on prompt depth and prompt length under the base-to-novel setting across 10 datasets, excluding ImageNet and its variants. Figure 5 summarizes the average results over these datasets. As shown in the left panel of Figure 5, model performance steadily improves with increasing adversarial prompt depth. However, performance gains plateau when the depth exceeds nine layers, showing diminishing returns. To avoid introducing excessive trainable parameters, we ultimately set the prompt depth to 9.

The right panel of Figure 5 illustrates the impact of prompt length on model performance. As the number of prompt tokens increases, the natural and robust performance on base classes remains relatively stable, whereas the natural and robust performance of the novel classes exhibits a declining trend. This indicates that excessive trainable prompt tokens are prone to overfit task-specific features, thereby undermining the task-agnostic generalization capability of VLMs. Similar performance trends have also been reported in the literature [57]. The model achieves optimal performance when the prompt length is set to 4.

![](images/c6a582fe945a4b1f01982c1bc8903629be3f09e971f1269e26acd5d9fec697b6.jpg)  
Figure 5: Analyze the impact of prompt depth (left) and prompt length (right) on the performance.

# C.5 Performance Across Different CLIP Architectures

We additionally evaluate CoAPT on the CLIP ViT-B/16 architecture under the base-to-novel benchmark to verify its scalability to higher-resolution architectures in terms of both natural accuracy and adversarial robustness. Compared to ViT-B/32, the ViT-B/16 architecture adopts finer image patching granularity, resulting in a greater number of input tokens and consequently exhibiting superior spatial resolution representation capacity. This structural advantage typically leads to enhanced performance in fine-grained visual tasks.

Table 12: Results of base-to-novel benchmarks on the ViT-B/16 architecture of CLIP under 11 datasets.   

<table><tr><td></td><td>Metric</td><td></td><td>Caltech101</td><td>DTD</td><td>EuroSAT</td><td>FGVCAircraft Food101</td><td></td><td></td><td>ImageNet Flowers101</td><td></td><td>OxfordPets StanfordCars</td><td>SUN397</td><td>UCF101</td><td>| Average</td></tr><tr><td></td><td>Base</td><td></td><td>97.93</td><td>78.13</td><td>93.64</td><td>41.54</td><td>83.26</td><td>71.76</td><td>97.63</td><td>94.10</td><td>77.49</td><td>79.21</td><td>83.82</td><td>81.68</td></tr><tr><td></td><td>M Novel</td><td></td><td>94.00</td><td>56.40</td><td>54.36</td><td>32.87</td><td>83.50</td><td>60.02</td><td>67.09</td><td>94.69</td><td>63.33</td><td>72.96</td><td>72.36</td><td>68.33</td></tr><tr><td>G</td><td>HM</td><td></td><td>95.92</td><td>65.51</td><td>68.79</td><td>36.70</td><td>83.38</td><td>65.36</td><td>79.53</td><td>94.39</td><td>69.70</td><td>75.95</td><td>77.67</td><td>74.41</td></tr><tr><td></td><td>Base</td><td></td><td>96.45</td><td>71.76</td><td>90.05</td><td>34.21</td><td>72.14</td><td>61.72</td><td>92.31</td><td>87.08</td><td>62.24</td><td>70.66</td><td>75.28</td><td>73.99</td></tr><tr><td></td><td>Novel</td><td></td><td>90.72</td><td>51.45</td><td>47.64</td><td>25.07</td><td>72.43</td><td>52.47</td><td>58.09</td><td>88.59</td><td>48.30</td><td>64.33</td><td>63.17</td><td>60.21</td></tr><tr><td>3</td><td>HM</td><td>93.50</td><td></td><td>59.93</td><td>62.31</td><td>28.94</td><td>72.28</td><td>56.72</td><td>71.30</td><td>87.83</td><td>54.40</td><td>67.35</td><td>68.70</td><td>66.39</td></tr></table>

Compared to the CLIP ViT-B/32 results reported in Table 1 of the main text, Table 12 demonstrates that the CLIP ViT-B/16 architecture achieves improvements of $3 . 1 \%$ and $6 . 2 3 \%$ in the HM of natural accuracy and robust accuracy, respectively. The high-resolution visual representations of the ViT-B/16 architecture provide CoAPT with a finer-grained and more stable latent space, enabling more effective reconstruction of natural generalization features disrupted by adversarial perturbations. Compared to the ViT-B/32 architecture, this enhanced representational capacity mitigates alignment errors and distributional shifts between language and vision embeddings, thereby significantly improving the natural generalization and adversarial robustness of robust CLIP. In contrast, the FAP method fails to achieve robustness gains under the ViT-B/16 architecture, further demonstrating the superiority of CoAPT in terms of scalability and stability.

# C.6 Impact of Reconstruction Loss Functions on Model Performance

CoAPT employs a Gaussian radial basis function (RBF) to measure the similarity between the language and vision branch embeddings of natural and robust CLIP representations in the latent space, effectively capturing the impact of input perturbations on the feature distributions. In Table 13, we systematically compare the performance of CoAPT on the base-to-novel benchmark under different configurations of Gaussian RBF and standard MSE loss functions. The Gaussian RBF demonstrates absolute superiority over MSE by $5 . 1 5 \%$ and $4 . 8 1 \%$ in natural HM and robust HM metrics, respectively. This is attributed to the fact that Gaussian RBF can effectively amplify the feature shifts caused by small-scale perturbations to acutely capture the subtle distributional changes, which not only promotes robustness training but also inhibits overfitting to a certain extent.

# C.7 Independent and Joint Vision-Language Adversarial Prompting

CoAPT employs deep contextualized joint vision-language adversarial prompting (JVLAP), which refines visual prompts based on linguistic prompts to facilitate cross-modal co-optimization via a vision-language coupling network. In Table 14, we additionally report the performance of CoAPT using independent vision-language adversarial prompting (IVLAP) under the base-to-novel benchmark. Compared to the JVLAP results in Table 1, IVLAP exhibits reductions of $0 . 2 9 \%$ and $0 . 3 7 \%$ in the HM of natural and robust accuracy, respectively. Although IVLAP shows slightly better performance on the Flowers101 and StanfordCars datasets, its performance on most other datasets is comparable to or slightly inferior to that of JVLAP.

Table 13: Results of base-to-novel benchmarks using Gaussian RBF and MSE loss functions under 11 datasets.   

<table><tr><td></td><td>Metric Caltech101 DTD</td><td>EuroSAT FGVCAircraft Food101 ImageNet Flowers101 OxfordPets StanfordCars SUN397</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>UCF101</td><td>|Average</td></tr><tr><td></td><td>Base</td><td>97.25</td><td>76.08</td><td>91.61</td><td>35.37</td><td>78.20</td><td>66.15</td><td>94.94</td><td>90.55</td><td>73.34</td><td>76.69</td><td>82.95</td><td>78.47</td></tr><tr><td></td><td>Novel</td><td>92.72</td><td>54.03</td><td>56.11</td><td>25.41</td><td>79.47</td><td>55.41</td><td>63.07</td><td>94.50</td><td>59.20</td><td>70.46</td><td>68.45</td><td>65.35</td></tr><tr><td></td><td>HM</td><td>94.93</td><td>63.18</td><td>69.60</td><td>29.58</td><td>78.83</td><td>60.30</td><td>75.79</td><td>92.49</td><td>65.51</td><td>73.44</td><td>75.01</td><td>71.31</td></tr><tr><td></td><td>Base</td><td>94.38</td><td>67.98</td><td>84.67</td><td>25.37</td><td>62.03</td><td>52.65</td><td>88.57</td><td>78.72</td><td>54.20</td><td>64.50</td><td>71.65</td><td>67.70</td></tr><tr><td>R</td><td>Novel</td><td>88.03</td><td>43.88</td><td>47.40</td><td>16.68</td><td>62.86</td><td>45.07</td><td>51.89</td><td>83.71</td><td>40.95</td><td>58.50</td><td>56.50</td><td>54.13</td></tr><tr><td></td><td>HM</td><td>91.09</td><td>53.33</td><td>60.78</td><td>20.12</td><td>62.44</td><td>48.57</td><td>65.44</td><td>81.13</td><td>46.65</td><td>61.35</td><td>63.18</td><td>60.16</td></tr><tr><td></td><td>Base</td><td>96.26</td><td>73.50</td><td>94.29</td><td>33.97</td><td>72.59</td><td>62.11</td><td>94.97</td><td>89.10</td><td>71.34</td><td>73.24</td><td>79.63</td><td>76.45</td></tr><tr><td></td><td>Novel</td><td>89.30</td><td>46.62</td><td>41.03</td><td>21.30</td><td>74.55</td><td>46.00</td><td>53.97</td><td>91.50</td><td>49.94</td><td>64.65</td><td>62.52</td><td>58.31</td></tr><tr><td></td><td>HM</td><td>92.65</td><td>57.05</td><td>57.17</td><td>26.18</td><td>73.56</td><td>52.85</td><td>68.83</td><td>90.28</td><td>58.75</td><td>68.67</td><td>70.04</td><td>66.16</td></tr><tr><td>2</td><td>Base</td><td>93.35</td><td>64.35</td><td>86.24</td><td>24.19</td><td>56.18</td><td>48.90</td><td>89.36</td><td>76.50</td><td>53.30</td><td>61.17</td><td>70.94</td><td>65.86</td></tr><tr><td>R</td><td>Novel</td><td>85.15</td><td>38.04</td><td>34.46</td><td>14.04</td><td>57.21</td><td>36.04</td><td>41.91</td><td>80.76</td><td>34.07</td><td>52.12</td><td>51.22</td><td>47.73</td></tr><tr><td></td><td>HM</td><td>89.06</td><td>47.82</td><td>49.24</td><td>17.77</td><td>56.69</td><td>41.50</td><td>57.06</td><td>78.57</td><td>41.57</td><td>56.28</td><td>59.49</td><td>55.35</td></tr></table>

Table 14: Performance of CoAPT using the IVLAP scheme on 11 datasets under the base-to-novel benchmark.   

<table><tr><td rowspan="2">M</td><td rowspan="2"></td><td rowspan="2">Metric</td><td rowspan="2">Caltech101</td><td rowspan="2">DTD</td><td rowspan="2"></td><td rowspan="2">EuroSAT FGVCAircraft Food101</td><td rowspan="2"></td><td rowspan="2">ImageNet</td><td rowspan="2">Flowers101</td><td rowspan="2"></td><td rowspan="2">OxfordPets StanfordCars</td><td rowspan="2">SUN397</td><td rowspan="2">UCF101</td><td rowspan="2">Average</td></tr><tr><td></td></tr><tr><td></td><td></td><td>Base</td><td>96.71</td><td>76.74</td><td>92.88</td><td>33.91</td><td>78.32</td><td>66.30</td><td>95.73</td><td>90.86</td><td>72.24</td><td>76.98</td><td>81.13</td><td>78.34</td></tr><tr><td rowspan="4">M</td><td>Novel</td><td></td><td>92.25</td><td>53.02</td><td>52.41</td><td>26.75</td><td>80.29</td><td>55.20</td><td>64.18</td><td>94.13</td><td>58.83</td><td>70.31</td><td>67.01</td><td>64.94</td></tr><tr><td></td><td>HM</td><td>94.43</td><td>62.71</td><td>67.01</td><td>29.91</td><td>79.29</td><td>60.24</td><td>76.84</td><td>92.46</td><td>64.85</td><td>73.50</td><td>73.40</td><td>71.02</td></tr><tr><td>Base</td><td></td><td>94.58</td><td>67.71</td><td>84.88</td><td>24.67</td><td>61.22</td><td>52.57</td><td>89.55</td><td>79.11</td><td>53.87</td><td>64.63</td><td>69.39</td><td>67.47</td></tr><tr><td>3</td><td>Novel HM</td><td>88.10 91.22</td><td>43.12 52.68</td><td>43.87 57.85</td><td>16.74 19.94</td><td>63.57 62.37</td><td>44.94 48.46</td><td>52.06 65.84</td><td>83.84 81.40</td><td>41.67 46.99</td><td>58.73 61.54</td><td>53.92 60.69</td><td>53.69 59.79</td></tr></table>

JVLAP shows more significant advantages in modeling cross-modal robustness. By jointly optimizing adversarial features of both vision and language branches within a unified framework, it more effectively captures the synergistic variations between the two modalities in the latent space, thereby enhancing the consistency and stability of modality alignment. This joint optimization not only mitigates performance bias caused by asymmetrical perturbation sensitivity between modalities but also preserves semantic consistency during adversarial training. Consequently, it significantly enhances the generalization capability of the model on novel categories, zero-shot recognition, and out-of-distribution scenarios.

# D Impact Statement

This work aims to support progress in robust machine learning by improving the resilience of visionlanguage models against adversarial threats. Although we do not anticipate any immediate negative consequences, it is important to remain aware of potential misuse in security-critical domains. One key outcome of our approach is the ability to preserve robustness with low-cost model adjustments, which offers practical value for time-sensitive applications on mobile and resource-limited devices. The techniques introduced here may contribute to safer and more dependable deployment of AI systems in real-world environments, particularly in areas like intelligent sensing and mobile security.

# E Reproducibility

To support reproducibility, we have included the anonymized source code in the supplementary materials for the review process. If the paper is accepted, we will release the complete codebase to the public.

# F Limitations

This work primarily investigates adversarial robustness against image-level perturbations, while multi-modal adversarial attacks that simultaneously affect both vision and language inputs remain underexplored. The current framework assumes that adversarial noise originates solely from the visual modality, which limits its applicability in scenarios involving adversarial manipulations in textual inputs. Although the proposed latent space reconstruction method shows strong generalization in experiments, its specific impact on generalization behavior and the theoretical analysis for its superiority over other techniques remain unexplained. The influence of latent space structure and distribution on model robustness and generalization requires further theoretical exploration. We leave these limitations as essential directions for future investigation.