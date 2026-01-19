# Agentic Reader Result
**Paper ID:** MZg06yaIW9
**Issue File:** MZg06yaIW9_0.md
**Status:** success
**Timestamp:** 2026-01-19T06:52:07.402256
**Model:** openai:gpt-5-mini
**Max Iterations:** 10

**Issue Content:**
## LINE 117-119

one can simply model

$$
q _ { \mathrm { a l i g n e d } } \left( y | X _ { \leq t } , Y _ { < t } \right) \approx \mathbb { P } \left[ Y _ { t } = y | X _ { \leq t } , Y _ { < t } \right] .
$$

117 Given $X \sim p ( X )$ , we sample auto-regressively from Eq. (5), with a streaming context $X$ ,

$$
\tilde { Y } _ { 1 } \sim q _ { \mathrm { a l i g n e d } } \left( \tilde { Y } _ { 1 } | X _ { 1 } \right) , \qquad \tilde { Y } _ { t } \sim q _ { \mathrm { a l i g n e d } } \left( \tilde { Y } _ { 1 } | X _ { \le t } , \tilde { Y } _ { < t } \right) .
$$

![](images/60d92ecf79d427d98419baa6566730737c8a49ddc60a4d265c04ef2690d8f3d4.jpg)  
Figure 2: DSM Architecture. Transformer is fed with the streaming input $X _ { t }$ . After a delay $\tau$ , a sampler is fed with the output of the backbone samples $\tilde { Y } _ { t }$ . At the next step, the backbone receives both the sampled value and next streaming input, whose embeddings are summed.

![](images/16cc9d4a3a186d1c691c319f3b27d96392328c97172ae27f12f7b644f40f86e9.jpg)  
Figure 3: DSM-TTS inference. The input words “Hello, world” are tokenized. Until the model action stream outputs a WORD, it is fed with PAD. Then the first word’s tokens are fed, including a look-ahead text stream. Once a delay $\tau = 5$ has accumulated, the model also outputs the audio.

We would want that given 118 $X \sim p ( X )$ , then $( X , { \tilde { Y } } ) \sim ( X , Y )$ , so that in particular $\mathbb { P } \left[ \tilde { Y } | X \right] \approx$ 119 $\mathbb { P } \left[ Y | X \right]$ . However this needs not be the case unless certain conditions are met.

20 The importance of causality. In particular, for $( X , { \tilde { Y } } ) \sim ( X , Y )$ to be true, $Y _ { > t }$ must be independent   
of $X _ { > t }$ , knowing $X _ { \leq t }$ . To realize that, one can look at a simple counter-example taking $X _ { t } \sim$   
$B ( 0 . 5 )$ independent


## Agentic Reader Analysis

Yes — there is a mathematical/formula typo in the paper.

Explanation and citation
- The paper defines an aligned sampler q_{aligned} and then gives the autoregressive sampling recipe. In the paragraph beginning "Given X ~ p(X), we sample auto-regressively from Eq. (5)..." (lines indicated in the reviewer snippet, see the displayed equations in the paper), the two sampling lines are written as

  \tilde{Y}_1 \sim q_{\mathrm{aligned}}(\tilde{Y}_1 \mid X_1), \qquad
  \tilde{Y}_t \sim q_{\mathrm{aligned}}(\tilde{Y}_1 \mid X_{\le t}, \tilde{Y}_{<t}).

  (This appears in the paper around the paragraph quoted in the review — the displayed sampling equations in the "Given X ~ p(X)" paragraph, lines 117–119 in the snippet.)

- The second displayed formula is inconsistent: its left-hand side is \tilde{Y}_t but the right-hand side conditions the distribution on \tilde{Y}_1 (i.e. q_{\mathrm{aligned}}(\tilde{Y}_1 | ...)). This is almost certainly a typographical/indexing error. The intended autoregressive update should be

  \tilde{Y}_t \sim q_{\mathrm{aligned}}(\tilde{Y}_t \mid X_{\le t}, \tilde{Y}_{<t}).

- (For context, the earlier displayed definition q_{\mathrm{aligned}}(y \mid X_{\le t},Y_{<t}) \approx P[Y_t = y \mid X_{\le t},Y_{<t}] is fine; the sampling line is where the index mismatch occurs.)

Suggested fix: replace the erroneous \tilde{Y}_1 on the right-hand side of the second sampling equation with \tilde{Y}_t so the two sides match.

Otherwise (separate from this typo) the reviewer also discusses a modeling/causality concern (about (X,\tilde{Y}) matching (X,Y)), but that is a conceptual modeling point, not a formula typo.