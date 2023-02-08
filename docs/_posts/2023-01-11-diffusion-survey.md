---
layout: post
title:  "Diffusion Survey"
date:   2023-01-11 10:30:31 +0900
category: Blog
tags: [Deep Learning, Computer Vision, Diffusion, Survey, Generative Model, Multimodal]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Diffusion Survey

## 1. Diffusion Introduction
![Diffusion](https://www.assemblyai.com/blog/content/images/2022/05/image-10.png)  
Process that can generate images from noise ([source](https://arxiv.org/pdf/2006.11239.pdf)).  

![Diffusion Process](https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image-1.png)  
Training involves the forward process $$q(x_{t}|x_{t-1})$$ of adding noise to $$x_{0}$$ and a backword process $$p_{\theta}(x_{t-1}|x_{t})$$ to predict the noise from the previous step.  

## 2. Background

### 2-1. Forward Process
Forward process $$q$$ is a Markov Chain that continuously adds more noise to the original imgae $$x_{0}$$.  
$$q(x_{t}|x_{t-1}) = \mathcal{N}(x_{t};\sqrt{1-\beta_{t}}x_{t-1},\beta_{t}I)$$  

$$\beta_{t}$$ is the variance schedule which continuosuly increases  
$$ 0 < \beta_{1} < \beta_{2} < ... < \beta_{T} < 1 $$  
Which results in $$x_{t}$$ getting closer to mean 0 and deviation of 1 from mean $$x_{0}$$ deviation 0  

So at each step an image can be drawn as:  
$$ x_{t} = \sqrt{1-\beta_{t}}x_{t-1} + \sqrt{\beta_{t}}\epsilon $$  
Where $$\epsilon \sim \mathcal{N}(0,I)$$

The variance schedule is not constant and can be set manually based on the domain distribution and there are further works on this topic.  

### 2-2. Parametrization
A neural network, $$ p_{\theta}(x_{t-1}|x_{t}) $$ with
$$\theta$$ as the parameters is learned with gradient descent.  

$$ p_{\theta}(x_{t-1}|x_{t}) = \mathcal{N}(x_{t-1};\mu_{\theta}(x_{t},t),\sum_{\theta}(x_{t},t)) $$  
Where mean and variance are respectively parametrized by $$\mu_{\theta}, and \sum_{\theta}$$

### 2-3. Reparametrization Trick
Since the sum of Gaussians is also Gussian, $$q(x_{t}|x_{0})$$ can be calculated as follows:  
$$ q(x_{t}|x_{0}) = \mathcal{N}(x_{t};\sqrt{\bar{a}_{t}}x_{0},(1-\bar{a}_{t})I)$$  
and  
$$x_{t} = \sqrt{\bar{a}_{t}}x_{0}+\sqrt{1-\bar{a}_{t}}\epsilon$$  
Where $$ a_{t} := 1-\beta_{t} $$ and $$ \bar{a}_{t} := \prod^{t}_{s=1}a_{s} $$  
  
So opposed to $$\beta_{t}$$, $$a_{t}$$ continously decreases in value:  
$$ 1 > a_{1} > a_{2} > ... > a_{T} > 0$$

With the definition of $$x_{t}$$ above, $$x_{0}$$ can be represented as follows:  
$$ x_{0} = \frac{1}{\sqrt{\bar{a}_{t}}}(x_{t}-\sqrt{1-\bar{a}_{t}}\epsilon_{t}) $$  
and the mean function can be written as follows ([source](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice:~:text=tx0-,Thanks,-to%20the%20nice)):  
$$ \mu_{\theta}(x_{t},t) = \frac{1}{\sqrt{a}_{t}}(x_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{a}_{t}}}\epsilon_{\theta}(x_{t},t)) $$  

### 2-4. Objective Function
**A. Variational Lower Bound (ELBO)**  
$$ L_{CE} = -log p_{\theta}(x_{0}) \leq \mathbb{E}_{q}[log\frac{q(x_{1:T}|x_{0})}{p_{\theta}(x_{0:T})}] = L_{VLB}$$ (<a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice:~:text=x0)-,It,-is%20also%20straightforward">Explanation</a>)  
Which is equivalent to the KL divergence between $$q(x_{1:T}|x_{0})$$ and $$p_{\theta}(x_{0:T})$$  
> The KL divergence between two Gaussian distributions can be expressed as the **L2-loss between their means**  
> $$ D_{KL}(p || q) = log\frac{\sigma_{2}}{\sigma_{1}} + \frac{\sigma^{2}_{1}+(\mu_{1}-\mu_{2})^{2}}{2\sigma^{2}_2} - \frac{1}{2} $$  

Thus the model can be optimized with $$L_{VLB} $$ instead of $$L_{CE}$$  

#### B. Complete Objective Function
The objective can be further rewritten with several terms:  
$$ L_{VLB}  $$  
$$= L_{T} + L_{T-1} + ... + L_{0}  $$  
$$= L_{T} + L_{t} + L_{0} $$  
where  
$$ L_{T} = D_{KL}(q(x_{T}|x_{0})||p_{\theta}(x_{T})) $$  
$$ L_{t} = D_{KL}(q(x_{t}|x_{t+1},x_{0})||p_{\theta}(x_{t}|x_{t+1})) $$ for $$1 \leq t \leq T -1 $$  
$$ L_{0} = -log p_{\theta}(x_{0}|x_{1}) $$

KL Divergences between two Gaussian distributions can be computed in closed form as mentioned above. And $$L_{T}$$ can be ignored during training since q has no learnable prameters and $$x_{T}$$ is a Gaussian noise.   

#### C. Final Objective Function $$L_{t}$$
To bring back some important functions (the reverse diffusion process $$ p_{\theta} $$, x_t, and the mean function $$ \mu_{\theta} $$)  
$$ p_{\theta}(x_{t-1}|x_{t}) = \mathcal{N}(x_{t-1};\mu_{\theta}(x_{t},t),\sum_{\theta}(x_{t},t)) $$  
$$x_{t} = \sqrt{\bar{a}_{t}}x_{0}+\sqrt{1-\bar{a}_{t}}\epsilon$$     
$$ \mu_{\theta} $$ predicts $$ \bar{\mu}_{t} = \frac{1}{\sqrt{a}_{t}}(x_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{a}_{t}}}\epsilon_{t}) $$  
Since $$x_{t}$$ is available during training time, function can be reparametrized to predict $$\epsilon_{t}$$  
$$ \mu_{\theta}(x_{t},t) = \frac{1}{\sqrt{a}_{t}}(x_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{a}_{t}}}\epsilon_{\theta}(x_{t},t)) $$  
$$L_{t}$$ is parameterized to minimize the difference from $$\bar{\mu}$$:  
$$L_{simple}$$   
$$ = \mathbb{E}_{t,x_{0},\epsilon_{t}}[\left\lVert \epsilon_{t} - \epsilon_{\theta}(x_{t},t) \right\rVert^2] $$

In essence, the model optimizes to predict Gaussian noise $$ \epsilon_{t} $$ and the loss function can be simply viewed as:  
$$ \left\lVert \epsilon_{t} - \epsilon_{\theta}(x_{t},t) \right\rVert^2 $$

The model can be trained as follows:  
![Diffusion Training](/assets/posts/blog/2.diffusion_survey/training.png "Training")  

Markov Chain
ELBO
Reparametrization Trick
Langevin Dynamics  
Variational Autoencoder  
Variational Lower bound
U-Net  
Gaussian Distribution, KL-Divergence and Mean Squared Loss  

## 3. Survey
### 3-1. Summary of Papers
#### 1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  
- UC Berkeley, NeurIPS 2020, Jun 19 2020  

1. Proposed diffusion models, parameterized Markov chain trained using variational inference to produce samples matching the data.  
2. First to demonstrate that diffusion models are cabaple of generating high quality samples.  
3. Implementation Details  
- Process variances $$\beta_{t}$$ could be learnable parameters but are fixed to constants, thus posterior $$q$$ has no learnable parameters so $$L_{T}$$ is ignored
- During the reverse process $$p_{\theta}$$ deviation is set to $$\sum_{\theta}(x_{t},t) = \sigma^{2}_{t}I$$ where $$\sigma_{t}^{2} = \beta_{t}$$
- $$L_{t-1}$$ can be written as the l2 loss of the predicted mean
4. Experiments  
- T = 1000, forward process variances are set to constants $$\beta_{1}=10^{-4}$$ to $$\beta_{T}=0.02$$
- Used U-Net backbone with group normalization. Parameteres are shared across time. Positional encoding and self-attention are used. 
- FID score of 3.17, better sample quality than most models in the literature


#### 2. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)  
- OpenAI, ICLR 2021 Reject , Feb 18 2021  

1. Core Contributions
    - Noise cosine schedule
    - Learning sigma (variance), reduces the number of diffusion steps required for training
2. Weaknesses
    - Similar FIDs as other approaches
    - log-likelihood is not paramount for generative tasks
3. Results
    - On CIFAR-10, improved negative log-likelihood from 3.99 (Basedline DDPM) to 2.94
4. Details
    - Hybrid objective outperforms in log-likelihood than optimizing for log-likelihood directly
    - Learn variance  
    > $$\sum_{\theta}(x_{t},t)=exp(\upsilon log \beta_{t}+(1-\upsilon)log \tilde{\beta}_{t})$$
    - Hybrid loss to incorporate $$\sum_{\theta}$$
    > $$L_{hypbrid} = L_{simple} + \lambda L_{vlb}$$
    - Noise schedule (cosine)
    > $$ \bar{\alpha}_{t} = \frac{f(t)}{f(0)}, f(t) = cos(\frac{t/T+s}{1+s} \cdot \frac{\pi}{2})^{2} $$  
    ![Cosine Noise Schedule](/assets/posts/blog/2.diffusion_survey/2-noise_schedule.png "Cosine Noise Schedule")  
    For reference $$\beta_{t} = 1 - \frac{\bar{\alpha}_{t}}{\bar{\alpha}_{t-1}}$$
    - Importance of log-likelihood
        - Generally believed that optimizing log-likelihood forces generative models to capture all of the modes the data distribution
        - Small improvements in log-likelihood can have a dramatic impact on sample quality and learnt feature representations.
5. Opinion
    - Importance of optimizing for log-likelihood?


#### 3. [Variational Diffusion Models](https://arxiv.org/abs/2107.00630)  
- Google Research, NeurIPS 2021, Jul 1 2021

1. Contributions
    - Introduces a family of diffusion based generative models with SOTA performance
    - Optimizes for noise schedule
    - VLB simplified in terms of the signal-to-noise ratio. 
    - State-of-the-art likelihoods on image density estimation benchmarks with often significantly faster optimization
    - Equivalence between models in the literature
    - Continuous-time VLB is invariant to the noise schedule, except for the signal-to-noise ratio at its endpoints
    - This enables the model to learn a noise schedule that minimizes the variance of the resulting VLB estimator, leading to faster optimization

#### 4. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)  
- Stanford University, ICLR 2021, Oct 6 2020

1. Contributions
    - New forward process (Non-Markovian)
    - Increases speed of reverse process (Objective is the same)
    - 10x, 50x faster to produce high quality samples

#### 5. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)  
- Runway ML, CVPR 2022, Dec 20 2021

1. Contributions
    - Faster high-resolution image synthesis with diffusion models by using the latent space instead of the pixel space.
    - Trains an autoencoder in the latent space, which is perceptually equivalent to the image space. It is trained with perceptual loss and patch-based adversarial objecgtive which enforces local realism. 
    - The encoder downsamples the image by the factor f, and use regularization techniques (KL) on the latent space. 
    - This learned latent space can be used for other downstream tasks. 
    - The 2 dimensional latent space can focus on the semantically important parts of the data more efficiently.

#### 6. [ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2108.02938)  
- SNU, Samsung, ICCV 2021 Oral, Aug 6 2021

1. Contributions
    - Propose Iterative Latent Variable Refinement (ILVR) to guide the generative process in DDPM.
    - Sample images from conditional distribution $$ p_{\theta}(x_{0:T} \mid c) = p(x_{T})\Pi^{T}_{t=1}p_{\theta}(x_{t-1} \mid x_{t}, c) $$ where c is the condition
    - $$ \phi_{N}(\cdot) $$ denotes a low-pass filtering operation, a sequence of downsampling and upsampling. Condition c ensures that the donwampled image $$ \phi_{N}(x_{0}) $$ equals to $$ \phi_{N}(y) $$


#### 7. [Semantic Image Synthesis via Diffusion Models](https://arxiv.org/abs/2207.00050)  
- Microsoft, Jun 30, 2022

1. Contributions
    - First to attempt diffusion modeles for semantic image synthesis. 
    - The noisy image is fed into the encoder of the denoising network.
    - The semantic layout is embedded into the the decoder of the denoising network by multi-layer spatially-adaptive normalization operators.
    - This highly improves the quality and semantic correlation of generated images.
    - Model is fine-tuned by randomly removing the semantic mask input.
    - Sampled based on both with and without the semantic mask.
    - With interpolation between the two, the model achieves higher fidelity and stronger correlation with the semantic mask input.


#### 8. [MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation](https://arxiv.org/abs/2205.09853)  
- Mila, NeurIPS 2022, May 19, 2022

1. Contributions
    - Model is trained by randonly and independenly masking all of the past or future frames. 

#### 9. [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

#### 10. [Diffused Heads: Diffusion Models Beat GANs on Talking-Face Generation](https://arxiv.org/abs/2301.03396)  

#### 11. [Cold Diffusion: Inverting Arbitrary ImageTransforms Without Noise](https://arxiv.org/abs/2208.09392)  

#### 12. [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)  

#### 13. [DiffTalk: Crafting Diffusion Models for Generalized Talking Head Synthesis]()

#### 14. [Speech Driven Video Editing via an Audio-Conditioned Diffusion Model]()

### 3-2. Insights
- Common features
- Direction
- Flaws, potential improvements

## Open Source

## Demos

## Code
Colab Notebook (PyTorch)

## Research Agendas

## References
1. [AssemplyAI](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)  
2. [Annotated Diffusion (HuggingFace)](https://huggingface.co/blog/annotated-diffusion)  
3. [Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)  

