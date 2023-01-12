---
layout: post
title:  "Diffusion Survey"
date:   2023-01-11 10:30:31 +0900
category: Blog
tags: [Deep Learning, Computer Vision, Diffusion, Survey, Multimodal]
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
To bring back the two importance functions (the reverse diffusion process $$ p_{\theta} $$, x_t, and the mean function $$ \mu_{\theta} $$)  
$$ p_{\theta}(x_{t-1}|x_{t}) = \mathcal{N}(x_{t-1};\mu_{\theta}(x_{t},t),\sum_{\theta}(x_{t},t)) $$  
$$x_{t} = \sqrt{\bar{a}_{t}}x_{0}+\sqrt{1-\bar{a}_{t}}\epsilon$$     
$$ \mu_{\theta} $$ predicts $$ \bar{\mu}_{t} = \frac{1}{\sqrt{a}_{t}}(x_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{a}_{t}}}\epsilon_{t}) $$  
Since $$x_{t}$$ is available during training time, the gaussian noise term can be reparametrized to predict $$\epsilon_{t}$$  
$$ \mu_{\theta}(x_{t},t) = \frac{1}{\sqrt{a}_{t}}(x_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{a}_{t}}}\epsilon_{\theta}(x_{t},t)) $$  
$$L_{t}$$ is parameterized to minimize the difference from $$\bar{\mu}$$:  
$$L_{t}$$   
$$ = \mathbb{E}_{t~[1,T],x_{0},\epsilon_{t}}[\left\lVert \epsilon_{t} - \epsilon_{\theta}(x_{t},t) \right\rVert^2] $$

So in essence, the model optimizes to predict the $$ \epsilon_{t} $$ and the loss function can be simplified as:  
$$ \left\lVert \epsilon_{t} - \epsilon_{\theta}(x_{t},t) \right\rVert^2 $$


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
1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  
Predicts mean of noise sampled from gaussian distribution.  
Noise $$B_{t}$$ scheduling  

2. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)  

3. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)  

4. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)  

5. [ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2108.02938)  

6. [Semantic Image Synthesis via Diffusion Models](https://arxiv.org/abs/2207.00050)  

7. [MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation](https://arxiv.org/abs/2205.09853)  

8. [Diffused Heads: Diffusion Models Beat GANs on Talking-Face Generation](https://arxiv.org/abs/2301.03396)  

9. [Cold Diffusion: Inverting Arbitrary ImageTransforms Without Noise](https://arxiv.org/abs/2208.09392)  

10. [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)  

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

