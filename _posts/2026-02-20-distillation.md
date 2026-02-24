---
layout: post
title: Modern Diffusion Distillation Overview
date: 2026-02-20 01:24:00
description: Recent developments of diffusion distillation techniques
tags: diffusion survey
categories: research-survey
---

# Flow Matching to Flow Maps to Distillation: A Deep Dive

# Table of Content
1. [Foundations](#1-foundations-flow-matching-vs-flow-maps)
    1. Diffusion
    2. Flow Matching
    3. Rectified Flow
    4. FlowMap
    5. Consistency Model
2. [MeanFlow Family](#2-meanflow)
    1. MeanFlow
    2. iMeanFlow
    3. AlphaFlow
    4. Improved MeanFlow
    5. Decoupled MeanFlow
3. [Flow Map](#3-freeflowmap)
    1. Data Free: FreeFlowMap
    2. Meta FlowMap
    3. TVM
4. [Score Distillation]()
    1. VSD
    2. DMD
    3. SiD
    4. Adaptive Matching Distillation
5. [Adversarial]()
    1. DiffRatio
    2. APT
6. [Video Generation]()
    1. CausVid
    2. Self-forcing
    3. TMD
7. [New Domains]()
    1. Drifting
    2. JIT
    3. PixelFlow
    4. LatentForcing
8. [Manifold]()
    1. Riemmian Manifold
    2. Optimal Transport

# Overview
Recent generative modeling utilize and develop upon flow maps and jvp based distillation techniques to reduce the number of function evaluations during inference. We focus on the Meanflow family, score distillation methods, and its applications in video generation. 

---

# 1. Foundations
## 1.1 Diffusion
Diffusion models define a **forward noising process** that gradually corrupts data into noise, and a **reverse process** that learns to reconstruct data from noise. The main reason diffusion models became dominant is that they are stable and high-quality, but the tradeoff is **slow iterative sampling**.

### 1.1.1 Forward process (discrete DDPM view)

In DDPM, the forward process is a Markov chain:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I\right),
$$

where $\beta_t \in (0,1)$ is a variance schedule.

A key closed form is:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\right),
$$

with

$$
\alpha_t = 1-\beta_t,\qquad \bar\alpha_t=\prod_{s=1}^t \alpha_s.
$$

So we can sample $x_t$ directly as:

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon \sim \mathcal{N}(0,I).
$$

This is the standard “data + Gaussian noise” interpolation used in many diffusion derivations.

### 1.1.2 Reverse process and denoising objective

The generative model learns the reverse transitions

$$
p_\theta(x_{t-1}\mid x_t),
$$

which are parameterized via a neural network (predicting noise, $x_0$, or velocity depending on parameterization).

The most common training objective is noise prediction:

$$
\mathcal{L}_{\text{simple}}(\theta)
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|\epsilon - \epsilon_\theta(x_t,t)\right\|^2
\right].
$$

This objective is simple and works extremely well, but inference still requires many reverse denoising steps.

### 1.1.3 Continuous-time diffusion (SDE view)

A continuous-time diffusion can be written as an SDE:

$$
dx = f(x,t)\,dt + g(t)\,dW_t,
$$

where $f$ is drift, $g$ is diffusion scale, and $W_t$ is a Wiener process.

The reverse-time generative dynamics also form an SDE involving the score:

$$
\nabla_x \log p_t(x).
$$

This is the bridge to **score-based generative modeling** and continuous-time transport formulations.

### 1.1.4 Probability flow ODE (deterministic counterpart)

Every diffusion SDE has an associated deterministic **probability flow ODE** that shares the same marginals $p_t$:

$$
\frac{dx}{dt}
=
f(x,t)
-
\frac{1}{2}g(t)^2 \nabla_x \log p_t(x).
$$

This is huge conceptually because it turns diffusion sampling into solving an ODE, which directly connects to:

- continuous normalizing flows,
- flow matching,
- rectified flow,
- and later flow-map distillation methods.

So diffusion is not just “denoising noise,” it is also a **continuous transport process** in disguise.

### 1.1.5 Why diffusion motivates distillation

Diffusion teachers are strong but slow because generation requires many function evaluations (NFEs). Distillation methods aim to compress this long trajectory into:

- **few-step samplers** (e.g. 2–8 steps),
- or even **one-step generators**,

while preserving the teacher’s learned transport geometry.

This is exactly why Chapter 1 naturally progresses from **Diffusion $\to$ Flow Matching $\to$ Flow Maps / Consistency / Distillation**.

## 1.2 Flow Matching
![Flow Matching](/assets/img/blogs/1_distillation/flowmatching.png)

*Figure 1. Flow Matching. from Sabour, Fidler, and Kreis (2025),* Align Your Flow: Scaling Continuous-Time Flow Map Distillation *(arXiv:2506.14603).*  

Flow Matching (FM) reframes generative modeling as directly learning a **time-dependent velocity field** that transports a simple source distribution (usually Gaussian noise) to the data distribution.

### 1.2.1 Core idea: learn the instantaneous velocity

Instead of learning reverse denoising conditionals, FM learns a vector field

$$
u_\theta(x,t)
$$

such that samples evolve via the ODE

$$
\frac{dx}{dt} = u_\theta(x,t),\qquad t\in[0,1].
$$

If this ODE is integrated from source noise at $t=0$ to $t=1$, the final samples should follow the data distribution.

### 1.2.2 Conditional path and target velocity

FM is usually trained by defining a conditional interpolation path between paired endpoints $(x_0,x_1)$:

$$
x_t = \psi_t(x_0,x_1).
$$

For simple linear interpolation:

$$
x_t = (1-t)x_0 + tx_1.
$$

The target velocity along this path is:

$$
\dot{x}_t = \frac{d}{dt}\psi_t(x_0,x_1).
$$

For the linear path, this becomes:

$$
\dot{x}_t = x_1 - x_0.
$$

The model is trained to match this conditional velocity in expectation:

$$
\mathcal{L}_{\text{FM}}(\theta)
=
\mathbb{E}_{t,x_0,x_1}
\left[
\left\|u_\theta(x_t,t)-\dot{x}_t\right\|^2
\right].
$$

So FM is “supervised vector field learning” on a chosen path family.

### 1.2.3 Why FM is attractive

Compared to score/diffusion training, FM gives a very clean ODE-learning objective and avoids explicit score estimation. It is especially natural when you want to reason about:

- transport geometry,
- ODE trajectories,
- and later finite-time maps (flow maps).

### 1.2.4 Limitation: local field, expensive sampling

FM learns a **local tangent** $u_\theta(x,t)$, not a finite-time jump. That means sampling still requires numerical integration:

$$
x_{t+\Delta t} \approx x_t + \Delta t\,u_\theta(x_t,t)
$$

(or higher-order solvers like Heun).

If trajectories are curved, discretization error accumulates, so many NFEs are needed. This is the main bottleneck that motivates:

- Rectified Flow (straighter trajectories),
- Flow Maps (direct time-to-time transport),
- and distillation methods (few-step or one-step generation).

## 1.3 Rectified Flow

Rectified Flow (RF) keeps the ODE/transport framing of FM, but explicitly pushes the learned trajectories to become **straighter**, which makes them much easier to sample with few steps.

### 1.3.1 Motivation: straight trajectories are cheap

If a trajectory is highly curved, Euler updates need many small steps. If a trajectory is nearly straight, even a coarse solver can track it accurately.

So RF is basically a geometry-aware fix to the FM sampling bottleneck.

### 1.3.2 Linear interpolation path and velocity target

A standard RF path is the same linear interpolation:

$$
x_t = (1-t)x_0 + tx_1,
$$

with instantaneous derivative

$$
\frac{dx_t}{dt} = x_1 - x_0.
$$

RF trains a velocity field to match this transport direction along the path:

$$
v_\theta(x_t,t) \approx x_1 - x_0.
$$

A common training objective is:

$$
\mathcal{L}_{\text{RF}}(\theta)
=
\mathbb{E}_{t,x_0,x_1}
\left[
\left\|v_\theta(x_t,t)-(x_1-x_0)\right\|^2
\right].
$$

This looks similar to FM, but the interpretation is sharper: RF cares about learning a transport field whose induced trajectories are easy to discretize.

### 1.3.3 Reflow (iterative rectification)

A major practical idea in RF is **reflow**:

1. Train an initial transport field.
2. Sample trajectories from the model.
3. Use those trajectories (or endpoint couplings) to retrain a straighter field.
4. Repeat.

Each round reduces curvature and improves few-step generation. This is why RF is often viewed as a bridge between classical diffusion/FM and modern one-step distillation.

### 1.3.4 Why RF matters for the rest of this blog

RF is the clean conceptual bridge to flow-map methods because it shifts the focus from “match local field” to “shape trajectories for fast transport.” Once you think this way, the next obvious step is:

> Why learn only the local tangent at all?  
> Why not learn the **finite-time map** directly?

That is exactly the flow-map perspective.

## 1.4 Flow Map
![Flow Map](/assets/img/blogs/1_distillation/flowmap.png)

*Figure 2. Flow Map. from Sabour, Fidler, and Kreis (2025),* Align Your Flow: Scaling Continuous-Time Flow Map Distillation *(arXiv:2506.14603).*  

Flow-map methods move beyond local vector fields and directly learn a **time-to-time transport operator**.

### 1.4.1 From vector field to finite-time map

Given a velocity field $u(x,t)$ and its ODE

$$
\frac{dx}{dt}=u(x,t),
$$

the associated flow map $\phi_u$ sends a state from time $t$ to time $s$:

$$
\phi_u(x_t,t,s)=x_s.
$$

So instead of learning only the local tangent $u(x,t)$, we learn the finite-time update:

$$
x_s \approx \phi_\theta(x_t,t,s).
$$

This is much more aligned with few-step sampling, because a single model evaluation can move across a large time interval.

### 1.4.2 Why flow maps help distillation

A flow field gives infinitesimal updates; a flow map gives finite jumps.

That means flow maps are naturally suited for:

- **few-step sampling** (large $t\to s$ jumps),
- **teacher-student distillation** (student imitates teacher transitions),
- **self-distillation** (model supervises its own multistep consistency),
- and **data-free distillation** variants (matching dynamics without original data).

This is the key conceptual move behind MeanFlow-family and FreeFlowMap-family methods.

### 1.4.3 Semigroup / composition structure

Exact flow maps satisfy a composition rule (semigroup property):

$$
\phi(x_t,t,r)
=
\phi\!\left(\phi(x_t,t,s),\,s,\,r\right)
\qquad \text{for } t \le s \le r.
$$

This property is incredibly important because it gives a built-in consistency constraint across time triples. Many modern distillation methods exploit some version of this:

- explicit composition matching,
- consistency losses,
- JVP-based local constraints that imply finite-time consistency.

### 1.4.4 Relation to FM and RF

- **FM** learns $u(x,t)$ (local instantaneous velocity)
- **RF** improves trajectory geometry (straighter ODE paths)
- **Flow Map** learns $\phi(x_t,t,s)$ (finite-time transport)

So flow maps are not a totally different universe; they are the natural next abstraction after FM/RF if your goal is **fast generation**.


## 1.5 Consistency Models
Consistency Models (CMs) attack the same bottleneck from another angle: instead of learning a vector field or even an explicit flow map, they learn a **cross-time consistent predictor** that maps noisy states to a shared target representation (often an estimate of clean data).

This makes them one of the foundational one-step / few-step distillation paradigms.

### 1.5.1 Core consistency idea

Suppose $x_t$ and $x_s$ lie on the same teacher trajectory (or same underlying denoising path). A consistency model $f_\theta$ is trained so that:
$$
f_\theta(x_t,t) \approx f_\theta(x_s,s),
$$
after the appropriate scaling/parameterization.

In words: different noise levels along the same trajectory should produce the same final prediction.

This is a **cross-time agreement constraint**, not just a local derivative-matching objective.

### 1.5.2 Why this enables one-step generation

Because the model is trained to collapse trajectory points to a common target, we can often sample by evaluating the model once (or very few times) from a noisy input.

This directly targets inference speed, unlike standard diffusion training which optimizes denoising accuracy at every step but does not inherently optimize for low-NFE sampling.

### 1.5.3 Teacher-student consistency distillation

A common setup is:

1. Start with a strong diffusion/score teacher.
2. Generate paired states $(x_t, x_s)$ on teacher trajectories.
3. Train the student consistency model to agree across those states.

This makes consistency models a very important predecessor to later:

- flow-map distillation,
- self-distillation,
- and JVP-based transport-map objectives.

### 1.5.4 Conceptual relation to flow maps

Consistency models and flow-map methods are closely related in spirit:

- **Flow map view:** learn explicit transport $\phi(x_t,t,s)$
- **Consistency view:** learn a representation/prediction that is invariant (or aligned) across times on the same trajectory

Both replace purely local supervision with **cross-time structure**, which is exactly what you need for few-step and one-step generation.

### 1.5.5 sCM
### 1.5.6 rCM

---

# 2. MeanFlow Family
## 2.1 MeanFlow
#### 2.1.1 Average velocity instead of instantaneous velocity
![Mean Flow](/assets/img/blogs/1_distillation/meanflow.png)

*Figure 3. Mean Flow, with different target timestep $t$. from Geng et al. (2025),* Mean Flows for One-step Generative Modeling *(arXiv:2505.13447).*  

MeanFlow defines an **average velocity**

$$
u(z_t, r, t)
$$

over the interval $[r,t]$, so the displacement is:

$$
(t-r)u(z_t,r,t).
$$

#### 2.1.2 The MeanFlow Identity (the central derivation)

Start from the definition:

$$
(t-r)u(z_t,r,t) = \int_r^t v(z_\tau,\tau)\, d\tau
$$

Differentiate both sides with respect to \(t\) (holding \(r\) fixed). By product rule + FTC:

$$
u(z_t,r,t) + (t-r)\frac{d}{dt}u(z_t,r,t) = v(z_t,t)
$$

Rearranging leads to the **MeanFlow Identity**:

$$
u(z_t,r,t) = v(z_t,t) - (t-r)\frac{d}{dt}u(z_t,r,t)
$$

#### 2.1.3 Main Points

- The authors rewrite an intractable target (the average velocity integral) into a trainable target:
  - By first taking the derivative of both sides.
  - Instantaneous velocity $v$ is available from FM-style interpolation.
  - Total derivative term computed via **JVP**.
- No explicit consistency regularizer is imposed.
- The consistency-like structure falls out from the definition of average velocity.

#### 2.1.4 MeanFlow loss

Parameterize $u_\theta(z_t,r,t)$, and regress to the identity-induced target:

$$
\mathcal{L}_{\text{MeanFlow}}(\theta)
=
\mathbb{E}\left[
\left\|
u_\theta(z_t,r,t) - \operatorname{sg}(u_{\text{tgt}})
\right\|_2^2
\right]
$$

with

$$
u_{\text{tgt}}
=
v_t - (t-r)\left(v_t \,\partial_z u_\theta + \partial_t u_\theta\right)
$$

where the total derivative is implemented through a JVP along tangent $(v_t, 0, 1)$.

#### 2.1.5 Sampling

Once you learn the average velocity, sampling is:

$$
z_r = z_t - (t-r)u(z_t,r,t)
$$

For 1-step:

$$
z_0 = z_1 - u(z_1,0,1),\quad z_1 \sim p_{\text{prior}}
$$

---

## 2.2 Improved MeanFlow (iMF)

iMF addresses two practical issues in MeanFlow:
1. the self-referential target,
2. fixed-CFG training (bad for inference-time flexibility).

#### 2.2.1 MeanFlow as a v-loss

iMF rewrites the MeanFlow identity into a **velocity regression form**:

$$
v(z_t)
=
u(z_t,r,t) + (t-r)\frac{d}{dt}u(z_t,r,t).
$$

Then parameterize the RHS with $u_\theta$:

$$
V_\theta
=
u_\theta(z_t,r,t) + (t-r)\,\mathrm{JVP}_{\mathrm{sg}}(u_\theta; v_\theta),
$$

and train with a Flow-Matching-like loss

$$
\mathcal{L}_{\mathrm{iMF}}
=
\mathbb{E}\left[\left\|V_\theta - (\epsilon-x)\right\|_2^2\right].
$$

This is cleaner because the regression target is now the standard FM target $(\epsilon-x)$ rather than an apparent target constructed from $u_\theta$.

### 2.2.2 Flexible CFG as conditioning

Original MeanFlow supports CFG in 1-NFE, but with a **fixed guidance scale** chosen at training time.

iMF fixes this by making the guidance scale part of the conditioning:
- guidance scale $\omega$ becomes an input condition,
- the same model can be sampled with different CFG scales at inference.

That is a big deal because the optimal CFG scale shifts with model size, training progress, and NFE.

### 2.2.3 In-context conditioning

iMF also upgrades conditioning architecture:
- conditions include $(r,t)$, class label $c$, and guidance-related variables $\Omega$,
- each condition is represented with multiple learnable tokens,
- all condition tokens are concatenated with image latent tokens and processed by the Transformer.

This allows:
- support richer heterogeneous conditioning more naturally,
- remove **adaLN-zero**,
- cut params significantly (they report about **1/3 reduction** in a base model setting).

## 2.3 AlphaFlow

AlphaFlow is the paper that gives the most useful conceptual interpretation of MeanFlow training.

#### 2.3.1 Core insight: MeanFlow decomposes into two losses

AlphaFlow shows the MeanFlow objective can be algebraically decomposed into:
1. **Trajectory Flow Matching (TFM)**
2. **Trajectory Consistency (TC)**

The decomposition (up to a constant) is:

$$
\mathcal{L}_{\mathrm{MF}}
=
\underbrace{\mathbb{E}\left[\|u_\theta(z_t,r,t)-v_t\|_2^2\right]}_{\mathcal{L}_{\mathrm{TFM}}}
+
\underbrace{\mathbb{E}\left[2(t-r)\,u_\theta^\top \frac{d u_\theta^-}{dt}\right]}_{\mathcal{L}_{\mathrm{TC}}}
+ C.
$$

Interpretation:
- **TFM** says “fit the trajectory-local velocity target.”
- **TC** says “be self-consistent along the trajectory.”
- MeanFlow is effectively a **consistency-like model with extra trajectory FM supervision**.

#### 2.3.2 Why MeanFlow often needs lots of border-case FM samples

AlphaFlow also explains a weird empirical fact from MeanFlow:
- MeanFlow works best when many samples use the border case $r=t$ (which looks like vanilla FM).

Their analysis shows this is not just a hack:
- the gradients of TFM and trajectory consistency are often **negatively correlated**,
- so the extra FM-style supervision helps stabilize and speed up training.

#### 2.3.3 α-Flow loss: one objective that unifies TFM, Shortcut, MeanFlow

AlphaFlow defines a family of losses parameterized by $\alpha$:

$$
\mathcal{L}_\alpha(\theta)
=
\mathbb{E}_{t,r,z_t}
\left[
\alpha^{-1}
\left\|
u_\theta(z_t,r,t)
-
\left(\alpha \,\tilde v_{s,t} + (1-\alpha)\,u_{\theta^-}(z_s,r,s)\right)
\right\|_2^2
\right],
$$

where

$$
s = \alpha r + (1-\alpha)t
$$

is an intermediate time.

This unifies several training objectives:
- $\alpha=1$ gives **trajectory flow matching** (with suitable $\tilde v_{s,t}$),
- $\alpha=\tfrac{1}{2}$ recovers a **Shortcut-style** objective,
- $\alpha \to 0$ recovers the **MeanFlow gradient**.

That is the key conceptual win: AlphaFlow puts FM, Shortcut, and MeanFlow on one continuum.

#### 2.3.4 Curriculum

Because TFM and TC conflict early in training, AlphaFlow uses a curriculum:
- start more FM-like (larger $\alpha$),
- gradually anneal toward MeanFlow-like behavior (smaller $\alpha$).

This disentangles optimization and improves convergence.

#### 2.3.5 Takeaways

AlphaFlow is best understood as:
- a **theory paper for MeanFlow optimization** (decomposition + gradient conflict),
- plus a **practical training recipe** (curriculum over $\alpha$) that improves one-step/few-step quality.

## 2.4 Accelerating and Improving MeanFlow

Understanding and Improving Mean Flow’s (UAIMF) main point is simple and very practical: MeanFlow training is bottlenecked by **slow velocity formation** and **bad temporal-gap scheduling**, so they speed up both. They propose two complementary components:  
1. accelerate the velocity-learning part with standard diffusion training tricks (they use **MinSNR** or **DTD**), and  
2. add a **progressive weighting** over the MeanFlow loss so the model learns small-gap average velocities first, then gradually expands to larger gaps. 

### 2.4.1 Why MeanFlow trains slowly

UAIMF analyzes MeanFlow training through two coupled subproblems:

- learning instantaneous velocity (the easier FM-like part),
- learning average velocity over larger timestep gaps (the harder MeanFlow part).

Their empirical claim is that **rapid velocity formation helps MeanFlow converge much faster**, and that the temporal gap matters a lot: large-gap average-velocity learning is harder and should be delayed. This is why they combine velocity acceleration + progressive gap weighting. 

### 2.4.2 Component 1: Accelerate the velocity part (MinSNR / DTD)

UAIMF tests one method from each category:

- **MinSNR** as a loss-weighting acceleration method
- **DTD** as a timestep-sampling acceleration method

and plugs them into MeanFlow training. They report both help, but emphasize that **DTD is more robust across model scales** because it changes the sampling distribution instead of interfering with MeanFlow’s own adaptive loss normalization. 

### 2.4.3 Component 2: Progressive weighting on the MeanFlow loss

UAIMF progressively reweights the MeanFlow term so training starts by emphasizing **small temporal gaps** (easy) and gradually transitions to **uniform weighting** (full MeanFlow objective). Their weighting is:

$$
\beta(\Delta t, s) = 1 - s + \lambda s (1 - \Delta t)
$$

where:

- $$\Delta t$$ is the temporal gap,
- $$s \in [0,1]$$ is training progress,
- $$\lambda$$ normalizes the expectation at initialization.

At initialization, the weighting prioritizes small gaps; by the end, it becomes uniform. They use a linear schedule by default:

$$
s = 1 - \frac{i}{T}
$$

and also discuss the generalized schedule:

$$
s = 1 - \left(\frac{i}{T}\right)^k
$$

with $$k=1$$ (linear) working best in their ablations. 

### 2.4.4 Why the two components work together

Their ablation is clean:

- velocity acceleration alone improves MeanFlow,
- progressive $$L_u$$ weighting alone improves MeanFlow,
- combining both works best.

They explicitly interpret this as:

- acceleration methods quickly establish the **instantaneous velocity foundation**
- progressive weighting improves **average velocity learning** over time

which is exactly the right mental model for MeanFlow optimization. 

## 2.5 Decoupled MeanFlow

Decoupled MeanFlow is the strongest architectural update in this line. The core idea is: **the encoder should care about the current timestep, and the decoder should care about the target timestep**. They decouple timestep conditioning and turn a pretrained flow model into a flow-map model with almost no architectural surgery. 

### 2.5.1 Core architectural idea: decouple encoder vs decoder conditioning

They reinterpret a flow model as:

$$
v_\theta = g_\theta \circ f_\theta
$$

with an encoder $$f_\theta$$ and decoder $$g_\theta$$. Then they argue the standard MeanFlow design is redundant because it feeds the next timestep $$r$$ everywhere. Their fix:

- encoder gets current timestep $$t$$
- decoder gets next timestep $$r$$

and the flow map becomes:

$$
u_\theta(x_t, t, r) = g_\theta(f_\theta(x_t, t), r)
$$

This is the defining DMF equation.

### 2.5.2 Why this matters: pretrained flow models already contain flow-map structure

DMF shows that a pretrained flow model can be **converted into a flow map without fine-tuning**, just by choosing an encoder/decoder split and decoding the representation with $$r$$. They report the converted DMF can even outperform the original flow model in some settings, which supports the claim that good flow-model representations are already enough for flow-map prediction. 

This is a major conceptual shift: instead of training flow maps from scratch, you can **reuse pretrained flow-model representations** and repurpose the decoder.

### 2.5.3 Representation-first view

DMF explicitly argues that **representation quality matters** for flow maps. They show:

- stronger pretrained encoders transfer better to flow-map fine-tuning
- freezing encoder + tuning decoder already gives a large speed/quality gain
- but true 1-step performance needs joint optimization (encoder cannot stay frozen forever)

This gives a practical recipe: pretrain a strong flow model first, then convert/fine-tune as DMF. 

### 2.5.4 Training recipe: FM warm-up + MF fine-tuning

DMF also proposes a better training pipeline:

1. train a flow model with FM loss
2. convert to DMF
3. fine-tune with MeanFlow loss

They justify this on compute grounds (MF/JVP is expensive) and show it scales better than training a flow map from scratch. This is one of the most important practical contributions in the paper.

### 2.5.5 Enhanced training techniques

#### (a) Adaptive weighted Cauchy loss

They note MF loss has high variance, then replace the raw MSE-style MF loss with a **Cauchy (Lorentzian) robust loss** and an adaptive weighting term over timestep pairs. Their DMF objective is written as an adaptive weighted Cauchy form over the MeanFlow residual. 

#### (b) Time proposal tailored to flow maps

They adapt timestep-pair sampling for flow maps (since you need ordered pairs with $$t>r$$). They sample two logit-normal values and sort them, and then bias the proposal toward larger gaps / smaller $$r$$ for better 1-step behavior, because converted DMF models are already strong near the diagonal $$r \approx t$$. 

#### (c) Model Guidance (MG)

They use **Model Guidance (MG)** to avoid the full compute cost of CFG during training, and note MG is especially effective for training high-quality few-step flow maps. This is part of why their 1-step/4-step results are strong. 

### 2.5.6 Takeaways

DMF is not just another MeanFlow variant. It reframes the problem:

- **MeanFlow** gives the right objective (average velocity via JVP)
- **UAIMF / AlphaFlow** improve optimization dynamics
- **DMF** improves the **architecture + training pipeline**, and shows pretrained flow models are the best starting point

They report SOTA-level few-step results and show 1-step / 4-step generation approaching much more expensive flow-model sampling with large inference-speed gains. 

---

# 3. FlowMap

## 3.1 Free FlowMap

Flow Map Distillation Without Data bias

#### 3.1.1 Teacher-data mismatch (the hidden bug in many distillation pipelines)

Traditional flow-map distillation often samples intermediate states $x_t$ from an **external dataset distribution** and supervises the student using teacher velocities at those states.

But the student is supposed to reproduce the teacher’s **sampling process**, i.e. the trajectory distribution induced by the teacher from the prior.

Supervision states coming from a mismatched distribution results in a **teacher-data mismatch**:
- the student is trained on states that are not on the teacher’s true rollout distribution,
- more augmentation can worsen it,
- student quality degrades.

This is a deep point because it says the standard “distill on data” recipe is fundamentally misaligned with the actual objective (imitate the teacher sampler).

#### 3.1.2 Prior-only / self-generated flow-map objective

Supervise entirely from the prior and the student’s own generated states.

They derive a sufficient optimality condition leading to a loss of the form

$$
\mathcal{L}_{\text{pred}}
=
\mathbb{E}_{z,\delta}
\left[
\|F_\theta(z,\delta)-\operatorname{sg}(u_{\text{target}})\|^2
\right]
$$

with

$$
u_{\text{target}}
=
u(f_\theta(z,\delta),1-\delta) - \delta \,\partial_\delta F_\theta(z,\delta)
$$

The key interpretation:
- $f_\theta(z,\delta)$ defines the student’s current trajectory,
- $\partial_\delta f_\theta$ is the **student’s generating velocity**,
- the loss is equivalent to aligning the student generating velocity with the teacher field:

$$
\partial_\delta f_\theta \approx u(f_\theta(z,\delta),1-\delta)
$$

So the student learns to **ride the teacher vector field along its own generated path**, starting from pure noise, no dataset needed.

This is the right fix for teacher-data mismatch.

#### 3.1.3 Gradient view

They explicitly write the optimization gradient in terms of a velocity mismatch

$$
\Delta v_{G,u} = v_G - u
$$

which is great because it makes gradient weighting / normalization tricks easier to reason about.

This is one of those “small” presentation choices that actually matters in practice.

#### 3.1.4 Correction objective (fixing distribution drift, not just local velocity)

Here’s the catch: aligning generating velocity locally is necessary, but in finite-capacity/discrete training the generated distribution can still drift.

So they add a **correction objective** motivated from minimizing a KL term over intermediate marginals and then translating score mismatch into **velocity mismatch** (via the score–velocity equivalence for linear interpolants).

This yields a gradient proportional to

$$
\nabla_\theta \;
\mathbb{E}_{z,n,r}
\left[
F_\theta(z,1)^\top \operatorname{sg}
\left(
v_N(I_r(f_\theta(z,1),n),r) - u(I_r(f_\theta(z,1),n),r)
\right)
\right]
$$

where:
- $u$ is the teacher marginal velocity,
- $v_N$ is the student-induced **noising** marginal velocity,
- $I_r(\cdot,\cdot)$ is the interpolation to intermediate time $r$.

Intuition:
- the prediction loss aligns the **student’s forward/generating flow**
- the correction loss aligns the **student-induced noising marginals** with the teacher’s marginals

That is a nice bidirectional correction mechanism.

---

## 3.2 Meta Flow Maps

Meta flow maps correspond to a **stochastic flow map**, which is important because deterministic flow maps are too rigid.

#### 3.2.1 Why stochastic flow maps?

Deterministic flow-map learning works if the transport map is the right object. But for many diffusion-like processes, especially when you want richer uncertainty handling, a **stochastic transition kernel**

$$
\kappa_{t,s}(z_t, z_s)
$$
is the right object.

The paper frames this using:
- **marginal consistency**
- **conditional consistency**
- a family of posterior conditionals $p_{1|t}$
- and a diagonal supervision view

The important conceptual upgrade is:
- instead of only learning deterministic trajectories,
- learn a transition operator consistent with the stochastic process structure.

#### 3.2.2 The diagonal condition (same role as FM diagonal supervision)

They derive that on the diagonal:

$$
\kappa_{t,t}(z_t, z_1) = p_{1|t}(z_1 \mid z_t)
$$

This is the stochastic analogue of “when $r=t$, your two-time object must match the one-time target.”

So the diagonal again plays the role of anchor supervision.

#### 3.2.3 Pathwise/consistency relation for stochastic flow maps

They also derive a consistency/composition condition (their Eq. 23 in the snippet):

$$
\kappa_{t,s}(z_t,z_s)
=
\mathbb{E}_{z_1 \sim p_{1|t}(\cdot|z_t)}
\big[
\kappa_{u,s}(z_u,z_s)
\big]
$$

(with the appropriate latent dependence through $z_1$/paths)

The exact notation is heavier, but the key idea is the same as flow-map composition:
- **two-time transitions must compose correctly through intermediate times**, but now in distributional form.

#### 3.2.4 Their training objective (MFM loss)

They build:
1. a **diagonal supervision loss** (fit the posterior on diagonal time pairs),
2. a **consistency loss** (enforce off-diagonal composition consistency),
3. and combine them into an MFM objective:

$$
\mathcal{L}_{\text{MFM}}
=
\mathcal{L}_{\text{diag}} + \lambda \mathcal{L}_{\text{cons}}
$$

This is the stochastic counterpart of the deterministic progression:
- diagonal target = “FM-like” anchor
- off-diagonal consistency = “flow-map-like” propagation

#### 3.2.5 Takeaways

This paper gives a more general lens:
- MeanFlow / deterministic flow maps are one branch
- stochastic transition learning is the broader object when uncertainty matters
- the “diagonal + consistency” decomposition is the unifying pattern

This is exactly the kind of conceptual bridge diffusion researchers should care about.

---

## 3.3 Transition Matching Distillation

Transition Matching Distillation (TMD) bridges engineering and theory based adaptation of MeanFlow to **video distillation**.

#### 3.3.1 Core problem setup

They want to distill a pretrained video diffusion teacher into a faster student. Direct one-stage distillation is hard in video because:
- the space is huge,
- temporal consistency matters,
- transformer-based video models make JVP painful (esp. attention kernels/FSDP/context parallelism).

So TMD uses **two stages**.

#### 3.3.2 Stage 1: Transition Matching MeanFlow (TM-MF)

This is the key new idea.

Instead of applying MeanFlow directly in the original latent/data space, they define an **inner transition** problem and parameterize a conditional inner flow map via average velocity:

$$
f_\theta(y_s,s,r;m) = y_s + (s-r)u_\theta(y_s,s,r;m)
$$

where $m$ is a feature extracted from the main backbone.

Then they use a MeanFlow-style objective to train this transition head.

A very practical (and nontrivial) design choice:
- they **reparameterize** the average velocity to stay aligned with the teacher head:

$$
u_\theta(y_s,s,r;m) = y_1 - \text{head}_\theta(y_s,s,r;m)
$$

This is not cosmetic. It keeps the new head close to teacher semantics, which improves stability.

#### 3.3.3 JVP issue and finite-difference approximation

This paper is very realistic about systems constraints:
- exact JVP is annoying with large-scale video transformer stacks (FlashAttention, FSDP, context parallelism),
- so they use a **finite-difference approximation** of the JVP.

That’s a practical compromise:
- theoretically less clean than exact JVP,
- but massively easier to integrate into production-grade training code.

#### 3.3.4 Stage 2: Distributional distillation objective

After TM-MF pretraining, they switch to a stronger distillation stage using a VSD/discriminator-style objective (their simplified algorithm shows):

$$
\mathcal{L} = \text{VSD}(\hat{x}) + \lambda \cdot \text{Discriminator}(\hat{x})
$$

So the conceptual split is:

- **Stage 1 (TM-MF):** learn a good transition-aware student parameterization, bootstrap geometry/dynamics
- **Stage 2:** sharpen sample quality and distribution match

This is a strong template for hard domains (video, 3D, multimodal) where pure one-shot distillation is brittle.

---
# 4. Score Distillation
## 4.1 Variational Score Distillation (VSD)

### 4.1.1 Score Distillation Sampling (SDS)

VSD explicitly frames **SDS** as the starting point for text-to-3D optimization.  
Given a differentiable renderer

$$
g(\theta, c)
$$

and a pretrained text-to-image diffusion model, SDS optimizes a **single 3D parameter** $\theta$ by minimizing a KL objective over noisy rendered images:

$$
L_{\mathrm{SDS}}(\theta)
:=
\mathbb{E}_{t,c}
\left[
\frac{\sigma_t}{\alpha_t}\,\omega(t)\,
D_{\mathrm{KL}}
\big(q_t^\theta(x_t \mid c)\,\|\,p_t(x_t \mid y_c)\big)
\right].
$$

The practical gradient (the one everyone uses) is approximated as:

$$
\nabla_\theta L_{\mathrm{SDS}}(\theta)
\approx
\mathbb{E}_{t,\epsilon,c}
\left[
\omega(t)\big(\epsilon_{\mathrm{pretrain}}(x_t,t,y_c)-\epsilon\big)
\frac{\partial g(\theta,c)}{\partial \theta}
\right],
$$

with

$$
x_t = \alpha_t g(\theta,c) + \sigma_t \epsilon.
$$

#### 4.1.2 Why SDS is limited

SDS uses a **single parameter** $\theta$ and directly optimizes it, which makes it a **mode-seeking** procedure in practice.  
VSD points out this is one reason SDS often gives over-smoothed / over-saturated / artifacted 3D results (especially in low-density regions of image space).

---

### 4.1.3 VSD: move from optimizing one 3D sample to learning a distribution over 3D parameters

The key VSD move is to define a **distribution over 3D parameters**

$$
\mu(\theta \mid y)
$$

instead of optimizing one fixed $\theta$.

Then VSD evolves particles by a **Wasserstein gradient flow / particle ODE**. The ODE uses a score difference between:

1. the score of noisy real images (from the pretrained diffusion model), and  
2. the score of noisy rendered images (estimated by a learnable network).

The VSD particle dynamics are:

$$
\frac{d\theta_\tau}{d\tau}
=
-\mathbb{E}_{t,\epsilon,c}
\left[
\omega(t)
\left(
-\sigma_t \nabla_{x_t}\log p_t(x_t \mid y_c)
-
\big(-\sigma_t \nabla_{x_t}\log q_t^{\mu_\tau}(x_t \mid c,y)\big)
\right)
\frac{\partial g(\theta_\tau,c)}{\partial \theta_\tau}
\right].
$$

This is the conceptual heart of VSD:

- **teacher score** pulls rendered images toward the text-conditioned image manifold,
- **rendered-image score** corrects for the current particle distribution,
- the difference gives a much better update than raw SDS.

---

### 4.1.4 The VSD score function estimator (this is the diffusion-distillation part)

VSD introduces a second noise predictor

$$
\epsilon_\phi(x_t,t,c,y)
$$

to estimate the score of noisy rendered images. It is trained with the **standard diffusion noise-prediction objective** on rendered images from the current particles:

$$
\min_\phi
\sum_{i=1}^n
\mathbb{E}_{t,\epsilon,c}
\left[
\left\|
\epsilon_\phi\big(\alpha_t g(\theta^{(i)},c)+\sigma_t\epsilon,\ t,\ c,\ y\big)-\epsilon
\right\|_2^2
\right].
$$

This is the crucial diffusion-distillation component in VSD:
- VSD **distills a score model for the rendered-image distribution** (not just the teacher),
- then uses the **difference of two scores** for particle updates.

In practice, VSD parameterizes $\epsilon_\phi$ as either:
- a small U-Net, or
- a **LoRA** adaptation of the pretrained diffusion model (usually better fidelity).

---

### 4.1.5 VSD gradient used for updating the 3D particles

Once $\epsilon_\phi$ is trained, the per-particle VSD gradient becomes:

$$
\nabla_\theta L_{\mathrm{VSD}}(\theta)
=
\mathbb{E}_{t,\epsilon,c}
\left[
\omega(t)\,
\big(
\epsilon_{\mathrm{pretrain}}(x_t,t,y_c)
-
\epsilon_\phi(x_t,t,c,y)
\big)\,
\frac{\partial g(\theta,c)}{\partial \theta}
\right],
$$

with

$$
x_t = \alpha_t g(\theta,c)+\sigma_t\epsilon.
$$

This is the clean algorithmic form:
- **SDS** uses $\epsilon_{\mathrm{pretrain}} - \epsilon$
- **VSD** uses $\epsilon_{\mathrm{pretrain}} - \epsilon_\phi$

That replacement is the whole game.

---

### 4.1.6 VSD vs SDS

VSD shows SDS is a **special case**:

- if you approximate the parameter distribution $\mu(\theta \mid y)$ by a single empirical point mass (one particle),
- then the rendered-image score term degenerates to the sampled noise $\epsilon$,
- and you recover vanilla SDS / SJC.

So the practical interpretation is:

- **SDS** = single-point / no-generalization approximation
- **VSD** = distributional score-distillation with a learned rendered-image score model

That is why VSD usually gives better diversity and better fidelity in text-to-3D than plain SDS.

---

## 4.2 DMD (Distribution Matching Distillation)

### 4.2.1 Core idea

DMD trains a **one-step generator** by taking a **distribution-matching gradient** (KL-driven, score-difference style). The distribution-matching improves realism and fixes pure regression collapse.

---

### 4.2.2 Distribution matching gradient

DMD computes a gradient by comparing two denoisers on the same noisy fake sample:

- $\mu_{\text{real}}$: pretrained denoiser trained on real data
- $\mu_{\text{fake}}$: denoiser trained on fake/generated data

For a fake image $x = G_\theta(z)$:

1. add random diffusion noise to get $x_t$  
2. denoise with both networks  
3. use the difference as the realism direction

The implementation-level gradient proxy is essentially:

$$
\mathrm{grad}
\propto
\frac{
\mu_{\text{fake}}(x_t,t)-\mu_{\text{real}}(x_t,t)
}{
\text{weighting\_factor}
}
$$

and they realize this as a stop-grad MSE objective on $x$ (so autograd yields the desired gradient direction).

This is the key DMD trick:
- avoid backprop through a long teacher trajectory,
- still get a **distribution-level** correction signal.

#### 4.2.3 Practical read on DMD

DMD is strong because it is **not just teacher regression**:
- the KL / distribution-matching gradient gives a realism signal beyond memorizing teacher trajectories,
- the regression term keeps training stable.

That combo is why DMD was a big step for one-step image generation.

---

## 4.3 SiD (Score identity Distillation)

### 4.3.1 Core framing: explicit score matching on fake-data diffusion

SiD reframes one-step diffusion distillation as minimizing a **score-difference loss** on the diffused fake-data distribution.

Define the score difference:

$$
\delta_{\phi,\theta}(x_t)
:=
S_\phi(x_t) - \nabla_{x_t}\log p_\theta(x_t),
$$

where:
- $S_\phi(x_t)$ is the pretrained teacher score,
- $p_\theta(x_t)$ is the diffused generator distribution.

Then SiD defines the theoretical objective (MESM / Fisher divergence):

$$
L_\theta
=
\mathbb{E}_{x_t \sim p_\theta(x_t)}
\left[
\|\delta_{\phi,\theta}(x_t)\|_2^2
\right].
$$

This is the cleanest formulation in this chapter conceptually:
- make the **fake-data score** match the **teacher score**.

---

### 4.3.2 Why SiD needs an auxiliary score network

The hard part is the generator score

$$
\nabla_{x_t}\log p_\theta(x_t),
$$

which is intractable directly.

SiD introduces a second network

$$
f_\psi(x_t,t)
$$

to approximate the conditional denoising mean / score-related quantity:

$$
f_{\psi^*(\theta)}(x_t,t)
=
\mathbb{E}[x_g \mid x_t]
=
x_t + \sigma_t^2 \nabla_{x_t}\log p_\theta(x_t).
$$

This gives a score-difference estimator:

$$
\delta_{\phi,\psi^*(\theta)}(x_t)
=
\sigma_t^{-2}\big(f_\phi(x_t,t)-f_{\psi^*(\theta)}(x_t,t)\big).
$$

So SiD turns score estimation into a diffusion-style denoising regression problem for $f_\psi$.

---

### 4.3.3 The naive approximation fails (important)

A very natural idea is to replace $\psi^*(\theta)$ with the current $\psi$ and use

$$
\delta_{\phi,\psi}(x_t)
=
\sigma_t^{-2}\big(f_\phi(x_t,t)-f_\psi(x_t,t)\big),
$$

then optimize the naive approximated loss

$$
L_\theta^{(1)}
=
\mathbb{E}\left[\|\delta_{\phi,\psi}(x_t)\|_2^2\right].
$$

SiD explicitly shows this can fail badly because:
- the approximation error in $f_\psi$ enters the objective in a destabilizing way,
- the loss depends on both score-estimation error and the true score difference.

This is the main reason SiD is not just “DMD but with a different notation.”

---

### 4.3.4 Projected score matching (SiD’s key fix)

SiD derives an alternative MESM form (Projected Score Matching) and then approximates it to get a much more stable objective:

$$
L_\theta^{(2)}
=
\mathbb{E}
\left[
\sigma_t^{-2}\,
\delta_{\phi,\psi}(x_t)^\top
\big(f_\phi(x_t,t)-x_g\big)
\right].
$$

This version is much more stable because it depends on the denoising residual

$$
f_\phi(x_t,t)-x_g
$$

instead of directly squaring the noisy score-difference estimate.

That is the algorithmic insight behind SiD:
- use score identities to derive a loss that still targets MESM,
- but behaves better under imperfect generator-score estimation.

---

### 4.3.5 Fused loss and alternating updates (the actual algorithm)

SiD trains with **alternating updates**:

1. **Update** $f_\psi$ (the generator-score estimator) with a diffusion / denoising loss.
2. **Update** $G_\theta$ using the SiD generator loss (built from the projected score-matching approximation / fused loss).

It also uses score gradients (backprop through both score networks), which distinguishes it from methods like Diff-Instruct and DMD.

In practice this is more memory/computation heavy, but it is a major reason SiD gets very strong one-step results.

---

# 5. Adversarial Distillation

# 6. Video Generation

## 6.3 Terminal Velocity Matching

Terminal Velocity Matching (TVM): A More Principled Objective for One/Few-Step Models

### 6.3.1 What it changes

Prior methods (FM/MeanFlow/FMM-style) mostly match local or initial-time velocity constraints.

TVM says: match the **terminal velocity** of the flow trajectory instead.

That sounds minor, but it changes the theory:
- they derive an explicit **2-Wasserstein upper bound**
- and motivate a more stable training target for one/few-step generation

### 6.3.2 Core theorem and loss structure

TVM introduces a terminal velocity target
$$
u^*(x_t,t,s) = \mathbb{E}[v_t \mid x_t]
$$
(at terminal pairing \(s\), with the paper’s precise conditioning)

Then they define a target involving a time derivative of the learned map:
$$
u^*_\theta(x_t,t,s)
=
u^*(x_t,t,s) - (t-s)\partial_s F_\theta(x_t,t,s)
$$

and train with a matching loss of the form
$$
\mathbb{E}\|F_\theta - u^*_\theta\|^2
$$

The crucial thing is **not** just the formula; it’s the theorem:
they show this objective upper-bounds the 2-Wasserstein distance (up to constants / residual terms in their theorem statement).

That gives TVM a stronger distributional interpretation than “just match a derivative identity.”

### 6.3.3 Significance

This is the first really clean signal that the field is maturing beyond:
- local consistency heuristics,
- empirical JVP identities,
- “it works in 1 step”

into:
- **distributional guarantees**
- explicit control over terminal behavior
- better theory-practice alignment

### 6.3.4 The systems contribution

TVM also points out a major implementation pain:
- JVP of scaled dot-product attention is poorly supported / inefficient in standard autograd stacks.
- Unlike prior works, TVM also propagates gradient through the JVP term (not just stop-grad around it), which is even harder.

They propose a FlashAttention kernel that fuses JVP with forward pass and supports backward through the JVP result.

That matters a lot if you care about scaling this family to modern DiT/transformer stacks.

---

# 7. New Domains

# 8. Manifold

# 7. Unifying View: The Field’s Progression in One Picture

Almost every method in this area can be seen as:

1. **Diagonal anchor**
   - when time-pair collapses, match a standard object (FM velocity / posterior / terminal target)
2. **Off-diagonal propagation**
   - via a differential identity (MeanFlow / FMM),
   - or consistency/composition (stochastic/meta flow maps),
   - or terminal-time control (TVM)

This “diagonal + propagation” lens is the best mental model for the literature.

---

# 8. Practical Research Takeaways (for top-lab diffusion folks)

### 8.1 If you’re doing one-step/few-step image generation
Start by deciding which failure mode you care about:
- **trajectory geometry / displacement quality** → MeanFlow-style
- **distributional mismatch under self-generated rollouts** → add FreeFlowMap-style correction
- **theory / Wasserstein control** → TVM-style objective

### 8.2 If you’re doing video / transformers
The math is not the main bottleneck; **JVP implementation is**.
TMD and TVM both basically scream this:
- attention kernels + JVP + distributed training are the real constraint
- finite-difference JVP is often the “actually trains” solution
- custom kernels become a differentiator

### 8.3 If you’re doing 3D / world models / stochastic simulators
The stochastic flow-map lens is probably the most future-proof:
- deterministic flow maps are great for one-step generation
- but world models usually need stochastic transitions
- the **diagonal posterior + consistency** formulation is much closer to what you actually want

---

# 9. A compact “theory stack” to remember

The whole area can be compressed into this stack:

1. **FM**: learn local tangent
2. **MeanFlow/FMM**: convert local tangent into a trainable two-time displacement rule (via differential identity/JVP)
3. **Self-distilled flow maps**: train on the student’s own rollout distribution (fix teacher-data mismatch)
4. **Correction terms**: explicitly align marginals / noising velocities
5. **Stochastic flow maps**: move from deterministic maps to transition kernels (posterior-diagonal + consistency)
6. **TVM**: choose a target (terminal velocity) that gives stronger distributional guarantees

That’s the real progression.

---

# 10. Where the field is likely going next

The next wave is probably a merge of these threads:

- **TVM-style distributional guarantees**
- **self-distilled prior-only training**
- **stochastic transition operators**
- **kernel-aware JVP training for transformers/video/world models**

In other words:
the future is not just “better one-step image generation,” it’s **learned transition operators** for high-dimensional structured dynamics (video, 3D, simulators, world models), with both:
- strong numerical behavior (few NFEs)
- and actual distributional control (Wasserstein/KL-style guarantees)

That’s exactly where flow maps stop being a distillation trick and become a real modeling primitive.