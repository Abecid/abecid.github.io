---
layout: post
title: Recent Flow-Map Distillation Methods
date: 2026-02-20 01:24:00
description: Recent developments of diffusion distillation techniques
tags: diffusion survey
categories: research-survey
---

Meta FlowMap

# Flow Matching to Flow Maps to Distillation: A Deep Dive
(MeanFlow, Flow Map Self-Distillation, Stochastic/Meta Flow Maps, TMD, TVM)

# Table of Content
1. [Foundations: Flow Matching and Flow Maps](#1-foundations-flow-matching-vs-flow-maps)
2. [MeanFlow](#2-meanflow)
3. [FreeFlowMap](#3-freeflowmap)
4. [Meta FlowMap](#4-meta-flow-map)
5. [Transition Matching Distillation](#5-transition-matching-distillation)
6. [Terminal Velocity Matching](#6-terminal-velocity-matching)

# Overview
Recent generative modeling utilize and develop upon flow maps and jvp based distillation techniques to reduce the number of function evaluations during inference. 

# Summary

The modern progression is:

1. **Flow Matching (FM)** learns the **instantaneous velocity field**.
2. **Flow-map methods** learn **time-to-time transport maps** (or average velocities), which are much more compatible with **1-step / few-step generation**.
3. **MeanFlow** gives a clean, derivation-first way to train a two-time flow-map model via a **JVP-based identity** (no extra consistency axiom).
4. **Flow Map Distillation without Data / self-distillation** (the “FreeFlowMap / How to build a consistency model” line) points out **teacher-data mismatch** and replaces data-dependent distillation with **prior-only self-generated supervision**, then adds a **correction objective** to fix distribution drift.
5. **Stochastic Flow Maps / Meta Flow Maps** generalize flow maps to **stochastic transitions** and derive **diagonal + consistency** objectives from conditional posterior structure; this is the right lens when deterministic flow maps are too restrictive.
6. **TMD** ports MeanFlow-style ideas into **video distillation**, using a **transition-matching MeanFlow pretraining stage** + a second-stage distributional distillation objective.
7. **TVM** shifts the target from “match initial/local velocity” to **match terminal velocity**, and gives a more principled guarantee (explicit **2-Wasserstein upper bound**), while also exposing a key practical systems issue: **JVP through transformer attention**.

---

# 1. Foundations: Flow Matching and Flow Maps
### 1.1 Flow Matching
![Flow Matching](/assets/img/blogs/1_distillation/flowmatching.png)

*Figure 1. Flow Matching. from Sabour, Fidler, and Kreis (2025),* Align Your Flow: Scaling Continuous-Time Flow Map Distillation *(arXiv:2506.14603).*  

Generative models learn to transport probability ditribution from prior, $p_{0}$, to a data distribution, $p_{1}$. 

Classic flow matching learns a vector field
$$
u(x,t)
$$
that acts as an instantaneous velocity at timestep $t$, which collectively defines an ODE trajectory. Sampling involves ODE integration from noise to data.

There are several bottlenecks to this approach: if the learned trajectory is curved, a decent solver (Euler, Heun) is required and many number of function evaluations (NFEs) to integrate over the ODE trajectory.

### 1.2 Flow Map
![Flow Map](/assets/img/blogs/1_distillation/flowmap.png)

*Figure 2. Flow Map. from Sabour, Fidler, and Kreis (2025),* Align Your Flow: Scaling Continuous-Time Flow Map Distillation *(arXiv:2506.14603).*  

Flow map formulation directly targets:
$$
\phi_u(x_t, t, s)
$$
which maps a state at time \(t\) to time \(s\), instead of learning only the local instantaneous tangent.

Recent methods have emerged developing upon this flow map formulation for fewer step, student-teacher, data-free distillation families.

---

# 2. MeanFlow
#### 2.1 Average velocity instead of instantaneous velocity
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

#### 2.2 The MeanFlow Identity (the central derivation)

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

#### 2.3 Main Points

- The authors rewrite an intractable target (the average velocity integral) into a trainable target:
  - By first taking the derivative of both sides.
  - Instantaneous velocity $v$ is available from FM-style interpolation.
  - Total derivative term computed via **JVP**.
- No explicit consistency regularizer is imposed.
- The consistency-like structure falls out from the definition of average velocity.

#### 2.4 MeanFlow loss

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

#### 2.5 Sampling

Once you learn the average velocity, sampling is:

$$
z_r = z_t - (t-r)u(z_t,r,t)
$$

For 1-step:

$$
z_0 = z_1 - u(z_1,0,1),\quad z_1 \sim p_{\text{prior}}
$$

---

# 3. FreeFlowMap

Flow Map Distillation Without Data

## 3.1 Teacher-data mismatch (the hidden bug in many distillation pipelines)

Traditional flow-map distillation often samples intermediate states \(x_t\) from an **external dataset distribution** and supervises the student using teacher velocities at those states.

But the student is supposed to reproduce the teacher’s **sampling process**, i.e. the trajectory distribution induced by the teacher from the prior.

Supervision states coming from a mismatched distribution results in a **teacher-data mismatch**:
- the student is trained on states that are not on the teacher’s true rollout distribution,
- more augmentation can worsen it,
- student quality degrades.

This is a deep point because it says the standard “distill on data” recipe is fundamentally misaligned with the actual objective (imitate the teacher sampler).

## 3.2 Prior-only / self-generated flow-map objective

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
- \(f_\theta(z,\delta)\) defines the student’s current trajectory,
- \(\partial_\delta f_\theta\) is the **student’s generating velocity**,
- the loss is equivalent to aligning the student generating velocity with the teacher field:
$$
\partial_\delta f_\theta \approx u(f_\theta(z,\delta),1-\delta)
$$

So the student learns to **ride the teacher vector field along its own generated path**, starting from pure noise, no dataset needed.

This is the right fix for teacher-data mismatch.

## 3.3 Gradient view

They explicitly write the optimization gradient in terms of a velocity mismatch
$$
\Delta v_{G,u} = v_G - u
$$
which is great because it makes gradient weighting / normalization tricks easier to reason about.

This is one of those “small” presentation choices that actually matters in practice.

## 3.4 Correction objective (fixing distribution drift, not just local velocity)

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
- \(u\) is the teacher marginal velocity,
- \(v_N\) is the student-induced **noising** marginal velocity,
- \(I_r(\cdot,\cdot)\) is the interpolation to intermediate time \(r\).

Intuition:
- the prediction loss aligns the **student’s forward/generating flow**
- the correction loss aligns the **student-induced noising marginals** with the teacher’s marginals

That is a nice bidirectional correction mechanism.

---

# 4. Meta Flow Maps

Meta flow maps correspond to a **stochastic flow map**, which is important because deterministic flow maps are too rigid.

### 4.1 Why stochastic flow maps?

Deterministic flow-map learning works if the transport map is the right object. But for many diffusion-like processes, especially when you want richer uncertainty handling, a **stochastic transition kernel**
$$
\kappa_{t,s}(z_t, z_s)
$$
is the right object.

The paper frames this using:
- **marginal consistency**
- **conditional consistency**
- a family of posterior conditionals \(p_{1|t}\)
- and a diagonal supervision view

The important conceptual upgrade is:
- instead of only learning deterministic trajectories,
- learn a transition operator consistent with the stochastic process structure.

### 4.2 The diagonal condition (same role as FM diagonal supervision)

They derive that on the diagonal:
$$
\kappa_{t,t}(z_t, z_1) = p_{1|t}(z_1 \mid z_t)
$$

This is the stochastic analogue of “when \(r=t\), your two-time object must match the one-time target.”

So the diagonal again plays the role of anchor supervision.

### 4.3 Pathwise/consistency relation for stochastic flow maps

They also derive a consistency/composition condition (their Eq. 23 in the snippet):
$$
\kappa_{t,s}(z_t,z_s)
=
\mathbb{E}_{z_1 \sim p_{1|t}(\cdot|z_t)}
\big[
\kappa_{u,s}(z_u,z_s)
\big]
$$
(with the appropriate latent dependence through \(z_1\)/paths)

The exact notation is heavier, but the key idea is the same as flow-map composition:
- **two-time transitions must compose correctly through intermediate times**, but now in distributional form.

### 4.4 Their training objective (MFM loss)

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

### 4.5 Why this matters for the broader field

This paper gives a more general lens:
- MeanFlow / deterministic flow maps are one branch
- stochastic transition learning is the broader object when uncertainty matters
- the “diagonal + consistency” decomposition is the unifying pattern

This is exactly the kind of conceptual bridge diffusion researchers should care about.

---

# 5. Transition Matching Distillation

Transition Matching Distillation (TMD) bridges engineering and theory based adaptation of MeanFlow to **video distillation**.

### 5.1 Core problem setup

They want to distill a pretrained video diffusion teacher into a faster student. Direct one-stage distillation is hard in video because:
- the space is huge,
- temporal consistency matters,
- transformer-based video models make JVP painful (esp. attention kernels/FSDP/context parallelism).

So TMD uses **two stages**.

### 5.2 Stage 1: Transition Matching MeanFlow (TM-MF)

This is the key new idea.

Instead of applying MeanFlow directly in the original latent/data space, they define an **inner transition** problem and parameterize a conditional inner flow map via average velocity:
$$
f_\theta(y_s,s,r;m) = y_s + (s-r)u_\theta(y_s,s,r;m)
$$

where \(m\) is a feature extracted from the main backbone.

Then they use a MeanFlow-style objective to train this transition head.

A very practical (and nontrivial) design choice:
- they **reparameterize** the average velocity to stay aligned with the teacher head:
$$
u_\theta(y_s,s,r;m) = y_1 - \text{head}_\theta(y_s,s,r;m)
$$

This is not cosmetic. It keeps the new head close to teacher semantics, which improves stability.

### 5.3 JVP issue and finite-difference approximation

This paper is very realistic about systems constraints:
- exact JVP is annoying with large-scale video transformer stacks (FlashAttention, FSDP, context parallelism),
- so they use a **finite-difference approximation** of the JVP.

That’s a practical compromise:
- theoretically less clean than exact JVP,
- but massively easier to integrate into production-grade training code.

### 5.4 Stage 2: Distributional distillation objective

After TM-MF pretraining, they switch to a stronger distillation stage using a VSD/discriminator-style objective (their simplified algorithm shows):
$$
\mathcal{L} = \text{VSD}(\hat{x}) + \lambda \cdot \text{Discriminator}(\hat{x})
$$

So the conceptual split is:

- **Stage 1 (TM-MF):** learn a good transition-aware student parameterization, bootstrap geometry/dynamics
- **Stage 2:** sharpen sample quality and distribution match

This is a strong template for hard domains (video, 3D, multimodal) where pure one-shot distillation is brittle.

---

# 6. Terminal Velocity Matching

Terminal Velocity Matching (TVM): A More Principled Objective for One/Few-Step Models

### 6.1 What it changes

Prior methods (FM/MeanFlow/FMM-style) mostly match local or initial-time velocity constraints.

TVM says: match the **terminal velocity** of the flow trajectory instead.

That sounds minor, but it changes the theory:
- they derive an explicit **2-Wasserstein upper bound**
- and motivate a more stable training target for one/few-step generation

### 6.2 Core theorem and loss structure

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

### 6.3 Significance

This is the first really clean signal that the field is maturing beyond:
- local consistency heuristics,
- empirical JVP identities,
- “it works in 1 step”

into:
- **distributional guarantees**
- explicit control over terminal behavior
- better theory-practice alignment

### 6.4 The systems contribution

TVM also points out a major implementation pain:
- JVP of scaled dot-product attention is poorly supported / inefficient in standard autograd stacks.
- Unlike prior works, TVM also propagates gradient through the JVP term (not just stop-grad around it), which is even harder.

They propose a FlashAttention kernel that fuses JVP with forward pass and supports backward through the JVP result.

That matters a lot if you care about scaling this family to modern DiT/transformer stacks.

---

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