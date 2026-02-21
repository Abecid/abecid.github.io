---
layout: post
title: Recent Flow-Map Distillation Methods
date: 2026-02-20 01:24:00
description: Recent developments of diffusion distillation techniques
tags: diffusion survey
categories: research-survey
---

Diffusion

FlowMap

MeanFlow

Data Free Distillation

Meta FlowMap

# Accelerating Generative Flows: From Flow Matching to Data-Free and Stochastic Distillation

This post covers the theoretical progression of flow-based generative modeling, detailing the transition from instantaneous vector field regression (Flow Matching) to integral operator parameterization (Flow Maps), and the recent frontier of data-free, stochastic, and terminal-differentiated distillation frameworks.



## 1. Fundamentals

**Flow Matching (FM)** constructs a deterministic coupling between a prior $p_{\text{prior}}(x_1)$ and a data distribution $p_{\text{data}}(x_0)$.
* **Interpolant:** $x_t = (1-t)x_0 + tx_1$ for $t \in [0, 1]$.
* **Conditional Velocity:** $v_t(x_t | x_0, x_1) = \frac{d}{dt}x_t = x_1 - x_0$.
* **Objective:** Regress a neural vector field $u_\theta$ against the conditional velocity.
    $$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, x_0, x_1} \|u_\theta(x_t, t) - (x_1 - x_0)\|^2$$
At optimality, $u_\theta(x_t, t)$ matches the marginal velocity field. Generation requires iterative numerical ODE integration (many NFEs).



**Flow Maps** circumvent iterative integration by parameterizing the flow's displacement operator.
* **Definition:** $\psi(x_t, t, s) = x_t + \int_t^s u(x_\tau, \tau) d\tau$.
* **Parameterization:** $f_\theta(x_t, t, s) = x_t + (s-t)F_\theta(x_t, t, s)$, where $F_\theta$ approximates the average velocity over the interval $[t, s]$.

## 2. Progression of Distillation Modalities

1.  **Trajectory-Based Distillation (Consistency Models, early Flow Maps):** Student directly regresses the teacher's deterministic ODE trajectory. *Limitation:* Compounding integration errors and high sensitivity to teacher curvature.
2.  **Distribution-Based Distillation (DMD, VSD):** Drops strict trajectory matching for direct distributional alignment (e.g., GAN loss, reverse KL). *Limitation:* Computationally expensive fake score estimation.
3.  **Data-Free & Analytic Frameworks (FreeFlow, MeanFlow, TVM, MFM):** Eliminate dependency on empirical dataset sampling by anchoring distillation strictly at the prior (FreeFlow), exploiting exact calculus identities (MeanFlow, TVM), or modeling true stochastic posteriors analytically (MFM).

## 3. Core Derivations and Objectives

### A. MeanFlow: Exact Average Velocity Matching
*Motivation:* Derives an analytic identity connecting average velocity to instantaneous velocity, enabling 1-NFE training without a pre-trained teacher.



**Derivation:**
Define average velocity $u(z_t, r, t) = \frac{1}{t-r}\int_r^t v(z_\tau, \tau) d\tau$.
Differentiate the integral w.r.t upper limit $t$ using the Leibniz rule:
$$\frac{d}{dt}\left[ (t-r)u(z_t, r, t) \right] = v(z_t, t)$$
Expand the total derivative $\frac{d}{dt}$:
$$u + (t-r)(\partial_t u + v \cdot \nabla_{z_t} u) = v(z_t, t)$$
**Objective:** The Jacobian-Vector Product (JVP) computes $\frac{d}{dt}u_\theta(z_t, r, t)$. The model regresses $u_\theta$ against the isolated $v_t$ term.
$$\mathcal{L}_{MF}(\theta) = \mathbb{E}_{t,r,x_0,x_1} \|u_\theta(z_t, r, t) - \text{sg}\left(v_t - (t-r)\frac{d}{dt}u_\theta(z_t, r, t)\right)\|^2$$

### B. FreeFlow: Data-Free Flow Map Distillation
*Motivation:* Static dataset trajectories induce "Teacher-Data Mismatch" under Classifier-Free Guidance. FreeFlow isolates distillation to the prior ($t=1$) where teacher/student distributions are identical.

**Derivation:**
Let $z \sim p_{\text{prior}}$ at $t=1$. Let integration interval be $\delta = 1-s$. The optimal flow map $F_{\theta^*}$ satisfies:
$$\delta F_{\theta^*}(z, \delta) = \int_1^{1-\delta} -u(x(\tau), \tau) d\tau$$
Differentiate w.r.t $\delta$:
$$F_{\theta^*}(z, \delta) + \delta \partial_\delta F_{\theta^*}(z, \delta) = u(f_{\theta^*}(z, \delta), 1-\delta)$$
**Objective:** (Predictor-Corrector)
$$\mathcal{L}_{pred} = \mathbb{E}_{z,\delta} \left\| F_\theta(z, \delta) + \delta \partial_\delta F_\theta(z, \delta) - u_{\text{teach}}(f_\theta(z, \delta), 1-\delta) \right\|^2$$
$$\mathcal{L}_{corr} = \mathbb{E}_{z,\delta,r,\epsilon} \left\| v_{\text{student}}(f_\theta(z,\delta)_r, r) - u_{\text{teach}}(f_\theta(z,\delta)_r, r) \right\|^2$$

### C. Terminal Velocity Matching (TVM)
*Motivation:* Differentiating displacement at the initial bound $t$ ignores terminal geometry. Differentiating at terminal bound $s$ upper-bounds the 2-Wasserstein distance to the data distribution.



**Derivation:**
Displacement $f(x_t, t, s) = \int_t^s u(x_r, r)dr$.
Differentiate w.r.t the terminal bound $s$:
$$\frac{d}{ds}f(x_t, t, s) = u(x_t + f(x_t, t, s), s)$$
**Objective:** Ground-truth terminal velocity $u$ is intractable, so substitute a proxy instantaneous velocity model $u_\theta$, jointly co-trained:
$$\mathcal{L}_{TVM}(\theta) = \mathbb{E} \left[ \left\|\frac{d}{ds}f_\theta(x_t, t, s) - u_\theta(x_t + f_\theta(x_t, t, s), s)\right\|_2^2 + \|u_\theta(x_s, s) - v_s\|_2^2 \right]$$

### D. Meta Flow Maps (MFM)
*Motivation:* RLHF/Reward alignment requires $\nabla V_t$, demanding differentiable samples from the stochastic conditional posterior $p_{1|t}(\cdot | x_t)$. Deterministic flow maps collapse this variance. MFM amortizes an infinite family of auxiliary flow ODEs to match $p_{1|t}$.



**Derivation:**
Construct an auxiliary ODE mapping noise $\epsilon \sim p_0$ to $p_{1|t}(\cdot | x_t)$.
Parameterize via $\bar{X}_{s,u}(\bar{x}; t, x_t) = \bar{x} + (u-s)\hat{v}_{s,u}(\bar{x}; t, x_t)$.
**Objective:** Analytically derive the true conditional ODE velocity $\bar{b}_s$ via GLASS flows. Match diagonal and enforce self-consistency.
$$\mathcal{L}_{diag} = \int_0^1 \mathbb{E} \|\hat{v}_{s,s}(\bar{I}_s; t, x_t) - \bar{b}_s(\bar{I}_s; t, x_t)\|^2 ds$$
$$\mathcal{L}_{cons} = \mathbb{E} \|\partial_s \bar{X}_{s,u} + \hat{v}_{s,s} \cdot \nabla \bar{X}_{s,u}\|^2$$

### E. Transition Matching Distillation (TMD)
*Motivation:* Video diffusion state spaces are too high-dimensional for direct trajectory distillation. TMD decouples the transformer backbone and flow head, chunking continuous trajectories into few-step discrete probability transitions.



**Architecture:**
Outer transition: $x_{t_{i-1}} = x_{t_i} - (t_i - t_{i-1})y$.
Inner flow parameterization: $y_{s_{j-1}} \approx f_\theta(y_{s_j}, s_j, s_{j-1}; m_\theta(x_{t_i}, t_i))$, where $m_\theta$ is the heavy semantic backbone and $f_\theta$ is a lightweight flow head.
**Objective:**
Stage 1 (Pretraining): Apply MeanFlow objective *strictly* to the isolated flow head $f_\theta$.
$$\mathcal{L}_{TM} = \left\| u_\theta(y_s, s, r; m) - \text{sg}\left(v - (s-r)\frac{d}{ds}u_\theta\right) \right\|^2$$
Stage 2 (Distribution Matching): Unroll inner transitions and optimize via variational score distillation (DMD2-v) against the full generative pipeline.