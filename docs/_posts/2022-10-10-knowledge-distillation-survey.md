---
layout: post
title:  "Knowledge Distillation Survey"
date:   2022-10-04 11:11:31 +0900
category: Research
tags: [Deep Learning, Computer Vision, Knowledge Distillation, Survey]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

[Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)
May 20, 2021

Abstract
Model compression and acceleration.
Knowledge distillation- small student model learning from a larger teacher model.
Knowledge distillation from the perspectives of knowledge categories, training schemes, teacher-student architecture, distillation algorithm.

1. Introduction
- Efficient deep models
    - Efficient building blocks: epthwise separable covolution
        - MobileNets
        - SuffleNets
    - Model compression and acceleration techniques
        - Parameter pruning and sharing
        - Low-rank factorization
        - Transferred compact convolutional filters
        - Knowledge distillation

Knowledge distillation: small student model supervised by a large teacher model

[Schematic structure image, Fig2]

2. Knowledge
    1. Response-based knowledge
    [Fig4]
    p: softmax with temperature
    loss: Kullback Leibler divergence loss
    2. Feature-based knowledge
    L2, L1, Cross-entropy, maximum mean discrepancy loss
    3. Relation-based knowledge
    Relationship between feature maps
    Equation: Correlation function of similarity functions for each model's feature maps

3. Distillation Schemes
    1. Offline Distillation
        - Teacher model trained
        - Teacher model is used to extract knowledge
        Focus on improving knowledge transfer
        - Design of knowledge
        - Loss function: features, distributions
    2. Online Distillation
    Teacher and student model updated simultaneously.
    (When a large-capacity high performance teacher model is not available)
    3. Self-Distillation
    Same networks used for the teacher and student models
    (Special case of online distillation)
    Self-attention distillation
        - Attention maps of its own layers as distillation targets for lower layers
    Snapshot distillation
        - Knowledge in the earlier epochs transferred to later epochs
        - Early exit layer tries to mimic the output of layer exit layers

4. Teacher-Student Architecture
    - Mostly the student network is smaller
    - Neural architecture search in knowledge distillation

5. Distillation Algorithms
    1. Adversarial Distillation
        - Adversarial generator used to generate synthetic data
        - Discriminator used to distinguish between student and teacher outputs
        - Teacher and student jointly optimized in online manner
    2. Multi-Teacher distillation
    3. Cross-Modal Distillation
    4. Graph based Distillation
    5. Attention-based Distillation
