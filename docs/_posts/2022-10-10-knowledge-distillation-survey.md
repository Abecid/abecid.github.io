---
layout: post
title:  "Knowledge Distillation Survey"
date:   2022-10-10 11:11:31 +0900
category: Research
tags: [Deep Learning, Computer Vision, Knowledge Distillation, Survey]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Knowledge Distillation: A Survey
[Source Paper](https://arxiv.org/abs/2006.05525)
May 20, 2021

### Abstract
- Model compression and acceleration.
- Knowledge distillation: small student model learning from a larger teacher model.
- Knowledge distillation from the perspectives of knowledge categories, training schemes, teacher-student architecture, and distillation algorithms.

### 1. Introduction
- Efficient deep models
    1. Efficient building blocks
        - depthwise separable covolution
            - MobileNets
            - SuffleNets
    2. Model compression and acceleration techniques
        - Parameter pruning and sharing
        - Low-rank factorization
        - Transferred compact convolutional filters
        - Knowledge distillation

Knowledge distillation: small student model supervised by a large teacher model

![Schematic structure image, Fig2](/assets/posts/research/2.knowledge-distillation-survey/fig2.png "Fig2")

### 2. Knowledge Categories
![Knowledge Types, Fig3](/assets/posts/research/2.knowledge-distillation-survey/fig3.png "Fig3")
1. Response-based knowledge
    ![Fig4](/assets/posts/research/2.knowledge-distillation-survey/fig4.png "Fig4")
    > Mimic the final prediction of the teacher model

    z (vector of logits): outputs of the last fully connected layer of a model  
    Distill Loss:  
    $$L_{R}=(z_{t}, z_{s})$$ (diveregence loss of logits)  

    - Image Classifinication  
        p: softmax with temperature  
        loss: Kullback Leibler divergence loss  
        ![Fig4](/assets/posts/research/2.knowledge-distillation-survey/eq1.png "eq1")  
        ![Fig4](/assets/posts/research/2.knowledge-distillation-survey/eq2.png "eq2")
2. Feature-based knowledge  
    Deep neural nets are good at learning feature representations with multiple layers.  
    ![Fig6](/assets/posts/research/2.knowledge-distillation-survey/fig6.png "Fig6")
    - Loss functions used for distill loss
        - L1
        - L2
        - Cross-entropy
        - Maximum Mean Discrepancy Loss
3. Relation-based knowledge  
    Previous knowledges use outputs of specific layers of the teacher model.  
    Relationship-based knowledge uses the relationship between feature maps  
    ![Fig7](/assets/posts/research/2.knowledge-distillation-survey/fig7.png "Fig7")

    - Flow of Solution Processes (FSP)
        - Proposed by [Yim et al. (2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf) to explore the relationships between different feature maps
        - Defined by the Gram matrix between two layers
        - Summarizes the relations between pairs of feature maps. Calculated using the inner products between features from two layers.
        - With the correlations between feature maps, knowledge distillation via singular value decomposition was proposed to extract key information in the feature maps.
    Correlation function of similarity functions for each model's feature maps  
    ![Eq5](/assets/posts/research/2.knowledge-distillation-survey/eq5.png "eq5")


### 3. Distillation Schemes
1. Offline Distillation
    - Teacher model trained
    - Teacher model is used to extract knowledge  
    Focus on improving knowledge transfer  
    - Design of knowledge
    - Loss function: features, distributions
2. Online Distillation
    Teacher and student model updated simultaneously.  
    Knowledge distillation framework is end-to-end trainable.  
    (When a large-capacity high performance teacher model is not available)
3. Self-Distillation
Same networks used for the teacher and student models
(Special case of online distillation)
Self-attention distillation
    - Attention maps of its own layers as distillation targets for lower layers
Snapshot distillation
    - Knowledge in the earlier epochs transferred to later epochs
    - Early exit layer tries to mimic the output of layer exit layers

### 4. Teacher-Student Architecture
- Mostly the student network is smaller
- Neural architecture search in knowledge distillation

### 5. Distillation Algorithms
1. Adversarial Distillation
    - Adversarial generator used to generate synthetic data
    - Discriminator used to distinguish between student and teacher outputs
    - Teacher and student jointly optimized in online manner
2. Multi-Teacher distillation
3. Cross-Modal Distillation
4. Graph based Distillation
5. Attention-based Distillation
