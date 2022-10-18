---
layout: post
title:  "Knowledge Distillation Survey"
date:   2022-10-10 11:11:31 +0900
category: [Research, Featured]
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

### Core Takeaways
1. Focus on Feature-based knowledge
2. Mix offline, online distillation
3. Adversarial Training
4. Attention-based distillation

***

## FEATURE-MAP-LEVEL ONLINE ADVERSARIAL KNOWLEDGE DISTILLATION
[Source](https://openreview.net/attachment?id=Bkl086VYvH&name=original_pdf)
ICLR 2020  

### Abstract
Online knowledge distillation that transfers the knowledge of the feature map using an adversarial training framework.  
By training a discrimination network to distinguish the featuremaps from different networks and training networks to fool it, networks can learn the other network's feature map distribution.  
Cyclic learning for training more than two networks together.  

### Introduction
Online distillation: training networks to learn from each other.  
First to apply feature map based knowledge distillation in online learning.  
![Fig2-1](/assets/posts/research/2.knowledge-distillation-survey/fig2-1.png "Fig2-1")
Direct alignment method only tries to minimize the distance between feature map points and ignores distributional differences.  

Newly proposed feature map based loss to distill the feature map indirectly via discriminators.  

### Related works
1. Model Compression
    - Knowledge distillation by Hinton et al. (2015)
        - Used softened logit (with termperature) which has higher entropy
        - Learn with conventional CE loss with labeled data and with the final outputs of a teacher network.  
    - Feature representation
        - [FitNet(2014)](/https://arxiv.org/abs/1412.6550), ATA(Zagoruyko 2016a), FT(Kim 2018), KTAN(Liu 2018) use intermediate feature representation to transfer knowledge
2. Online Knowledge Distillation
    - [DML (Zhang 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf) 
        - Ensemble of students learn from each other. Achieves results even better than offline Knowledge Distillation in one benchmark
    - ONE (Lan 2018)
        - Rather than mutually distilling between the networks, ONE generates a gated ensemble logit of the training networks and uses it as a target to align for each network.  

    - Current drawbacks
        - Dependent only on the logit and do not make any use of the feature map information
3. Proposed Method
- Adversarial training to let each to learn each other's distribution.  
- Cyclic learning scheme for training more than two networks simultaneously. The number of discriminators is K (number of networks participating)  
    - One-way cyclic manner.  

### Core Takeaways
1. Use a discriminator when training online distillation with feautre-based knowledge

***

## Show, Attend and Distill:Knowledge Distillation via Attention-based Feature Matching
[Source](https://ojs.aaai.org/index.php/AAAI/article/download/16969/16776)  
2021 AAAI

### Abstract
Most studies manually tie intermediate features of the teacher and student and transfer knowledge through pre-defined links.  
Proposed method utilizes an attention-based meta-network that learns relative similarities between features, and applies identified similarities to control distillation intensities of all possible pairs.  
As a result, the proposed method determines competent links more efficiently than previous approaches.  

### Introduction
Most studies manually link the teacher and student features and perform distillation through the links individually.  
This manual selection does not consider the similarity between the teacher and student features, which risks forcing an incorrect intermediate processs to the student.  
Also the link selection limits fully utilizing the whole knowledge of the teacher by choosing a few of all possible linkns.  

Feature link selection: Jang et al 2019.  
New feature linking method proposed based on the attention mechanism.  

### Core Takeaways
1. Link features using an attention mechanism

***

## Online Knowledge Distillation via Collaborative Learning
[Srouce](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Online_Knowledge_Distillation_via_Collaborative_Learning_CVPR_2020_paper.pdf)

### Abstract
Knowledge Distillation via Collaberative Learning: KDCL
- Treats all models as students and collaboratively trains them. 

### Introduction
- Students directly learn from the prediction of other students in Deep Mutual Learning
    - Output of students could conflict with each other and does harm to model with high performance.  
- ONE: multi-branch network while establishing teacher on the fly.  
    - Knowledge transfer only accurs at the upper layers; inflexible
    - Gate module is the soft target; which is not a guarantee

### Core Takeaways
1. Student pool training
