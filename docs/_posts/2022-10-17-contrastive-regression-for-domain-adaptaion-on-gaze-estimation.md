---
layout: post
title:  "Contrastive Regression for Domain Adaptation on Gaze Estimation"
date:   2022-10-17 14:39:31 +0900
category: Paper Review
tags: [Deep Learning, Computer Vision, Gaze Estimation, Domain Adaptation, Contrastive Learning]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Contrastive Regression for Domain Adaptation on Gaze Estimation  
[Source Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Contrastive_Regression_for_Domain_Adaptation_on_Gaze_Estimation_CVPR_2022_paper.pdf)  
CVPR 2022

![Title](/assets/posts/paper_review/7.contrastive_domain_gaze/title.png "Title")  

## Abstract
Appearance-based Gaze Estimation depends on expensive and cumbersome annotation capture. When lacking precise aannotation, the large domain gap hinders the performance of trained models on new domains.  

A novel gaze adaptation appraoch, Contrastive Regression Gaze Adaptation (CRGA) for generalizing gaze estimation on the target domain in an unsupervised manner.  

- CRGA leverages
    - Contrastive Domain Generalization (CDG) module to learn the stable representation from the source domain
    - Contrastive Self-training Adaptation (CSA) module to learn from the psudo labels on the target domain

- Contrastive Regression (CR) loss
    - Novel contrastive loss of regression by pulling features with closer gaze directions closer together while pushing features with farther gaze directions farther apart.  

- Experimentation
    - Source domain: ETH-XGAZE, Gaze-360
    - Target domain: MPIIGAZE, RT-GENE, Gaze-Capture, EyeDiap respectively.
    - Remarkable performance improvements compared with the baseline models and also outperforms the state-of-the-art domain adaptation approaches on gaze adaptation tasks.


## 1. Introduction
![Fig1](/assets/posts/paper_review/7.contrastive_domain_gaze/fig1.png "Fig1")  
**Despite the success of appearance-based gaze estimation, expensive and cumbersme annotation capture contraints its application.**  
- Large-scale datasets have been proposed to alletiae this problem.
    - Promising performance in teh within-dataset test (training and testing from same dataset)
    - Degraded daramatically in cross-dataset test, due to the gap between different domains (different subjeccts, background environments, illuminations)

- Other works
    - Narrow the cross-dataset gap
        - Collaborative model ensembles
        - Additional annotations
    - Inter-person gap
        - Learn the personal error between visual axis and optical axis
            - Adversarial training
            - Few shot learning
    - Drawbacks
        - Lacks a self-supervised approach to address the cross-dataset gap

- Existing unsupervised and supervised contrastive learning for classification cannot accomodate to gaze regression tasks.  
    - Unsupervised contrastive learning
        - positive: different views of same image; negative: views of other images
        - Extracts global semantic information that benefits classification tasks
        - However, global semantic information could mislead regression tasks
    - Supervised contrastive learning

> CRGA proposed for generalizing gaze estimation on the target domain in an unsupervsied maner  
> Contrastive Regression (CR) loss to learn robust domain-invariant representation for regression tasks

- First to introduce contrastive learning into regression tasks to improve domain generalization and adaptation performance.

## 2. Related Works
### 2-1. Domain Adaptive Gaze Estimation
- Performance is degraded on new domains
    - Large-scale diverse datasets
- Align data distribution cross domain
    - Data preprocessing to normalized space
    - GAN methods
    - Rotation-aware latent representation of gaze with meta-learning
    - Adversarial training
    - Ensemble of networks to learn collaboratively with the guidance of outliers

### 2-2. Constrastive Learning
> Surpasses supervised methods when transferring the representation to cross-domain and downstream tasks

## 3. Methodology
### 3-1. Preliminary: Domain Adaptation
Predictive Function  
$$f : x \rightarrow g$$  

Learn $$f$$ on the source domain $$S$$ to achieve minimum error on the target domain $$T$$  
$$\min_{f}\mathbb{E}_{(x^{\tau}, y^{\tau})}[L(f(x^{\tau}),g^{\tau})]$$

### 3-2. Contrastive Regression
Propose a novel contrastive regression framework to learn robust and invariant representation for regression tasks.  

Relationship between labels reveal the relationship between the features.  

- CR loss  
    $$-log\dfrac{\sum_{k}\sigma(S_{i,k}) \cdot f_{k}(y_{k},x)}{\sum_{j}|S_{i,j}| \cdot f_{j}(y_{j},x)}$$  
    > Pull features with closer gaze directions closer together while pushing features with farther gaze directions farther apart.  

- Similarity function
    - gradient near zero of cosine similarity is too small
    - -log KL function as similarity  
    $$S_{i,j}=log\dfrac{0.07}{|g_{i}-g_{j}|}$$

### 3-3. Contrastive Regression Gaze Adaptation
![Fig2](/assets/posts/paper_review/7.contrastive_domain_gaze/fig2.png "Fig2")  
CRGA consists of two modules
- Contrastive Domain Generalization (CDG)
    - Learn stable representations from the source domain using CDG loss
- Contrastive Self-training Adaptation (CSA)
    - Uses contrastive self-training with pseudo labeling
    - Improve adaptation performance on the target domain

#### 1) Contrastive Domain Generalization
Data augmentation oeprators: $$A, \tilde{A}$$  
Input Images  
$$I = A(\textit{input}), \tilde{I}=\tilde{A}(\textit{input})$$  

Get feeatures with $$f(\cdot)$$  
$$V = f(I), \tilde{V} = f(\tilde{I})$$  

Map features to projection space with projection head $$r(\cdot)$$   
$$z = r(v)$$  

Introduce the cosine similarity for the $$\ell_{2}$$ normalized z with the temperature parameter $$\tau$$
$$f_{k}(u_{k},x) into exp(sim(z_{i},z_{k}/\tau))$$ 

$$L_{CDG}$$  
![Eq7](/assets/posts/paper_review/7.contrastive_domain_gaze/eq7.png "Eq7")

#### 2) Contrastive Self-training Adaptation
Target model generates pseudo gaze direction  
$$\tilde{g}^{\tau} = r^{\tau}(f^{\tau}(x_{\tau}))$$

Use $$\tilde{g}^{\tau}$$  as the label of the target data  
$$L_{CSA}$$
![Eq8](/assets/posts/paper_review/7.contrastive_domain_gaze/eq8.png "Eq8")

Source data used as the regularizaiton term with an annealed temperature $$\gamma$$ (from 0 to 1)
![Eq9](/assets/posts/paper_review/7.contrastive_domain_gaze/eq9.png "Eq9")

## 4. Experiments
