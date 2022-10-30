---
layout: post
title:  "General Contrastive Regression Loss for IR"
date:   2022-10-11 17:11:31 +0900
category: [Research]
tags: [Deep Learning, Computer Vision, Knowledge Distillation, IR, Unfinished]
header-includes:
   - \usepackage{bbm}
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Abstract
Simple Framework for IR image based gaze estimation. 
Gaze estimation appearance based leveraging deep neural networks. 
Knowledge distillation and transfer learning via supervised learning has limits as the feature space is different. Here we adopt a contrastive regression loss to improve the feature extractor and improve the performance of IR image based gaze estimation based on supervised baseline models. We were able to improve performance by 30%. 

## Introduction
Simple Framework for IR image based gaze estimation. 
Gaze estimation appearance based leveraging deep neural networks. 
Knowledge distillation and transfer learning via supervised learning has limits as the feature space is different. Here we adopt a contrastive regression loss to improve the feature extractor and improve the performance of IR image based gaze estimation based on supervised baseline models. 

## Related Works
Appearance based gaze estimation

Knowledge distillation

Contrastive loss
NT-Xent
SimCLR
Contrastive Regression Loss

## Method
Backbone gaze estimation architecture following iTracker. 
Data augmentation to create positive pairs or an IR image and an black and white image converted from a RGB image taken at the same time. 

Supervised loss, l2 loss on the 2-dimensional camera plane. 
Propose a novel contrastive regression loss function for gaze estimation. 
Contrary to classification tasks, distancing method has to be scaled relative to the sample's proximity to the anchor sample inferred by the labels[CRL]. Also it should be scaled accordingly as the similarities between samples are relatively similar between gaze samples. 

Normalized Temperature-scaled Cross Entropy Loss  
$$ \mathbb{l}_{i,j} = -\log\frac{\exp\left(\text{sim}\left(\mathbf{z}_{i}, \mathbf{z}_{j}\right)/\tau\right)}{\sum^{2N}_{k=1}\mathcal{1}_{[k\neq{i}]}\exp\left(\text{sim}\left(\mathbf{z}_{i}, \mathbf{z}_{k}\right)/\tau\right)} $$

$$ \frac{\sum_{j} exp(sim(z_{i}, z_{j})/\tau)}{\sum_{k}\mathbb{1}_{k \ne i}\lvert S_{i, k} \rvert \cdot exp(sim(z_{i}, z_{k})/\tau)} $$

$$ S = -log \frac{sim(g_{i}, g_{k})}{cos(\pi/60)} $$

Assume 3 degrees or less are the same

$$ \sigma $$ is the Relu activation function. 

Wang et el. proposed contrastive regression loss with similarity weight.

Here propose a different similarity pair based on labeled similarity. 

Also a different similarity function. 

Finally a new training architecture. 
- Pretrain Teacher model with RGB Image dataset
- Generate RGB Image Pseudo label
- Find cosine similarity between pairs in the batch (stochastically)
- Domain Discriminator + Attention in Feature Map Latent Vector
- Total Loss: CRL + Supervised Loss

## Experiments

## Ablation Study
Epoch
Hyperparameter Tuning

### Architecture

### Methodologies

### Math

### Performance

### Challenges, Future Directions

### Citations
