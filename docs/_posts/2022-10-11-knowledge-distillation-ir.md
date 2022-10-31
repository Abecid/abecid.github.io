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
Appearance-based gaze estimation has been very successful with the use of deep learning. With ever more datasets and bigger models the state-of-the-art is constantly improving. Even though there have been much progress in domain generalization for gaze estimation, most of the recent work have been focused on cross-dataset performance, where illuminations, head pose, and lighting are different. Also there are clear limitations relying solely on supervised learning for domain adaptation. This paper proposes a novel framework for gaze estimation with near-infrared images using contrastive learning. We propose a novel contrastive loss function that effectively clusters the features of different samples in the latent space and incorporate it into our novel framework for near infrared based gaze estimation. Our model outperforms previous domain generalization models in Infrared based gaze estimation and with thorough experiments in ablation studies we prove the efficacy of our method. 


## 1. Introduction
Gaze estimation is a crucial technology that has many critical applications including in virtual reality medical analysis. Appearance based gaze estimation leveraging deep learning and convolutional neural networks have gained great traction over the past several years. There have been much progress in appearance based gaze estimation via supervised learning and domain adaptation for cross-dataset performance accounting for different lighting, illuminations, and head poses. However, near-infrared gaze estimation has not gained much traction in the literature. With broader usage of gaze estimation in edge devices, there is increasing need for gaze estimation in the dark. And infrared images are much more reliable in these settings. 

To the best of our knowledge, we are the first to improve gaze estimation with near infrared images with appearance based models and contrastive learning. 

We propose a novel framework for gaze estimation with near infrared images and propose a novel contrastive loss function for better clustering of features in the latent space. 

Our model achieves significant performance improvements compared to existing domain generalization methods. 


## 2. Related Works
### 2.1. Appearance based gaze estimation
Appearance based gaze estimation

### 2.2. Domain adaptation

### 2.3. Contrastive learning
NT-Xent
SimCLR
Contrastive Regression Loss

## 3. Method
### 3.1. Framework
Backbone gaze estimation architecture following iTracker. 

Pretrain Teacher model with RGB Image dataset
Generate RGB Image Pseudo label

Data augmentation to create positive pairs or an IR image and an black and white image converted from a RGB image taken at the same time. 

L2 loss for supervised learning

### 3.2. Contrastive regression loss for gaze estimation
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

### 3.3. Final loss
Finally a new training architecture. 
- Total Loss: CRL + Supervised Loss

- Domain Discriminator + Attention in Feature Map Latent Vector

## 4. Experiments
### 4.1. Datasets
### 4.2. Training Details
### 4.3. Near infrared performance
### 4.4. Ablation Study
Epoch
Hyperparameter Tuning
Loss function

## 5. Conclusion

### Citations
