---
layout: post
title:  "Generalizing Gaze Estimation with Rotation Consistency"
date:   2022-10-18 13:55:31 +0900
category: Paper Review
tags: [Deep Learning, Computer Vision, Gaze Estimation]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Generalizing Gaze Estimation with Rotation Consistency  
[Source Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Bao_Generalizing_Gaze_Estimation_With_Rotation_Consistency_CVPR_2022_paper.pdf)  
CVPR 2022

![Title](/assets/posts/paper_review/8.generalizing_gaze_rotation/title.png "Title")  

## Abstract
Generalizing gaze estimation to unseen environments is still challenging.  
Authors discover the rotation-consistency property in gaze estimation.  
Introduce the 'sub-label' for unsupervised domain adaptation.  
Authors propose the Rotation-enhanced Unsupervised Domain Adaptation (RUDA) for gaze estimation.  
- Rotate the original images with different angles for training
- Conduct domain adaptation under the constraint of rotation consistency
- Target domain images are assigned with sub-labels, derived from relative rotation angles rather than untouchable real labels
- A novel distribution loss that facilitates the domain adaptation
Evaluation of RUDA framework on four cross-domain tasks

## 1. Introduction
- Existing appearance-based gaze estimation methods suffer from severe performance degradation when adapting to new domains, cased by:
    - Subject appearance
    - Image quality
    - Shooting angle
    - Illumination

**Target domain labels are not usually accessible, cannot directly train the gaze estimator in target domain.**  
- Unsupervised domain adaptation aim to find a gaze-relevant constraint generalizing the model to target domain without label.  
    - Domain discriminator by adversarial learning
    - Appearance discriminator and a head pose classifier for adaptation
    - Guide the model with outliers

> Authors find that human gaze, as a 3D direction vector, is rotation-consistent.  
Define the relative rotation angle as the sub-label (not an absolute angle)  

**Authors sropose Rotation-enhanced Unsupervised Domain Adaptation (RUDA) framework for gaze estimation.**  
- Creates sub-labels between original and randomly rotated images
- Estimator is generalized to target domain via rotation consistency of estimation results

- Contributions
    1. RUDA
        - Trains a rotation-augmented model in source domain
        - Adapt the model to target domain with physically-constrained gaze directions
    2. Found rotation consistency property
        - Used to generate sub-labels for unsupervised gaze adaptation
        - Design a novel distribution loss which supervise the model with rotation consistency and sub-labels
    3. Experimental results
        - RUDA achieves consistent improvement over the baseline model on four cross-domain gaze estimation tasks
        - Even outperforms some state-of-the-art methods trained on target domain with labels

## 2. Related Work
1. Gaze Estimation
    - Calibration-free appearance-based gaze estimation with CNN
        - Various methods using different outputs
            - Eye images
            - Face images
            - Both
    - Cross-domain gaze estimation
        - Person-specific gaze estimation network with few sampels via meta-learning
        - Eliminate inter-personal diversity by ensuring predictino consistency
        - Eliminate gaze-irrelevant feature; improve cross dataset accuracy without target domain data
        - Plug-and-play cross-domain gaze estimation with the guidance of outliers
        - Redirect head and gaze in a self-supervised manner by embedding transformation including rotation
    
2. Unsupervised Domain Adaptation
    - Representation subspace distance (RSD) that aligns features from two domains for regression tasks
    - Adversarial learning

    > Most previous methods are designed for classification tasks instead of regression tasks.  
    > UDA for gaze estimation still remains to be explored

## 3. Rotation Consistency in Gaze Estimation
Main challenges in unsupervised gaze adaptation tasks
- Shortage of target domain samples
- Absence of ground truth labels

Augmentations proposed
- Color jittering
- Introducing noise

However, existing data augmentation appraoches only bring limited performance improvement, when adopted in unsupervised gaze adaptation tasks

**Rotation Consistency**  
![Eq1](/assets/posts/paper_review/8.generalizing_gaze_rotation/eq1.png "eq1")  
F: Gaze mapping function from the image to gaze direction
I: Input image
R: Rotation matrix of the input image
$$R^{g}$$: Rotation matrix of the gaze direction

![Eq2](/assets/posts/paper_review/8.generalizing_gaze_rotation/eq2.png "eq2")  
![Eq3](/assets/posts/paper_review/8.generalizing_gaze_rotation/eq3.png "eq3")  

## 4. Method
### 4-1. Task Definition
### 4-2. Rotation-Enhanced Unsupervsied Domain Adaptation for Gaze Estimation
### 4-3. Implementation Details

## 5. Experiments
