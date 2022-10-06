---
layout: post
title:  "Gaze Estimation Unsupervsed Learning Survey"
date:   2022-10-04 11:11:31 +0900
category: Research
tags: [Deep Learning, Computer Vision, Gaze, Gaze Estimation, Self-Supervised Learning, Unsupervised Learning, Survey]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

[Unsupervised Representation Learning for Gaze Estimation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Unsupervised_Representation_Learning_for_Gaze_Estimation_CVPR_2020_paper.pdf)
CVPR 2020
1. Abstract
- Learn a low dimensional gaze representation without gaze annotations. Use a Gaze redirection network. 
- Use the gaze representation difference of the input and target images (of the redirection network).
- Redirection loss allows the joint training of both the redirection network and the gaze representation network. Also uses warping field regularization.
- Promising results on few-shot gaze estimation: competitive results with less than 100 calibration samples.
- Cross-dataset gze estimation, gaze network pretraining demonstrate the validity of the proposed framework

2. Intro
- Gaze estimation is important and can be used in different ways
- Data annotation is costly
- Authors propose an unsupervised approach with few calibration samples.
- Gaze redirection network

3. Key Points

