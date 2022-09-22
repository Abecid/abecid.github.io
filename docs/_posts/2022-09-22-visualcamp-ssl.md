---
layout: post
title:  "Visualcamp : SSL"
date:   2022-09-22 15:48:31 +0900
category: Research
tags: [Deep Learning, Computer Vision, Self-Supervised Learning, Knowledge Distillation, Visualcamp]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<h2>Intro</h2>
Self-supervised learning

IR and RGB loss (taken at same instance)

IR-IR with similar x,y points
RGB-RGB with similar x,y points

IR-RGB with similar x,y points

loss function
- cross entropy
- mse
- further modification
    - apply both gradients to teacher with alpha parameter

architecture
- 1 model (shared weights) v. 2 models (teacher and student model)
- more layers?


