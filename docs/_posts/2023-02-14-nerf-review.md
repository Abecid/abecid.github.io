---
layout: post
title:  "NeRF Overview"
date:   2023-02-14 10:20:31 +0900
category: Blog
tags: [Deep Learning, Computer Vision, NERF, Survey]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# NeRF Overview
Novel View Synthesis

## 1. Background
### 1-1. Singed distance functions  
- Given a spatial [x, y, z] point as input, SDFs will output the distance from that point to the nearest surface of the underlying object being represented.
- negative sign: inside, positive sign: outside the surface
- $$$$ SDF(x) = s:x \subseteq \mathbb{R}^{3}, s \in \mathbb{R} $$$$
> **Usefulness**
> Stores a function instead of a direct representation of 3D shape (more efficient)




Applications