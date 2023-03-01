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
- $$ SDF(x) = s:x \in \mathbb{R}^{3}, s \in \mathbb{R} $$
> **Usefulness**
> Stores a function instead of a direct representation of 3D shape (more efficient)

### 1-2. DeepSDF
![Demo](/assets/posts/blog/5.nerf/deepsdf.png "Demo")  
- Use a feedforward neural network to project SDF from (x,y,z)
- Represent a shape
  - $$X := {(x,s) : SDF(x) = s} $$ where x is [x,y,z]
- Training
  ![Train](/assets/posts/blog/5.nerf/train.png "Train")  
  ![autodecoder](/assets/posts/blog/5.nerf/autodecoder.png "autodecoder")  
> Different from auto-encoders, auto-decoders directly accept a latent vector as input. 
- Inference
  1. Get samples of SDF values
  2. Determine the best latent vector for those samples
  3. Run inference on the samples to get the SDF values
  4. Visualize the shape via [Maching Cubes](https://graphics.stanford.edu/~mdfisher/MarchingCubes.html#:~:text=Marching%20cubes%20is%20a%20simple,a%20region%20of%20the%20function.)
- Limitations
  - Requires 3D geometry to run inference
  - Searching for the best latent vector is expensive

### 1-3. ONets

Applications