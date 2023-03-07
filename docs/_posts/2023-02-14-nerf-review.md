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
- **Usefulness**  
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
- Strengths
  - Compression
    - More efficient thatn voxels and mesh
  - Fix broken geometry
    - Given partial or noisy representation, recover an accurate mesh
  - Intepolating latent space
    - Produce new shapes by interpolating latent vectors
- Limitations
  - Requires 3D geometry to run inference
  - Searching for the best latent vector is expensive

### 1-3. ONets

### 1-4. Scene Representation Networks

## 2. NeRF
### 2-1. Introduction
![ner](/assets/posts/blog/5.nerf/nerf-image2.png "nerf-image2")  
- When an image is taken, a camera is at a certain pose (orientation and angle in the world coordinate system). Novel view synthesis is the domain of generating an image with an arbitrary target pose based on the provided source images and their respective camera poses[1].
- Process
    1. Conducting ray tracing for each pixel to generate a sample set of 3D points.
    2. Using those generated points and the 2D viewing direction as 5D inputs to the neural network to produce a 4D output of colors and density.
    3. Use classical volume rendering techniques to accumulate the output to synthesize a 2D image.
    ![fig2](/assets/posts/blog/5.nerf/fig2.png "f2")  
    - Neural Radiance Field Scene Representation
        - (a) Synthesize images by sampling 5D coordinates(location and viewing direction) along camera rays 
        - (b) Feed those locations into an MLP to produce a color and volume density
        - (c) Use volume rendering techniques to composite these values into an image
        - (d) This rendering function is differentiable, the scene representation can be optimized by minimizing the residual between synthesized and ground truth observed images
    >  Minimizing this error across multiple views encourages the network to predict a coherent model of the scene by assigning high volume densities and accurate colors to the locations that contain the true underlying scene content.

### 2-2. Optimization
1. Positional Encoding
    - Neural networks are biased towards learning lower frequency functions. 
    - Transform input to higher dimension to enable better fitting of data that contains high frequency variation  
    $$ F_{\Theta} = F_{\Theta}' \circ \gamma $$  
    $$ F_{\Theta}$$ is learned while $$\gamma$$ projects the input from $$\mathbb{R}$$ to $$\mathbb{R}^{2L}$$

  
2. Hierarchical Volume Sampling

### 2-3. Neural Volume Rendering

### 2-4. Characteristics

## 3. Beyond NeRF

## 4. Recent Applications of NeRF

## 5. Research Directions

## Sources
[1] https://paperswithcode.com/task/novel-view-synthesis