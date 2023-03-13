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
![Ray Casting](/assets/posts/blog/5.nerf/circle-distance.png "Ray Casting")  
- Given a spatial [x, y, z] point as input, SDFs will output the distance from that point to the nearest surface of the underlying object being represented.
- negative sign: inside, positive sign: outside the surface
- $$ SDF(x) = s:x \in \mathbb{R}^{3}, s \in \mathbb{R} $$
- **Usefulness**  
    - Stores a function instead of a direct representation of 3D shape (more efficient)

### 1-2. Ray Tracing
1. Ray Casting  
  ![Ray Casting](/assets/posts/blog/5.nerf/raycasting.gif "Ray Casting")  
  - Casting a ray from the camera through each pixel on the screen and into the scene.
  - When the ray intersects with an object in the scene, the color of the object at that point is calculated and used to color the pixel on the screen.
  - Ray casting is a fast technique and is commonly used for simple scenes or for generating 2D images from 3D scenes.
2. Ray Tracing  
  ![Ray Casting](/assets/posts/blog/5.nerf/ray-tracing-image-1.jpeg "Ray Casting")  
  ![Ray Casting](/assets/posts/blog/5.nerf/gta-5-reflection-quality.webp "Ray Casting")  
  - Tracing rays of light from the camera into the scene, and then bouncing those rays off objects in the scene, and tracing them to the light sources.
  - Simulates the behavior of light more accurately than ray casting, resulting in more realistic lighting and shadows in the rendered scene
  - However, it is more computationally expensive than ray casting.
3. Ray Marching
  - Repeatedly stepping along a ray from the camera into the scene, and testing whether the ray intersects with a surface or volume (Evaluate SDF).
  - If the ray intersects with a surface, the color of the surface is calculated, and if it intersects with a volume, the color and opacity of the volume are calculated.
  - Ray marching is typically slower than ray casting or ray tracing, but it allows for the rendering of very complex shapes that cannot be represented by simple geometry.

### 1-3. Marching Cubes  
![Demo](/assets/posts/blog/5.nerf/MarchingCubesCases.png "Demo") 
- Creates a triangle mesh from an implicit function
-  Iterates ("marches") over a uniform grid of cubes superimposed over a region of the function

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
![ner](/assets/posts/blog/5.nerf/SRN.png "nerf-image2")  
- Represent a scene implicitly as acontinuous, differentiable function that maps a 3D world coordinate to a feature-based representationof the scene properties at that coordinate

## 2. NeRF
### 2-1. Introduction
![ner](/assets/posts/blog/5.nerf/nerf-image2.png "nerf-image2")  
- Input : (x,y,z,$$\theta, \phi$$) -> Output : (c, $$\sigma$$)
    - 3D location : x,y,z
    - Viewing Direction : $$\theta$$, $$\phi$$
    - Color : c
    - Volume Density : $$\sigma$$
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

- Multiview Consistent
  - Volume Density: $$ \sigma(x) $$
    - Function of location x (x,y,z)
    - Differential probability of a ray terminating at location x
  - Color: $$ c(r(t), d) $$
    - Function of location and viewing direction d
    - Location is 256-dimensional feature vector from 8 FCC


### 2-2. Volume Rendering  
1. Color at Ray
  - $$C(r) = \int^{t_f}_{t_n}T(t)σ(r(t))c(r(t),d)dt$$  
  - $$T(t) = exp(−\int^{t}_{t_n}σ(r(s))ds)$$
    - Accumulated  transmittance  along  the  ray
    - Probability  that  the  ray  travels  from $$t_n$$ to t without  hitting any other particle
  - $$ \sigma(x) $$ : differential probability of a ray terminating at location x
  - $$ r(t) = o + td $$ : 3D point on the camera ray at a distance t from the camera center
      - o : Origin starting point of the ray (Camera center)
      - t : Distance
      - d : Direction of the camera ray (unit vector)
  - c : color (R, G, B)
  - $$t_n$$ : Near bounds
  - $$t_f$$ : Far bounds
2. Quadrature
    1. Partition  $$[t_n,t_f]$$  into N evenly-spaced bins
    2. sampling $$t_i ∼ \mathcal{u}[t_n+\frac{(i−1)}{N}(t_f−t_n), t_n+\frac{i}{N}(t_f−t_n)]$$
    3. Estimation $$\hat{C}(r) =\sum_{i=1}^N T_i(1−exp(−σ_iδ_i))c_i$$
        1. probability of ray not colliding : $$T_i= exp(− \sum_{j=1}^{i−1} σ_jδ_j)$$
        2.  $$δ_i=t_i+1−t_i$$ : Distance between adjacent samples

### 2-3. Optimization
1. Positional Encoding
    - Neural networks are biased towards learning lower frequency functions. 
    - Transform input to higher dimension to enable better fitting of data that contains high frequency variation  
    $$ F_{\Theta} = F_{\Theta}' \circ \gamma $$  
    $$ F_{\Theta}$$ is learned while $$\gamma$$ projects the input from $$\mathbb{R}$$ to $$\mathbb{R}^{2L}$$  
    $$ F_{\Theta}' $$ is a MLP  
    $$ \gamma(p) = (sin(2^{0}\pi p), cos(2^{0}\pi p), ... ,sin(2^{L-1}\pi p), cos(2^{L-1}\pi p))  $$

2. Hierarchical Volume Sampling
    1. Properties
        - Evaluating Neural Field Network at N queries along each camera ray is inefficient
            - Free space and occluded regions are sampled repeatedly
        - Increase rendering efficiency
            - Optimize two networks: "coarse" and "fine"  
    2. Coarse Network
        - Sample $$N_{c}$$ locations
        - Evaluate the Coarse network at these positions  
        - $$\hat{C}_c(r) =∑^{N_c}_{i=1}w_i c_i$$
        - $$w_i=T_i(1−exp(−σ_iδ_i))$$

    3. Fine Network
        - Sample $$N_f$$ locations from piecewise-constant PDF C along the ray
        - Compute rendered color for $$N_c + N_f$$ samples
        - This procedure allocates moresamples to regions we expect to contain visible content.

3. Loss  
$$L=∑_{r∈R}[\lVert \hat{C}_c(r)−C(r)\rVert^2_2+\lVert\hat{C}_f(r)−C(r)\rVert^2_2]$$

### 2-3. Results
![fig2](/assets/posts/blog/5.nerf/results.png "f2")  

### 2-4. Characteristics
- Thoroughly outperform both baselines that also optimize a separate networkper scene (NV and SRN) in all scenarios
- The biggest practical tradeoffs between these methods are time versus space.
  - All compared single scene methods take at least 12 hours to train per scene.
  - In contrast, LLFF can process a small input dataset in under 10 minutes. However,LLFF produces a large 3D voxel grid for every input image, resulting in enor-mous storage requirement
- NeRF requires a large amount of training data, which can be expensive and time-consuming to collect.

## 3. Beyond NeRF
[Awesome NeRF](https://github.com/awesome-NeRF/awesome-NeRF/tree/main)

## 4. Recent Applications of NeRF

## 5. Research Directions

## Sources
[1] https://paperswithcode.com/task/novel-view-synthesis