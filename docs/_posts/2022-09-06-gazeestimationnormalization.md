---
layout: post
title:  "Revisiting Data Normalization for Appearance-Based Gaze Estimation"
date:   2022-09-06 10:45:31 +0900
category: Paper Review
tags: [Deep Learning, Computer Vision, Gaze, Normalization]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

<img src="/assets/posts/paper_review/5.gazenormalization/FrontPage.png">

<h2>Intro</h2>
Revisiting Data Normalization for Appearance-Based Gaze Estimation
<a href="https://perceptualui.org/publications/zhang18_etra.pdf">Source Paper (PDF File)</a>
<h4>Context</h4>
Appearance-based gaze estimation
Challanges
- Variability in head pose
- user-camera distance
Normalization
- Cancel out geometric variability: Map input images to normalized space

Paper states
> Role and importance of data normalization remains unclear
And claims to suty for the first time using principled evaluations on both simulated and real data. Proposes modification to remove the scaling factor and the new formulation performs significantly better (between 9.5% and 32.7%). 

Recent development of appearance-based gaze estimation.
Fixed head pose -> fixed distance between user and camera -> Real-world environment.

Large data contains sufficient variabilty but manual collection and annotation is costly.

Image resizing is equivalent to 3D scaling rather than 3D shifting.
(Image Samples)


<h2>Methodology</h2>
> The key idea is to standardize the translation and rotation between camera and face coordinate system via camera rotation and scaling.

<h4>Eye Image Normalization</h4>

Starts from an aribitrary pose of the target face.
x-axis: line connecting midpoints of the two eyes from right eye to left eye
y-axis: perpendicular to the x-axis inside the triangle plane from the eye to the mouth
z-axis: perpendicular to the triangle and pointing backwards from the face.

$e_{r}$: The midpoint of the right eye as the origin of the head coordinate system.
$R_{r}$: Translation and rotation from the camera coordinate system to the head coordinate system.

Normalized image must meet three conditions
1. The normalized camera looks at the origin of the head coordinate system and the center of the eye is located at the center of the normalized image.
2. The x-axes of the head and camera coordinate systems are on the same plane (x-axis of the head coordinate system appears horizontal)
3. The normalized camera is located at a fixed distance $d_{n} from the eye center and the eye always has the same size.

$z_{c}$: Rotated z-axis of the camera coordinate system
$x_{r}$: x-axis of the head coordinate system

$z_{c}$ has to be $e_{r}$.
$y_{c} = z_{c} * x_{r} $ 
$y_{c} \perp z_{c}$, $y_{c} \perp x_{r}$
$x_{c} = y_{c} * z_{c} $

(R Image)
(S Image)
M = S * R

This is for a 3D face mesh.

If the input is a 2D face image,
(W Image)
$C_{r}$: Original camera projection matrix from camera caliberation
$C_{n}$: Camera projection matrix for the normalized camera

<h4>Modified Data Normalization</h4>
It is important to handle the geometric transformation caused by the eye image normalization and apply the same transformation to the gaze direction vector.

$g_{r}$: ground-truth gaze direction vector
$g_{n}$: normalized gaze vector

For 3D space, (g_n = M * g_r)
$R_{n}$: Head rotation matrix
(R_n = R * R_r)

A modified version of data nomalization for 2d images is proposed.
Propose to only rotate the original gaze vector to obtain the normalized gaze vector
(g_n = R * g_r)

<h2>Performance</h2>
Since this paper emphasizes itself for being the first to quantize the effects of normalization, the boosted performace is an essential component of this paper.

<h4>Evaluation on Synthetic Images</h4>

<h4>Evaluation on Real Images</h4>

<h2>Code</h2>


<h2>Challenging Aspects</h2>
1. Method
Vector, Camera Coordinates & World Coordinates
2. Code
MediaPipe, Conversion of 2D Landmark points to Pixel Points

<h2>Thoughts</h2>
There could be more ways to normalize image data.
Besides learning, data pre-processing is also very important.
Intuition for neural networks to learn well is to restrict the differences between the training data to learn.

<h2>References</h2>
