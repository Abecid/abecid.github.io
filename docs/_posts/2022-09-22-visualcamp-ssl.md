---
layout: post
title:  "Visualcamp : SSL Research"
date:   2022-09-22 15:48:31 +0900
category: Research
tags: [Deep Learning, Computer Vision, Self-Supervised Learning, Knowledge Distillation, Visualcamp, Hidden]
---

{% if page.tags.size > 0 %}
  Tags: {{ page.tags | sort | join: ", " }}
{% endif %}

## Self-supervised Learning Research Log

<details open>
  <summary>2022</summary>
  <details>
    <summary>September</summary>
    <h4>Broad Plan</h4>
    <ol>
      <li>SSL Structure</li>
        <ul>
          <li>IR and RGB loss (taken at same instance)</li>
          <li>IR-IR with similar x,y points</li>
          <li>RGB-RGB with similar x,y points</li>
          <li>IR-RGB with similar x,y points  </li>
          <li>right, left, center x,y point estimates  </li>
        </ul>
      <li>Loss Function</li>
        <ul>
          <li>cross entropy  </li>
          <li>mse  </li>
          <li>further modification  </li>
            <ul>
              <li>apply both gradients to teacher with alpha parameter  </li>
            </ul>
        </ul>
      <li>Architecture</li>
        <ul>
          <li>1 model (shared weights) v. 2 models (teacher and student model)  </li>
          <li>more layers?  </li>
        </ul>
    </ol>
    <hr>
    <h4>9.22</h4>
    <ul>
      <li>Measure the loss between IR and RGB outputs with the same pretrained model  </li>
      <li>Loss between right,left,center estimates of the model  </li>
    </ul>

  </details>
  <details>
  <summary>October</summary>
  <h4>Broad Plan</h4>
  <ul>
    <li>Fine-tune bigger Res-net model: Train both RGB and IR image models  </li>
    <li>Featuremap loss  </li>
    <li>Domain adversarial training  </li>
    <li>Pool student models and RGB model, train together</li>
    <li>Attention embedding</li>
    <li>Paper</li>
      <ul>
        <li>Knowledge Distillation and Domain Generalization at Once</li>
        <li>Novel architecture and loss function to combine both</li>
        <li>Scarlett is all you need</li>
      </ul>
  </ul>
  <h4>Week 3.</h4>
  <ul>
    <li>Convert IR to RGB camera extrinsic calibration</li>
    <li>Convert target gaze coordinate from world coordinate systme to gaze-model coordinate system</li>
    <li>Use entire training dataset (currently using 1/3)</li>
    <li>Train Res-net with CW1 and Everyone</li>
    <li>Train a separate network on distance data</li>
    <li>Student Pool</li>
      <ul>
        <li>Multiple students learn from teacher</li>
        <li>Students teach each other</li>
      </ul>
    <li>Discriminator</li>
    <li>Featuremap; attention-based</li>
    <li>Adjust Learning Rate</li>
      <ul>
        <li>Decay Rate</li>
        <li>Schedule</li>
      </ul>
    <li>Supervsied Learning</li>
  </ul>
  </details>
  <summary>Future Work</summary>
  <ol>
    <li>Pretrain a larger model: Resnet on our datasets</li>
      <ul>
        <li>Utilization for knowledge distillation on gaze model (both RGB and IR)</li>
      </ul>
    <li>Knowledge distillation on a smaller gaze model! </li>
      <ul>
        <li>Neural architecture search</li>
      </ul>
    <li>Knowledge distillaion - domain generalization</li>
      <ul>
        <li>Unified model for both</li>
        <li>Adjust loss function</li>
      </ul>
    <li>Diffusion</li>
    <li>NERF</li>
    <li>Neural Architecture Search</li>
  </ol>
</details>


