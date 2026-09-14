---
layout: post
_styles: |
  .post-content figure { margin: 1.25rem 0 1.5rem; }
  .post-content figure img { display: block; width: 100%; border-radius: 8px; }
  .post-content figcaption.caption {
    max-width: 76ch;
    margin: 0.5rem 0 0;
    text-align: left;
    font-size: 0.8rem;
    line-height: 1.45;
  }
  .post-content img { max-width: 100%; height: auto; }
  .post-content mjx-container[display="true"] {
    display: block;
    max-width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0.25em 0;
  }
  .post-content table { display: block; max-width: 100%; overflow-x: auto; font-size: 0.9rem; }
  .post-content h2 { margin-top: 2.5rem; font-size: 1.5rem; scroll-margin-top: 5rem; }
  .post-content th, .post-content td { padding: 0.4rem 0.9rem 0.4rem 0; vertical-align: top; border-bottom: 1px solid var(--global-divider-color); }
  .post-content table { margin: 1rem 0 1.5rem; }
title: "GPU Systems: Follow the Bottleneck"
date: 2026-07-28
description: "GPU memory, kernels, communication, and serving—in diagrams."
tags: gpu-systems cuda
categories: research-survey
published: true
related_posts: false
---

## 1. Inside a GPU

A **streaming multiprocessor (SM)** executes thread blocks. Each **warp** contains 32 threads.

{% include figure.liquid path="assets/img/blogs/gpu-systems/gpu-layout.svg" mobile_path="assets/img/blogs/gpu-systems/gpu-layout-mobile.svg" alt="GPU package with HBM beside the die. Inside the die, L2 connects multiple SMs. Each SM contains registers, compute units, and L1/shared memory." width=900 height=530 mobile_width=390 mobile_height=626 zoomable=true avoid_scaling=true %}

**L1** caches automatically. **Shared memory** is managed by the program, normally within a block. Registers hold each thread’s values. [CUDA execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).

## 2. Read adjacent addresses

**Coalescing:** neighboring lanes read neighboring values.

{% include figure.liquid path="assets/img/blogs/gpu-systems/coalescing.svg" mobile_path="assets/img/blogs/gpu-systems/coalescing-mobile.svg" alt="One warp reads 32 FP32 values. Adjacent addresses touch four 32-byte sectors; addresses eight floats apart touch 32 sectors." caption="Aligned FP32 loads: same 128 useful bytes, 4 versus 32 sectors." width=900 height=438 mobile_width=390 mobile_height=538 zoomable=true avoid_scaling=true %}

## 3. Reuse data on chip

**Tiling:** load a matrix tile once; reuse it for many multiply-adds.

$$
I=\frac{\text{FLOPs}}{\text{HBM bytes}},\qquad
P\leq\min(P_{\rm peak},\;B I).
$$

{% include figure.liquid path="assets/img/blogs/gpu-systems/roofline.svg" mobile_path="assets/img/blogs/gpu-systems/roofline-mobile.svg" alt="A roofline rises with arithmetic intensity until reaching 120 TFLOP/s at 60 FLOP per byte, then stays flat." caption="Hypothetical GPU: 2 TB/s bandwidth, 120 TFLOP/s compute." width=900 height=451 mobile_width=390 mobile_height=439 zoomable=true avoid_scaling=true %}

FP32 vector add: **1 FLOP / 12 bytes**, so its bandwidth ceiling is **0.167 TFLOP/s** here.

- **Memory-bound:** reduce bytes; increase reuse.
- **Compute-bound:** improve matrix layouts and instruction throughput.
- **Below both roofs:** inspect stalls, dependencies, and available parallel work.

More resident warps can hide latency. Excess registers/shared memory reduce residency; register spills add memory traffic. [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

## 4. Fuse intermediate writes away

For $y=\phi(ax+b)$, keep intermediate values in registers.

{% include figure.liquid path="assets/img/blogs/gpu-systems/fusion.svg" mobile_path="assets/img/blogs/gpu-systems/fusion-mobile.svg" alt="Three separate operations read and write global memory six times. A fused operation keeps intermediates on chip, leaving one input read and one output write." caption="Scalar a, b; FP32. Ideal traffic: 24N → 8N bytes; caches can reduce HBM traffic." width=900 height=476 mobile_width=390 mobile_height=465 zoomable=true avoid_scaling=true %}

Compare the compiled/library baseline first. Fusion helps only if saved traffic and launches outweigh register pressure. [Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).

## 5. Measure the time you can remove

The **critical path** is the dependency chain that determines completion time. Optimize time exposed on that path.

{% include figure.liquid path="assets/img/blogs/gpu-systems/amdahl.svg" mobile_path="assets/img/blogs/gpu-systems/amdahl-mobile.svg" alt="A 100 ms run contains 95 ms of other work and a 5 ms target kernel. A threefold kernel speedup reduces total time to 96.67 ms." caption="Calculated example: 3× kernel speedup → 1.034× application speedup." width=900 height=344 mobile_width=390 mobile_height=399 zoomable=true avoid_scaling=true %}

$$
S=\frac{1}{(1-f)+f/s}
$$

$f$: non-overlapped runtime fraction. $s$: speedup of that fraction.

| Inspect                   | Tool                                                                                        |
| :------------------------ | :------------------------------------------------------------------------------------------ |
| Operators and allocations | [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) |
| CPU gaps, copies, overlap | [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)               |
| One expensive kernel      | [Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)           |

**Warm up. Time through GPU completion.** Compare identical shapes, precision, hardware, and batch policy. Report latency, throughput, peak memory, and quality.

## 6. Split work; overlap communication

| Parallelism        | Split across GPUs                      |
| :----------------- | :------------------------------------- |
| Data               | Batch; replicate model                 |
| Fully sharded data | Parameters, gradients, optimizer state |
| Tensor             | Operations within a layer              |
| Context            | Tokens                                 |
| Pipeline           | Layers                                 |
| Expert             | MoE experts                            |

[AllReduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) gives every rank the same reduced values. AllGather assembles shards; ReduceScatter reduces, then partitions.

{% include figure.liquid path="assets/img/blogs/gpu-systems/overlap.svg" mobile_path="assets/img/blogs/gpu-systems/overlap-mobile.svg" alt="Eight milliseconds of compute plus six milliseconds of communication take fourteen milliseconds serially. Starting communication at four milliseconds reduces completion to ten milliseconds." caption="Illustrative schedule: 4 ms hidden, 2 ms communication exposed." width=900 height=414 mobile_width=390 mobile_height=447 zoomable=true avoid_scaling=true %}

Launch collectives when inputs become ready. Put frequent exchanges on fast links. Resource contention can reduce overlap.

**Scaling efficiency:** $E_N=X_N/(NX_1)$, for throughput $X_N$ on $N$ GPUs. State whether global work or per-GPU work is fixed.

## 7. Refill finished batch slots

**Continuous batching** admits waiting requests at iteration boundaries.

{% include figure.liquid path="assets/img/blogs/gpu-systems/batching.svg" mobile_path="assets/img/blogs/gpu-systems/batching-mobile.svg" alt="A requires four iterations; B and C need two each. Continuous batching replaces B with C while A runs, finishing at iteration four instead of six." caption="Two slots; equal-cost iterations; compatible requests." width=900 height=414 mobile_width=390 mobile_height=447 zoomable=true avoid_scaling=true %}

- **Paged KV cache:** allocate request state in blocks; reduce wasted capacity.
- **CUDA Graphs:** replay compatible launch sequences; reduce CPU overhead.
- **Goodput:** completed requests/s that meet the latency target.

[vLLM runtime](https://docs.vllm.ai/) · [CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)

## 8. Lower precision; check quality

| Format          | Bytes/value | Exponent bits |
| :-------------- | ----------: | ------------: |
| FP32            |           4 |             8 |
| BF16            |           2 |             8 |
| FP8 E4M3 / E5M2 |           1 |         4 / 5 |

$$
q=Q(x/s),\qquad \hat x=sq.
$$

**Scaling** keeps values within the usable range. Tensor or block scales add overhead. Measure speed and memory alongside overflow, convergence, and output quality. [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

## 9. Fewer denoising calls move the bottleneck

$$
T_{\rm total}\approx K T_{\rm denoiser}+T_{\rm fixed}
$$

{% include figure.liquid path="assets/img/blogs/gpu-systems/denoising-budget.svg" mobile_path="assets/img/blogs/gpu-systems/denoising-budget-mobile.svg" alt="With forty milliseconds per denoiser call and two hundred milliseconds of fixed work, twenty calls cost one thousand milliseconds, four cost three hundred sixty, and one costs two hundred forty." caption="Calculated, no overlap: 40 ms/call + 200 ms conditioning, VAE, and output work." width=900 height=356 mobile_width=390 mobile_height=411 zoomable=true avoid_scaling=true %}

[Flow maps]({% post_url 2026-02-24-flowmap %}) reduce $K$. Compare quality too. At one call, fixed work is **83%** of this pipeline.
