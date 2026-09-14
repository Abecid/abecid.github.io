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
  .post-content h3 { font-size: 1.2rem; margin-top: 1.8rem; scroll-margin-top: 5rem; }
  .post-content h2 { margin-top: 2.5rem; font-size: 1.5rem; scroll-margin-top: 5rem; }
  .post-content th, .post-content td { padding: 0.4rem 0.9rem 0.4rem 0; vertical-align: top; border-bottom: 1px solid var(--global-divider-color); }
  .post-content table { margin: 1rem 0 1.5rem; }
title: "GPU Systems: Follow the Bottleneck"
date: 2026-07-28
description: "How GPU execution and data movement shape kernel optimization, distributed training, and inference serving."
tags: gpu-systems cuda
categories: research-survey
published: true
related_posts: false
---

Making an AI workload faster requires understanding what keeps its GPU busy or waiting. A matrix multiplication, a normalization layer, and an autoregressive decoding step use the same hardware but move different amounts of data and expose different kinds of parallelism. The useful optimization depends on that difference.

This post builds a way to reason about those costs: start inside one GPU, follow data through a kernel, then extend the same reasoning to multiple GPUs and inference requests. The objective is to connect each technique to **the cost it removes, the tradeoff it introduces, and its effect on the whole workload**.

## Contents

1. [Inside a GPU: execution and memory](#1-inside-a-gpu-execution-and-memory)
2. [Move fewer bytes: coalescing, tiling, and rooflines](#2-move-fewer-bytes)
3. [Build efficient kernels: fusion and occupancy](#3-build-efficient-kernels)
4. [Use lower precision](#4-use-lower-precision)
5. [Measure the application before optimizing](#5-measure-the-application)
6. [Scale across GPUs](#6-scale-across-gpus)
7. [Keep inference work moving](#7-keep-inference-work-moving)
8. [Put it together: video generation](#8-put-it-together-video-generation)

## 1. Inside a GPU: execution and memory

A **kernel** is a function executed by many GPU threads. A launch groups those threads into **blocks**, and the collection of blocks forms a **grid**. Each block runs on one **streaming multiprocessor (SM)**; an SM can host several blocks. Within an SM, threads execute in **warps of 32**. For vector addition, thread $i$ might compute just $c_i=a_i+b_i$: the speed comes from many threads doing that work concurrently.

{% include figure.liquid path="assets/img/blogs/gpu-systems/gpu-layout.svg" mobile_path="assets/img/blogs/gpu-systems/gpu-layout-mobile.svg" alt="GPU package with HBM beside the die. Inside the die, L2 connects multiple SMs. Each SM contains registers, compute units, and L1/shared memory." width=900 height=530 mobile_width=390 mobile_height=626 zoomable=true avoid_scaling=true %}

The diagram shows where their data lives. **High bandwidth memory (HBM)** is the large device memory beside the GPU die. **L2** is a cache on the GPU die shared across SMs; **L1** serves accesses within an SM. **Shared memory** is explicitly managed storage that lets threads in a block reuse data and cooperate. **Registers** hold each thread’s working values. Tensor cores, inside the compute units, accelerate matrix operations. The capacity shrinks as storage moves closer to computation, so kernels must choose what to keep there. [CUDA execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).

## 2. Move fewer bytes

### Coalescing: combine a warp’s memory accesses

Consider the vector addition kernel again. A warp needs 32 values from each input, but memory is transferred in sectors rather than one independent transfer per requested float. If neighboring threads read neighboring 32-bit floating point (FP32) values, their 128 useful bytes fit into **four aligned 32-byte sectors**. This is **coalescing**: the mapping from threads to elements makes the transferred bytes useful.

{% include figure.liquid path="assets/img/blogs/gpu-systems/coalescing.svg" mobile_path="assets/img/blogs/gpu-systems/coalescing-mobile.svg" alt="One warp reads 32 FP32 values. Adjacent addresses touch four 32-byte sectors; addresses eight floats apart touch 32 sectors." caption="Aligned FP32 loads: same 128 useful bytes, 4 versus 32 sectors." width=900 height=438 mobile_width=390 mobile_height=538 zoomable=true avoid_scaling=true %}

Change the mapping so each thread reads every eighth float, and those same 32 values occupy **32 sectors**. The kernel still performs 32 additions, but the memory system handles far more data. This is why tensor strides and layouts matter even when the arithmetic is unchanged. For a matrix stored in row major order, mapping neighboring threads along a row usually gives contiguous accesses; mapping them down a column often creates a large stride. The sector counts here assume aligned loads and 32 active lanes. [CUDA memory access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory).

### Tiling: reuse the bytes you loaded

Coalescing makes each transfer efficient. **Tiling** reduces how often the same inputs need transferring. In $C=AB$, one row of $A$ contributes to many columns of $C$, and one column of $B$ contributes to many rows. A tiled kernel loads small patches of $A$ and $B$ into shared memory or registers, computes a patch of $C$, then advances along the reduction dimension. The partial output stays in registers while successive input tiles accumulate into it.

{% include figure.liquid path="assets/img/blogs/gpu-systems/tiling.svg" mobile_path="assets/img/blogs/gpu-systems/tiling-mobile.svg" alt="A 4 × 4 A tile and a 4 × 4 B tile produce sixteen output values. Each A value is reused across four output columns, and each B value across four output rows." caption="One 4 × 4 tile multiplication: 32 FP32 input values support 128 FLOPs. Output traffic excluded." width=900 height=618 mobile_width=390 mobile_height=638 zoomable=true avoid_scaling=true %}

For square tiles of width $b$, one tile multiplication performs approximately $2b^3$ floating point operations (FLOPs) from $2b^2$ input elements. With FP32 inputs, that is $b/4$ FLOPs per input byte, excluding output traffic. Larger tiles therefore offer more reuse but require more storage on chip. [Triton’s matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) implements this pattern with separate output tiles and a loop over the reduction dimension.

### Rooflines: decide whether bytes or arithmetic limit performance

The link between reuse and speed is **arithmetic intensity**, $I$: operations performed per byte transferred from the memory level being considered. For HBM bandwidth $B$ and peak compute throughput $P_{\rm peak}$ at the chosen precision, attainable throughput $P$ is bounded by:

$$
I=\frac{\text{FLOPs}}{\text{HBM bytes}},\qquad
P\leq\min(P_{\rm peak},\;B I).
$$

{% include figure.liquid path="assets/img/blogs/gpu-systems/roofline.svg" mobile_path="assets/img/blogs/gpu-systems/roofline-mobile.svg" alt="A roofline rises with arithmetic intensity until reaching 120 TFLOP/s at 60 FLOP per byte, then stays flat." caption="Hypothetical GPU: 2 TB/s bandwidth, 120 TFLOP/s compute." width=900 height=451 mobile_width=390 mobile_height=439 zoomable=true avoid_scaling=true %}

On this hypothetical device, the two ceilings meet at $120/2=60$ FLOPs/byte. FP32 vector addition reads two inputs and writes one output: **12 bytes for one FLOP**, giving a bandwidth ceiling of about **0.167 TFLOP/s**. Low FLOP/s is expected for that operation. Tiled matrix multiplication can reuse inputs enough to approach the compute ceiling instead. A kernel far below both roofs needs a different diagnosis: dependencies, poor memory accesses, or insufficient parallel work may be preventing it from reaching either limit.

## 3. Build efficient kernels

### Fusion: keep intermediate results inside the kernel

Tiling reuses inputs within an operation. **Fusion** carries that idea across operations. For $y=\phi(ax+b)$, three separate elementwise kernels may write the scaled tensor, read it to add the bias, then write and read again for the activation. With scalar $a$ and $b$, these intermediates contain no information that needs to survive after $y$ is produced.

{% include figure.liquid path="assets/img/blogs/gpu-systems/fusion.svg" mobile_path="assets/img/blogs/gpu-systems/fusion-mobile.svg" alt="Three separate operations read and write global memory six times. A fused operation keeps intermediates on chip, leaving one input read and one output write." caption="Scalar a, b; FP32. Ideal traffic: 24N → 8N bytes; caches can reduce HBM traffic." width=900 height=476 mobile_width=390 mobile_height=465 zoomable=true avoid_scaling=true %}

A fused kernel keeps those values in registers and writes only the final result. For $N$ elements, the ideal FP32 array traffic falls from **24N to 8N bytes**, and three launches become one. Cache hits can reduce the actual HBM savings, while excessive register use can erase them through **spills**, which store excess working values in device memory. Compare against the framework compiler and vendor libraries first; they may already perform this fusion. [Fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) applies the same idea to a reduction: load a row, compute its statistics and normalization on chip, and write once.

### Occupancy: leave enough independent work to hide waiting

A warp can stall while waiting for memory or a previous instruction. The SM can issue work from another ready warp during that wait. **Occupancy** measures resident warps relative to the hardware maximum; it tells us how much of that potential parallelism is available. Blocks consume registers and shared memory, so a larger tile or a more heavily fused kernel can reduce the number of blocks that fit on an SM.

This creates a tradeoff between **reuse and concurrency**. A larger tile may run faster despite lower occupancy because it avoids repeated HBM reads. It may also run slower because too few independent warps remain, or because registers spill into memory. Choose tile sizes by measuring the resulting kernel, then use register, shared memory, and stall counters to explain the result. Occupancy alone does not tell you which choice is faster. [CUDA resource usage](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#occupancy).

## 4. Use lower precision

Lower precision reduces bytes per stored value and can enable faster matrix instructions. The exponent sets the range of magnitudes; the mantissa controls how finely nearby values can be distinguished. **BF16** retains FP32’s eight exponent bits, preserving a wide dynamic range with fewer significant bits. **FP8** halves storage again, but its smaller exponent and mantissa fields make the choice of scaling more important. E4M3 spends more bits on precision; E5M2 spends more on range.

| Format   | Bytes/value | Exponent bits | Mantissa bits |
| :------- | ----------: | ------------: | ------------: |
| FP32     |           4 |             8 |            23 |
| BF16     |           2 |             8 |             7 |
| FP8 E4M3 |           1 |             4 |             3 |
| FP8 E5M2 |           1 |             5 |             2 |

$$
q=Q(x/s),\qquad \hat x=sq.
$$

Here $Q$ rounds into the chosen format, and $s$ sets the scale. With this convention, too small a scale can overflow large values; too large a scale can push small ones toward zero. Scales for each block adapt better to local ranges but add metadata and computation. The relevant result is the whole workload’s speed and memory use at acceptable quality: conversions, sensitive operations retained in higher precision, and changed convergence all affect that tradeoff. [Transformer Engine’s FP8 guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

## 5. Measure the application

### Estimate the available speedup

An efficient kernel matters only as much as the application depends on it. The **critical path** is the dependency chain that determines completion time. If two operations overlap, shortening the one that already finishes early may leave latency unchanged. This is why a profiler’s summed kernel durations are not a budget of elapsed time: concurrent operations can cover the same time interval.

{% include figure.liquid path="assets/img/blogs/gpu-systems/amdahl.svg" mobile_path="assets/img/blogs/gpu-systems/amdahl-mobile.svg" alt="A 100 ms run contains 95 ms of other work and a 5 ms target kernel. A threefold kernel speedup reduces total time to 96.67 ms." caption="Calculated example: 3× kernel speedup → 1.034× application speedup." width=900 height=344 mobile_width=390 mobile_height=399 zoomable=true avoid_scaling=true %}

If a fraction $f$ of runtime does not overlap other work and is accelerated by $s$, Amdahl’s law gives the total speedup below. In the diagram, making a 5 ms kernel three times faster saves only 3.33 ms from a 100 ms run. Even eliminating it entirely leaves 95 ms. Estimate this budget before implementing a replacement; it distinguishes a worthwhile application improvement from an impressive isolated benchmark.

$$
S=\frac{1}{(1-f)+f/s}.
$$

### Use the profiler at the right scale

Start with the application timeline to find the expensive phase, then inspect the relevant operators and kernels. A gap between GPU kernels may come from CPU dispatch, a transfer, or synchronization. CUDA launches are asynchronous, so a CPU timer around the launch alone can measure submission rather than execution. Warm up the exact workload and measure through GPU completion; keep shapes, precision, hardware, and batch policy fixed.

| Tool                                                                                        | What it helps explain                                        |
| :------------------------------------------------------------------------------------------ | :----------------------------------------------------------- |
| [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | Framework operators, allocations, and their kernels          |
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)               | CPU/GPU timing, transfers, dependencies, and overlap         |
| [Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)           | Memory traffic, resource usage, and stalls inside one kernel |

After changing a kernel, remeasure the full workload. Extra copies, dispatch overhead, or lost overlap can consume its local saving. The useful evidence is a shorter completion time **and a trace showing which cost disappeared**. Record numerical agreement too: changing accumulation order or precision can change outputs even when the timing improves.

## 6. Scale across GPUs

### Partition the resource that does not fit

One GPU has finite model state memory, activation memory, and compute. A parallelism scheme chooses which of those demands to split. **Data parallelism** gives each GPU different examples while keeping a model replica on each; gradients must then be combined. **Fully sharded data parallelism** also partitions parameters, gradients, and optimizer state, exchanging the pieces needed for the current layer. The memory saving therefore introduces communication into the execution schedule.

| Parallelism        | What is split?            | Typical exchange                            |
| :----------------- | :------------------------ | :------------------------------------------ |
| Data               | Batch                     | Gradient AllReduce                          |
| Fully sharded data | Model and optimizer state | Parameter AllGather; gradient ReduceScatter |
| Tensor             | Operations within a layer | Partial activations/results                 |
| Context            | Sequence positions        | Attention information across positions      |
| Pipeline           | Groups of layers          | Activations and their gradients             |
| Expert             | MoE experts               | Tokens dispatched to selected experts       |

The right split follows the pressure: large parameter state suggests sharding; long video sequences can make activation or context partitioning important. Communication frequency then determines placement. Tensor parallel layers exchange data repeatedly, so their groups benefit from fast local links. Pipeline stages exchange at layer group boundaries. Adding GPUs helps only if the reduced local work outweighs these exchanges and any imbalance. [PyTorch distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html).

### Overlap communication with ready computation

[NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) describe how tensors move between GPU processes, called ranks. **AllReduce** combines corresponding values and returns the result to every rank; **AllGather** assembles shards; **ReduceScatter** combines values and leaves each rank one result shard. During backpropagation, gradients computed earlier can be reduced while later gradients are still being computed. Grouping gradients into buckets controls when these exchanges become ready.

{% include figure.liquid path="assets/img/blogs/gpu-systems/overlap.svg" mobile_path="assets/img/blogs/gpu-systems/overlap-mobile.svg" alt="Eight milliseconds of compute plus six milliseconds of communication take fourteen milliseconds serially. Starting communication at four milliseconds reduces completion to ten milliseconds." caption="Illustrative schedule: 4 ms hidden, 2 ms communication exposed." width=900 height=414 mobile_width=390 mobile_height=447 zoomable=true avoid_scaling=true %}

Here, 8 ms of compute plus 6 ms of communication takes 14 ms if serialized. Starting communication at 4 ms hides 4 ms and finishes at 10 ms; the remaining 2 ms is **exposed communication**. Launching earlier requires ready inputs, and overlapping operations can compete for GPU resources. Measure the exposed tail rather than treating every millisecond of NCCL activity as lost time. For scaling comparisons, hold either the global workload fixed (**strong scaling**) or the workload per GPU fixed (**weak scaling**), and report which one you used.

## 7. Keep inference work moving

### Continuous batching: replace requests as they finish

Training often repeats a regular batch shape. Serving handles requests that arrive and finish at different times. A static batch can leave capacity unused when short requests finish but the batch waits for a long one. **Continuous batching** updates the active group at iteration boundaries: finished requests leave, and compatible waiting requests enter their slots.

{% include figure.liquid path="assets/img/blogs/gpu-systems/batching.svg" mobile_path="assets/img/blogs/gpu-systems/batching-mobile.svg" alt="A requires four iterations; B and C need two each. Continuous batching replaces B with C while A runs, finishing at iteration four instead of six." caption="Two slots; iterations of equal cost; compatible requests." width=900 height=414 mobile_width=390 mobile_height=447 zoomable=true avoid_scaling=true %}

In this example, A needs four iterations while B and C need two each. Refilling B’s slot with C finishes all three in four iterations instead of six. Actual iterations vary with batch size and request type, so the diagram is a schedule, not a measured speedup. Larger batches can also increase an individual request’s latency. Evaluate throughput together with queue time and latency targets; **goodput** counts requests that finish within those targets.

### Manage request state and launch overhead

Autoregressive decoding retains previous attention keys and values in a **KV cache**. Reserving each request’s maximum possible cache in advance wastes capacity, especially when output lengths differ. Paged allocation grows request state in blocks and releases blocks on completion, allowing more active requests to share memory. The scheduler must still balance **prefill**, which processes input tokens together, against repeated **decode** steps that generate new tokens. [vLLM](https://docs.vllm.ai/) combines these scheduling and memory mechanisms.

Repeated short kernels introduce another cost: the CPU must keep submitting work. **CUDA Graphs** capture a compatible launch sequence and replay it with less dispatch overhead. That is useful when a trace shows gaps between otherwise efficient kernels. Shapes, addresses, and control flow constrain reuse, so runtimes maintain compatible graph variants or fall back when a request does not fit. Graph replay addresses launch overhead; it complements the memory and arithmetic optimizations above. [CUDA Graphs in vLLM](https://docs.vllm.ai/en/stable/design/cuda_graphs/).

## 8. Put it together: video generation

A diffusion video pipeline repeats a denoiser, then converts its output latents to pixels with a **variational autoencoder (VAE)** and encodes the video. Kernel fusion, tiling, and lower precision can reduce the cost of each denoiser call. [Flow maps and distillation]({% post_url 2026-02-24-flowmap %}) instead reduce how many calls the sampler needs. These changes act on different terms in the latency budget:

$$
T_{\rm total}\approx K T_{\rm denoiser}+T_{\rm fixed}.
$$

{% include figure.liquid path="assets/img/blogs/gpu-systems/denoising-budget.svg" mobile_path="assets/img/blogs/gpu-systems/denoising-budget-mobile.svg" alt="With forty milliseconds per denoiser call and two hundred milliseconds of fixed work, twenty calls cost one thousand milliseconds, four cost three hundred sixty, and one costs two hundred forty." caption="Calculated, no overlap: 40 ms/call + 200 ms conditioning, VAE, and output work." width=900 height=356 mobile_width=390 mobile_height=411 zoomable=true avoid_scaling=true %}

With 40 ms per call and 200 ms of fixed work, reducing $K$ from 20 to 4 changes latency from 1,000 to 360 ms: **five times fewer denoiser calls, but only 2.78× overall speedup**. At one call, fixed work accounts for 83% of latency. The next useful target may therefore be VAE decoding or output encoding. This is the same reasoning used for one kernel, now applied to the entire pipeline: measure the remaining cost after each improvement, and include a quality comparison when changing the sampler.
