---
layout: post
title: "The Modern AI GPU Systems Stack: A Comprehensive Guide"
date: 2026-07-28
description: "A guide to performance modeling, GPU execution, kernel engineering, distributed training, inference serving, low precision, and reliability for modern foundation models."
tags:
  - machine-learning-systems
  - gpu-systems
  - cuda
  - triton
  - distributed-training
  - inference-serving
  - world-models
categories: research-survey
published: false
---

# The Modern AI GPU Systems Stack: A Comprehensive Guide

*From CUDA execution and Triton kernels to distributed training, inference serving, low-precision numerics, and fault-tolerant runtimes.*

Modern AI models are usually described at the algorithmic level: transformer blocks, attention, diffusion objectives, latent spaces, distillation, and scaling laws. None of those abstractions explains why a training step takes twice as long as expected, why eight GPUs deliver far less than an 8× throughput gain, or why an inference server fails under concurrent load despite running perfectly in a notebook.

Those are **systems questions**.

A model is not executed as one mathematical object. It is lowered through a stack:

| Layer | What it controls | Typical failure mode |
|---|---|---|
| Model and workload | Tensor shapes, sequence length, sparsity, number of steps, recomputation | Too much inherent work or memory |
| Framework and compiler | Graph capture, operator decomposition, fusion, scheduling | Excess launches, graph breaks, redundant materialization |
| GPU kernels | Tiling, memory access, reductions, tensor-core use | Bandwidth waste, low reuse, poor occupancy, serialization |
| Device runtime | Streams, synchronization, memory allocation, CUDA Graphs | Idle gaps, launch overhead, allocator churn |
| Distributed runtime | Sharding, collectives, rank topology, overlap | Communication dominates or fails to overlap |
| Training runtime | Data loading, checkpointing, restart, logging | GPU starvation, long pauses, irreproducible recovery |
| Serving runtime | Admission, batching, caching, scheduling, cancellation | Tail-latency explosion, fragmentation, out-of-memory failure |
| Numerical format | BF16, FP8, scaling policy, accumulation precision | Instability or hidden quality loss |

GPU optimization is therefore not “make a kernel fast.” It is the end-to-end discipline of locating the active bottleneck, changing the correct layer of the stack, and proving that the result is faster **under the same workload and quality constraints**.

This guide develops that systems view from a single kernel to a multi-node training or inference runtime, with special attention to video diffusion and world models.

## Contents

1. Performance objectives and the critical path
2. CUDA execution, memory, and roofline reasoning
3. Profiling from the application timeline to one kernel
4. Triton and custom kernel engineering
5. Distributed training, collectives, topology, and scaling
6. Fault-tolerant training runtimes
7. Inference scheduling, memory management, and observability
8. BF16, FP8, and scaling policies
9. Video- and world-model-specific bottlenecks
10. End-to-end optimization methodology
11. A practical implementation sequence
12. What strong systems evidence looks like
13. Primary references

---

## 1. The governing principle: optimize the critical path

The first question is not “How busy is the GPU?” It is:

> What resource currently determines end-to-end latency or throughput?

A workload may be limited by:

- arithmetic throughput;
- HBM or cache bandwidth;
- dependency latency;
- insufficient parallel work;
- CPU dispatch and kernel-launch overhead;
- synchronization between streams;
- GPU-to-GPU communication;
- data loading or checkpoint I/O;
- queueing and memory pressure in a serving system.

These bottlenecks interact, and many phases overlap. A useful conceptual decomposition is

\[
T_{\text{critical}}
\approx
T_{\text{compute}}
+T_{\text{exposed memory}}
+T_{\text{exposed communication}}
+T_{\text{launch}}
+T_{\text{idle}}
+T_{\text{I/O}},
\]

where “exposed” means work that is not hidden behind another phase. This is not an accounting identity: compute, memory traffic, communication, and I/O can overlap. Its purpose is to force the correct question—**which costs remain on the critical path?**

### Latency, throughput, and goodput are different objectives

- **Latency** is the time to complete one unit of work: one training step, one request, one generated video, or one autoregressive chunk.
- **Throughput** is completed work per unit time: samples/s, tokens/s, frames/s, or requests/s.
- **Goodput** is useful work that satisfies the system's quality or service constraint. A server processing 100 requests/s is not achieving 100 requests/s of goodput if half violate the latency objective or fail.

A change that improves throughput can worsen single-request latency. Larger batches are the standard example. Systems work begins by declaring which objective matters and under what workload distribution.

### Amdahl's law kills cosmetic optimizations

Suppose an operation consumes fraction \(f\) of total runtime and is accelerated by a factor \(s\). The maximum end-to-end speedup is

\[
S_{\text{total}}
=
\frac{1}{(1-f)+f/s}.
\]

If a kernel is only 5% of runtime, making it infinitely fast improves the full application by at most

\[
\frac{1}{0.95}\approx 1.053.
\]

That is why a 3× microbenchmark result can produce a nearly invisible model-level improvement. The operation was never important enough.

The practical rule is simple:

> Rank opportunities by end-to-end critical-path contribution, not by how intellectually impressive the kernel looks.

---

## 2. Performance fundamentals: compute, memory, and parallelism

### 2.1 The CUDA execution model

A CUDA kernel launches a **grid** of thread **blocks**. Each block contains threads, and the GPU schedules threads in groups called **warps**—32 threads on NVIDIA GPUs. Blocks execute on streaming multiprocessors, or SMs.

The hierarchy matters because different forms of communication and synchronization are available at different levels:

- threads in one warp execute in a tightly coupled way;
- threads in one block can cooperate through shared memory and block-level synchronization;
- different blocks are generally independent during a conventional kernel launch;
- communication across blocks usually requires global memory or a new kernel boundary, unless a specialized execution mechanism is used.

A high-level tensor operation may become many kernel launches, and each launch chooses a mapping from logical tensor elements to blocks, warps, and threads. Performance depends on whether that mapping creates enough parallel work, uses the memory system efficiently, and avoids unnecessary synchronization.

### 2.2 The GPU memory hierarchy

The memory hierarchy is the core of GPU performance:

| Storage | Scope | Relative capacity | Relative speed | Main systems implication |
|---|---:|---:|---:|---|
| Registers | Per thread | Tiny | Fastest | Excess use reduces resident warps and may spill |
| Shared memory | Per block | Small | Very fast | Enables reuse and cooperation; layout can cause bank conflicts |
| L1 cache | Per SM | Small | Fast | Helps local reuse but is not a substitute for good access patterns |
| L2 cache | Device-wide | Moderate | Faster than HBM | Reuse across blocks may benefit, subject to working-set size |
| HBM/global memory | Device-wide | Large | High bandwidth, high latency | Repeated reads and writes often dominate elementwise workloads |
| Host or remote memory | Outside device | Much larger | Far slower | Transfers must be minimized, pinned, batched, and overlapped |

The GPU is fast because it can sustain enormous parallel throughput—not because any individual global-memory access is cheap. A good kernel either performs enough independent work to hide latency or reuses data from registers/shared memory so that expensive transfers are amortized.

### 2.3 Coalescing

Threads in a warp should access nearby memory addresses whenever possible. When adjacent threads access adjacent elements, the hardware can serve the warp with a small number of memory transactions. Strided or scattered access may require many transactions and waste bandwidth.

This is **memory coalescing**. It is one of the first properties to inspect when a supposedly simple kernel underperforms.

### 2.4 Tiling and reuse

Matrix multiplication illustrates the central optimization pattern. A naive implementation repeatedly loads matrix elements from HBM. A tiled implementation loads submatrices into shared memory or registers, reuses them for many multiply-accumulate operations, and only then advances to the next tile.

The deeper principle is:

> Spend expensive bandwidth once, then reuse the data as many times as the algorithm permits.

The same idea appears in attention, convolution, normalization, reductions, and fused pointwise operations.

### 2.5 Arithmetic intensity and the roofline model

Define arithmetic intensity as

\[
I
=
\frac{\text{floating-point operations}}{\text{bytes transferred from the limiting memory level}}.
\]

A simplified roofline bound is

\[
P_{\text{attainable}}
\leq
\min\left(P_{\text{peak}},\;B_{\text{memory}}I\right),
\]

where:

- \(P_{\text{attainable}}\) is achieved arithmetic throughput;
- \(P_{\text{peak}}\) is peak compute throughput for the relevant datatype and instruction path;
- \(B_{\text{memory}}\) is sustained bandwidth from the limiting memory level;
- \(I\) is arithmetic intensity.

Low-intensity operations are usually memory-bound. High-intensity operations may become compute-bound. An operation far below both roofs is often losing performance to poor occupancy, dependencies, uncoalesced access, serialization, synchronization, or launch overhead.

The roofline model is not a complete simulator. It is a classification tool: it tells you whether the next optimization should primarily reduce bytes, increase useful arithmetic throughput, or repair execution inefficiency.

### 2.6 Occupancy is a latency-hiding mechanism, not a score

**Occupancy** measures how many warps are resident on an SM relative to the hardware maximum. More resident warps can hide latency: while one warp waits for data, another can execute.

But maximizing occupancy is not the objective. A kernel with lower occupancy can be faster if it uses more registers to avoid spills, keeps more useful data on chip, or performs more work per thread. Conversely, high occupancy does not rescue a kernel with poor memory access or low instruction-level efficiency.

The correct question is:

> Is the kernel short of independent work needed to hide its active latency, and which resource—registers, shared memory, block size, or dependencies—is limiting that work?

### 2.7 Streams and synchronization

CUDA streams provide ordered command sequences that may execute concurrently with other streams when dependencies and hardware resources permit. They are the basis for overlapping:

- host-to-device copies with computation;
- VAE encoding or decoding with denoiser work;
- communication with gradient computation;
- checkpoint staging with continued training.

Synchronization destroys overlap when inserted unnecessarily. Common sources include:

- explicit device synchronization;
- reading a GPU result on the CPU;
- scalar extraction such as poorly placed `.item()` calls;
- allocator behavior;
- implicit dependencies between default and auxiliary streams;
- collectives that are launched too late to overlap.

The Nsight Systems timeline is usually the fastest way to see these mistakes.

---

## 3. Measurement: from the application timeline to one kernel

Optimization without measurement is performance folklore. The profiling stack should be used from coarse to fine.

### 3.1 PyTorch Profiler: operator-level attribution

PyTorch Profiler is the fast first pass. It answers:

- Which framework operators dominate aggregate CPU and CUDA time?
- Which operations allocate the most memory?
- How often is each operation called?
- Are graph breaks or unexpected copies visible?

Its output is useful for ranking suspects, but framework operator names do not always correspond one-to-one with GPU kernels. A single operator may launch many kernels; several operators may be fused by a compiler.

### 3.2 Nsight Systems: the global critical path

Nsight Systems shows the complete timeline across Python, CPU threads, CUDA API calls, kernels, memory transfers, streams, and NCCL collectives. It answers system-level questions:

- Why is the GPU idle?
- Is the CPU feeding the device fast enough?
- Are communication and computation overlapping?
- Does the VAE run serially with the denoiser when it could overlap?
- Is checkpointing freezing the loop?
- Are many tiny kernels creating launch-bound execution?
- Which synchronization event creates the critical gap?

Semantic ranges—using NVTX or framework annotations—are essential. Without them, the trace becomes a wall of kernel names instead of a model execution narrative.

### 3.3 Nsight Compute: kernel-level diagnosis

Once one kernel is known to matter, Nsight Compute provides the microscope:

- achieved memory bandwidth;
- cache hit rates;
- global-memory transaction efficiency;
- tensor-core utilization;
- instruction mix;
- register and shared-memory pressure;
- theoretical and achieved occupancy;
- warp stall reasons;
- roofline position.

The correct order is:

1. identify the expensive phase globally;
2. identify the expensive operation inside that phase;
3. isolate the important kernel;
4. inspect the kernel's limiting resource.

Starting with a random kernel reverses the causal chain.

### 3.4 A credible baseline report

For a real model, a baseline should include at least:

- end-to-end training-step time or inference latency;
- throughput in domain units, such as frames/s or videos/GPU-hour;
- forward, backward, optimizer, VAE, data-loading, checkpoint, and serialization time;
- peak and steady-state HBM usage;
- kernel-launch count and distribution;
- GPU idle fraction;
- NCCL time and the exposed fraction of communication;
- achieved TFLOP/s or model FLOPs utilization when the FLOP estimate is defensible;
- exact hardware, software versions, tensor shapes, precision, and batch policy;
- p50, p95, and p99 latency for a service rather than only the mean.

When model FLOPs are known, model FLOPs utilization can be reported as

\[
\operatorname{MFU}
=
\frac{F_{\text{model per step}}}
{T_{\text{step}}\,N\,P_{\text{peak}}},
\]

where \(F_{\text{model per step}}\) is the algorithmic model FLOP count, \(T_{\text{step}}\) is measured step time, \(N\) is the number of GPUs, and \(P_{\text{peak}}\) is peak per-GPU throughput for the relevant precision. MFU is useful only when the FLOP accounting and hardware peak are stated consistently; it does not capture data loading, communication, or useful work omitted from the model FLOP estimate.

The hard part is not creating the trace. It is explaining every large region of the critical path and selecting the highest-value next experiment.

---

## 4. Kernel engineering: fusion, tiling, and persistence

Framework code is written in tensor operations. GPUs execute kernels. The gap between those levels is where many important performance losses occur.

### 4.1 Why Triton is the right middle layer

CUDA C++ provides the deepest control, but Triton exposes the most important kernel-design problems with a shorter development loop:

- decomposition of work into programs;
- blocked tensor layouts;
- coalesced loads and stores;
- masking and edge handling;
- reductions;
- shared on-chip reuse;
- fusion;
- launch geometry;
- autotuning;
- persistent execution.

For modern ML workloads, Triton is often the fastest route from a profiler finding to a correct integrated kernel. CUDA C++ becomes necessary when the required control, hardware primitive, or integration path lies below what Triton exposes cleanly.

### 4.2 The canonical kernel progression—and what each kernel teaches

The standard Triton examples form a compact map of GPU concepts rather than a checklist:

| Kernel | Main systems idea |
|---|---|
| Vector addition | Program indexing, masks, validation, benchmark mechanics |
| Fused softmax | Reductions and eliminating repeated HBM traffic |
| Matrix multiplication | Tiling, reuse, launch geometry, tensor-core layouts |
| Layer normalization | Parallel reductions and a custom backward pass |
| Fused attention | Online normalization and avoiding a materialized attention matrix |
| Persistent matrix multiplication | Keeping work resident and reducing repeated scheduling overhead |

The real objective is not tutorial completion. It is the ability to inspect a PyTorch trace, choose one expensive operation, and replace it with a correct faster implementation.

### 4.3 Fusion

Consider a pointwise chain

\[
y = \phi(a \odot x + b),
\]

where \(x\) is read from HBM, scaled by \(a\), shifted by \(b\), transformed by \(\phi\), and written back.

An eager implementation may materialize intermediate tensors and launch several kernels. A fused kernel reads each required input, performs the chain in registers, and writes the final output once.

Fusion is most valuable when:

- operations are bandwidth-bound;
- intermediate tensors are large;
- launch overhead is material;
- the fused kernel does not create prohibitive register pressure;
- the shape is common enough to justify specialized code.

Fusion is not automatically beneficial. An oversized fused kernel can reduce occupancy, complicate scheduling, or prevent reuse of a highly optimized library primitive.

### 4.4 Example: fused adaptive normalization in a video DiT

A video diffusion transformer may repeatedly compute

\[
y = \operatorname{Norm}(x) \odot (1+s) + b,
\]

where:

- \(x\) is a hidden activation;
- \(\operatorname{Norm}\) is LayerNorm or RMSNorm;
- \(s\) is a conditioning-dependent scale;
- \(b\) is a conditioning-dependent shift;
- \(\odot\) is elementwise multiplication.

A naive execution can launch separate kernels for statistics, normalization, scale adjustment, multiplication, and addition. A fused implementation can:

1. load the hidden row;
2. compute normalization statistics, usually with FP32 accumulation;
3. normalize the values;
4. apply the conditioning modulation;
5. write the output once.

This is a plausible optimization because the operation repeats across many transformer blocks. It is still only valuable if profiling shows that the repeated launches and memory traffic occupy a meaningful fraction of the critical path.

### 4.5 Persistent kernels

A conventional launch assigns a finite tile of work to each program instance. A **persistent kernel** keeps a controlled set of programs resident and lets each program process multiple work units. This can reduce scheduling overhead, improve cache reuse, or enable specialized producer-consumer pipelines.

Persistence is powerful when the workload and hardware mapping are stable. It can be harmful when it creates load imbalance, monopolizes SMs, or assumes shapes that do not generalize.

### 4.6 The correct kernel-development protocol

A custom kernel should pass four gates.

#### Gate 1: relevance

- The operation is visible in an end-to-end trace.
- Its contribution is large enough for the expected speedup to matter under Amdahl's law.
- The model uses a stable family of shapes rather than one synthetic benchmark shape.

#### Gate 2: correctness

- Forward results match a trusted reference under declared tolerances.
- Backward gradients match when training requires them.
- Edge shapes, non-power-of-two dimensions, and all production dtypes are tested.
- NaNs, infinities, accumulation precision, and deterministic behavior are checked.

#### Gate 3: isolated performance

- Warm-up is complete.
- Compilation time is excluded from steady-state measurements.
- GPU synchronization is handled correctly.
- Enough repetitions are run to report stable percentiles.
- The comparison includes the strongest relevant baseline: eager, compiled, vendor library, or another fused implementation.

#### Gate 4: end-to-end impact

- The kernel is integrated into the actual model.
- The full trace is rerun.
- Model-level latency or throughput improves.
- Memory and output quality are revalidated.

A kernel that clears only the microbenchmark gate is a research exercise, not a systems result.

---

## 5. Distributed training: partitioning computation, state, and communication

Multi-GPU training is not one technique. It is a set of different tensor-partitioning strategies composed around the model's memory and communication structure.

The central questions are:

> What is replicated? What is sharded? Which ranks communicate? Which collective moves the data? How many bytes move? Can that movement overlap useful computation?

### 5.1 The main parallelism dimensions

| Strategy | What is partitioned? | Common communication | Primary purpose |
|---|---|---|---|
| Data parallelism / DDP | Input batch; each rank holds a full model replica | Gradient AllReduce | Increase throughput when the model fits per GPU |
| Fully sharded data parallelism / FSDP | Parameters, gradients, optimizer state | Parameter AllGather, gradient ReduceScatter | Reduce per-rank model-state memory |
| Tensor parallelism | Individual linear algebra operations, heads, or channels | AllReduce, AllGather, ReduceScatter | Split layers too large or inefficient for one GPU |
| Sequence/context parallelism | Tokens or spatiotemporal positions | AllGather, ReduceScatter, AllToAll, point-to-point exchange | Reduce activation memory for long sequences |
| Pipeline parallelism | Consecutive groups of layers | Activation and gradient Send/Recv | Partition very deep models across devices |
| Expert parallelism | Mixture-of-experts experts and routed tokens | AllToAll or fused dispatch/combine | Scale sparse expert capacity |

No strategy dominates universally. The right composition depends on:

- parameter count;
- activation size;
- sequence or video-token length;
- batch size;
- interconnect topology;
- compute-to-communication ratio;
- acceptable implementation complexity;
- fault-tolerance and checkpointing requirements.

For video models, context or sequence parallelism is often unusually important because activation memory grows across both space and time.

### 5.2 Collectives as tensor transformations

The most important NCCL collectives should be understood as explicit data transformations:

- **AllReduce:** combine corresponding values across ranks and return the result to every rank.
- **AllGather:** gather each rank's shard so that every rank receives the complete tensor.
- **ReduceScatter:** reduce corresponding values and leave each rank with one output shard.
- **AllToAll:** send a different shard from every source rank to every destination rank.
- **Send/Recv:** point-to-point transfer, commonly used between pipeline stages.

ReduceScatter followed by AllGather is functionally equivalent to an AllReduce, though performance and memory behavior depend on implementation and scheduling.

For a ring AllReduce over \(N\) ranks and a tensor of \(M\) bytes, each rank transfers approximately

\[
2\frac{N-1}{N}M
\]

bytes: one reduce-scatter phase and one all-gather phase. This approximation makes an important point visible: large gradient or activation tensors can put communication directly on the critical path unless they are bucketed and overlapped.

### 5.3 Topology is part of the algorithm

The same logical collective behaves differently over:

- NVLink or NVSwitch within a node;
- PCIe between local devices;
- InfiniBand across nodes;
- Ethernet when no high-performance fabric is available.

Rank placement matters. A tensor-parallel group with heavy per-layer communication should usually use the fastest local links available. Data-parallel groups can often tolerate slower cross-node links better because communication occurs at coarser boundaries and may overlap the backward pass.

A parallelism plan that ignores physical topology is incomplete.

### 5.4 Communication overlap

Suppose one phase requires compute time \(T_c\) and communication time \(T_m\). If they are serialized, the phase costs

\[
T_c + T_m.
\]

If communication is launched early and overlaps perfectly with independent computation, the lower bound becomes

\[
\max(T_c,T_m).
\]

Real systems sit between these extremes. Overlap depends on:

- when tensors become ready;
- bucket size;
- stream scheduling;
- process-group configuration;
- available SM and copy-engine resources;
- whether communication kernels contend with compute kernels;
- load balance across ranks.

The key metric is not total NCCL time. It is **exposed NCCL time on the critical path**.

### 5.5 Megatron Core as an executable reference architecture

Megatron Core is valuable because it implements tensor, pipeline, context, data, and expert parallelism in one composable system rather than as isolated diagrams. Its code makes concrete:

- process-group construction;
- partitioned linear layers;
- sequence-parallel normalization;
- pipeline schedules and bubbles;
- distributed optimizer state;
- mixed-precision integration;
- distributed checkpointing.

It is one of the highest-density open-source references for understanding how large transformer-family models are actually partitioned and executed.[^megatron]

### 5.6 Measuring scaling honestly

For \(N\) GPUs, define parallel efficiency as

\[
E_N
=
\frac{X_N}{N X_1},
\]

where:

- \(X_N\) is throughput on \(N\) GPUs;
- \(X_1\) is throughput on one GPU under a comparable workload.

The scaling regime must also be declared:

- **Strong scaling** keeps the global workload fixed and asks how much faster the same work completes as GPUs are added.
- **Weak scaling** increases the workload with GPU count and asks whether per-GPU throughput remains stable.

The same efficiency number can tell a different story under these two regimes. A credible scaling report includes:

- throughput on 1, 2, 4, and 8 GPUs, and beyond when available;
- \(E_N\) at each scale;
- per-rank HBM use;
- total and exposed communication time;
- rank imbalance;
- pipeline bubble fraction;
- global and microbatch sizes;
- tensor shapes and sequence length;
- precision and activation-checkpointing policy;
- node and interconnect topology.

“We ran on eight GPUs” is not a scaling result. It is merely a launch configuration.

---

## 6. Training-runtime reliability is part of performance

Long training jobs fail. Nodes disappear, preemptions happen, filesystems stall, and one rank can crash after every other rank has entered a collective. A robust runtime treats failure as an expected operating condition.

### 6.1 What a restartable checkpoint must preserve

At minimum:

- model parameters;
- optimizer state;
- learning-rate scheduler state;
- gradient-scaler state when used;
- random-number-generator state;
- data-loader and sampler position;
- distributed sharding metadata;
- model and data configuration;
- code version or commit identifier.

Restoring only the model weights is not resuming training. It is initializing a related new run.

### 6.2 Data correctness after restart

A distributed sampler must avoid silently duplicating or skipping examples after recovery. This is especially important for streaming datasets, large sharded video corpora, and jobs that checkpoint between gradient-accumulation boundaries.

A strong test records sample identifiers and verifies that a preemption-resume cycle produces the expected coverage and ordering semantics.

### 6.3 Asynchronous checkpointing

Synchronous checkpointing can pause every rank while state is copied and written. Asynchronous checkpointing moves some or all of that work off the foreground critical path, often by staging state into CPU buffers and writing concurrently.[^dcp]

The relevant metrics are:

- foreground pause per checkpoint;
- additional host-memory pressure;
- write bandwidth;
- time until a checkpoint is durably committed;
- behavior when a failure occurs during an in-flight save.

A checkpointing improvement can save more cluster time than a glamorous kernel optimization. Systems value is determined by critical-path impact, not by proximity to CUDA.

### 6.4 Operational observability

A training runtime should expose:

- step-time distributions, not only averages;
- data-loader wait time;
- per-rank memory and utilization;
- collective duration and stragglers;
- checkpoint state and age;
- numerical anomalies;
- failed or retried samples;
- progress measured in useful units such as frames or tokens processed.

Without observability, performance regressions and partial failures become anecdotes instead of diagnosable events.

---

## 7. Inference serving: scheduling and memory management on a shared GPU

Serving is not wrapping a model in an HTTP endpoint. The hard problem is managing a finite accelerator under variable arrivals, shapes, output lengths, memory demands, cancellations, and failures.

A request typically moves through:

1. admission and queueing;
2. preprocessing or conditioning;
3. model initialization or prefill;
4. iterative execution;
5. postprocessing and serialization;
6. streaming or final response.

The scheduler controls which requests share the device at each boundary.

### 7.1 Continuous and in-flight batching

Static batching waits for a fixed group, executes it together, and releases the entire group when complete. This wastes capacity when requests have different lengths or arrive continuously.

Continuous or in-flight batching allows the active batch to change at iteration boundaries. Completed requests leave; newly admitted requests enter when compatible. Mature LLM runtimes use this to increase utilization while controlling latency.[^vllm][^trtllm]

For a diffusion or world-model server, the analogous boundary may be:

- a denoising step;
- an autoregressive video chunk;
- a simulator transition;
- a refinement stage;
- a VAE encode/decode phase.

The scheduler must respect state compatibility. Requests with different shapes, precision modes, guidance settings, or state layouts may not form a valid tensor batch.

### 7.2 Memory pools and paged state

Autoregressive LLM runtimes devote substantial engineering to the KV cache. Paged allocation reduces fragmentation and lets the runtime manage request state in blocks rather than reserving one worst-case contiguous tensor per request. vLLM's PagedAttention, for example, is part of a broader runtime combining paged KV management, continuous batching, chunked prefill, prefix caching, and CUDA Graph execution.[^vllm]

Video and world models have different state, but the transferable principle is the same:

> Treat persistent model state as a managed memory resource with explicit allocation, reuse, eviction, and admission policies.

Possible state includes:

- conditioning embeddings;
- temporal attention caches;
- latent history;
- recurrent world state;
- 3D memory or feature grids;
- VAE intermediates;
- control or action histories.

### 7.3 Separating setup from iterative execution

LLM serving distinguishes **prefill** from **decode** because their computational profiles differ. Prefill processes many input tokens with high parallelism; decode processes a small number of new tokens repeatedly and is often launch- or memory-sensitive.

An analogous decomposition for video generation is:

- conditioning and text/video/3D encoding;
- initial latent or state preparation;
- repeated denoising or causal rollout;
- VAE decoding and output encoding.

Separating these phases enables different batching, hardware placement, and scheduling policies. SGLang and other modern runtimes also explore prefill/decode disaggregation to prevent long setup work from repeatedly interrupting latency-sensitive iterative work.[^sglang]

### 7.4 CUDA Graphs

CUDA Graphs capture a stable sequence of launches and replay it with much lower CPU launch overhead. They are attractive when:

- shapes are stable;
- control flow is fixed;
- memory addresses can remain stable;
- the same execution segment repeats many times.

Iterative denoising and causal rollout can be excellent candidates, but dynamic shapes, changing batch composition, and request-specific branches complicate capture. Production runtimes therefore maintain multiple captured graph sizes, pad to compatible sizes, or fall back to eager execution when a batch cannot use a graph.[^vllm-graphs]

### 7.5 Speculative execution

Speculative decoding uses a cheaper proposal process and a more expensive verifier to complete multiple accepted steps per target-model iteration. The general systems pattern is broader than language:

1. generate a cheap candidate trajectory or update;
2. verify or correct it with the expensive model;
3. retain the computational gain when acceptance is high.

For diffusion or world models, possible analogues include cheap draft denoisers, lower-resolution proposals, distilled transition models, or adaptive-step integration. These are algorithmic changes as well as systems changes, so quality and workload equivalence must be reported explicitly.

### 7.6 Admission control

A production server should predict whether a request fits before launching it. Admission can use:

- estimated peak HBM;
- shape and frame count;
- number of iterative steps;
- cache footprint;
- current fragmentation and active allocations;
- service priority;
- latency budget.

The alternative—accept everything and hope the allocator survives—is not high utilization. It is uncontrolled failure.

### 7.7 Tail latency and saturation

As offered load approaches service capacity, queueing delay rises sharply. Little's law provides a useful steady-state identity:

\[
L = \lambda W,
\]

where \(L\) is the average number of requests in the system, \(\lambda\) is completed-request throughput, and \(W\) is average time in the system. It does not predict the full latency distribution, but it makes one operational fact unavoidable: sustained queue growth means arrivals exceed effective service capacity.

Mean latency hides the behavior users actually feel. A serving benchmark should report:

- p50, p95, and p99 end-to-end latency;
- queueing time separated from execution time;
- throughput across an arrival-rate sweep;
- time to first output and total completion time where streaming exists;
- GPU utilization and HBM at each load level;
- the saturation point where queueing grows without bound;
- cancellation and overload behavior;
- goodput under the declared service-level objective.

A single successful request proves only functional correctness.

### 7.8 What transfers from LLM runtimes to video and world models

| LLM runtime mechanism | Transferable systems principle | Video/world-model analogue |
|---|---|---|
| Continuous batching | Change the active batch at iterative boundaries | Regroup at denoising steps or causal chunks |
| Paged KV cache | Manage persistent state in reusable blocks | Latent, temporal-cache, 3D-memory, or conditioning pools |
| Prefill/decode scheduling | Separate setup-heavy and iteration-heavy phases | Encoder/conditioning versus denoising/rollout |
| Prefix caching | Reuse shared request state | Shared prompts, camera context, scene history, or controls |
| CUDA Graphs | Replay stable launch sequences | Fixed-shape denoising or simulator transitions |
| Speculative decoding | Cheap proposal plus expensive verification | Draft rollout, distilled denoiser, adaptive refinement |
| Quantized execution | Trade numerical precision for capacity and speed | FP8 transformer blocks or quantized caches |

The implementation is model-specific. The runtime principles are not.

---

## 8. Low precision: performance constrained by numerical behavior

Low precision reduces memory traffic and can unlock higher-throughput tensor-core paths. It also changes the numerical system. Treating FP8 as a compiler switch is a category error.

### 8.1 A scaling mental model

Conceptually, a tensor \(x\) is scaled before low-precision representation:

\[
q = Q(x/s),
\qquad
\hat{x}=s q,
\]

where:

- \(s\) is a scale factor;
- \(Q\) maps values into the representable low-precision format;
- \(q\) is the stored low-precision value;
- \(\hat{x}\) is the reconstructed value used by later computation.

The scale trades overflow against quantization resolution. A scale too small causes large values to overflow or saturate. A scale too large wastes precision on the tensor's typical values.

Floating-point formats are not integer quantizers, so the exact mapping differs, but this model captures why scale selection matters.

### 8.2 BF16 versus FP8

BF16 keeps the eight-bit exponent range of FP32 but has far fewer mantissa bits. It is therefore robust to a wide dynamic range and has become a common training baseline.

FP8 uses eight total bits and requires more active scaling management. Different FP8 formats trade exponent range against precision, and many systems use different formats for forward tensors and backward gradients.

The benefit can include:

- higher matrix-multiplication throughput on supported hardware;
- lower activation and communication volume;
- lower memory use;
- larger feasible batches or models.

The risk includes:

- overflow or underflow;
- quantization noise;
- unstable scale statistics;
- convergence changes;
- operations that must remain in BF16 or FP32.

### 8.3 Scaling policies

Transformer Engine exposes several practical FP8 recipes.[^te]

| Policy | Scale granularity and timing | Main trade-off |
|---|---|---|
| Current per-tensor scaling | One scale derived from the current tensor | Responsive, but computing the current maximum can require an additional tensor read |
| Delayed per-tensor scaling | One scale predicted from historical maxima | Avoids some current-tensor overhead, but can lag distribution shifts |
| Blockwise scaling | Independent scales for local tensor blocks | Better adaptation to outliers and local dynamic range, with extra scale metadata and hardware constraints |
| Microscaling formats | Small blocks with compact shared scales | Strong locality and efficiency on supported newer hardware |

Blockwise schemes can also reduce the need to synchronize a single global scale across distributed shards because scales are local to blocks.[^te-block]

### 8.4 The experiment that matters

On one real model, compare at least:

1. BF16 training baseline;
2. FP8 training;
3. BF16 inference;
4. FP8 inference;
5. per-tensor and blockwise recipes where supported.

Hold the workload fixed and report:

- step time and throughput;
- end-to-end inference latency;
- peak and steady-state HBM;
- communication volume where precision affects collectives;
- convergence speed;
- validation loss;
- final domain metrics;
- overflow, NaN, and scale statistics;
- qualitative regressions;
- operations retained in higher precision.

The quality tolerance should be declared before examining the final result. Otherwise “no meaningful quality loss” becomes a moving target.

Toy networks are useful for checking API mechanics. They do not establish that a precision strategy works for a real video model with long training dynamics and sensitive perceptual outputs.

---

## 9. Why video diffusion and world models are distinct systems workloads

The same GPU principles apply across foundation models, but video and world models place unusual pressure on the stack.

### 9.1 Spatiotemporal token growth

Suppose a latent video has:

- \(T\) temporal positions;
- spatial size \(H \times W\);
- patch sizes \(p_t, p_h, p_w\).

The approximate token count is

\[
L
=
\frac{T}{p_t}
\frac{H}{p_h}
\frac{W}{p_w}.
\]

Full self-attention has roughly quadratic dependence on token count:

\[
\text{attention work}
= O(L^2 d),
\]

where \(d\) is the head or hidden dimension up to architecture-specific factors. Attention-memory pressure also grows with pairwise token interactions unless the implementation avoids materializing the full matrix.

This makes temporal length, spatial resolution, and patching inseparable systems variables. Doubling one dimension can affect more than one bottleneck at once.

Many practical video architectures use factorized, local, sparse, windowed, or causal attention, changing the exact complexity. The systems lesson remains: the token geometry must be treated explicitly when designing parallelism and kernels.

### 9.2 Iterative execution

Diffusion inference repeats a large denoiser for multiple steps. If one denoiser call costs \(T_d\), VAE and conditioning cost \(T_o\), and the sampler uses \(K\) model evaluations, then a first-order latency model is

\[
T_{\text{generation}}
\approx
K T_d + T_o.
\]

This creates several optimization axes:

- reduce \(T_d\) through kernels, compilation, precision, and parallelism;
- reduce \(K\) through distillation or better integration;
- overlap or accelerate \(T_o\);
- batch compatible requests across iterative steps.

Reducing \(K\) changes the algorithm and may change quality. It should not be mixed into a claim about pure runtime optimization unless the comparison includes the new quality-speed trade-off.

### 9.3 VAE and output pipelines

Video VAEs can consume material latency and memory. They may also create host-side work through frame conversion, video encoding, and storage. An optimized denoiser can reveal the VAE or serialization path as the next bottleneck.

Potential systems work includes:

- tiled encode/decode;
- asynchronous execution on separate streams or devices;
- precision tuning;
- batched frame conversion;
- pipelined video encoding;
- avoiding unnecessary CPU round trips.

### 9.4 Causal state and long-horizon memory

Interactive world models often maintain temporal context, latent history, attention caches, geometry, or action state across chunks. This makes inference resemble a stateful service rather than an independent image-generation call.

The runtime must decide:

- which state remains on GPU;
- what can be compressed or evicted;
- how state is partitioned across ranks;
- how requests migrate between workers;
- how cancellation frees resources;
- how long sessions are protected from fragmentation or starvation.

These are the same class of problems solved by cache managers and schedulers in mature LLM runtimes, but with model-specific state semantics.

### 9.5 A bottleneck map for a modern video model

| Phase | Common bottleneck | Likely systems lever |
|---|---|---|
| Text/action/3D conditioning | Small kernels, CPU dispatch, redundant encoding | Caching, fusion, graph capture |
| Spatiotemporal attention | Compute, HBM, quadratic intermediates | Flash-style kernels, context parallelism, locality |
| MLP and projections | Tensor-core throughput | Layouts, fused epilogues, FP8 |
| Normalization/modulation | Bandwidth and launch count | Triton fusion |
| Denoising loop | Repeated launches and total model evaluations | CUDA Graphs, batching, distillation with quality controls |
| VAE decode | Compute, memory, output size | Tiling, overlap, precision, separate placement |
| Multi-GPU execution | Activation and gradient communication | Topology-aware TP/CP/FSDP and overlap |
| Serving | Variable shapes, state memory, queueing | Admission, pools, scheduling, bucketing |

The model architecture tells the systems engineer where reuse and parallelism are possible. The profiler tells them which possibility is currently valuable.

---

## 10. An end-to-end optimization methodology

A credible optimization project follows a disciplined loop.

### Step 1: Freeze the benchmark contract

Record:

- model checkpoint;
- input or prompt set;
- random seeds;
- resolution and frame count;
- sequence length;
- sampler and number of model evaluations;
- batch and concurrency policy;
- precision;
- hardware and software versions;
- quality metrics and tolerance.

Changing these silently invalidates the comparison.

### Step 2: Profile the complete system

Capture framework attribution and a system timeline. Mark major phases. Identify the actual critical path and quantify its largest components.

### Step 3: Form one causal hypothesis

Examples:

- repeated normalization/modulation launches are bandwidth- and launch-bound;
- gradient communication begins too late to overlap backward computation;
- the CPU creates gaps between short denoiser kernels;
- variable-shape requests prevent graph replay and efficient batching;
- per-tensor FP8 scaling is sensitive to activation outliers.

The hypothesis should predict what a successful trace will look like.

### Step 4: Change one intended variable

Implement the kernel, scheduling policy, sharding plan, checkpoint path, or precision recipe. Avoid bundling several unrelated changes into one benchmark.

### Step 5: Validate locally

Test correctness, numerical behavior, failure modes, and isolated performance.

### Step 6: Re-profile end to end

Measure whether the critical path moved. A successful optimization often reveals a new bottleneck. That is progress, not failure.

### Step 7: Revalidate quality and reliability

Run the declared domain metrics, qualitative checks, restart tests, and load tests. Faster wrong outputs are not an optimization.

### Step 8: Publish the evidence

A strong report contains:

- before/after traces;
- matched workload configuration;
- microbenchmark and end-to-end results;
- memory measurements;
- scaling curves;
- quality deltas;
- limitations and shapes where the optimization does not help;
- reproducible commands and environment information.

### Benchmark integrity rules

- Do not change resolution, frame count, sequence length, or step count without labeling it a different workload.
- Do not compare a cold baseline against a warmed optimized path.
- Do not include compilation in one measurement and exclude it from the other.
- Do not report only the best run; report distributions.
- Do not claim multi-GPU scalability without a scaling curve.
- Do not claim low-precision success without convergence and final-quality measurements.
- Do not call queue coalescing “tensor batching” unless the model actually executes a batched tensor path.
- Do not stop at a microbenchmark when the claim concerns an application.

---

## 11. A practical implementation sequence through the stack

The guide is organized by systems layer, but the most efficient hands-on sequence is still cumulative.

### Performance fundamentals

Study and actively use:

- CUDA's programming and memory model;
- streams and synchronization;
- occupancy and latency hiding;
- coalescing and shared-memory tiling;
- roofline reasoning;
- PyTorch Profiler;
- Nsight Systems;
- Nsight Compute.

The deliverable is an end-to-end profile of a real model with a defensible bottleneck analysis—not a collection of notes.

### Kernel engineering

Work through:

- vector addition;
- fused softmax;
- matrix multiplication;
- layer normalization;
- fused attention;
- persistent matrix multiplication.

Then replace one expensive operation in an actual PyTorch trace with a correct faster kernel and prove the model-level delta.

### Distributed execution

Learn:

- DDP and FSDP2;
- tensor parallelism;
- sequence/context parallelism;
- pipeline parallelism;
- NCCL collectives;
- rank topology;
- communication overlap;
- Megatron Core.

The deliverable is a 1/2/4/8-GPU scaling report plus deterministic, data-correct checkpoint resume.

### Serving systems

Study and benchmark:

- vLLM;
- SGLang;
- TensorRT-LLM;
- continuous/in-flight batching;
- paged state management;
- prefill/decode scheduling;
- speculative execution;
- quantization;
- CUDA Graphs;
- admission control and load testing.

The deliverable is a service with bounded memory behavior and a latency-throughput curve under controlled arrivals.

### Low precision

Use Transformer Engine to compare:

- a BF16 baseline;
- FP8 training;
- FP8 inference;
- per-tensor versus blockwise scaling;
- throughput and HBM;
- convergence and final quality.

Run the study on a real video or world model. Small demos establish mechanics, not systems evidence.

---

## 12. What strong systems evidence looks like

The highest-value artifacts are concrete:

- profiler traces with explained idle regions;
- a model-relevant kernel that improves end-to-end runtime;
- parallel scaling curves with communication analysis;
- restart-equivalent distributed training;
- a load-tested inference runtime with tail-latency metrics;
- a controlled BF16/FP8 quality-performance study;
- reproducible benchmark scripts and machine-readable results.

The valuable professional profile is not:

> “A model researcher who has also read about deployment.”

It is:

> **“A video and world-model researcher who can make frontier models train and run substantially faster on real GPU clusters—and prove it without sacrificing quality.”**

That combination is rare because it requires fluency across mathematical models, numerical behavior, kernels, communication, scheduling, and production reliability. It is also exactly where increasingly large and interactive generative models encounter their hardest practical constraints.

---

## Primary references

### CUDA and profiling

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)

### Triton and kernels

- [Triton tutorial gallery](https://triton-lang.org/main/getting-started/tutorials/)
- [Vector addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Layer normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [Fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Persistent matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
- [Block-scaled matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)

### Distributed training

- [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [DistributedDataParallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [FSDP2 `fully_shard`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [Pipeline Parallelism](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
- [Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [Asynchronous Distributed Checkpointing](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)

### Inference serving

- [vLLM documentation](https://docs.vllm.ai/)
- [SGLang documentation](https://docs.sglang.ai/)
- [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

### Low precision

- [Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)
- [FP8 and FP4 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [FP8 current scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html)
- [FP8 delayed scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)
- [FP8 blockwise scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html)

[^megatron]: NVIDIA, [Megatron Core User Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html).
[^dcp]: PyTorch, [Asynchronous Saving with Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html).
[^vllm]: vLLM, [official documentation](https://docs.vllm.ai/).
[^vllm-graphs]: vLLM, [CUDA Graphs design documentation](https://docs.vllm.ai/en/stable/design/cuda_graphs/).
[^sglang]: SGLang, [official documentation](https://docs.sglang.ai/) and [prefill/decode disaggregation](https://docs.sglang.ai/advanced_features/pd_disaggregation.html).
[^trtllm]: NVIDIA, [TensorRT-LLM overview](https://nvidia.github.io/TensorRT-LLM/overview.html).
[^te]: NVIDIA, [Using FP8 and FP4 with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).
[^te-block]: NVIDIA, [FP8 Blockwise Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html).
