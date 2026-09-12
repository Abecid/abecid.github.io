---
layout: post
_styles: |
  .post-content figure { margin: 1.75rem 0 2rem; }
  .post-content figure img { display: block; width: 100%; }
  .post-content figcaption.caption {
    max-width: 76ch;
    margin: 0.75rem 0 0;
    text-align: left;
    font-size: 0.875rem;
    line-height: 1.55;
  }
  .post-content img { max-width: 100%; height: auto; }
  .post-content mjx-container[display="true"] {
    display: block;
    max-width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0.25em 0;
  }
  .post-content table { display: block; max-width: 100%; overflow-x: auto; }
title: "GPU Systems: Follow the Bottleneck"
date: 2026-07-28
description: "How to find the cost that matters, estimate the speedup budget, and improve GPU kernels, distributed training, and serving."
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

A kernel gets 3× faster. The training step improves by 3%. Both numbers can be correct.

GPU optimization starts by finding what delays the result: bytes moved, arithmetic, launch gaps, or communication that finishes too late. Estimate how much time removing that cost could save, change one mechanism, and check the full workload again.

This post develops that reasoning from one kernel to distributed training and serving. Video generation is a recurring example: a diffusion transformer (DiT) repeatedly updates latents, and a variational autoencoder (VAE) converts them to pixels. Accelerating one stage can make another stage dominate.

## Contents

1. [The critical path and Amdahl’s law](#1-the-governing-principle-optimize-the-critical-path)
2. [Execution, memory, and roofline reasoning](#2-performance-fundamentals-compute-memory-and-parallelism)
3. [Measure the application, then the kernel](#3-measurement-from-the-application-timeline-to-one-kernel)
4. [Fusion, tiling, and custom kernels](#4-kernel-engineering-fusion-tiling-and-persistence)
5. [Distributed training and overlap](#5-distributed-training-partitioning-computation-state-and-communication)
6. [Checkpoints and recovery](#6-training-runtime-reliability-is-part-of-performance)
7. [Serving: batching, state, and queues](#7-inference-serving-scheduling-and-memory-management-on-a-shared-gpu)
8. [Low precision and scaling](#8-low-precision-performance-constrained-by-numerical-behavior)
9. [Video and world-model workloads](#9-why-video-diffusion-and-world-models-are-distinct-systems-workloads)
10. [Run a controlled optimization experiment](#10-an-end-to-end-optimization-methodology)
11. [Experiments and evidence](#11-an-experiment-sequence-you-can-actually-run)

---

## 1. The governing principle: optimize the critical path

A **critical path** is the longest dependency chain required to finish the measured work. Shortening an operation off that path may free resources without reducing latency.

Think of two concurrent operations taking 8 ms and 6 ms, with a result that needs both. Making the 6 ms operation twice as fast still leaves an 8 ms wait. Making the 8 ms operation twice as fast leaves a 6 ms wait: the bottleneck moves.

An **exposed cost** is time that remains after overlap. This is why summed kernel, memory, and communication durations do not form a valid latency budget: they can cover the same wall-clock interval.

### Latency, throughput, and goodput are different objectives

- **Latency** is the time to complete one unit of work: one training step, one request, one generated video, or one autoregressive chunk.
- **Throughput** is completed work per unit time: samples/s, tokens/s, frames/s, or requests/s.
- **Goodput** is useful work that satisfies the system's quality or service constraint. A server processing 100 requests/s is not achieving 100 requests/s of goodput if half violate the latency objective or fail.

A change that improves throughput can worsen single-request latency. Larger batches are the standard example. Systems work begins by declaring which objective matters and under what workload distribution.

### Amdahl sets the optimization budget

For a fixed workload, suppose a non-overlapped operation takes fraction $f$ of runtime. Accelerate only that operation by $s$, leaving the rest unchanged:

$$
S_{\text{total}}
=
\frac{1}{(1-f)+f/s}.
$$

If a kernel is only 5% of runtime, making it infinitely fast improves the full application by at most

$$
\frac{1}{0.95}\approx 1.053.
$$

At 3× kernel speed, the application gains only $1/(0.95+0.05/3) \approx 1.034$—about 3.4%. The microbenchmark is having a better day than the application.

Rank opportunities by removable critical-path time. For overlapping work, use the timeline to estimate that fraction; adding up kernel durations can double-count time.

{% include figure.liquid path="assets/img/blogs/gpu-systems/amdahl.svg" mobile_path="assets/img/blogs/gpu-systems/amdahl-mobile.svg" alt="A 100 ms run contains 95 ms of other work and 5 ms in the target kernel. A threefold kernel speedup saves only 3.33 ms." caption="Figure 1. Amdahl in milliseconds. Illustrative arithmetic, not a benchmark: even deleting the target kernel leaves 95 ms." width=1260 height=525 mobile_width=825 mobile_height=570 zoomable=true avoid_scaling=true %}

---

## 2. Performance fundamentals: compute, memory, and parallelism

### 2.1 The CUDA execution model

A CUDA kernel launches a **grid** of thread **blocks**. Each block contains threads, and the GPU schedules threads in groups called **warps**—32 threads on NVIDIA GPUs. Blocks execute on streaming multiprocessors, or SMs.

Threads in a block can cooperate through shared memory and block-level synchronization. Different blocks are generally independent in a conventional launch; cross-block coordination needs global memory plus an appropriate synchronization mechanism, often a new kernel boundary.

The mapping from tensor elements to blocks, warps, and lanes determines parallelism and memory access. The same formula can run very differently under a different mapping.

### 2.2 The GPU memory hierarchy

**HBM** is high-bandwidth device memory. **SRAM** is the smaller on-chip storage used for registers, shared memory, and caches. **Tensor cores** are specialized matrix arithmetic units. Moving data to those units is often harder than doing the arithmetic:

| Storage               |          Scope | Relative capacity |               Relative speed | Main systems implication                                           |
| --------------------- | -------------: | ----------------: | ---------------------------: | ------------------------------------------------------------------ |
| Registers             |     Per thread |              Tiny |                      Fastest | Excess use reduces resident warps and may spill                    |
| Shared memory         |      Per block |             Small |                    Very fast | Enables reuse and cooperation; layout can cause bank conflicts     |
| L1 cache              |         Per SM |             Small |                         Fast | Helps local reuse but is not a substitute for good access patterns |
| L2 cache              |    Device-wide |          Moderate |              Faster than HBM | Reuse across blocks may benefit, subject to working-set size       |
| HBM/global memory     |    Device-wide |             Large | High bandwidth, high latency | Repeated reads and writes often dominate elementwise workloads     |
| Host or remote memory | Outside device |       Much larger |                   Far slower | Transfers must be minimized, pinned, batched, and overlapped       |

The GPU is fast because it can sustain enormous parallel throughput—not because any individual global-memory access is cheap. A good kernel either performs enough independent work to hide latency or reuses data from registers/shared memory so that expensive transfers are amortized.

### 2.3 Coalescing

Threads in a warp should access nearby memory addresses whenever possible. When adjacent threads access adjacent elements, the hardware can serve the warp with a small number of memory transactions. Strided or scattered access may require many transactions and waste bandwidth.

This is **memory coalescing**. For example, a warp reading 32 adjacent FP32 elements requests a contiguous 128 bytes. Giving those lanes addresses separated by a large stride can require many more memory sectors to serve the same useful payload. Alignment and the architecture determine the exact transaction count.

### 2.4 Tiling and reuse

Matrix multiplication illustrates the central optimization pattern. A naive implementation repeatedly loads matrix elements from HBM. A tiled implementation loads submatrices into shared memory or registers, reuses them for many multiply-accumulate operations, and only then advances to the next tile.

Tiling increases useful arithmetic per transferred byte. Attention, convolution, and reductions exploit the same trade: keep a working set on chip long enough to reuse it, without exhausting the registers or shared memory needed for parallel execution.

### 2.5 Arithmetic intensity and the roofline model

Define arithmetic intensity as

$$
I
=
\frac{\text{floating-point operations}}{\text{bytes transferred from the limiting memory level}}.
$$

A simplified roofline bound is

$$
P_{\text{attainable}}
\leq
\min\left(P_{\text{peak}},\;B_{\text{memory}}I\right),
$$

where:

- $P_{\text{attainable}}$ is achieved arithmetic throughput;
- $P_{\text{peak}}$ is peak compute throughput for the relevant datatype and instruction path;
- $B_{\text{memory}}$ is sustained bandwidth from the limiting memory level;
- $I$ is arithmetic intensity.

Low-intensity operations are usually memory-bound. High-intensity operations may become compute-bound. An operation far below both roofs is often losing performance to poor occupancy, dependencies, uncoalesced access, serialization, synchronization, or launch overhead.

The roofline model is not a complete simulator. It is a classification tool: it tells you whether the next optimization should primarily reduce bytes, increase useful arithmetic throughput, or repair execution inefficiency.

{% include figure.liquid path="assets/img/blogs/gpu-systems/roofline.svg" mobile_path="assets/img/blogs/gpu-systems/roofline-mobile.svg" alt="A calculated roofline rises with arithmetic intensity until reaching 120 TFLOP/s at 60 FLOP per byte. Arrows show removing stalls at fixed intensity, then increasing reuse." caption="Figure 2. Hypothetical roofline at 2 TB/s and 120 TFLOP/s. Removing stalls moves a kernel upward at fixed intensity. Reusing data raises intensity and can raise its bandwidth ceiling. The square marks the vector-add bound; the other points illustrate possible improvements, not measurements." width=1260 height=660 mobile_width=825 mobile_height=660 zoomable=true avoid_scaling=true %}

**Put numbers on it.** Suppose a hypothetical device sustains 2 TB/s and 120 TFLOP/s for the chosen arithmetic path. The roofline knee is $120/2=60$ FLOP/byte. FP32 vector addition reads two inputs and writes one output: roughly 12 bytes per addition, so $I\approx1/12$ FLOP/byte. Its bandwidth roof is only about 0.167 TFLOP/s. Low achieved FLOP/s is expected here; counting tensor cores will not rescue an operation that mostly moves bytes. Cache residency and extra traffic change the estimate.

### 2.6 Occupancy is a latency-hiding mechanism, not a score

**Occupancy** measures how many warps are resident on an SM relative to the hardware maximum. More resident warps can hide latency: while one warp waits for data, another can execute.

But maximizing occupancy is not the objective. A kernel with lower occupancy can be faster if it uses more registers to avoid spills, keeps more useful data on chip, or performs more work per thread. Conversely, high occupancy does not rescue a kernel with poor memory access or low instruction-level efficiency.

Check which resource limits independent work: registers, shared memory, block size, or an instruction dependency. That determines whether adding resident warps can help.

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

Start with the full workload. A profiler’s summed CUDA duration can exceed wall time because kernels overlap; a CPU timer around asynchronous launches can measure only submission. Decide which time you want before interpreting a number.

| Tool             | Question it resolves                                                     | Common misreading                                                       |
| ---------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| PyTorch Profiler | Which operators allocate memory, launch work, and dominate execution?    | Assuming one framework operator equals one kernel                       |
| Nsight Systems   | Which CPU, kernel, transfer, or collective dependency delays completion? | Treating total NCCL duration as fully exposed communication             |
| Nsight Compute   | Why does a known expensive kernel miss its memory or compute roof?       | Optimizing a detailed counter before establishing application relevance |

Use named ranges for conditioning, denoising, backward, optimizer, VAE, and checkpointing. An unexplained gap between two kernels is a lead: inspect the preceding CPU work, stream dependency, scalar readback, or allocation. Do not assume the GPU needs a better kernel while it is waiting for Python to submit one.

### 3.1 Time the intended boundary

For single-process end-to-end CUDA latency, pass the workload function to this helper. It includes CPU dispatch and completion of queued device work:

```python
import time
import torch

def latency_samples_ms(run_workload, warmup=10, repeats=30):
    # Warm the exact shape/dtype path; account for compile time separately.
    for _ in range(warmup):
        run_workload()
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_workload()
        torch.cuda.synchronize()
        samples_ms.append(1000 * (time.perf_counter() - start))
    return samples_ms
```

This measures serialized trials on the current device; it is not a concurrent-serving benchmark. Warm-up counts depend on the workload. CUDA events are useful for device intervals, but their stream placement and dependencies must include the intended work. For a distributed step, measure completion across all participating ranks. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).

### 3.2 What to record

Keep the workload definition beside the result: shapes, precision, checkpoint, batch policy, hardware, software versions, warm-up, and timing boundary. Record latency distributions, throughput, peak HBM, and an annotated trace. Include domain quality metrics when the change affects numerical or algorithmic behavior.

If the model FLOP count is defensible, model FLOPs utilization is

$$
\operatorname{MFU}=\frac{F_{\mathrm{model/step}}}{T_{\mathrm{step}}N P_{\mathrm{peak}}}.
$$

The numerator is total algorithmic model work across $N$ GPUs; the denominator uses elapsed step time and the per-GPU peak for the selected precision. Communication, input stalls, and other overhead lower MFU by lengthening the denominator. MFU does not identify which overhead caused that loss, and its value depends on whether recomputation is included in the FLOP convention.

---

## 4. Kernel engineering: fusion, tiling, and persistence

### 4.1 Choosing an implementation layer

First compare eager execution with the framework compiler and the relevant vendor/library primitive. A custom kernel is justified when profiling exposes a useful specialization those paths miss. Triton provides blocked tensor operations, masks, reductions, launch control, and autotuning with a short development loop. CUDA C++ offers lower-level control when the required primitive or scheduling mechanism needs it.

A compiler can remove intermediate tensors automatically. Check the generated execution before spending a weekend manually fusing something it already fused.

### 4.2 Pick a reference kernel for the bottleneck

Use [fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) to study redundant memory traffic, [matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) for tiled reuse, and [fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) for avoiding a large intermediate matrix. Each gives a concrete implementation of an IO argument.

### 4.3 Fusion

Consider a pointwise chain

$$
y = \phi(a \odot x + b),
$$

where $x$ is read from HBM, scaled by $a$, shifted by $b$, transformed by $\phi$, and written back.

An eager implementation may materialize intermediate tensors and launch several kernels. A fused kernel reads each required input, performs the chain in registers, and writes the final output once.

The strongest candidates combine large intermediate tensors, little arithmetic, and repeated launches. The limit is the on-chip working set: excessive register use can reduce occupancy or spill intermediates back to memory. Check the compiled/library baseline too; it may already fuse this chain.

{% include figure.liquid path="assets/img/blogs/gpu-systems/fusion.svg" mobile_path="assets/img/blogs/gpu-systems/fusion-mobile.svg" alt="Three operators send intermediate tensors down to HBM and back up, making six memory transfers. Fusion keeps both intermediate results on chip, leaving one input read and one output write." caption="Figure 3. Six HBM crossings become two. With scalar a and b and N elements of w bytes, ideal traffic falls from 6Nw to 2Nw. Cache hits, tensor-valued coefficients, and register spills change the count." width=1350 height=765 mobile_width=840 mobile_height=780 zoomable=true avoid_scaling=true %}

The same IO argument drives [Triton’s fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html). A row reduction adds a constraint: the working row must fit the chosen on-chip strategy.

### 4.4 Example: fused adaptive normalization in a video DiT

A video diffusion transformer may repeatedly compute

$$
y = \operatorname{Norm}(x) \odot (1+s) + b,
$$

Here $x$ is a hidden activation, $s$ and $b$ are conditioning-dependent scale and shift, and $\operatorname{Norm}$ is LayerNorm or RMSNorm.

A fused kernel loads a hidden row, computes its statistics (usually accumulating in FP32), normalizes it, applies the modulation, and writes once. The row reduction makes this harder than a pointwise chain: its layout and working-set size must support cooperation across lanes. Repetition across transformer blocks makes a small per-row saving worth measuring.

### 4.5 Persistent kernels

A conventional launch assigns a finite tile of work to each program instance. A **persistent kernel** keeps a controlled set of programs resident and lets each program process multiple work units. This can reduce scheduling overhead, improve cache reuse, or enable specialized producer-consumer pipelines.

Persistence is powerful when the workload and hardware mapping are stable. It can be harmful when it creates load imbalance, monopolizes SMs, or assumes shapes that do not generalize.

### 4.6 Proving the replacement works

Validate forward values and, when needed, backward gradients against a trusted implementation. Include edge shapes, non-power-of-two dimensions, dtypes, and declared numerical tolerances. Compare against the strongest applicable baseline, including compiled and library paths.

Benchmark the actual shape distribution after compilation and warm-up, then integrate the kernel and repeat the full workload. An isolated gain can disappear through extra copies, dispatch, register spills, or reduced overlap. The trace should show which cost disappeared, not merely a different kernel name.

---

## 5. Distributed training: partitioning computation, state, and communication

A parallelism scheme decides what each rank stores, what it computes, and when it must exchange tensors. Follow one tensor through those three decisions before choosing a combination of schemes.

### 5.1 The main parallelism dimensions

| Strategy                              | What is partitioned?                                     | Common communication                                        | Primary purpose                                   |
| ------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------- |
| Data parallelism / DDP                | Input batch; each rank holds a full model replica        | Gradient AllReduce                                          | Increase throughput when the model fits per GPU   |
| Fully sharded data parallelism / FSDP | Parameters, gradients, optimizer state                   | Parameter AllGather, gradient ReduceScatter                 | Reduce per-rank model-state memory                |
| Tensor parallelism                    | Individual linear algebra operations, heads, or channels | AllReduce, AllGather, ReduceScatter                         | Split layers too large or inefficient for one GPU |
| Sequence/context parallelism          | Tokens or spatiotemporal positions                       | AllGather, ReduceScatter, AllToAll, point-to-point exchange | Reduce activation memory for long sequences       |
| Pipeline parallelism                  | Consecutive groups of layers                             | Activation and gradient Send/Recv                           | Partition very deep models across devices         |
| Expert parallelism                    | Mixture-of-experts experts and routed tokens             | AllToAll or fused dispatch/combine                          | Scale sparse expert capacity                      |

Parameter state, live activations, and interconnect bandwidth constrain the choice. For video models, long spatiotemporal sequences can make activation sharding as important as parameter sharding.

### 5.2 Collectives as tensor transformations

NVIDIA’s NCCL library implements GPU collectives. Read each as a tensor transformation:

- **AllReduce:** combine corresponding values across ranks and return the result to every rank.
- **AllGather:** gather each rank's shard so that every rank receives the complete tensor.
- **ReduceScatter:** reduce corresponding values and leave each rank with one output shard.
- **AllToAll:** send a different shard from every source rank to every destination rank.
- **Send/Recv:** point-to-point transfer, commonly used between pipeline stages.

ReduceScatter followed by AllGather is functionally equivalent to an AllReduce, though performance and memory behavior depend on implementation and scheduling.

For a ring AllReduce over $N$ ranks and a tensor of $M$ bytes, each rank sends approximately

$$
2\frac{N-1}{N}M
$$

bytes, and receives the same amount. The two phases are reduce-scatter and all-gather; summing sent and received bytes doubles this expression. **Bucketing** groups ready tensors into collective calls: small buckets can launch earlier but pay more per-call overhead; large buckets amortize overhead but may delay overlap.

### 5.3 Topology is part of the algorithm

Place groups with frequent per-layer exchanges on the fastest available links, commonly local NVLink/NVSwitch. Coarser data-parallel exchanges can often tolerate cross-node links better because they may overlap the backward pass. The actual topology—PCIe, local fabric, and inter-node network—sets the cost of the same logical collective.

### 5.4 Communication overlap

Suppose one phase requires compute time $T_c$ and communication time $T_m$. If they are serialized, the phase costs

$$
T_c + T_m.
$$

If communication is launched early and overlaps perfectly with independent computation, the lower bound becomes

$$
\max(T_c,T_m).
$$

Useful overlap requires ready tensors and independent work. Buckets that become ready too late leave a tail; communication kernels can also contend with compute for device resources. Measure **exposed NCCL time on the critical path**.

{% include figure.liquid path="assets/img/blogs/gpu-systems/overlap.svg" mobile_path="assets/img/blogs/gpu-systems/overlap-mobile.svg" alt="An illustrative schedule takes 14 ms when 8 ms compute and 6 ms communication serialize, versus 10 ms when communication starts after 4 ms." caption="Figure 4. Launch readiness sets the overlap budget. Starting the 6 ms collective at t=4 hides 4 ms behind compute and leaves 2 ms exposed. The ideal 8 ms bound would require earlier readiness and no resource contention." width=1260 height=660 mobile_width=825 mobile_height=660 zoomable=true avoid_scaling=true %}

### 5.5 An implementation to inspect

Megatron Core composes tensor, pipeline, context, data, and expert parallelism. Use its process-group construction, partitioned layers, pipeline schedules, and optimizer/checkpoint paths to connect the partition table to executable code.[^megatron]

For a model-state memory estimate, state your optimizer assumptions. One common mixed-precision Adam layout has a 2-byte parameter, a 2-byte gradient, a 4-byte master parameter, and two 4-byte moments: about 16 bytes per parameter before activations, temporary buffers, or allocator overhead. Sharding reduces persistent per-rank state, but all-gathered layer parameters and live activations still determine the peak. Dividing total model bytes by GPU count is an optimistic starting point, not a fit guarantee.

### 5.6 Measuring scaling honestly

For $N$ GPUs, define parallel efficiency as

$$
E_N
=
\frac{X_N}{N X_1},
$$

where:

- $X_N$ is throughput on $N$ GPUs;
- $X_1$ is throughput on one GPU under a comparable workload.

The scaling regime must also be declared:

- **Strong scaling** keeps the global workload fixed and asks how much faster the same work completes as GPUs are added.
- **Weak scaling** increases the workload with GPU count and asks whether per-GPU throughput remains stable.

Report both the scaling regime and global/microbatch sizes. Keep per-rank HBM, topology, exposed collective time, imbalance, and pipeline idle time beside the 1/2/4/8-GPU throughput curve. If the model cannot fit on one GPU, choose and disclose a feasible reference configuration instead of inventing a single-GPU baseline.

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

The ordering matters: capture a consistent snapshot, finish staging it into independently owned buffers, then let training mutate the live state while writing proceeds. Mark the checkpoint usable only after its required shards and metadata are complete. “The background thread started” is not a durability guarantee. Measure staging pause, host-memory pressure, write time, and recovery from an interrupted save.

### 6.4 Test recovery as part of the runtime

At a known step, save, interrupt, restore, and compare the resumed sample IDs and optimizer/RNG state with an uninterrupted reference. Record the numerical tolerance rather than assuming every distributed kernel is bitwise deterministic. During real runs, track step-time tails, per-rank waits, checkpoint age, and failed/retried samples. A straggler that appears only every thousand steps still spends cluster time.

---

## 7. Inference serving: scheduling and memory management on a shared GPU

A serving runtime shares finite compute and memory across arrivals with different shapes, output lengths, and deadlines. Its pipeline is admission → conditioning/prefill → repeated model execution → decode/serialization. The scheduler decides what runs together and when a request’s state can be released.

### 7.1 Continuous and in-flight batching

Static batching waits for a fixed group, executes it together, and releases the entire group when complete. This wastes capacity when requests have different lengths or arrive continuously.

Continuous or in-flight batching allows the active batch to change at iteration boundaries. Completed requests leave; newly admitted requests enter when compatible. Mature LLM runtimes use this to increase utilization while controlling latency.[^vllm][^trtllm]

For diffusion or world models, scheduling boundaries can be denoising steps or causal chunks. Requests still need compatible tensor shapes, state layouts, and execution paths; an HTTP queue full of jobs is not automatically a GPU batch.

{% include figure.liquid path="assets/img/blogs/gpu-systems/batching.svg" mobile_path="assets/img/blogs/gpu-systems/batching-mobile.svg" alt="Static batching leaves a hatched empty slot after B ends at iteration 2. Continuous batching immediately fills that slot with C, finishing all requests by iteration 4 rather than 6." caption="Figure 5. A needs four iterations; B and queued C need two each. Hatching marks an unused slot. Static batching waits for A before admitting C; continuous batching refills B’s slot at the next boundary. Equal-cost iterations are assumed here; real prefill costs and batch-dependent iteration times must be measured." width=1260 height=645 mobile_width=825 mobile_height=645 zoomable=true avoid_scaling=true %}

The scheduler must respect state compatibility. Requests with different shapes, precision modes, guidance settings, or state layouts may not form a valid tensor batch.

### 7.2 Memory pools and paged state

Autoregressive LLM runtimes devote substantial engineering to the KV cache. Paged allocation reduces fragmentation and lets the runtime manage request state in blocks rather than reserving one worst-case contiguous tensor per request. vLLM's PagedAttention, for example, is part of a broader runtime combining paged KV management, continuous batching, chunked prefill, prefix caching, and CUDA Graph execution.[^vllm]

For video, persistent state may include conditioning embeddings, temporal key/value caches, latent history, or 3D feature grids. Give each an allocation, reuse, and eviction policy. A key/value cache stores earlier attention keys and values; reuse is valid only when the prefix, model configuration, and relevant conditioning agree.

### 7.3 Separating setup from iterative execution

LLM serving distinguishes **prefill** from **decode** because their computational profiles differ. Prefill processes many input tokens with high parallelism; decode processes a small number of new tokens repeatedly and is often launch- or memory-sensitive.

An analogous decomposition for video generation is:

- conditioning and text/video/3D encoding;
- initial latent or state preparation;
- repeated denoising or causal rollout;
- VAE decoding and output encoding.

Separating these phases enables different batching, hardware placement, and scheduling policies. SGLang and other modern runtimes also explore prefill/decode disaggregation to prevent long setup work from repeatedly interrupting latency-sensitive iterative work.[^sglang]

### 7.4 CUDA Graphs

CUDA Graphs capture a stable launch sequence and replay it with lower CPU launch overhead. Repeated fixed-shape denoising is a natural candidate. Changing shapes, addresses, control flow, or batch composition can invalidate that reuse, so runtimes keep multiple graph sizes, pad to compatible sizes, or fall back to eager execution.[^vllm-graphs]

### 7.5 Speculative execution

Speculative decoding uses a cheaper proposal process and a more expensive verifier to complete multiple accepted steps per target-model iteration. The general systems pattern is broader than language:

1. generate a cheap candidate trajectory or update;
2. verify or correct it with the expensive model;
3. retain the computational gain when acceptance is high.

For diffusion or world models, possible analogues include cheap draft denoisers, lower-resolution proposals, distilled transition models, or adaptive-step integration. These are algorithmic changes as well as systems changes, so quality and workload equivalence must be reported explicitly.

### 7.6 Admission control

Estimate peak HBM from shapes, length, cache growth, temporary workspaces, and active reservations before starting a request. Combine that capacity check with a queue deadline and priority policy. Reject or defer work that cannot fit; cancellation should reclaim its state promptly. Accepting requests faster than they can finish just converts a throughput problem into a queueing problem.

### 7.7 Tail latency and saturation

As offered load approaches service capacity, queueing delay rises sharply. Little's law provides a useful steady-state identity:

$$
L = \lambda W,
$$

where $L$ is the average number of requests in the system, $\lambda$ is completed-request throughput, and $W$ is average time in the system. It does not predict the full latency distribution, but it makes one operational fact unavoidable: sustained queue growth means arrivals exceed effective service capacity.

Sweep arrival rate and report p50/p95/p99 latency, goodput, memory use, and rejection rate. Separate queue time from execution; for streaming, also separate time to first output from total completion time. Test cancellation and overload at the point where the queue stops draining.

### 7.8 Transfer mechanisms, then check model semantics

Continuous batching transfers when work has safe iteration boundaries. Cache pooling transfers when state has a clear lifetime. Graph replay transfers when launch structure and addresses remain stable. A video pipeline may satisfy one condition and violate another—for example, a dynamic-resolution VAE after a fixed-shape denoising loop. Profile and schedule those phases separately.

---

## 8. Low precision: performance constrained by numerical behavior

Low precision reduces memory traffic and can unlock higher-throughput tensor-core paths. It also changes the numerical system. Treating FP8 as a compiler switch is a category error.

### 8.1 What the scale does

Conceptually, a tensor $x$ is scaled before low-precision representation:

$$
q = Q(x/s),
\qquad
\hat{x}=s q,
$$

where:

- $s$ is a scale factor;
- $Q$ maps values into the representable low-precision format;
- $q$ is the stored low-precision value;
- $\hat{x}$ is the reconstructed value used by later computation.

The scale trades overflow against quantization resolution. With this division convention, a scale too small can push large values beyond the format’s range; a scale too large can push small values toward subnormal values or zero. Floating-point spacing is relative within the normal range, so the precision effect is subtler than a uniform integer quantization grid.

### 8.2 BF16 versus FP8

BF16 keeps the eight-bit exponent range of FP32 but has far fewer mantissa bits. It is therefore robust to a wide dynamic range and has become a common training baseline.

FP8 uses eight total bits and requires more active scaling management. Different FP8 formats trade exponent range against precision, and many systems use different formats for forward tensors and backward gradients.

The savings apply to tensors and operations that actually use the reduced format. Extra range checks, scaling passes, metadata, or conversions cost time; sensitive operations may need BF16 or FP32. Evaluate those costs alongside overflow, underflow, and convergence.

### 8.3 Scaling policies

Transformer Engine exposes several practical FP8 recipes.[^te]

| Policy                     | Scale granularity and timing               | Main trade-off                                                                                            |
| -------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Current per-tensor scaling | One scale derived from the current tensor  | Responsive, but computing the current maximum can require an additional tensor read                       |
| Delayed per-tensor scaling | One scale predicted from historical maxima | Avoids some current-tensor overhead, but can lag distribution shifts                                      |
| Blockwise scaling          | Independent scales for local tensor blocks | Better adaptation to outliers and local dynamic range, with extra scale metadata and hardware constraints |
| Microscaling formats       | Small blocks with compact shared scales    | Strong locality and efficiency on supported newer hardware                                                |

Blockwise schemes can also reduce the need to synchronize a single global scale across distributed shards because scales are local to blocks.[^te-block]

### 8.4 Isolate numerical and runtime effects

Compare BF16 with each supported FP8 recipe under the same workload and evaluation. Track saturation, underflow, scale distributions, and operations retained in higher precision. Then measure training convergence or inference quality alongside latency and HBM. Quantization overhead can erase a small-GEMM speedup; tensor-core acceleration does not accelerate every operation in a block.

Declare the quality tolerance before comparing final results. A short forward-pass check establishes implementation compatibility, not training convergence or preservation of video motion.

---

## 9. Why video diffusion and world models are distinct systems workloads

The same GPU principles apply across foundation models, but video and world models place unusual pressure on the stack.

### 9.1 Spatiotemporal token growth

Suppose a latent video has:

- $T$ temporal positions;
- spatial size $H \times W$;
- patch sizes $p_t, p_h, p_w$.

The approximate token count is

$$
L
=
\frac{T}{p_t}
\frac{H}{p_h}
\frac{W}{p_w}.
$$

Full self-attention has roughly quadratic dependence on token count:

$$
\text{attention work}
= O(L^2 d),
$$

where $d$ is the head or hidden dimension up to architecture-specific factors. Attention-memory pressure also grows with pairwise token interactions unless the implementation avoids materializing the full matrix.

This makes temporal length, spatial resolution, and patching inseparable systems variables. Doubling one dimension can affect more than one bottleneck at once.

Many practical video architectures use factorized, local, sparse, windowed, or causal attention, changing the exact complexity. The systems lesson remains: the token geometry must be treated explicitly when designing parallelism and kernels.

### 9.2 Iterative execution

Diffusion inference repeats a large denoiser for multiple steps. If one denoiser call costs $T_d$, VAE and conditioning cost $T_o$, and the sampler uses $K$ model evaluations, then a first-order latency model is

$$
T_{\text{generation}}
\approx
K T_d + T_o.
$$

This creates several optimization axes:

- reduce $T_d$ through kernels, compilation, precision, and parallelism;
- reduce $K$ through distillation or better integration;
- overlap or accelerate $T_o$;
- batch compatible requests across iterative steps.

Reducing $K$ changes the algorithm and may change quality. It should not be mixed into a claim about pure runtime optimization unless the comparison includes the new quality-speed trade-off.

**Watch the bottleneck move.** Consider an illustrative pipeline with 40 ms per denoiser evaluation and 200 ms of fixed conditioning, VAE, and encoding work. Twenty evaluations cost 1,000 ms. Four cost 360 ms: a 5× reduction in denoiser calls buys only 2.78× end-to-end speedup. At one evaluation, fixed work is 83% of latency.

{% include figure.liquid path="assets/img/blogs/gpu-systems/denoising-budget.svg" mobile_path="assets/img/blogs/gpu-systems/denoising-budget-mobile.svg" alt="Stacked latency bars for 20, 4, and 1 denoiser evaluations show the fixed 200 ms pipeline cost dominating as evaluations decrease." caption="Figure 6. Calculated scenario: 40 ms per model evaluation plus 200 ms fixed work, with no overlap. These are assumptions, not measured GPU results. Reducing NFE also requires a separate quality comparison." width=1260 height=555 mobile_width=825 mobile_height=555 zoomable=true avoid_scaling=true %}

[Flow maps and distillation]({% post_url 2026-02-24-flowmap %}) target the number of evaluations. Once that succeeds, optimizing VAE decode or output encoding may matter more than another denoiser kernel.

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

| Phase                       | Common bottleneck                               | Likely systems lever                                      |
| --------------------------- | ----------------------------------------------- | --------------------------------------------------------- |
| Text/action/3D conditioning | Small kernels, CPU dispatch, redundant encoding | Caching, fusion, graph capture                            |
| Spatiotemporal attention    | Compute, HBM, quadratic intermediates           | Flash-style kernels, context parallelism, locality        |
| MLP and projections         | Tensor-core throughput                          | Layouts, fused epilogues, FP8                             |
| Normalization/modulation    | Bandwidth and launch count                      | Triton fusion                                             |
| Denoising loop              | Repeated launches and total model evaluations   | CUDA Graphs, batching, distillation with quality controls |
| VAE decode                  | Compute, memory, output size                    | Tiling, overlap, precision, separate placement            |
| Multi-GPU execution         | Activation and gradient communication           | Topology-aware TP/CP/FSDP and overlap                     |
| Serving                     | Variable shapes, state memory, queueing         | Admission, pools, scheduling, bucketing                   |

The model architecture tells the systems engineer where reuse and parallelism are possible. The profiler tells them which possibility is currently valuable.

---

## 10. An end-to-end optimization methodology

Use one experiment loop throughout the stack:

1. **Fix the contract.** Declare checkpoint, shapes, prompts/data, precision, quality tolerance, batch/concurrency, hardware, and timing boundaries.
2. **Locate exposed cost.** Capture a timeline and estimate the maximum gain from removing the suspected bottleneck.
3. **Predict the trace change.** “Fewer normalization launches and fewer HBM writes” is testable. “Better GPU utilization” is too vague.
4. **Change one mechanism.** Validate numerics and the relevant failure cases, then rerun the full workload.
5. **Explain the delta.** Compare latency distributions, memory, quality, and the trace. Keep compile costs and workload changes explicit.

If several things change together, measure them separately before attributing the result. Save negative results too: a fusion that slows the application through register pressure is a useful boundary on the design.

---

## 11. An experiment sequence you can actually run

| Experiment              | Change one mechanism                                                                | Evidence to keep                                                                |
| ----------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Establish the baseline  | Fix shapes, batch, precision, hardware, warm-up and timing boundaries               | End-to-end latency, throughput, peak HBM, annotated timeline                    |
| Remove a measured cost  | Fuse a repeated operation, improve tiling, or remove an unnecessary synchronization | Numerical comparison, kernel timing, full-step delta                            |
| Scale the same workload | Test 1/2/4/8 GPUs; state strong or weak scaling                                     | Throughput, exposed collective time, rank imbalance, topology                   |
| Interrupt training      | Resume a checkpoint at a known step                                                 | Sample IDs, optimizer and RNG state, loss continuity within declared tolerances |
| Load the server         | Sweep arrival rate and request lengths                                              | Goodput, p50/p95/p99, queue time, memory, rejection rate                        |
| Lower precision         | Compare against BF16 under the same evaluation                                      | Speed, HBM, saturation/overflow diagnostics, convergence and sample quality     |

Keep the baseline trace and the changed trace beside the configuration that produced each. Report compile and warm-up costs separately when they matter to deployment. Repeat enough runs to separate the delta from timing noise.

If a kernel gets faster and the application does not, find where the saved time went: a different bottleneck, added dispatch, more copies, or lost overlap. That failed speedup is useful evidence. Naming the kernel `fast_v2_final` is not.

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
