# High-Performance AI Compute & Serving Lab: CUDA, TensorRT & vLLM

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Managed with uv](https://img.shields.io/badge/Package%20Manager-uv-purple.svg)](https://astral.sh/uv)
[![Architecture Dual-Mode](https://img.shields.io/badge/Mode-Local%20%2F%20Remote%20GPU-green.svg)](#dual-mode-architecture)

A rigorous, hands-on engineering testbed and conceptual knowledge base exploring the core mechanics, performance trade-offs, failure modes, and production patterns of:
1. **CUDA:** Hardware execution hierarchy, warp scheduling, memory coalescing, bank conflicts, and custom kernels.
2. **TensorRT & TensorRT-LLM:** Graph compilation, horizontal/vertical layer fusion, kernel auto-tuning, and quantization (FP16/INT8/FP8).
3. **vLLM & PagedAttention:** Zero-waste virtual KV cache management, continuous batching, chunked prefill, and high-concurrency serving.
4. **Agentic AI Pipeline Integration:** Low-latency tool-use execution, Automatic Prefix Caching (APC) for massive system prompts and tool schemas, structured output logit-masking (XGrammar), and swarm concurrency.

---

## Architecture Spectrum

```
+--------------------------------------------------------------------------------------------------+
|                                     AGENTIC AI PIPELINE                                          |
|  Multi-Turn ReAct Loops | Tool Execution Loops | Swarm Concurrency | Structured Outputs (JSON)   |
+--------------------------------------------------------------------------------------------------+
                                                │
                                                ▼  (HTTP / gRPC / IPC)
+--------------------------------------------------------------------------------------------------+
|                                    HIGH-CONCURRENCY SERVING                                      |
|                                       vLLM / SGLang                                              |
|  - PagedAttention (Zero-Waste KV Memory)            - Continuous (Iteration-Level) Batching     |
|  - Automatic Prefix Caching (APC)                  - Chunked Prefill & Speculative Decoding     |
|  - Guided Decoding (XGrammar / Outlines)           - Distributed Tensor Parallelism             |
+--------------------------------------------------------------------------------------------------+
                                                │
                                                ▼  (Model Backend / Graph Execution)
+--------------------------------------------------------------------------------------------------+
|                                 GRAPH COMPILER & ACCELERATOR                                     |
|                                 TensorRT / TensorRT-LLM (TRT)                                    |
|  - Graph Optimization & Layer Fusion               - Precision Quantization (FP16/INT8/FP8)     |
|  - Architecture-Specific Kernel Auto-Tuning        - Static Engine Scheduling & Dynamic Shapes  |
+--------------------------------------------------------------------------------------------------+
                                                │
                                                ▼  (Hardware Primitives)
+--------------------------------------------------------------------------------------------------+
|                                     CUDA PLATFORM & RUNTIME                                      |
|  - Thread Hierarchy: Grids -> Blocks -> Warps (32 threads) -> Threads                            |
|  - Memory Hierarchy: Registers -> Shared Memory -> L1/L2 -> HBM3e/GDDR                           |
|  - Asynchronous Execution: Streams, Events, and CUDA Graphs                                      |
+--------------------------------------------------------------------------------------------------+
```

---

## 5-Milestone Progressive Roadmap

* **[x] Milestone 1: CUDA Architecture & Execution Model Fundamentals**
  - [x] Hardware Hierarchy: SMs, Warps (32 threads), SIMT, Latency Hiding.
  - [x] Memory Coalescing & Stride Penalties.
  - [x] Lab 01 Implementation: Vector Addition, Thread Math & Coalescing Benchmark.
  - [x] Notes: `notes/00_master_taxonomy_cuda_trt_vllm.md`, `notes/01_cuda_fundamentals.md`.

* **[ ] Milestone 2: Kernel Optimization, Shared Memory & Triton**
  - [ ] Shared Memory Tiling for GEMM (General Matrix Multiply).
  - [ ] Eliminating 32-way Bank Conflicts via Column Padding and XOR Swizzling.
  - [ ] Reduction Progression: Harris Stages 1 through 6 (`__shfl_down_sync`).
  - [ ] Lab 02 Implementation: Tiled GEMM & Bank Conflict Profiler.

* **[ ] Milestone 3: TensorRT Graph Compilation & Quantization**
  - [ ] The AOT Pipeline: PyTorch $\to$ ONNX $\to$ TRT Execution Engine (`.plan`).
  - [ ] Horizontal & Vertical Layer Fusions (CBR, FMHA, RMSNorm+Residual).
  - [ ] Dynamic Shapes & Optimization Profiles (`min`, `opt`, `max`).
  - [ ] Quantization: FP32 vs. FP16 vs. INT8 PTQ (Calibrators) & FP8 (E4M3/E5M2).
  - [ ] Lab 03 Implementation: Model Export, Dynamic Profiles & Quantization Profiler.

* **[ ] Milestone 4: vLLM, PagedAttention & High-Throughput Serving**
  - [ ] KV Cache Memory Math: Derivations, GQA vs MHA, Fragmentation analysis.
  - [ ] PagedAttention Implementation: Logical-to-Physical Block Tables.
  - [ ] Continuous (Iteration-Level) Batching vs Static Batching.
  - [ ] Chunked Prefill & Smoothing Inter-Token Latency ($P_{99}$).
  - [ ] Lab 04 Implementation: PagedAttention Simulator & Serving Benchmarks.

* **[ ] Milestone 5: Agentic AI Engineering & Production Deployment**
  - [ ] Automatic Prefix Caching (APC) Evaluation for 50+ Tool Schemas.
  - [ ] Guided Decoding with XGrammar: 100% Guaranteed JSON Tool Calling.
  - [ ] Multi-Agent Swarm Concurrency & Consistent Hash Routing.
  - [ ] Lab 05 Implementation: Agentic Tool Latency Benchmark (TTFT vs TPOT).

---

## Conceptual Notes & Technical Deep Dives

* [`notes/00_master_taxonomy_cuda_trt_vllm.md`](file:///notes/00_master_taxonomy_cuda_trt_vllm.md):
  The definitive conceptual reference. What, Where, Types, Why, Why Not, When, and How for the core technologies.
* [`notes/01_cuda_fundamentals.md`](file:///notes/01_cuda_fundamentals.md):
  SM architecture, memory hierarchy latencies, warp scheduling, memory coalescing, and bank conflicts.
* [`notes/02_tensorrt_compilation_and_quantization.md`](file:///notes/02_tensorrt_compilation_and_quantization.md):
  AOT graph compiler pipeline, layer fusion, dynamic profiles, and INT8/FP8 quantization.
* [`notes/03_vllm_and_paged_attention.md`](file:///notes/03_vllm_and_paged_attention.md):
  KV cache memory math, virtual memory paging, continuous batching, and chunked prefill.
* [`notes/04_agentic_ai_engineering_guide.md`](file:///notes/04_agentic_ai_engineering_guide.md):
  Optimizing serving for Agentic AI: TTFT elimination via prefix caching, canonical prompt layouts, and guided JSON decoding.
* [`notes/05_sglang_and_radix_attention.md`](file:///notes/05_sglang_and_radix_attention.md):
  SGLang deep dive: Dynamic RadixTree KV reuse vs block hash chains, Jump-Forward FSM decoding for structured JSON, and DeepSeek-R1 serving.
* [`notes/06_megatron_distributed_systems.md`](file:///notes/06_megatron_distributed_systems.md):
  Megatron-LM & 3D/4D Parallelism: Tensor Parallelism (column/row GEMM math), Pipeline Parallelism (1F1B schedule), Sequence & Context Parallelism.
* [`notes/07_miles_and_tinker_agentic_post_training.md`](file:///notes/07_miles_and_tinker_agentic_post_training.md):
  Emerging post-training & agentic infrastructure: MILES (distributed SGLang rollouts + Megatron training via Ray) and Tinker (cloud-native training API primitives).

---

## Getting Started

### 1. Environment Setup
This repository uses [`uv`](https://astral.sh/uv) to manage Python 3.12 dependencies:

```bash
# Sync environment
uv sync
```

### 2. Hardware Diagnostics
Run the diagnostic inspector to verify local hardware vs cloud GPU mode:

```bash
uv run python scripts/check_env.py
```

### 3. Running Lab 01 Exercises

```bash
# Run vector addition and inspect thread block math
uv run python labs/lab01_cuda_foundations/vector_add.py

# Benchmark memory coalescing & stride penalty
uv run python labs/lab01_cuda_foundations/coalescing_bench.py

# Run unit tests
uv run pytest labs/lab01_cuda_foundations/verify.py -v
```

---

## Dual-Mode Architecture

* **Local Host Mode:** If running on a CPU-only workstation, scripts automatically operate in **Simulation & Verification Mode**, executing exact algorithmic decomposition, cache profiling, and ONNX graph exports.
* **Target GPU Cloud Mode:** When dispatched to an NVIDIA GPU (via Docker, RunPod, Modal, or SSH remote), scripts automatically switch to **Native CUDA Kernel Execution** with full hardware performance.
