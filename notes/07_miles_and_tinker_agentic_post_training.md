# Technical Guide: MILES & Tinker — Emerging Post-Training & Agentic Infrastructure

---

## 1. The Post-Training & Agentic RL Shift

Frontier AI systems (DeepSeek-R1, OpenAI o1/o3, Qwen-2.5-Coder) achieve autonomous reasoning and multi-turn tool proficiency through **post-training reinforcement learning (RL)**:
- **Group Relative Policy Optimization (GRPO)**
- **Proximal Policy Optimization (PPO)**
- **Multi-turn code & tool execution in interactive sandboxes**

### The Systems-Engineering Conflict:
1. **Rollout Phase (Inference):** Requires high KV cache reuse, dynamic sequence batching, speculative decoding, and low latency across millions of environment-interaction steps.
2. **Update Phase (Training):** Requires 3D distributed parallelism (TP + PP + DP + EP), autograd backward graphs, and optimizer states.

**MILES** and **Tinker** represent the two leading approaches solving this challenge from infrastructure-level and developer-level perspectives.

---

## 2. MILES (RadixArk): Enterprise Distributed Post-Training

* **Origin:** Developed by **RadixArk** (arXiv:2609.08368, *Miles v0.1: Production-Level Post-Training*), built as the evolutionary successor to `slime`.
* **Repository:** [`https://github.com/radixark/miles`](https://github.com/radixark/miles)

```
+-----------------------------------------------------------------------------------+
|                                  MILES RUNTIME                                    |
|                      (Orchestrated via Ray / Ray Clusters)                        |
+-----------------------------------------------------------------------------------+
       │                                                              │
       ▼                                                              ▼
+-------------------------------+              +-----------------------------------+
|       ROLLOUT ENGINE          |              |         TRAINING ENGINE           |
|          (SGLang)             |              |      (Megatron-LM / FSDP2)        |
+-------------------------------+              +-----------------------------------+
| - RadixAttention (KV reuse)   |              | - 3D Parallelism (TP + PP + DP)   |
| - Token-in-Token-Out (TITO)   |              | - Expert Parallelism (EP) for MoE |
| - Rollout Routing Replay (R3) |              | - Gradient accumulation & clipping|
| - Agentic Sandbox Execution   |              | - AdamW / Distributed Optimizer   |
+-------------------------------+              +-----------------------------------+
       ▲                                                              │
       │                   Fast Weight Synchronization                │
       +══════════════════════════════════════════════════════════════+
                   - Direct P2P GPU-to-GPU NCCL / RDMA Transfer
                   - Low-precision (FP8/BF16) Zero-Disk Streaming
                   - Eliminates synchronization bubble idle time
```

### Core Innovations:
1. **Engine Decoupling (SGLang + Megatron-LM):** Uses **SGLang** as the high-throughput inference worker pool and **Megatron-LM / FSDP2** as the distributed gradient worker pool.
2. **Zero-Disk P2P Weight Sync:** Replaces slow disk checkpoints with peer-to-peer memory transfers directly over NVLink and InfiniBand RDMA, syncing updated training weights to SGLang workers in sub-seconds.
3. **Rollout Routing Replay (R3):** Leverages SGLang's RadixAttention to preserve multi-turn agent prefixes across environment trajectories (Bash, Python REPL, web tools).

---

## 3. Tinker (Thinking Machines Lab): API-First Training Primitives

* **Origin:** Developed by **Thinking Machines Lab** (founded by former OpenAI CTO Mira Murati).
* **Repository:** [`https://github.com/thinking-machines-lab/tinker-cookbook`](https://github.com/thinking-machines-lab/tinker-cookbook) (`pip install tinker`)
* **Philosophy:** Instead of forcing developers to manage Slurm clusters, CUDA drivers, and Ray workers, Tinker exposes low-level training and sampling primitives over a clean, pipelined cloud API.

```
  +-------------------------------------------------------+
  |              LOCAL DEVELOPER / CPU MACHINE            |
  | (Jupyter Notebook, Python Script, or Agentic Sandbox) |
  +-------------------------------------------------------+
        │                  │                     │
        │ .forward_backward() │ .optim_step()    │ .sample()
        ▼                  ▼                     ▼
  +-------------------------------------------------------+
  |             TINKER REMOTE TRAINING SERVICE            |
  |              (Thinking Machines Managed GPU)          |
  +-------------------------------------------------------+
  | - Persistent Base Model in GPU VRAM (e.g. Qwen, Llama)|
  | - Dynamic Multi-Tenant LoRA Adapters                  |
  | - Low-overhead Backprop & Optimizer State Execution   |
  | - Instantaneous Adapter Switching for Rollouts        |
  +-------------------------------------------------------+
```

### Core Primitives:
* `training_client.forward_backward(batch, loss_fn)`: Computes autograd gradients on remote GPUs and returns loss statistics asynchronously.
* `training_client.optim_step(optimizer_params)`: Updates weights on remote GPUs using AdamW/SGD.
* `sampling_client.sample(prompts, ...)`: Generates completions using the *newly updated* policy weights immediately.

---

## 4. Where They Fit: The Complete Modern AI Systems Stack

```
+-----------------------------------------------------------------------------------------+
| Level 4: Agent Applications & Environments                                              |
| Tools, Code Interpreters, SWE-bench, OSWorld, LangGraph, AutoGen                        |
+-----------------------------------------------------------------------------------------+
       │ Trajectories & Feedback                                 ▲ Policy Generations
       ▼                                                         │
+-----------------------------------------------------------------------------------------+
| Level 3: Post-Training, Fine-Tuning & RL Frameworks                                    |
| [MILES] (Self-hosted distributed cluster)  │  [TINKER] (API-driven primitives runtime)   |
| [veRL / HybridFlow]                       │  [OpenRLHF / Slime]                        |
+-----------------------------------------------------------------------------------------+
       │ Model Weights & Gradients                               ▲ Fast KV-Cache Rollouts
       ▼                                                         │
+-----------------------------------------------------------------------------------------+
| Level 2: Distributed Training & Inference Backends                                      |
| Training: NVIDIA Megatron-LM, FSDP2       │  Inference: SGLang, vLLM, TensorRT-LLM      |
+-----------------------------------------------------------------------------------------+
       │ Computation Graphs & Communication Calls                │
       ▼                                                         ▼
+-----------------------------------------------------------------------------------------+
| Level 1: Kernel & Acceleration Libraries                                                |
| FlashAttention, FlashDecoding, CUTLASS, Triton, NCCL, NVLink, RDMA                      |
+-----------------------------------------------------------------------------------------+
       │ Hardware Execution                                      │
       ▼                                                         ▼
+-----------------------------------------------------------------------------------------+
| Level 0: Physical Hardware & Compute                                                    |
| NVIDIA Hopper (H100/H200), Blackwell (B200/GB200/GB300), AMD Instinct (MI300X)          |
+-----------------------------------------------------------------------------------------+
```

---

## 5. Comparative Decision Matrix

| Dimension | MILES (`radixark/miles`) | Tinker (`thinking-machines-lab`) |
| :--- | :--- | :--- |
| **Primary Philosophy** | Full cluster infrastructure optimization | Developer experience & cloud API abstraction |
| **Hosting Model** | **Self-hosted** (Bare-metal, AWS, GCP, Slurm, K8s) | **Cloud API & Managed GPU Service** |
| **Inference Backend** | Native **SGLang** (RadixAttention, TITO) | Remote server-side managed sampling |
| **Training Backend** | **Megatron-LM / FSDP2** | Proprietary managed backend (LoRA / Full) |
| **Weight Synchronization**| Zero-disk direct P2P NCCL/RDMA | Managed hot-swapping behind cloud API |
| **Target Scale** | 30B–670B+ Dense/MoE models, multi-node clusters | Interactive agent RL, fast prototyping |
| **User Profile** | AI Infrastructure Engineers, Foundation Model Labs | AI Engineers, Agent Builders, Researchers |
