# Technical Guide: Megatron-LM & Distributed Systems Architecture

---

## 1. Why Megatron-LM is the Bedrock of Distributed AI

Developed by NVIDIA Applied Deep Learning Research (Mohammad Shoeybi et al.), **Megatron-LM** is the foundational framework that invented **3D/4D Parallelism** for Large Language Models.

### The Memory Barrier:
Training a 70B parameter model in FP16 requires:
$$\text{Memory}_{Train} = 2 \times 70\text{GB (Weights + Gradients)} + 12 \times 70\text{GB (Adam Optimizer States)} = 980 \text{ GB}$$
A single NVIDIA H100 has 80 GB of VRAM. Slicing models across multi-GPU nodes is mathematically mandatory.

---

## 2. Multi-Dimensional Parallelism Breakdown

### 2.1 Tensor Parallelism (TP): Intra-Node Weight Sharding
Slices individual linear projection weight matrices within a layer across $k$ GPUs connected via high-speed **NVLink** (900 GB/s on H100).

```
Multi-Head Attention Tensor Parallelism (TP = 2):
                       Input Activation X [B, S, H]
                       (Replicated on GPU 0 & GPU 1)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼ (Column-Parallel)                     ▼ (Column-Parallel)
             [GPU 0]                                 [GPU 1]
         W_q1, W_k1, W_v1                        W_q2, W_k2, W_v2
                 │                                       │
                 ▼                                       ▼
           Q1, K1, V1                              Q2, K2, V2
     (Heads 0 to h/2 - 1)                   (Heads h/2 to h - 1)
                 │                                       │
                 ▼ (Local Attention)                     ▼ (Local Attention)
              Attn_1                                  Attn_2
                 │                                       │
                 ▼ (Row-Parallel)                        ▼ (Row-Parallel)
             W_out_1                                 W_out_2
                 │                                       │
                 ▼                                       ▼
            Z1 [B, S, H]                            Z2 [B, S, H]
                 └───────────────────┬───────────────────┘
                                     ▼
                       [ ALL-REDUCE (SUM) via NVLink ]
                                     ▼
                     Final Attention Output Z [B, S, H]
```

1. **Column-Parallel Layer ($W_{qkv}$, $W_{gate}$, $W_{up}$):** Slices weight matrix along columns. Requires **zero communication** during forward execution.
2. **Row-Parallel Layer ($W_{out}$, $W_{down}$):** Slices weight matrix along rows. Requires **one All-Reduce (Sum)** to synchronize final activations.
3. **Communication Cost:** Exactly **two All-Reduces per Transformer layer** in the forward pass, and two in the backward pass.

---

### 2.2 Pipeline Parallelism (PP) & The 1F1B Schedule
When a model exceeds the memory of an 8-GPU node, Pipeline Parallelism divides the $L$ layers vertically across $p$ nodes.

#### The 1F1B (One Forward, One Backward) Breakthrough:
Instead of running all forward passes before backward passes (which causes activation memory to explode), Megatron-LM pairs one forward microbatch with one backward microbatch in steady-state:

```
1F1B Schedule (Steady State):
Stage 3:       [F1]    [F2]    [F3]    [F4][B1][B2][B3][B4]
Stage 2:     [F1]  [F2]    [F3]    [B1][F4][B2][B3][B4]
Stage 1:   [F1]  [F2]  [B1][F3][B2][F4][B3][B4]
Stage 0: [F1]  [F2][B1][F3][B2][F4][B3][B4]
         |<- Warmup ->|<- Steady State (1F1B) ->|<- Cooldown ->|
```

- **Activation Memory:** Strictly bounded by pipeline stages $O(p)$, independent of total microbatches $m$.

---

### 2.3 Sequence Parallelism (SP) and Context Parallelism (CP)

* **Sequence Parallelism (SP):** Partitions activation memory along the sequence dimension ($S / TP$) during LayerNorm and Dropout. By replacing the Row-Parallel All-Reduce with a **Reduce-Scatter** followed by an **All-Gather**, SP reduces LayerNorm activation memory by a factor of $TP$ with **zero added network communication**.
* **Context Parallelism (CP / Ring Attention):** For ultra-long contexts (32k to 1M+ tokens), queries remain stationary while keys and values are circulated across GPUs in a peer-to-peer ring topology using non-blocking RDMA transfers.

---

## 3. Hardware Interconnect Mapping

```
+-----------------------------------------------------------------------------------------+
|                                    8-GPU SERVER NODE                                    |
| ┌────────┐  NVLink (900 GB/s - 1.8 TB/s)  ┌────────┐     NVLink      ┌────────┐        |
| │ GPU 0  │◄══════════════════════════════►│ GPU 1  │◄════ ... ═════►│ GPU 7  │        |
| └───┬────┘                                └───┬────┘                 └───┬────┘        |
+─────┼─────────────────────────────────────────┼──────────────────────────┼──────────────+
      │ ConnectX-7 / ConnectX-8                 │ InfiniBand               │ InfiniBand
      ▼ (400 Gbps - 800 Gbps / 50-100 GB/s)     ▼ (High Latency ~2-5 µs)   ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
|                                 INTER-NODE INFINIBAND FABRIC                            |
|                               (Multi-Node Cluster Interconnect)                         |
+-----------------------------------------------------------------------------------------+
```

| Parallelism Dimension | Communication Frequency | Required Interconnect |
| :--- | :--- | :--- |
| **Tensor Parallelism (TP)** | Extremely High (2 All-Reduces per layer) | **NVLink Intra-Node Only** ($900\text{ GB/s}$) |
| **Sequence Parallelism (SP)** | Tied directly to TP (Reduce-Scatter / All-Gather) | **NVLink Intra-Node Only** |
| **Pipeline Parallelism (PP)** | Low (Only layer boundary activations) | **InfiniBand / RoCE Inter-Node** ($400\text{ Gbps}$) |
| **Data Parallelism (DP/ZeRO)**| Once per microbatch / iteration step | **InfiniBand / RoCE Inter-Node** |
