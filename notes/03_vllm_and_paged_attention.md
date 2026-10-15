# Technical Guide: vLLM, PagedAttention, and High-Throughput Serving

---

## 1. The KV Cache Memory Problem

In autoregressive generation, past tokens must be preserved in GPU High Bandwidth Memory (HBM) to compute self-attention:

$$\text{Memory}_{KV} = 2 \times L \times H_{kv} \times d_{head} \times b \times S \times B$$

Where:
- $L$: Number of hidden layers
- $H_{kv}$: Number of Key-Value attention heads (reduced in Grouped-Query Attention / GQA)
- $d_{head}$: Hidden dimension per head
- $b$: Precision bytes (FP16 = 2, FP8 = 1)
- $S$: Total sequence length
- $B$: Batch size (concurrent active sequences)

### The Fragmentation Crisis
In traditional serving (pre-PagedAttention), systems pre-allocated contiguous memory up to the request's requested maximum context length $N_{max}$ (e.g., 4096 tokens):
1. **Internal Fragmentation:** If the request stopped after 150 tokens, the remaining 3946 slots (~96% of allocated memory) were reserved and wasted.
2. **External Fragmentation:** Dynamic requests of varying lengths created scattered memory holes, making contiguous allocation impossible even when total free VRAM was high.
3. **Total Waste:** Traditional serving utilized only **20% to 40%** of available GPU memory for active tokens.

---

## 2. PagedAttention Mechanics (Kwon et al., SOSP 2023)

PagedAttention eliminates contiguous allocation requirements by treating GPU VRAM like operating system virtual memory:

```
Logical KV Cache (Sequence A)
+---------------+---------------+---------------+
| Logical Blk 0 | Logical Blk 1 | Logical Blk 2 |
| Tokens 0 - 15 | Tokens 16 - 31| Tokens 32 - 47|
+-------+-------+-------+-------+-------+-------+
        │               │               │
        ▼               ▼               ▼
+---------------+---------------+---------------+
| Physical Blk 7| Physical Blk 2| Physical Blk 9|  <-- Physical HBM Pages
+---------------+---------------+---------------+      (Non-contiguous)
```

1. **Physical Blocks:** VRAM is divided into fixed-size physical blocks (pages), typically sized to 16 or 32 tokens.
2. **Logical Block Table:** A mapping table translates contiguous logical sequence tokens to non-contiguous physical VRAM blocks.
3. **Zero External Fragmentation:** Every physical block is identical in size; any free page can satisfy any sequence's allocation request.
4. **Near-Zero Internal Fragmentation:** Waste is strictly bounded to the final unfilled block of each sequence ($<4\%$ average waste).
5. **Copy-on-Write (CoW) Forking:** Parallel sampling ($n > 1$) and beam search share physical blocks until a sequence generates a unique continuation token, slashing memory consumption by up to 80%.

---

## 3. Batching Paradigms

```
Static Batching:
Seq 1: [Prefill][Decode][Decode][Decode]----------------- (Idle / Padded)
Seq 2: [Prefill][Decode][Decode][Decode][Decode][Decode]- (Idle / Padded)
Seq 3: [Prefill][Decode][Decode]------------------------- (Idle / Padded)
                                                         ^ All wait for Seq 2

Continuous (Iteration-Level) Batching:
Iter 1: [Seq 1 Dec][Seq 2 Dec][Seq 3 Dec]
Iter 2: [Seq 1 Dec][Seq 2 Dec][Seq 4 Prefill Chunk]  <-- Seq 3 finished; Seq 4 injected
Iter 3: [Seq 1 Dec][Seq 2 Dec][Seq 4 Dec]
```

- **Continuous (Iteration-Level) Batching:** The scheduler operates per token iteration rather than per request. As soon as a request emits an `EOS` token, its physical memory is recycled immediately, and waiting prompts are scheduled on the next iteration.

---

## 4. Chunked Prefill & Latency Smoothing

```
+------------------------------------+--------------------------------------------------+
| Prefill Phase                      | Decode Phase                                     |
+------------------------------------+--------------------------------------------------+
| - Processes entire input prompt    | - Generates 1 token per sequence per step        |
| - Compute-Bound (TFLOPs limited)   | - Memory-Bandwidth-Bound (HBM GB/s limited)      |
| - High Arithmetic Intensity        | - Low Arithmetic Intensity (~0.5 - 2 FLOPs/byte) |
| - Dictates TTFT                    | - Dictates TPOT / Inter-Token Latency            |
+------------------------------------+--------------------------------------------------+
```

### The Problem:
A large prompt prefill (e.g., 16k tokens) monopolizes the GPU for hundreds of milliseconds, stalling all ongoing decodes and causing severe **Time-Per-Output-Token (TPOT)** spikes.

### The Solution: Chunked Prefill (`--enable-chunked-prefill`)
Slices long prompts into fixed chunks (e.g., 512 tokens) and co-schedules them with ongoing decode steps.
- The compute-heavy prefill chunk saturates idle Tensor Cores without exceeding acceptable per-iteration execution boundaries.
- Delivers smooth, predictable Inter-Token Latency ($P_{99}$).
