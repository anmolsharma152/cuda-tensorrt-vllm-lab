# Technical Guide: TensorRT Compilation, Layer Fusion & Quantization

---

## 1. The TensorRT Compilation Pipeline

NVIDIA TensorRT is an optimizing ahead-of-time (AOT) compiler that converts static and dynamic neural network graphs into serialized execution plans (`.plan` or `.engine`).

```
[PyTorch / HuggingFace]
          │
          ▼  (torch.onnx.export / torch.export)
    [ONNX Graph]
          │
          ▼
+─────────────────────────────────────────────────────────────+
|               TENSORRT OPTIMIZATION PHASES                  |
| 1. Graph Parsing & Algebraic Simplification (Constant fold) |
| 2. Horizontal & Vertical Layer Fusion                       |
| 3. Format & Layout Optimization (NCHW -> NHWC / NC32HW32)   |
| 4. Kernel Selection & Architecture Auto-Tuning (SM-locked)  |
| 5. Zero-Copy Static Memory Planning & Activation Offsetting |
+─────────────────────────────────────────────────────────────+
          │
          ▼
   [Serialized Engine (.plan)]
          │
          ▼
[Runtime Context (IExecutionContext)] ──> Non-blocking enqueue_v3 on CUDA Stream
```

---

## 2. Layer & Tensor Fusion Mechanics

### 1. Vertical Fusion
Combines sequential operations into a single kernel to eliminate round-trips to High Bandwidth Memory (HBM):
- **CBR Fusion:** $\text{Conv2D} \rightarrow \text{BatchNorm} \rightarrow \text{ReLU}$ merged into one fused kernel.
- **Attention Fusion (FMHA):** Merges $Q \times K^T$, softmax scaling, causal masking, and multiplication by $V$ into a single fused kernel kept in GPU SRAM (Shared Memory).
- **RMSNorm + Residual:** Combines elementwise addition with root-mean-square normalization.

### 2. Horizontal Fusion
Combines parallel layers that share identical inputs:
- Merging separate Query, Key, and Value linear projections ($W_q, W_k, W_v$) into a single unified wide GEMM ($W_{qkv}$).

---

## 3. Dynamic Shapes & Optimization Profiles

In real-world deployment, batch sizes and sequence lengths vary dynamically. TensorRT manages dynamic axes using **Optimization Profiles**:
- For every dynamic dimension (represented by `-1`), you define:
  * `minShapes`: Smallest allowable dimension (e.g., `1x1x224x224`).
  * `optShapes`: The expected runtime dimension mode/median (e.g., `16x3x224x224`).
  * `maxShapes`: Largest allowable dimension (e.g., `64x3x224x224`).

> [!WARNING]
> **Tactic Mismatch Penalty:** TensorRT benchmarks and auto-tunes kernel tile sizes specifically targeting `optShapes`. If production inputs deviate significantly from `optShapes`, throughput can degrade by up to 40% due to suboptimal thread block sizing.

---

## 4. Precision & Quantization Paradigms

```
FP32 (32-bit): [Sign: 1] [Exponent: 8] [Mantissa: 23]  - Baseline
FP16 (16-bit): [Sign: 1] [Exponent: 5] [Mantissa: 10]  - 2x throughput, 50% memory
FP8  ( 8-bit): [Sign: 1] [Exponent: 4] [Mantissa:  3] (E4M3 for weights/activations)
INT8 ( 8-bit): Symmetric Integer [-128, 127] with dynamic scaling factor S
```

### Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)
- **PTQ:** Weights and activations are quantized without retraining. Because activation distributions vary dynamically, PTQ requires a representative **calibration dataset** (~500 batches). Calibrators (`IInt8EntropyCalibrator2`, `IInt8MinMaxCalibrator`) measure activation histograms to calculate scaling factor $S$:
  $$\text{Tensor}_{\text{int8}} = \text{clip}\left(\text{round}\left(\frac{X}{S}\right), -128, 127\right)$$
- **QAT:** Quantization error is modeled during training using PyTorch Q/DQ (`QuantizeLinear` / `DequantizeLinear`) nodes. TensorRT fuses these nodes directly into INT8 Tensor Core operations without calibration.

---

## 5. TensorRT-LLM and Modern Serving

Standard TensorRT processes single-pass forward graphs. **TensorRT-LLM** extends TensorRT with LLM-native execution primitives:
1. **In-Flight Batching:** Iteration-level scheduling where requests enter and leave the batch dynamically at each generated token.
2. **Paged KV Caching:** Non-contiguous block memory allocation for Key-Value activations.
3. **Multi-GPU Parallelism:** Native Megatron-style column/row Tensor Parallelism (TP) with fused NCCL All-Reduce kernels.
4. **C++ Executor API:** High-concurrency asynchronous runtime handling streaming outputs without Python GIL overhead.
