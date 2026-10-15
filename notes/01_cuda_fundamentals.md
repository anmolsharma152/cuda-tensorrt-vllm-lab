# Technical Guide: CUDA Architecture, Thread Hierarchy, and Memory Coalescing

---

## 1. Streaming Multiprocessor (SM) Microarchitecture

An NVIDIA GPU is an array of Streaming Multiprocessors (SMs) connected to a shared L2 cache and High Bandwidth Memory (HBM/GDDR).

```
+-----------------------------------------------------------------------------------+
|                                 GPU Device                                        |
|  +-----------------------------------------------------------------------------+  |
|  |                             L2 Cache                                        |  |
|  +-----------------------------------------------------------------------------+  |
|  |    SM 0                      SM 1                      SM N                 |  |
|  | +-------------------------+ +-------------------------+ +-----------------+ |  |
|  | | Warp Scheduler 0..3     | | Warp Scheduler 0..3     | |                 | |  |
|  | | Register File (64K regs)| | Register File (64K regs)| |                 | |  |
|  | | FP32 / INT32 / FP64 Core| | FP32 / INT32 / FP64 Core| |      ...        | |  |
|  | | Tensor Cores (4th/5th G)| | Tensor Cores (4th/5th G)| |                 | |  |
|  | | Shared Memory / L1 SRAM | | Shared Memory / L1 SRAM | |                 | |  |
|  | +-------------------------+ +-------------------------+ +-----------------+ |  |
|  +-----------------------------------------------------------------------------+  |
|  +-----------------------------------------------------------------------------+  |
|  |                        HBM3e / GDDR6X Memory                                |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### Key Components of an SM:
1. **Sub-Cores (Processing Blocks):** Each SM is partitioned into 4 sub-cores. Each sub-core operates independently with its own Warp Scheduler, Instruction Dispatch Unit, and a 64 KB slice of the 256 KB Register File.
2. **Warps (32 Threads):** The fundamental unit of execution. 32 contiguous logical threads execute the same instruction simultaneously in Single Instruction, Multiple Threads (SIMT) mode.
3. **Latency Hiding via Occupancy:** GPUs lack complex branch predictors and out-of-order execution pipelines. Instead, they hide instruction latency (e.g., 200–400 cycles for global memory reads) by immediately switching execution context to another active warp on the next clock cycle with zero overhead.

---

## 2. Memory Hierarchy & Latency Profiles

| Memory Tier | Scope | Latency | Bandwidth (H100) | Sizing |
| :--- | :--- | :--- | :--- | :--- |
| **Registers** | Per Thread | ~1 cycle | ~30 TB/s (aggregate) | 64K 32-bit registers / SM (256 KB) |
| **Shared Memory / L1** | Per Block | ~20–30 cycles | ~15 TB/s (aggregate) | Up to 228 KB / SM |
| **L2 Cache** | Device-wide | ~150–200 cycles | ~5.5 TB/s | 50 MB (H100) to 128 MB (B200) |
| **Global Memory (HBM3e)**| Device-wide | ~300–500 cycles | 3.35 TB/s | 80 GB to 192 GB |
| **Constant Memory** | Device-wide | 1 cycle (on broadcast)| Broadcast throughput | 64 KB total (cached in L1) |

### Register Pressure & Spilling
If a kernel uses too many registers per thread (e.g., >64 registers on a high-thread block), the compiler is forced to spill excess registers to **Local Memory**.
Local Memory physically resides in off-chip DRAM (Global Memory), causing severe throughput degradation. Compilers allow controlling register limits via `--maxrregcount=N`.

---

## 3. Memory Coalescing Mechanics

Global memory is read in discrete hardware transactions of **32 bytes** (L2 sector) or **128 bytes** (L1 cache line).

### Coalescing Scenarios (for 32-bit `float` elements, Warp of 32 threads = 128 bytes total):

1. **Fully Coalesced (Stride = 1):**
   ```
   Warp Threads:  T0  T1  T2  T3 ... T31
   Global Addr:  [00  04  08  12 ... 124]  --> Exactly ONE 128-byte aligned transaction.
   Bus Efficiency: 100%
   ```
2. **Strided Access (Stride = 2):**
   ```
   Warp Threads:  T0      T1      T2      T3 ... T15
   Global Addr:  [00  --  08  --  16  --  24 ... 120] --> TWO 128-byte transactions needed.
   Bus Efficiency: 50%
   ```
3. **Severe Strided Access (Stride = 32):**
   ```
   Warp Threads:  T0 -> Addr 0, T1 -> Addr 128, T2 -> Addr 256 ...
   --> 32 SEPARATE 32-byte transactions (1024 bytes transferred for 128 bytes needed).
   Bus Efficiency: 12.5% (or 3.125% un-cached).
   ```

### Vectorized Memory Instructions (`float4`)
By casting contiguous pointers to `float4` (or 16-byte packed structs), NVCC emits `LDG.E.128` instructions:
- Each thread loads 4 floats (16 bytes) in a single instruction.
- Halves or quarters the total instructions issued to the warp scheduler.

---

## 4. Shared Memory Bank Conflicts

Shared Memory is organized into **32 independent banks** of 4-byte width:

$$\text{Bank Index} = \left(\frac{\text{Byte Address}}{4}\right) \pmod{32}$$

- **Conflict-Free:** 32 threads in a warp access 32 distinct banks $\rightarrow$ Serviced in **1 cycle**.
- **Broadcast:** Multiple threads read the *exact same address* in a bank $\rightarrow$ Serviced in **1 cycle** via broadcast.
- **$K$-Way Conflict:** $K$ threads access *different addresses within the same bank* $\rightarrow$ Access is serialized $K$ times (up to $32\times$ slower).

### Mitigation via Padding
For a 2D tile `float tile[32][32]`:
- Reading column-wise causes all 32 threads to hit the same bank (32-way bank conflict).
- Adding a single dummy column `float tile[32][33]` offsets subsequent rows by 1 bank, eliminating conflicts completely:
  $$\text{Bank} = (i \times 33 + x) \pmod{32} = (i + x) \pmod{32}$$

---

## 5. Warp Shuffle Primitives (`__shfl_down_sync`)

Warp shuffle instructions permit threads in a warp to exchange data directly through **register-to-register pathways**, bypassing shared memory and thread barrier synchronizations:

```cpp
// 32-thread Butterfly Reduction in Registers
__device__ inline float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val; // Lane 0 contains the reduction sum
}
```
Latency: **1 clock cycle**, with 0 shared memory overhead.
