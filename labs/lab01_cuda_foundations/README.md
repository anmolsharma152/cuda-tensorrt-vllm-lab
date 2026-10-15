# Lab 01: Hardware Hierarchy, Warp Execution & Memory Coalescing

## Overview
This lab grounds you in the fundamental execution model of NVIDIA GPUs:
1. **Thread Hierarchies & Index Arithmetic:** Mapping 1D/2D grid and block dimensions to linear memory coordinates.
2. **Memory Coalescing:** Understanding how the warp memory controller services global memory requests in 32-byte or 128-byte transactions.
3. **Stride Penalty Analysis:** Measuring effective memory bandwidth collapse when memory accesses are non-contiguous.

---

## Files in this Lab

* [`vector_add.py`](file:///labs/lab01_cuda_foundations/vector_add.py):
  Implements vector addition using explicit thread/block indexing. Includes dual-mode dispatch: native CUDA kernel via PyTorch C++/inline/CUDA runtime if GPU available, and an algorithmic simulation showing grid/block dispatch math on CPU.
* [`coalescing_bench.py`](file:///labs/lab01_cuda_foundations/coalescing_bench.py):
  Benchmarks memory access performance across varying strides ($1, 2, 4, 8, 16, 32$). Proves how uncoalesced memory access degrades throughput by up to $10\times$.
* [`verify.py`](file:///labs/lab01_cuda_foundations/verify.py):
  Assertion and correctness test suite validating mathematical correctness and memory alignment rules.

---

## Quickstart & Verification Commands

```bash
# 1. Run environment diagnostic
uv run python scripts/check_env.py

# 2. Run vector addition with thread index inspection
uv run python labs/lab01_cuda_foundations/vector_add.py

# 3. Run memory coalescing benchmark
uv run python labs/lab01_cuda_foundations/coalescing_bench.py

# 4. Run automated unit assertions
uv run pytest labs/lab01_cuda_foundations/verify.py -v
```
