#!/usr/bin/env python3
"""
Lab 01: Memory Coalescing Benchmark.

Demonstrates the effect of memory access stride on effective memory bandwidth.
On GPUs:
- Stride 1: 32 threads in a warp access contiguous addresses -> Single 128-byte transaction.
- Stride 2..32: Addresses diverge across multiple 32-byte sectors -> Multiple transactions,
  wasting memory bus bandwidth and inducing memory pipeline stalls.
On CPUs:
- Demonstrates cache-line utilization (64-byte CPU cache lines).
"""

import time
import numpy as np
from tabulate import tabulate

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def bench_stride_cpu(n: int = 16_000_000, strides: list[int] = [1, 2, 4, 8, 16, 32]):
    print(f"\n[Benchmarking Memory Access Stride Impact (N = {n:,} floats)]")
    data = np.ones(n * max(strides), dtype=np.float32)

    results = []
    base_time = None

    for s in strides:
        indices = np.arange(0, n * s, s, dtype=np.int64)

        # Warmup
        _ = np.sum(data[indices[:1000]])

        # Measure read bandwidth
        start = time.perf_counter()
        _ = np.sum(data[indices])
        duration = time.perf_counter() - start

        if base_time is None:
            base_time = duration

        slowdown = duration / base_time
        bytes_requested = n * 4
        effective_gb_s = (bytes_requested / duration) / 1e9

        results.append([
            f"Stride {s}",
            f"{duration * 1000:.2f} ms",
            f"{effective_gb_s:.2f} GB/s",
            f"{slowdown:.2f}x"
        ])

    headers = ["Access Pattern", "Execution Time", "Effective Bandwidth", "Relative Penalty"]
    print(tabulate(results, headers=headers, tablefmt="fancy_grid"))
    print("\n[Key Takeaway]")
    print("  * As stride increases, memory controller transaction efficiency plummets.")
    print("  * On GPUs, Stride 32 causes a single 32-thread warp to issue 32 distinct")
    print("    memory transactions instead of 1, cutting bandwidth efficiency to ~3-12%.")


def main():
    print("=" * 65)
    print("  LAB 01: Memory Coalescing & Stride Benchmark")
    print("=" * 65)
    bench_stride_cpu(n=4_000_000)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
