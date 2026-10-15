#!/usr/bin/env python3
"""
Lab 01: Vector Addition & Thread Indexing Decomposition.

Demonstrates:
1. 1D Thread/Block/Grid linear index calculation:
   i = blockIdx.x * blockDim.x + threadIdx.x
2. Boundary protection against non-multiple array sizes:
   if (i < n) c[i] = a[i] + b[i]
3. Dual-mode execution: Native PyTorch CUDA kernel on NVIDIA GPU,
   or algorithmic thread-block decomposition simulation on CPU.
"""

import time
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def simulate_gpu_grid(n: int, block_size: int = 256):
    """
    Simulates the GPU warp/thread scheduling logic on CPU.
    Deconstructs gridDim, blockDim, blockIdx, threadIdx.
    """
    grid_size = (n + block_size - 1) // block_size
    print(f"\n[Hardware Execution Configuration]")
    print(f"  * Total Elements (N)  : {n:,}")
    print(f"  * Threads per Block   : {block_size}")
    print(f"  * Number of Blocks    : {grid_size}")
    print(f"  * Total Warps per Blk : {block_size // 32}")
    print(f"  * Total Allocated Thrs: {grid_size * block_size:,}")

    a = np.ones(n, dtype=np.float32) * 1.5
    b = np.ones(n, dtype=np.float32) * 2.5
    c_sim = np.zeros(n, dtype=np.float32)

    # Simulation loop modeling GPU block dispatch
    for block_idx in range(grid_size):
        for thread_idx in range(block_size):
            global_idx = block_idx * block_size + thread_idx
            # Boundary guard (crucial in every CUDA kernel)
            if global_idx < n:
                c_sim[global_idx] = a[global_idx] + b[global_idx]

    # Verify against numpy reference
    c_ref = a + b
    assert np.allclose(c_sim, c_ref), "Simulation failed numerical verification!"
    print(f"  * Verification Status : PASSED (c[0..3] = {c_sim[:4]})")
    return c_sim


def run_pytorch_cuda_or_cpu(n: int = 10_000_000):
    if not HAS_TORCH:
        print("[!] PyTorch not found. Running NumPy simulation only.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[PyTorch Execution Mode: {device.type.upper()}]")

    a = torch.ones(n, dtype=torch.float32, device=device) * 1.5
    b = torch.ones(n, dtype=torch.float32, device=device) * 2.5

    # Warm-up
    _ = a + b
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    num_runs = 20
    for _ in range(num_runs):
        c = a + b
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_runs

    # Bandwidth formula: Read A (4 bytes) + Read B (4 bytes) + Write C (4 bytes) = 12 bytes / element
    bytes_transferred = 3 * n * 4
    bandwidth_gb_s = (bytes_transferred / elapsed) / 1e9

    print(f"  * Array Size          : {n:,} elements ({bytes_transferred / (1024**2):.1f} MB memory traffic)")
    print(f"  * Avg Kernel Time     : {elapsed * 1000:.3f} ms")
    print(f"  * Effective Bandwidth : {bandwidth_gb_s:.2f} GB/s")


def main():
    print("=" * 65)
    print("  LAB 01: Vector Addition & Thread Indexing")
    print("=" * 65)

    simulate_gpu_grid(n=1_000_000, block_size=256)
    run_pytorch_cuda_or_cpu(n=10_000_000)

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
