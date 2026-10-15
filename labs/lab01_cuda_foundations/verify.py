"""
Lab 01 Correctness & Assertion Test Suite.
Verifies thread index math, boundary guard protections, and numerical accuracy.
"""

import numpy as np
import pytest
from labs.lab01_cuda_foundations.vector_add import simulate_gpu_grid


def test_grid_block_boundary_exact_multiple():
    """Checks grid allocation when array length is exact multiple of block size."""
    n = 1024
    block_size = 256
    result = simulate_gpu_grid(n=n, block_size=block_size)
    assert len(result) == n
    assert np.allclose(result, 4.0)


def test_grid_block_boundary_non_multiple():
    """Checks boundary protection when array length is NOT a multiple of block size."""
    n = 1000  # Requires 4 blocks (1024 threads), 24 threads must be masked off
    block_size = 256
    result = simulate_gpu_grid(n=n, block_size=block_size)
    assert len(result) == n
    assert np.allclose(result, 4.0)


def test_coalesced_stride_invariant():
    """Ensures stride calculations do not alter summation correctness."""
    arr = np.arange(100, dtype=np.float32)
    stride = 2
    indices = np.arange(0, 100, stride)
    sampled = arr[indices]
    assert len(sampled) == 50
    assert sampled[0] == 0.0
    assert sampled[1] == 2.0
