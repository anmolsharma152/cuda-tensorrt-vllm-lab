#!/usr/bin/env bash
# Unified lab runner script
set -euo pipefail

LAB="${1:-lab01}"

echo "=========================================================="
echo " Running Lab: ${LAB}"
echo "=========================================================="

case "${LAB}" in
    lab01|1)
        echo "--> [1/3] Running Vector Add & Grid Decomposition..."
        uv run python labs/lab01_cuda_foundations/vector_add.py
        echo ""
        echo "--> [2/3] Running Memory Coalescing Benchmark..."
        uv run python labs/lab01_cuda_foundations/coalescing_bench.py
        echo ""
        echo "--> [3/3] Running Automated Verification Test Suite..."
        uv run pytest labs/lab01_cuda_foundations/verify.py -v
        ;;
    env|check)
        uv run python scripts/check_env.py
        ;;
    *)
        echo "Unknown lab: ${LAB}"
        echo "Usage: ./scripts/run_lab.sh [env|lab01]"
        exit 1
        ;;
esac

echo "=========================================================="
echo " Finished Lab: ${LAB} successfully!"
echo "=========================================================="
