#!/usr/bin/env python3
"""
Diagnostic Environment & Hardware Inspector for CUDA, TensorRT, and vLLM.
Performs an audit of local host compute capabilities, available drivers,
compiler toolchains, and detects whether execution is running in Local Fallback Mode
or Target GPU Cloud/Container Mode.
"""

import os
import platform
import subprocess
import sys
from typing import Dict, Any


def run_cmd(cmd: list[str]) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "Not available"


def check_python_environment() -> Dict[str, Any]:
    info = {
        "Python Version": sys.version.split()[0],
        "Platform": platform.platform(),
        "Architecture": platform.machine(),
        "Executable": sys.executable,
    }
    return info


def check_cuda_toolchain() -> Dict[str, Any]:
    info = {}
    # nvcc
    nvcc_out = run_cmd(["nvcc", "--version"])
    if "release" in nvcc_out:
        for line in nvcc_out.splitlines():
            if "release" in line:
                info["NVCC Version"] = line.strip()
    else:
        info["NVCC Version"] = "Not found (CPU host / no CUDA toolkit)"

    # nvidia-smi
    smi_out = run_cmd(["nvidia-smi"])
    if "NVIDIA-SMI" in smi_out:
        info["nvidia-smi"] = "Active"
    else:
        info["nvidia-smi"] = "Not found (No discrete NVIDIA GPU driver detected)"

    return info


def check_frameworks() -> Dict[str, Any]:
    info = {}
    # PyTorch
    try:
        import torch
        info["PyTorch Installed"] = torch.__version__
        info["CUDA Available in Torch"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["Device Name"] = torch.cuda.get_device_name(0)
            info["Device Count"] = torch.cuda.device_count()
            info["Compute Capability"] = torch.cuda.get_device_capability(0)
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["Total VRAM (GB)"] = f"{total_mem_gb:.2f} GB"
        else:
            info["Device Mode"] = "CPU / Fallback Emulation Mode"
    except ImportError:
        info["PyTorch Installed"] = "Not installed yet"

    # TensorRT
    try:
        import tensorrt as trt
        info["TensorRT Installed"] = trt.__version__
    except ImportError:
        info["TensorRT Installed"] = "Not installed (Requires CUDA target)"

    # vLLM
    try:
        import vllm
        info["vLLM Installed"] = vllm.__version__
    except ImportError:
        info["vLLM Installed"] = "Not installed (Requires CUDA target)"

    return info


def main():
    print("=" * 70)
    print("  CUDA, TensorRT, vLLM Testbed - Environment & Hardware Diagnostic")
    print("=" * 70)

    print("\n[1] Python & Operating System:")
    for k, v in check_python_environment().items():
        print(f"  * {k:25}: {v}")

    print("\n[2] CUDA Toolchain & Hardware Driver:")
    for k, v in check_cuda_toolchain().items():
        print(f"  * {k:25}: {v}")

    print("\n[3] Deep Learning & Inference Frameworks:")
    for k, v in check_frameworks().items():
        print(f"  * {k:25}: {v}")

    print("\n" + "=" * 70)
    # Summary recommendation
    smi_present = run_cmd(["which", "nvidia-smi"]) != "Not available"
    if smi_present:
        print("  STATUS: Target NVIDIA GPU detected. Native kernel execution enabled.")
    else:
        print("  STATUS: Host running in Local Mode (CPU / Simulation Fallback).")
        print("          All lab exercises will execute algorithmic verification,")
        print("          profiling models, and graph exports using CPU/emulated backends,")
        print("          with production GPU code paths ready for remote container deployment.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
