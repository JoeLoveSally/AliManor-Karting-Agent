"""Cooperative CUDA budget selection for offline tiny-policy experiments.

The allocator fraction is a PyTorch allocation limit, not a reservation or
isolation mechanism. The NVML preflight is best-effort and avoids initializing
CUDA when a co-resident inference service occupies most of the GPU.
"""
from __future__ import annotations

import subprocess

import torch


def nvidia_memory_mib() -> tuple[int, int] | None:
    """Best-effort free/total MiB without creating a CUDA context."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        line = result.stdout.strip().splitlines()[0]
        free, total = (int(item.strip()) for item in line.split(","))
        if 0 <= free <= total and total > 0:
            return free, total
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return None


def select_training_device(
    requested: str = "auto", *, cuda_memory_fraction: float = 0.03,
    min_cuda_free_mib: int = 2048, allow_cpu_fallback: bool = False,
) -> tuple[torch.device, dict]:
    """Check CUDA usability before expensive video decoding.

    Explicit CUDA remains fail-fast unless CPU fallback is opted into.
    Auto always falls back to CPU when CUDA is unavailable or overloaded.
    """
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda or auto")
    if not 0.0 < cuda_memory_fraction <= 1.0:
        raise ValueError("cuda_memory_fraction must be in (0,1]")
    if min_cuda_free_mib < 0:
        raise ValueError("min_cuda_free_mib must be nonnegative")

    meta = {
        "requested_device": requested,
        "cuda_memory_fraction": cuda_memory_fraction,
        "min_cuda_free_mib": min_cuda_free_mib,
        "cpu_fallback_allowed": requested == "auto" or allow_cpu_fallback,
        "gpu_free_mib_before": None,
        "gpu_total_mib": None,
    }
    if requested == "cpu":
        return torch.device("cpu"), {**meta, "reason": "explicit_cpu"}

    can_fallback = requested == "auto" or allow_cpu_fallback

    def decline(reason: str, *, cause: Exception | None = None):
        if can_fallback:
            return torch.device("cpu"), {**meta, "reason": reason}
        raise RuntimeError(
            f"CUDA preflight failed ({reason}); use --device cpu or "
            "--allow-cpu-fallback to leave vLLM undisturbed"
        ) from cause

    memory = nvidia_memory_mib()
    if memory is not None:
        free, total = memory
        meta["gpu_free_mib_before"] = free
        meta["gpu_total_mib"] = total
        if free < min_cuda_free_mib:
            return decline("cuda_free_memory_below_floor")

    try:
        if not torch.cuda.is_available():
            return decline("cuda_not_available")
        torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction, device=0)
        # Test context creation now, not after decoding thirteen large videos.
        torch.empty((1,), device="cuda:0")
        return torch.device("cuda:0"), {**meta, "reason": "cuda_preflight_ok"}
    except RuntimeError as exc:
        return decline("cuda_initialization_failed", cause=exc)
