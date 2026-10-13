"""CPU-only mock tests for shared-GPU memory preflight and fallback."""
from unittest.mock import patch

import pytest
import torch

from karting_agent.train import gpu_budget as budget


def test_explicit_cpu_does_not_initialize_nvidia_or_cuda():
    with patch.object(
        budget, "nvidia_memory_mib",
        side_effect=AssertionError("NVML queried in CPU mode"),
    ):
        device, info = budget.select_training_device("cpu")
    assert device.type == "cpu"
    assert info["reason"] == "explicit_cpu"


def test_low_gpu_free_memory_auto_falls_back_before_cuda_init():
    with patch.object(budget, "nvidia_memory_mib", return_value=(700, 128000)):
        with patch.object(
            torch.cuda, "is_available",
            side_effect=AssertionError("CUDA initialized despite low memory"),
        ):
            device, info = budget.select_training_device(
                "auto", min_cuda_free_mib=2048
            )
    assert device.type == "cpu"
    assert info["gpu_free_mib_before"] == 700


def test_explicit_cuda_fail_fast_or_opt_in_cpu_fallback():
    with patch.object(budget, "nvidia_memory_mib", return_value=(700, 128000)):
        with pytest.raises(RuntimeError, match="--allow-cpu-fallback"):
            budget.select_training_device("cuda")
        device, info = budget.select_training_device(
            "cuda", allow_cpu_fallback=True
        )
    assert device.type == "cpu"
    assert info["reason"] == "cuda_free_memory_below_floor"


def test_budget_is_applied_before_cuda_context():
    calls = []
    with patch.object(budget, "nvidia_memory_mib", return_value=(5000, 128000)):
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(
                torch.cuda, "set_per_process_memory_fraction",
                side_effect=lambda *a, **k: calls.append("budget")
            ):
                with patch.object(
                    torch, "empty",
                    side_effect=lambda *a, **k: calls.append("context")
                ):
                    device, info = budget.select_training_device(
                        "auto", cuda_memory_fraction=0.02,
                    )
    assert device.type == "cuda"
    assert calls == ["budget", "context"]
    assert info["cuda_memory_fraction"] == 0.02


def test_cuda_context_oom_auto_falls_back():
    with patch.object(budget, "nvidia_memory_mib", return_value=(4500, 128000)):
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(
                torch.cuda, "set_per_process_memory_fraction",
                side_effect=torch.AcceleratorError("CUDA out of memory"),
            ):
                device, info = budget.select_training_device("auto")
    assert device.type == "cpu"
    assert info["reason"] == "cuda_initialization_failed"


@pytest.mark.parametrize(
    "kwargs", [
        {"cuda_memory_fraction": 0},
        {"cuda_memory_fraction": 1.1},
        {"min_cuda_free_mib": -1},
        {"requested": "mps"},
    ],
)
def test_rejects_invalid_budget_arguments(kwargs):
    with pytest.raises(ValueError):
        budget.select_training_device(**kwargs)


def test_nvidia_smi_nvml_probe_parsing():
    from types import SimpleNamespace
    with patch.object(
        budget.subprocess, "run",
        return_value=SimpleNamespace(stdout="4096, 128000\n")
    ):
        assert budget.nvidia_memory_mib() == (4096, 128000)
