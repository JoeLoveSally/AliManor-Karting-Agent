"""Regression: frozen shadow replay must use CUDA for both inference paths.

These tests are CPU executable. The actual CUDA kernel parity is verified
separately by the original Spark frozen Validation metadata.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from karting_agent.train.shadow_replay import replay_fixed_video
from karting_agent.train.shadow_teacher import canonical_teacher_probabilities


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "audit_tiny_policy_shadow.py"


def test_strict_cuda_preflight_precedes_video_decoding():
    source = CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "main")
    assert main is not None
    assert source.index("device, device_budget = select_training_device(") < (
        source.index("frames=cache_required_frames(")
    )
    assert "'cuda', cuda_memory_fraction=args.cuda_memory_fraction" in source
    assert "allow_cpu_fallback=False" in source
    assert "if device.type != 'cuda':" in source


def test_both_frozen_teacher_and_self_fed_pass_selected_device():
    source = CLI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    teacher = [call for call in calls if isinstance(call.func, ast.Name)
               and call.func.id == "canonical_teacher_probabilities"]
    shadow = [call for call in calls if isinstance(call.func, ast.Name)
              and call.func.id == "replay_fixed_video"]
    assert len(teacher) == len(shadow) == 1
    for call in (*teacher, *shadow):
        device_kw = next(kw for kw in call.keywords if kw.arg == "device")
        assert isinstance(device_kw.value, ast.Name)
        assert device_kw.value.id == "device"
    assert "model = model.to(device).eval()" in source
    assert "compare_frozen_validation(item['training_metadata']" in source


def test_no_replacement_of_threshold_check_or_test_data_exposure():
    source = CLI.read_text(encoding="utf-8")
    assert "threshold=.5,tolerance_ms=100." in source
    assert "['validation']" in source
    assert "['test']" not in source
    assert "optimizer.step(" not in source
    assert "allow_cpu_fallback=True" not in source


class DeviceSpy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.received = []

    def forward(self, images, controls):
        assert images.device == self.weight.device
        assert controls.device == self.weight.device
        self.received.append((str(images.device), len(images)))
        return torch.zeros((len(images),), device=images.device) + self.weight


def _expert_features(stamps, events):
    return np.zeros((5, 3), np.float32)


def _sample(i):
    t = 200.0 + i * 50.0
    return SimpleNamespace(
        video="validation.mp4",
        input_timestamps_ms=(t - 200, t - 150, t - 100, t - 50, t),
        target_timestamp_ms=t + 100,
        current_pressed=False,
        target_pressed=False,
    )


def test_explicit_device_is_used_by_shadow_forward():
    model = DeviceSpy()
    rows = [_sample(0), _sample(1)]
    teacher = {
        (s.video, s.target_timestamp_ms): .5 for s in rows
    }
    replay = replay_fixed_video(
        model, rows,
        frames_for_sample=lambda _: np.zeros((5, 3, 16, 16), np.float32),
        expert_events=(SimpleNamespace(timestamp_ms=0., pressed=False),),
        expert_features=_expert_features,
        teacher_probabilities=teacher,
        device=torch.device("cpu"),
    )
    assert len(replay["rows"]) == 2
    assert model.received == [("cpu", 1), ("cpu", 1)]
    assert all(row["teacher_probability"] == .5 for row in replay["rows"])


def test_canonical_teacher_device_routing_preserves_original_batching():
    class TeacherDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = [SimpleNamespace(
                video="validation.mp4", target_timestamp_ms=float(i)
            ) for i in range(35)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            return (
                torch.zeros((5, 3, 16, 16)),
                torch.zeros((5, 3)),
                torch.tensor(0., dtype=torch.float32),
            )

    model = DeviceSpy()
    report = canonical_teacher_probabilities(
        model, TeacherDataset(), batch_size=16, device=torch.device("cpu")
    )
    assert len(report) == 35
    assert model.received == [("cpu", 16), ("cpu", 16), ("cpu", 3)]


def test_invalid_device_is_not_silently_corrected():
    model = DeviceSpy()
    if torch.cuda.is_available():
        pytest.skip("CPU-only invalid-device assertion")
    with pytest.raises((RuntimeError, AssertionError)):
        replay_fixed_video(
            model, [_sample(0)],
            frames_for_sample=lambda _: np.zeros((5, 3, 16, 16), np.float32),
            expert_events=(SimpleNamespace(timestamp_ms=0., pressed=False),),
            expert_features=_expert_features,
            device="cuda",
        )
