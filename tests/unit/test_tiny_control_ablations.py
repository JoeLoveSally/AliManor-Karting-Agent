"""CPU-only contract tests for matched Tiny RGB control-input ablations."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from karting_agent.model.tiny_control_ablation import (  # noqa: E402
    CONTROL_MASKS, TinyControlAblation,
)


def test_all_variants_have_same_trainable_weights():
    states = [TinyControlAblation(name).state_dict() for name in CONTROL_MASKS]
    assert set(states[0]) == set(states[1]) == set(states[2])
    assert all(states[0][key].shape == states[1][key].shape == states[2][key].shape
               for key in states[0])
    assert "control_mask" not in states[0]


def test_vision_only_ignores_action_channels_but_not_time_delta():
    model = TinyControlAblation("vision_only").eval()
    frames = torch.rand(2, 5, 3, 32, 32)
    control = torch.rand(2, 5, 3)
    changed = control.clone()
    changed[..., :2] = 1 - changed[..., :2]
    with torch.inference_mode():
        first = model(frames, control)
        second = model(frames, changed)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert tuple(model.control_mask.tolist()) == (0., 0., 1.)


def test_state_only_ignores_age_but_observes_state():
    model = TinyControlAblation("state_only").eval()
    frames = torch.rand(2, 5, 3, 32, 32)
    control = torch.rand(2, 5, 3)
    changed = control.clone()
    changed[..., 1] += .9
    with torch.inference_mode():
        before, after = model(frames, control), model(frames, changed)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert tuple(model.control_mask.tolist()) == (1., 0., 1.)


def test_full_wrapper_matches_original_policy_when_weights_identical():
    model = TinyControlAblation("full").eval()
    frames = torch.rand(2, 5, 3, 32, 32)
    control = torch.rand(2, 5, 3)
    with torch.inference_mode():
        torch.testing.assert_close(model(frames, control),
                                   model.policy(frames, control), rtol=0, atol=0)


def test_invalid_variant_and_control_shape_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        TinyControlAblation("rgb_hsv")
    model = TinyControlAblation("full")
    with pytest.raises(ValueError, match="shape"):
        model(torch.rand(1, 5, 3, 32, 32), torch.rand(1, 5, 2))


def test_variant_paths_do_not_overlap_frozen_baseline(tmp_path):
    script = ROOT / "scripts/train_tiny_control_ablations.py"
    spec = importlib.util.spec_from_file_location("train_tiny_control_ablations_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = [module.variant_paths(tmp_path, v, smoke=False)
             for v in CONTROL_MASKS]
    assert len({path for pair in paths for path in pair}) == 6
    assert all("tiny_policy" not in path.name for pair in paths for path in pair)
    assert module.variant_paths(tmp_path, "full", smoke=True) != paths[0]
    with pytest.raises(ValueError, match="unknown"):
        module.variant_paths(tmp_path, "fake", smoke=False)
