from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from karting_agent.model.action_history import encode_action_history, history_feature_width
from karting_agent.model.action_history_residual import ActionHistoryResidualPolicy


def test_history_is_causal_and_preserves_frame_states():
    features = encode_action_history(
        (0, 50, 100, 150, 200), 200,
        initial_pressed=False,
        transitions=((120, True), (250, False)),  # Future event must not leak.
        current_pressed=True,
    )
    assert features.dtype == np.float32
    assert features.shape == (history_feature_width(5),)
    assert features[::3][:5].tolist() == [0., 0., 0., 1., 1.]
    assert features[2::3][:5].tolist() == [0., 0., 0., 1., 1.]
    assert features[-2] == pytest.approx(80 / 500)


def test_initial_hold_is_not_falsely_labeled_a_recent_switch():
    features = encode_action_history((0, 50, 100), 100, initial_pressed=True,
                                     transitions=(), current_pressed=True)
    assert features[2::3][:3].tolist() == [0., 0., 0.]
    assert features[-2:].tolist() == [1., 0.]


def test_current_state_mismatch_is_rejected():
    with pytest.raises(ValueError, match="disagrees"):
        encode_action_history((0, 50), 50, initial_pressed=False,
                              transitions=((20, True),), current_pressed=False)


def test_bad_time_axes_and_duplicate_switches_are_rejected():
    with pytest.raises(ValueError, match="ordered"):
        encode_action_history((10, 0), 10, initial_pressed=False,
                              transitions=(), current_pressed=False)
    with pytest.raises(ValueError, match="alternate"):
        encode_action_history((0, 50), 50, initial_pressed=False,
                              transitions=((10, True), (20, True)), current_pressed=True)


class TinyBaseline(nn.Module):
    visual_feature_dim = 4
    horizon_count = 3

    def __init__(self):
        super().__init__()
        self.visual_encoder = nn.Sequential(nn.Flatten(), nn.Linear(6, 4), nn.ReLU())
        self.state_embedding = nn.Embedding(2, 2)
        self.switch_head = nn.Sequential(nn.Linear(6, 3), nn.Dropout(p=0.5))


def test_zero_residual_preserves_baseline_and_only_adapter_trains():
    base = TinyBaseline().eval()
    model = ActionHistoryResidualPolicy(base, history_width=8)
    model.train()
    assert not model.base.training
    images = torch.rand(3, 1, 2, 3)
    states = torch.tensor([0, 1, 1])
    history = torch.rand(3, 8)
    corrected, baseline = model(images, states, history)
    torch.testing.assert_close(corrected, baseline, rtol=0, atol=0)
    target = torch.ones_like(corrected)
    torch.nn.functional.binary_cross_entropy_with_logits(corrected, target).backward()
    assert model.residual[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in base.parameters())


def test_residual_input_requires_explicit_history_width():
    model = ActionHistoryResidualPolicy(TinyBaseline(), history_width=8)
    with pytest.raises(ValueError, match="incorrect"):
        model(torch.rand(1, 1, 2, 3), torch.tensor([0]), torch.zeros(1, 7))
