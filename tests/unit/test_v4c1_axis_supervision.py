from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


def test_axis_target_is_axial_not_directional() -> None:
    from scripts.build_axis_pseudo_labels import axis_target

    zero = axis_target(0.0)
    half_turn = axis_target(180.0)
    quarter_turn = axis_target(90.0)

    assert zero[0] == pytest.approx(half_turn[0], abs=1e-7)
    assert zero[1] == pytest.approx(half_turn[1], abs=1e-7)
    assert quarter_turn[0] == pytest.approx(-1.0, abs=1e-7)
    assert quarter_turn[1] == pytest.approx(0.0, abs=1e-7)


def test_axis_label_loader_reads_weighted_vector(tmp_path: Path) -> None:
    from karting_agent.train.axis_labels import load_axis_pseudo_labels

    path = tmp_path / "axis.jsonl"
    path.write_text(
        json.dumps(
            {
                "video": "data/raw/a.mp4",
                "frame_index": 12,
                "angle_deg": 45.0,
                "target_x": math.cos(math.pi / 2),
                "target_y": math.sin(math.pi / 2),
                "weight": 0.8,
                "straight_confidence": 0.9,
                "corner_score": 0.1,
                "road_area_fraction": 0.4,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    labels = load_axis_pseudo_labels(path)
    label = labels[("data/raw/a.mp4", 12)]
    assert label.weight == pytest.approx(0.8)
    assert label.target_x == pytest.approx(0.0, abs=1e-7)
    assert label.target_y == pytest.approx(1.0, abs=1e-7)


def test_v4c1_model_keeps_v3_input_and_adds_axis_head() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from karting_agent.model.axis_supervised import build_axis_supervised_model

    model = build_axis_supervised_model(
        pretrained=False,
        frame_stack=5,
        horizon_count=3,
        visual_feature_dim=16,
        state_embedding_dim=4,
        hidden_dim=8,
    ).eval()
    inputs = torch.randn(2, 15, 64, 64)
    states = torch.tensor([0, 1])

    with torch.inference_mode():
        switch_logits, future_logits, axis_vector = model(inputs, states)

    assert switch_logits.shape == (2, 3)
    assert future_logits.shape == (2, 3)
    assert axis_vector.shape == (2, 2)


def test_weighted_axis_loss_ignores_zero_weight_samples() -> None:
    torch = pytest.importorskip("torch")
    from scripts.train_model_v4c1 import weighted_axis_loss

    prediction = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    weights = torch.tensor([1.0, 0.0])

    loss, _, normalized_weights = weighted_axis_loss(prediction, target, weights)

    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    assert normalized_weights.tolist() == pytest.approx([1.0, 0.0])
