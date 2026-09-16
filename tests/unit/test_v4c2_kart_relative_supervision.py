from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_heading_error_target_wraps_axially() -> None:
    from scripts.build_kart_relative_pseudo_labels import heading_error_target

    pos = heading_error_target(90.0)
    neg = heading_error_target(-90.0)
    zero = heading_error_target(0.0)

    assert pos == pytest.approx(neg, abs=1e-7)
    assert zero == pytest.approx((1.0, 0.0), abs=1e-7)
    assert pos == pytest.approx((-1.0, 0.0), abs=1e-7)


def test_kart_relative_label_loader_reads_targets(tmp_path: Path) -> None:
    from karting_agent.train.kart_relative_labels import (
        load_kart_relative_pseudo_labels,
    )

    path = tmp_path / "relation.jsonl"
    path.write_text(
        json.dumps(
            {
                "video": "data/raw/a.mp4",
                "frame_index": 42,
                "lateral_target": 0.4,
                "heading_target_x": 0.5,
                "heading_target_y": 0.8660254,
                "edge_risk_target": 1.0,
                "weight": 0.7,
                "raw_lateral_offset_norm": 0.8,
                "heading_error_deg": 30.0,
                "inside_road": True,
                "confidence": 0.7,
                "heading_quality": 0.9,
                "road_support_fraction": 0.8,
                "valid_cross_sections": 2,
                "cross_section_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    label = load_kart_relative_pseudo_labels(path)[("data/raw/a.mp4", 42)]
    assert label.lateral_target == pytest.approx(0.4)
    assert label.heading_error_deg == pytest.approx(30.0)
    assert label.edge_risk_target == pytest.approx(1.0)
    assert label.weight == pytest.approx(0.7)


def test_v4c2_model_keeps_v3_control_and_adds_relation_heads() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from karting_agent.model.kart_relative_supervised import build_kart_relative_model

    model = build_kart_relative_model(
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
        switch_logits, future_logits, lateral, heading, edge_logit = model(
            inputs, states
        )

    assert switch_logits.shape == (2, 3)
    assert future_logits.shape == (2, 3)
    assert lateral.shape == (2,)
    assert heading.shape == (2, 2)
    assert edge_logit.shape == (2,)


def test_v4c2_weighted_losses_ignore_zero_weight_samples() -> None:
    torch = pytest.importorskip("torch")
    from scripts.train_model_v4c2 import (
        weighted_edge_risk_loss,
        weighted_heading_loss,
        weighted_lateral_loss,
    )

    weights = torch.tensor([1.0, 0.0])
    lateral_loss = weighted_lateral_loss(
        torch.tensor([0.25, 9.0]),
        torch.tensor([0.25, 0.0]),
        weights,
    )
    heading_loss, _ = weighted_heading_loss(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        weights,
    )
    edge_loss = weighted_edge_risk_loss(
        torch.tensor([20.0, -20.0]),
        torch.tensor([1.0, 1.0]),
        weights,
    )

    assert lateral_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert heading_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert edge_loss.item() < 1e-6


def test_kart_relative_metrics_report_expected_geometry_error() -> None:
    np = pytest.importorskip("numpy")
    from scripts.evaluate_model_v4c2 import kart_relative_metrics

    result = kart_relative_metrics(
        lateral_prediction=np.asarray([0.2, 99.0]),
        heading_prediction=np.asarray([[1.0, 0.0], [-1.0, 0.0]]),
        edge_probability=np.asarray([0.9, 0.1]),
        lateral_target=np.asarray([0.1, 0.0]),
        heading_target=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        edge_target=np.asarray([1.0, 0.0]),
        weights=np.asarray([1.0, 0.0]),
    )

    assert result["samples"] == 1
    assert result["weighted_lateral_mae"] == pytest.approx(0.1)
    assert result["weighted_heading_error_deg"] == pytest.approx(0.0)
    assert result["edge_risk"]["f1"] == pytest.approx(1.0)
