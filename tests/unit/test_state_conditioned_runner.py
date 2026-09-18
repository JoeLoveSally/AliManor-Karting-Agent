from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from karting_agent.model.runner import _runtime_spec
from karting_agent.model.state_conditioned_runner import (
    _SUPPORTED_MODEL_FAMILIES,
    _control_state_dict,
    _switch_probability_source,
)


def test_v4c2_runtime_family_is_supported() -> None:
    assert "state_conditioned_kart_relative_v4c2" in _SUPPORTED_MODEL_FAMILIES


def test_v5_h0_runtime_family_is_supported() -> None:
    assert "state_conditioned_kart_relative_v5_h0" in _SUPPORTED_MODEL_FAMILIES


def test_v4c2_runtime_strips_only_auxiliary_heads() -> None:
    state_dict = {
        "visual_encoder.features.0.0.weight": object(),
        "state_embedding.weight": object(),
        "switch_head.0.weight": object(),
        "future_action_head.weight": object(),
        "lateral_head.weight": object(),
        "heading_error_head.weight": object(),
        "edge_risk_head.weight": object(),
    }

    control_state = _control_state_dict(
        state_dict,
        "state_conditioned_kart_relative_v4c2",
    )

    assert set(control_state) == {
        "visual_encoder.features.0.0.weight",
        "state_embedding.weight",
        "switch_head.0.weight",
        "future_action_head.weight",
    }
    assert set(state_dict) - set(control_state) == {
        "lateral_head.weight",
        "heading_error_head.weight",
        "edge_risk_head.weight",
    }


def test_switch_probability_source_defaults_to_native_head() -> None:
    assert _switch_probability_source({}) == "native_switch"


def test_switch_probability_source_accepts_future_action_projection() -> None:
    assert (
        _switch_probability_source(
            {"switch_probability_source": "future_action_projection"}
        )
        == "future_action_projection"
    )


def test_switch_probability_source_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="unsupported switch_probability_source"):
        _switch_probability_source({"switch_probability_source": "bad"})


def test_runtime_spec_accepts_legacy_config_path(tmp_path: Path) -> None:
    config = tmp_path / "train_v4c2.yaml"
    config.write_text(
        """
model:
  architecture: mobilenet_v3_small
  frame_stack: 5
dataset:
  history_ms: 200
  frame_interval_ms: 50
  prediction_horizons_ms: [100, 200, 300]
  control_horizon_ms: 100
preprocess:
  mask_touch_area: true
  touch_roi: [0.78, 0.82, 0.98, 0.98]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    spec = _runtime_spec(
        {
            "config": str(config),
            "architecture": "mobilenet_v3_small",
            "frame_stack": 5,
            "prediction_horizons_ms": [100, 200, 300],
            "control_horizon_ms": 100,
        }
    )

    assert spec.frame_offsets_ms == (-200.0, -150.0, -100.0, -50.0, 0.0)
    assert spec.control_horizon_ms == 100.0


def test_v5_h0_runtime_strips_kart_relative_auxiliary_heads() -> None:
    state_dict = {
        "visual_encoder.features.0.0.weight": object(),
        "state_embedding.weight": object(),
        "switch_head.0.weight": object(),
        "future_action_head.weight": object(),
        "lateral_head.weight": object(),
        "heading_error_head.weight": object(),
        "edge_risk_head.weight": object(),
    }

    control_state = _control_state_dict(
        state_dict,
        "state_conditioned_kart_relative_v5_h0",
    )

    assert set(control_state) == {
        "visual_encoder.features.0.0.weight",
        "state_embedding.weight",
        "switch_head.0.weight",
        "future_action_head.weight",
    }
