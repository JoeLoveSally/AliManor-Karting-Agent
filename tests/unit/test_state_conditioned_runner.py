import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from karting_agent.model.state_conditioned_runner import (
    _SUPPORTED_MODEL_FAMILIES,
    _control_state_dict,
)


def test_v4c2_runtime_family_is_supported() -> None:
    assert "state_conditioned_kart_relative_v4c2" in _SUPPORTED_MODEL_FAMILIES


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
