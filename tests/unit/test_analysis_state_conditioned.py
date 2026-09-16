from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_model_focus as focus  # noqa: E402


def test_pre_action_pressed_recovers_runtime_input_state() -> None:
    assert focus.pre_action_pressed({"action": "HOLD", "pressed": True}) is True
    assert focus.pre_action_pressed({"action": "HOLD", "pressed": False}) is False
    assert focus.pre_action_pressed({"action": "PRESS", "pressed": True}) is False
    assert focus.pre_action_pressed({"action": "RELEASE", "pressed": False}) is True


def test_load_diagnostic_runner_selects_state_conditioned_family(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"model_family": "state_conditioned_kart_relative_v4c2"}),
        encoding="utf-8",
    )

    class FakeStateRunner:
        def __init__(
            self,
            model_path: Path,
            *,
            metadata_path: Path,
            device: str | None,
        ) -> None:
            self.model_path = model_path
            self.metadata_path = metadata_path
            self.device = device

    monkeypatch.setattr(focus, "StateConditionedModelRunner", FakeStateRunner)
    runner = focus.load_diagnostic_runner(
        tmp_path / "model.pt",
        metadata_path=metadata,
        device="cpu",
    )

    assert isinstance(runner, FakeStateRunner)
    assert runner.metadata_path == metadata.resolve()
    assert runner.device == "cpu"


def test_top_cells_maps_switch_direction_to_release_when_pressed() -> None:
    sensitivity = np.array([[0.4, -0.2]], dtype=np.float32)
    valid = np.ones_like(sensitivity, dtype=np.float32)

    result = focus.top_cells(
        sensitivity,
        valid,
        current_pressed=True,
        count=1,
    )

    assert result["switch_evidence"][0]["col"] == 0
    assert result["release_evidence"][0]["col"] == 0
    assert result["press_evidence"][0]["col"] == 1
