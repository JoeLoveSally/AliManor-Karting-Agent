from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from karting_agent.data_flow.execute.mock import MockExecutor
from karting_agent.data_flow.input.common import Frame
from karting_agent.model.event_time_runner import EventTimeActionRunner
from karting_agent.model.temporal_delta import transform_temporal_input_numpy
from karting_agent.runtime.event_time_action_engine import (
    ActionRuntimeConfig,
    EventTimeActionEngine,
)
from karting_agent.vision.preprocess import PreprocessConfig, stack_frames


class FixedActionModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = iter(probabilities)
        self.inputs: list[np.ndarray] = []

    def predict_action(self, normalized_rgb_stack: np.ndarray) -> float:
        self.inputs.append(normalized_rgb_stack.copy())
        return next(self.probabilities)


def frame(timestamp_ms: float, index: int) -> Frame:
    return Frame(
        image=np.full((32, 32, 3), index % 255, dtype=np.uint8),
        frame_index=index,
        timestamp_ms=timestamp_ms,
    )


def make_engine(probabilities: list[float]):
    model = FixedActionModel(probabilities)
    executor = MockExecutor()
    engine = EventTimeActionEngine(
        model=model,
        executor=executor,
        preprocess_config=PreprocessConfig(),
        config=ActionRuntimeConfig(target_fps=30),
    )
    return engine, model, executor


def test_history_waits_for_full_200ms_and_preserves_train_stack_order() -> None:
    engine, model, executor = make_engine([0.1])
    for index, timestamp in enumerate((0.0, 50.0, 100.0, 150.0)):
        assert engine.ingest(frame(timestamp, index)) is None
    step = engine.ingest(frame(200.0, 4))
    assert step is not None
    assert step["history_frame_indices"] == (0, 1, 2, 3, 4)
    expected = stack_frames(
        [frame(t, i).image for i, t in enumerate((0, 50, 100, 150, 200))],
        PreprocessConfig(),
    )
    np.testing.assert_array_equal(model.inputs[0], expected)
    assert executor.calls == []
    assert engine.shutdown() is False
    assert executor.calls == []


def test_minimum_hold_pending_executes_only_on_arriving_frame() -> None:
    engine, _, executor = make_engine([0.99, 0.01, 0.01, 0.01, 0.99])
    for index, timestamp in enumerate((0.0, 50.0, 100.0, 150.0)):
        assert engine.ingest(frame(timestamp, index)) is None
    first = engine.ingest(frame(200.0, 4))
    assert first is not None
    assert first["events"][0]["reason"] == "action_mismatch"
    assert executor.calls == [True]
    # Use timestamps that actually cross the 30fps scheduler deadlines.
    held = engine.ingest(frame(233.334, 5))
    assert held is not None
    assert held["decoder_reason"] == "action_hold"
    assert held["pending_due_ms"] == pytest.approx(300.0)
    assert executor.calls == [True]
    assert engine.ingest(frame(266.668, 6)) is not None
    assert executor.calls == [True]
    executed = engine.ingest(frame(333.334, 7))
    assert executed is not None
    assert executed["events"][0]["reason"] == "pending_execute"
    assert executed["events"][0]["pending_due_at_ms"] == pytest.approx(300.0)
    assert executed["events"][0]["observation_timestamp_ms"] == pytest.approx(333.334)
    assert engine.decoder.last_switch_ms == pytest.approx(333.334)
    assert executor.calls == [True, False]
    # A reverse action at 366ms is inside the *real* hold begun at 333ms.
    opposite = engine.ingest(frame(366.668, 8))
    assert opposite is not None
    assert opposite["decoder_reason"] == "action_hold"
    assert opposite["pending_due_ms"] == pytest.approx(433.334)
    assert executor.calls == [True, False]
    assert engine.shutdown() is False


def test_shutdown_releases_after_failure_and_prevents_further_ingest() -> None:
    engine, _, executor = make_engine([0.99])
    for i in range(5):
        step = engine.ingest(frame(i * 50.0, i))
    assert step is not None
    assert engine.shutdown() is True
    assert executor.calls == [True, False]
    assert engine.shutdown() is False
    with pytest.raises(RuntimeError, match="after shutdown"):
        engine.ingest(frame(250.0, 5))


def test_config_rejects_wrong_history_and_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        ActionRuntimeConfig(frame_offsets_ms=(-150.0, -100.0, -50.0, 0.0)).validate()
    with pytest.raises(ValueError):
        ActionRuntimeConfig(action_threshold=1.0).validate()


def test_model_runner_matches_direct_offline_inference(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from karting_agent.model.event_time import build_event_time_model

    torch.manual_seed(7)
    metadata = {
        "model_family": "event_time_v4c4",
        "architecture": "mobilenet_v3_small",
        "frame_stack": 5,
        "input_representation": "current_rgb_plus_adjacent_deltas",
        "event_time_classes": 7,
        "no_event_class": 6,
        "event_time_bin_ms": 50.0,
        "visual_feature_dim": 16,
        "hidden_dim": 16,
    }
    model = build_event_time_model(
        architecture="mobilenet_v3_small", frame_stack=5, pretrained=False,
        event_time_classes=7, visual_feature_dim=16, hidden_dim=16,
    )
    model.eval()
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    runner = EventTimeActionRunner(tmp_path / "model.pt", device="cpu", torch_num_threads=2)
    stacked = np.random.default_rng(5).normal(size=(15, 224, 224)).astype(np.float32)
    transformed = transform_temporal_input_numpy(
        stacked, frame_stack=5, representation="current_rgb_plus_adjacent_deltas"
    )
    with torch.inference_mode():
        logits, *_ = model(torch.from_numpy(transformed).unsqueeze(0))
        expected = float(torch.sigmoid(logits)[0].item())
    assert runner.predict_action(stacked) == pytest.approx(expected, abs=1e-6)


def test_entrypoint_requires_explicit_touch_coordinates() -> None:
    import importlib.util
    import sys

    script_path = Path(__file__).resolve().parents[2] / "scripts/run_adb_closed_loop_v4c4.py"
    spec = importlib.util.spec_from_file_location("run_adb_closed_loop_v4c4", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script_path.parent))
    try:
        spec.loader.exec_module(module)
        with pytest.raises(ValueError, match="explicit --x and --y"):
            module.main(["--arm"])
        with pytest.raises(ValueError, match="requires --arm"):
            module.main(["--wait-for-start"])
    finally:
        sys.path.remove(str(script_path.parent))
