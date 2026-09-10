import numpy as np
import pytest

from karting_agent.control.controller import (
    ControlAction,
    HysteresisConfig,
    HysteresisController,
)
from karting_agent.data_flow.execute.mock import MockExecutor
from karting_agent.data_flow.input.common import Frame
from karting_agent.model.runner import _runtime_spec
from karting_agent.runtime.engine import RuntimeEngine, RuntimeEngineConfig
from karting_agent.vision.preprocess import PreprocessConfig
from karting_agent.vision.temporal import TemporalFrameBuffer


def _frame(timestamp_ms: float, frame_index: int) -> Frame:
    return Frame(
        image=np.full((4, 6, 3), frame_index, dtype=np.uint8),
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
    )


def test_temporal_buffer_selects_only_observed_history() -> None:
    buffer = TemporalFrameBuffer(history_ms=100.0)
    for frame in (
        _frame(0.0, 0),
        _frame(40.0, 1),
        _frame(80.0, 2),
        _frame(120.0, 3),
    ):
        buffer.add(frame)

    stack = buffer.stack_at(
        120.0,
        (-100.0, -50.0, 0.0),
        PreprocessConfig(input_width=6, input_height=4, mask_touch_area=False),
    )

    assert stack is not None
    assert stack.frame_timestamps_ms == (0.0, 40.0, 120.0)
    assert stack.frame_indices == (0, 1, 3)
    assert stack.input.shape == (9, 4, 6)


def test_temporal_buffer_rejects_non_monotonic_input() -> None:
    buffer = TemporalFrameBuffer(history_ms=100.0)
    buffer.add(_frame(100.0, 0))

    with pytest.raises(ValueError):
        buffer.add(_frame(90.0, 1))


def test_runtime_engine_drives_hysteresis_and_mock_executor() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self._probabilities = iter((0.60, 0.40))

        def predict(self, inputs: np.ndarray) -> float:
            assert inputs.shape == (9, 4, 6)
            return next(self._probabilities)

    executor = MockExecutor()
    controller = HysteresisController(
        HysteresisConfig(press_threshold=0.55, release_threshold=0.45)
    )
    engine = RuntimeEngine(
        model=FakeModel(),
        preprocess_config=PreprocessConfig(
            input_width=6,
            input_height=4,
            mask_touch_area=False,
        ),
        controller=controller,
        executor=executor,
        config=RuntimeEngineConfig(
            target_fps=10.0,
            frame_offsets_ms=(-100.0, -50.0, 0.0),
            prediction_horizon_ms=100.0,
        ),
    )

    steps = []
    for timestamp_ms, frame_index in (
        (0.0, 0),
        (50.0, 1),
        (100.0, 2),
        (150.0, 3),
        (200.0, 4),
    ):
        step = engine.ingest(_frame(timestamp_ms, frame_index))
        if step is not None:
            steps.append(step)

    assert [step.action for step in steps] == [
        ControlAction.PRESS,
        ControlAction.RELEASE,
    ]
    assert executor.calls == [True, False]
    assert [step.observation_timestamp_ms for step in steps] == [100.0, 200.0]
    assert steps[0].input_frame_timestamps_ms == (0.0, 50.0, 100.0)
    assert steps[0].prediction_target_timestamp_ms == 200.0
    assert engine.shutdown() is False


def test_runtime_engine_shutdown_releases_pressed_state() -> None:
    class PressModel:
        def predict(self, inputs: np.ndarray) -> float:
            return 0.9

    executor = MockExecutor()
    controller = HysteresisController(
        HysteresisConfig(press_threshold=0.55, release_threshold=0.45)
    )
    engine = RuntimeEngine(
        model=PressModel(),
        preprocess_config=PreprocessConfig(
            input_width=6,
            input_height=4,
            mask_touch_area=False,
        ),
        controller=controller,
        executor=executor,
        config=RuntimeEngineConfig(
            target_fps=10.0,
            frame_offsets_ms=(0.0,),
            prediction_horizon_ms=100.0,
        ),
    )

    assert engine.ingest(_frame(0.0, 0)) is not None
    assert controller.pressed is True
    assert engine.shutdown() is True
    assert controller.pressed is False
    assert executor.calls == [True, False]


def test_model_runtime_spec_uses_artifact_training_metadata() -> None:
    spec = _runtime_spec(
        {
            "architecture": "mobilenet_v3_small",
            "frame_stack": 3,
            "config": {
                "model": {
                    "architecture": "mobilenet_v3_small",
                    "input_size": 224,
                    "frame_stack": 3,
                },
                "dataset": {
                    "history_ms": 100,
                    "frame_interval_ms": 50,
                    "prediction_horizon_ms": 100,
                },
                "preprocess": {
                    "mask_touch_area": True,
                    "touch_roi": [0.78, 0.82, 0.98, 0.98],
                },
            },
        }
    )

    assert spec.frame_offsets_ms == (-100.0, -50.0, 0.0)
    assert spec.prediction_horizon_ms == 100.0
    assert spec.preprocess_config.input_width == 224
    assert spec.preprocess_config.touch_roi == (0.78, 0.82, 0.98, 0.98)
