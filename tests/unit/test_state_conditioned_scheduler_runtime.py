from __future__ import annotations

import numpy as np

from karting_agent.control.controller import ControlAction
from karting_agent.data_flow.input.common import Frame
from karting_agent.runtime.multi_horizon_scheduler import (
    MultiHorizonSchedulerConfig,
    MultiHorizonSwitchScheduler,
)
from karting_agent.runtime.state_conditioned_engine import (
    StateConditionedRuntimeConfig,
    StateConditionedRuntimeEngine,
)
from karting_agent.vision.preprocess import PreprocessConfig


class RecordingExecutor:
    def __init__(self) -> None:
        self.states: list[bool] = []

    def set_pressed(self, pressed: bool) -> None:
        self.states.append(bool(pressed))


class SequenceModel:
    def __init__(self, outputs: list[tuple[float, ...]]) -> None:
        self.outputs = iter(outputs)

    def predict_switch_all(
        self, inputs: np.ndarray, current_pressed: bool
    ) -> tuple[float, ...]:
        del inputs, current_pressed
        return next(self.outputs)

    def predict_switch(self, inputs: np.ndarray, current_pressed: bool) -> float:
        del inputs, current_pressed
        raise AssertionError("scheduler runtime must use predict_switch_all")


class FixedModel:
    def predict_switch(self, inputs: np.ndarray, current_pressed: bool) -> float:
        del inputs, current_pressed
        return 0.7

    def predict_switch_all(
        self, inputs: np.ndarray, current_pressed: bool
    ) -> tuple[float, ...]:
        del inputs, current_pressed
        raise AssertionError("fixed runtime must use predict_switch")


def make_frame(index: int, timestamp_ms: float) -> Frame:
    return Frame(
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        frame_index=index,
        timestamp_ms=timestamp_ms,
    )


def make_runtime_config() -> StateConditionedRuntimeConfig:
    return StateConditionedRuntimeConfig(
        target_fps=30.0,
        frame_offsets_ms=(0.0,),
        prediction_horizon_ms=200.0,
        switch_threshold=0.6,
    )


def make_preprocess_config() -> PreprocessConfig:
    return PreprocessConfig(
        input_width=8,
        input_height=8,
        mask_touch_area=False,
    )


def test_scheduler_runtime_arms_executes_and_respects_min_hold() -> None:
    executor = RecordingExecutor()
    scheduler = MultiHorizonSwitchScheduler(
        MultiHorizonSchedulerConfig(
            horizons_ms=(100.0, 200.0, 300.0),
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
            threshold=0.6,
            min_state_hold_ms=100.0,
        )
    )
    model = SequenceModel(
        [
            (0.01, 0.05, 0.80),
            (0.02, 0.10, 0.85),
            (0.99, 0.90, 0.01),
            (0.99, 0.90, 0.01),
        ]
    )
    engine = StateConditionedRuntimeEngine(
        model=model,
        preprocess_config=make_preprocess_config(),
        executor=executor,
        config=make_runtime_config(),
        scheduler=scheduler,
    )

    armed = engine.ingest(make_frame(0, 0.0))
    executed = engine.ingest(make_frame(1, 100.0))
    held = engine.ingest(make_frame(2, 150.0))
    reversed_step = engine.ingest(make_frame(3, 210.0))

    assert armed is not None
    assert armed.action is ControlAction.HOLD
    assert armed.scheduler_reason == "pending_armed"
    assert armed.probabilities == (0.01, 0.05, 0.80)

    assert executed is not None
    assert executed.action is ControlAction.PRESS
    assert executed.scheduler_reason == "pending_execute"
    assert executed.pressed is True

    assert held is not None
    assert held.action is ControlAction.HOLD
    assert held.scheduler_reason == "min_hold"
    assert held.pressed is True

    assert reversed_step is not None
    assert reversed_step.action is ControlAction.RELEASE
    assert reversed_step.scheduler_reason == "primary"
    assert reversed_step.pressed is False
    assert executor.states == [True, False]


def test_fixed_runtime_path_remains_unchanged() -> None:
    executor = RecordingExecutor()
    engine = StateConditionedRuntimeEngine(
        model=FixedModel(),
        preprocess_config=make_preprocess_config(),
        executor=executor,
        config=make_runtime_config(),
    )

    step = engine.ingest(make_frame(0, 0.0))

    assert step is not None
    assert step.action is ControlAction.PRESS
    assert step.pressed is True
    assert step.probability == 0.7
    assert step.probabilities == (0.7,)
    assert step.scheduler_reason is None
    assert executor.states == [True]
