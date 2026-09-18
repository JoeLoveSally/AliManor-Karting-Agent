from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from karting_agent.control.controller import ControlAction
from karting_agent.data_flow.input.common import Frame
from karting_agent.runtime.consensus_dense_scheduler import (
    ConsensusDenseConfig,
    ConsensusDenseHorizonScheduler,
)
from karting_agent.runtime.consensus_transition_lifecycle import (
    ConsensusTransitionLifecycle,
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
        self,
        inputs: np.ndarray,
        current_pressed: bool,
    ) -> tuple[float, ...]:
        del inputs, current_pressed
        return next(self.outputs)

    def predict_switch(self, inputs: np.ndarray, current_pressed: bool) -> float:
        del inputs, current_pressed
        raise AssertionError("consensus runtime must use predict_switch_all")


class CoherentSequenceModel:
    def __init__(
        self,
        outputs: list[tuple[tuple[float, ...], tuple[float, ...]]],
    ) -> None:
        self.outputs = iter(outputs)
        self.calls = 0

    def predict_native_switch_and_future_all(
        self,
        inputs: np.ndarray,
        current_pressed: bool,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        del inputs, current_pressed
        self.calls += 1
        return next(self.outputs)

    def predict_switch_all(
        self,
        inputs: np.ndarray,
        current_pressed: bool,
    ) -> tuple[float, ...]:
        del inputs, current_pressed
        raise AssertionError("coherent H0 path must use one-pass native+future output")

    def predict_switch(self, inputs: np.ndarray, current_pressed: bool) -> float:
        del inputs, current_pressed
        raise AssertionError("coherent H0 path must not use predict_switch")


class ManualTimer:
    def __init__(self, interval: float, callback: Callable[[], None]) -> None:
        self.interval = float(interval)
        self.callback = callback
        self.started = False
        self.cancelled = False
        self.daemon = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if self.started and not self.cancelled:
            self.callback()


class ManualTimerFactory:
    def __init__(self) -> None:
        self.timers: list[ManualTimer] = []

    def __call__(self, interval: float, callback: Callable[[], None]) -> ManualTimer:
        timer = ManualTimer(interval, callback)
        self.timers.append(timer)
        return timer


def make_frame(index: int, timestamp_ms: float) -> Frame:
    return Frame(
        image=np.zeros((16, 16, 3), dtype=np.uint8),
        frame_index=index,
        timestamp_ms=timestamp_ms,
    )


def make_preprocess_config() -> PreprocessConfig:
    return PreprocessConfig(
        input_width=8,
        input_height=8,
        mask_touch_area=False,
    )


def make_scheduler() -> ConsensusDenseHorizonScheduler:
    return ConsensusDenseHorizonScheduler(
        ConsensusDenseConfig(
            horizons_ms=(0.0, 50.0, 100.0, 150.0),
            threshold=0.6,
            min_state_hold_ms=100.0,
            min_consensus_heads=3,
            consensus_window_ms=60.0,
        )
    )


def test_consensus_deadline_lifecycle_blocks_h0_correction_until_confirmation() -> None:
    executor = RecordingExecutor()
    timers = ManualTimerFactory()
    model = SequenceModel(
        [
            (0.1, 0.1, 0.1, 0.7),
            (0.1, 0.1, 0.7, 0.8),
            (0.1, 0.7, 0.8, 0.9),
            # Current PRESS state wants to switch back, but the old RELEASE
            # state's H0 has not yet caught up to the scheduled PRESS.
            (0.9, 0.1, 0.1, 0.1),
            (0.5, 0.1, 0.1, 0.1),
            # Old-state H0 now confirms the earlier scheduled transition.
            (0.9, 0.1, 0.1, 0.1),
            (0.7, 0.1, 0.1, 0.1),
            # On the next observation ordinary H0 control resumes.
            (0.9, 0.1, 0.1, 0.1),
        ]
    )
    engine = StateConditionedRuntimeEngine(
        model=model,
        preprocess_config=make_preprocess_config(),
        executor=executor,
        config=StateConditionedRuntimeConfig(
            target_fps=30.0,
            frame_offsets_ms=(0.0,),
            prediction_horizon_ms=0.0,
            switch_threshold=0.6,
            execute_pending_at_due=True,
        ),
        scheduler=make_scheduler(),
        transition_lifecycle=ConsensusTransitionLifecycle(
            threshold=0.6,
            min_state_hold_ms=100.0,
        ),
        timer_factory=timers,
        clock=lambda: 10.0,
    )

    assert engine.ingest(make_frame(0, 0.0)) is not None
    assert engine.ingest(make_frame(1, 50.0)) is not None
    armed = engine.ingest(make_frame(2, 100.0))
    assert armed is not None
    assert armed.scheduler_reason == "consensus_armed"
    assert armed.pending_due_ms is not None

    timers.timers[-1].fire()
    deadline_events = engine.drain_deadline_events()
    assert executor.states == [True]
    assert len(deadline_events) == 1
    assert deadline_events[0].action is ControlAction.PRESS
    assert deadline_events[0].scheduler_reason == "consensus_execute_deadline"

    waiting = engine.ingest(make_frame(3, 170.0))
    assert waiting is not None
    assert waiting.action is ControlAction.HOLD
    assert waiting.pressed is True
    assert waiting.lifecycle_status == "waiting"

    confirmed = engine.ingest(make_frame(4, 210.0))
    assert confirmed is not None
    assert confirmed.action is ControlAction.HOLD
    assert confirmed.pressed is True
    assert confirmed.lifecycle_status == "confirmed"

    resumed = engine.ingest(make_frame(5, 250.0))
    assert resumed is not None
    assert resumed.action is ControlAction.RELEASE
    assert resumed.scheduler_reason == "primary"
    assert executor.states == [True, False]


def test_lifecycle_requires_consensus_scheduler() -> None:
    lifecycle = ConsensusTransitionLifecycle(
        threshold=0.6,
        min_state_hold_ms=100.0,
    )
    with pytest.raises(ValueError, match="requires ConsensusDenseHorizonScheduler"):
        StateConditionedRuntimeEngine(
            model=SequenceModel([]),
            preprocess_config=make_preprocess_config(),
            executor=RecordingExecutor(),
            config=StateConditionedRuntimeConfig(
                target_fps=30.0,
                frame_offsets_ms=(0.0,),
                prediction_horizon_ms=0.0,
            ),
            transition_lifecycle=lifecycle,
        )


def test_future_action_h0_uses_one_pass_and_coherent_lifecycle() -> None:
    executor = RecordingExecutor()
    timers = ManualTimerFactory()
    model = CoherentSequenceModel(
        [
            ((0.9, 0.1, 0.1, 0.7), (0.1, 0.0, 0.0, 0.0)),
            ((0.9, 0.1, 0.7, 0.8), (0.1, 0.0, 0.0, 0.0)),
            ((0.9, 0.7, 0.8, 0.9), (0.1, 0.0, 0.0, 0.0)),
            ((0.9, 0.1, 0.1, 0.1), (0.8, 0.0, 0.0, 0.0)),
            ((0.9, 0.1, 0.1, 0.1), (0.2, 0.0, 0.0, 0.0)),
        ]
    )
    engine = StateConditionedRuntimeEngine(
        model=model,
        preprocess_config=make_preprocess_config(),
        executor=executor,
        config=StateConditionedRuntimeConfig(
            target_fps=30.0,
            frame_offsets_ms=(0.0,),
            prediction_horizon_ms=0.0,
            switch_threshold=0.6,
            execute_pending_at_due=True,
            control_h0_source="future_action_projection",
        ),
        scheduler=make_scheduler(),
        transition_lifecycle=ConsensusTransitionLifecycle(
            threshold=0.6,
            min_state_hold_ms=100.0,
        ),
        timer_factory=timers,
        clock=lambda: 10.0,
    )

    engine.ingest(make_frame(0, 0.0))
    engine.ingest(make_frame(1, 50.0))
    armed = engine.ingest(make_frame(2, 100.0))
    assert armed is not None
    assert armed.scheduler_reason == "consensus_armed"
    assert armed.raw_control_probability == pytest.approx(0.1)

    timers.timers[-1].fire()
    deadline_events = engine.drain_deadline_events()
    assert len(deadline_events) == 1
    assert deadline_events[0].action is ControlAction.PRESS

    confirmed = engine.ingest(make_frame(3, 170.0))
    assert confirmed is not None
    assert confirmed.action is ControlAction.HOLD
    assert confirmed.lifecycle_status == "confirmed"
    assert confirmed.raw_control_probability == pytest.approx(0.2)
    assert confirmed.lifecycle_previous_state_probability == pytest.approx(0.8)

    resumed = engine.ingest(make_frame(4, 210.0))
    assert resumed is not None
    assert resumed.action is ControlAction.RELEASE
    assert resumed.scheduler_reason == "primary"
    assert model.calls == 5
    assert executor.states == [True, False]


def test_future_action_h0_requires_consensus_scheduler() -> None:
    with pytest.raises(
        ValueError,
        match="requires ConsensusDenseHorizonScheduler",
    ):
        StateConditionedRuntimeEngine(
            model=CoherentSequenceModel([]),
            preprocess_config=make_preprocess_config(),
            executor=RecordingExecutor(),
            config=StateConditionedRuntimeConfig(
                target_fps=30.0,
                frame_offsets_ms=(0.0,),
                prediction_horizon_ms=0.0,
                control_h0_source="future_action_projection",
            ),
        )
