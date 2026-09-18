from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

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


def make_runtime_config(
    *,
    execute_pending_at_due: bool = False,
    min_state_hold_ms: float = 0.0,
) -> StateConditionedRuntimeConfig:
    return StateConditionedRuntimeConfig(
        target_fps=30.0,
        frame_offsets_ms=(0.0,),
        prediction_horizon_ms=200.0,
        switch_threshold=0.6,
        min_state_hold_ms=min_state_hold_ms,
        execute_pending_at_due=execute_pending_at_due,
    )


def make_preprocess_config() -> PreprocessConfig:
    return PreprocessConfig(
        input_width=8,
        input_height=8,
        mask_touch_area=False,
    )


def make_scheduler() -> MultiHorizonSwitchScheduler:
    return MultiHorizonSwitchScheduler(
        MultiHorizonSchedulerConfig(
            horizons_ms=(100.0, 200.0, 300.0),
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
            threshold=0.6,
            min_state_hold_ms=100.0,
        )
    )


def test_scheduler_runtime_arms_executes_and_respects_min_hold() -> None:
    executor = RecordingExecutor()
    scheduler = make_scheduler()
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


def test_deadline_runtime_rechecks_pending_on_intermediate_frames_and_fires_timer() -> None:
    executor = RecordingExecutor()
    timers = ManualTimerFactory()
    model = SequenceModel(
        [
            (0.10, 0.55, 0.80),
            (0.10, 0.56, 0.85),
            (0.99, 0.90, 0.01),
        ]
    )
    engine = StateConditionedRuntimeEngine(
        model=model,
        preprocess_config=make_preprocess_config(),
        executor=executor,
        config=make_runtime_config(execute_pending_at_due=True),
        scheduler=make_scheduler(),
        timer_factory=timers,
        clock=lambda: 10.0,
    )

    armed = engine.ingest(make_frame(0, 0.0))
    monitored = engine.ingest(make_frame(1, 10.0))

    assert armed is not None
    assert armed.scheduler_reason == "pending_armed"
    assert armed.pending_due_ms == pytest.approx(20.0)
    assert monitored is not None
    assert monitored.scheduler_reason == "pending_wait"
    assert monitored.inference_trigger == "pending_monitor"
    assert len(timers.timers) == 2
    assert timers.timers[0].cancelled is True
    assert timers.timers[1].cancelled is False

    timers.timers[1].fire()
    events = engine.drain_deadline_events()

    assert executor.states == [True]
    assert len(events) == 1
    assert events[0].action is ControlAction.PRESS
    assert events[0].pressed is True
    assert events[0].timestamp_ms == pytest.approx(20.0)
    assert events[0].scheduler_reason == "pending_execute_deadline"

    held = engine.ingest(make_frame(2, 35.0))
    assert held is not None
    assert held.scheduler_reason == "min_hold"
    assert held.action is ControlAction.HOLD
    assert held.pressed is True


def test_deadline_runtime_can_arm_during_min_hold_and_execute_at_hold_expiry() -> None:
    executor = RecordingExecutor()
    timers = ManualTimerFactory()
    scheduler = MultiHorizonSwitchScheduler(
        MultiHorizonSchedulerConfig(
            horizons_ms=(100.0, 200.0, 300.0),
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
            threshold=0.6,
            min_state_hold_ms=100.0,
            arm_pending_during_min_hold=True,
        )
    )
    model = SequenceModel(
        [
            (0.10, 0.90, 0.90),
            (0.10, 0.59, 0.90),
        ]
    )
    engine = StateConditionedRuntimeEngine(
        model=model,
        preprocess_config=make_preprocess_config(),
        executor=executor,
        config=make_runtime_config(execute_pending_at_due=True),
        scheduler=scheduler,
        timer_factory=timers,
        clock=lambda: 10.0,
    )

    first = engine.ingest(make_frame(0, 0.0))
    armed = engine.ingest(make_frame(1, 50.0))

    assert first is not None
    assert first.action is ControlAction.PRESS
    assert first.scheduler_reason == "primary"
    assert armed is not None
    assert armed.action is ControlAction.HOLD
    assert armed.scheduler_reason == "pending_armed"
    assert armed.pending_due_ms == pytest.approx(100.0)
    assert armed.pending_delay_ms == pytest.approx(50.0)
    assert timers.timers[-1].interval == pytest.approx(0.05)

    timers.timers[-1].fire()
    events = engine.drain_deadline_events()

    assert executor.states == [True, False]
    assert len(events) == 1
    assert events[0].action is ControlAction.RELEASE
    assert events[0].pressed is False
    assert events[0].timestamp_ms == pytest.approx(100.0)
    assert events[0].scheduler_reason == "pending_execute_deadline"


def test_deadline_runtime_cancels_timer_when_warning_disappears() -> None:
    executor = RecordingExecutor()
    timers = ManualTimerFactory()
    model = SequenceModel(
        [
            (0.10, 0.55, 0.80),
            (0.10, 0.10, 0.40),
        ]
    )
    engine = StateConditionedRuntimeEngine(
        model=model,
        preprocess_config=make_preprocess_config(),
        executor=executor,
        config=make_runtime_config(execute_pending_at_due=True),
        scheduler=make_scheduler(),
        timer_factory=timers,
        clock=lambda: 10.0,
    )

    armed = engine.ingest(make_frame(0, 0.0))
    cancelled = engine.ingest(make_frame(1, 10.0))

    assert armed is not None
    assert cancelled is not None
    assert cancelled.scheduler_reason == "pending_cancelled"
    assert timers.timers[0].cancelled is True
    timers.timers[0].fire()
    assert executor.states == []
    assert engine.drain_deadline_events() == []


def test_deadline_monitor_is_opt_in() -> None:
    executor = RecordingExecutor()
    engine = StateConditionedRuntimeEngine(
        model=SequenceModel([(0.10, 0.55, 0.80)]),
        preprocess_config=make_preprocess_config(),
        executor=executor,
        config=make_runtime_config(execute_pending_at_due=False),
        scheduler=make_scheduler(),
    )

    armed = engine.ingest(make_frame(0, 0.0))
    intermediate = engine.ingest(make_frame(1, 10.0))

    assert armed is not None
    assert armed.scheduler_reason == "pending_armed"
    assert intermediate is None
    assert executor.states == []


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


def test_fixed_runtime_can_apply_direct_min_state_hold() -> None:
    executor = RecordingExecutor()
    engine = StateConditionedRuntimeEngine(
        model=FixedModel(),
        preprocess_config=make_preprocess_config(),
        executor=executor,
        config=make_runtime_config(min_state_hold_ms=100.0),
    )

    first = engine.ingest(make_frame(0, 0.0))
    blocked = engine.ingest(make_frame(1, 50.0))
    reversed_step = engine.ingest(make_frame(2, 100.0))
    blocked_again = engine.ingest(make_frame(3, 150.0))

    assert first is not None
    assert first.action is ControlAction.PRESS
    assert first.scheduler_reason == "primary"
    assert blocked is not None
    assert blocked.action is ControlAction.HOLD
    assert blocked.scheduler_reason == "min_hold"
    assert reversed_step is not None
    assert reversed_step.action is ControlAction.RELEASE
    assert reversed_step.scheduler_reason == "primary"
    assert blocked_again is not None
    assert blocked_again.action is ControlAction.HOLD
    assert blocked_again.scheduler_reason == "min_hold"
    assert executor.states == [True, False]


def test_runtime_preserves_bounded_reversal_seen_during_min_hold() -> None:
    executor = RecordingExecutor()
    scheduler = MultiHorizonSwitchScheduler(
        MultiHorizonSchedulerConfig(
            horizons_ms=(100.0, 200.0, 300.0),
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
            threshold=0.6,
            min_state_hold_ms=100.0,
            reserve_bounded_reversal_during_min_hold=True,
        )
    )
    model = SequenceModel(
        [
            (0.10, 0.70, 0.90),
            (0.98, 0.86, 0.03),
            (0.25, 0.10, 0.05),
            (0.10, 0.05, 0.05),
        ]
    )
    engine = StateConditionedRuntimeEngine(
        model=model,
        preprocess_config=make_preprocess_config(),
        executor=executor,
        config=make_runtime_config(),
        scheduler=scheduler,
    )

    first = engine.ingest(make_frame(0, 0.0))
    armed = engine.ingest(make_frame(1, 50.0))
    waiting = engine.ingest(make_frame(2, 80.0))
    reversed_step = engine.ingest(make_frame(3, 100.0))

    assert first is not None
    assert first.action is ControlAction.PRESS
    assert first.scheduler_reason == "primary"

    assert armed is not None
    assert armed.action is ControlAction.HOLD
    assert armed.scheduler_reason == "hold_reversal_armed"
    assert armed.pending_due_ms == pytest.approx(100.0)

    assert waiting is not None
    assert waiting.action is ControlAction.HOLD
    assert waiting.scheduler_reason == "hold_reversal_wait"

    assert reversed_step is not None
    assert reversed_step.action is ControlAction.RELEASE
    assert reversed_step.scheduler_reason == "hold_reversal_execute"
    assert executor.states == [True, False]
