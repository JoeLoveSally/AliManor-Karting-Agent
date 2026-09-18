"""State-conditioned KEEP/SWITCH runtime for model v3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import RLock, Timer
import time
from typing import Callable, Protocol

import numpy as np

from karting_agent.control.controller import ControlAction
from karting_agent.data_flow.execute.common import Executor
from karting_agent.data_flow.input.common import Frame
from karting_agent.runtime.engine import RuntimeStep
from karting_agent.runtime.consensus_dense_scheduler import (
    ConsensusDenseHorizonScheduler,
)
from karting_agent.runtime.consensus_transition_lifecycle import (
    ConsensusTransitionLifecycle,
)
from karting_agent.runtime.multi_horizon_scheduler import MultiHorizonSwitchScheduler
from karting_agent.vision.preprocess import PreprocessConfig
from karting_agent.vision.temporal import TemporalFrameBuffer


class SwitchModel(Protocol):
    def predict_switch(self, inputs: np.ndarray, current_pressed: bool) -> float:
        """Return probability that the current physical state should be flipped."""

    def predict_switch_all(
        self, inputs: np.ndarray, current_pressed: bool
    ) -> tuple[float, ...]:
        """Return switch probabilities for all prediction horizons."""


class TimerHandle(Protocol):
    def start(self) -> None: ...

    def cancel(self) -> None: ...


TimerFactory = Callable[[float, Callable[[], None]], TimerHandle]


@dataclass(frozen=True)
class StateConditionedRuntimeConfig:
    target_fps: float
    frame_offsets_ms: tuple[float, ...]
    prediction_horizon_ms: float
    switch_threshold: float = 0.6
    min_state_hold_ms: float = 0.0
    execute_pending_at_due: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target_fps must be finite and > 0")
        if not self.frame_offsets_ms:
            raise ValueError("frame_offsets_ms must not be empty")
        if tuple(sorted(self.frame_offsets_ms)) != self.frame_offsets_ms:
            raise ValueError("frame_offsets_ms must be sorted")
        if self.frame_offsets_ms[-1] > 1e-6:
            raise ValueError("frame_offsets_ms must not look into the future")
        if (
            not math.isfinite(self.prediction_horizon_ms)
            or self.prediction_horizon_ms < 0
        ):
            raise ValueError("prediction_horizon_ms must be finite and >= 0")
        if not 0.0 < self.switch_threshold < 1.0:
            raise ValueError("switch_threshold must be in (0, 1)")
        if not math.isfinite(self.min_state_hold_ms) or self.min_state_hold_ms < 0.0:
            raise ValueError("min_state_hold_ms must be finite and >= 0")

    @property
    def history_ms(self) -> float:
        return max(0.0, -min(self.frame_offsets_ms))


@dataclass(frozen=True)
class StateConditionedRuntimeStep(RuntimeStep):
    """Runtime step with optional multi-horizon scheduler diagnostics."""

    probabilities: tuple[float, ...] = ()
    scheduler_reason: str | None = None
    pending_due_ms: float | None = None
    pending_delay_ms: float | None = None
    inference_trigger: str = "regular"
    lifecycle_status: str | None = None


@dataclass(frozen=True)
class DeadlineControlEvent:
    """A control transition executed by the pending-deadline timer."""

    timestamp_ms: float
    action: ControlAction
    pressed: bool
    scheduler_reason: str
    pending_due_ms: float
    timer_lateness_ms: float


class StateConditionedRuntimeEngine:
    """Run v3 by feeding the model the physical state active at each step."""

    def __init__(
        self,
        *,
        model: SwitchModel,
        preprocess_config: PreprocessConfig,
        executor: Executor,
        config: StateConditionedRuntimeConfig,
        initial_pressed: bool = False,
        scheduler: (
            MultiHorizonSwitchScheduler
            | ConsensusDenseHorizonScheduler
            | None
        ) = None,
        transition_lifecycle: ConsensusTransitionLifecycle | None = None,
        timer_factory: TimerFactory = Timer,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        config.validate()
        if config.execute_pending_at_due and scheduler is None:
            raise ValueError("execute_pending_at_due requires a scheduler")
        if transition_lifecycle is not None and not isinstance(
            scheduler, ConsensusDenseHorizonScheduler
        ):
            raise ValueError(
                "transition_lifecycle requires ConsensusDenseHorizonScheduler"
            )
        if scheduler is not None:
            scheduler_config = scheduler.config
            if not math.isclose(
                scheduler_config.control_horizon_ms,
                config.prediction_horizon_ms,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "scheduler control horizon must match runtime prediction horizon"
                )
            if not math.isclose(
                scheduler_config.threshold,
                config.switch_threshold,
                abs_tol=1e-9,
            ):
                raise ValueError("scheduler threshold must match runtime switch threshold")
        self.model = model
        self.preprocess_config = preprocess_config
        self.executor = executor
        self.config = config
        self.scheduler = scheduler
        self.transition_lifecycle = transition_lifecycle
        self.pressed = bool(initial_pressed)
        self._last_direct_switch_ms: float | None = None
        self._buffer = TemporalFrameBuffer(config.history_ms)
        self._interval_ms = 1000.0 / config.target_fps
        self._next_regular_due_ms: float | None = None
        self._timer_factory = timer_factory
        self._clock = clock
        self._lock = RLock()
        self._pending_timer: TimerHandle | None = None
        self._pending_timer_due_ms: float | None = None
        self._pending_timer_wall_due: float | None = None
        self._pending_timer_generation = 0
        self._deadline_events: list[DeadlineControlEvent] = []

    def drain_deadline_events(self) -> list[DeadlineControlEvent]:
        with self._lock:
            events = list(self._deadline_events)
            self._deadline_events.clear()
            return events

    def _cancel_pending_timer_locked(self) -> None:
        self._pending_timer_generation += 1
        timer = self._pending_timer
        self._pending_timer = None
        self._pending_timer_due_ms = None
        self._pending_timer_wall_due = None
        if timer is not None:
            timer.cancel()

    def _schedule_pending_timer_locked(
        self,
        *,
        frame_timestamp_ms: float,
        pending_due_ms: float,
        processing_elapsed_ms: float,
    ) -> None:
        if not self.config.execute_pending_at_due:
            return
        self._cancel_pending_timer_locked()
        self._pending_timer_generation += 1
        generation = self._pending_timer_generation
        remaining_ms = max(
            0.0,
            pending_due_ms - frame_timestamp_ms - processing_elapsed_ms,
        )
        wall_due = self._clock() + remaining_ms / 1000.0

        def fire() -> None:
            self._execute_pending_timer(generation, pending_due_ms, wall_due)

        timer = self._timer_factory(remaining_ms / 1000.0, fire)
        if hasattr(timer, "daemon"):
            setattr(timer, "daemon", True)
        self._pending_timer = timer
        self._pending_timer_due_ms = pending_due_ms
        self._pending_timer_wall_due = wall_due
        timer.start()

    def _execute_pending_timer(
        self,
        generation: int,
        pending_due_ms: float,
        wall_due: float,
    ) -> None:
        with self._lock:
            if generation != self._pending_timer_generation:
                return
            if self.scheduler is None or not self.config.execute_pending_at_due:
                return
            self._pending_timer = None
            self._pending_timer_due_ms = None
            self._pending_timer_wall_due = None
            execution = self.scheduler.execute_pending_if_due(
                timestamp_ms=pending_due_ms,
                current_pressed=self.pressed,
            )
            if execution is None:
                return

            before = self.pressed
            self.pressed = not self.pressed
            if self.transition_lifecycle is not None:
                self.transition_lifecycle.begin(
                    executed_at_ms=pending_due_ms,
                    previous_pressed=before,
                    current_pressed=self.pressed,
                    evidence_lead_ms=float(execution.delay_ms),
                )
            action = ControlAction.PRESS if self.pressed else ControlAction.RELEASE
            self.executor.set_pressed(self.pressed)
            lateness_ms = max(0.0, (self._clock() - wall_due) * 1000.0)
            deadline_reason = (
                "consensus_execute_deadline"
                if self.transition_lifecycle is not None
                else "pending_execute_deadline"
            )
            self._deadline_events.append(
                DeadlineControlEvent(
                    timestamp_ms=pending_due_ms,
                    action=action,
                    pressed=self.pressed,
                    scheduler_reason=deadline_reason,
                    pending_due_ms=pending_due_ms,
                    timer_lateness_ms=lateness_ms,
                )
            )

    def ingest(self, frame: Frame) -> StateConditionedRuntimeStep | None:
        with self._lock:
            ingest_started = self._clock()
            self._buffer.add(frame)
            if self._next_regular_due_ms is None:
                self._next_regular_due_ms = (
                    frame.timestamp_ms + self.config.history_ms
                )

            regular_due = frame.timestamp_ms + 1e-6 >= self._next_regular_due_ms
            pending_monitor = bool(
                self.config.execute_pending_at_due
                and self.scheduler is not None
                and self.scheduler.pending_due_ms is not None
            )
            if not regular_due and not pending_monitor:
                return None

            if regular_due:
                while self._next_regular_due_ms <= frame.timestamp_ms + 1e-6:
                    self._next_regular_due_ms += self._interval_ms

            if regular_due and pending_monitor:
                inference_trigger = "regular+pending_monitor"
            elif pending_monitor:
                inference_trigger = "pending_monitor"
            else:
                inference_trigger = "regular"

            temporal = self._buffer.stack_at(
                frame.timestamp_ms,
                self.config.frame_offsets_ms,
                self.preprocess_config,
            )
            if temporal is None:
                return None

            before = self.pressed
            started = time.perf_counter()
            lifecycle_status: str | None = None
            if self.scheduler is None:
                probability = float(self.model.predict_switch(temporal.input, before))
                probabilities = (probability,)
            else:
                probabilities = tuple(
                    float(value)
                    for value in self.model.predict_switch_all(temporal.input, before)
                )
                control_index = self.scheduler.config.horizons_ms.index(
                    self.scheduler.config.control_horizon_ms
                )
                if self.transition_lifecycle is not None and self.transition_lifecycle.active:
                    pending = self.transition_lifecycle.pending
                    assert pending is not None
                    previous_probabilities = tuple(
                        float(value)
                        for value in self.model.predict_switch_all(
                            temporal.input,
                            pending.previous_pressed,
                        )
                    )
                    lifecycle_status = self.transition_lifecycle.observe(
                        timestamp_ms=frame.timestamp_ms,
                        previous_state_h0_probability=previous_probabilities[
                            control_index
                        ],
                    )
                    if lifecycle_status in {"waiting", "confirmed"}:
                        masked = list(probabilities)
                        masked[control_index] = 0.0
                        probabilities = tuple(masked)
                probability = probabilities[control_index]
            inference_ms = (time.perf_counter() - started) * 1000.0

            if self.scheduler is None:
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError("model probability must be finite and in [0, 1]")
                wants_switch = probability >= self.config.switch_threshold
                inside_direct_hold = (
                    wants_switch
                    and self.config.min_state_hold_ms > 0.0
                    and self._last_direct_switch_ms is not None
                    and frame.timestamp_ms
                    < self._last_direct_switch_ms + self.config.min_state_hold_ms - 1e-6
                )
                should_switch = wants_switch and not inside_direct_hold
                if self.config.min_state_hold_ms > 0.0:
                    if inside_direct_hold:
                        scheduler_reason = "min_hold"
                    elif should_switch:
                        scheduler_reason = "primary"
                    else:
                        scheduler_reason = "hold"
                else:
                    scheduler_reason = None
                pending_due_ms = None
                pending_delay_ms = None
            else:
                decision = self.scheduler.update(
                    timestamp_ms=frame.timestamp_ms,
                    probabilities=probabilities,
                    current_pressed=before,
                )
                should_switch = decision.switch
                scheduler_reason = decision.reason
                pending_due_ms = decision.pending_due_ms
                pending_delay_ms = decision.pending_delay_ms

            if should_switch:
                self._cancel_pending_timer_locked()
                self.pressed = not before
                if self.scheduler is None:
                    self._last_direct_switch_ms = frame.timestamp_ms
                elif self.transition_lifecycle is not None:
                    if scheduler_reason == "consensus_execute":
                        evidence_lead_ms = float(
                            getattr(decision, "evidence_lead_ms", 0.0) or 0.0
                        )
                        self.transition_lifecycle.begin(
                            executed_at_ms=frame.timestamp_ms,
                            previous_pressed=before,
                            current_pressed=self.pressed,
                            evidence_lead_ms=evidence_lead_ms,
                        )
                    else:
                        self.transition_lifecycle.clear()
                action = ControlAction.PRESS if self.pressed else ControlAction.RELEASE
                self.executor.set_pressed(self.pressed)
            else:
                action = ControlAction.HOLD

            if self.scheduler is not None and self.config.execute_pending_at_due:
                active_due_ms = self.scheduler.pending_due_ms
                if active_due_ms is None:
                    self._cancel_pending_timer_locked()
                else:
                    processing_elapsed_ms = (self._clock() - ingest_started) * 1000.0
                    self._schedule_pending_timer_locked(
                        frame_timestamp_ms=frame.timestamp_ms,
                        pending_due_ms=active_due_ms,
                        processing_elapsed_ms=processing_elapsed_ms,
                    )

            return StateConditionedRuntimeStep(
                observation_timestamp_ms=frame.timestamp_ms,
                prediction_target_timestamp_ms=(
                    frame.timestamp_ms + self.config.prediction_horizon_ms
                ),
                input_frame_timestamps_ms=temporal.frame_timestamps_ms,
                input_frame_indices=temporal.frame_indices,
                probability=probability,
                action=action,
                pressed=self.pressed,
                inference_ms=inference_ms,
                probabilities=probabilities,
                scheduler_reason=scheduler_reason,
                pending_due_ms=pending_due_ms,
                pending_delay_ms=pending_delay_ms,
                inference_trigger=inference_trigger,
                lifecycle_status=lifecycle_status,
            )

    def shutdown(self) -> bool:
        with self._lock:
            self._cancel_pending_timer_locked()
            if self.scheduler is not None:
                self.scheduler.reset()
            if self.transition_lifecycle is not None:
                self.transition_lifecycle.clear()
            if not self.pressed:
                return False
            self.executor.set_pressed(False)
            self.pressed = False
            return True
