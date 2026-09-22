"""Frame-driven V4-C4 current-action controller (no event-time head or timer).

Pending min-hold transitions execute at the next observed frame at/after the due
instant. Planned deadlines and actual host command timestamps are kept separate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Protocol

import numpy as np

from karting_agent.data_flow.execute.common import Executor
from karting_agent.data_flow.input.common import Frame
from karting_agent.runtime.event_time_policy import (
    EventTimePolicyConfig,
    EventTimePolicyDecoder,
)
from karting_agent.vision.preprocess import PreprocessConfig
from karting_agent.vision.temporal import TemporalFrameBuffer


class ActionModel(Protocol):
    def predict_action(self, normalized_rgb_stack: np.ndarray) -> float: ...


@dataclass(frozen=True)
class ActionRuntimeConfig:
    target_fps: float = 30.0
    frame_offsets_ms: tuple[float, ...] = (-200.0, -150.0, -100.0, -50.0, 0.0)
    action_threshold: float = 0.5
    min_state_hold_ms: float = 100.0
    event_time_bin_ms: float = 50.0
    no_event_class: int = 6

    def validate(self) -> None:
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target_fps must be finite and positive")
        if (
            len(self.frame_offsets_ms) != 5
            or tuple(sorted(self.frame_offsets_ms)) != self.frame_offsets_ms
        ):
            raise ValueError("expected five ascending frame offsets")
        if self.frame_offsets_ms[-1] != 0 or self.frame_offsets_ms[0] >= 0:
            raise ValueError("offsets must start in the past and end at now")
        if self.no_event_class < 1:
            raise ValueError("no_event_class must be >= 1")
        self.decoder_config().validate()

    def decoder_config(self) -> EventTimePolicyConfig:
        return EventTimePolicyConfig(
            bin_ms=self.event_time_bin_ms,
            event_bins=self.no_event_class,
            no_event_class=self.no_event_class,
            action_threshold=self.action_threshold,
            min_state_hold_ms=self.min_state_hold_ms,
        )


class EventTimeActionEngine:
    def __init__(
        self,
        *,
        model: ActionModel,
        executor: Executor,
        preprocess_config: PreprocessConfig,
        config: ActionRuntimeConfig = ActionRuntimeConfig(),
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        config.validate()
        self.model = model
        self.executor = executor
        self.preprocess_config = preprocess_config
        self.config = config
        self.clock = clock
        self.decoder = EventTimePolicyDecoder(config.decoder_config())
        self.buffer = TemporalFrameBuffer(-config.frame_offsets_ms[0])
        self.pressed = False
        self.stopped = False
        self._next_due_ms: float | None = None

    def _send(
        self,
        *,
        pressed: bool,
        frame: Frame,
        reason: str,
        due_at_ms: float | None = None,
    ) -> dict[str, object]:
        if pressed == self.pressed:
            raise RuntimeError("attempted a redundant control transition")
        # Host command timing does not measure when Android physically applies touch.
        start_ms = self.clock() * 1000.0
        self.executor.set_pressed(pressed)
        returned_ms = self.clock() * 1000.0
        self.pressed = pressed
        return {
            "pressed": pressed,
            "reason": reason,
            "observation_timestamp_ms": frame.timestamp_ms,
            "source_frame_index": frame.frame_index,
            "pending_due_at_ms": due_at_ms,
            "host_command_started_monotonic_ms": start_ms,
            "host_command_returned_monotonic_ms": returned_ms,
        }

    def ingest(self, frame: Frame) -> dict[str, object] | None:
        if self.stopped:
            raise RuntimeError("cannot ingest after shutdown")
        self.buffer.add(frame)
        if self._next_due_ms is None:
            self._next_due_ms = frame.timestamp_ms - self.config.frame_offsets_ms[0]
        regular_due = frame.timestamp_ms + 1e-6 >= self._next_due_ms
        pending_due = self.decoder.pending_due_ms
        pending_ready = (
            pending_due is not None and frame.timestamp_ms + 1e-6 >= pending_due
        )
        if not regular_due and not pending_ready:
            return None
        temporal = self.buffer.stack_at(
            frame.timestamp_ms, self.config.frame_offsets_ms, self.preprocess_config
        )
        if temporal is None:
            return None
        if regular_due:
            while self._next_due_ms <= frame.timestamp_ms + 1e-6:
                self._next_due_ms += 1000.0 / self.config.target_fps
        # Preserve offline replay ordering: consume pending hold before processing
        # the current observation, but use actual observation time for the NEXT
        # minimum hold; a delayed frame must not shorten the real hold interval.
        events: list[dict[str, object]] = []
        if pending_ready:
            pending = self.decoder.execute_pending_if_due(
                timestamp_ms=frame.timestamp_ms,
                current_pressed=self.pressed,
                execution_timestamp_ms=frame.timestamp_ms,
            )
            if pending is not None:
                events.append(
                    self._send(
                        pressed=pending.state_after,
                        frame=frame,
                        reason="pending_execute",
                        due_at_ms=pending.due_at_ms,
                    )
                )
        started = self.clock()
        probability = float(self.model.predict_action(temporal.input))
        inference_ms = (self.clock() - started) * 1000.0
        decision = self.decoder.update(
            timestamp_ms=frame.timestamp_ms,
            current_pressed=self.pressed,
            current_action_probability=probability,
            event_class=self.config.no_event_class,
        )
        if decision.switch:
            events.append(
                self._send(
                    pressed=decision.state_after,
                    frame=frame,
                    reason=decision.reason,
                )
            )
        return {
            "observation_timestamp_ms": frame.timestamp_ms,
            "source_frame_index": frame.frame_index,
            "history_frame_indices": temporal.frame_indices,
            "history_timestamps_ms": temporal.frame_timestamps_ms,
            "action_probability": probability,
            "desired_pressed": decision.desired_pressed,
            "pressed": self.pressed,
            "decoder_reason": decision.reason,
            "pending_due_ms": self.decoder.pending_due_ms,
            "inference_ms": inference_ms,
            "events": events,
        }

    def shutdown(self) -> bool:
        if self.stopped:
            return False
        self.stopped = True
        self.decoder.reset()
        self.buffer.clear()
        if not self.pressed:
            return False
        # MockExecutor is local only: dry-run cleanup never sends Android touch.
        self.executor.set_pressed(False)
        self.pressed = False
        return True
