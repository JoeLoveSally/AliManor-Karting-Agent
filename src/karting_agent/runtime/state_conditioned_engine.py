"""State-conditioned KEEP/SWITCH runtime for model v3."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol

import numpy as np

from karting_agent.control.controller import ControlAction
from karting_agent.data_flow.execute.common import Executor
from karting_agent.data_flow.input.common import Frame
from karting_agent.runtime.engine import RuntimeStep
from karting_agent.vision.preprocess import PreprocessConfig
from karting_agent.vision.temporal import TemporalFrameBuffer


class SwitchModel(Protocol):
    def predict_switch(self, inputs: np.ndarray, current_pressed: bool) -> float:
        """Return probability that the current physical state should be flipped."""


@dataclass(frozen=True)
class StateConditionedRuntimeConfig:
    target_fps: float
    frame_offsets_ms: tuple[float, ...]
    prediction_horizon_ms: float
    switch_threshold: float = 0.6

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

    @property
    def history_ms(self) -> float:
        return max(0.0, -min(self.frame_offsets_ms))


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
    ) -> None:
        config.validate()
        self.model = model
        self.preprocess_config = preprocess_config
        self.executor = executor
        self.config = config
        self.pressed = bool(initial_pressed)
        self._buffer = TemporalFrameBuffer(config.history_ms)
        self._interval_ms = 1000.0 / config.target_fps
        self._next_due_ms: float | None = None

    def ingest(self, frame: Frame) -> RuntimeStep | None:
        self._buffer.add(frame)
        if self._next_due_ms is None:
            self._next_due_ms = frame.timestamp_ms + self.config.history_ms
        if frame.timestamp_ms + 1e-6 < self._next_due_ms:
            return None

        while self._next_due_ms <= frame.timestamp_ms + 1e-6:
            self._next_due_ms += self._interval_ms

        temporal = self._buffer.stack_at(
            frame.timestamp_ms,
            self.config.frame_offsets_ms,
            self.preprocess_config,
        )
        if temporal is None:
            return None

        before = self.pressed
        started = time.perf_counter()
        probability = float(self.model.predict_switch(temporal.input, before))
        inference_ms = (time.perf_counter() - started) * 1000.0
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("model probability must be finite and in [0, 1]")

        if probability >= self.config.switch_threshold:
            self.pressed = not before
            action = ControlAction.PRESS if self.pressed else ControlAction.RELEASE
            self.executor.set_pressed(self.pressed)
        else:
            action = ControlAction.HOLD

        return RuntimeStep(
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
        )

    def shutdown(self) -> bool:
        if not self.pressed:
            return False
        self.executor.set_pressed(False)
        self.pressed = False
        return True
