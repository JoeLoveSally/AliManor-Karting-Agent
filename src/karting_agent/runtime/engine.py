"""Core single-input runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol

import numpy as np

from karting_agent.control.controller import ControlAction, HysteresisController
from karting_agent.data_flow.execute.common import Executor
from karting_agent.data_flow.input.common import Frame
from karting_agent.vision.preprocess import PreprocessConfig
from karting_agent.vision.temporal import TemporalFrameBuffer


class ProbabilityModel(Protocol):
    def predict(self, inputs: np.ndarray) -> float:
        """Return PRESS probability for one temporal stack."""


@dataclass(frozen=True)
class RuntimeEngineConfig:
    target_fps: float
    frame_offsets_ms: tuple[float, ...]
    prediction_horizon_ms: float

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

    @property
    def history_ms(self) -> float:
        return max(0.0, -min(self.frame_offsets_ms))


@dataclass(frozen=True)
class RuntimeStep:
    observation_timestamp_ms: float
    prediction_target_timestamp_ms: float
    input_frame_timestamps_ms: tuple[float, ...]
    input_frame_indices: tuple[int, ...]
    probability: float
    action: ControlAction
    pressed: bool
    inference_ms: float


class RuntimeEngine:
    """Run causal temporal inference at a capped target frequency."""

    def __init__(
        self,
        *,
        model: ProbabilityModel,
        preprocess_config: PreprocessConfig,
        controller: HysteresisController,
        executor: Executor,
        config: RuntimeEngineConfig,
    ) -> None:
        config.validate()
        self.model = model
        self.preprocess_config = preprocess_config
        self.controller = controller
        self.executor = executor
        self.config = config
        self._buffer = TemporalFrameBuffer(config.history_ms)
        self._interval_ms = 1000.0 / config.target_fps
        self._next_due_ms: float | None = None

    def ingest(self, frame: Frame) -> RuntimeStep | None:
        self._buffer.add(frame)
        if self._next_due_ms is None:
            self._next_due_ms = frame.timestamp_ms + self.config.history_ms
        if frame.timestamp_ms + 1e-6 < self._next_due_ms:
            return None

        # Do not replay missed ticks in a burst. Runtime always uses the newest
        # frame available when the next inference slot becomes due.
        while self._next_due_ms <= frame.timestamp_ms + 1e-6:
            self._next_due_ms += self._interval_ms

        temporal = self._buffer.stack_at(
            frame.timestamp_ms,
            self.config.frame_offsets_ms,
            self.preprocess_config,
        )
        if temporal is None:
            return None

        started = time.perf_counter()
        probability = float(self.model.predict(temporal.input))
        inference_ms = (time.perf_counter() - started) * 1000.0
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("model probability must be finite and in [0, 1]")

        action = self.controller.update(probability)
        if action is ControlAction.PRESS:
            self.executor.set_pressed(True)
        elif action is ControlAction.RELEASE:
            self.executor.set_pressed(False)

        return RuntimeStep(
            observation_timestamp_ms=frame.timestamp_ms,
            prediction_target_timestamp_ms=(
                frame.timestamp_ms + self.config.prediction_horizon_ms
            ),
            input_frame_timestamps_ms=temporal.frame_timestamps_ms,
            input_frame_indices=temporal.frame_indices,
            probability=probability,
            action=action,
            pressed=self.controller.pressed,
            inference_ms=inference_ms,
        )

    def shutdown(self) -> bool:
        """Return whether a safety RELEASE side effect was required."""
        if not self.controller.pressed:
            return False
        self.executor.set_pressed(False)
        self.controller.reset(pressed=False)
        return True
