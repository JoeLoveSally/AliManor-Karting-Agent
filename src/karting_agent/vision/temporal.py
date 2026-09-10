"""Causal temporal frame selection for runtime inference."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from karting_agent.data_flow.input.common import Frame
from karting_agent.vision.preprocess import PreprocessConfig, stack_frames


@dataclass(frozen=True)
class TemporalStack:
    input: np.ndarray
    frame_timestamps_ms: tuple[float, ...]
    frame_indices: tuple[int, ...]


class TemporalFrameBuffer:
    """Keep recent frames and build history using only frames already observed."""

    def __init__(self, history_ms: float, *, retention_margin_ms: float = 100.0) -> None:
        if history_ms < 0:
            raise ValueError("history_ms must be >= 0")
        if retention_margin_ms < 0:
            raise ValueError("retention_margin_ms must be >= 0")
        self.history_ms = float(history_ms)
        self.retention_margin_ms = float(retention_margin_ms)
        self._frames: deque[Frame] = deque()

    def clear(self) -> None:
        self._frames.clear()

    def add(self, frame: Frame) -> None:
        if self._frames and frame.timestamp_ms < self._frames[-1].timestamp_ms:
            raise ValueError("frame timestamps must be non-decreasing")
        self._frames.append(frame)
        cutoff_ms = frame.timestamp_ms - self.history_ms - self.retention_margin_ms
        while len(self._frames) > 1 and self._frames[1].timestamp_ms <= cutoff_ms:
            self._frames.popleft()

    def _latest_at_or_before(self, timestamp_ms: float) -> Frame | None:
        for frame in reversed(self._frames):
            if frame.timestamp_ms <= timestamp_ms + 1e-6:
                return frame
        return None

    def stack_at(
        self,
        timestamp_ms: float,
        offsets_ms: Sequence[float],
        preprocess_config: PreprocessConfig,
    ) -> TemporalStack | None:
        if not offsets_ms:
            raise ValueError("offsets_ms must not be empty")
        values = tuple(float(value) for value in offsets_ms)
        if tuple(sorted(values)) != values:
            raise ValueError("offsets_ms must be sorted in ascending order")
        if values[-1] > 1e-6:
            raise ValueError("offsets_ms must not look into the future")

        selected: list[Frame] = []
        for offset_ms in values:
            frame = self._latest_at_or_before(timestamp_ms + offset_ms)
            if frame is None:
                return None
            selected.append(frame)

        return TemporalStack(
            input=stack_frames([frame.image for frame in selected], preprocess_config),
            frame_timestamps_ms=tuple(frame.timestamp_ms for frame in selected),
            frame_indices=tuple(frame.frame_index for frame in selected),
        )
