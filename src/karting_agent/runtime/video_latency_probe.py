"""Image-only marker detector for an ADB video freshness probe.

The module has no Android dependencies and never stores captured screen content.
A marker is a large central-screen change across consecutive decoded frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def changed_pixel_fraction(
    baseline: np.ndarray,
    observed: np.ndarray,
    *,
    pixel_delta: int = 32,
) -> float:
    """Return fraction of central pixels with a sufficiently large BGR change.

    Ignore the outer 10% on each edge: clock, notification/status and navigation
    icons should not by themselves constitute a screen-change marker.
    """
    if baseline.shape != observed.shape or baseline.ndim != 3 or baseline.shape[2] != 3:
        raise ValueError("baseline and observed must be identically shaped HxWx3 images")
    if baseline.dtype != np.uint8 or observed.dtype != np.uint8:
        raise ValueError("baseline and observed must contain uint8 BGR pixels")
    if not 1 <= pixel_delta <= 255:
        raise ValueError("pixel_delta must be within [1, 255]")
    height, width = baseline.shape[:2]
    y0, y1 = height // 10, height - height // 10
    x0, x1 = width // 10, width - width // 10
    if y1 <= y0 or x1 <= x0:
        raise ValueError("image is too small to measure a central crop")
    first = baseline[y0:y1, x0:x1].astype(np.int16)
    second = observed[y0:y1, x0:x1].astype(np.int16)
    return float(np.mean(np.max(np.abs(first - second), axis=2) >= pixel_delta))


@dataclass
class ScreenChangeGate:
    baseline: np.ndarray
    changed_fraction_threshold: float = 0.25
    pixel_delta: int = 32
    consecutive_frames: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.changed_fraction_threshold <= 1.0:
            raise ValueError("changed_fraction_threshold must be within (0, 1]")
        if self.consecutive_frames < 1:
            raise ValueError("consecutive_frames must be >= 1")
        self.baseline = self.baseline.copy()
        self._candidate_time_ms: float | None = None
        self._candidate_frame_index: int | None = None
        self._consecutive = 0

    def observe(
        self,
        *,
        image: np.ndarray,
        observed_monotonic_ms: float,
        frame_index: int,
    ) -> tuple[float, int, float] | None:
        """Return first marker (time, frame index, changed fraction) once confirmed.

        observed_monotonic_ms is the host's monotonic read time, not an Android
        capture timestamp. The onset is deliberately the FIRST qualifying frame,
        not the later frame that confirms the marker.
        """
        if not math.isfinite(observed_monotonic_ms):
            raise ValueError("observed_monotonic_ms must be finite")
        fraction = changed_pixel_fraction(
            self.baseline, image, pixel_delta=self.pixel_delta
        )
        if fraction < self.changed_fraction_threshold:
            self._candidate_time_ms = None
            self._candidate_frame_index = None
            self._consecutive = 0
            return None
        if self._consecutive == 0:
            self._candidate_time_ms = float(observed_monotonic_ms)
            self._candidate_frame_index = int(frame_index)
        self._consecutive += 1
        if self._consecutive < self.consecutive_frames:
            return None
        assert self._candidate_time_ms is not None
        assert self._candidate_frame_index is not None
        return (self._candidate_time_ms, self._candidate_frame_index, fraction)
