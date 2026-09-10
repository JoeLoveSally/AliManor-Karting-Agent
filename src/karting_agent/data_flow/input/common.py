"""Shared input types."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Frame:
    """One decoded input frame with a monotonic source timestamp."""

    image: np.ndarray
    frame_index: int
    timestamp_ms: float

    def __post_init__(self) -> None:
        if self.image is None or self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("image must be a BGR image with shape HxWx3")
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if not math.isfinite(self.timestamp_ms) or self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be finite and >= 0")
