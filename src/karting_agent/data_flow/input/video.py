"""Sequential MP4 input for replay."""

from __future__ import annotations

import math
from pathlib import Path

import cv2

from karting_agent.data_flow.input.common import Frame


class VideoInput:
    """Decode a video sequentially without dataset manifests or random seeking."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open video: {self.path}")

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self._capture.release()
            raise RuntimeError(f"invalid video FPS: {self.path}")

        raw_frame_count = float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = (
            int(round(raw_frame_count))
            if math.isfinite(raw_frame_count) and raw_frame_count >= 0
            else 0
        )
        self._next_frame_index = 0
        self._closed = False

    @property
    def duration_ms(self) -> float:
        return self.frame_count * 1000.0 / self.fps if self.frame_count else 0.0

    def read(self) -> Frame | None:
        if self._closed:
            return None

        ok, image = self._capture.read()
        if not ok:
            return None

        frame_index = self._next_frame_index
        self._next_frame_index += 1
        return Frame(
            image=image,
            frame_index=frame_index,
            timestamp_ms=frame_index * 1000.0 / self.fps,
        )

    def close(self) -> None:
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> "VideoInput":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
