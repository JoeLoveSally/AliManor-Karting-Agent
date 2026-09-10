"""ADB screenshot input for development-only closed-loop POC."""

from __future__ import annotations

import time

import cv2
import numpy as np

from karting_agent.data_flow.adb import AdbClient
from karting_agent.data_flow.input.common import Frame


class AdbInput:
    """Capture Android screenshots one-by-one through ``adb exec-out screencap``."""

    def __init__(self, client: AdbClient) -> None:
        self.client = client
        self._origin = time.perf_counter()
        self._frame_index = 0
        self.capture_latencies_ms: list[float] = []
        self.last_capture_ms = 0.0

    def read(self) -> Frame:
        started = time.perf_counter()
        payload = self.client.run("exec-out", "screencap", "-p")
        finished = time.perf_counter()
        self.last_capture_ms = (finished - started) * 1000.0
        self.capture_latencies_ms.append(self.last_capture_ms)

        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("ADB screencap returned an invalid PNG image")

        frame = Frame(
            image=image,
            frame_index=self._frame_index,
            timestamp_ms=(finished - self._origin) * 1000.0,
        )
        self._frame_index += 1
        return frame

    def close(self) -> None:
        return None
