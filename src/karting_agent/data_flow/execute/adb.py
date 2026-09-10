"""ADB touch executor for development-only closed-loop POC."""

from __future__ import annotations

import time

from karting_agent.data_flow.adb import AdbClient


class AdbExecutor:
    """Translate controller state changes into Android DOWN / UP events."""

    def __init__(self, client: AdbClient, *, x: int, y: int) -> None:
        if x < 0 or y < 0:
            raise ValueError("ADB touch coordinates must be >= 0")
        self.client = client
        self.x = int(x)
        self.y = int(y)
        self.pressed = False
        self.execute_latencies_ms: list[float] = []
        self.last_execute_ms = 0.0

    def set_pressed(self, pressed: bool) -> None:
        pressed = bool(pressed)
        if pressed == self.pressed:
            return

        event = "DOWN" if pressed else "UP"
        started = time.perf_counter()
        self.client.run(
            "shell",
            "input",
            "motionevent",
            event,
            str(self.x),
            str(self.y),
        )
        self.last_execute_ms = (time.perf_counter() - started) * 1000.0
        self.execute_latencies_ms.append(self.last_execute_ms)
        self.pressed = pressed
