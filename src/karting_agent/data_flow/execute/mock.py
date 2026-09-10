"""No-op executor that records requested control states."""

from __future__ import annotations


class MockExecutor:
    def __init__(self) -> None:
        self.pressed = False
        self.calls: list[bool] = []

    def set_pressed(self, pressed: bool) -> None:
        self.pressed = bool(pressed)
        self.calls.append(self.pressed)
