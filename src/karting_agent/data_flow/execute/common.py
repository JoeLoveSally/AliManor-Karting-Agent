"""Executor protocol for control side effects."""

from __future__ import annotations

from typing import Protocol


class Executor(Protocol):
    def set_pressed(self, pressed: bool) -> None:
        """Set the physical control state."""
