"""Hysteresis-based PRESS/RELEASE control decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math


class ControlAction(str, Enum):
    PRESS = "PRESS"
    RELEASE = "RELEASE"
    HOLD = "HOLD"


@dataclass(frozen=True)
class HysteresisConfig:
    press_threshold: float = 0.7
    release_threshold: float = 0.3

    def validate(self) -> None:
        if not (0.0 <= self.release_threshold < self.press_threshold <= 1.0):
            raise ValueError(
                "thresholds must satisfy 0 <= release_threshold < "
                "press_threshold <= 1"
            )


class HysteresisController:
    """Convert PRESS probabilities into stateful control actions."""

    def __init__(
        self,
        config: HysteresisConfig | None = None,
        *,
        initial_pressed: bool = False,
    ) -> None:
        self.config = config or HysteresisConfig()
        self.config.validate()
        self._pressed = bool(initial_pressed)

    @property
    def pressed(self) -> bool:
        return self._pressed

    def reset(self, *, pressed: bool = False) -> None:
        self._pressed = bool(pressed)

    def update(self, probability: float) -> ControlAction:
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be finite and in [0, 1]")

        if self._pressed:
            if probability <= self.config.release_threshold:
                self._pressed = False
                return ControlAction.RELEASE
            return ControlAction.HOLD

        if probability >= self.config.press_threshold:
            self._pressed = True
            return ControlAction.PRESS
        return ControlAction.HOLD


def hysteresis_states(
    probabilities: Sequence[float],
    config: HysteresisConfig | None = None,
    *,
    initial_pressed: bool | None = None,
) -> tuple[bool, ...]:
    """Return controller states for an ordered probability sequence.

    Offline evaluation infers the first state from the midpoint between the
    thresholds when ``initial_pressed`` is omitted. Runtime should construct
    ``HysteresisController`` explicitly with its safety-required initial state.
    """
    if not probabilities:
        return ()

    resolved = config or HysteresisConfig()
    resolved.validate()
    values = [float(value) for value in probabilities]
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("probabilities must be finite and in [0, 1]")

    start = (
        values[0] >= (resolved.press_threshold + resolved.release_threshold) / 2.0
        if initial_pressed is None
        else bool(initial_pressed)
    )
    controller = HysteresisController(resolved, initial_pressed=start)
    states = [start]

    for probability in values[1:]:
        controller.update(probability)
        states.append(controller.pressed)
    return tuple(states)
