"""Closed-loop replay helpers for an immediate H0 KEEP/SWITCH policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence


@dataclass(frozen=True)
class H0Decision:
    timestamp_ms: float
    state_before: bool
    state_after: bool
    probability: float
    switched: bool
    blocked_by_min_hold: bool


def simulate_h0_closed_loop(
    timestamps_ms: Sequence[float],
    switch_if_release: Sequence[float],
    switch_if_press: Sequence[float],
    *,
    initial_pressed: bool,
    threshold: float = 0.6,
    min_state_hold_ms: float = 0.0,
) -> list[H0Decision]:
    """Replay H0 decisions against the controller's own physical action state.

    At each observation the probability source is selected from the model output
    conditioned on the *simulated* current state, not from the recorded expert
    state. A threshold crossing toggles the state immediately unless it falls
    inside ``min_state_hold_ms`` after the previous executed toggle.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if not math.isfinite(min_state_hold_ms) or min_state_hold_ms < 0.0:
        raise ValueError("min_state_hold_ms must be finite and >= 0")
    if not (
        len(timestamps_ms) == len(switch_if_release) == len(switch_if_press)
    ):
        raise ValueError("timestamps and probability sequences must have equal length")

    state = bool(initial_pressed)
    last_switch_ms: float | None = None
    last_timestamp_ms: float | None = None
    decisions: list[H0Decision] = []

    for raw_timestamp, raw_release, raw_press in zip(
        timestamps_ms,
        switch_if_release,
        switch_if_press,
        strict=True,
    ):
        timestamp_ms = float(raw_timestamp)
        release_probability = float(raw_release)
        press_probability = float(raw_press)
        if not math.isfinite(timestamp_ms):
            raise ValueError("timestamps must be finite")
        if last_timestamp_ms is not None and timestamp_ms < last_timestamp_ms - 1e-6:
            raise ValueError("timestamps must be nondecreasing")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (release_probability, press_probability)
        ):
            raise ValueError("switch probabilities must be finite and in [0, 1]")

        probability = press_probability if state else release_probability
        wants_switch = probability >= threshold
        inside_hold = (
            wants_switch
            and last_switch_ms is not None
            and timestamp_ms < last_switch_ms + min_state_hold_ms - 1e-6
        )
        before = state
        switched = wants_switch and not inside_hold
        if switched:
            state = not state
            last_switch_ms = timestamp_ms

        decisions.append(
            H0Decision(
                timestamp_ms=timestamp_ms,
                state_before=before,
                state_after=state,
                probability=probability,
                switched=switched,
                blocked_by_min_hold=bool(inside_hold),
            )
        )
        last_timestamp_ms = timestamp_ms

    return decisions
