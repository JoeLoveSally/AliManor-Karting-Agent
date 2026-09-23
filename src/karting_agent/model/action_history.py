"""Causal *command/label* history features shared by training and offline inference.

A command timestamp is not an observed in-game steering-effect timestamp.
Only events already known at inference time may be supplied at runtime.
"""

from __future__ import annotations

from bisect import bisect_right
import math
from typing import Sequence

import numpy as np


def history_feature_width(frame_stack: int) -> int:
    if frame_stack < 1:
        raise ValueError("frame_stack must be positive")
    # Per frame: state, capped age / cap, age-known flag; plus current age and flag.
    return 3 * frame_stack + 2


def encode_action_history(
    frame_timestamps_ms: Sequence[float],
    observation_timestamp_ms: float,
    *,
    initial_pressed: bool,
    transitions: Sequence[tuple[float, bool]],
    current_pressed: bool,
    max_age_ms: float = 500.0,
) -> np.ndarray:
    """Encode past states without consulting transitions after the observation.

    ``transitions`` contains actual state CHANGES, excluding the initial state.
    Equal-time events are considered visible at that timestamp. Call this
    BEFORE appending an action that is chosen at the current observation.
    """
    frame_times = tuple(float(value) for value in frame_timestamps_ms)
    observed = float(observation_timestamp_ms)
    if not frame_times or not math.isfinite(observed):
        raise ValueError("finite observation and at least one frame required")
    if not math.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError("max_age_ms must be finite and positive")
    if any(not math.isfinite(value) for value in frame_times):
        raise ValueError("frame timestamps must be finite")
    if tuple(sorted(frame_times)) != frame_times or frame_times[-1] > observed + 1e-6:
        raise ValueError("frames must be ordered and no later than observation")

    times = [float(timestamp) for timestamp, _ in transitions]
    if any(not math.isfinite(value) for value in times):
        raise ValueError("transition timestamps must be finite")
    if any(left >= right for left, right in zip(times, times[1:])):
        raise ValueError("transition timestamps must be strictly increasing")
    previous_state = bool(initial_pressed)
    for _, state in transitions:
        if bool(state) == previous_state:
            raise ValueError("transitions must alternate physical states")
        previous_state = bool(state)

    def at(timestamp: float) -> tuple[bool, float, float]:
        index = bisect_right(times, timestamp) - 1
        if index < 0:
            return bool(initial_pressed), 1.0, 0.0
        age = min(max(timestamp - times[index], 0.0), max_age_ms) / max_age_ms
        return bool(transitions[index][1]), age, 1.0

    observed_state, observed_age, observed_known = at(observed)
    if observed_state != bool(current_pressed):
        raise ValueError("current_pressed disagrees with causal action history")
    encoded: list[float] = []
    for timestamp in frame_times:
        pressed, age, known = at(timestamp)
        encoded.extend((float(pressed), age, known))
    encoded.extend((observed_age, observed_known))
    return np.asarray(encoded, dtype=np.float32)
