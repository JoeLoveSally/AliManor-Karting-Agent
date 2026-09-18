"""Helpers for comparing aligned live multi-horizon control runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LiveTransition:
    ordinal: int
    step_index: int
    timestamp_ms: float
    action: str
    pressed: bool
    source_frame: int
    reason: str | None
    probabilities: tuple[float, ...]


def extract_transitions(steps: Sequence[dict[str, object]]) -> list[LiveTransition]:
    """Return executed PRESS/RELEASE transitions in chronological order."""

    transitions: list[LiveTransition] = []
    previous_timestamp: float | None = None
    for step_index, step in enumerate(steps):
        timestamp_ms = float(step["observation_timestamp_ms"])
        if previous_timestamp is not None and timestamp_ms <= previous_timestamp:
            raise ValueError("runtime step timestamps must be strictly increasing")
        previous_timestamp = timestamp_ms

        action = str(step["action"])
        if action not in {"PRESS", "RELEASE"}:
            continue

        raw_indices = step.get("input_frame_indices")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError("transition step is missing input_frame_indices")
        raw_probabilities = step.get("probabilities")
        if not isinstance(raw_probabilities, list) or not raw_probabilities:
            raise ValueError("transition step is missing probabilities")

        transitions.append(
            LiveTransition(
                ordinal=len(transitions) + 1,
                step_index=step_index,
                timestamp_ms=timestamp_ms,
                action=action,
                pressed=bool(step["pressed"]),
                source_frame=int(raw_indices[-1]),
                reason=(
                    None
                    if step.get("scheduler_reason") is None
                    else str(step["scheduler_reason"])
                ),
                probabilities=tuple(float(value) for value in raw_probabilities),
            )
        )
    return transitions


def nearest_step(
    steps: Sequence[dict[str, object]],
    target_timestamp_ms: float,
) -> tuple[int, dict[str, object]]:
    """Return the runtime step nearest to a target timestamp."""

    if not steps:
        raise ValueError("run contains no runtime steps")

    best_index = 0
    best_distance = float("inf")
    for index, step in enumerate(steps):
        timestamp_ms = float(step["observation_timestamp_ms"])
        distance = abs(timestamp_ms - target_timestamp_ms)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index, steps[best_index]


def aligned_snapshots(
    steps: Sequence[dict[str, object]],
    *,
    anchor_timestamp_ms: float,
    offsets_ms: Sequence[float],
) -> list[dict[str, object]]:
    """Sample runtime rows nearest to fixed offsets from an anchor transition."""

    snapshots: list[dict[str, object]] = []
    for offset_ms in offsets_ms:
        target_timestamp_ms = anchor_timestamp_ms + float(offset_ms)
        step_index, step = nearest_step(steps, target_timestamp_ms)
        raw_indices = step.get("input_frame_indices")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError("runtime step is missing input_frame_indices")
        raw_probabilities = step.get("probabilities")
        if not isinstance(raw_probabilities, list) or not raw_probabilities:
            raise ValueError("runtime step is missing probabilities")
        actual_timestamp_ms = float(step["observation_timestamp_ms"])
        snapshots.append(
            {
                "requested_offset_ms": float(offset_ms),
                "actual_offset_ms": actual_timestamp_ms - anchor_timestamp_ms,
                "step_index": step_index,
                "timestamp_ms": actual_timestamp_ms,
                "source_frame": int(raw_indices[-1]),
                "action": str(step["action"]),
                "pressed": bool(step["pressed"]),
                "reason": step.get("scheduler_reason"),
                "probabilities": [float(value) for value in raw_probabilities],
            }
        )
    return snapshots
