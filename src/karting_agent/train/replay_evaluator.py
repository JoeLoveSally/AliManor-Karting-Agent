"""Evaluate replay JSON state sequences against recorded action labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from karting_agent.train.sequence_evaluator import SequencePoint

ReplayTimeline = Literal["target", "observation"]


def replay_points(
    steps: Sequence[Mapping[str, object]],
    *,
    video: str,
    timeline: ReplayTimeline,
) -> tuple[SequencePoint, ...]:
    """Convert recorded Runtime steps into sequence-evaluator points.

    ``target`` places each controller state at the model's future-action target
    timestamp. ``observation`` places it at the timestamp where Runtime emitted
    the decision, before any real executor/hardware latency.
    """
    if timeline not in ("target", "observation"):
        raise ValueError(f"unsupported replay timeline: {timeline}")

    timestamp_key = (
        "prediction_target_timestamp_ms"
        if timeline == "target"
        else "observation_timestamp_ms"
    )
    points: list[SequencePoint] = []
    previous_timestamp: float | None = None
    for step in steps:
        if timestamp_key not in step or "pressed" not in step:
            raise ValueError(f"replay step missing {timestamp_key!r} or 'pressed'")
        timestamp_ms = float(step[timestamp_key])
        if previous_timestamp is not None and timestamp_ms <= previous_timestamp:
            raise ValueError("replay timestamps must be strictly increasing")
        previous_timestamp = timestamp_ms
        pressed = bool(step["pressed"])
        points.append(
            SequencePoint(
                video=video,
                timestamp_ms=timestamp_ms,
                probability=1.0 if pressed else 0.0,
            )
        )
    return tuple(points)
