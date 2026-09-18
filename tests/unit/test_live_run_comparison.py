from __future__ import annotations

import pytest

from karting_agent.train.live_run_comparison import (
    aligned_snapshots,
    extract_transitions,
    nearest_step,
)


def _step(
    timestamp_ms: float,
    *,
    action: str = "HOLD",
    pressed: bool = False,
    frame: int = 0,
    probabilities: list[float] | None = None,
    reason: str = "hold",
) -> dict[str, object]:
    return {
        "observation_timestamp_ms": timestamp_ms,
        "action": action,
        "pressed": pressed,
        "input_frame_indices": [frame],
        "probabilities": probabilities or [0.1, 0.2, 0.3],
        "scheduler_reason": reason,
    }


def test_extract_transitions_keeps_executed_order() -> None:
    steps = [
        _step(0.0, frame=0),
        _step(
            100.0,
            action="PRESS",
            pressed=True,
            frame=5,
            probabilities=[0.1, 0.4, 0.9],
            reason="pending_execute",
        ),
        _step(150.0, pressed=True, frame=8),
        _step(
            300.0,
            action="RELEASE",
            pressed=False,
            frame=16,
            probabilities=[0.2, 0.7, 0.8],
            reason="primary",
        ),
    ]

    transitions = extract_transitions(steps)

    assert [item.ordinal for item in transitions] == [1, 2]
    assert [item.action for item in transitions] == ["PRESS", "RELEASE"]
    assert transitions[0].source_frame == 5
    assert transitions[0].reason == "pending_execute"
    assert transitions[1].probabilities == pytest.approx((0.2, 0.7, 0.8))


def test_nearest_step_prefers_first_equal_distance() -> None:
    steps = [
        _step(0.0, frame=0),
        _step(100.0, frame=5),
        _step(200.0, frame=10),
    ]

    index, step = nearest_step(steps, 150.0)

    assert index == 1
    assert step["input_frame_indices"] == [5]


def test_aligned_snapshots_use_relative_anchor_offsets() -> None:
    steps = [
        _step(900.0, frame=9),
        _step(1000.0, action="PRESS", pressed=True, frame=10),
        _step(1060.0, pressed=True, frame=11),
        _step(1110.0, pressed=True, frame=12),
    ]

    snapshots = aligned_snapshots(
        steps,
        anchor_timestamp_ms=1000.0,
        offsets_ms=(0.0, 50.0, 100.0),
    )

    assert [item["source_frame"] for item in snapshots] == [10, 11, 12]
    assert [item["actual_offset_ms"] for item in snapshots] == pytest.approx(
        [0.0, 60.0, 110.0]
    )


def test_extract_transitions_rejects_nonmonotonic_timestamps() -> None:
    steps = [
        _step(100.0),
        _step(90.0),
    ]

    with pytest.raises(ValueError, match="strictly increasing"):
        extract_transitions(steps)
