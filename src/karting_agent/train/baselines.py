"""Simple label-only baselines for temporal control prediction."""

from __future__ import annotations

from collections.abc import Sequence

from karting_agent.train.dataset import DatasetSample
from karting_agent.train.labels.touch_marker import ActionEvent, action_at_timestamp
from karting_agent.train.sequence_evaluator import SequencePoint


def persistence_prediction(
    sample: DatasetSample,
    events: Sequence[ActionEvent],
) -> bool:
    """Predict the future action by copying the ground-truth action at input time t."""
    if not sample.input_timestamps_ms:
        raise ValueError("sample has no input timestamps")
    current_timestamp_ms = sample.input_timestamps_ms[-1]
    return action_at_timestamp(events, current_timestamp_ms)


def persistence_points(
    samples: Sequence[DatasetSample],
    events: Sequence[ActionEvent],
) -> tuple[SequencePoint, ...]:
    """Build target-time sequence points for the action(t) -> action(t+horizon) baseline."""
    return tuple(
        SequencePoint(
            video=sample.video,
            timestamp_ms=sample.target_timestamp_ms,
            probability=1.0 if persistence_prediction(sample, events) else 0.0,
        )
        for sample in samples
    )
