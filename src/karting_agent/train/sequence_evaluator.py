"""Sequence-level metrics for temporal PRESS/RELEASE predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SequencePoint:
    video: str
    timestamp_ms: float
    probability: float


@dataclass(frozen=True)
class Transition:
    video: str
    timestamp_ms: float
    pressed: bool


@dataclass(frozen=True)
class ReleaseSegment:
    video: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def start_transition(self) -> Transition:
        return Transition(self.video, self.start_ms, False)

    @property
    def end_transition(self) -> Transition:
        return Transition(self.video, self.end_ms, True)


@dataclass(frozen=True)
class TransitionMatch:
    expected: Transition
    predicted: Transition

    @property
    def error_ms(self) -> float:
        return self.predicted.timestamp_ms - self.expected.timestamp_ms


@dataclass(frozen=True)
class SequenceEvaluation:
    samples: int
    coverage_start_ms: float
    coverage_end_ms: float
    threshold: float
    tolerance_ms: float
    expected_transitions: tuple[Transition, ...]
    predicted_transitions: tuple[Transition, ...]
    matches: tuple[TransitionMatch, ...]
    release_segments: tuple[ReleaseSegment, ...]
    release_detected: tuple[bool, ...]

    def summary(self) -> dict[str, object]:
        return _summarize_evaluation(self)


def transitions_from_points(
    points: Sequence[SequencePoint], threshold: float = 0.5
) -> tuple[Transition, ...]:
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0, 1)")
    if not points:
        return ()

    video = points[0].video
    previous_timestamp = float("-inf")
    for point in points:
        if point.video != video:
            raise ValueError("all sequence points must belong to one video")
        if point.timestamp_ms <= previous_timestamp:
            raise ValueError("sequence points must be strictly increasing by timestamp")
        previous_timestamp = point.timestamp_ms

    states = [point.probability >= threshold for point in points]
    return tuple(
        Transition(
            video=video,
            timestamp_ms=points[index].timestamp_ms,
            pressed=states[index],
        )
        for index in range(1, len(points))
        if states[index] != states[index - 1]
    )


def match_transitions(
    expected: Sequence[Transition],
    predicted: Sequence[Transition],
    tolerance_ms: float,
) -> tuple[TransitionMatch, ...]:
    """Greedily match same-direction transitions by nearest timing error."""
    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be >= 0")

    candidates: list[tuple[float, int, int]] = []
    for expected_index, expected_event in enumerate(expected):
        for predicted_index, predicted_event in enumerate(predicted):
            if expected_event.video != predicted_event.video:
                continue
            if expected_event.pressed != predicted_event.pressed:
                continue
            distance = abs(predicted_event.timestamp_ms - expected_event.timestamp_ms)
            if distance <= tolerance_ms:
                candidates.append((distance, expected_index, predicted_index))

    matched_expected: set[int] = set()
    matched_predicted: set[int] = set()
    matches: list[TransitionMatch] = []
    for _, expected_index, predicted_index in sorted(candidates):
        if expected_index in matched_expected or predicted_index in matched_predicted:
            continue
        matched_expected.add(expected_index)
        matched_predicted.add(predicted_index)
        matches.append(
            TransitionMatch(expected[expected_index], predicted[predicted_index])
        )

    return tuple(sorted(matches, key=lambda match: match.expected.timestamp_ms))


def evaluate_sequence(
    points: Sequence[SequencePoint],
    expected_transitions: Sequence[Transition],
    release_segments: Sequence[ReleaseSegment],
    *,
    threshold: float = 0.5,
    tolerance_ms: float = 100.0,
) -> SequenceEvaluation:
    if not points:
        raise ValueError("points must not be empty")

    coverage_start = points[0].timestamp_ms
    coverage_end = points[-1].timestamp_ms
    video = points[0].video
    if any(point.video != video for point in points):
        raise ValueError("all sequence points must belong to one video")

    expected = tuple(
        event
        for event in expected_transitions
        if event.video == video
        and coverage_start <= event.timestamp_ms <= coverage_end
    )
    predicted = transitions_from_points(points, threshold)
    matches = match_transitions(expected, predicted, tolerance_ms)
    matched_expected = {match.expected for match in matches}

    evaluable_segments = tuple(
        segment
        for segment in release_segments
        if segment.video == video
        and coverage_start <= segment.start_ms
        and segment.end_ms <= coverage_end
    )
    detected = tuple(
        segment.start_transition in matched_expected
        and segment.end_transition in matched_expected
        for segment in evaluable_segments
    )

    return SequenceEvaluation(
        samples=len(points),
        coverage_start_ms=coverage_start,
        coverage_end_ms=coverage_end,
        threshold=threshold,
        tolerance_ms=tolerance_ms,
        expected_transitions=expected,
        predicted_transitions=predicted,
        matches=matches,
        release_segments=evaluable_segments,
        release_detected=detected,
    )


def combine_evaluations(
    evaluations: Sequence[SequenceEvaluation],
) -> SequenceEvaluation:
    if not evaluations:
        raise ValueError("evaluations must not be empty")

    threshold = evaluations[0].threshold
    tolerance_ms = evaluations[0].tolerance_ms
    if any(item.threshold != threshold for item in evaluations):
        raise ValueError("all evaluations must use the same threshold")
    if any(item.tolerance_ms != tolerance_ms for item in evaluations):
        raise ValueError("all evaluations must use the same tolerance")

    return SequenceEvaluation(
        samples=sum(item.samples for item in evaluations),
        coverage_start_ms=min(item.coverage_start_ms for item in evaluations),
        coverage_end_ms=max(item.coverage_end_ms for item in evaluations),
        threshold=threshold,
        tolerance_ms=tolerance_ms,
        expected_transitions=tuple(
            event for item in evaluations for event in item.expected_transitions
        ),
        predicted_transitions=tuple(
            event for item in evaluations for event in item.predicted_transitions
        ),
        matches=tuple(match for item in evaluations for match in item.matches),
        release_segments=tuple(
            segment for item in evaluations for segment in item.release_segments
        ),
        release_detected=tuple(
            detected for item in evaluations for detected in item.release_detected
        ),
    )


def _transition_metrics(
    expected: Sequence[Transition],
    predicted: Sequence[Transition],
    matches: Sequence[TransitionMatch],
    pressed: bool | None,
) -> dict[str, float | int]:
    if pressed is None:
        filtered_expected = expected
        filtered_predicted = predicted
        filtered_matches = matches
    else:
        filtered_expected = [event for event in expected if event.pressed == pressed]
        filtered_predicted = [event for event in predicted if event.pressed == pressed]
        filtered_matches = [
            match for match in matches if match.expected.pressed == pressed
        ]

    true_positive = len(filtered_matches)
    false_positive = len(filtered_predicted) - true_positive
    false_negative = len(filtered_expected) - true_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "ground_truth": len(filtered_expected),
        "predicted": len(filtered_predicted),
        "matched": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _timing_metrics(
    matches: Sequence[TransitionMatch], pressed: bool
) -> dict[str, float | int]:
    errors = np.asarray(
        [match.error_ms for match in matches if match.expected.pressed == pressed],
        dtype=np.float64,
    )
    if errors.size == 0:
        return {
            "matched": 0,
            "mean_error_ms": 0.0,
            "mae_ms": 0.0,
            "p50_abs_ms": 0.0,
            "p95_abs_ms": 0.0,
            "max_abs_ms": 0.0,
        }

    absolute = np.abs(errors)
    return {
        "matched": int(errors.size),
        "mean_error_ms": float(errors.mean()),
        "mae_ms": float(absolute.mean()),
        "p50_abs_ms": float(np.percentile(absolute, 50)),
        "p95_abs_ms": float(np.percentile(absolute, 95)),
        "max_abs_ms": float(absolute.max()),
    }


def _recall_summary(
    segments: Sequence[ReleaseSegment],
    detected: Sequence[bool],
    predicate,
) -> dict[str, float | int]:
    selected = [
        flag
        for segment, flag in zip(segments, detected)
        if predicate(segment.duration_ms)
    ]
    count = len(selected)
    matched = sum(selected)
    return {
        "segments": count,
        "detected": matched,
        "recall": matched / count if count else 0.0,
    }


def _summarize_evaluation(evaluation: SequenceEvaluation) -> dict[str, object]:
    segments = evaluation.release_segments
    detected = evaluation.release_detected
    return {
        "samples": evaluation.samples,
        "coverage_start_ms": evaluation.coverage_start_ms,
        "coverage_end_ms": evaluation.coverage_end_ms,
        "threshold": evaluation.threshold,
        "transition_tolerance_ms": evaluation.tolerance_ms,
        "transition": {
            "all": _transition_metrics(
                evaluation.expected_transitions,
                evaluation.predicted_transitions,
                evaluation.matches,
                None,
            ),
            "press": _transition_metrics(
                evaluation.expected_transitions,
                evaluation.predicted_transitions,
                evaluation.matches,
                True,
            ),
            "release": _transition_metrics(
                evaluation.expected_transitions,
                evaluation.predicted_transitions,
                evaluation.matches,
                False,
            ),
        },
        "onset_timing_ms": {
            "press": _timing_metrics(evaluation.matches, True),
            "release": _timing_metrics(evaluation.matches, False),
        },
        "release_segment_recall": {
            "lt_100ms": _recall_summary(
                segments, detected, lambda value: value < 100.0
            ),
            "short_100_300ms": _recall_summary(
                segments, detected, lambda value: 100.0 <= value <= 300.0
            ),
            "100_200ms": _recall_summary(
                segments, detected, lambda value: 100.0 <= value < 200.0
            ),
            "200_300ms": _recall_summary(
                segments, detected, lambda value: 200.0 <= value <= 300.0
            ),
            "gt_300ms": _recall_summary(
                segments, detected, lambda value: value > 300.0
            ),
        },
    }
