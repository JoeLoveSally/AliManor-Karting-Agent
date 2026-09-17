"""Helpers for diagnosing discrete multi-horizon switch patterns."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence


def switch_pattern(current_pressed: bool, future_actions: Sequence[bool]) -> str:
    """Encode whether each future action differs from the supplied current state."""

    current = bool(current_pressed)
    return "".join("1" if bool(action) != current else "0" for action in future_actions)


def threshold_pattern(probabilities: Sequence[float], threshold: float) -> str:
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    return "".join("1" if float(value) >= threshold else "0" for value in probabilities)


def pattern_counts(patterns: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(patterns).items()))


def weighted_pattern_counts(
    patterns: Sequence[str],
    weights: Sequence[float],
) -> dict[str, float]:
    if len(patterns) != len(weights):
        raise ValueError("patterns and weights must have the same length")
    totals: dict[str, float] = defaultdict(float)
    for pattern, weight in zip(patterns, weights, strict=True):
        value = float(weight)
        if value < 0.0:
            raise ValueError("weights must be >= 0")
        totals[pattern] += value
    return dict(sorted(totals.items()))


def binary_metrics(predicted: Sequence[bool], target: Sequence[bool]) -> dict[str, float | int]:
    if len(predicted) != len(target):
        raise ValueError("predicted and target must have the same length")
    tp = fp = fn = tn = 0
    for pred, truth in zip(predicted, target, strict=True):
        pred = bool(pred)
        truth = bool(truth)
        if pred and truth:
            tp += 1
        elif pred:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": tp + fp + fn + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def one_vs_rest_metrics(
    target_patterns: Sequence[str],
    predicted_patterns: Sequence[str],
    pattern: str,
) -> dict[str, float | int]:
    return binary_metrics(
        [value == pattern for value in predicted_patterns],
        [value == pattern for value in target_patterns],
    )


def confusion_counts(
    target_patterns: Sequence[str],
    predicted_patterns: Sequence[str],
) -> dict[str, dict[str, int]]:
    if len(target_patterns) != len(predicted_patterns):
        raise ValueError("target_patterns and predicted_patterns must have the same length")
    rows: dict[str, Counter[str]] = defaultdict(Counter)
    for truth, predicted in zip(target_patterns, predicted_patterns, strict=True):
        rows[truth][predicted] += 1
    return {
        truth: dict(sorted(row.items()))
        for truth, row in sorted(rows.items())
    }


def per_horizon_metrics(
    target_patterns: Sequence[str],
    predicted_patterns: Sequence[str],
) -> list[dict[str, float | int]]:
    if len(target_patterns) != len(predicted_patterns):
        raise ValueError("target_patterns and predicted_patterns must have the same length")
    if not target_patterns:
        return []
    width = len(target_patterns[0])
    if any(len(value) != width for value in target_patterns + predicted_patterns):
        raise ValueError("all patterns must have the same width")
    return [
        binary_metrics(
            [pattern[index] == "1" for pattern in predicted_patterns],
            [pattern[index] == "1" for pattern in target_patterns],
        )
        for index in range(width)
    ]
