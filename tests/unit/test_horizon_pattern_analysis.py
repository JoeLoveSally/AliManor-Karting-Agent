from __future__ import annotations

import pytest

from karting_agent.train.horizon_pattern_analysis import (
    confusion_counts,
    one_vs_rest_metrics,
    pattern_counts,
    per_horizon_metrics,
    switch_pattern,
    threshold_pattern,
    weighted_pattern_counts,
)


def test_switch_pattern_is_relative_to_current_state() -> None:
    assert switch_pattern(False, (True, False, False)) == "100"
    assert switch_pattern(True, (True, False, False)) == "011"


def test_threshold_pattern_uses_inclusive_threshold() -> None:
    assert threshold_pattern((0.59, 0.60, 0.90), 0.60) == "011"


def test_pattern_counts_and_weighted_counts() -> None:
    patterns = ["000", "100", "100"]
    assert pattern_counts(patterns) == {"000": 1, "100": 2}
    assert weighted_pattern_counts(patterns, [1.0, 2.0, 3.0]) == {
        "000": 1.0,
        "100": 5.0,
    }


def test_one_vs_rest_pattern_metrics() -> None:
    metrics = one_vs_rest_metrics(
        ["100", "100", "000", "000"],
        ["100", "000", "100", "000"],
        "100",
    )
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_confusion_and_per_horizon_metrics() -> None:
    truth = ["100", "011", "000"]
    predicted = ["100", "001", "000"]
    assert confusion_counts(truth, predicted) == {
        "000": {"000": 1},
        "011": {"001": 1},
        "100": {"100": 1},
    }
    metrics = per_horizon_metrics(truth, predicted)
    assert len(metrics) == 3
    assert metrics[0]["f1"] == pytest.approx(1.0)
    assert metrics[1]["recall"] == pytest.approx(0.0)
    assert metrics[2]["f1"] == pytest.approx(1.0)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        weighted_pattern_counts(["000"], [])
    with pytest.raises(ValueError):
        confusion_counts(["000"], [])
