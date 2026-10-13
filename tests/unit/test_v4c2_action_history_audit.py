from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_v4c2_action_history import audit_run, commanded_transitions  # noqa: E402


def example():
    return {
        "runtime": {"initial_pressed": False},
        "steps": [
            {"observation_timestamp_ms": 100.0,
             "input_frame_timestamps_ms": [0.0, 50.0, 100.0],
             "input_frame_indices": [0, 1, 2],
             "probabilities": [0.01, 0.01, 0.01],
             "action": "PRESS", "pressed": True},
            {"observation_timestamp_ms": 125.0,
             "input_frame_timestamps_ms": [25.0, 75.0, 125.0],
             "input_frame_indices": [1, 2, 3],
             "probabilities": [0.99, 0.92, 0.02],
             "action": "HOLD", "pressed": True},
            {"observation_timestamp_ms": 800.0,
             "input_frame_timestamps_ms": [600.0, 700.0, 800.0],
             "input_frame_indices": [4, 5, 6],
             "probabilities": [0.01, 0.01, 0.01],
             "action": "RELEASE", "pressed": False},
        ],
        "deadline_events": [],
    }


def test_transient_not_automatically_reversal():
    result = audit_run(example())
    assert result["transitions"] == 2
    assert result["transient_both_high_steps"] == 1
    assert result["transient_both_high_with_no_reverse_within_500ms"] == 1
    row = result["transient_both_high_examples"][0]
    assert row["inferred_frame_command_states"] == [0, 0, 1]
    assert row["mixed_command_history"]
    assert row["next_command_transition_ms"] == pytest.approx(675.0)


def test_merge_timer_and_regular_transitions():
    run = example()
    run["steps"][0]["action"] = "HOLD"
    run["deadline_events"].append({"timestamp_ms": 90.0, "action": "PRESS"})
    assert commanded_transitions(run) == [(90.0, True), (800.0, False)]


def test_reject_non_alternating_transitions():
    run = example()
    run["deadline_events"].append({"timestamp_ms": 99.0, "action": "PRESS"})
    with pytest.raises(ValueError, match="alternate"):
        commanded_transitions(run)


def test_pretransition_history_not_marked_mixed():
    run = example()
    run["steps"][1]["input_frame_timestamps_ms"] = [115.0, 120.0, 125.0]
    result = audit_run(run)
    assert not result["transient_both_high_examples"][0]["mixed_command_history"]


def test_reject_future_frame_and_empty_input():
    run = example()
    run["steps"][0]["input_frame_timestamps_ms"] = [0.0, 101.0]
    with pytest.raises(ValueError, match="frame timestamps"):
        audit_run(run)
    run = example()
    run["steps"][0]["input_frame_timestamps_ms"] = []
    with pytest.raises(ValueError, match="recorded step"):
        audit_run(run)


def test_frame_at_command_timestamp_is_still_pre_command():
    run = example()
    result = audit_run(run)
    assert result["steps"] == 3
    assert result["mixed_command_history_steps"] == 1
