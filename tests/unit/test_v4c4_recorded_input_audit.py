from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from karting_agent.vision.preprocess import PreprocessConfig


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    from diagnose_v4c4_recorded_input import (  # noqa: E402
        replay_recorded_steps,
        summarize,
        validate_run,
    )
finally:
    sys.path.remove(str(SCRIPTS))


def make_step(frame: int, history: list[int], probability: float) -> dict[str, object]:
    return {
        "source_frame_index": frame,
        "observation_timestamp_ms": frame * 20.0,
        "history_frame_indices": history,
        "action_probability": probability,
    }


def test_audit_rejects_missing_or_unreliable_frame_mapping() -> None:
    payload = {
        "policy": "v4c4_current_action_only",
        "recording_frame_mapping_valid": False,
        "steps": [make_step(4, [0, 1, 2, 3, 4], 0.2)],
    }
    with pytest.raises(ValueError, match="recording_frame_mapping_valid"):
        validate_run(payload)
    payload["recording_frame_mapping_valid"] = True
    assert len(validate_run(payload)) == 1


def test_audit_rejects_noncausal_or_nonmonotonic_indices() -> None:
    base = {"policy": "v4c4_current_action_only", "recording_frame_mapping_valid": True}
    for steps in (
        [make_step(4, [0, 1, 2, 5, 4], 0.5)],
        [make_step(4, [0, 1, 2, 3, 4], 0.5), make_step(4, [0, 1, 2, 3, 4], 0.5)],
        [make_step(4, [0, 1, 2, 3, 4], float("nan"))],
    ):
        with pytest.raises(ValueError):
            validate_run({**base, "steps": steps})


def test_summary_counts_decision_disagreements_and_large_errors() -> None:
    predictions = [
        {"source_frame_index": 4, "logged_probability": 0.49,
         "replayed_probability": 0.51, "absolute_error": 0.02},
        {"source_frame_index": 5, "logged_probability": 0.2,
         "replayed_probability": 0.2, "absolute_error": 0.0},
    ]
    result = summarize(predictions, decision_threshold=0.5, max_acceptable_error=0.05)
    assert result["decision_disagreements"] == 1
    assert result["decision_disagreement_frames"] == [4]
    assert result["passes_requested_audit_tolerance"] is False


def test_video_replay_reads_shared_history_frames_once(tmp_path: Path) -> None:
    path = tmp_path / "capture.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 96))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPEG video writer unavailable")
    for index in range(6):
        image = np.full((96, 64, 3), index * 20, dtype=np.uint8)
        writer.write(image)
    writer.release()

    recorded_inputs: list[np.ndarray] = []

    def fake_predict(normalized_rgb_stack: np.ndarray) -> float:
        recorded_inputs.append(normalized_rgb_stack.copy())
        return 0.6 if len(recorded_inputs) == 1 else 0.4

    steps = [
        make_step(4, [0, 1, 2, 3, 4], 0.6),
        make_step(5, [1, 2, 3, 4, 5], 0.4),
    ]
    result = replay_recorded_steps(
        path, steps, predict_action=fake_predict, preprocess_config=PreprocessConfig()
    )
    assert [row["source_frame_index"] for row in result] == [4, 5]
    assert len(recorded_inputs) == 2
    np.testing.assert_array_equal(recorded_inputs[0][3:], recorded_inputs[1][:-3])
    summary = summarize(result, decision_threshold=0.5, max_acceptable_error=0.05)
    assert summary["passes_requested_audit_tolerance"] is True
