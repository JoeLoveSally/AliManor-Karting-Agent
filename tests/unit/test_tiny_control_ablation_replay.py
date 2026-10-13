"""Pure CPU tests for independent Tiny control-ablation video replay diagnostics."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "evaluate_tiny_control_ablation_replay_test",
    ROOT / "scripts/evaluate_tiny_control_ablation_replay.py",
)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def point(t, teacher, virtual):
    return {
        "target_ms": float(t), "teacher_prob": float(teacher),
        "rollout_prob": float(virtual),
    }


def test_release_windows_measure_later_virtual_release_and_conflicts():
    rows = [point(210, .2, .9), point(260, .1, .3), point(350, .1, .1)]
    windows = subject.release_windows(rows, [200.])
    assert len(windows) == 1
    assert windows[0]["prediction_rows"] == 3
    assert windows[0]["teacher_release_virtual_press_conflict"]
    assert windows[0]["conflict_rows"] == 1
    assert windows[0]["first_teacher_release_after_ms"] == 10.
    assert windows[0]["first_virtual_release_after_ms"] == 60.
    assert windows[0]["virtual_minus_teacher_first_release_ms"] == 50.
    summary = subject.aggregate_release_windows(windows)
    assert summary["human_release_events"] == 1
    assert summary["virtual_first_release_later"] == 1
    assert summary["median_virtual_minus_teacher_first_release_ms"] == 50.


def test_missing_virtual_release_is_reported_without_invented_delay():
    windows = subject.release_windows([point(100, .2, .9)], [100.])
    assert windows[0]["first_teacher_release_after_ms"] == 0.
    assert windows[0]["first_virtual_release_after_ms"] is None
    assert windows[0]["virtual_minus_teacher_first_release_ms"] is None
    summary = subject.aggregate_release_windows(windows)
    assert summary["events_with_both_first_release_predictions"] == 0
    assert summary["virtual_first_release_absent_in_window"] == 1
    assert summary["median_virtual_minus_teacher_first_release_ms"] is None


def test_release_window_uses_absolute_target_ms():
    rows = [point(800, .2, .9), point(950, .1, .1)]
    windows = subject.release_windows(rows, [1000.], left_ms=100., right_ms=100.)
    assert windows[0]["prediction_rows"] == 1
    assert windows[0]["first_teacher_release_after_ms"] is None
    assert windows[0]["first_virtual_release_after_ms"] is None


def test_report_enforces_variant_mask_split_and_non_smoke():
    from karting_agent.model.tiny_control_ablation import CONTROL_MASKS

    split = SimpleNamespace(name="v4c2_temporal_v2",
                            test=("test-a.mp4", "test-b.mp4"))
    report = {
        "kind": "tiny_rgb_control_input_ablation_training",
        "variant": "vision_only",
        "smoke_unrepresentative": False,
        "control_mask": list(CONTROL_MASKS["vision_only"]),
        "split": split.name,
        "target": "expert_PRESS_at_observation_plus_100ms",
        "test_videos_not_evaluated": list(split.test),
    }
    subject.validate_training_report(report, variant="vision_only", split=split)
    with pytest.raises(ValueError, match="incompatible"):
        subject.validate_training_report(report, variant="full", split=split)
    with pytest.raises(ValueError, match="incompatible"):
        subject.validate_training_report({**report, "smoke_unrepresentative": True},
                                         variant="vision_only", split=split)
    with pytest.raises(ValueError, match="incompatible"):
        subject.validate_training_report({**report, "test_videos_not_evaluated": []},
                                         variant="vision_only", split=split)


def test_sha256_is_content_derived(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"frozen experimental model")
    assert subject.sha256(weights) == hashlib.sha256(weights.read_bytes()).hexdigest()


def test_replay_script_is_read_only_and_no_heldout_test_evaluator():
    source = (ROOT / "scripts/evaluate_tiny_control_ablation_replay.py").read_text()
    assert "from evaluate_history_candidate import" in source
    assert "replay_candidate" in source
    assert 'with output.open("x"' in source
    assert 'with rows_path.open("x"' in source
    assert ".backward(" not in source
    assert ".optimizer" not in source
    assert "input motionevent" not in source
