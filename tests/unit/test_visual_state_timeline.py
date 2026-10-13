"""Pure-Python regression tests for recorded CV-state timeline diagnostics."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_visual_state_timeline.py"
spec = importlib.util.spec_from_file_location("_visual_state_timeline_test", SCRIPT)
assert spec is not None and spec.loader is not None
subject = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subject
spec.loader.exec_module(subject)


def row(frame, time_ms, *, valid=True, lateral_rate=None, heading_rate=None):
    return {
        "frame_index": frame, "timestamp_ms": time_ms,
        "state_valid": valid, "kart_detected": valid,
        "lateral_rate_per_s": lateral_rate,
        "heading_rate_deg_per_s": heading_rate,
        "lateral_offset_norm": 0.0 if valid else None,
        "heading_error_axial_deg": 0.0 if valid else None,
        "tracker_mode": "stable" if valid else "missing",
        "relation_confidence": 0.2 if valid else None,
    }


def test_rate_percentiles_with_single_value_are_not_zero():
    stats = subject.rate_stats([
        row(1, 1.0, lateral_rate=-12.5, heading_rate=90.0)
    ])
    assert stats["abs_lateral_rate_p50_p95"] == [12.5, 12.5]
    assert stats["abs_heading_rate_p50_p95_deg_s"] == [90.0, 90.0]


def test_binning_uses_actual_host_elapsed_time_not_frame_rate():
    rows = [row(0, 10100.0), row(1, 11000.0, valid=False),
            row(2, 11900.0, valid=False), row(3, 12100.0)]
    result = subject.analyze_timeline(
        rows, control_start_ms=10000.0, window_ms=1000.0
    )
    windows = result["per_window"]
    assert [w["window_index"] for w in windows] == [0, 1, 2]
    assert windows[1]["samples"] == 2
    assert windows[1]["valid_state_rate"] == 0.0


def test_longest_invalid_run_keeps_correct_frames_and_counts():
    rows = [row(1, 100), row(2, 150, valid=False),
            row(3, 200, valid=False), row(4, 250),
            row(5, 300, valid=False), row(6, 350)]
    report = subject.analyze_timeline(rows, control_start_ms=100)
    gaps = report["longest_invalid_runs"]
    assert gaps[0]["count"] == 2
    assert [gaps[0]["start_frame"], gaps[0]["end_frame"]] == [2, 3]
    assert gaps[1]["count"] == 1


def test_ranked_spikes_relate_to_nearest_recorded_action():
    rows = [row(10, 100, lateral_rate=1, heading_rate=10),
            row(20, 200, lateral_rate=-40, heading_rate=-200)]
    actions = [{"timestamp_ms": 220.0, "action": "RELEASE",
                "source": "step", "index": 9}]
    previews = [(9, "frame_000009.jpg"), (21, "frame_000021.jpg")]
    report = subject.analyze_timeline(
        rows, events=actions, previews=previews, top_k=1,
    )
    lateral = report["ranked_spikes"][0]
    assert lateral["frame_index"] == 20
    assert lateral["value"] == -40.0
    assert lateral["nearest_action"]["action"] == "RELEASE"
    assert lateral["nearest_action"]["delta_from_frame_ms"] == pytest.approx(-20)
    assert lateral["nearest_existing_preview"]["offset_frames"] == 1


def test_empty_action_list_and_absent_preview_are_explicitly_null():
    report = subject.analyze_timeline(
        [row(1, 100, lateral_rate=2.0, heading_rate=3.0)]
    )
    assert report["ranked_spikes"][0]["nearest_action"] is None
    assert report["ranked_spikes"][0]["nearest_existing_preview"] is None
    assert "No gameplay phase" in report["warning"]


def test_events_from_run_preserves_nominal_deadline_provenance():
    run = {
        "steps": [
            {"observation_timestamp_ms": 20.0, "action": "PRESS"},
            {"observation_timestamp_ms": 70.0, "action": "HOLD"},
        ],
        "deadline_events": [
            {"timestamp_ms": 15.0, "action": "RELEASE"},
        ],
    }
    events = subject.events_from_run(run)
    assert [event["action"] for event in events] == ["RELEASE", "PRESS"]
    assert events[0]["source"] == "deadline_nominal"


def test_nonmonotonic_source_rows_are_rejected(tmp_path):
    path = tmp_path / "measurements.jsonl"
    path.write_text(
        '{"frame_index":1,"timestamp_ms":100}\n'
        '{"frame_index":2,"timestamp_ms":90}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-monotonic"):
        subject.load_measurements(path)


def test_axial_seam_rate_spike_masked_without_hiding_raw_measurement():
    first = row(1, 100, valid=True, lateral_rate=None, heading_rate=None)
    first["road_axis_axial_deg"] = 179.0
    first["lateral_offset_norm"] = 0.7
    second = row(2, 150, valid=True, lateral_rate=-28, heading_rate=8)
    second["road_axis_axial_deg"] = 1.0
    second["lateral_offset_norm"] = -0.7

    clean, warnings = subject.suppress_axial_seam_rates([first, second])
    assert len(warnings) == 1
    assert warnings[0]["road_axis_before_deg"] == 179.0
    assert warnings[0]["road_axis_after_deg"] == 1.0
    assert clean[1]["lateral_rate_per_s"] is None
    assert clean[1]["heading_rate_deg_per_s"] is None
    assert second["lateral_rate_per_s"] == -28

    report = subject.analyze_timeline([first, second], control_start_ms=100)
    assert len(report["axis_seam_suspicions"]) == 1
    assert report["original_rate_statistics"]["valid_rate_pair_coverage"] == 0.5
    assert report["per_window"][0]["valid_rate_pair_coverage"] == 0.0


def test_large_true_axis_rotation_not_marked_as_representation_seam():
    first = row(1, 100)
    second = row(2, 150, lateral_rate=6)
    first["road_axis_axial_deg"] = 125.0
    second["road_axis_axial_deg"] = 15.0
    output, flagged = subject.suppress_axial_seam_rates([first, second])
    assert not flagged
    assert output[1]["lateral_rate_per_s"] == 6


def test_small_axis_change_inside_chart_is_not_masked():
    first = row(1, 100)
    second = row(2, 150, lateral_rate=3)
    first["road_axis_axial_deg"] = 40.0
    second["road_axis_axial_deg"] = 42.0
    output, flagged = subject.suppress_axial_seam_rates([first, second])
    assert not flagged
    assert output[1]["lateral_rate_per_s"] == 3
