"""Pure deterministic offline evaluation tests; no Android/checkpoints."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate_tiny_live_recording.py"
spec = importlib.util.spec_from_file_location("evaluate_tiny_live_recording", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_touch_timeline_and_causal_feedback():
    rows = [
        {"kind": "touch_transition", "host_received_perf_counter_ms": 900, "pressed": True},
        {"kind": "touch_transition", "host_received_perf_counter_ms": 950, "pressed": False},
        {"kind": "touch_transition", "host_received_perf_counter_ms": 1150, "pressed": True},
        {"kind": "touch_transition", "host_received_perf_counter_ms": 1250, "pressed": False},
    ]
    timeline = module.touch_timeline(rows, 1000)
    assert timeline == [(0.0, False), (150.0, True), (250.0, False)]
    features = module.control_features([0, 100, 200], timeline)
    assert list(features[:, 0]) == [0, 0, 1]
    assert features[1, 1] == pytest.approx(.2)
    assert features[2, 1] == pytest.approx(.1)


def test_real_short_release_preserved():
    events = [(0, False), (100, True), (600, False), (700, True), (1000, False)]
    assert module.timeline_state(events, 650) is False
    assert module.timeline_state(events, 701) is True


def test_kernel_touch_duration_not_host_arrival_duration():
    rows = [
        {"kind": "touch_transition", "host_received_perf_counter_ms": 1100,
         "device_timestamp_ms": 1000, "pressed": True},
        {"kind": "touch_transition", "host_received_perf_counter_ms": 1680,
         "device_timestamp_ms": 1500, "pressed": False},
    ]
    assert module.physical_press_spans(rows, 1000, 0, 1000) == [(100, 680, 500)]


def test_extra_hold_constant_clock_shift_cancels():
    real = [(0, False), (100, True), (600, False), (700, True), (1000, False)]
    virtual = [(0, False), (120, True), (800, False), (920, True), (1300, False)]
    reference = module.paired_duration_drift(virtual, real, 0, 1500, tol_ms=250)
    shifted = [(0, False)] + [(t + 30, s) for t, s in real[1:]]
    shifted_result = module.paired_duration_drift(
        virtual, shifted, 0, 1500, tol_ms=250
    )
    assert reference["pairs_matched_by_press_onset"] == 2
    assert reference["duration_difference_median_ms"] == pytest.approx(130)
    assert shifted_result["duration_difference_median_ms"] == pytest.approx(130)


def test_ffmpeg_status_not_sufficient_to_reject_matching_frame_metadata():
    summary = {"recording": {"recorded_frames": 2, "frame_mapping_valid": False,
                             "recording_error": "255"}}
    frames = [
        {"frame_index": 0, "host_decoded_ms": 11, "host_decoded_perf_counter_ms": 1011},
        {"frame_index": 1, "host_decoded_ms": 21, "host_decoded_perf_counter_ms": 1021},
    ]
    assert module.frame_consistency(
        summary, frames, [{"frame_index": 1, "observation_ms": 21}]
    )["frame_metadata_consistent"]
    with pytest.raises(ValueError, match="timestamp"):
        module.frame_consistency(
            summary, frames, [{"frame_index": 1, "observation_ms": 22}]
        )


def test_nearest_event_matching_one_to_one_and_empty_precision():
    prediction = [(100, True), (200, True), (300, False)]
    truth = [(110, True), (290, False)]
    result = module.match_events(prediction, truth, pressed=True, tol_ms=50)
    assert result["matched"] == 1 and result["predicted"] == 2
    assert module.state_scores([(100, False)], [(0, False)])["press_precision"] is None
