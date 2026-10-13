"""Read-only RELEASE calibration tests, no Android or model checkpoints."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/diagnose_release_timing.py"
spec = importlib.util.spec_from_file_location("diagnose_release_timing", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(kind, device, host, pressed=None):
    result = {
        "kind": kind, "device_timestamp_ms": device,
        "host_received_perf_counter_ms": host,
    }
    if pressed is not None:
        result["pressed"] = pressed
    return result


def test_host_kernel_offset_cannot_remove_unknown_adb_delay():
    rows = [
        event("kernel_event", 100, 300),
        event("kernel_event", 200, 412),
        event("kernel_event", 300, 503),
    ]
    info = module.clock_offset_diagnostics(rows)
    assert info["arrival_excess_above_fastest_observation_ms"]["max"] == pytest.approx(12)
    assert "not calibrated" in info["interpretation"]


def test_touch_dot_detector_and_optical_only_one_frame_filter():
    image = np.full((800, 360, 3), (210, 175, 85), dtype=np.uint8)
    assert not module.gray_touch_dot(image)
    cv2.circle(image, (299, 722), 5, (204, 205, 207), -1)
    assert module.gray_touch_dot(image)
    states = [False, True, True, False, True, True, True, False]
    assert module.clean_optical_only(states) == [
        False, True, True, True, True, True, True, False,
    ]


def test_video_overlay_matching_excludes_short_taps():
    rows = [
        event("touch_transition", 0, 1000, True),
        event("touch_transition", 500, 1500, False),
        event("touch_transition", 800, 1800, True),
        event("touch_transition", 860, 1860, False),
        event("touch_transition", 2000, 3000, True),
        event("touch_transition", 2700, 3700, False),
    ]
    visual = [(1060, True), (1630, False), (3030, True), (3850, False)]
    paired = module.pair_visual_with_kernel(rows, visual)
    assert paired["paired_episodes"] == 2
    assert paired["visual_hold_duration_excess_ms"]["p50"] == pytest.approx(95)


def test_conditional_future_release_and_press():
    predictions = [
        {"target_ms": 200., "probabilities": {"teacher": .1, "virtual": .9}},
        {"target_ms": 300., "probabilities": {"teacher": .9, "virtual": .9}},
        {"target_ms": 500., "probabilities": {"teacher": .9, "virtual": .1}},
    ]
    truth = [(0., False), (100., True), (190., False), (400., True)]
    info = module.future_transition_diagnostics(predictions, truth, 0, 600)
    assert info["teacher"]["upcoming_release"] == {
        "correct": 1, "total": 1, "accuracy": 1.0,
    }
    assert info["virtual"]["upcoming_release"]["accuracy"] == 0.0
    assert info["teacher"]["stable_pressed"]["total"] == 1


def test_overlay_ambiguity_rejected_with_rapid_second_press():
    rows = [
        event("touch_transition", 0, 1000, True),
        event("touch_transition", 600, 1600, False),
        event("touch_transition", 700, 1700, True),
        event("touch_transition", 1300, 2300, False),
    ]
    info = module.pair_visual_with_kernel(
        rows, [(1080, True), (2400, False)]
    )
    assert info["paired_episodes"] == 0
