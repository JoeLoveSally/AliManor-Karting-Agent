"""No-GPU and no-video regression tests for event-aligned timing audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("audit_tiny_control_episode_timing", ROOT / "scripts/audit_tiny_control_episode_timing.py")
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def touch(t, pressed, kernel):
    return {"kind": "touch_transition", "host_received_perf_counter_ms": t,
            "device_timestamp_ms": kernel, "pressed": pressed}


def test_human_hold_uses_kernel_duration_and_host_origin():
    raw = [touch(1050, True, 500), touch(1110, False, 561),
           touch(1200, True, 651), touch(1320, False, 772)]
    rows = subject.human_episodes(raw, origin_ms=1000, start_ms=0, end_ms=400)
    assert len(rows) == 2
    assert rows[0]["kernel_hold_ms"] == 61
    assert rows[0]["host_hold_ms"] == 60
    assert rows[0]["host_minus_kernel_hold_ms"] == -1


def test_model_episodes_reconstruct_t_plus_100_transitions_without_chatter_dropping():
    entries = [
        {"variant": "full", "target_ms": t, "rollout_prob": p}
        for t, p in [(100, .2), (150, .9), (200, .8), (250, .1),
                     (300, .8), (350, .1)]
    ]
    episodes = subject.model_episodes(entries, "full")
    assert [(x["press_ms"], x["release_ms"]) for x in episodes] == [
        (150, 250), (300, 350)
    ]
    with pytest.raises(ValueError, match="missing"):
        subject.model_episodes(entries, "vision_only")


def test_episode_matching_and_flagging_does_not_erase_early_release_outlier():
    human = [
        dict(human_index=0, press_ms=150, release_ms=400, host_hold_ms=250,
             kernel_hold_ms=250, host_minus_kernel_hold_ms=0),
        dict(human_index=1, press_ms=700, release_ms=1000, host_hold_ms=300,
             kernel_hold_ms=300, host_minus_kernel_hold_ms=0),
    ]
    model = [
        dict(model_index=0, press_ms=165, release_ms=475, hold_ms=310),
        dict(model_index=1, press_ms=690, release_ms=750, hold_ms=60),
    ]
    pairs = subject.onset_pairs(human, model)
    assert len(pairs) == 2
    assert pairs[0]["hold_minus_kernel_ms"] == 60
    assert pairs[1]["release_minus_host_ms"] == -250
    audit, rows = subject.audit(human, {v: model for v in subject.VARIANTS})
    assert audit["variants"]["full"]["release_plausible_subset"]["n"] == 1
    assert audit["variants"]["full"]["ambiguous_human_indices"] == [1]
    assert len(rows) == 6


def test_optical_matching_same_human_episode_not_just_statistical_median():
    base = dict(human_index=0, model_index=0, human_press_ms=150,
                human_release_ms=400, model_press_ms=165, model_release_ms=475,
                kernel_hold_ms=250, host_hold_ms=250, model_hold_ms=310,
                press_minus_host_ms=15, release_minus_host_ms=75,
                hold_minus_kernel_ms=60, hold_minus_host_ms=60,
                host_minus_kernel_hold_ms=0)
    optical = {
        "_origin_ms": 1000,
        "optical_overlay": {"paired": {"examples": [
            {"host_press_ms": 1150, "host_release_ms": 1400,
             "overlay_press_lag_ms": 70, "overlay_release_lag_ms": 120,
             "overlay_hold_excess_ms": 50}
        ]}},
    }
    pairs = [base]
    subject.attach_optical(pairs, optical)
    assert pairs[0]["model_press_minus_overlay_ms"] == -55
    assert pairs[0]["model_release_minus_overlay_ms"] == -45
    assert pairs[0]["model_hold_minus_overlay_ms"] == 10


def test_nonmonotonic_kernel_touch_stream_rejected():
    with pytest.raises(ValueError, match="nonmonotonic"):
        subject.human_episodes(
            [touch(1000, True, 20), touch(900, False, 40)],
            origin_ms=0, start_ms=0, end_ms=2000,
        )


def test_summary_empty_without_dangerous_zero_imputation():
    assert subject.summarize([]) == {
        "n": 0, "median_ms": None, "p05_ms": None, "p95_ms": None,
    }
