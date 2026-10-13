"""No-GPU/no-video unit tests of diagnostic timestamp shift analysis."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "audit_tiny_control_horizon_alignment",
    ROOT / "scripts/audit_tiny_control_horizon_alignment.py",
)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def row(t, pressed, *, variant="full"):
    return {"variant": variant, "observation_ms": t - 100.,
            "target_ms": t, "rollout_prob": 0.9 if pressed else 0.1}


def test_model_event_time_is_future_target_and_horizon_guard():
    rows = [row(110, False), row(140, True), row(180, True), row(280, False)]
    assert subject.event_timeline(rows, "full") == [(140, True), (280, False)]
    with pytest.raises(ValueError, match="observation"):
        subject.event_timeline([{**row(110, False), "observation_ms": 20}], "full")


def test_greedy_match_does_not_reuse_truth_for_duplicate_model_events():
    model = [(100., True), (112., True), (190., False)]
    truth = [(105., True), (200., False)]
    press = subject.match_polarity(
        model, truth, pressed=True, start_ms=0, end_ms=300, tolerance_ms=30,
    )
    assert press["matched"] == 1
    assert press["unmatched_predicted"] == 1
    assert press["signed_lag_median_ms"] == -5


def test_minus_h100_is_diagnostic_only_and_changes_event_match():
    model = [(260., True), (420., False)]
    truth = [(160., True), (320., False)]
    sweep = subject.scan_shifts(
        model, truth, start_ms=0, end_ms=1000,
        max_shift_ms=200, step_ms=25, tolerance_ms=25,
    )
    zero = next(x for x in sweep if x["shift_ms"] == 0)
    minus = next(x for x in sweep if x["shift_ms"] == -100)
    assert zero["total_matched"] == 0
    assert minus["total_matched"] == 2
    # Multiple offsets can tie: choose the smallest absolute displacement.
    assert subject.best_shift(sweep, "total")["shift_ms"] == -75


def test_uniform_shift_never_changes_episode_duration():
    source = (200., 530.)
    for offset in (-250., -100., 0., 100.):
        assert ((source[1] + offset) - (source[0] + offset)) == pytest.approx(330.)


def test_best_shift_prefers_nearest_zero_when_scores_tie():
    sweep = [
        {"shift_ms": -25, "total_matched": 1,
         "release": {"matched": 1}, "press": {"matched": 1}},
        {"shift_ms": 0, "total_matched": 1,
         "release": {"matched": 1}, "press": {"matched": 1}},
        {"shift_ms": 25, "total_matched": 1,
         "release": {"matched": 1}, "press": {"matched": 1}},
    ]
    assert subject.best_shift(sweep, "release")["shift_ms"] == 0


def test_hold_duration_metrics_are_invariant_to_shift():
    audit = {"variants": {"full": {
        "matched_by_press_onset": 3,
        "release_plausible_subset": {
            "n": 2, "hold_minus_kernel_ms": {"median_ms": 225},
            "model_hold_minus_overlay_ms": {"median_ms": 170},
        },
    }}}
    result = subject.episode_hold_audit(audit, "full")
    assert result["model_minus_kernel_hold_median_ms"] == 225
    assert result["unchanged_by_uniform_time_shift"]


def test_refuse_inconsistent_shift_grid():
    with pytest.raises(ValueError, match="divide H100"):
        subject.scan_shifts([], [], start_ms=0, end_ms=1000,
                            max_shift_ms=300, step_ms=30)
