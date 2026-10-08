"""Pure-Python checks for saved V4-C2 pending-warning attribution."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_v4c2_pending_attribution.py"
spec = importlib.util.spec_from_file_location("_v4c2_pending_attribution_test", SCRIPT)
assert spec is not None and spec.loader is not None
subject = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subject
spec.loader.exec_module(subject)


def example_entry(base, adapter, threshold=0.60):
    return {
        "candidate_horizons": "full",
        "baseline_threshold_all_horizons": 0.60,
        "candidate_threshold_all_horizons": threshold,
        "status": "state_divergence",
        "first_pending_plan_change": {
            "step": 21,
            "observation_ms": 11687.7,
            "logged_probabilities": base,
            "candidate_probabilities": adapter,
            "reference_due_ms": 11757.5,
            "candidate_due_ms": 11768.7,
        },
        "first_difference": {"kind": "recorded_timer_transition_not_due_for_candidate"},
    }


def test_warning_gate_requires_monotone_all_horizons():
    gate = subject.warning_gate([0.011975, 0.008379, 0.824569], 0.60)
    assert gate["threshold_crosses_h200_h300"] is True
    assert gate["h100_gt_h200"] is True
    assert gate["monotonic"] is False
    assert gate["anticipation_gate_eligible"] is False
    assert gate["ungated_interpolated_delay_ms"] is not None


def test_first_plan_h200_only_hybrid_creates_artificial_nonmonotonicity():
    entry = example_entry(
        [0.011975, 0.081090, 0.824569],
        [0.000476, 0.008379, 0.738586],
    )
    result = subject.analyze_first_plan("historical", "full:0.6", entry)
    gates = result["actual_scheduler_gate"]
    assert gates["baseline"]["anticipation_gate_eligible"]
    assert gates["full_candidate"]["anticipation_gate_eligible"]
    assert gates["h200_only_synthetic"]["anticipation_gate_eligible"] is False
    sensitivity = result["fixed_baseline_threshold_sensitivity"]
    assert (
        sensitivity["h300_only_synthetic"]["delay_change_from_baseline_ms"]
        > sensitivity["h200_only_synthetic"]["delay_change_from_baseline_ms"]
    )
    assert sensitivity["full_candidate"]["delay_change_from_baseline_ms"] > 0


def test_h300_suppression_can_prevent_warning():
    entry = example_entry(
        [0.007110, 0.033147, 0.619962],
        [0.000366, 0.002283, 0.441328],
    )
    result = subject.analyze_first_plan("baseline", "full:0.6", entry)
    gate = result["actual_scheduler_gate"]["full_candidate"]
    assert gate["monotonic"] is True
    assert gate["h300_at_least_threshold"] is False
    assert gate["ungated_interpolated_delay_ms"] is None


def test_higher_threshold_alone_can_block_initial_candidate_warning():
    entry = example_entry(
        [0.011975, 0.081090, 0.824569],
        [0.000476, 0.008379, 0.738586],
        threshold=0.76,
    )
    result = subject.analyze_first_plan("historical", "full:0.76", entry)
    assert not result["actual_scheduler_gate"]["full_candidate"][
        "anticipation_gate_eligible"
    ]
    assert result["fixed_baseline_threshold_sensitivity"]["full_candidate"][
        "anticipation_gate_eligible"
    ]


def test_saved_prefix_reports_only_include_full_variant():
    obj = {"runs": {"historical": {
        "full:0.6": example_entry(
            [0.011975, 0.081090, 0.824569],
            [0.000476, 0.008379, 0.738586],
        ),
        "control_only:0.6": {
            "candidate_horizons": "control_only",
            "first_pending_plan_change": {},
        },
    }}}
    report = subject.analyze_reports([obj])
    assert len(report["cases"]) == 1
    assert report["cases"][0]["variant"] == "full:0.6"
