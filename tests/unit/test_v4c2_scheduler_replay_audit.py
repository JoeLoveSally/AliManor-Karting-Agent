"""Small deterministic controls for read-only V4-C2 scheduler parity audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/audit_v4c2_scheduler_replay.py"
spec = importlib.util.spec_from_file_location("_scheduler_parity_audit_test", SCRIPT)
assert spec is not None and spec.loader is not None
audit_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit_module
spec.loader.exec_module(audit_module)
scheduler_module = audit_module.load_scheduler(audit_module.DEFAULT_SCHEDULER)


def example_run():
    cfg = {"horizons_ms": [100.0, 200.0, 300.0],
           "control_horizon_ms": 200.0, "anticipation_horizon_ms": 300.0,
           "threshold": .6, "min_state_hold_ms": 100.0,
           "pending_advance_ms": 0.0, "arm_pending_during_min_hold": True}
    steps = [
        {"observation_timestamp_ms": 0.0, "probabilities": [.01, .01, .01],
         "action": "HOLD", "pressed": False, "scheduler_reason": "hold", "pending_due_ms": None},
        {"observation_timestamp_ms": 100.0, "probabilities": [.001, .1, .9],
         "action": "HOLD", "pressed": False, "scheduler_reason": "pending_armed", "pending_due_ms": 162.5},
        {"observation_timestamp_ms": 150.0, "probabilities": [.001, .1, .9],
         "action": "HOLD", "pressed": False, "scheduler_reason": "pending_wait", "pending_due_ms": 162.5},
        {"observation_timestamp_ms": 180.0, "probabilities": [.99, .8, .01],
         "action": "HOLD", "pressed": True, "scheduler_reason": "min_hold", "pending_due_ms": None},
        {"observation_timestamp_ms": 300.0, "probabilities": [.001, .001, .001],
         "action": "HOLD", "pressed": True, "scheduler_reason": "hold", "pending_due_ms": None},
    ]
    return {"runtime": {"initial_pressed": False,
                        "execute_pending_at_due": True,
                        "multi_horizon_scheduler": cfg},
            "steps": steps,
            "deadline_events": [{"timestamp_ms": 162.5, "action": "PRESS", "pressed": True}]}


def test_timer_and_steps_have_complete_parity():
    report = audit_module.audit(example_run(), scheduler_module)
    assert report["parity_passed"] is True
    assert report["steps_checked"] == 5
    assert report["replayed_deadlines"] == 1


def test_missing_recorded_timer_blocks_parity():
    run = example_run()
    run["deadline_events"] = []
    report = audit_module.audit(run, scheduler_module)
    assert report["parity_passed"] is False
    assert report["first_errors"][0]["kind"] == "unexpected_timer_event"


def test_different_recorded_action_blocks_parity():
    run = example_run()
    run["steps"][0]["action"] = "PRESS"
    run["steps"][0]["pressed"] = True
    report = audit_module.audit(run, scheduler_module)
    assert report["parity_passed"] is False
    assert report["first_errors"][0]["kind"] == "pre_state_mismatch"


def test_new_opt_in_source_option_cannot_be_silently_dropped():
    run = example_run()
    run["runtime"]["multi_horizon_scheduler"]["execute_short_horizon_overdue"] = True
    with pytest.raises(ValueError, match="scheduler source lacks recorded options"):
        audit_module.audit(run, scheduler_module)
