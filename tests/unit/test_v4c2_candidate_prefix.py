"""CPU-only synthetic cases for factual-prefix scheduler comparison."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "_v4c2_candidate_prefix_test", SCRIPTS / "audit_v4c2_candidate_prefix.py"
)
assert spec is not None and spec.loader is not None
subject = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subject
spec.loader.exec_module(subject)
from audit_v4c2_scheduler_replay import load_scheduler  # noqa: E402

module = load_scheduler(ROOT / "src/karting_agent/runtime/multi_horizon_scheduler.py")


def step(t, probs, action="HOLD", pressed=False, reason="hold", pending_due=None):
    return {
        "observation_timestamp_ms": float(t),
        "probabilities": list(probs),
        "action": action,
        "pressed": pressed,
        "scheduler_reason": reason,
        "pending_due_ms": pending_due,
    }


def run(*steps, deadlines=()):
    return {
        "runtime": {
            "initial_pressed": False,
            "execute_pending_at_due": True,
            "multi_horizon_scheduler": {
                "horizons_ms": [100, 200, 300],
                "control_horizon_ms": 200,
                "anticipation_horizon_ms": 300,
                "threshold": 0.6,
                "arm_pending_during_min_hold": True,
                "min_state_hold_ms": 100,
                "pending_advance_ms": 0,
            },
        },
        "steps": list(steps),
        "deadline_events": list(deadlines),
    }


def predictor(run, replacements=None):
    replacements = replacements or {}
    def predict(i, step, before):
        base = tuple(step["probabilities"])
        return base, tuple(replacements.get(i, base))
    return predict


def test_identical_predictions_replay_without_difference():
    obj = run(
        step(0, [.01, .01, .01]),
        step(30, [.01, .01, .01]),
    )
    result = subject.compare_prefix(
        obj, module, predictor(obj), candidate_threshold=0.60
    )
    assert result["status"] == "no_executable_difference_observed"
    assert result["common_prefix_steps"] == 2
    assert result["first_pending_plan_change"] is None


def test_immediate_switch_difference_stops_at_current_step():
    obj = run(
        step(0, [.01, .01, .01]),
        step(150, [.99, .90, .99], action="PRESS",
             pressed=True, reason="primary"),
        step(280, [.01, .01, .01], pressed=True),
    )
    result = subject.compare_prefix(
        obj, module, predictor(obj, {1: [.01, .10, .10]}),
        candidate_threshold=0.60,
    )
    assert result["status"] == "decision_disagreement"
    assert result["first_difference"]["step"] == 1
    assert result["first_difference"]["candidate_action"] == "HOLD"
    assert result["common_prefix_steps"] == 2


def test_different_pending_time_can_be_cancelled_without_executing():
    obj = run(
        step(0, [.001, .001, .001]),
        step(100, [.001, .1, .9], reason="pending_armed",
             pending_due=162.5),
        step(150, [.001, .1, .9], reason="pending_wait",
             pending_due=162.5),
        step(152, [.001, .001, .001], reason="pending_cancelled",
             pending_due=162.5),
        step(200, [.001, .001, .001]),
    )
    result = subject.compare_prefix(
        obj, module, predictor(obj, {1: [.001, .2, .9]}),
        candidate_threshold=0.60,
    )
    assert result["status"] == "no_executable_difference_observed"
    assert result["first_pending_plan_change"]["step"] == 1


def test_candidate_deadline_before_next_observation_is_uncertain():
    obj = run(
        step(0, [.001, .001, .001]),
        step(100, [.001, .1, .9], reason="pending_armed",
             pending_due=162.5),
        step(150, [.001, .1, .9], reason="pending_wait",
             pending_due=162.5),
        step(152, [.001, .001, .001], reason="pending_cancelled",
             pending_due=162.5),
    )
    result = subject.compare_prefix(
        obj, module, predictor(obj, {1: [.001, .5, .9]}),
        candidate_threshold=0.60,
    )
    assert result["status"] == "timer_uncertain"
    assert result["first_difference"]["kind"] == "unobserved_candidate_deadline_due"
    assert result["first_difference"]["step"] == 2


def test_recorded_timer_with_matching_candidate_due_is_consumed():
    obj = run(
        step(0, [.01, .01, .01]),
        step(100, [.001, .1, .9], reason="pending_armed", pending_due=162.5),
        step(150, [.001, .1, .9], reason="pending_wait", pending_due=162.5),
        step(180, [.99, .80, .01], reason="min_hold", pressed=True),
        step(300, [.01, .01, .01], pressed=True),
        deadlines=[{"timestamp_ms": 162.5, "action": "PRESS",
                    "pressed": True}],
    )
    result = subject.compare_prefix(
        obj, module, predictor(obj), candidate_threshold=0.60
    )
    assert result["status"] == "no_executable_difference_observed"
    assert result["recorded_deadlines_consumed"] == 1


def test_h200_baseline_fidelity_failure_blocks_candidate():
    obj = run(step(0, [.01, .01, .01]))
    result = subject.compare_prefix(
        obj, module, lambda i, s, before: ((.01, .50, .01), (.01, .01, .01)),
        candidate_threshold=0.60,
    )
    assert result["status"] == "blocked_model_fidelity"
    assert result["first_difference"]["kind"] == "h200_replay_mismatch"



def test_recorded_timer_with_candidate_deadline_after_observation_is_state_divergence():
    obj = run(
        step(0, [.01, .01, .01]),
        step(100, [.001, .1, .9], reason="pending_armed", pending_due=162.5),
        step(150, [.001, .1, .9], reason="pending_wait", pending_due=162.5),
        step(180, [.99, .80, .01], reason="min_hold", pressed=True),
        step(300, [.01, .01, .01], pressed=True),
        deadlines=[{"timestamp_ms": 162.5, "action": "PRESS",
                    "pressed": True}],
    )
    # The candidate's pending deadline moves to roughly 185.5ms.
    # At the recorded 180ms observation, a candidate callback cannot
    # already have executed.
    result = subject.compare_prefix(
        obj, module, predictor(obj, {1: [.001, .01, .7]}),
        candidate_threshold=0.60,
    )
    assert result["status"] == "state_divergence"
    assert result["first_difference"]["step"] == 3
    assert result["first_difference"]["kind"] == (
        "recorded_timer_transition_not_due_for_candidate"
    )
    assert result["first_difference"]["candidate_remaining_ms"] > 0


def test_recorded_timer_with_no_candidate_pending_is_state_divergence():
    obj = run(
        step(0, [.01, .01, .01]),
        step(100, [.001, .1, .9], reason="pending_armed", pending_due=162.5),
        step(150, [.001, .1, .9], reason="pending_wait", pending_due=162.5),
        step(180, [.99, .80, .01], reason="min_hold", pressed=True),
        deadlines=[{"timestamp_ms": 162.5, "action": "PRESS",
                    "pressed": True}],
    )
    result = subject.compare_prefix(
        obj, module, predictor(obj, {
            1: [.001, .01, .10],
            2: [.001, .01, .10],
        }), candidate_threshold=0.60,
    )
    assert result["status"] == "state_divergence"
    assert result["first_difference"]["candidate_due_ms"] is None
    assert result["first_difference"]["candidate_state_before_observation"] is False
    assert result["first_difference"]["recorded_state_before_observation"] is True
