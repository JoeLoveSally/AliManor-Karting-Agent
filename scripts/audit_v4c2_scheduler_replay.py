#!/usr/bin/env python3
"""Audit recorded V4-C2 scheduler decisions without model inference or ADB.

This is a deterministic, nominal-deadline replay. It does NOT reproduce wall-clock
inference latency or guarantee timer-thread ordering near an observation. A
mismatch blocks downstream counterfactual scheduler claims; no recorded action
may be silently replaced with a simulated action.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULER = ROOT / "src/karting_agent/runtime/multi_horizon_scheduler.py"


def load_scheduler(path: Path):
    """Load one explicit source file; never overwrite the live runtime module."""
    path = path.resolve(strict=True)
    spec = importlib.util.spec_from_file_location("_v4c2_scheduler_audit_source", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load scheduler source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # required for dataclasses with postponed hints
    spec.loader.exec_module(module)
    return module


def make_scheduler(run: dict, module):
    raw = run.get("runtime", {}).get("multi_horizon_scheduler")
    if not isinstance(raw, dict):
        raise ValueError("record has no multi-horizon scheduler config")
    if not run["runtime"].get("execute_pending_at_due"):
        raise ValueError("audit supports only recorded deadline-timer runs")
    cls = module.MultiHorizonSchedulerConfig
    names = {field.name for field in fields(cls)}
    extras = {key: value for key, value in raw.items() if key not in names}
    # An older source may omit newer opt-in flags; never drop active behavior.
    harmless = {
        "execute_short_horizon_overdue": False,
        "reserve_bounded_reversal_during_min_hold": False,
        "anticipation_direction": "both",
    }
    unsupported = {key: value for key, value in extras.items()
                   if key not in harmless or value != harmless[key]}
    if unsupported:
        raise ValueError(f"scheduler source lacks recorded options: {unsupported}")
    kwargs = {key: value for key, value in raw.items() if key in names}
    kwargs["horizons_ms"] = tuple(float(v) for v in raw["horizons_ms"])
    cfg = cls(**kwargs)
    cfg.validate()
    return module.MultiHorizonSwitchScheduler(cfg), extras


def close_float(left, right, tolerance_ms=0.10):
    return (left is None and right is None) or (
        left is not None and right is not None
        and math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance_ms)
    )


def audit(run: dict, module, *, max_errors: int = 5):
    scheduler, ignored_default_options = make_scheduler(run, module)
    steps = run.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("record lacks steps")
    deadlines = sorted(run.get("deadline_events", []), key=lambda item: float(item["timestamp_ms"]))
    errors: list[dict] = []
    emitted: list[dict] = []
    state = bool(run["runtime"].get("initial_pressed", False))
    last_observation = float("-inf")
    deadline_index = 0
    step_checked = 0

    def fail(kind, index, expected, actual):
        errors.append({"kind": kind, "index": index, "expected": expected, "replayed": actual})

    for i, step in enumerate(steps):
        t = float(step["observation_timestamp_ms"])
        if not math.isfinite(t) or t <= last_observation:
            fail("observation_order", i, f"> {last_observation}", t)
            break
        last_observation = t
        # Nominal timer callbacks occurring before this observation. An actual
        # callback can be late; that uncertainty is exposed as a parity failure.
        pending_at = scheduler.pending_due_ms
        if pending_at is not None and float(pending_at) < t - 1e-6:
            fired = scheduler.execute_pending_if_due(
                timestamp_ms=float(pending_at), current_pressed=state)
            if fired is None:
                fail("timer_did_not_fire", i, pending_at, None)
                break
            state = not state
            event = {"timestamp_ms": float(pending_at),
                     "action": "PRESS" if state else "RELEASE", "pressed": state}
            emitted.append(event)
            if deadline_index >= len(deadlines):
                fail("unexpected_timer_event", i, None, event)
                break
            recorded = deadlines[deadline_index]
            deadline_index += 1
            if (not close_float(recorded["timestamp_ms"], event["timestamp_ms"])
                    or recorded["action"] != event["action"]
                    or bool(recorded["pressed"]) != event["pressed"]):
                fail("timer_event_mismatch", i, recorded, event)
                break
        if deadline_index < len(deadlines) and float(deadlines[deadline_index]["timestamp_ms"]) < t - 1e-6:
            fail("missing_recorded_timer_event", i, deadlines[deadline_index], None)
            break
        expected_before = (not bool(step["pressed"]) if step["action"] in ("PRESS", "RELEASE")
                           else bool(step["pressed"]))
        if state != expected_before:
            fail("pre_state_mismatch", i, int(expected_before), int(state))
            break
        decision = scheduler.update(timestamp_ms=t,
                                    probabilities=step["probabilities"],
                                    current_pressed=state)
        action = ("PRESS" if not state else "RELEASE") if decision.switch else "HOLD"
        if decision.switch:
            state = not state
        actual = {"action": action, "pressed": state, "reason": decision.reason,
                  "pending_due_ms": decision.pending_due_ms}
        expected = {"action": step["action"], "pressed": bool(step["pressed"]),
                    "reason": step.get("scheduler_reason"),
                    "pending_due_ms": step.get("pending_due_ms")}
        step_checked += 1
        if (expected["action"] != actual["action"]
                or expected["pressed"] != actual["pressed"]
                or expected["reason"] != actual["reason"]
                or not close_float(expected["pending_due_ms"], actual["pending_due_ms"])):
            fail("step_mismatch", i, expected, actual)
            break
        if len(errors) >= max_errors:
            break

    if not errors and deadline_index < len(deadlines):
        fail("unmatched_deadline_events_after_last_step", len(steps),
             deadlines[deadline_index:deadline_index + max_errors], None)
    return {
        "steps_total": len(steps), "steps_checked": step_checked,
        "logged_deadlines": len(deadlines), "replayed_deadlines": len(emitted),
        "recorded_optional_defaults_not_in_source": ignored_default_options,
        "parity_passed": not errors and step_checked == len(steps),
        "first_errors": errors[:max_errors],
        "warning": "Nominal timer replay only; source revision and callback timing must match original. No candidate model was evaluated.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--scheduler-source", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/analysis/v4c2_scheduler_parity.json")
    args = parser.parse_args()
    source = args.scheduler_source.resolve(strict=True)
    module = load_scheduler(source)
    report = {"scheduler_source": str(source),
              "scheduler_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "runs": {}}
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        try:
            result = audit(run, module)
        except (ValueError, KeyError, TypeError) as exc:
            result = {"parity_passed": False, "blocked": str(exc)}
        report["runs"][path.stem] = result
        print(json.dumps({"run": path.stem, **result}, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved scheduler parity gate: {args.output}")


if __name__ == "__main__":
    main()
