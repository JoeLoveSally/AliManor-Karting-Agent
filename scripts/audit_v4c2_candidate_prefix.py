#!/usr/bin/env python3
"""Read-only factual-prefix scheduler comparison for a V4-C2 history adapter.

The candidate sees *recorded* video and command history while both policies
still share the same physical state. A different immediate switch is an
on-record decision disagreement. An unresolvable candidate timer deadline is
reported as timer-order uncertainty, never as a confirmed actuator event.

No ADB, control executor, weights update, or Armed runtime imports.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v4c2_scheduler_replay import (  # noqa: E402
    audit, close_float, load_scheduler, make_scheduler,
)


def factual_pre_state(step: dict) -> bool:
    state = bool(step["pressed"])
    return not state if step["action"] in ("PRESS", "RELEASE") else state


def _as_probabilities(values, width: int) -> tuple[float, ...]:
    probs = tuple(float(x) for x in values)
    if len(probs) != width or any(not 0 <= x <= 1 for x in probs):
        raise ValueError("candidate probabilities have invalid shape/range")
    return probs


def compare_prefix(
    run: dict,
    scheduler_module,
    predict,
    *,
    candidate_threshold: float,
    timer_tolerance_ms: float = 0.10,
    max_replay_h200_error: float = 0.05,
    candidate_horizons: str = "full",
) -> dict:
    """Compare recorded baseline and candidate until the first unsafe prefix.

    predict(index, step, pressed_before) -> (base_all, candidate_all).
    Only baseline *logged* probabilities drive the reference scheduler.
    Baseline re-inference is a fidelity check, not its control signal.
    """
    if not 0 < candidate_threshold < 1:
        raise ValueError("candidate threshold must be in (0, 1)")
    if candidate_horizons not in {"full", "control_only"}:
        raise ValueError("candidate_horizons must be full or control_only")
    if timer_tolerance_ms < 0 or max_replay_h200_error < 0:
        raise ValueError("tolerances must be non-negative")

    gate = audit(run, scheduler_module, timer_policy="recorded")
    if not gate["parity_passed"]:
        return {"status": "blocked_baseline_parity", "gate": gate}

    baseline, _ = make_scheduler(run, scheduler_module)
    original_cfg = baseline.config
    cfg = replace(original_cfg, threshold=float(candidate_threshold))
    candidate = scheduler_module.MultiHorizonSwitchScheduler(cfg)
    horizons = tuple(float(v) for v in original_cfg.horizons_ms)
    control_index = horizons.index(float(original_cfg.control_horizon_ms))
    recorded_deadlines = sorted(
        run.get("deadline_events", []), key=lambda item: float(item["timestamp_ms"])
    )
    pending_change = None
    state = bool(run["runtime"].get("initial_pressed", False))
    consumed_deadlines = 0
    checked = 0
    previous_t = float("-inf")

    def result(status, kind=None, *, index=None, t=None, details=None):
        return {
            "status": status,
            "first_difference": (
                {"kind": kind, "step": index, "observation_ms": t, **(details or {})}
                if kind is not None else None
            ),
            "first_pending_plan_change": pending_change,
            "common_prefix_steps": checked,
            "recorded_deadlines_consumed": consumed_deadlines,
            "candidate_threshold_all_horizons": candidate_threshold,
            "candidate_horizons": candidate_horizons,
            "baseline_threshold_all_horizons": float(original_cfg.threshold),
            "warning": (
                "This compares a factual original-policy prefix, not a new "
                "closed-loop trajectory. Candidate timer callbacks may race "
                "with observations. H200-selected 0.76 is not calibrated "
                "for all horizons."
            ),
        }

    for index, step in enumerate(run["steps"]):
        t = float(step["observation_timestamp_ms"])
        before = factual_pre_state(step)
        if t <= previous_t:
            return result("blocked", "observation_order", index=index, t=t)
        previous_t = t

        # When the recorded pre-state changes between observations, the actual
        # run records which timer won the race. Candidate must have the *same*
        # pending deadline to reuse this factual callback; otherwise stop.
        while before != state:
            if consumed_deadlines >= len(recorded_deadlines):
                return result("blocked", "missing_recorded_callback",
                              index=index, t=t)
            logged = recorded_deadlines[consumed_deadlines]
            due = float(logged["timestamp_ms"])
            base_due = baseline.pending_due_ms
            cand_due = candidate.pending_due_ms
            if due > t + 1e-6 or base_due is None or not close_float(
                due, base_due, timer_tolerance_ms
            ):
                return result("blocked", "reference_callback_inconsistent",
                              index=index, t=t,
                              details={"logged_due_ms": due, "base_due_ms": base_due})
            if cand_due is None or cand_due > t + 1e-6:
                # Since no candidate primary switch occurred on the shared
                # prefix, and the candidate pending deadline has not yet
                # arrived (or is absent), it cannot have reached the recorded
                # post-timer state by this observation. This is an observed-
                # state divergence under the scheduler model, not a guess
                # about when a callback thread might have run.
                return result("state_divergence", "recorded_timer_transition_not_due_for_candidate",
                              index=index, t=t,
                              details={"recorded_due_ms": due,
                                       "candidate_due_ms": cand_due,
                                       "candidate_remaining_ms": (
                                           None if cand_due is None else cand_due - t),
                                       "recorded_action": logged["action"],
                                       "recorded_state_before_observation": before,
                                       "candidate_state_before_observation": state})
            if not close_float(due, cand_due, timer_tolerance_ms):
                # Candidate deadline is in the past, but a callback may have
                # lost a race with the model observation. The original timer
                # callback order cannot be copied across a changed deadline.
                return result("timer_uncertain", "candidate_callback_order_unknown",
                              index=index, t=t,
                              details={"recorded_due_ms": due,
                                       "candidate_due_ms": cand_due,
                                       "recorded_action": logged["action"]})
            base_fired = baseline.execute_pending_if_due(
                timestamp_ms=due, current_pressed=state
            )
            cand_fired = candidate.execute_pending_if_due(
                timestamp_ms=due, current_pressed=state
            )
            if base_fired is None or cand_fired is None:
                return result("timer_uncertain", "pending_callback_replay_failed",
                              index=index, t=t,
                              details={"recorded_due_ms": due})
            state = not state
            consumed_deadlines += 1
            if (bool(logged["pressed"]) != state
                    or logged["action"] != ("PRESS" if state else "RELEASE")):
                return result("blocked", "logged_callback_state_invalid",
                              index=index, t=t)

        # A candidate-only deadline may have become runnable before this
        # observation. It might fire, or this frame might cancel it. Do not
        # assume the recorded callback ordering still applies to that timer.
        base_due = baseline.pending_due_ms
        cand_due = candidate.pending_due_ms
        if cand_due is not None and cand_due < t - 1e-6:
            if base_due is None or not close_float(
                cand_due, base_due, timer_tolerance_ms
            ):
                return result("timer_uncertain", "unobserved_candidate_deadline_due",
                              index=index, t=t,
                              details={"candidate_due_ms": cand_due,
                                       "reference_due_ms": base_due})
        if state != before:
            return result("blocked", "factual_pre_state_mismatch",
                          index=index, t=t,
                          details={"recorded_pressed": before,
                                   "replayed_pressed": state})

        logged_all = _as_probabilities(step["probabilities"], len(horizons))
        replayed_all, candidate_all = predict(index, step, before)
        replayed_all = _as_probabilities(replayed_all, len(horizons))
        candidate_all = _as_probabilities(candidate_all, len(horizons))
        if candidate_horizons == "control_only":
            # Diagnostic ablation: keep the recorded baseline H100/H300
            # *exactly* and substitute only residual H200. This is not a
            # deployment configuration or separate per-horizon threshold.
            candidate_all = tuple(
                candidate_all[j] if j == control_index else logged_all[j]
                for j in range(len(horizons))
            )
        replay_err = abs(replayed_all[control_index] - logged_all[control_index])
        if replay_err > max_replay_h200_error:
            return result("blocked_model_fidelity", "h200_replay_mismatch",
                          index=index, t=t,
                          details={"logged_h200": logged_all[control_index],
                                   "replayed_h200": replayed_all[control_index],
                                   "absolute_error": replay_err})

        base_decision = baseline.update(
            timestamp_ms=t, probabilities=logged_all, current_pressed=state
        )
        candidate_decision = candidate.update(
            timestamp_ms=t, probabilities=candidate_all, current_pressed=state
        )
        logged_switch = step["action"] in ("PRESS", "RELEASE")
        if (base_decision.switch != logged_switch
                or base_decision.reason != step.get("scheduler_reason")):
            return result("blocked", "reference_decision_mismatch",
                          index=index, t=t)
        checked += 1
        if candidate_decision.switch != logged_switch:
            return result("decision_disagreement", "immediate_switch_difference",
                          index=index, t=t,
                          details={
                              "pressed_before": state,
                              "recorded_action": step["action"],
                              "candidate_action": (
                                  ("RELEASE" if state else "PRESS")
                                  if candidate_decision.switch else "HOLD"
                              ),
                              "recorded_reason": step.get("scheduler_reason"),
                              "candidate_reason": candidate_decision.reason,
                              "logged_probabilities": logged_all,
                              "candidate_probabilities": candidate_all,
                              "baseline_pending_due_ms": base_decision.pending_due_ms,
                              "candidate_pending_due_ms": candidate_decision.pending_due_ms,
                          })

        if pending_change is None and not close_float(
            base_decision.pending_due_ms,
            candidate_decision.pending_due_ms,
            timer_tolerance_ms,
        ):
            pending_change = {
                "step": index,
                "observation_ms": t,
                "recorded_reason": base_decision.reason,
                "candidate_reason": candidate_decision.reason,
                "reference_due_ms": base_decision.pending_due_ms,
                "candidate_due_ms": candidate_decision.pending_due_ms,
                "logged_probabilities": logged_all,
                "candidate_probabilities": candidate_all,
            }

        if logged_switch:
            state = not state

    # A final pending mismatch is a planning difference, not an executed act.
    if consumed_deadlines < len(recorded_deadlines):
        return result("timer_uncertain", "callbacks_after_last_observation",
                      index=len(run["steps"]), t=previous_t,
                      details={"unconsumed_recorded_callbacks":
                               len(recorded_deadlines) - consumed_deadlines})
    return result("no_executable_difference_observed")


def _model_predictor(run, json_path: Path, runner):
    # Import CV2 only when processing real MP4s; pure scheduler tests remain
    # independent of torch and video dependencies.
    from shadow_v4c2_history_runs import frames_by_index, preflight
    from karting_agent.vision.preprocess import stack_frames

    initial, times, transitions, count = preflight(run)
    spec = runner.baseline.spec
    runtime = run["runtime"]
    if list(runtime["frame_offsets_ms"]) != list(spec.frame_offsets_ms):
        raise ValueError("recorded frame offsets differ from model")
    if list(runtime["prediction_horizons_ms"]) != list(spec.prediction_horizons_ms):
        raise ValueError("recorded prediction horizons differ from model")
    if any(len(s["input_frame_indices"]) != runner.frame_stack for s in run["steps"]):
        raise ValueError("recorded frame stack mismatch")
    mp4 = json_path.with_suffix(".mp4")
    needed = {int(n) for step in run["steps"] for n in step["input_frame_indices"]}
    frames = frames_by_index(mp4, needed, count, run["recording"]["resolution"])

    def predict(index, step, pressed_before):
        observed = float(step["observation_timestamp_ms"])
        indices = [int(x) for x in step["input_frame_indices"]]
        inputs = stack_frames(
            [frames[x] for x in indices], spec.preprocess_config
        )
        base_all = runner.baseline.predict_switch_all(inputs, pressed_before)
        adapted_all = runner.predict_switch_all(
            inputs,
            frame_timestamps_ms=step["input_frame_timestamps_ms"],
            observation_timestamp_ms=observed,
            initial_pressed=initial,
            transitions=transitions[:bisect_left(times, observed)],
            current_pressed=pressed_before,
        )
        return base_all, adapted_all

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "artifacts/models/mobilenet_v3_small_v4c2_temporal_v2"
    adapter = ROOT / "artifacts/models/mobilenet_v3_small_v4c2_history_residual_v1"
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--scheduler-source", type=Path, required=True)
    parser.add_argument("--candidate-thresholds", type=float, nargs="+",
                        default=[0.60, 0.76])
    parser.add_argument("--candidate-horizons", nargs="+",
                        choices=("full", "control_only"), default=["full", "control_only"])
    parser.add_argument("--base-model", type=Path, default=base / "model.pt")
    parser.add_argument("--base-metadata", type=Path, default=base / "metadata_h200.json")
    parser.add_argument("--adapter", type=Path, default=adapter / "history_adapter.pt")
    parser.add_argument("--adapter-metadata", type=Path,
                        default=adapter / "history_adapter_metadata.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"report exists: {args.output}")

    from karting_agent.model.action_history_runner import ActionHistoryAdapterRunner
    runner = ActionHistoryAdapterRunner(
        base_model_path=args.base_model,
        base_metadata_path=args.base_metadata,
        adapter_path=args.adapter,
        adapter_metadata_path=args.adapter_metadata,
        device=args.device,
    )
    module = load_scheduler(args.scheduler_source)
    report = {
        "analysis": "factual-prefix decisions and timer uncertainty, NOT autonomous candidate outcome",
        "scheduler_source": str(args.scheduler_source.resolve()),
        "scheduler_sha256": hashlib.sha256(args.scheduler_source.read_bytes()).hexdigest(),
        "base_model_sha256": hashlib.sha256(args.base_model.read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
        "runs": {},
    }
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        predictor = _model_predictor(run, path, runner)
        per_threshold = {}
        for horizon_variant in args.candidate_horizons:
            for threshold in args.candidate_thresholds:
                key = f"{horizon_variant}:{threshold:g}"
                per_threshold[key] = compare_prefix(
                    run, module, predictor, candidate_threshold=threshold,
                    candidate_horizons=horizon_variant,
                )
                r = per_threshold[key]
                print(json.dumps({
                    "run": path.stem, "variant": horizon_variant,
                    "threshold": threshold, "status": r["status"],
                    "prefix_steps": r.get("common_prefix_steps"),
                    "first_difference": r.get("first_difference"),
                    "first_pending_plan_change": r.get("first_pending_plan_change"),
                }, ensure_ascii=False), flush=True)
        report["runs"][path.stem] = per_threshold
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved factual-prefix audit: {args.output}")


if __name__ == "__main__":
    main()
