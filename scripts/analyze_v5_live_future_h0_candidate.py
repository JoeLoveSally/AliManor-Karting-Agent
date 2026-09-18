#!/usr/bin/env python3
"""Find the first causal divergence of the coherent future-H0 live candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v5_live_horizons as live  # noqa: E402

from karting_agent.model.state_conditioned_runner import (  # noqa: E402
    StateConditionedModelRunner,
)
from karting_agent.runtime.consensus_dense_scheduler import (  # noqa: E402
    ConsensusDenseConfig,
    ConsensusDenseHorizonScheduler,
)
from karting_agent.runtime.consensus_transition_lifecycle import (  # noqa: E402
    ConsensusTransitionLifecycle,
)
from karting_agent.vision.preprocess import stack_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the coherent future-H0 candidate only while its physical "
            "action history matches the recorded run."
        )
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--recording", type=Path, default=None)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--min-state-hold-ms", type=float, default=100.0)
    parser.add_argument("--consensus-heads", type=int, default=3)
    parser.add_argument("--consensus-window-ms", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _recorded_transitions(run: dict[str, object]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    deadline_events = run.get("deadline_events", [])
    if isinstance(deadline_events, list):
        for raw in deadline_events:
            if not isinstance(raw, dict):
                continue
            events.append(
                {
                    "timestamp_ms": float(raw["timestamp_ms"]),
                    "pressed": bool(raw["pressed"]),
                    "source": str(raw.get("scheduler_reason", "deadline")),
                }
            )

    steps = run["steps"]
    assert isinstance(steps, list)
    for raw in steps:
        if not isinstance(raw, dict):
            continue
        action = str(raw.get("action", "HOLD"))
        if action not in {"PRESS", "RELEASE"}:
            continue
        events.append(
            {
                "timestamp_ms": float(raw["observation_timestamp_ms"]),
                "pressed": bool(raw["pressed"]),
                "source": str(raw.get("scheduler_reason", "primary")),
            }
        )
    events.sort(key=lambda item: float(item["timestamp_ms"]))
    return events


def _matches(
    candidate: dict[str, object],
    recorded: dict[str, object],
    *,
    tolerance_ms: float = 2.0,
) -> bool:
    return (
        bool(candidate["pressed"]) == bool(recorded["pressed"])
        and abs(
            float(candidate["timestamp_ms"])
            - float(recorded["timestamp_ms"])
        )
        <= tolerance_ms
    )


def main() -> int:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    if args.min_state_hold_ms < 0:
        raise ValueError("--min-state-hold-ms must be >= 0")

    run_path = args.run.resolve()
    run = live._load_run(run_path)
    runtime = run.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("run JSON has no runtime metadata")
    recording_path = live._recording_path(run, args.recording)

    model = StateConditionedModelRunner(
        args.model.resolve(),
        metadata_path=args.metadata.resolve(),
        device=args.device,
    )
    horizons = model.spec.prediction_horizons_ms
    if 0.0 not in horizons:
        raise ValueError("candidate requires an H0 prediction head")
    h0_index = horizons.index(0.0)

    scheduler = ConsensusDenseHorizonScheduler(
        ConsensusDenseConfig(
            horizons_ms=horizons,
            threshold=args.threshold,
            min_state_hold_ms=args.min_state_hold_ms,
            min_consensus_heads=args.consensus_heads,
            consensus_window_ms=args.consensus_window_ms,
        )
    )
    lifecycle = ConsensusTransitionLifecycle(
        threshold=args.threshold,
        min_state_hold_ms=args.min_state_hold_ms,
    )

    steps = run["steps"]
    assert isinstance(steps, list)
    required_indices: set[int] = set()
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            raise ValueError("runtime step must be a mapping")
        raw_indices = raw_step.get("input_frame_indices")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError("runtime step is missing input_frame_indices")
        required_indices.update(int(value) for value in raw_indices)
    frames = live._decode_required_frames(recording_path, required_indices)

    recorded_events = _recorded_transitions(run)
    recorded_index = 0
    state = bool(runtime.get("initial_pressed", False))
    matched_events: list[dict[str, object]] = []
    candidate_events: list[dict[str, object]] = []
    divergence: dict[str, object] | None = None

    def accept_candidate(event: dict[str, object]) -> bool:
        nonlocal recorded_index, divergence
        candidate_events.append(event)
        if recorded_index >= len(recorded_events):
            divergence = {
                "kind": "candidate_extra_transition",
                "candidate": event,
                "recorded": None,
            }
            return False
        recorded = recorded_events[recorded_index]
        if not _matches(event, recorded):
            divergence = {
                "kind": "candidate_transition_mismatch",
                "candidate": event,
                "recorded": recorded,
            }
            return False
        matched_events.append(
            {
                "candidate": event,
                "recorded": recorded,
            }
        )
        recorded_index += 1
        return True

    for step_index, raw_step in enumerate(steps):
        assert isinstance(raw_step, dict)
        timestamp_ms = float(raw_step["observation_timestamp_ms"])

        due_ms = scheduler.pending_due_ms
        if due_ms is not None and due_ms <= timestamp_ms + 1e-6:
            previous_state = state
            execution = scheduler.execute_pending_if_due(
                timestamp_ms=due_ms,
                current_pressed=state,
            )
            if execution is not None:
                state = not state
                lifecycle.begin(
                    executed_at_ms=due_ms,
                    previous_pressed=previous_state,
                    current_pressed=state,
                    evidence_lead_ms=execution.delay_ms,
                )
                event = {
                    "timestamp_ms": due_ms,
                    "pressed": state,
                    "source": "consensus_execute_deadline",
                    "step_index": step_index,
                    "armed_at_ms": execution.armed_at_ms,
                    "delay_ms": execution.delay_ms,
                    "support_heads_ms": list(execution.support_heads_ms),
                    "due_spread_ms": execution.due_spread_ms,
                }
                if not accept_candidate(event):
                    break

        if (
            recorded_index < len(recorded_events)
            and float(recorded_events[recorded_index]["timestamp_ms"])
            < timestamp_ms - 1e-6
        ):
            divergence = {
                "kind": "candidate_missing_transition",
                "candidate": None,
                "recorded": recorded_events[recorded_index],
                "step_index": step_index,
            }
            break

        frame_indices = tuple(int(value) for value in raw_step["input_frame_indices"])
        inputs = stack_frames(
            [frames[index] for index in frame_indices],
            model.spec.preprocess_config,
        )
        native_switch, future_action = model.predict_native_switch_and_future_all(
            inputs,
            state,
        )
        probabilities = list(float(value) for value in native_switch)
        desired_press_h0 = float(future_action[h0_index])
        projected_h0 = 1.0 - desired_press_h0 if state else desired_press_h0
        probabilities[h0_index] = projected_h0

        lifecycle_status: str | None = None
        previous_state_h0: float | None = None
        if lifecycle.active:
            pending = lifecycle.pending
            assert pending is not None
            previous_state_h0 = (
                1.0 - desired_press_h0
                if pending.previous_pressed
                else desired_press_h0
            )
            lifecycle_status = lifecycle.observe(
                timestamp_ms=timestamp_ms,
                previous_state_h0_probability=previous_state_h0,
            )
            if lifecycle_status in {"waiting", "confirmed"}:
                probabilities[h0_index] = 0.0

        decision = scheduler.update(
            timestamp_ms=timestamp_ms,
            probabilities=tuple(probabilities),
            current_pressed=state,
        )
        if decision.switch:
            previous_state = state
            state = not state
            if decision.reason == "consensus_execute":
                lifecycle.begin(
                    executed_at_ms=timestamp_ms,
                    previous_pressed=previous_state,
                    current_pressed=state,
                    evidence_lead_ms=float(decision.evidence_lead_ms or 0.0),
                )
            else:
                lifecycle.clear()
            event = {
                "timestamp_ms": timestamp_ms,
                "pressed": state,
                "source": decision.reason,
                "step_index": step_index,
                "source_frame": int(frame_indices[-1]),
                "desired_press_h0": desired_press_h0,
                "projected_h0": projected_h0,
                "lifecycle_status": lifecycle_status,
                "lifecycle_previous_state_h0": previous_state_h0,
            }
            if not accept_candidate(event):
                break

        if (
            recorded_index < len(recorded_events)
            and abs(
                float(recorded_events[recorded_index]["timestamp_ms"])
                - timestamp_ms
            )
            <= 1e-6
        ):
            divergence = {
                "kind": "candidate_missing_transition",
                "candidate": None,
                "recorded": recorded_events[recorded_index],
                "step_index": step_index,
                "source_frame": int(frame_indices[-1]),
                "desired_press_h0": desired_press_h0,
                "projected_h0": projected_h0,
                "lifecycle_status": lifecycle_status,
                "lifecycle_previous_state_h0": previous_state_h0,
            }
            break

    print(
        f"run={run_path.name} recording={recording_path.name} "
        f"device={model.device} matched_recorded_transitions={len(matched_events)}",
        flush=True,
    )
    for index, item in enumerate(matched_events):
        event = item["candidate"]
        assert isinstance(event, dict)
        print(
            f"  match[{index}] t={float(event['timestamp_ms']):.1f}ms "
            f"state={'PRESS' if event['pressed'] else 'RELEASE'} "
            f"source={event['source']}",
            flush=True,
        )

    if divergence is None:
        print("first causal divergence: none within recorded run", flush=True)
    else:
        print(
            "first causal divergence: "
            + json.dumps(divergence, sort_keys=True),
            flush=True,
        )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else run_path.with_name(
            run_path.stem + "_future_h0_first_divergence.json"
        )
    )
    payload = {
        "run": str(run_path),
        "recording": str(recording_path),
        "model": str(args.model.resolve()),
        "metadata": str(args.metadata.resolve()),
        "threshold": args.threshold,
        "min_state_hold_ms": args.min_state_hold_ms,
        "consensus_heads": args.consensus_heads,
        "consensus_window_ms": args.consensus_window_ms,
        "matched_recorded_transitions": matched_events,
        "first_causal_divergence": divergence,
        "candidate_events_until_divergence": candidate_events,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
