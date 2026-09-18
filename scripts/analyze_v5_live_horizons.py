#!/usr/bin/env python3
"""Replay a recorded V5-H0 run and recover all dense switch-horizon probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.model.state_conditioned_runner import (  # noqa: E402
    StateConditionedModelRunner,
)
from karting_agent.runtime.consensus_dense_scheduler import (  # noqa: E402
    ConsensusDenseConfig,
    ConsensusDenseHorizonScheduler,
)
from karting_agent.train.live_horizon_analysis import (  # noqa: E402
    crossing_lead_ms,
    first_threshold_crossing,
)
from karting_agent.vision.preprocess import stack_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct all V5 dense-horizon probabilities from a recorded live run. "
            "The pre-first-switch comparison is causal because every candidate sees "
            "the exact same visuals and physical state up to the first executed switch."
        )
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--recording", type=Path, default=None)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--window-steps", type=int, default=12)
    parser.add_argument("--min-state-hold-ms", type=float, default=100.0)
    parser.add_argument("--consensus-heads", type=int, default=3)
    parser.add_argument("--consensus-window-ms", type=float, default=60.0)
    return parser.parse_args()


def _load_run(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run JSON root must be a mapping")
    recording = payload.get("recording")
    if not isinstance(recording, dict):
        raise ValueError("run JSON has no recording metadata")
    if not bool(recording.get("frame_mapping_valid", False)):
        raise ValueError("run recording/source-frame mapping is not valid")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("run JSON contains no runtime steps")
    return payload


def _recording_path(run: dict[str, object], override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    recording = run["recording"]
    assert isinstance(recording, dict)
    raw = recording.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("run recording path is missing")
    return Path(raw).resolve()


def _decode_required_frames(path: Path, required_indices: set[int]) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"recording not found: {path}")
    if not required_indices:
        raise ValueError("no source frames requested")
    if min(required_indices) < 0:
        raise ValueError("source frame indices must be >= 0")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open recording: {path}")

    frames: dict[int, np.ndarray] = {}
    max_index = max(required_indices)
    try:
        index = 0
        while index <= max_index:
            ok, frame = capture.read()
            if not ok:
                break
            if index in required_indices:
                frames[index] = frame
            index += 1
    finally:
        capture.release()

    missing = sorted(required_indices - set(frames))
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise RuntimeError(
            f"recording is missing {len(missing)} required source frame(s): {preview}"
        )
    return frames


def _state_after(action: str, state_before: bool) -> bool:
    if action == "PRESS":
        return True
    if action == "RELEASE":
        return False
    if action == "HOLD":
        return state_before
    raise ValueError(f"unsupported runtime action: {action!r}")


def main() -> int:
    args = parse_args()
    if args.window_steps < 1:
        raise ValueError("--window-steps must be >= 1")
    if args.min_state_hold_ms < 0:
        raise ValueError("--min-state-hold-ms must be >= 0")
    if args.consensus_heads < 2:
        raise ValueError("--consensus-heads must be >= 2")
    if args.consensus_window_ms < 0:
        raise ValueError("--consensus-window-ms must be >= 0")

    run_path = args.run.resolve()
    run = _load_run(run_path)
    runtime = run.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("run JSON has no runtime metadata")
    threshold = (
        float(runtime.get("switch_threshold", 0.6))
        if args.threshold is None
        else float(args.threshold)
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")

    recording_path = _recording_path(run, args.recording)
    model = StateConditionedModelRunner(
        args.model.resolve(),
        metadata_path=args.metadata.resolve(),
        device=args.device,
    )
    horizons = model.spec.prediction_horizons_ms
    if 0.0 not in horizons:
        raise ValueError("analysis requires an H0 output")
    h0_index = horizons.index(0.0)

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
    frames = _decode_required_frames(recording_path, required_indices)

    initial_pressed = bool(runtime.get("initial_pressed", False))
    state = initial_pressed
    rows: list[dict[str, object]] = []
    reproduction_errors: list[float] = []

    for step_index, raw_step in enumerate(steps):
        assert isinstance(raw_step, dict)
        frame_indices = tuple(int(value) for value in raw_step["input_frame_indices"])
        inputs = stack_frames(
            [frames[index] for index in frame_indices],
            model.spec.preprocess_config,
        )
        probabilities = model.predict_switch_all(inputs, state)
        runtime_probability = float(raw_step["probability"])
        reproduction_errors.append(abs(probabilities[h0_index] - runtime_probability))

        action = str(raw_step["action"])
        after = _state_after(action, state)
        recorded_after = bool(raw_step["pressed"])
        if after != recorded_after:
            raise RuntimeError(
                f"state reconstruction mismatch at step {step_index}: "
                f"action={action} before={state} calculated_after={after} "
                f"recorded_after={recorded_after}"
            )

        rows.append(
            {
                "step_index": step_index,
                "timestamp_ms": float(raw_step["observation_timestamp_ms"]),
                "source_frame": int(frame_indices[-1]),
                "state_before": state,
                "state_after": after,
                "action": action,
                "reason": raw_step.get("scheduler_reason"),
                "runtime_h0_probability": runtime_probability,
                "probabilities": list(probabilities),
            }
        )
        state = after

    primary_indices = [
        index
        for index, row in enumerate(rows)
        if row["action"] in {"PRESS", "RELEASE"}
    ]
    if not primary_indices:
        raise ValueError("run contains no executed state transition")
    first_switch_index = primary_indices[0]
    first_switch = rows[first_switch_index]
    first_switch_timestamp = float(first_switch["timestamp_ms"])

    # Up to and including the first executed switch all candidates see the same
    # visual trajectory and RELEASE/PRESS physical state as the actual run.
    causal_rows = rows[: first_switch_index + 1]
    causal_timestamps = [float(row["timestamp_ms"]) for row in causal_rows]
    crossings: dict[str, object] = {}
    reference_crossing = first_threshold_crossing(
        causal_timestamps,
        [float(row["probabilities"][h0_index]) for row in causal_rows],
        threshold=threshold,
    )
    if reference_crossing is None:
        raise RuntimeError("H0 did not cross threshold at the first executed switch")
    _, reference_timestamp = reference_crossing

    causal_scheduler = ConsensusDenseHorizonScheduler(
        ConsensusDenseConfig(
            horizons_ms=horizons,
            threshold=threshold,
            min_state_hold_ms=float(args.min_state_hold_ms),
            min_consensus_heads=int(args.consensus_heads),
            consensus_window_ms=float(args.consensus_window_ms),
        )
    )
    causal_candidate: dict[str, object] | None = None
    causal_state = initial_pressed
    for row in causal_rows:
        timestamp_ms = float(row["timestamp_ms"])
        due_ms = causal_scheduler.pending_due_ms
        if due_ms is not None and due_ms <= timestamp_ms + 1e-6:
            execution = causal_scheduler.execute_pending_if_due(
                timestamp_ms=due_ms,
                current_pressed=causal_state,
            )
            if execution is not None:
                causal_candidate = {
                    "timestamp_ms": due_ms,
                    "reason": "consensus_execute_deadline",
                    "state_before": causal_state,
                    "state_after": not causal_state,
                    "armed_at_ms": execution.armed_at_ms,
                    "delay_ms": execution.delay_ms,
                    "support_heads_ms": list(execution.support_heads_ms),
                    "due_spread_ms": execution.due_spread_ms,
                }
                break

        decision = causal_scheduler.update(
            timestamp_ms=timestamp_ms,
            probabilities=tuple(float(value) for value in row["probabilities"]),
            current_pressed=causal_state,
        )
        if decision.switch:
            causal_candidate = {
                "timestamp_ms": timestamp_ms,
                "reason": decision.reason,
                "state_before": causal_state,
                "state_after": not causal_state,
                "armed_at_ms": None,
                "delay_ms": decision.evidence_lead_ms,
                "support_heads_ms": list(decision.consensus_heads_ms),
                "due_spread_ms": decision.consensus_due_spread_ms,
            }
            break

    if causal_candidate is None:
        due_ms = causal_scheduler.pending_due_ms
        if due_ms is not None and due_ms < first_switch_timestamp - 1e-6:
            execution = causal_scheduler.execute_pending_if_due(
                timestamp_ms=due_ms,
                current_pressed=causal_state,
            )
            if execution is not None:
                causal_candidate = {
                    "timestamp_ms": due_ms,
                    "reason": "consensus_execute_deadline",
                    "state_before": causal_state,
                    "state_after": not causal_state,
                    "armed_at_ms": execution.armed_at_ms,
                    "delay_ms": execution.delay_ms,
                    "support_heads_ms": list(execution.support_heads_ms),
                    "due_spread_ms": execution.due_spread_ms,
                }

    print(
        f"run={run_path.name} recording={recording_path.name} "
        f"steps={len(rows)} device={model.device} threshold={threshold:.2f}",
        flush=True,
    )
    print(
        "runtime reproduction: "
        f"h0_abs_error_mean={np.mean(reproduction_errors):.6f} "
        f"p95={np.percentile(reproduction_errors, 95):.6f} "
        f"max={np.max(reproduction_errors):.6f}",
        flush=True,
    )
    print(
        "first executed switch: "
        f"step={first_switch_index} source_frame={first_switch['source_frame']} "
        f"t={first_switch_timestamp:.1f}ms action={first_switch['action']} "
        f"h0={float(first_switch['probabilities'][h0_index]):.3f}",
        flush=True,
    )
    if causal_candidate is None:
        print("causal consensus first switch: no earlier candidate", flush=True)
    else:
        candidate_timestamp = float(causal_candidate["timestamp_ms"])
        lead_vs_actual = first_switch_timestamp - candidate_timestamp
        heads = ",".join(
            f"h{float(value):g}"
            for value in causal_candidate["support_heads_ms"]
        )
        print(
            "causal consensus first switch: "
            f"t={candidate_timestamp:.1f}ms "
            f"reason={causal_candidate['reason']} "
            f"lead_vs_actual={lead_vs_actual:+.1f}ms "
            f"armed_at={causal_candidate['armed_at_ms']} "
            f"delay={causal_candidate['delay_ms']} "
            f"heads=[{heads}] spread={causal_candidate['due_spread_ms']}",
            flush=True,
        )
    print("pre-first-switch threshold crossings:", flush=True)

    for horizon_index, horizon_ms in enumerate(horizons):
        probabilities = [
            float(row["probabilities"][horizon_index]) for row in causal_rows
        ]
        crossing = first_threshold_crossing(
            causal_timestamps,
            probabilities,
            threshold=threshold,
        )
        if crossing is None:
            crossings[str(horizon_ms)] = None
            print(f"  h{horizon_ms:g}: no crossing", flush=True)
            continue
        crossing_index, crossing_timestamp = crossing
        row = causal_rows[crossing_index]
        lead_ms = crossing_lead_ms(reference_timestamp, crossing_timestamp)
        crossings[str(horizon_ms)] = {
            "step_index": int(row["step_index"]),
            "source_frame": int(row["source_frame"]),
            "timestamp_ms": crossing_timestamp,
            "probability": probabilities[crossing_index],
            "lead_vs_h0_ms": lead_ms,
        }
        print(
            f"  h{horizon_ms:g}: source_frame={row['source_frame']} "
            f"t={crossing_timestamp:.1f}ms p={probabilities[crossing_index]:.3f} "
            f"lead_vs_h0={lead_ms:+.1f}ms",
            flush=True,
        )

    window_start = max(0, first_switch_index - args.window_steps)
    print("dense probabilities before first switch:", flush=True)
    header = "  frame      t_ms state " + " ".join(
        f"h{horizon:g}".rjust(7) for horizon in horizons
    )
    print(header, flush=True)
    for row in rows[window_start : first_switch_index + 1]:
        values = " ".join(
            f"{float(value):7.3f}" for value in row["probabilities"]
        )
        print(
            f"  {int(row['source_frame']):5d} "
            f"{float(row['timestamp_ms']):9.1f} "
            f"{'P' if row['state_before'] else 'R':>5} {values}",
            flush=True,
        )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else run_path.with_name(run_path.stem + "_dense_horizons.json")
    )
    payload = {
        "run": str(run_path),
        "recording": str(recording_path),
        "model": str(args.model.resolve()),
        "metadata": str(args.metadata.resolve()),
        "threshold": threshold,
        "horizons_ms": list(horizons),
        "runtime_reproduction": {
            "h0_abs_error_mean": float(np.mean(reproduction_errors)),
            "h0_abs_error_p95": float(np.percentile(reproduction_errors, 95)),
            "h0_abs_error_max": float(np.max(reproduction_errors)),
        },
        "first_executed_switch": first_switch,
        "pre_first_switch_crossings": crossings,
        "causal_consensus_first_switch": causal_candidate,
        "steps": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
