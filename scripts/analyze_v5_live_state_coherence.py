#!/usr/bin/env python3
"""Analyze V5 live H0 state consistency on a recorded runtime trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

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
from karting_agent.runtime.h0_coherence import (  # noqa: E402
    desired_press_from_switch_pair,
)
from karting_agent.vision.preprocess import stack_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze V5 H0 conditioned-state contradictions on a live run."
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--recording", type=Path, default=None)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _state_before(raw_step: dict[str, object]) -> bool:
    action = str(raw_step["action"])
    state_after = bool(raw_step["pressed"])
    if action == "HOLD":
        return state_after
    if action == "PRESS":
        if not state_after:
            raise ValueError("PRESS step must end in pressed state")
        return False
    if action == "RELEASE":
        if state_after:
            raise ValueError("RELEASE step must end in released state")
        return True
    raise ValueError(f"unsupported runtime action: {action!r}")


def main() -> int:
    args = parse_args()
    run_path = args.run.resolve()
    run = live._load_run(run_path)
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

    recording_path = live._recording_path(run, args.recording)
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
    frames = live._decode_required_frames(recording_path, required_indices)

    rows: list[dict[str, object]] = []
    reproduction_errors: list[float] = []
    complement_errors: list[float] = []

    for step_index, raw_step in enumerate(steps):
        assert isinstance(raw_step, dict)
        frame_indices = tuple(int(value) for value in raw_step["input_frame_indices"])
        inputs = stack_frames(
            [frames[index] for index in frame_indices],
            model.spec.preprocess_config,
        )
        switch_if_release = model.predict_switch_all(inputs, False)
        switch_if_press = model.predict_switch_all(inputs, True)
        future_press = model.predict_future_all(inputs)

        release_h0 = float(switch_if_release[h0_index])
        press_h0 = float(switch_if_press[h0_index])
        state_before = _state_before(raw_step)
        native_h0 = press_h0 if state_before else release_h0
        raw_runtime_h0 = raw_step.get("raw_control_probability")
        if raw_runtime_h0 is None:
            raw_runtime_h0 = raw_step["probability"]
        reproduction_errors.append(abs(native_h0 - float(raw_runtime_h0)))

        desired_press_pair = float(
            desired_press_from_switch_pair(
                np.asarray([release_h0]),
                np.asarray([press_h0]),
            )[0]
        )
        desired_press_future = float(future_press[h0_index])
        paired_current_switch = (
            1.0 - desired_press_pair if state_before else desired_press_pair
        )
        future_current_switch = (
            1.0 - desired_press_future
            if state_before
            else desired_press_future
        )
        complement_error = abs(release_h0 + press_h0 - 1.0)
        complement_errors.append(complement_error)
        both_switch = release_h0 >= threshold and press_h0 >= threshold
        both_keep = release_h0 < threshold and press_h0 < threshold

        rows.append(
            {
                "step_index": step_index,
                "timestamp_ms": float(raw_step["observation_timestamp_ms"]),
                "source_frame": int(frame_indices[-1]),
                "state_before": state_before,
                "state_after": bool(raw_step["pressed"]),
                "action": str(raw_step["action"]),
                "scheduler_reason": raw_step.get("scheduler_reason"),
                "lifecycle_status": raw_step.get("lifecycle_status"),
                "switch_if_release_h0": release_h0,
                "switch_if_press_h0": press_h0,
                "native_current_h0": native_h0,
                "paired_desired_press_h0": desired_press_pair,
                "paired_current_switch_h0": paired_current_switch,
                "future_action_desired_press_h0": desired_press_future,
                "future_action_current_switch_h0": future_current_switch,
                "both_switch": both_switch,
                "both_keep": both_keep,
                "complement_error": complement_error,
            }
        )

    both_switch_rows = [row for row in rows if bool(row["both_switch"])]
    both_keep_rows = [row for row in rows if bool(row["both_keep"])]
    interesting_rows = [
        row
        for row in rows
        if (
            row["action"] != "HOLD"
            or row["scheduler_reason"]
            in {"consensus_armed", "consensus_cancelled"}
            or row["lifecycle_status"] is not None
            or bool(row["both_switch"])
        )
    ]

    print(
        f"run={run_path.name} recording={recording_path.name} "
        f"steps={len(rows)} device={model.device} threshold={threshold:.2f}",
        flush=True,
    )
    print(
        "runtime reproduction: "
        f"raw_h0_abs_error_mean={np.mean(reproduction_errors):.6f} "
        f"p95={np.percentile(reproduction_errors, 95):.6f} "
        f"max={np.max(reproduction_errors):.6f}",
        flush=True,
    )
    print(
        "state consistency: "
        f"both_switch={len(both_switch_rows)} "
        f"both_keep={len(both_keep_rows)} "
        f"comp_err_mean={np.mean(complement_errors):.3f} "
        f"p95={np.percentile(complement_errors, 95):.3f} "
        f"max={np.max(complement_errors):.3f}",
        flush=True,
    )
    print("interesting H0 states:", flush=True)
    for row in interesting_rows:
        print(
            f"  f{row['source_frame']} t={row['timestamp_ms']:.1f}ms "
            f"state={'P' if row['state_before'] else 'R'} "
            f"action={row['action']} reason={row['scheduler_reason']} "
            f"life={row['lifecycle_status']} "
            f"switch_R={row['switch_if_release_h0']:.3f} "
            f"switch_P={row['switch_if_press_h0']:.3f} "
            f"both_switch={row['both_switch']} "
            f"pair_press={row['paired_desired_press_h0']:.3f} "
            f"future_press={row['future_action_desired_press_h0']:.3f}",
            flush=True,
        )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else run_path.with_name(run_path.stem + "_h0_coherence.json")
    )
    payload = {
        "run": str(run_path),
        "recording": str(recording_path),
        "model": str(args.model.resolve()),
        "metadata": str(args.metadata.resolve()),
        "threshold": threshold,
        "runtime_reproduction": {
            "raw_h0_abs_error_mean": float(np.mean(reproduction_errors)),
            "raw_h0_abs_error_p95": float(
                np.percentile(reproduction_errors, 95)
            ),
            "raw_h0_abs_error_max": float(np.max(reproduction_errors)),
        },
        "state_consistency": {
            "both_switch": len(both_switch_rows),
            "both_keep": len(both_keep_rows),
            "complement_error_mean": float(np.mean(complement_errors)),
            "complement_error_p95": float(
                np.percentile(complement_errors, 95)
            ),
            "complement_error_max": float(np.max(complement_errors)),
        },
        "interesting_rows": interesting_rows,
        "steps": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
