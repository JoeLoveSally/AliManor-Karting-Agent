#!/usr/bin/env python3
"""Audit temporal action-state mixing in recorded V4-C2 runtime steps.

The inferred state of a decoded frame is the most recent *commanded* action
at its logged ingestion time, NOT a measurement of the in-game steering state.
No model inference or physical controls are performed.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import json
from pathlib import Path
from typing import Any


def commanded_transitions(run: dict[str, Any]) -> list[tuple[float, bool]]:
    """Merge regular and timer-triggered transitions in chronological order."""
    changes: list[tuple[float, bool]] = []
    for step in run.get("steps", []):
        action = step.get("action")
        if action in ("PRESS", "RELEASE"):
            changes.append((float(step["observation_timestamp_ms"]), action == "PRESS"))
    for event in run.get("deadline_events", []):
        action = event.get("action")
        if action in ("PRESS", "RELEASE"):
            changes.append((float(event["timestamp_ms"]), action == "PRESS"))
    changes.sort(key=lambda item: item[0])
    for previous, current in zip(changes, changes[1:]):
        if previous[0] >= current[0]:
            raise ValueError("transition timestamps must be unique and increasing")
        if previous[1] == current[1]:
            raise ValueError("consecutive transitions must alternate PRESS/RELEASE")
    return changes


def audit_run(run: dict[str, Any], *, threshold: float = 0.6) -> dict[str, Any]:
    """Return deterministic audit; never treat an inferred state as sensor truth."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    steps = run.get("steps", [])
    if not steps:
        raise ValueError("run contains no recorded steps")
    changes = commanded_transitions(run)
    timestamps = [time for time, _ in changes]
    initial = bool(run.get("runtime", {}).get("initial_pressed", False))
    observations: list[dict[str, Any]] = []
    for step in steps:
        observed_at = float(step["observation_timestamp_ms"])
        frame_times = [float(value) for value in step["input_frame_timestamps_ms"]]
        probabilities = [float(value) for value in step["probabilities"]]
        if not frame_times or len(probabilities) < 2:
            raise ValueError("recorded step must include input frames and at least two horizons")
        if frame_times != sorted(frame_times) or frame_times[-1] > observed_at + 1e-6:
            raise ValueError("frame timestamps must be ordered and no later than observation")
        last_index = bisect_right(timestamps, observed_at) - 1
        last_time = timestamps[last_index] if last_index >= 0 else None
        ages = None if last_time is None else observed_at - last_time
        inferred_states = [
            changes[idx][1] if (idx := bisect_left(timestamps, frame_time) - 1) >= 0
            else initial
            for frame_time in frame_times
        ]
        mixed = len(set(inferred_states)) > 1
        both_high = probabilities[0] >= threshold and probabilities[1] >= threshold
        next_index = bisect_right(timestamps, observed_at)
        next_gap = (
            timestamps[next_index] - observed_at if next_index < len(timestamps) else None
        )
        observations.append(
            {
                "timestamp_ms": observed_at,
                "source_frame": step.get("input_frame_indices", [None])[-1],
                "state_after_step": bool(step["pressed"]),
                "since_last_transition_ms": ages,
                "inferred_frame_command_states": [int(value) for value in inferred_states],
                "mixed_command_history": mixed,
                "h100": probabilities[0],
                "h200": probabilities[1],
                "both_high": both_high,
                "next_command_transition_ms": next_gap,
                "transient_both_high": bool(ages is not None and 0 < ages < 100 and both_high),
            }
        )
    transient = [item for item in observations if item["transient_both_high"]]
    early_misleading = [
        item for item in transient
        if item["next_command_transition_ms"] is None
        or item["next_command_transition_ms"] > 500
    ]
    return {
        "steps": len(observations),
        "transitions": len(changes),
        "mixed_command_history_steps": sum(int(item["mixed_command_history"]) for item in observations),
        "transient_both_high_steps": len(transient),
        "transient_both_high_with_no_reverse_within_500ms": len(early_misleading),
        "first_transient_both_high_with_no_reverse_within_500ms": early_misleading[:3],
        "transient_both_high_examples": transient[:5],
        "interpretation_warning": (
            "Frame-state labels are inferred from command timestamps, not from "
            "rendered pixels or measured control effects. Time since command "
            "does not establish when the game visually responded."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Recorded V4-C2 run JSON files")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = {
        str(path): audit_run(json.loads(path.read_text(encoding="utf-8")))
        for path in args.runs
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
