#!/usr/bin/env python3
"""Read-only diagnostic: does a global timestamp shift explain RELEASE lag?

Uses frozen three-model fixed-video replay JSON/JSONL, recorded getevent
transitions, frame timing metadata, and episode-level duration audit. Evaluates
observed predictions as-is and *diagnostic* constant shifts; shifted events
are NOT physically executable forecasts, and shift selection is NOT training.
No video decoding, model loading, GPU, Android access, or source modification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

VARIANTS = ("full", "vision_only", "state_only")
FORECAST_HORIZON_MS = 100.0


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def event_timeline(rows: list[dict], variant: str) -> list[tuple[float, bool]]:
    """Reconstruct virtual action changes at target, not observation, time."""
    subset = [r for r in rows if r.get("variant") == variant]
    if not subset:
        raise ValueError(f"missing predictions for {variant}")
    subset.sort(key=lambda r: float(r["target_ms"]))
    previous_target = float("-inf")
    state = False  # Same initial virtual state as evaluate_history_candidate.
    transitions = []
    for row in subset:
        target = float(row["target_ms"])
        observation = float(row["observation_ms"])
        if target <= previous_target:
            raise ValueError("duplicate or nonmonotonic target timestamps")
        if abs(target - observation - FORECAST_HORIZON_MS) > 0.01:
            raise ValueError("target timestamp does not equal observation +100ms")
        previous_target = target
        pressed = float(row["rollout_prob"]) >= 0.5
        if pressed != state:
            transitions.append((target, pressed))
            state = pressed
    return transitions


def kernel_events(touches: list[dict], origin_ms: float,
                  start: float, end: float) -> list[tuple[float, bool]]:
    """Host arrival timestamps mapped to decoded-frame monotonic origin."""
    transitions = [r for r in touches if r.get("kind") == "touch_transition"]
    times = [float(r["host_received_perf_counter_ms"]) for r in transitions]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("nonmonotonic physical touch host-arrival timeline")
    return [(float(t - origin_ms), bool(r["pressed"]))
            for t, r in zip(times, transitions) if start <= t - origin_ms < end]


def match_polarity(model_events, human_events, *, pressed: bool,
                   shift_ms: float = 0.0, tolerance_ms: float = 150.0,
                   start_ms: float, end_ms: float) -> dict:
    """Greedy closest one-to-one matching, held constant for every shift."""
    predicted = [t + shift_ms for t, state in model_events
                 if state == pressed and start_ms <= t + shift_ms < end_ms]
    truth = [t for t, state in human_events if state == pressed]
    candidates = sorted(
        (abs(a - b), i, j, a - b)
        for i, a in enumerate(predicted) for j, b in enumerate(truth)
        if abs(a - b) <= tolerance_ms
    )
    used_pred, used_truth, errors = set(), set(), []
    for _, i, j, lag in candidates:
        if i in used_pred or j in used_truth:
            continue
        used_pred.add(i)
        used_truth.add(j)
        errors.append(lag)
    return {
        "actual": len(truth), "predicted": len(predicted),
        "matched": len(errors), "unmatched_actual": len(truth) - len(errors),
        "unmatched_predicted": len(predicted) - len(errors),
        "signed_lag_median_ms": float(statistics.median(errors)) if errors else None,
        "absolute_lag_median_ms": float(statistics.median(map(abs, errors))) if errors else None,
    }


def scan_shifts(model_events, human_events, *, start_ms: float,
                end_ms: float, max_shift_ms: int = 300,
                step_ms: int = 25, tolerance_ms: float = 150.0) -> list[dict]:
    if max_shift_ms < FORECAST_HORIZON_MS or step_ms < 1 or max_shift_ms % step_ms:
        raise ValueError("max shift must be >=100ms and an integer multiple of step")
    if 100 % step_ms != 0 or tolerance_ms <= 0:
        raise ValueError("shift step must divide H100; tolerance must be positive")
    result = []
    for shift in range(-max_shift_ms, max_shift_ms + 1, step_ms):
        press = match_polarity(
            model_events, human_events, pressed=True, shift_ms=shift,
            tolerance_ms=tolerance_ms, start_ms=start_ms, end_ms=end_ms,
        )
        release = match_polarity(
            model_events, human_events, pressed=False, shift_ms=shift,
            tolerance_ms=tolerance_ms, start_ms=start_ms, end_ms=end_ms,
        )
        result.append({
            "shift_ms": shift,
            "press": press, "release": release,
            "total_matched": press["matched"] + release["matched"],
        })
    return result


def best_shift(scan: list[dict], criterion: str) -> dict:
    """Post-hoc descriptive optimum, never a deployment adjustment."""
    if criterion not in ("press", "release", "total"):
        raise ValueError("invalid shift criterion")

    def matched(row):
        return row["total_matched"] if criterion == "total" else row[criterion]["matched"]

    return max(scan, key=lambda row: (matched(row), -abs(row["shift_ms"]),
                                      -row["shift_ms"]))


def episode_hold_audit(audit: dict, variant: str) -> dict:
    """A constant timestamp shift leaves every paired hold duration unchanged."""
    info = audit["variants"][variant]
    measures = info["release_plausible_subset"]
    return {
        "press_onset_pairs": info["matched_by_press_onset"],
        "release_plausible_pairs": measures["n"],
        "model_minus_kernel_hold_median_ms": measures["hold_minus_kernel_ms"]["median_ms"],
        "model_minus_optical_overlay_hold_median_ms": (
            measures["model_hold_minus_overlay_ms"]["median_ms"]
        ),
        "unchanged_by_uniform_time_shift": True,
    }


def compute(stem: Path, *, max_shift_ms: int, step_ms: int,
            tolerance_ms: float) -> dict:
    summary = json.loads(Path(f"{stem}_tiny_control_replay.json").read_text(encoding="utf-8"))
    episode = json.loads(Path(f"{stem}_episode_timing_audit.json").read_text(encoding="utf-8"))
    if (summary.get("kind") != "tiny_rgb_control_ablation_fixed_video_self_fed_replay"
            or summary.get("run_stem") != str(stem)
            or summary.get("read_only") is not True
            or episode.get("kind") != "tiny_control_model_vs_touch_episode_timing_audit"
            or episode.get("source") != str(stem)
            or episode.get("read_only") is not True):
        raise ValueError("source fixed-video replay or episode audit provenance mismatch")
    rows = read_jsonl(Path(f"{stem}_tiny_control_replay.jsonl"))
    frames = read_jsonl(Path(f"{stem}_frames.jsonl"))
    touches = read_jsonl(Path(f"{stem}_touch.jsonl"))
    if not frames or not rows:
        raise ValueError("empty replay or decoded-frame sidecar")
    expected_n = int(summary["frame_integrity"]["prediction_rows"])
    start, end = map(float, summary["window_ms"])
    if start >= end or list(map(float, episode["window_ms"])) != [start, end]:
        raise ValueError("mismatched evaluation time window")
    origin = float(frames[0]["host_decoded_perf_counter_ms"]
                   - frames[0]["host_decoded_ms"])
    human = kernel_events(touches, origin, start, end)
    if not human:
        raise ValueError("no human touch events")
    output = {
        "kind": "tiny_control_h100_uniform_offset_diagnostic",
        "run_stem": str(stem), "read_only": True,
        "source_replay": str(Path(f"{stem}_tiny_control_replay.json")),
        "source_episode_audit": str(Path(f"{stem}_episode_timing_audit.json")),
        "source_frame_integrity": summary["frame_integrity"],
        "window_ms": [start, end], "horizon_ms": FORECAST_HORIZON_MS,
        "event_match_tolerance_ms": tolerance_ms,
        "diagnostic_shifts_ms": list(range(-max_shift_ms, max_shift_ms + 1, step_ms)),
        "human_touch_events": len(human), "variants": {},
        "interpretation": [
            "t+100ms is the trained prediction target; negative 100ms diagnostic"
            " shift is not a valid correction for the model forecast.",
            "A constant time shift changes event alignment but cannot change"
            " a predicted episode's PRESS-to-RELEASE duration.",
            "Optimal shifts are chosen post hoc on this same run; never use"
            " them as a deployed correction or model-selection metric.",
            "Kernel getevent host-received time and video host decode time"
            " are not physically capture-synchronized.",
            "The fixed video is human-controlled; none of these scores"
            " measures autonomous kart driving quality.",
        ],
    }
    for variant in VARIANTS:
        if (sum(row.get("variant") == variant for row in rows) != expected_n
                or variant not in summary["variants"]
                or variant not in episode["variants"]):
            raise ValueError(f"missing or inconsistent {variant} replay records")
        events = event_timeline(rows, variant)
        expected_switches = summary["variants"][variant]["simulated_virtual_switches"]
        if len(events) != expected_switches:
            raise ValueError(f"{variant} action count differs from source replay")
        scan = scan_shifts(events, human, start_ms=start, end_ms=end,
                           max_shift_ms=max_shift_ms, step_ms=step_ms,
                           tolerance_ms=tolerance_ms)
        lookup = {row["shift_ms"]: row for row in scan}
        output["variants"][variant] = {
            "action_transitions": len(events),
            "original_target_time": lookup[0],
            "invalid_target_to_observation_time_probe": lookup[-100],
            "best_press_posthoc": best_shift(scan, "press"),
            "best_release_posthoc": best_shift(scan, "release"),
            "best_joint_posthoc": best_shift(scan, "total"),
            "episode_hold_duration": episode_hold_audit(episode, variant),
            "offset_sensitivity": scan,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-stem", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shift-ms", type=int, default=300)
    parser.add_argument("--step-ms", type=int, default=25)
    parser.add_argument("--event-tolerance-ms", type=float, default=150.0)
    args = parser.parse_args()
    if not args.output_dir.is_dir():
        parser.error("--output-dir must be an existing directory")
    stem = args.run_stem.with_suffix("") if args.run_stem.suffix == ".mp4" else args.run_stem
    output = args.output_dir / f"{stem.name}_h100_alignment_audit.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report = compute(stem, max_shift_ms=args.max_shift_ms,
                     step_ms=args.step_ms, tolerance_ms=args.event_tolerance_ms)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for variant, row in report["variants"].items():
        base = row["original_target_time"]
        prior = row["invalid_target_to_observation_time_probe"]
        hold = row["episode_hold_duration"]["model_minus_kernel_hold_median_ms"]
        print(
            f"{variant}: target PRESS {base['press']['matched']}/{base['press']['actual']}"
            f" RELEASE {base['release']['matched']}/{base['release']['actual']};"
            f" -100ms PRESS {prior['press']['matched']}/{prior['press']['actual']}"
            f" RELEASE {prior['release']['matched']}/{prior['release']['actual']};"
            f" model-minus-kernel hold {hold:.1f}ms"
        )
    print("Report:", output)


if __name__ == "__main__":
    main()
