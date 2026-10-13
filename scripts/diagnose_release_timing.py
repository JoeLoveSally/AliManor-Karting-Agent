#!/usr/bin/env python3
"""Read-only RELEASE timing and visual touch-overlay diagnostics.

This tool never sends Android controls, trains, or overwrites source recordings.
Touch-circle visibility is UI feedback, NOT the timestamp of physical touch.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path

import cv2
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile_summary(values) -> dict:
    if not values:
        return {"count": 0}
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()), "max": float(values.max()),
    }


def clock_offset_diagnostics(touches: list[dict]) -> dict:
    rows = [r for r in touches if r.get("kind") == "kernel_event"]
    differences = [
        r["host_received_perf_counter_ms"] - r["device_timestamp_ms"]
        for r in rows
    ]
    if not differences:
        raise ValueError("missing raw kernel events for clock diagnostics")
    minimum = min(differences)
    return {
        "host_minus_kernel_ms": percentile_summary(differences),
        "arrival_excess_above_fastest_observation_ms": percentile_summary(
            [value - minimum for value in differences]
        ),
        "interpretation": (
            "Clock origin difference and unknown ADB buffering delay cannot "
            "be separated. Lower-envelope offset is not calibrated physical "
            "event time."
        ),
    }


def gray_touch_dot(frame: np.ndarray) -> bool:
    """Find a bright, low-saturation compact circular component at lower right.

    Calibrated to the 360x800 Android screenrecord overlay; do not use it
    as a training label or treat it as ground truth on new visual themes.
    """
    h, w = frame.shape[:2]
    if w < 200 or h < 400:
        raise ValueError("unsupported debug recording resolution")
    x0, x1 = int(w * .66), int(w * .99)
    y0, y1 = int(h * .80), int(h * .985)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    binary = np.uint8((hsv[:, :, 1] <= 74) & (hsv[:, :, 2] >= 130))
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    scale = (w / 360.0) ** 2
    linear = w / 360.0
    for x, y, cw, ch, area in stats[1:]:
        if (55 * scale <= area <= 190 * scale
                and 7 * linear <= cw <= 18 * linear
                and 7 * linear <= ch <= 18 * linear
                and .7 <= cw / ch <= 1.45
                and .55 <= area / (cw * ch) <= .92):
            return True
    return False


def clean_optical_only(states: list[bool]) -> list[bool]:
    """One-frame cleanup applies ONLY to UI marker observations, not labels."""
    cleaned = list(states)
    for i in range(1, len(states) - 1):
        if states[i - 1] == states[i + 1] != states[i]:
            cleaned[i] = states[i - 1]
    return cleaned


def visual_transitions(states: list[bool], times: list[float]):
    if len(states) != len(times):
        raise ValueError("visual states and frame timestamps differ")
    return [
        (times[i], states[i])
        for i in range(1, len(states)) if states[i] != states[i - 1]
    ]


def detect_visual(video: Path, frames: list[dict]) -> dict:
    reader = cv2.VideoCapture(str(video))
    if not reader.isOpened():
        raise RuntimeError(f"cannot decode {video}")
    states = []
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            states.append(gray_touch_dot(frame))
    finally:
        reader.release()
    if len(states) != len(frames):
        raise ValueError("MP4 decoded frames do not match timestamp sidecar")
    cleaned = clean_optical_only(states)
    return {
        "detected_frames": sum(states), "total_frames": len(states),
        "isolated_visual_glitches_suppressed": sum(
            a != b for a, b in zip(states, cleaned)
        ),
        "transitions": visual_transitions(
            cleaned, [float(r["host_decoded_perf_counter_ms"]) for r in frames]
        ),
        "method": (
            "Bright low-saturation circular blob in lower-right ROI; "
            "video-specific heuristic, not validated physical touch labels"
        ),
    }


def pair_visual_with_kernel(touch_rows: list[dict],
                            visual: list[tuple[float, bool]],
                            *, min_press_duration_ms: float = 200.0) -> dict:
    """One-to-one pair long, isolated physical holds to optical UI episodes."""
    kernel = [
        (float(r["host_received_perf_counter_ms"]), bool(r["pressed"]))
        for r in touch_rows if r.get("kind") == "touch_transition"
    ]
    onsets = [t for t, p in visual if p]
    offsets = [t for t, p in visual if not p]
    used_on, used_off = set(), set()
    on_lags, off_lags, inflation, examples = [], [], [], []
    eligible = 0
    for i in range(len(kernel) - 1):
        start, pressed = kernel[i]
        if (not pressed or kernel[i + 1][1]
                or kernel[i + 1][0] - start < min_press_duration_ms):
            continue
        stop = kernel[i + 1][0]
        if i + 2 < len(kernel) and kernel[i + 2][0] - stop < 300:
            continue  # Ambiguous overlapping touch-overlay fade.
        eligible += 1
        on = [
            (abs(t - (start + 65)), j, t)
            for j, t in enumerate(onsets)
            if j not in used_on and -10 <= t - start <= 180
        ]
        off = [
            (abs(t - (stop + 120)), j, t)
            for j, t in enumerate(offsets)
            if j not in used_off and -10 <= t - stop <= 320
        ]
        if not on or not off:
            continue
        _, on_idx, visual_start = min(on)
        _, off_idx, visual_stop = min(off)
        if visual_stop <= visual_start:
            continue
        used_on.add(on_idx)
        used_off.add(off_idx)
        on_lag = visual_start - start
        off_lag = visual_stop - stop
        on_lags.append(on_lag)
        off_lags.append(off_lag)
        inflation.append(off_lag - on_lag)
        examples.append({
            "host_press_ms": round(start, 3),
            "host_release_ms": round(stop, 3),
            "overlay_press_lag_ms": round(on_lag, 2),
            "overlay_release_lag_ms": round(off_lag, 2),
            "overlay_hold_excess_ms": round(off_lag - on_lag, 2),
        })
    return {
        "eligible_long_isolated_episodes": eligible,
        "paired_episodes": len(on_lags),
        "visual_press_minus_host_touch_arrival_ms": percentile_summary(on_lags),
        "visual_release_minus_host_touch_arrival_ms": percentile_summary(off_lags),
        "visual_hold_duration_excess_ms": percentile_summary(inflation),
        "examples": examples,
        "caveat": (
            "UI overlay may linger after physical release. Frame timestamps "
            "denote host decoder availability, not Android capture. This is "
            "not a calibrated end-to-end physical latency measurement."
        ),
    }


def truth_events(touches, origin_ms, shift_ms):
    result = [(0.0, False)]
    for r in touches:
        if r.get("kind") != "touch_transition":
            continue
        t = float(r["host_received_perf_counter_ms"] - origin_ms + shift_ms)
        if t >= 0 and bool(r["pressed"]) != result[-1][1]:
            result.append((t, bool(r["pressed"])))
    return result


def future_transition_diagnostics(predictions, truth, start_ms, end_ms,
                                  horizon_ms=100.0):
    """Upcoming RELEASE/PRESS conditional on current actual state at t.

    Sensitivity scans only shift the reference timestamps; teacher input is
    NOT rerun at each offset, so they are not separate causal experiments.
    """
    if not predictions:
        raise ValueError("missing offline prediction rows")
    event_times = [t for t, _ in truth]
    keys = sorted(predictions[0]["probabilities"])
    result = {}
    for key in keys:
        counts = {
            "upcoming_release": [0, 0], "upcoming_press": [0, 0],
            "stable_pressed": [0, 0], "stable_released": [0, 0],
        }
        for row in predictions:
            target = float(row["target_ms"])
            now = target - horizon_ms
            if now < start_ms or target >= end_ms:
                continue
            current_index = max(0, bisect_right(event_times, now) - 1)
            target_index = max(0, bisect_right(event_times, target) - 1)
            current = truth[current_index][1]
            future = truth[target_index][1]
            changes = [
                e for e in truth[current_index + 1:target_index + 1]
                if now < e[0] <= target
            ]
            if changes and current == future:
                continue  # Two opposite transitions in one forecast horizon.
            name = (
                "upcoming_release" if current and not future else
                "upcoming_press" if not current and future else
                "stable_pressed" if current else "stable_released"
            )
            predicted = row["probabilities"][key] >= .5
            counts[name][0] += int(predicted == future)
            counts[name][1] += 1
        result[key] = {
            name: {
                "correct": correct, "total": total,
                "accuracy": correct / total if total else None,
            }
            for name, (correct, total) in counts.items()
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-stem", type=Path, required=True)
    parser.add_argument("--offline-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--race-start-ms", type=float, default=5000.0)
    parser.add_argument("--race-end-ms", type=float, default=56800.0)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if not args.output_dir.is_dir() or args.race_end_ms <= args.race_start_ms:
        parser.error("output-dir must exist; race-end-ms must exceed race-start-ms")
    stem = args.run_stem.with_suffix("") if args.run_stem.suffix == ".mp4" else args.run_stem
    touches = read_jsonl(Path(str(stem) + "_touch.jsonl"))
    frames = read_jsonl(Path(str(stem) + "_frames.jsonl"))
    if not frames:
        raise ValueError("missing decoded frame timestamps")
    origin = (frames[0]["host_decoded_perf_counter_ms"]
              - frames[0]["host_decoded_ms"])
    result = {
        "kind": "tiny_release_timing_diagnostic", "source": str(stem),
        "read_only": True,
        "window_ms": [args.race_start_ms, args.race_end_ms],
        "kernel_clock": clock_offset_diagnostics(touches),
        "optical_overlay": None, "forecast_shift_sensitivity": None,
        "limitations": [
            "Kernel clock offset and unknown ADB buffering delays are inseparable",
            "Decoder availability differs from original Android capture time",
            "Gray touch overlay is UI feedback, not a physical event timestamp",
            "Shift-dependent forecasting numbers are not calibrated latency",
        ],
    }
    if not args.no_video:
        visual = detect_visual(Path(str(stem) + ".mp4"), frames)
        result["optical_overlay"] = {
            k: v for k, v in visual.items() if k != "transitions"
        }
        result["optical_overlay"]["paired"] = pair_visual_with_kernel(
            touches, visual["transitions"]
        )
    if args.offline_predictions is not None:
        predictions = read_jsonl(args.offline_predictions)
        result["forecast_shift_sensitivity"] = {
            str(shift): future_transition_diagnostics(
                predictions, truth_events(touches, origin, shift),
                args.race_start_ms, args.race_end_ms,
            )
            for shift in (-150, -100, -50, 0, 50, 100, 150)
        }
    output = args.output_dir / (stem.name + "_release_timing_diagnostics.json")
    with output.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print("RELEASE diagnostics:", output)
    if result["optical_overlay"] is not None:
        paired = result["optical_overlay"]["paired"]
        print("Matched visual/physical holds:", paired["paired_episodes"])
        print("Visual hold inflation:",
              paired["visual_hold_duration_excess_ms"])


if __name__ == "__main__":
    main()
