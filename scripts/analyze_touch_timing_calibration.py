#!/usr/bin/env python3
"""Summarize passive touch timing traces without changing any labels or weights.

Preserves short kernel PRESS/RELEASE pulses (including <100ms). Separately
compares long isolated episodes with a heuristic on-screen touch overlay.
The optical detector reports UI latency, not calibrated physical/capture time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
BINS = (("lt_100ms", 0, 100), ("100_200ms", 100, 200),
        ("200_300ms", 200, 300), ("300_600ms", 300, 600),
        ("ge_600ms", 600, float("inf")))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def distribution(samples) -> dict:
    series = sorted(float(value) for value in samples)
    if not series:
        return {"n": 0, "median_ms": None, "p95_ms": None}
    p95_index = .95 * (len(series) - 1)
    lower = int(p95_index)
    upper = min(lower + 1, len(series) - 1)
    p95 = series[lower] + (series[upper] - series[lower]) * (p95_index - lower)
    return {"n": len(series), "median_ms": statistics.median(series),
            "p95_ms": p95}


def physical_episodes(raw: list[dict]) -> dict:
    """Use untouched kernel clock deltas, never optical marker smoothing."""
    transitions = [r for r in raw if r.get("kind") == "touch_transition"]
    if not transitions:
        raise ValueError("no touch_transition events; check getevent device/permissions")
    if any(bool(a["pressed"]) == bool(b["pressed"])
           for a, b in zip(transitions, transitions[1:])):
        raise ValueError("touch transitions do not alternate")
    for field in ("device_timestamp_ms", "host_received_perf_counter_ms"):
        if any(float(b[field]) <= float(a[field])
               for a, b in zip(transitions, transitions[1:])):
            raise ValueError(f"nonmonotonic {field}")
    episodes, gaps = [], []
    for a, b in zip(transitions, transitions[1:]):
        delta_kernel = float(b["device_timestamp_ms"]) - float(a["device_timestamp_ms"])
        delta_host = (float(b["host_received_perf_counter_ms"])
                      - float(a["host_received_perf_counter_ms"]))
        if bool(a["pressed"]):
            episodes.append({
                "kernel_press_ms": float(a["device_timestamp_ms"]),
                "kernel_release_ms": float(b["device_timestamp_ms"]),
                "kernel_duration_ms": delta_kernel,
                "host_duration_ms": delta_host,
                "host_minus_kernel_duration_ms": delta_host - delta_kernel,
            })
        else:
            gaps.append(delta_kernel)
    durations = [row["kernel_duration_ms"] for row in episodes]
    return {
        "touch_transition_count": len(transitions),
        "complete_press_release_episodes": len(episodes),
        "incomplete_or_boundary_transitions": (
            int(not bool(transitions[0]["pressed"]))
            + int(bool(transitions[-1]["pressed"]))
        ),
        "kernel_hold_ms": distribution(durations),
        "host_minus_kernel_hold_ms": distribution([
            row["host_minus_kernel_duration_ms"] for row in episodes
        ]),
        "kernel_release_to_next_press_ms": distribution(gaps),
        "kernel_duration_bins": {
            name: {"count": sum(lower <= t < upper for t in durations),
                   "kernel_hold_ms": distribution([
                       t for t in durations if lower <= t < upper
                   ])}
            for name, lower, upper in BINS
        },
        "kernel_episodes": episodes,
        "caveat": "No short corrective pulse is removed or re-labeled."
    }


def validate_frames(frames: list[dict], reads: list[dict], summary: dict) -> dict:
    if not frames:
        raise ValueError("missing decoded-frame sidecar")
    indices = [int(f["frame_index"]) for f in frames]
    if indices != list(range(len(indices))):
        raise ValueError("decoded frame indices must be dense from zero")
    times = [float(f["host_decoded_perf_counter_ms"]) for f in frames]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("nonmonotonic decoded frame times")
    if len(frames) != int(summary["decoded_frames"]):
        raise ValueError("summary and frame sidecar disagree")
    read_indices = [int(r["frame_index"]) for r in reads]
    if any(a >= b for a, b in zip(read_indices, read_indices[1:])):
        raise ValueError("consumer read frame indices not strictly increasing")
    if any(i < 0 or i >= len(frames) for i in read_indices):
        raise ValueError("consumer read frame index missing from decoded sidecar")
    delays = []
    for r in reads:
        frame = frames[int(r["frame_index"])]
        decoder = float(frame["host_decoded_perf_counter_ms"])
        if abs(decoder - float(r["host_decoded_perf_counter_ms"])) > .02:
            raise ValueError("consumer decoder timestamp does not match sidecar")
        delay = float(r["host_read_perf_counter_ms"]) - decoder
        if delay < -.02:
            raise ValueError("consumer time earlier than decoder availability")
        delays.append(max(0, delay))
    return {
        "decoded_frame_count": len(frames),
        "consumed_frame_count": len(reads),
        "decoder_interval_ms": distribution([
            b - a for a, b in zip(times, times[1:])
        ]),
        "decoder_to_consumer_read_ms": distribution(delays),
        "consumer_skipped_decoded_frames": len(frames) - len(reads),
        "debug_mp4_recorder_error": summary.get("recording_error"),
        "debug_mp4_mapping_valid_reported": summary.get("recording_frame_mapping_valid"),
    }


def analyze(stem: Path, *, with_video: bool = True) -> dict:
    from diagnose_release_timing import (
        clock_offset_diagnostics, detect_visual, pair_visual_with_kernel,
    )
    source = json.loads(Path(f"{stem}_summary.json").read_text(encoding="utf-8"))
    if (source.get("kind") != "passive_touch_timing_calibration_trace"
            or source.get("read_only") is not True):
        raise ValueError("source is not a passive calibration trace")
    touches = read_jsonl(Path(f"{stem}_touch.jsonl"))
    frames = read_jsonl(Path(f"{stem}_frames.jsonl"))
    reads = read_jsonl(Path(f"{stem}_reads.jsonl"))
    visual = None
    if with_video:
        # Existing detector verifies that MP4 decode count equals sidecar
        # and refuses mismapped/faulty recordings. Its finding is only optical.
        observations = detect_visual(Path(f"{stem}.mp4"), frames)
        visual = {
            "detected_frames": observations["detected_frames"],
            "total_frames": observations["total_frames"],
            "isolated_visual_glitches_suppressed": observations[
                "isolated_visual_glitches_suppressed"
            ],
            "paired_isolated_long_holds": pair_visual_with_kernel(
                touches, observations["transitions"]
            ),
        }
    quality = validate_frames(frames, reads, source)
    physical = physical_episodes(touches)
    return {
        "kind": "passive_touch_timing_calibration_analysis",
        "run_stem": str(stem), "read_only": True,
        "source_capture_error": source.get("capture_error"),
        "touch_capture": source.get("touch"),
        "frame_timing": quality,
        "kernel_host_clock": clock_offset_diagnostics(touches),
        "raw_physical_episodes": physical,
        "optical_touch_indicator": visual,
        "limitations": [
            "Event CLOCK origin and getevent USB/ADB transit are not separately calibrated.",
            "BGR decoder timestamps measure host availability, not capture time.",
            "Optical circle may lag and persist after kernel UP; it is not ground truth.",
            "Optical isolated-long-hold heuristic intentionally EXCLUDES rapid pulses.",
            "Do not train physical action targets or derive deployment shifts from this report.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-stem", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true",
                        help="Analyze kernel/frame timing only when MP4 cannot be decoded")
    args = parser.parse_args()
    if not args.output_dir.is_dir():
        parser.error("--output-dir must exist")
    stem = args.run_stem.with_suffix("") if args.run_stem.suffix == ".mp4" else args.run_stem
    dest = args.output_dir / f"{stem.name}_analysis.json"
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite {dest}")
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    report = analyze(stem, with_video=not args.no_video)
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    episodes = report["raw_physical_episodes"]
    print("Complete kernel holds:", episodes["complete_press_release_episodes"])
    for label, bucket in episodes["kernel_duration_bins"].items():
        print(f"  {label}: {bucket['count']}")
    optical = report["optical_touch_indicator"]
    if optical is not None:
        print("Long/isolated optical pairs:",
              optical["paired_isolated_long_holds"]["paired_episodes"])
        print("Visual hold excess:", optical["paired_isolated_long_holds"][
            "visual_hold_duration_excess_ms"
        ])
    print("Report:", dest)


if __name__ == "__main__":
    main()
