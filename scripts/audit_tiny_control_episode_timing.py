#!/usr/bin/env python3
"""Audit model PRESS/RELEASE episode durations against kernel and UI overlay.

Purely read-only inputs: precomputed fixed-video virtual replay, kernel
getevent rows, frame timestamps, and optional previous optical timing report.
No ADB, ML weights, training, video decode, or Test split. Timing origins are
uncalibrated; optical circle visibility is not physical control latency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

VARIANTS = ("full", "vision_only", "state_only")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(values) -> dict:
    values = sorted(float(x) for x in values)
    if not values:
        return {"n": 0, "median_ms": None, "p05_ms": None, "p95_ms": None}
    def percentile(p: float) -> float:
        position = (len(values) - 1) * p
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "p05_ms": percentile(.05),
        "p95_ms": percentile(.95),
    }


def human_episodes(touches: list[dict], *, origin_ms: float,
                   start_ms: float, end_ms: float) -> list[dict]:
    transitions = [row for row in touches if row.get("kind") == "touch_transition"]
    if len(transitions) < 2:
        raise ValueError("need at least two touch transitions")
    out = []
    last = -math.inf
    for a, b in zip(transitions, transitions[1:]):
        a_host = float(a["host_received_perf_counter_ms"])
        b_host = float(b["host_received_perf_counter_ms"])
        if a_host <= last or b_host <= a_host:
            raise ValueError("nonmonotonic touch transition host timestamps")
        last = a_host
        if not bool(a["pressed"]) or bool(b["pressed"]):
            continue
        press, release = a_host - origin_ms, b_host - origin_ms
        if press < start_ms or release >= end_ms:
            continue
        kernel_duration = float(b["device_timestamp_ms"]) - float(a["device_timestamp_ms"])
        if kernel_duration <= 0:
            raise ValueError("nonpositive kernel hold duration")
        out.append({
            "human_index": len(out),
            "press_ms": press, "release_ms": release,
            "press_perf_ms": a_host, "release_perf_ms": b_host,
            "host_hold_ms": release - press,
            "kernel_hold_ms": kernel_duration,
            "host_minus_kernel_hold_ms": release - press - kernel_duration,
        })
    return out


def model_episodes(records: list[dict], variant: str) -> list[dict]:
    points = sorted(
        (float(r["target_ms"]), bool(float(r["rollout_prob"]) >= .5))
        for r in records if r.get("variant") == variant
    )
    if not points:
        raise ValueError(f"missing predictions for {variant}")
    if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
        raise ValueError("duplicate or nonmonotonic target time")
    events = []
    pressed = False  # Simulator initial_pressed=False at video start.
    for t, next_pressed in points:
        if pressed != next_pressed:
            events.append((t, next_pressed))
            pressed = next_pressed
    out = []
    for a, b in zip(events, events[1:]):
        if a[1] and not b[1] and b[0] > a[0]:
            out.append({"model_index": len(out), "press_ms": a[0],
                        "release_ms": b[0], "hold_ms": b[0] - a[0]})
    return out


def onset_pairs(humans: list[dict], models: list[dict],
                *, press_tolerance_ms: float = 150) -> list[dict]:
    """Nearest-onset one-to-one match. Preserve early-release mispairs visibly."""
    candidates = sorted(
        (abs(m["press_ms"] - h["press_ms"]), i, j)
        for i, h in enumerate(humans) for j, m in enumerate(models)
        if abs(m["press_ms"] - h["press_ms"]) <= press_tolerance_ms
    )
    seen_h, seen_m, result = set(), set(), []
    for _, i, j in candidates:
        if i in seen_h or j in seen_m:
            continue
        seen_h.add(i)
        seen_m.add(j)
        h, m = humans[i], models[j]
        result.append({
            "human_index": h["human_index"], "model_index": m["model_index"],
            "human_press_ms": h["press_ms"], "human_release_ms": h["release_ms"],
            "model_press_ms": m["press_ms"], "model_release_ms": m["release_ms"],
            "kernel_hold_ms": h["kernel_hold_ms"],
            "host_hold_ms": h["host_hold_ms"],
            "model_hold_ms": m["hold_ms"],
            "press_minus_host_ms": m["press_ms"] - h["press_ms"],
            "release_minus_host_ms": m["release_ms"] - h["release_ms"],
            "hold_minus_kernel_ms": m["hold_ms"] - h["kernel_hold_ms"],
            "hold_minus_host_ms": m["hold_ms"] - h["host_hold_ms"],
            "host_minus_kernel_hold_ms": h["host_minus_kernel_hold_ms"],
        })
    return sorted(result, key=lambda r: r["human_index"])


def attach_optical(pairs: list[dict], optical: dict | None) -> None:
    """Attach same-episode UI marker timestamps; not an action ground truth."""
    if optical is None:
        return
    examples = optical["optical_overlay"]["paired"]["examples"]
    used = set()
    for row in pairs:
        candidates = sorted(
            (abs(ex["host_press_ms"] - (row["human_press_ms"] + optical["_origin_ms"])),
             i, ex)
            for i, ex in enumerate(examples)
            if i not in used
            and abs(ex["host_press_ms"] - (row["human_press_ms"] + optical["_origin_ms"])) <= 5
            and abs(ex["host_release_ms"] - (row["human_release_ms"] + optical["_origin_ms"])) <= 5
        )
        if not candidates:
            continue
        _, i, ex = candidates[0]
        used.add(i)
        overlay_press = row["human_press_ms"] + float(ex["overlay_press_lag_ms"])
        overlay_release = row["human_release_ms"] + float(ex["overlay_release_lag_ms"])
        row.update({
            "overlay_press_lag_ms": float(ex["overlay_press_lag_ms"]),
            "overlay_release_lag_ms": float(ex["overlay_release_lag_ms"]),
            "overlay_hold_excess_ms": float(ex["overlay_hold_excess_ms"]),
            "model_press_minus_overlay_ms": row["model_press_ms"] - overlay_press,
            "model_release_minus_overlay_ms": row["model_release_ms"] - overlay_release,
            "model_hold_minus_overlay_ms": row["model_hold_ms"]
            - (overlay_release - overlay_press),
        })


def audit(humans, grouped_models, *, optical=None,
          press_tolerance_ms=150., plausible_release_min_ms=-150.,
          plausible_release_max_ms=500.):
    out = {
        "human_episode_count": len(humans),
        "human_kernel_hold": summarize([h["kernel_hold_ms"] for h in humans]),
        "human_host_minus_kernel_hold": summarize([
            h["host_minus_kernel_hold_ms"] for h in humans
        ]),
        "variants": {},
    }
    all_rows = []
    for variant in VARIANTS:
        models = grouped_models[variant]
        pairs = onset_pairs(humans, models, press_tolerance_ms=press_tolerance_ms)
        attach_optical(pairs, optical)
        valid = [p for p in pairs if plausible_release_min_ms <=
                 p["release_minus_host_ms"] <= plausible_release_max_ms]
        ambiguous = [p for p in pairs if p not in valid]
        if len(valid) + len(ambiguous) != len(pairs):
            raise AssertionError("invalid episode partition")
        def measures(rows):
            return {
                name: summarize([p[name] for p in rows if name in p])
                for name in (
                    "press_minus_host_ms", "release_minus_host_ms",
                    "hold_minus_kernel_ms", "hold_minus_host_ms",
                    "model_press_minus_overlay_ms", "model_release_minus_overlay_ms",
                    "model_hold_minus_overlay_ms",
                )
            }
        out["variants"][variant] = {
            "model_complete_episodes": len(models),
            "matched_by_press_onset": len(pairs),
            "unmatched_human": len(humans) - len(pairs),
            "ambiguous_release_episode_count": len(ambiguous),
            "ambiguous_human_indices": [p["human_index"] for p in ambiguous],
            "all_matched_episodes": measures(pairs),
            "release_plausible_subset": {"n": len(valid), **measures(valid)},
            "qualifier": (
                "Subset removes severe release mismatches for robust timing display; "
                "excluded episodes remain present in CSV and counts. Not a metric of driving success."
            ),
        }
        for p in pairs:
            p["variant"] = variant
            p["release_pairing_plausible"] = p in valid
            all_rows.append(p)
    return out, all_rows


def run(args):
    stem = args.run_stem.with_suffix("") if args.run_stem.suffix == ".mp4" else args.run_stem
    if (not args.output_dir.is_dir() or
            not 0 <= args.start_ms < args.end_ms or args.press_tolerance_ms <= 0):
        raise ValueError("invalid output directory, time range or pairing tolerance")
    output = args.output_dir / f"{stem.name}_episode_timing_audit.json"
    details = args.output_dir / f"{stem.name}_episode_timing_audit.csv"
    if output.exists() or details.exists():
        raise FileExistsError("refusing to overwrite existing timing audit")
    source_report = json.loads(Path(f"{stem}_tiny_control_replay.json").read_text(encoding="utf-8"))
    if (source_report.get("kind") != "tiny_rgb_control_ablation_fixed_video_self_fed_replay"
            or source_report.get("run_stem") != str(stem)
            or source_report.get("read_only") is not True
            or not all(v in source_report["variants"] for v in VARIANTS)):
        raise ValueError("source replay provenance mismatch")
    frames = read_jsonl(Path(f"{stem}_frames.jsonl"))
    touches = read_jsonl(Path(f"{stem}_touch.jsonl"))
    records = read_jsonl(Path(f"{stem}_tiny_control_replay.jsonl"))
    if not frames or not records:
        raise ValueError("empty frame sidecar or model replay")
    expected = int(source_report["frame_integrity"]["prediction_rows"])
    if any(sum(r.get("variant") == name for r in records) != expected
           for name in VARIANTS):
        raise ValueError("model replay prediction counts disagree with source sidecar")
    origin = float(frames[0]["host_decoded_perf_counter_ms"] - frames[0]["host_decoded_ms"])
    humans = human_episodes(touches, origin_ms=origin,
                            start_ms=args.start_ms, end_ms=args.end_ms)
    optical = None
    if args.optical_report is not None:
        optical = json.loads(args.optical_report.read_text(encoding="utf-8"))
        if (optical.get("kind") != "tiny_release_timing_diagnostic"
                or optical.get("source") != str(stem)
                or not isinstance(optical.get("optical_overlay", {}).get("paired", {}).get("examples"), list)):
            raise ValueError("optical timing report provenance mismatch")
        optical["_origin_ms"] = origin
    models = {variant: model_episodes(records, variant) for variant in VARIANTS}
    results, rows = audit(humans, models, optical=optical,
                          press_tolerance_ms=args.press_tolerance_ms)
    report = {
        "kind": "tiny_control_model_vs_touch_episode_timing_audit",
        "source": str(stem), "read_only": True,
        "model_replay": str(Path(f"{stem}_tiny_control_replay.json")),
        "human_reference": "Android getevent timestamps / host arrival times",
        "optical_reference": str(args.optical_report) if optical is not None else None,
        "source_frame_integrity": source_report["frame_integrity"],
        "window_ms": [args.start_ms, args.end_ms],
        "press_matching_tolerance_ms": args.press_tolerance_ms,
        "release_plausibility_window_ms": [-150, 500],
        **results,
        "limitations": [
            "Recorded video is a fixed human trajectory, not model-acted counterfactual pixels.",
            "Human physical event timestamps and host-decoded video have uncalibrated capture latency.",
            "A visible circular touch overlay is not proof of actual touch actuation timing.",
            "Overlay onset/offset heuristics are video-specific and may have asymmetric UI fade.",
            "First RELEASE after a matched model PRESS can be an unrelated short correction.",
            "Outliers are shown, not silently dropped; subset excludes unmatched or severe mispairs.",
            "Do not optimize model policy thresholds against this recording.",
        ],
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    fields = (
        "variant", "human_index", "model_index", "release_pairing_plausible",
        "human_press_ms", "human_release_ms", "model_press_ms", "model_release_ms",
        "press_minus_host_ms", "release_minus_host_ms", "host_hold_ms",
        "kernel_hold_ms", "model_hold_ms", "hold_minus_host_ms", "hold_minus_kernel_ms",
        "host_minus_kernel_hold_ms", "overlay_press_lag_ms", "overlay_release_lag_ms",
        "overlay_hold_excess_ms", "model_press_minus_overlay_ms",
        "model_release_minus_overlay_ms", "model_hold_minus_overlay_ms",
    )
    with details.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Summary:", output)
    print("Per-episode CSV:", details)
    for name, info in report["variants"].items():
        subset = info["release_plausible_subset"]
        hold = subset["hold_minus_kernel_ms"]["median_ms"]
        residual = subset["model_hold_minus_overlay_ms"]["median_ms"]
        print(f"{name}: matched={info['matched_by_press_onset']}/"
              f"{report['human_episode_count']}, plausible={subset['n']}, "
              f"hold-kernel={hold}ms, hold-overlay={residual}ms")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-stem", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optical-report", type=Path, default=None)
    parser.add_argument("--start-ms", type=float, default=5000.)
    parser.add_argument("--end-ms", type=float, default=56800.)
    parser.add_argument("--press-tolerance-ms", type=float, default=150.)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
