#!/usr/bin/env python3
"""Postprocess saved visual-state JSONLs without decoding MP4 or executing control.

Outputs a per-second validity timeline, contiguous missing-state intervals,
ranked derivative anomalies and nearest recorded command events. It cannot
determine whether a visual scene is pre-failure, post-failure, or recoverable.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path
import re


def load_measurements(path: Path) -> list[dict]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw.strip():
            row = json.loads(raw)
            if not isinstance(row, dict) or "timestamp_ms" not in row or "frame_index" not in row:
                raise ValueError(f"invalid measurement at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no measurements: {path}")
    for i, r in enumerate(rows):
        t = float(r["timestamp_ms"])
        if not math.isfinite(t):
            raise ValueError(f"non-finite time at index {i}")
        if i and (t <= float(rows[i - 1]["timestamp_ms"])
                  or int(r["frame_index"]) <= int(rows[i - 1]["frame_index"])):
            raise ValueError(f"non-monotonic measurements at index {i}")
    return rows


def events_from_run(run: dict) -> list[dict]:
    events = []
    for i, step in enumerate(run.get("steps", [])):
        action = step.get("action")
        if action in ("PRESS", "RELEASE"):
            events.append({
                "timestamp_ms": float(step["observation_timestamp_ms"]),
                "action": action, "source": "step", "index": i,
            })
    for i, event in enumerate(run.get("deadline_events", [])):
        action = event.get("action")
        if action in ("PRESS", "RELEASE"):
            events.append({
                "timestamp_ms": float(event["timestamp_ms"]),
                "action": action, "source": "deadline_nominal", "index": i,
            })
    events.sort(key=lambda x: x["timestamp_ms"])
    return events


def rate_stats(rows: list[dict]) -> dict:
    def fraction(key: str) -> float:
        return sum(bool(r.get(key)) for r in rows) / len(rows) if rows else 0.0
    def q(values: list[float], frac: float) -> float | None:
        if not values:
            return None
        vals = sorted(values)
        pos = frac * (len(vals) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    lat = [abs(float(r["lateral_rate_per_s"])) for r in rows
           if r.get("lateral_rate_per_s") is not None]
    head = [abs(float(r["heading_rate_deg_per_s"])) for r in rows
            if r.get("heading_rate_deg_per_s") is not None]
    return {
        "samples": len(rows),
        "kart_detected_rate": fraction("kart_detected"),
        "valid_state_rate": fraction("state_valid"),
        "valid_rate_pair_coverage": len(lat) / len(rows) if rows else 0.0,
        "abs_lateral_rate_p50_p95": [q(lat, 0.50), q(lat, 0.95)],
        "abs_heading_rate_p50_p95_deg_s": [q(head, 0.50), q(head, 0.95)],
    }


def _nearest_event(t: float, events: list[dict], times: list[float]) -> dict | None:
    if not events:
        return None
    i = bisect_left(times, t)
    candidates = [events[j] for j in (i - 1, i) if 0 <= j < len(events)]
    event = min(candidates, key=lambda e: abs(float(e["timestamp_ms"]) - t))
    return {**event, "delta_from_frame_ms": round(t - float(event["timestamp_ms"]), 3)}


def _nearest_preview(frame: int, previews: list[tuple[int, str]]):
    if not previews:
        return None
    index, path = min(previews, key=lambda item: abs(item[0] - frame))
    return {"path": path, "source_frame": index, "offset_frames": index - frame}


def _annotate(r: dict, start: float, events: list[dict],
              event_times: list[float], previews: list[tuple[int, str]],
              issue: str, score: float | None = None) -> dict:
    frame = int(r["frame_index"])
    t = float(r["timestamp_ms"])
    return {
        "issue": issue, "frame_index": frame, "elapsed_from_control_ms": round(t - start, 2),
        "value": score, "state_valid": bool(r.get("state_valid")),
        "lateral_offset_norm": r.get("lateral_offset_norm"),
        "heading_error_axial_deg": r.get("heading_error_axial_deg"),
        "tracker_mode": r.get("tracker_mode"),
        "relation_confidence": r.get("relation_confidence"),
        "nearest_action": _nearest_event(t, events, event_times),
        "nearest_existing_preview": _nearest_preview(frame, previews),
    }


def analyze_timeline(rows: list[dict], *, control_start_ms: float | None = None,
                     events: list[dict] | None = None,
                     previews: list[tuple[int, str]] | None = None,
                     window_ms: float = 1000.0,
                     top_k: int = 6) -> dict:
    if window_ms <= 0 or top_k < 1:
        raise ValueError("window_ms must be positive and top_k >= 1")
    start = float(rows[0]["timestamp_ms"] if control_start_ms is None else control_start_ms)
    events = sorted(events or [], key=lambda x: float(x["timestamp_ms"]))
    times = [float(e["timestamp_ms"]) for e in events]
    previews = previews or []
    bins: dict[int, list[dict]] = {}
    for r in rows:
        bucket = math.floor((float(r["timestamp_ms"]) - start) / window_ms)
        bins.setdefault(bucket, []).append(r)
    windows = [
        {"window_index": k, "from_control_ms": k * window_ms,
         "to_control_ms": (k + 1) * window_ms, **rate_stats(v)}
        for k, v in sorted(bins.items())
    ]

    gaps: list[list[dict]] = []
    current = []
    for r in rows:
        if not bool(r.get("state_valid")):
            current.append(r)
        elif current:
            gaps.append(current)
            current = []
    if current:
        gaps.append(current)
    gaps.sort(key=len, reverse=True)
    missing = []
    for gap in gaps[:top_k]:
        first = gap[0]
        last = gap[-1]
        missing.append({
            "count": len(gap),
            "start_frame": int(first["frame_index"]),
            "end_frame": int(last["frame_index"]),
            "from_control_ms": round(float(first["timestamp_ms"]) - start, 1),
            "to_control_ms": round(float(last["timestamp_ms"]) - start, 1),
            "first": _annotate(first, start, events, times, previews, "invalid_state"),
            "last": _annotate(last, start, events, times, previews, "invalid_state"),
        })

    spikes = []
    for key, name in (("lateral_rate_per_s", "lateral_rate_spike"),
                      ("heading_rate_deg_per_s", "heading_rate_spike")):
        valid_rates = sorted(
            (r for r in rows
             if r.get(key) is not None and math.isfinite(float(r[key]))),
            key=lambda r: abs(float(r[key])),
            reverse=True,
        )
        spikes.extend(
            _annotate(r, start, events, times, previews, name, float(r[key]))
            for r in valid_rates[:top_k]
        )
    return {
        "control_start_ms": start,
        "clock": "same timestamp coordinate as measurement input",
        "per_window": windows,
        "longest_invalid_runs": missing,
        "ranked_spikes": spikes,
        "warning": (
            "No segmentation into normal, recoverable drift or failure state "
            "was inferred. Derivative magnitude is NOT a physical speed or "
            "proof of detector error. Inspect actual preview frames and "
            "annotate gameplay phase before selecting a controller."
        ),
    }


def existing_previews(measurement_dir: Path):
    matches = []
    for path in sorted((measurement_dir / "previews").glob("frame_*.jpg")):
        found = re.fullmatch(r"frame_(\d+)\.jpg", path.name)
        if found:
            matches.append((int(found.group(1)), str(path)))
    return matches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("measurement_dirs", nargs="+", type=Path)
    ap.add_argument("--adb-runs", type=Path, default=Path("artifacts/adb_runs"))
    ap.add_argument("--window-ms", type=float, default=1000.0)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"will not overwrite existing report: {args.output}")
    result = {"mode": "offline_saved_measurements", "runs": {}}
    for folder in args.measurement_dirs:
        name = folder.name
        rows = load_measurements(folder / "measurements.jsonl")
        run_path = args.adb_runs / f"{name}.json"
        if not run_path.exists():
            raise FileNotFoundError(f"matching ADB JSON required: {run_path}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        start = float(run["recording"]["control_start_timestamp_ms"])
        values = analyze_timeline(
            rows, control_start_ms=start,
            events=events_from_run(run), previews=existing_previews(folder),
            window_ms=args.window_ms, top_k=args.top_k,
        )
        result["runs"][name] = values
        print(json.dumps({
            "run": name,
            "windows": [
                {"seconds": window["window_index"],
                 "valid": round(window["valid_state_rate"], 3),
                 "trend": round(window["valid_rate_pair_coverage"], 3)}
                for window in values["per_window"]
            ],
            "longest_invalid": [
                {"frames": [g["start_frame"], g["end_frame"]],
                 "count": g["count"]}
                for g in values["longest_invalid_runs"][:3]
            ],
            "worst_lateral_spike": next(
                (x for x in values["ranked_spikes"]
                 if x["issue"] == "lateral_rate_spike"), None
            ),
            "worst_heading_spike": next(
                (x for x in values["ranked_spikes"]
                 if x["issue"] == "heading_rate_spike"), None
            ),
        }, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Saved visual-state timeline audit: {args.output}")


if __name__ == "__main__":
    main()
