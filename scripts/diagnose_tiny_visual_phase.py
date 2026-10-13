#!/usr/bin/env python3
"""Noncausal oracle experiment: shift only five RGB source frames in time.

The model, its original frame-delta control inputs, and the scored H100 target
timestamps remain FIXED. A positive image shift may reveal pixels from after
the target event. This measures VISUAL PHASE, not executable policy latency.
Read-only recorded MP4 / frozen vision_only weights; no ADB or retraining.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHIFTS = (0, 50, 100, 150, 200, 250, 300)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nearest_time_index(timestamps: list[float], target: float) -> tuple[int, float]:
    """Closest decoded frame by host-availability timestamp, ties to earlier."""
    i = bisect_left(timestamps, target)
    choices = [j for j in (i - 1, i) if 0 <= j < len(timestamps)]
    if not choices:
        raise ValueError("empty frame timestamps")
    index = min(choices, key=lambda j: (abs(timestamps[j] - target), j))
    return index, float(timestamps[index] - target)


def plan_windows(predictions, frames, *, shifts_ms=DEFAULT_SHIFTS,
                 start_ms=5000.0, end_ms=56800.0, max_skew_ms=45.0):
    """Select the SAME predictions at all shifts; never alter target times.

    Missing shifted frames drop an entire row for ALL shifts, so model
    comparisons cannot be explained by different prediction subsets.
    """
    shifts = tuple(int(s) for s in shifts_ms)
    if (not shifts or len(set(shifts)) != len(shifts) or 0 not in shifts
            or any(s < 0 for s in shifts) or max_skew_ms <= 0
            or start_ms >= end_ms):
        raise ValueError("expect unique nonnegative shifts including zero and valid limits")
    if not frames:
        raise ValueError("no decoded frames")
    times = [float(f["host_decoded_ms"]) for f in frames]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("decoded frame times must increase strictly")
    plans = []
    eligible = 0
    rejected = 0
    gaps = {s: [] for s in shifts}
    # Include a prior decision and trailing decisions for onset scoring.
    for row in predictions:
        target = float(row["models"]["rgb"]["target_ms"])
        if not start_ms - 400 <= target < end_ms + 400:
            continue
        eligible += 1
        original = [int(i) for i in row["input_indices"]]
        stamps = [float(t) for t in row["input_timestamp_ms"]]
        if (len(original) != 5 or len(stamps) != 5
                or abs(target - float(row["observation_ms"]) - 100.0) > .01):
            raise ValueError("invalid 5-frame, H100 source prediction")
        if any(i < 0 or i >= len(times) or abs(times[i] - t) > .01
               for i, t in zip(original, stamps)):
            raise ValueError("original frame index/timestamp sidecar mismatch")
        indices_by_shift = {}
        deviations = {}
        for shift in shifts:
            if shift == 0:
                indices_by_shift[shift] = original
                deviations[shift] = [0.] * 5
                continue
            chosen = [nearest_time_index(times, t + shift) for t in stamps]
            indices_by_shift[shift] = [i for i, _ in chosen]
            deviations[shift] = [dt for _, dt in chosen]
        if any(abs(dt) > max_skew_ms for dev in deviations.values() for dt in dev):
            rejected += 1
            continue
        if any(any(b < a for a, b in zip(indices, indices[1:]))
               for indices in indices_by_shift.values()):
            raise ValueError("shifted video frame sequence not monotonic")
        for shift, dev in deviations.items():
            gaps[shift].extend(map(abs, dev))
        plans.append({
            "frame_index": int(row["frame_index"]),
            "target_ms": target,
            "observation_ms": float(row["observation_ms"]),
            "input_timestamp_ms": stamps,
            "input_indices_by_shift": indices_by_shift,
        })
    if not plans or eligible < 1:
        raise ValueError("no valid common observation rows")
    return plans, {
        "eligible_rows": eligible, "common_rows": len(plans),
        "excluded_due_to_nearest_frame_gap": rejected,
        "retention_fraction": len(plans) / eligible,
        "max_nearest_frame_error_ms": max_skew_ms,
        "nearest_frame_error_p95_by_shift_ms": {
            str(shift): float(np.percentile(dev, 95)) for shift, dev in gaps.items()
        },
    }


def fixed_control_dt(stamps) -> np.ndarray:
    """Vision-only ignores state and action age, but retains ORIGINAL dt."""
    stamps = np.asarray(stamps, dtype=np.float64)
    if stamps.shape != (5,) or (np.diff(stamps) < 0).any():
        raise ValueError("bad five-frame time vector")
    values = np.zeros((5, 3), dtype=np.float32)
    values[1:, 2] = np.minimum(np.maximum(np.diff(stamps), 0), 200) / 200
    return values


def infer_oracle(model, plans, features, *, shifts_ms, batch_size=48):
    """Same model parameters + fixed dt, only RGB feature indices can vary."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    output = [{"frame_index": row["frame_index"],
               "target_ms": row["target_ms"],
               "observation_ms": row["observation_ms"],
               "probabilities": {}} for row in plans]
    with torch.inference_mode():
        for shift in shifts_ms:
            for first in range(0, len(plans), batch_size):
                block = plans[first:first + batch_size]
                images = torch.from_numpy(np.stack([
                    np.stack([features[i] for i in r["input_indices_by_shift"][shift]])
                    for r in block
                ]).astype(np.float32) / 255.0)
                controls = torch.from_numpy(np.stack([
                    fixed_control_dt(r["input_timestamp_ms"]) for r in block
                ]))
                logits = model(images, controls)
                values = torch.sigmoid(logits).cpu().numpy()
                if values.shape != (len(block),) or not np.isfinite(values).all():
                    raise ValueError("nonfinite or mismatched model probability")
                for j, p in enumerate(values):
                    output[first + j]["probabilities"][str(shift)] = float(p)
    return output


def model_events(rows, shift: int):
    """Changes in predicted H100 states, always at ORIGINAL target times."""
    changes = []
    last = None
    for row in rows:
        pressed = row["probabilities"][str(shift)] >= .5
        if last is not None and pressed != last:
            changes.append((row["target_ms"], bool(pressed)))
        last = pressed
    return changes


def post_release_probe(rows, releases, shift, *, lower_ms=100, upper_ms=200):
    """For each release, sample closest predicted target at +150ms."""
    observations = []
    for released_at in releases:
        candidates = [
            (abs((row["target_ms"] - released_at) - 150.0), row)
            for row in rows
            if lower_ms <= row["target_ms"] - released_at <= upper_ms
        ]
        if candidates:
            row = min(candidates, key=lambda x: x[0])[1]
            observations.append(row["probabilities"][str(shift)])
    return {
        "reference_releases": len(releases),
        "release_events_with_prediction_near_plus150ms": len(observations),
        "median_p_press": float(statistics.median(observations)) if observations else None,
        "predicted_release_fraction": (
            sum(p < .5 for p in observations) / len(observations)
            if observations else None
        ),
    }


def compute_scores(rows, actual, shifts, start, end):
    from evaluate_tiny_live_recording import (
        match_events, state_scores, window,
    )

    gt = window(actual[1:], start, end)
    releases = [t for t, pressed in gt if not pressed]
    out = {}
    for shift in shifts:
        events = window(model_events(rows, shift), start, end)
        points = [(r["target_ms"], r["probabilities"][str(shift)] >= .5)
                  for r in rows if start <= r["target_ms"] < end]
        out[str(shift)] = {
            "press": match_events(events, gt, pressed=True),
            "release": match_events(events, gt, pressed=False),
            "state_at_original_target": state_scores(points, actual),
            "release_plus150ms": post_release_probe(rows, releases, shift),
            "visual_frame_after_original_target": shift > 100,
            "note": "The target timestamp NEVER changes between image shifts.",
        }
    return out


def run(args):
    if (not args.output_dir.is_dir() or not args.checkpoint_dir.is_dir()
            or args.batch_size < 1 or args.cpu_threads < 1
            or not 0 <= args.start_ms < args.end_ms):
        raise ValueError("invalid input directories, CPU threads, batch or window")
    stem = args.run_stem.with_suffix("") if args.run_stem.suffix == ".mp4" else args.run_stem
    report_path = args.output_dir / f"{stem.name}_visual_phase_oracle.json"
    probabilities_path = args.output_dir / f"{stem.name}_visual_phase_oracle.jsonl"
    if report_path.exists() or probabilities_path.exists():
        raise FileExistsError("refusing to overwrite oracle phase diagnostics")
    for directory in (ROOT / "src", ROOT / "scripts"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import torch
    from evaluate_history_candidate import load_required_rgb
    from evaluate_tiny_live_recording import (
        frame_consistency, read_jsonl, touch_timeline,
    )
    from evaluate_tiny_control_ablation_replay import validate_training_report
    from karting_agent.model.tiny_control_ablation import TinyControlAblation
    from karting_agent.train.trainer import load_video_split

    torch.set_num_threads(args.cpu_threads)
    summary = json.loads(Path(f"{stem}_summary.json").read_text(encoding="utf-8"))
    source = json.loads(Path(f"{stem}_tiny_control_replay.json").read_text(encoding="utf-8"))
    if (source.get("kind") != "tiny_rgb_control_ablation_fixed_video_self_fed_replay"
            or source.get("run_stem") != str(stem)
            or source.get("source_rgb_parity", {}).get("passes") is not True):
        raise ValueError("source frozen video replay provenance or RGB parity invalid")
    pred = [r for r in read_jsonl(Path(f"{stem}.jsonl")) if r.get("kind") == "prediction"]
    frames = read_jsonl(Path(f"{stem}_frames.jsonl"))
    touches = read_jsonl(Path(f"{stem}_touch.jsonl"))
    integrity = frame_consistency(summary, frames, pred)
    if (source["frame_integrity"]["frames"] != len(frames)
            or source["frame_integrity"]["prediction_rows"] != len(pred)):
        raise ValueError("decoded-frame source mismatch")
    model_path = args.checkpoint_dir / "tiny_control_vision_only.pt"
    training_path = args.checkpoint_dir / "tiny_control_vision_only.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    validate_training_report(training, variant="vision_only",
                             split=load_video_split(args.config))
    provenance = source["checkpoint_provenance"]["vision_only"]
    if (sha256(model_path) != provenance["checkpoint_sha256"]
            or sha256(training_path) != provenance["training_report_sha256"]):
        raise ValueError("vision-only weights/report differ from previous frozen replay")
    model = TinyControlAblation("vision_only")
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    plans, mapping = plan_windows(
        pred, frames, shifts_ms=args.shifts_ms, start_ms=args.start_ms,
        end_ms=args.end_ms, max_skew_ms=args.max_frame_skew_ms,
    )
    indices = sorted({
        index for p in plans for shift in args.shifts_ms
        for index in p["input_indices_by_shift"][shift]
    })
    features = load_required_rgb(
        Path(f"{stem}.mp4"), [{"input_indices": indices}], len(frames),
        args.config, args.road_config,
    )
    rows = infer_oracle(model, plans, features, shifts_ms=args.shifts_ms,
                        batch_size=args.batch_size)
    # Absolute 0ms parity against the PREVIOUS vision-only replay is mandatory.
    original = {
        int(r["frame_index"]): float(r["rollout_prob"])
        for r in read_jsonl(Path(f"{stem}_tiny_control_replay.jsonl"))
        if r.get("variant") == "vision_only"
    }
    if len(original) != len(pred):
        raise ValueError("original vision-only replay prediction count mismatch")
    if any(r["frame_index"] not in original for r in rows):
        raise ValueError("missing vision-only baseline prediction")
    differences = np.asarray([
        abs(r["probabilities"]["0"] - original[r["frame_index"]]) for r in rows
    ])
    parity = {
        "rows": len(rows),
        "p95_abs_probability_error": float(np.percentile(differences, 95)),
        "max_abs_probability_error": float(differences.max()),
        "passes": bool(np.percentile(differences, 95) <= .00001
                       and differences.max() <= .00005),
    }
    if not parity["passes"]:
        raise ValueError(f"0ms vision-only frozen checkpoint parity failed: {parity}")
    origin = float(frames[0]["host_decoded_perf_counter_ms"]
                   - frames[0]["host_decoded_ms"])
    actual = touch_timeline(touches, origin)
    scores = compute_scores(rows, actual, args.shifts_ms, args.start_ms, args.end_ms)
    result = {
        "kind": "tiny_vision_only_noncausal_visual_phase_oracle",
        "read_only": True, "model": "frozen_tiny_control_vision_only",
        "run_stem": str(stem), "checkpoint_sha256": sha256(model_path),
        "window_ms": [args.start_ms, args.end_ms],
        "horizon_ms": 100, "original_target_time_preserved": True,
        "original_action_features_and_frame_dt_preserved": True,
        "shifts_ms": list(args.shifts_ms), "mapping": mapping,
        "frame_integrity": integrity, "vision_only_replay_parity": parity,
        "scores": scores,
        "limitations": [
            "Deliberately acausal visual lookahead; NOT a viable real-time control correction.",
            "Only RGB inputs change; original target timestamps and frame-delta features are held fixed.",
            "A shift over +100ms uses visual information from after the ORIGINAL H100 target.",
            "A vision-only model is insensitive to virtual/teacher state features by construction.",
            "getevent host arrival time is not synchronized with Android capture/encoding.",
            "MP4 debug recorder reported a nonzero FFmpeg exit, even though sampled replay parity passed.",
            "Fixed human-driven pixels cannot measure closed-loop physical policy quality.",
            "Do not select deployment timing, threshold, or model hyperparameters with this recording.",
        ],
    }
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with probabilities_path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("0ms parity:", parity)
    print("Common rows:", mapping["common_rows"], "of", mapping["eligible_rows"])
    for shift in args.shifts_ms:
        s = scores[str(shift)]
        print(f"RGB visual lookahead +{shift}ms:"
              f" PRESS {s['press']['matched']}/{s['press']['actual']}"
              f" RELEASE {s['release']['matched']}/{s['release']['actual']}"
              f" P(PRESS)@RELEASE+150 {s['release_plus150ms']['median_p_press']}")
    print("Report:", report_path)
    print("Predictions:", probabilities_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-stem", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/train_v4c2_temporal_v2.yaml")
    parser.add_argument("--road-config", type=Path,
                        default=ROOT / "configs/geometry_pseudo_labels.yaml")
    parser.add_argument("--start-ms", type=float, default=5000.)
    parser.add_argument("--end-ms", type=float, default=56800.)
    parser.add_argument("--shifts-ms", nargs="+", type=int,
                        default=list(DEFAULT_SHIFTS))
    parser.add_argument("--max-frame-skew-ms", type=float, default=45.)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--cpu-threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
