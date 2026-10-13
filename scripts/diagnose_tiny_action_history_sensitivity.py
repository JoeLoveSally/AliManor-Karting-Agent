#!/usr/bin/env python3
"""Read-only Tiny RGB action-history sensitivity on a recorded Shadow video.

All scenarios share the same decoded video frames and frozen checkpoint.
Teacher/recorded virtual histories come from different timelines; hybrid
features are intentionally *incoherent* mechanistic perturbations, not valid
control trajectories. No model training or Android control is performed.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = (
    "teacher", "virtual", "virtual_state_teacher_age", "teacher_state_virtual_age",
    "held_press_200", "held_press_350", "held_press_500",
    "held_release_200", "held_release_350", "held_release_500",
)
RELATIVE_WINDOW_MS = (-200.0, 300.0)


def coherent_held_controls(stamps, *, pressed: bool, age_at_observation_ms: float):
    """Single held state over the entire observation history, with legal age.

    The hypothetical last change occurs at t_last - age. Requiring the age
    to cover the full frame span avoids inventing a mid-window transition.
    """
    values = np.asarray(stamps, dtype=np.float64)
    if (values.ndim != 1 or len(values) != 5 or not np.isfinite(values).all()
            or np.any(np.diff(values) < 0)):
        raise ValueError("expected five nondecreasing finite image timestamps")
    duration = float(age_at_observation_ms)
    if not math.isfinite(duration) or duration < values[-1] - values[0]:
        raise ValueError("held history must start no later than the first frame")
    controls = np.zeros((len(values), 3), dtype=np.float32)
    controls[:, 0] = float(pressed)
    controls[:, 1] = np.minimum((duration + values - values[-1]) / 500.0, 1.0)
    controls[1:, 2] = np.minimum(np.diff(values), 200.0) / 200.0
    return controls


def make_scenarios(stamps, teacher: np.ndarray, virtual: np.ndarray):
    """Generate controlled interventions; hybrids are off-manifold by design."""
    if (teacher.shape != (5, 3) or virtual.shape != (5, 3)
            or not np.isfinite(teacher).all() or not np.isfinite(virtual).all()):
        raise ValueError("expected finite [5,3] causal controls")
    if not np.allclose(teacher[:, 2], virtual[:, 2], atol=1e-7):
        raise ValueError("teacher and virtual history must use identical frame deltas")
    state_swap = teacher.copy()
    state_swap[:, 0] = virtual[:, 0]
    age_swap = teacher.copy()
    age_swap[:, 1] = virtual[:, 1]
    result = {
        "teacher": teacher.copy(),
        "virtual": virtual.copy(),
        "virtual_state_teacher_age": state_swap,
        "teacher_state_virtual_age": age_swap,
    }
    for pressed, name in ((True, "press"), (False, "release")):
        for age in (200, 350, 500):
            # A recorded five-frame grid can span slightly more than 200ms.
            # Use the requested age or that frame span, whichever is larger.
            effective_age = max(float(age), float(stamps[-1] - stamps[0]))
            result[f"held_{name}_{age}"] = coherent_held_controls(
                stamps, pressed=pressed, age_at_observation_ms=effective_age
            )
    return result


def release_events(timeline, start_ms: float, end_ms: float):
    return [float(t) for t, pressed in timeline[1:]
            if not pressed and start_ms <= t < end_ms]


def select_release_rows(predictions, releases, *, lower=-200.0, upper=300.0):
    """Select a prediction once if its target is near any human RELEASE."""
    if not releases:
        raise ValueError("no reference RELEASE events inside evaluation window")
    ordered = sorted(releases)
    selected = []
    for row in predictions:
        t = float(row["models"]["rgb"]["target_ms"])
        idx = bisect_left(ordered, t)
        offsets = [t - ordered[j] for j in (idx - 1, idx)
                   if 0 <= j < len(ordered) and lower <= t - ordered[j] <= upper]
        if offsets:
            selected.append((row, min(offsets, key=abs)))
    return selected


def sensitivity_metrics(results):
    """Paired model perturbation metrics, *not* closed-loop control scores."""
    if not results:
        raise ValueError("no prediction results")
    out = {}
    baseline = np.asarray([r["probabilities"]["teacher"] for r in results])
    for name in SCENARIOS:
        scores = np.asarray([r["probabilities"][name] for r in results])
        delta = scores - baseline
        out[name] = {
            "n": len(scores),
            "median_p_press": float(np.median(scores)),
            "median_delta_from_teacher": float(np.median(delta)),
            "p95_abs_delta_from_teacher": float(np.percentile(abs(delta), 95)),
            "threshold_disagreements_with_teacher": int(
                np.count_nonzero((scores >= .5) != (baseline >= .5))
            ),
        }
    effects = {
        "state_replacement": np.asarray([
            r["probabilities"]["virtual_state_teacher_age"]
            - r["probabilities"]["teacher"] for r in results
        ]),
        "age_replacement": np.asarray([
            r["probabilities"]["teacher_state_virtual_age"]
            - r["probabilities"]["teacher"] for r in results
        ]),
        "interaction": np.asarray([
            r["probabilities"]["virtual"]
            - r["probabilities"]["virtual_state_teacher_age"]
            - r["probabilities"]["teacher_state_virtual_age"]
            + r["probabilities"]["teacher"] for r in results
        ]),
    }
    return out, {
        key: {"median_delta_p_press": float(np.median(value)),
              "p95_abs_delta_p_press": float(np.percentile(abs(value), 95))}
        for key, value in effects.items()
    }


def release_bin_summary(results, *, centers=(0, 50, 100, 150, 200, 250), half_width=20):
    out = {}
    for center in centers:
        subset = [r for r in results if abs(r["relative_to_release_ms"] - center)
                  <= half_width]
        if not subset:
            out[str(center)] = {"n": 0}
            continue
        info = {"n": len(subset), "median_p_press": {
            name: float(statistics.median(r["probabilities"][name] for r in subset))
            for name in SCENARIOS
        }}
        info["teacher_release_virtual_hold_count"] = sum(
            r["probabilities"]["teacher"] < .5
            and r["probabilities"]["virtual"] >= .5 for r in subset
        )
        out[str(center)] = info
    return out


def predict_scenarios(model, selected, features, teacher_events, virtual_events,
                      *, batch_size: int = 32):
    """Run all paired scenarios using the same frozen CPU model and pixels."""
    import torch
    from evaluate_tiny_live_recording import control_features

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    records = []
    for row, relative in selected:
        stamps = [float(t) for t in row["input_timestamp_ms"]]
        teacher = control_features(stamps, teacher_events)
        virtual = control_features(stamps, virtual_events)
        scenario_controls = make_scenarios(stamps, teacher, virtual)
        images = np.stack([features[int(i)] for i in row["input_indices"]])
        records.append({
            "frame_index": int(row["frame_index"]),
            "observation_ms": float(row["observation_ms"]),
            "target_ms": float(row["models"]["rgb"]["target_ms"]),
            "relative_to_release_ms": float(relative),
            "logged_live_p_press": float(row["models"]["rgb"]["probability"]),
            "images": images,
            "controls": scenario_controls,
            "probabilities": {},
        })
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            block = records[offset:offset + batch_size]
            images = torch.from_numpy(np.stack([r["images"] for r in block])
                                      .astype(np.float32) / 255.0)
            for name in SCENARIOS:
                control = torch.from_numpy(
                    np.stack([r["controls"][name] for r in block])
                )
                probs = torch.sigmoid(model(images, control)).cpu().numpy()
                if probs.shape != (len(block),) or not np.isfinite(probs).all():
                    raise ValueError("unexpected or nonfinite model probability")
                for record, prob in zip(block, probs):
                    record["probabilities"][name] = float(prob)
    for r in records:
        del r["images"]
        del r["controls"]
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-stem", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/train_v4c2_temporal_v2.yaml")
    parser.add_argument("--road-config", type=Path,
                        default=ROOT / "configs/geometry_pseudo_labels.yaml")
    parser.add_argument("--race-start-ms", type=float, default=5000)
    parser.add_argument("--race-end-ms", type=float, default=56800)
    parser.add_argument("--touch-shift-ms", type=float, default=0,
                        help="Exploratory uncalibrated shift, NOT a physical latency fix")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if (not args.output_dir.is_dir() or not args.checkpoint_dir.is_dir()
            or args.cpu_threads < 1 or args.batch_size < 1
            or not 0 <= args.race_start_ms < args.race_end_ms):
        parser.error("invalid directory, batch size, threads, or race window")

    import torch
    for folder in (ROOT / "src", ROOT / "scripts"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    from evaluate_tiny_live_recording import (
        read_jsonl, frame_consistency, touch_timeline,
    )
    from evaluate_history_candidate import load_required_rgb
    from run_tiny_live_shadow import load_frozen_models

    torch.set_num_threads(args.cpu_threads)
    stem = args.run_stem.with_suffix("") if args.run_stem.suffix == ".mp4" else args.run_stem
    outfile = args.output_dir / f"{stem.name}_action_history_sensitivity.json"
    detail = args.output_dir / f"{stem.name}_action_history_sensitivity.jsonl"
    if outfile.exists() or detail.exists():
        raise FileExistsError("refusing to overwrite an existing diagnosis")
    summary = json.loads(Path(f"{stem}_summary.json").read_text(encoding="utf-8"))
    pred = [r for r in read_jsonl(Path(f"{stem}.jsonl"))
            if r.get("kind") == "prediction"]
    touch = read_jsonl(Path(f"{stem}_touch.jsonl"))
    frames = read_jsonl(Path(f"{stem}_frames.jsonl"))
    integrity = frame_consistency(summary, frames, pred)
    if not frames or not pred:
        raise ValueError("no recorded frames or predictions")
    origin = float(frames[0]["host_decoded_perf_counter_ms"]
                   - frames[0]["host_decoded_ms"])
    teacher_events = touch_timeline(touch, origin, args.touch_shift_ms)
    virtual_events = [
        (float(e["ms"]), bool(e["pressed"]))
        for e in summary["summary"]["virtual_events"]["rgb"]
    ]
    releases = release_events(teacher_events, args.race_start_ms, args.race_end_ms)
    selected = select_release_rows(pred, releases,
                                   lower=RELATIVE_WINDOW_MS[0],
                                   upper=RELATIVE_WINDOW_MS[1])
    if not selected:
        raise ValueError("no prediction rows around reference RELEASE events")
    models, provenance = load_frozen_models(args.checkpoint_dir, args.config)
    if provenance["rgb"]["weights_sha256"] != summary["models"]["rgb"]["weights_sha256"]:
        raise ValueError("RGB checkpoint does not match the recorded Shadow run")
    features = load_required_rgb(
        Path(f"{stem}.mp4"), [row for row, _ in selected],
        len(frames), args.config, args.road_config,
    )
    rows = predict_scenarios(models["rgb"], selected, features,
                             teacher_events, virtual_events,
                             batch_size=args.batch_size)
    deltas = np.abs(np.asarray([
        r["probabilities"]["virtual"] - r["logged_live_p_press"]
        for r in rows
    ]))
    parity = {"n": len(deltas), "p95_abs_prob_delta": float(np.percentile(deltas, 95)),
              "max_abs_prob_delta": float(np.max(deltas)),
              "passes": bool(np.percentile(deltas, 95) <= .005 and np.max(deltas) <= .05)}
    if not parity["passes"]:
        raise ValueError(f"frozen RGB live replay parity failed; do not interpret: {parity}")
    metrics, factors = sensitivity_metrics(rows)
    result = {
        "kind": "tiny_action_history_sensitivity_read_only",
        "recorded_run": str(stem), "model": provenance["rgb"],
        "video_integrity": integrity, "parity": parity,
        "race_window_ms": [args.race_start_ms, args.race_end_ms],
        "touch_shift_ms_uncalibrated": args.touch_shift_ms,
        "release_events": len(releases), "selected_prediction_rows": len(rows),
        "scenarios": metrics, "factorial_diagnostic_effects": factors,
        "release_relative_target_bins_ms": release_bin_summary(rows),
        "caveats": [
            "This is a fixed human-driven video; virtual state cannot change vehicle pixels.",
            "Physical touch timestamps are host arrival times, not synchronized capture times.",
            "State-only and age-only swaps can be causally inconsistent: they are sensitivity probes, not feasible actions.",
            "Held-history probes are internally causal but may not match the pictured physical state.",
            "Per-observation scenarios do not roll out their own predicted control states.",
            "A passing probability parity check validates the sampled RGB model inputs, not frame-perfect video authenticity.",
        ],
    }
    with outfile.open("x", encoding="utf-8") as fd:
        json.dump(result, fd, indent=2, ensure_ascii=False)
        fd.write("\n")
    with detail.open("x", encoding="utf-8") as fd:
        for row in rows:
            fd.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Summary:", outfile)
    print("Paired predictions:", detail)
    print("Releases:", len(releases), "selected predictions:", len(rows))
    print("Replay parity:", parity)
    print("Sensitivity effects:", factors)


if __name__ == "__main__":
    main()
