#!/usr/bin/env python3
"""Read-only, independent self-fed replays of trained Tiny RGB ablations.

Identical fixed expert video pixels and timestamps for every variant; each
variant owns a separate causal virtual action queue. This is NOT a physical
closed-loop simulation, and getevent host arrival time is NOT calibrated to
video capture time. No ADB commands, model training, Test split or overwrites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("full", "vision_only", "state_only")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_training_report(report: dict, *, variant: str, split) -> None:
    from karting_agent.model.tiny_control_ablation import CONTROL_MASKS

    if (variant not in VARIANTS or
            report.get("kind") != "tiny_rgb_control_input_ablation_training" or
            report.get("variant") != variant or
            report.get("smoke_unrepresentative") is not False or
            report.get("control_mask") != list(CONTROL_MASKS[variant]) or
            report.get("split") != split.name or
            report.get("target") != "expert_PRESS_at_observation_plus_100ms" or
            tuple(report.get("test_videos_not_evaluated", ())) != tuple(split.test)):
        raise ValueError(f"untrusted or incompatible full-training report: {variant}")


def load_models(checkpoints: Path, split):
    import torch
    from karting_agent.model.tiny_control_ablation import TinyControlAblation

    models, provenance = {}, {}
    for variant in VARIANTS:
        weights = checkpoints / f"tiny_control_{variant}.pt"
        metadata = checkpoints / f"tiny_control_{variant}.json"
        if not weights.is_file() or not metadata.is_file():
            raise FileNotFoundError(f"missing {variant} frozen checkpoint or metadata")
        report = json.loads(metadata.read_text(encoding="utf-8"))
        validate_training_report(report, variant=variant, split=split)
        model = TinyControlAblation(variant)
        model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True),
            strict=True,
        )
        model.eval()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError(f"nonfinite {variant} model weights")
        models[variant] = model
        provenance[variant] = {
            "checkpoint_sha256": sha256(weights),
            "training_report_sha256": sha256(metadata),
            "selected_epoch": min(
                report["history"], key=lambda x: x["validation"]["bce"]
            )["epoch"],
            "validation": report["validation_sequence"]["candidate"],
        }
    return models, provenance


def release_windows(rows: list[dict], releases: list[float],
                    *, left_ms: float = 200.0, right_ms: float = 300.0):
    """Descriptive fixed-video probabilities around uncalibrated RELEASE time."""
    out = []
    for release in releases:
        nearby = [
            r for r in rows
            if -left_ms <= float(r["target_ms"]) - release <= right_ms
        ]
        after = [r for r in nearby if 0 <= float(r["target_ms"]) - release <= right_ms]

        def first_release(key: str):
            return next(
                (float(r["target_ms"]) - release for r in after if r[key] < .5),
                None,
            )

        teacher_at = first_release("teacher_prob")
        virtual_at = first_release("rollout_prob")
        conflicts = [
            r for r in nearby
            if r["teacher_prob"] < .5 and r["rollout_prob"] >= .5
        ]
        out.append({
            "human_release_host_ms_uncalibrated": release,
            "prediction_rows": len(nearby),
            "teacher_release_virtual_press_conflict": bool(conflicts),
            "conflict_rows": len(conflicts),
            "first_teacher_release_after_ms": teacher_at,
            "first_virtual_release_after_ms": virtual_at,
            "virtual_minus_teacher_first_release_ms": (
                virtual_at - teacher_at
                if teacher_at is not None and virtual_at is not None else None
            ),
        })
    return out


def aggregate_release_windows(windows: list[dict]):
    paired = [
        x["virtual_minus_teacher_first_release_ms"]
        for x in windows if x["virtual_minus_teacher_first_release_ms"] is not None
    ]
    return {
        "human_release_events": len(windows),
        "events_with_teacher_release_virtual_press_conflict": sum(
            x["teacher_release_virtual_press_conflict"] for x in windows
        ),
        "events_with_both_first_release_predictions": len(paired),
        "virtual_first_release_later": sum(x > 0 for x in paired),
        "virtual_first_release_earlier": sum(x < 0 for x in paired),
        "virtual_first_release_equal": sum(x == 0 for x in paired),
        "virtual_first_release_absent_in_window": sum(
            x["first_virtual_release_after_ms"] is None for x in windows
        ),
        "median_virtual_minus_teacher_first_release_ms": (
            __import__("statistics").median(paired) if paired else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-stem", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/train_v4c2_temporal_v2.yaml")
    parser.add_argument("--road-config", type=Path,
                        default=ROOT / "configs/geometry_pseudo_labels.yaml")
    parser.add_argument("--race-start-ms", type=float, default=5000.0)
    parser.add_argument("--race-end-ms", type=float, default=56800.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if (not args.output_dir.is_dir() or not args.checkpoint_dir.is_dir()
            or not 0 <= args.race_start_ms < args.race_end_ms
            or args.cpu_threads < 1):
        parser.error("invalid output/checkpoint directory, time range or CPU threads")

    for directory in (ROOT / "src", ROOT / "scripts"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import torch
    from evaluate_history_candidate import (
        compare_run, load_required_rgb, replay_candidate,
    )
    from evaluate_tiny_live_recording import (
        frame_consistency, physical_press_spans, read_jsonl,
        touch_timeline, window,
    )
    from karting_agent.train.trainer import load_video_split

    torch.set_num_threads(args.cpu_threads)
    stem = args.run_stem.with_suffix("") if args.run_stem.suffix == ".mp4" else args.run_stem
    output = args.output_dir / f"{stem.name}_tiny_control_replay.json"
    rows_path = args.output_dir / f"{stem.name}_tiny_control_replay.jsonl"
    if output.exists() or rows_path.exists():
        raise FileExistsError("refusing to overwrite an existing Tiny control replay")
    # Previous paired original-checkpoint CPU replay must have passed its
    # strict numerical parity before accepting the logged frame-to-pixel join.
    parity_path = Path(f"{stem}_action_history_sensitivity.json")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    summary = json.loads(Path(f"{stem}_summary.json").read_text(encoding="utf-8"))
    if (parity.get("recorded_run") != str(stem) or
            parity.get("parity", {}).get("passes") is not True or
            parity.get("model", {}).get("weights_sha256") !=
            summary["models"]["rgb"]["weights_sha256"]):
        raise ValueError("recorded Shadow source did not pass verified frozen RGB parity")

    pred = [
        row for row in read_jsonl(Path(f"{stem}.jsonl"))
        if row.get("kind") == "prediction"
    ]
    frames = read_jsonl(Path(f"{stem}_frames.jsonl"))
    touches = read_jsonl(Path(f"{stem}_touch.jsonl"))
    integrity = frame_consistency(summary, frames, pred)
    if not frames or not pred:
        raise ValueError("recording lacks decoded frames or model predictions")
    origin = float(frames[0]["host_decoded_perf_counter_ms"]
                   - frames[0]["host_decoded_ms"])
    human = touch_timeline(touches, origin)
    physical = physical_press_spans(
        touches, origin, args.race_start_ms, args.race_end_ms
    )
    releases = [
        t for t, pressed in window(human[1:], args.race_start_ms, args.race_end_ms)
        if not pressed
    ]
    if not releases:
        raise ValueError("no recorded RELEASE events in evaluation window")
    split = load_video_split(args.config)
    models, provenance = load_models(args.checkpoint_dir, split)
    images = load_required_rgb(
        Path(f"{stem}.mp4"), pred, len(frames), args.config, args.road_config,
    )

    result = {
        "kind": "tiny_rgb_control_ablation_fixed_video_self_fed_replay",
        "read_only": True,
        "fixed_human_video_pixels": True,
        "uncalibrated_getevent_host_arrival_time": True,
        "idealized_zero_inference_latency": True,
        "prediction_horizon_ms": 100.0,
        "threshold": 0.5,
        "run_stem": str(stem),
        "window_ms": [args.race_start_ms, args.race_end_ms],
        "frame_integrity": integrity,
        "source_rgb_parity": parity["parity"],
        "checkpoint_provenance": provenance,
        "variants": {},
        "limitations": [
            "Every policy sees fixed human video: this is NOT counterfactual vehicle motion.",
            "Model self-fed histories are simulated separately; a virtual action never changes pixels.",
            "Original live Shadow used measured inference readiness, but this replay uses ideal t+100ms execution.",
            "Human touch host arrival and decoder availability have unknown relative capture delays.",
            "Training Validation was teacher-forced: replay disagreement is diagnostic, NOT closed-loop success.",
            "The original debug MP4 may have returned nonzero ffmpeg exit status; sampled RGB parity passed.",
        ],
    }
    details = []
    for variant in VARIANTS:
        rows, events = replay_candidate(models[variant], images, pred, human)
        scores = compare_run(
            rows, human, events, physical, args.race_start_ms, args.race_end_ms,
        )
        per_release = release_windows(rows, releases)
        summary_release = aggregate_release_windows(per_release)
        result["variants"][variant] = {
            "scores": scores,
            "release_window_summary": summary_release,
            "per_release": per_release,
            "simulated_virtual_switches": len(events) - 1,
        }
        details.extend({"variant": variant, **row} for row in rows)
        virtual = scores["rollout"]
        print(
            f"{variant}: rollout RELEASE {virtual['release_events']['matched']}"
            f"/{virtual['release_events']['actual']}, "
            f"PRESS {virtual['press_events']['matched']}"
            f"/{virtual['press_events']['actual']}, "
            f"conflicts {summary_release['events_with_teacher_release_virtual_press_conflict']}"
            f"/{len(releases)}", flush=True,
        )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with rows_path.open("x", encoding="utf-8") as handle:
        for item in details:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Summary:", output, flush=True)
    print("Detailed paired probabilities:", rows_path, flush=True)


if __name__ == "__main__":
    main()
