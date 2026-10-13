#!/usr/bin/env python3
"""Compare a new RGB action-history robustness checkpoint on the same recorded race.

Replay uses the fixed expert video pixels. In rollout mode feedback is the
candidate model's OWN scheduled action history, not the old Shadow baseline.
No controls are sent to Android; the trajectory is not counterfactual physics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from evaluate_tiny_live_recording import (  # noqa: E402
    read_jsonl, frame_consistency, touch_timeline, physical_press_spans,
    state_scores, match_events, paired_duration_drift, control_features,
    window,
)
from diagnose_release_timing import future_transition_diagnostics  # noqa: E402
from experiment_tiny_temporal_policy import prepare_video_feature  # noqa: E402
from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from karting_agent.model.tiny_temporal_policy import TinyTemporalPolicy  # noqa: E402
from karting_agent.train.heldout_guard import require_frozen_checkpoint  # noqa: E402
from karting_agent.train.shadow_control import ShadowController  # noqa: E402
from karting_agent.train.trainer import load_video_split  # noqa: E402
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def load_candidate(directory: Path, config: Path):
    weights = directory / "tiny_policy_rgb.pt"
    report_path = directory / "tiny_policy_rgb.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require_frozen_checkpoint(
        report, mode="rgb", split=load_video_split(config)
    )
    robustness = report.get("history_robustness", {})
    probability = robustness.get("probability")
    if not isinstance(probability, (float, int)) or not 0 < probability <= 1:
        raise ValueError("candidate must have nonzero training history corruption")
    model = TinyTemporalPolicy(image_channels=3)
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    return model, {
        "weights_sha256": sha256(weights), "metadata_sha256": sha256(report_path),
        "history_robustness": robustness, "split": report["split"],
    }


def load_required_rgb(video: Path, pred: list[dict], frame_count: int,
                      config: Path, road_config: Path):
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    preprocess = preprocess_config_from_mapping(raw)
    road, _, _ = load_road_config(road_config)
    needed = {int(i) for row in pred for i in row["input_indices"]}
    if not needed or max(needed) >= frame_count:
        raise ValueError("invalid requested recorded frame indices")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open recording: {video}")
    result = {}
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in needed:
                result[index] = prepare_video_feature(
                    frame, preprocess=preprocess, road_config=road
                )[:3]
            index += 1
    finally:
        capture.release()
    if index != frame_count or len(result) != len(needed):
        raise ValueError(
            f"MP4 frame count/mapping invalid: decoded {index}, expected {frame_count}"
        )
    return result


def infer(model, features, image_indices, stamps, controls):
    images = np.stack([features[int(i)] for i in image_indices])
    images = np.asarray(images, dtype=np.float32)[None] / 255.0
    with torch.inference_mode():
        probability = torch.sigmoid(model(
            torch.from_numpy(images), torch.from_numpy(controls[None])
        )).reshape(-1)
    if probability.numel() != 1:
        raise ValueError("expected scalar PRESS prediction")
    return float(probability[0])


def replay_candidate(model, features, predictions, actual):
    """Teacher and autonomous feedback over IDENTICAL recorded video frames."""
    controller = ShadowController(initial_pressed=False, horizon_ms=100.,
                                  threshold=.5, latency_ms=0.)
    rows = []
    for row in predictions:
        observe = float(row["observation_ms"])
        stamps = tuple(map(float, row["input_timestamp_ms"]))
        target = float(row["models"]["rgb"]["target_ms"])
        if abs(target - observe - 100.) > .01:
            raise ValueError("unexpected forecast horizon in original Shadow log")
        controller.advance_to(observe)
        expert_ctl = control_features(stamps, actual)
        virtual_ctl = controller.control_features(stamps)
        expert = infer(model, features, row["input_indices"], stamps, expert_ctl)
        rollout = infer(model, features, row["input_indices"], stamps, virtual_ctl)
        controller.propose(
            observe_ms=observe, target_ms=target,
            probability=rollout, ready_ms=observe,
        )
        rows.append({
            "frame_index": row["frame_index"], "observation_ms": observe,
            "target_ms": target, "teacher_prob": expert,
            "rollout_prob": rollout,
        })
    controller.flush()
    events = [(float(item.time_ms), bool(item.pressed))
              for item in controller.events]
    return rows, events


def compare_run(rows, actual, virtual, touch_spans, start, end):
    expert_points = [(r["target_ms"], r["teacher_prob"] >= .5)
                     for r in rows if start <= r["target_ms"] < end]
    rollout_points = [(r["target_ms"], r["rollout_prob"] >= .5)
                      for r in rows if start <= r["target_ms"] < end]
    result = {}
    human = window(actual[1:], start, end)
    for name, points in (("teacher", expert_points), ("rollout", rollout_points)):
        if name == "teacher":
            # Clean transition points derived from decisions at 30Hz; this is
            # NOT an executed queue, so compare only these discrete decisions.
            from evaluate_tiny_live_recording import transitions_from_points
            actions = [(0., False)] + transitions_from_points(points)
        else:
            actions = virtual
        events = window(actions[1:], start, end)
        result[name] = {
            "state_at_target": state_scores(points, actual),
            "press_events": match_events(events, human, pressed=True),
            "release_events": match_events(events, human, pressed=False),
        }
        if name == "rollout":
            result[name]["paired_hold_duration"] = paired_duration_drift(
                actions, actual, start, end, physical_spans=touch_spans
            )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-stem", type=Path, required=True)
    ap.add_argument("--candidate-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/train_v4c2_temporal_v2.yaml")
    ap.add_argument("--road-config", type=Path,
                    default=ROOT / "configs/geometry_pseudo_labels.yaml")
    ap.add_argument("--race-start-ms", type=float, default=5000.)
    ap.add_argument("--race-end-ms", type=float, default=56800.)
    ap.add_argument("--cpu-threads", type=int, default=4)
    args = ap.parse_args()
    if (not args.output_dir.is_dir() or args.cpu_threads < 1
            or args.race_end_ms <= args.race_start_ms):
        ap.error("output-dir must exist, cpu-threads positive, race range valid")
    torch.set_num_threads(args.cpu_threads)
    stem = args.run_stem
    summary = json.loads(Path(str(stem) + "_summary.json").read_text())
    pred = [r for r in read_jsonl(Path(str(stem) + ".jsonl"))
            if r.get("kind") == "prediction"]
    touches = read_jsonl(Path(str(stem) + "_touch.jsonl"))
    frames = read_jsonl(Path(str(stem) + "_frames.jsonl"))
    integrity = frame_consistency(summary, frames, pred)
    model, provenance = load_candidate(args.candidate_dir, args.config)
    if provenance["weights_sha256"] == summary["models"]["rgb"]["weights_sha256"]:
        raise ValueError("candidate checkpoint unexpectedly equals frozen baseline")
    features = load_required_rgb(
        Path(str(stem) + ".mp4"), pred, len(frames),
        args.config, args.road_config
    )
    origin = float(frames[0]["host_decoded_perf_counter_ms"]
                   - frames[0]["host_decoded_ms"])
    truth = touch_timeline(touches, origin)
    phys = physical_press_spans(
        touches, origin, args.race_start_ms, args.race_end_ms
    )
    rows, events = replay_candidate(model, features, pred, truth)
    scored = compare_run(rows, truth, events, phys,
                         args.race_start_ms, args.race_end_ms)

    offline_ref = args.run_stem.parent / (
        args.run_stem.name + "_offline_eval_summary.json"
    )
    baseline = None
    if offline_ref.is_file():
        old = json.loads(offline_ref.read_text(encoding="utf-8"))
        if old.get("ablations", {}).get("interpretation_valid_only_if_parity_passed"):
            baseline = old["recorded_virtual"]["rgb"]
    result = {
        "kind": "action_history_robustness_candidate_evaluation",
        "read_only": True, "same_recorded_video": True,
        "no_counterfactual_scene_control": True,
        "same_original_training_preprocessing": True,
        "candidate": provenance, "video_integrity": integrity,
        "window_ms": [args.race_start_ms, args.race_end_ms],
        "uncalibrated_reference_time": True,
        "candidate_scores": scored,
        "recorded_frozen_rgb_baseline": baseline,
        "candidate_forecast_shift_sensitivity": {
            str(offset): future_transition_diagnostics(
                [
                    {"target_ms": row["target_ms"], "probabilities": {
                        "candidate_teacher": row["teacher_prob"],
                        "candidate_rollout": row["rollout_prob"],
                    }}
                    for row in rows
                ],
                touch_timeline(touches, origin, offset),
                args.race_start_ms, args.race_end_ms,
            )
            for offset in (-150, -100, -50, 0, 50, 100, 150)
        },
        "notes": [
            "The candidate autoregressively feeds its own scheduled action history.",
            "The phone video is unchanged by predicted actions: no closed-loop physics.",
            "Touch/video relative time remains uncalibrated.",
            "Candidate deployment requires separate live Shadow evaluation; original recorded "
            "live parity only validates source video/preprocess, not candidate predictions.",
        ],
    }
    out = args.output_dir / (
        stem.name + "_ahr_" + str(provenance["history_robustness"]["probability"])
        .replace(".", "p") + "_evaluation.json"
    )
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("Candidate evaluation:", out)
    for name in ("teacher", "rollout"):
        row = scored[name]
        print(name, "acc=", round(row["state_at_target"]["accuracy"], 4),
              "PRESS", row["press_events"]["matched"], "/",
              row["press_events"]["actual"],
              "RELEASE", row["release_events"]["matched"], "/",
              row["release_events"]["actual"])


if __name__ == "__main__":
    main()
