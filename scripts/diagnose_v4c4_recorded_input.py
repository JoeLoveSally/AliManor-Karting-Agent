#!/usr/bin/env python3
"""Re-run V4-C4 inference on an existing ADB debug MP4, without Android access.

Use the precise source-frame indices recorded by the armed/dry-run controller.
This tests whether frames reconstructed from the debug video reproduce logged
probabilities; it is NOT a closed-loop or camera-capture latency measurement.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_adb_closed_loop as legacy  # noqa: E402
from karting_agent.model.event_time_runner import EventTimeActionRunner  # noqa: E402
from karting_agent.vision.preprocess import (  # noqa: E402
    preprocess_config_from_mapping,
    stack_frames,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit logged V4-C4 probabilities against the recorded ADB video (no ADB calls)."
    )
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None,
                        help="Recorded MP4; defaults to --run-json with .mp4 extension")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/models/mobilenet_v3_small_v4c4_event_time/model.pt")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--train-config", type=Path,
                        default=ROOT / "configs/train_v4c4_event_time.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--max-acceptable-probability-error", type=float, default=0.05,
                        help="Audit alarm threshold, NOT the control policy threshold")
    return parser.parse_args(argv)


def validate_run(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        raise ValueError("run JSON root must be an object")
    if payload.get("policy") != "v4c4_current_action_only":
        raise ValueError("run JSON was not produced by V4-C4 action-only runtime")
    if payload.get("recording_frame_mapping_valid") is not True:
        raise ValueError("recording_frame_mapping_valid is not true; cannot trust MP4 frame indices")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("run JSON has no model steps")
    result: list[dict[str, object]] = []
    last_observation_ms = float("-inf")
    last_frame_index = -1
    for raw in steps:
        if not isinstance(raw, dict):
            raise ValueError("step must be an object")
        history = raw.get("history_frame_indices")
        if not isinstance(history, list) or len(history) != 5:
            raise ValueError("each step needs five recorded history_frame_indices")
        indices = [int(index) for index in history]
        current_index = int(raw["source_frame_index"])
        timestamp_ms = float(raw["observation_timestamp_ms"])
        if (indices != sorted(indices) or indices[-1] != current_index
                or min(indices) < 0 or current_index <= last_frame_index
                or timestamp_ms <= last_observation_ms):
            raise ValueError("noncausal, unsorted, or nonmonotonic recorded history")
        probability = float(raw["action_probability"])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("invalid logged action probability")
        last_frame_index = current_index
        last_observation_ms = timestamp_ms
        result.append(raw)
    return result


def replay_recorded_steps(
    video_path: Path,
    steps: list[dict[str, object]],
    *,
    predict_action,
    preprocess_config,
) -> list[dict[str, object]]:
    """Stream each MP4 frame once; retain only images needed by future steps."""
    references = Counter(
        int(index) for step in steps for index in step["history_frame_indices"]
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open recorded MP4: {video_path}")
    current_frame_index = -1
    cached: dict[int, np.ndarray] = {}
    predictions: list[dict[str, object]] = []
    try:
        for step in steps:
            history = [int(index) for index in step["history_frame_indices"]]
            until = history[-1]
            while current_frame_index < until:
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"recorded MP4 ended before source frame {until}")
                current_frame_index += 1
                if current_frame_index in references:
                    cached[current_frame_index] = image
            stacked = stack_frames([cached[index] for index in history], preprocess_config)
            predicted = float(predict_action(stacked))
            logged = float(step["action_probability"])
            if not np.isfinite(predicted) or not 0.0 <= predicted <= 1.0:
                raise ValueError("replayed model produced an invalid probability")
            predictions.append({
                "source_frame_index": int(step["source_frame_index"]),
                "observation_timestamp_ms": float(step["observation_timestamp_ms"]),
                "logged_probability": logged,
                "replayed_probability": predicted,
                "absolute_error": abs(predicted - logged),
            })
            for index in history:
                references[index] -= 1
                if references[index] == 0:
                    del references[index]
                    del cached[index]
    finally:
        cap.release()
    return predictions


def summarize(
    predictions: list[dict[str, object]], *, decision_threshold: float,
    max_acceptable_error: float,
) -> dict[str, object]:
    if not 0 < decision_threshold < 1:
        raise ValueError("decision threshold must be strictly between 0 and 1")
    if max_acceptable_error < 0:
        raise ValueError("acceptable error must be nonnegative")
    errors = np.array([item["absolute_error"] for item in predictions], dtype=float)
    disagreements = [
        item for item in predictions
        if (item["logged_probability"] >= decision_threshold)
        != (item["replayed_probability"] >= decision_threshold)
    ]
    anomalies = sorted(predictions, key=lambda item: item["absolute_error"], reverse=True)[:10]
    return {
        "steps": len(predictions),
        "mean_absolute_error": float(errors.mean()),
        "p95_absolute_error": float(np.percentile(errors, 95)),
        "max_absolute_error": float(errors.max()),
        "decision_threshold": decision_threshold,
        "decision_disagreements": len(disagreements),
        "decision_disagreement_frames": [item["source_frame_index"] for item in disagreements],
        "max_acceptable_error": max_acceptable_error,
        "passes_requested_audit_tolerance": bool(errors.max() <= max_acceptable_error and not disagreements),
        "largest_errors": anomalies,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_path = args.run_json.resolve()
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    steps = validate_run(payload)
    video_path = (args.video if args.video is not None else run_path.with_suffix(".mp4")).resolve()
    preprocess = preprocess_config_from_mapping(legacy.load_mapping(args.train_config))
    model = EventTimeActionRunner(
        args.model, metadata_path=args.metadata, device=args.device,
        torch_num_threads=args.torch_num_threads,
    )
    result = summarize(
        replay_recorded_steps(
            video_path, steps, predict_action=model.predict_action,
            preprocess_config=preprocess,
        ),
        decision_threshold=args.decision_threshold,
        max_acceptable_error=args.max_acceptable_probability_error,
    )
    result["run_json"] = str(run_path)
    result["recording"] = str(video_path)
    result["model"] = str(args.model.resolve())
    result["warning"] = (
        "MP4 frames are decoded again and may differ by pixel rounding. "
        "This test checks replay consistency, NOT screen-capture latency, "
        "Android touch latency, visual domain shift, or closed-loop success."
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if result["passes_requested_audit_tolerance"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
