#!/usr/bin/env python3
"""Evaluate the label-only action persistence baseline on an ordered split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.train.baselines import persistence_points
from karting_agent.train.labels.touch_marker import ActionEvent
from karting_agent.train.sequence_evaluator import (
    ReleaseSegment,
    Transition,
    combine_evaluations,
    evaluate_sequence,
)
from karting_agent.train.trainer import load_samples, load_video_split, partition_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate action(t) -> action(t+horizon) persistence baseline."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Evaluate validation by default; test must be selected explicitly.",
    )
    parser.add_argument("--tolerance-ms", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_raw_config(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("train config root must be a mapping")
    return raw


def evaluation_tolerance(raw: dict[str, object], override: float | None) -> float:
    config = raw.get("evaluation", {})
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be a mapping")
    tolerance_ms = (
        float(config.get("transition_tolerance_ms", 100.0))
        if override is None
        else override
    )
    if tolerance_ms < 0:
        raise ValueError("tolerance must be >= 0")
    return tolerance_ms


def prediction_horizon(raw: dict[str, object]) -> float:
    dataset = raw.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("dataset config must be a mapping")
    return float(dataset.get("prediction_horizon_ms", 100.0))


def load_label_data(
    video: str,
) -> tuple[list[ActionEvent], list[Transition], list[ReleaseSegment]]:
    path = ROOT / "data" / "processed" / "labels" / f"{Path(video).stem}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"label file has no events: {path}")

    events = [
        ActionEvent(
            frame_index=int(raw_event["frame_index"]),
            timestamp_ms=float(raw_event["timestamp_ms"]),
            pressed=bool(raw_event["pressed"]),
        )
        for raw_event in raw_events
    ]
    transitions = [
        Transition(video=video, timestamp_ms=event.timestamp_ms, pressed=event.pressed)
        for event in events[1:]
    ]
    releases = [
        ReleaseSegment(video=video, start_ms=current.timestamp_ms, end_ms=nxt.timestamp_ms)
        for current, nxt in zip(events, events[1:])
        if not current.pressed and nxt.pressed
    ]
    return events, transitions, releases


def default_output(split: str, tolerance_ms: float) -> Path:
    tolerance_label = f"{tolerance_ms:g}".replace(".", "p")
    return (
        ROOT
        / "artifacts"
        / "baselines"
        / "persistence"
        / f"{split}_t{tolerance_label}_sequence.json"
    )


def main() -> int:
    args = parse_args()
    raw = load_raw_config(args.config)
    tolerance_ms = evaluation_tolerance(raw, args.tolerance_ms)
    horizon_ms = prediction_horizon(raw)
    split_config = load_video_split(args.config)

    all_samples = load_samples(ROOT / "data" / "processed" / "samples.jsonl")
    partitions = partition_samples(all_samples, split_config)
    samples = partitions[args.split]
    samples_by_video = defaultdict(list)
    for sample in samples:
        samples_by_video[sample.video].append(sample)

    print(
        f"Baseline: persistence action(t) -> action(t+{horizon_ms:g}ms); "
        f"split={args.split}; samples={len(samples)}; tolerance={tolerance_ms:.1f}ms",
        flush=True,
    )

    evaluations = []
    per_video: dict[str, object] = {}
    total_correct = 0
    configured_videos = getattr(split_config, args.split)

    for video in configured_videos:
        video_samples = samples_by_video.get(video)
        if not video_samples:
            raise RuntimeError(f"no samples for configured video: {video}")

        events, transitions, releases = load_label_data(video)
        points = persistence_points(video_samples, events)
        predicted_states = [point.probability >= 0.5 for point in points]
        sample_correct = sum(
            predicted == sample.target_pressed
            for predicted, sample in zip(predicted_states, video_samples)
        )
        total_correct += sample_correct

        evaluation = evaluate_sequence(
            points,
            transitions,
            releases,
            threshold=0.5,
            tolerance_ms=tolerance_ms,
        )
        evaluations.append(evaluation)
        summary = evaluation.summary()
        per_video[video] = {
            "sample_accuracy": sample_correct / len(video_samples),
            **summary,
        }

        transition = summary["transition"]["all"]
        press_timing = summary["onset_timing_ms"]["press"]
        release_timing = summary["onset_timing_ms"]["release"]
        short = summary["release_segment_recall"]["short_100_300ms"]
        print(
            f"{Path(video).name}: sample_acc={sample_correct / len(video_samples):.3f}, "
            f"transition_f1={transition['f1']:.3f}, "
            f"matched={transition['matched']}/{transition['ground_truth']}, "
            f"press_mean={press_timing['mean_error_ms']:.1f}ms, "
            f"release_mean={release_timing['mean_error_ms']:.1f}ms, "
            f"short_recall={short['recall']:.3f} "
            f"({short['detected']}/{short['segments']})",
            flush=True,
        )

    aggregate = combine_evaluations(evaluations).summary()
    sample_accuracy = total_correct / len(samples) if samples else 0.0
    transition = aggregate["transition"]["all"]
    press_timing = aggregate["onset_timing_ms"]["press"]
    release_timing = aggregate["onset_timing_ms"]["release"]
    short = aggregate["release_segment_recall"]["short_100_300ms"]

    print(
        "aggregate: "
        f"sample_acc={sample_accuracy:.3f}, "
        f"transition_f1={transition['f1']:.3f}, "
        f"precision={transition['precision']:.3f}, "
        f"recall={transition['recall']:.3f}, "
        f"press_mean={press_timing['mean_error_ms']:.1f}ms, "
        f"press_mae={press_timing['mae_ms']:.1f}ms, "
        f"release_mean={release_timing['mean_error_ms']:.1f}ms, "
        f"release_mae={release_timing['mae_ms']:.1f}ms, "
        f"short_recall={short['recall']:.3f} "
        f"({short['detected']}/{short['segments']})",
        flush=True,
    )

    output_path = args.output.resolve() if args.output is not None else default_output(
        args.split, tolerance_ms
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": "action_persistence",
        "definition": f"action(t) -> action(t+{horizon_ms:g}ms)",
        "split": args.split,
        "prediction_horizon_ms": horizon_ms,
        "sample_accuracy": sample_accuracy,
        "aggregate": aggregate,
        "per_video": per_video,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Evaluation: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
