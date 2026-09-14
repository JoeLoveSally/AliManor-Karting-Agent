#!/usr/bin/env python3
"""Evaluate a v2 multi-horizon artifact with correct per-horizon subsets and baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.model.runner import ModelRunner
from karting_agent.train.evaluator import BinaryMetricAccumulator
from karting_agent.train.frame_cache import frame_cache_root_from_config
from karting_agent.train.labels.touch_marker import ActionEvent, action_at_timestamp
from karting_agent.train.sequence_evaluator import (
    ReleaseSegment,
    SequencePoint,
    Transition,
    combine_evaluations,
    evaluate_sequence,
)
from karting_agent.train.trainer import load_samples, load_train_loop_config, load_video_split, partition_samples
from karting_agent.train.video_dataset import TemporalVideoDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate v2 model heads independently, including per-horizon transition/short "
            "subsets, sequence timing, and action-persistence baselines."
        )
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_v2.yaml")
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "processed" / "v2" / "samples.jsonl",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "v2" / "labels",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tolerance-ms", type=float, default=None)
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Fail instead of falling back to MP4 when the frame cache is missing.",
    )
    return parser.parse_args()


def load_yaml_mapping(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def artifact_dir_from_config(raw: dict[str, object]) -> Path:
    artifact = raw.get("artifact", {})
    if not isinstance(artifact, dict):
        raise ValueError("artifact config must be a mapping")
    name = str(artifact.get("name", "mobilenet_v3_small_v2"))
    return ROOT / "artifacts" / "models" / name


def evaluation_tolerance(raw: dict[str, object], override: float | None) -> float:
    evaluation = raw.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation config must be a mapping")
    tolerance = (
        float(evaluation.get("transition_tolerance_ms", 100.0))
        if override is None
        else override
    )
    if tolerance < 0:
        raise ValueError("transition tolerance must be >= 0")
    return tolerance


def load_label_data(
    labels_dir: Path,
    video: str,
) -> tuple[list[ActionEvent], list[Transition], list[ReleaseSegment]]:
    path = labels_dir / f"{Path(video).stem}.json"
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


def sample_horizon_fields(sample, horizon_count: int):
    targets = sample.target_states
    timestamps = sample.target_timestamps_ms or (sample.target_timestamp_ms,)
    near_transition = sample.near_transition_by_horizon or (sample.near_transition,)
    near_short = sample.near_short_correction_by_horizon or (sample.near_short_correction,)
    fields = (targets, timestamps, near_transition, near_short)
    if any(len(values) != horizon_count for values in fields):
        raise ValueError(
            f"sample horizon metadata mismatch for {sample.video}: "
            f"expected {horizon_count}, got {[len(values) for values in fields]}"
        )
    return targets, timestamps, near_transition, near_short


def metric_dict(probabilities, targets, mask, threshold: float) -> dict[str, float | int]:
    accumulator = BinaryMetricAccumulator(threshold)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    accumulator.update(probabilities[mask], targets[mask])
    return accumulator.result().to_dict()


def sequence_summary(
    samples,
    probabilities: np.ndarray,
    horizon_index: int,
    *,
    label_data: dict[str, tuple[list[ActionEvent], list[Transition], list[ReleaseSegment]]],
    threshold: float,
    tolerance_ms: float,
    persistence: bool,
) -> dict[str, object]:
    points_by_video: dict[str, list[SequencePoint]] = defaultdict(list)
    for sample_index, sample in enumerate(samples):
        _, timestamps, _, _ = sample_horizon_fields(sample, probabilities.shape[1])
        if persistence:
            events = label_data[sample.video][0]
            probability = 1.0 if action_at_timestamp(events, sample.input_timestamps_ms[-1]) else 0.0
        else:
            probability = float(probabilities[sample_index, horizon_index])
        points_by_video[sample.video].append(
            SequencePoint(
                video=sample.video,
                timestamp_ms=float(timestamps[horizon_index]),
                probability=probability,
            )
        )

    evaluations = []
    for video, points in points_by_video.items():
        _, transitions, releases = label_data[video]
        evaluations.append(
            evaluate_sequence(
                points,
                transitions,
                releases,
                threshold=threshold,
                tolerance_ms=tolerance_ms,
            )
        )
    return combine_evaluations(evaluations).summary()


def main() -> int:
    args = parse_args()
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be in (0, 1)")

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: python -m pip install -e ".[train]"'
        ) from exc

    config_path = args.config.resolve()
    samples_path = args.samples.resolve()
    labels_dir = args.labels_dir.resolve()
    raw = load_yaml_mapping(config_path)
    loop_config = load_train_loop_config(config_path)
    split_config = load_video_split(config_path)
    tolerance_ms = evaluation_tolerance(raw, args.tolerance_ms)
    artifact_dir = artifact_dir_from_config(raw)
    model_path = args.model.resolve() if args.model is not None else artifact_dir / "model.pt"
    output_path = (
        args.output.resolve()
        if args.output is not None
        else artifact_dir / "evaluation" / f"{args.split}_v2.json"
    )

    runner = ModelRunner(model_path, device=args.device)
    horizons = runner.spec.prediction_horizons_ms
    horizon_count = len(horizons)
    all_samples = load_samples(samples_path)
    partitions = partition_samples(all_samples, split_config)
    samples = partitions[args.split]
    for sample in samples:
        sample_horizon_fields(sample, horizon_count)

    cache_root = frame_cache_root_from_config(raw, ROOT)
    dataset = TemporalVideoDataset(
        samples,
        project_root=ROOT,
        preprocess_config=runner.spec.preprocess_config,
        cache_root=cache_root,
        require_cache=args.require_cache,
    )
    num_workers = loop_config.num_workers if args.num_workers is None else args.num_workers
    loader_kwargs = {
        "batch_size": loop_config.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": runner.device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    probability_batches: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["input"].to(
                    device=runner.device,
                    dtype=torch.float32,
                    non_blocking=runner.device.type == "cuda",
                )
                probabilities = torch.sigmoid(runner.model(inputs))
                if probabilities.ndim == 1:
                    probabilities = probabilities.unsqueeze(1)
                probability_batches.append(probabilities.detach().cpu().numpy())
    finally:
        dataset.close()

    probabilities = np.concatenate(probability_batches, axis=0)
    if probabilities.shape != (len(samples), horizon_count):
        raise RuntimeError(
            f"prediction shape mismatch: {probabilities.shape} != {(len(samples), horizon_count)}"
        )

    videos = getattr(split_config, args.split)
    label_data = {
        video: load_label_data(labels_dir, video)
        for video in videos
    }

    print(
        f"Device: {runner.device}; split={args.split}; samples={len(samples)}; "
        f"horizons={horizons}; threshold={args.threshold:.3f}; "
        f"tolerance={tolerance_ms:.1f}ms",
        flush=True,
    )

    results: dict[str, object] = {}
    for horizon_index, horizon_ms in enumerate(horizons):
        targets = np.asarray(
            [sample_horizon_fields(sample, horizon_count)[0][horizon_index] for sample in samples],
            dtype=np.float32,
        )
        transition_mask = np.asarray(
            [sample_horizon_fields(sample, horizon_count)[2][horizon_index] for sample in samples],
            dtype=bool,
        )
        short_mask = np.asarray(
            [sample_horizon_fields(sample, horizon_count)[3][horizon_index] for sample in samples],
            dtype=bool,
        )
        head_probabilities = probabilities[:, horizon_index]
        all_mask = np.ones(len(samples), dtype=bool)

        current_states = np.asarray(
            [
                action_at_timestamp(label_data[sample.video][0], sample.input_timestamps_ms[-1])
                for sample in samples
            ],
            dtype=np.float32,
        )

        model_metrics = {
            "all": metric_dict(head_probabilities, targets, all_mask, args.threshold),
            "near_transition": metric_dict(
                head_probabilities, targets, transition_mask, args.threshold
            ),
            "near_short_correction": metric_dict(
                head_probabilities, targets, short_mask, args.threshold
            ),
            "sequence": sequence_summary(
                samples,
                probabilities,
                horizon_index,
                label_data=label_data,
                threshold=args.threshold,
                tolerance_ms=tolerance_ms,
                persistence=False,
            ),
        }
        persistence_probabilities = current_states
        persistence_metrics = {
            "all": metric_dict(
                persistence_probabilities, targets, all_mask, args.threshold
            ),
            "near_transition": metric_dict(
                persistence_probabilities, targets, transition_mask, args.threshold
            ),
            "near_short_correction": metric_dict(
                persistence_probabilities, targets, short_mask, args.threshold
            ),
            "sequence": sequence_summary(
                samples,
                probabilities,
                horizon_index,
                label_data=label_data,
                threshold=args.threshold,
                tolerance_ms=tolerance_ms,
                persistence=True,
            ),
        }

        key = f"{horizon_ms:g}ms"
        results[key] = {
            "horizon_ms": horizon_ms,
            "model": model_metrics,
            "persistence": persistence_metrics,
        }

        model_sequence = model_metrics["sequence"]
        persistence_sequence = persistence_metrics["sequence"]
        model_transition = model_sequence["transition"]["all"]
        persistence_transition = persistence_sequence["transition"]["all"]
        model_short = model_sequence["release_segment_recall"]["short_100_300ms"]
        persistence_short = persistence_sequence["release_segment_recall"]["short_100_300ms"]
        model_press = model_sequence["onset_timing_ms"]["press"]
        model_release = model_sequence["onset_timing_ms"]["release"]

        print(
            f"h{horizon_ms:g}/model: "
            f"f1={model_metrics['all']['f1']:.3f}, "
            f"near_transition_f1={model_metrics['near_transition']['f1']:.3f}, "
            f"near_short_f1={model_metrics['near_short_correction']['f1']:.3f}, "
            f"seq_transition_f1={model_transition['f1']:.3f}, "
            f"press_mean={model_press['mean_error_ms']:.1f}ms, "
            f"release_mean={model_release['mean_error_ms']:.1f}ms, "
            f"short_recall={model_short['recall']:.3f}",
            flush=True,
        )
        print(
            f"h{horizon_ms:g}/persistence: "
            f"acc={persistence_metrics['all']['accuracy']:.3f}, "
            f"f1={persistence_metrics['all']['f1']:.3f}, "
            f"near_transition_f1={persistence_metrics['near_transition']['f1']:.3f}, "
            f"near_short_f1={persistence_metrics['near_short_correction']['f1']:.3f}, "
            f"seq_transition_f1={persistence_transition['f1']:.3f}, "
            f"short_recall={persistence_short['recall']:.3f}",
            flush=True,
        )

    payload = {
        "split": args.split,
        "model": str(model_path),
        "samples": str(samples_path),
        "labels_dir": str(labels_dir),
        "threshold": args.threshold,
        "transition_tolerance_ms": tolerance_ms,
        "prediction_horizons_ms": list(horizons),
        "control_horizon_ms": runner.spec.control_horizon_ms,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Evaluation: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
