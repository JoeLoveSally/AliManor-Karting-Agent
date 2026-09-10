#!/usr/bin/env python3
"""Evaluate a trained model as an ordered control sequence."""

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

from karting_agent.model.base import build_model
from karting_agent.train.frame_cache import frame_cache_root_from_config
from karting_agent.train.sequence_evaluator import (
    ReleaseSegment,
    SequencePoint,
    Transition,
    combine_evaluations,
    evaluate_sequence,
)
from karting_agent.train.trainer import (
    load_samples,
    load_train_loop_config,
    load_video_split,
    partition_samples,
)
from karting_agent.train.video_dataset import TemporalVideoDataset
from karting_agent.vision.preprocess import preprocess_config_from_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequence-level model evaluation.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Evaluate validation by default; test must be selected explicitly.",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tolerance-ms", type=float, default=None)
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Fail instead of falling back to MP4 when frame cache is missing.",
    )
    return parser.parse_args()


def load_raw_config(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("train config root must be a mapping")
    return raw


def select_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluation_parameters(
    raw: dict[str, object],
    *,
    threshold_override: float | None,
    tolerance_override: float | None,
) -> tuple[float, float]:
    config = raw.get("evaluation", {})
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be a mapping")
    threshold = (
        float(config.get("threshold", 0.5))
        if threshold_override is None
        else threshold_override
    )
    tolerance_ms = (
        float(config.get("transition_tolerance_ms", 100.0))
        if tolerance_override is None
        else tolerance_override
    )
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0, 1)")
    if tolerance_ms < 0:
        raise ValueError("tolerance must be >= 0")
    return threshold, tolerance_ms


def artifact_paths(
    raw: dict[str, object],
    split_name: str,
    model_override: Path | None,
    output_override: Path | None,
) -> tuple[Path, Path]:
    artifact = raw.get("artifact", {})
    model = raw.get("model", {})
    if not isinstance(artifact, dict) or not isinstance(model, dict):
        raise ValueError("model/artifact config must be mappings")
    architecture = str(model.get("architecture", "mobilenet_v3_small"))
    artifact_name = str(artifact.get("name", f"{architecture}_{split_name}"))
    artifact_dir = ROOT / "artifacts" / "models" / artifact_name
    model_path = (
        model_override.resolve()
        if model_override is not None
        else artifact_dir / "model.pt"
    )
    output_path = (
        output_override.resolve()
        if output_override is not None
        else artifact_dir / "evaluation" / f"{split_name}_sequence.json"
    )
    return model_path, output_path


def load_label_data(video: str) -> tuple[list[Transition], list[ReleaseSegment]]:
    path = ROOT / "data" / "processed" / "labels" / f"{Path(video).stem}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"label file has no events: {path}")

    events: list[tuple[float, bool]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            raise ValueError(f"invalid event in {path}")
        events.append((float(raw_event["timestamp_ms"]), bool(raw_event["pressed"])))

    transitions = [
        Transition(video=video, timestamp_ms=timestamp_ms, pressed=pressed)
        for timestamp_ms, pressed in events[1:]
    ]
    releases = [
        ReleaseSegment(video=video, start_ms=start_ms, end_ms=end_ms)
        for (start_ms, pressed), (end_ms, next_pressed) in zip(events, events[1:])
        if not pressed and next_pressed
    ]
    return transitions, releases


def load_state_dict(torch, path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> int:
    args = parse_args()
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: python -m pip install -e ".[train]"'
        ) from exc

    raw = load_raw_config(args.config)
    loop_config = load_train_loop_config(args.config)
    split_config = load_video_split(args.config)
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)
    threshold, tolerance_ms = evaluation_parameters(
        raw,
        threshold_override=args.threshold,
        tolerance_override=args.tolerance_ms,
    )
    model_path, output_path = artifact_paths(
        raw,
        split_config.name,
        args.model,
        args.output,
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"model artifact not found: {model_path}")

    all_samples = load_samples(ROOT / "data" / "processed" / "samples.jsonl")
    partitions = partition_samples(all_samples, split_config)
    samples = partitions[args.split]
    num_workers = loop_config.num_workers if args.num_workers is None else args.num_workers

    dataset = TemporalVideoDataset(
        samples,
        project_root=ROOT,
        preprocess_config=preprocess_config,
        cache_root=cache_root,
        require_cache=args.require_cache,
    )
    device = select_device(torch)
    loader_kwargs = {
        "batch_size": loop_config.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    model_config = raw.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a mapping")
    model = build_model(
        str(model_config.get("architecture", "mobilenet_v3_small")),
        frame_stack=int(model_config.get("frame_stack", 3)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(load_state_dict(torch, model_path, device))
    model.eval()

    print(
        f"Device: {device}; split={args.split}; samples={len(samples)}; "
        f"threshold={threshold:.3f}; tolerance={tolerance_ms:.1f}ms",
        flush=True,
    )
    print(
        "Frame cache: "
        + (
            f"{cache_root} (required={args.require_cache})"
            if cache_root is not None
            else "disabled"
        ),
        flush=True,
    )

    probabilities: list[float] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["input"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                probabilities.extend(
                    torch.sigmoid(model(inputs)).detach().cpu().tolist()
                )
    finally:
        dataset.close()

    if len(probabilities) != len(samples):
        raise RuntimeError(
            f"prediction count mismatch: {len(probabilities)} != {len(samples)}"
        )

    points_by_video: dict[str, list[SequencePoint]] = defaultdict(list)
    for sample, probability in zip(samples, probabilities):
        points_by_video[sample.video].append(
            SequencePoint(
                video=sample.video,
                timestamp_ms=sample.target_timestamp_ms,
                probability=float(probability),
            )
        )

    evaluations = []
    per_video: dict[str, object] = {}
    configured_videos = getattr(split_config, args.split)
    for video in configured_videos:
        points = points_by_video.get(video)
        if not points:
            raise RuntimeError(f"no predictions for configured video: {video}")
        transitions, releases = load_label_data(video)
        evaluation = evaluate_sequence(
            points,
            transitions,
            releases,
            threshold=threshold,
            tolerance_ms=tolerance_ms,
        )
        evaluations.append(evaluation)
        summary = evaluation.summary()
        per_video[video] = summary
        transition = summary["transition"]["all"]
        short = summary["release_segment_recall"]["short_100_300ms"]
        print(
            f"{Path(video).name}: transition_f1={transition['f1']:.3f}, "
            f"matched={transition['matched']}/{transition['ground_truth']}, "
            f"short_recall={short['recall']:.3f} "
            f"({short['detected']}/{short['segments']})",
            flush=True,
        )

    aggregate = combine_evaluations(evaluations).summary()
    transition = aggregate["transition"]["all"]
    press_timing = aggregate["onset_timing_ms"]["press"]
    release_timing = aggregate["onset_timing_ms"]["release"]
    short = aggregate["release_segment_recall"]["short_100_300ms"]
    print(
        "aggregate: "
        f"transition_f1={transition['f1']:.3f}, "
        f"precision={transition['precision']:.3f}, "
        f"recall={transition['recall']:.3f}, "
        f"press_mae={press_timing['mae_ms']:.1f}ms, "
        f"release_mae={release_timing['mae_ms']:.1f}ms, "
        f"short_recall={short['recall']:.3f} "
        f"({short['detected']}/{short['segments']})",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "model": str(model_path),
        "aggregate": aggregate,
        "per_video": per_video,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Evaluation: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
