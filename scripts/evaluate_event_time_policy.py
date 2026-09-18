#!/usr/bin/env python3
"""Replay V4-C4 current-action and event-time policy on held-out expert visuals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.model.event_time import build_event_time_model  # noqa: E402
from karting_agent.model.temporal_delta import (  # noqa: E402
    transform_temporal_input_torch,
    validate_temporal_input_representation,
)
from karting_agent.runtime.event_time_policy import (  # noqa: E402
    EventTimePolicyConfig,
    EventTimePolicyDecoder,
)
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.labels.touch_marker import ActionEvent  # noqa: E402
from karting_agent.train.sequence_evaluator import (  # noqa: E402
    ReleaseSegment,
    SequencePoint,
    Transition,
    combine_evaluations,
    evaluate_sequence,
)
from karting_agent.train.state_conditioned_dataset import (  # noqa: E402
    StateConditionedVideoDataset,
    load_v3_samples,
)
from karting_agent.train.trainer import (  # noqa: E402
    load_train_loop_config,
    load_video_split,
    partition_samples,
)
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a V4-C4 event-time policy on a held-out split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_v4c4_event_time.yaml",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "v4c3_temporal_delta_dense"
            / "samples.jsonl"
        ),
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "v4c3_temporal_delta_dense"
            / "labels"
        ),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--action-threshold", type=float, default=0.5)
    parser.add_argument("--min-state-hold-ms", type=float, default=100.0)
    parser.add_argument("--tolerance-ms", type=float, default=100.0)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    return raw


def _select_device(torch, requested: str | None):
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _artifact_dir(raw: dict[str, object]) -> Path:
    artifact = raw.get("artifact", {})
    if not isinstance(artifact, dict):
        raise ValueError("artifact config must be a mapping")
    return ROOT / "artifacts" / "models" / str(artifact["name"])


def _load_label_data(
    labels_dir: Path,
    video: str,
) -> tuple[list[Transition], list[ReleaseSegment]]:
    path = labels_dir / f"{Path(video).stem}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"label file has no events: {path}")
    events = [
        ActionEvent(
            frame_index=int(raw["frame_index"]),
            timestamp_ms=float(raw["timestamp_ms"]),
            pressed=bool(raw["pressed"]),
        )
        for raw in raw_events
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
    return transitions, releases


def _points_and_reasons(
    *,
    video: str,
    samples,
    indices: list[int],
    action_probabilities: np.ndarray,
    event_classes: np.ndarray,
    decoder_config: EventTimePolicyConfig,
    use_event_time: bool,
) -> tuple[list[SequencePoint], Counter[str]]:
    ordered = sorted(
        indices,
        key=lambda index: float(samples[index].input_timestamps_ms[-1]),
    )
    if not ordered:
        return [], Counter()

    initial = samples[ordered[0]].current_pressed
    if initial is None:
        raise ValueError("sample is missing current_pressed")
    state = bool(initial)
    first_observation_ms = float(samples[ordered[0]].input_timestamps_ms[-1])
    points = [
        SequencePoint(
            video=video,
            timestamp_ms=first_observation_ms - 1e-3,
            probability=1.0 if state else 0.0,
        )
    ]
    reasons: Counter[str] = Counter()
    decoder = EventTimePolicyDecoder(decoder_config)

    for index in ordered:
        observation_ms = float(samples[index].input_timestamps_ms[-1])

        pending = decoder.execute_pending_if_due(
            timestamp_ms=observation_ms,
            current_pressed=state,
        )
        if pending is not None:
            state = pending.state_after
            points.append(
                SequencePoint(
                    video=video,
                    timestamp_ms=pending.due_at_ms,
                    probability=1.0 if state else 0.0,
                )
            )
            reasons["pending_execute"] += 1

        event_class = (
            int(event_classes[index])
            if use_event_time
            else decoder_config.no_event_class
        )
        decision = decoder.update(
            timestamp_ms=observation_ms,
            current_pressed=state,
            current_action_probability=float(action_probabilities[index]),
            event_class=event_class,
        )
        reasons[decision.reason] += 1
        if decision.switch:
            state = decision.state_after
            points.append(
                SequencePoint(
                    video=video,
                    timestamp_ms=observation_ms,
                    probability=1.0 if state else 0.0,
                )
            )

    last_observation_ms = float(samples[ordered[-1]].input_timestamps_ms[-1])
    if points[-1].timestamp_ms < last_observation_ms - 1e-6:
        points.append(
            SequencePoint(
                video=video,
                timestamp_ms=last_observation_ms,
                probability=1.0 if state else 0.0,
            )
        )
    return points, reasons


def _chatter(points_by_video: dict[str, list[SequencePoint]]) -> dict[str, int]:
    transitions = 0
    lt_100 = 0
    lt_200 = 0
    for points in points_by_video.values():
        states = [point.probability >= 0.5 for point in points]
        timestamps = np.asarray(
            [
                points[index].timestamp_ms
                for index in range(1, len(points))
                if states[index] != states[index - 1]
            ],
            dtype=np.float64,
        )
        transitions += int(timestamps.size)
        intervals = np.diff(timestamps)
        lt_100 += int((intervals < 100.0).sum()) if intervals.size else 0
        lt_200 += int((intervals < 200.0).sum()) if intervals.size else 0
    return {
        "transitions": transitions,
        "lt_100ms": lt_100,
        "lt_200ms": lt_200,
    }


def _summary(
    points_by_video: dict[str, list[SequencePoint]],
    label_data: dict[str, tuple[list[Transition], list[ReleaseSegment]]],
    *,
    tolerance_ms: float,
) -> dict[str, object]:
    evaluations = []
    for video, points in points_by_video.items():
        transitions, releases = label_data[video]
        evaluations.append(
            evaluate_sequence(
                points,
                transitions,
                releases,
                threshold=0.5,
                tolerance_ms=tolerance_ms,
            )
        )
    combined = combine_evaluations(evaluations)
    return {
        "sequence": combined.summary(),
        "chatter": _chatter(points_by_video),
    }


def _print_summary(name: str, summary: dict[str, object]) -> None:
    sequence = summary["sequence"]
    chatter = summary["chatter"]
    transition = sequence["transition"]["all"]
    short = sequence["release_segment_recall"]["short_100_300ms"]
    press = sequence["onset_timing_ms"]["press"]
    release = sequence["onset_timing_ms"]["release"]
    print(
        f"{name}: f1={transition['f1']:.3f} "
        f"matched={transition['matched']}/{transition['ground_truth']} "
        f"pred={transition['predicted']} "
        f"short_recall={short['recall']:.3f} "
        f"press_mae={press['mae_ms']:.1f}ms "
        f"press_bias={press['mean_error_ms']:+.1f}ms "
        f"release_mae={release['mae_ms']:.1f}ms "
        f"release_bias={release['mean_error_ms']:+.1f}ms "
        f"chatter_lt100={chatter['lt_100ms']} "
        f"lt200={chatter['lt_200ms']}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if not 0.0 < args.action_threshold < 1.0:
        raise ValueError("--action-threshold must be in (0, 1)")
    if args.min_state_hold_ms < 0:
        raise ValueError("--min-state-hold-ms must be >= 0")
    if args.tolerance_ms < 0:
        raise ValueError("--tolerance-ms must be >= 0")

    import torch
    from torch.utils.data import DataLoader

    config_path = args.config.resolve()
    raw = _load_yaml(config_path)
    split_config = load_video_split(config_path)
    loop = load_train_loop_config(config_path)
    preprocess = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)
    artifact = _artifact_dir(raw)
    model_path = args.model.resolve() if args.model else artifact / "model.pt"
    metadata_path = args.metadata.resolve() if args.metadata else artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_family") != "event_time_v4c4":
        raise ValueError("metadata is not a V4-C4 event-time artifact")

    frame_stack = int(metadata["frame_stack"])
    input_representation = validate_temporal_input_representation(
        str(metadata["input_representation"])
    )
    bin_ms = float(metadata["event_time_bin_ms"])
    max_horizon_ms = float(metadata["event_time_max_horizon_ms"])
    event_time_classes = int(metadata["event_time_classes"])
    no_event_class = int(metadata["no_event_class"])
    event_bins = event_time_classes - 1
    if no_event_class != event_bins:
        raise ValueError("V4-C4 metadata has inconsistent event classes")

    model = build_event_time_model(
        str(metadata["architecture"]),
        frame_stack=frame_stack,
        pretrained=False,
        event_time_classes=event_time_classes,
        visual_feature_dim=int(metadata["visual_feature_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
    )
    device = _select_device(torch, args.device)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    all_samples = load_v3_samples(args.samples.resolve())
    samples = partition_samples(all_samples, split_config)[args.split]
    dataset = StateConditionedVideoDataset(
        samples,
        project_root=ROOT,
        preprocess_config=preprocess,
        cache_root=cache_root,
        require_cache=args.require_cache,
        counterfactual_states=False,
    )
    num_workers = loop.num_workers if args.num_workers is None else args.num_workers
    loader = DataLoader(
        dataset,
        batch_size=loop.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    action_batches: list[np.ndarray] = []
    event_batches: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["input"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                inputs = transform_temporal_input_torch(
                    inputs,
                    frame_stack=frame_stack,
                    representation=input_representation,
                )
                action_logit, event_logits, _, _, _ = model(inputs)
                action_batches.append(
                    torch.sigmoid(action_logit).cpu().numpy()
                )
                event_batches.append(
                    event_logits.argmax(dim=1).cpu().numpy()
                )
    finally:
        dataset.close()

    action_probabilities = np.concatenate(action_batches, axis=0)
    event_classes = np.concatenate(event_batches, axis=0)
    if action_probabilities.shape != (len(samples),):
        raise RuntimeError("current-action prediction shape mismatch")
    if event_classes.shape != (len(samples),):
        raise RuntimeError("event-time prediction shape mismatch")

    decoder_config = EventTimePolicyConfig(
        bin_ms=bin_ms,
        event_bins=event_bins,
        no_event_class=no_event_class,
        action_threshold=float(args.action_threshold),
        min_state_hold_ms=float(args.min_state_hold_ms),
    )

    indices_by_video: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_video[sample.video].append(index)
    labels_dir = args.labels_dir.resolve()
    label_data = {
        video: _load_label_data(labels_dir, video)
        for video in indices_by_video
    }

    action_only_points: dict[str, list[SequencePoint]] = {}
    event_time_points: dict[str, list[SequencePoint]] = {}
    action_only_reasons: Counter[str] = Counter()
    event_time_reasons: Counter[str] = Counter()
    for video, indices in indices_by_video.items():
        points, reasons = _points_and_reasons(
            video=video,
            samples=samples,
            indices=indices,
            action_probabilities=action_probabilities,
            event_classes=event_classes,
            decoder_config=decoder_config,
            use_event_time=False,
        )
        action_only_points[video] = points
        action_only_reasons.update(reasons)

        points, reasons = _points_and_reasons(
            video=video,
            samples=samples,
            indices=indices,
            action_probabilities=action_probabilities,
            event_classes=event_classes,
            decoder_config=decoder_config,
            use_event_time=True,
        )
        event_time_points[video] = points
        event_time_reasons.update(reasons)

    action_only_summary = _summary(
        action_only_points,
        label_data,
        tolerance_ms=float(args.tolerance_ms),
    )
    event_time_summary = _summary(
        event_time_points,
        label_data,
        tolerance_ms=float(args.tolerance_ms),
    )

    print(
        f"split={args.split} samples={len(samples)} device={device} "
        f"action_threshold={args.action_threshold:.2f} "
        f"min_hold={args.min_state_hold_ms:g}ms "
        f"bin={bin_ms:g}ms max_horizon={max_horizon_ms:g}ms",
        flush=True,
    )
    _print_summary("current_action_only", action_only_summary)
    _print_summary("event_time", event_time_summary)
    print(
        "current-action reasons: "
        + ", ".join(
            f"{key}={value}" for key, value in sorted(action_only_reasons.items())
        ),
        flush=True,
    )
    print(
        "event-time reasons: "
        + ", ".join(
            f"{key}={value}" for key, value in sorted(event_time_reasons.items())
        ),
        flush=True,
    )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else artifact / "evaluation" / f"{args.split}_event_time_policy.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "samples": len(samples),
                "model": str(model_path),
                "metadata": str(metadata_path),
                "decoder_config": {
                    "bin_ms": bin_ms,
                    "max_horizon_ms": max_horizon_ms,
                    "event_bins": event_bins,
                    "no_event_class": no_event_class,
                    "action_threshold": args.action_threshold,
                    "min_state_hold_ms": args.min_state_hold_ms,
                },
                "current_action_only": action_only_summary,
                "event_time": event_time_summary,
                "current_action_reasons": dict(action_only_reasons),
                "event_time_reasons": dict(event_time_reasons),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
