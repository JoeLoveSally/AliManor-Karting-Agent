#!/usr/bin/env python3
"""Evaluate v4-A static switch quality and stateful offline replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.model.sequential_state_conditioned import (  # noqa: E402
    build_sequential_state_conditioned_model,
)
from karting_agent.train.evaluator import BinaryMetricAccumulator  # noqa: E402
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
    SequentialStateConditionedVideoDataset,
    load_v3_samples,
)
from karting_agent.train.trainer import (  # noqa: E402
    load_train_loop_config,
    load_video_split,
    partition_samples,
)
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


@dataclass(frozen=True)
class StatefulDecision:
    video: str
    observation_timestamp_ms: float
    target_timestamp_ms: float
    probability: float
    state_before: bool
    switched: bool
    state_after: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v4-A sequential policy.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "train_v4a.yaml"
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "processed" / "v3" / "samples.jsonl",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "v3" / "labels",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.5, 0.6, 0.7, 0.8, 0.9)
    )
    parser.add_argument("--tolerance-ms", type=float, default=None)
    return parser.parse_args()


def load_yaml_mapping(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def select_device(torch, requested: str | None):
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def artifact_dir_from_config(raw: dict[str, object]) -> Path:
    artifact = raw.get("artifact", {})
    if not isinstance(artifact, dict):
        raise ValueError("artifact config must be a mapping")
    return ROOT / "artifacts" / "models" / str(
        artifact.get("name", "mobilenet_v3_small_v4a")
    )


def evaluation_tolerance(raw: dict[str, object], override: float | None) -> float:
    evaluation = raw.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation config must be a mapping")
    value = (
        float(evaluation.get("transition_tolerance_ms", 100.0))
        if override is None
        else float(override)
    )
    if value < 0:
        raise ValueError("transition tolerance must be >= 0")
    return value


def load_label_data(
    labels_dir: Path, video: str
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


def binary_metrics(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> dict[str, float | int]:
    metric = BinaryMetricAccumulator(threshold)
    metric.update(probabilities, targets)
    return metric.result().to_dict()


def false_positive_rate(
    probabilities: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> float:
    selected = np.asarray(mask, dtype=bool) & (np.asarray(targets) < 0.5)
    count = int(selected.sum())
    if count == 0:
        return 0.0
    return float((np.asarray(probabilities)[selected] >= threshold).sum() / count)


def simulate_video(
    video: str,
    samples,
    sample_indices: list[int],
    switch_if_release: np.ndarray,
    switch_if_press: np.ndarray,
    *,
    horizon_index: int,
    horizon_ms: float,
    threshold: float,
) -> list[StatefulDecision]:
    ordered = sorted(
        sample_indices,
        key=lambda index: float(samples[index].input_timestamps_ms[-1]),
    )
    if not ordered:
        return []
    initial = samples[ordered[0]].current_pressed
    if initial is None:
        raise ValueError("sample is missing current_pressed")
    state = bool(initial)
    decisions: list[StatefulDecision] = []
    for index in ordered:
        sample = samples[index]
        observation_ms = float(sample.input_timestamps_ms[-1])
        probabilities = switch_if_press if state else switch_if_release
        probability = float(probabilities[index, horizon_index])
        before = state
        switched = probability >= threshold
        if switched:
            state = not state
        decisions.append(
            StatefulDecision(
                video=video,
                observation_timestamp_ms=observation_ms,
                target_timestamp_ms=observation_ms + horizon_ms,
                probability=probability,
                state_before=before,
                switched=switched,
                state_after=state,
            )
        )
    return decisions


def points_from_decisions(
    decisions: list[StatefulDecision], *, target_timeline: bool
) -> list[SequencePoint]:
    if not decisions:
        return []
    first = decisions[0]
    first_timestamp = (
        first.target_timestamp_ms if target_timeline else first.observation_timestamp_ms
    )
    points = [
        SequencePoint(
            video=first.video,
            timestamp_ms=first_timestamp - 1e-3,
            probability=1.0 if first.state_before else 0.0,
        )
    ]
    for decision in decisions:
        timestamp = (
            decision.target_timestamp_ms
            if target_timeline
            else decision.observation_timestamp_ms
        )
        points.append(
            SequencePoint(
                video=decision.video,
                timestamp_ms=timestamp,
                probability=1.0 if decision.state_after else 0.0,
            )
        )
    return points


def chatter_summary(decisions: list[StatefulDecision]) -> dict[str, int]:
    timestamps = [
        decision.observation_timestamp_ms for decision in decisions if decision.switched
    ]
    intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
    return {
        "transitions": len(timestamps),
        "lt_100ms": int((intervals < 100.0).sum()) if intervals.size else 0,
        "lt_200ms": int((intervals < 200.0).sum()) if intervals.size else 0,
    }


def stateful_sequence_summary(
    samples,
    switch_if_release: np.ndarray,
    switch_if_press: np.ndarray,
    *,
    horizon_index: int,
    horizon_ms: float,
    threshold: float,
    tolerance_ms: float,
    label_data: dict[str, tuple[list[Transition], list[ReleaseSegment]]],
) -> dict[str, object]:
    indices_by_video: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_video[sample.video].append(index)

    target_evaluations = []
    observation_evaluations = []
    chatter = {"transitions": 0, "lt_100ms": 0, "lt_200ms": 0}
    for video, indices in indices_by_video.items():
        decisions = simulate_video(
            video,
            samples,
            indices,
            switch_if_release,
            switch_if_press,
            horizon_index=horizon_index,
            horizon_ms=horizon_ms,
            threshold=threshold,
        )
        transitions, releases = label_data[video]
        target_evaluations.append(
            evaluate_sequence(
                points_from_decisions(decisions, target_timeline=True),
                transitions,
                releases,
                threshold=0.5,
                tolerance_ms=tolerance_ms,
            )
        )
        observation_evaluations.append(
            evaluate_sequence(
                points_from_decisions(decisions, target_timeline=False),
                transitions,
                releases,
                threshold=0.5,
                tolerance_ms=tolerance_ms,
            )
        )
        local = chatter_summary(decisions)
        for key in chatter:
            chatter[key] += local[key]

    return {
        "target_timeline": combine_evaluations(target_evaluations).summary(),
        "observation_timeline": combine_evaluations(observation_evaluations).summary(),
        "chatter": chatter,
    }


def main() -> int:
    args = parse_args()
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    thresholds = tuple(float(value) for value in args.thresholds)
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("all thresholds must be in (0, 1)")

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: python -m pip install -e ".[train]"'
        ) from exc

    config_path = args.config.resolve()
    raw = load_yaml_mapping(config_path)
    split_config = load_video_split(config_path)
    loop_config = load_train_loop_config(config_path)
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)
    tolerance_ms = evaluation_tolerance(raw, args.tolerance_ms)
    artifact_dir = artifact_dir_from_config(raw)
    model_path = args.model.resolve() if args.model else artifact_dir / "model.pt"
    metadata_path = (
        args.metadata.resolve() if args.metadata else artifact_dir / "metadata.json"
    )
    output_path = (
        args.output.resolve()
        if args.output
        else artifact_dir / "evaluation" / f"{args.split}_v4a.json"
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    horizons = tuple(float(value) for value in metadata["prediction_horizons_ms"])
    control_horizon = float(metadata["control_horizon_ms"])
    control_index = horizons.index(control_horizon)
    frame_stack = int(metadata["frame_stack"])
    model = build_sequential_state_conditioned_model(
        str(metadata["architecture"]),
        pretrained=False,
        frame_stack=frame_stack,
        horizon_count=len(horizons),
        visual_feature_dim=int(metadata["visual_feature_dim"]),
        gru_hidden_dim=int(metadata["gru_hidden_dim"]),
        gru_layers=int(metadata["gru_layers"]),
        state_embedding_dim=int(metadata["state_embedding_dim"]),
        policy_hidden_dim=int(metadata["policy_hidden_dim"]),
    )
    device = select_device(torch, args.device)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    all_samples = load_v3_samples(args.samples.resolve())
    partitions = partition_samples(all_samples, split_config)
    samples = partitions[args.split]
    dataset = SequentialStateConditionedVideoDataset(
        samples,
        frame_stack=frame_stack,
        project_root=ROOT,
        preprocess_config=preprocess_config,
        cache_root=cache_root,
        require_cache=args.require_cache,
        counterfactual_states=False,
    )
    num_workers = loop_config.num_workers if args.num_workers is None else args.num_workers
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

    release_batches: list[np.ndarray] = []
    press_batches: list[np.ndarray] = []
    future_batches: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["input"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                batch_size = int(inputs.shape[0])
                release_states = torch.zeros(batch_size, device=device, dtype=torch.long)
                press_states = torch.ones(batch_size, device=device, dtype=torch.long)
                release_logits, future_logits = model(inputs, release_states)
                press_logits, _ = model(inputs, press_states)
                release_batches.append(torch.sigmoid(release_logits).cpu().numpy())
                press_batches.append(torch.sigmoid(press_logits).cpu().numpy())
                future_batches.append(torch.sigmoid(future_logits).cpu().numpy())
    finally:
        dataset.close()

    switch_if_release = np.concatenate(release_batches, axis=0)
    switch_if_press = np.concatenate(press_batches, axis=0)
    future_probabilities = np.concatenate(future_batches, axis=0)
    expert_states = np.asarray(
        [bool(sample.current_pressed) for sample in samples], dtype=bool
    )
    future_targets = np.asarray(
        [sample.target_states for sample in samples], dtype=np.float32
    )
    switch_targets = np.not_equal(
        future_targets >= 0.5, expert_states[:, None]
    ).astype(np.float32)
    expert_switch_probabilities = np.where(
        expert_states[:, None], switch_if_press, switch_if_release
    )
    near_transition = np.asarray(
        [
            sample.near_transition_by_horizon
            or tuple(sample.near_transition for _ in horizons)
            for sample in samples
        ],
        dtype=bool,
    )

    videos = getattr(split_config, args.split)
    label_data = {
        video: load_label_data(args.labels_dir.resolve(), video) for video in videos
    }
    auxiliary = {
        f"{horizon:g}ms": binary_metrics(
            future_probabilities[:, index], future_targets[:, index], 0.5
        )
        for index, horizon in enumerate(horizons)
    }
    print(
        f"Device: {device}; split={args.split}; samples={len(samples)}; "
        f"horizons={horizons}; control_horizon={control_horizon:g}ms; "
        f"tolerance={tolerance_ms:.1f}ms",
        flush=True,
    )
    print(
        "Aux future-action F1: "
        + ", ".join(
            f"h{horizon:g}={auxiliary[f'{horizon:g}ms']['f1']:.3f}"
            for horizon in horizons
        ),
        flush=True,
    )

    threshold_results: dict[str, object] = {}
    primary_targets = switch_targets[:, control_index]
    primary_probabilities = expert_switch_probabilities[:, control_index]
    stable_mask = ~near_transition[:, control_index]
    for threshold in thresholds:
        static = binary_metrics(primary_probabilities, primary_targets, threshold)
        stable_fpr = false_positive_rate(
            primary_probabilities, primary_targets, stable_mask, threshold
        )
        all_negative_fpr = false_positive_rate(
            primary_probabilities,
            primary_targets,
            np.ones(len(samples), dtype=bool),
            threshold,
        )
        stateful = stateful_sequence_summary(
            samples,
            switch_if_release,
            switch_if_press,
            horizon_index=control_index,
            horizon_ms=control_horizon,
            threshold=threshold,
            tolerance_ms=tolerance_ms,
            label_data=label_data,
        )
        target_transition = stateful["target_timeline"]["transition"]["all"]
        observation_transition = stateful["observation_timeline"]["transition"]["all"]
        target_short = stateful["target_timeline"]["release_segment_recall"][
            "short_100_300ms"
        ]
        chatter = stateful["chatter"]
        print(
            f"threshold={threshold:.2f}/static: p={static['precision']:.3f}, "
            f"r={static['recall']:.3f}, f1={static['f1']:.3f}, "
            f"pred_pos={static['predicted_positive_rate']:.3f}, "
            f"stable_fpr={stable_fpr:.3f}, all_negative_fpr={all_negative_fpr:.3f}",
            flush=True,
        )
        print(
            f"threshold={threshold:.2f}/stateful-target: "
            f"transition_f1={target_transition['f1']:.3f}, "
            f"matched={target_transition['matched']}/{target_transition['ground_truth']}, "
            f"predicted={target_transition['predicted']}, "
            f"short_recall={target_short['recall']:.3f}, "
            f"chatter_lt100={chatter['lt_100ms']}, chatter_lt200={chatter['lt_200ms']}",
            flush=True,
        )
        print(
            f"threshold={threshold:.2f}/stateful-observation: "
            f"transition_f1={observation_transition['f1']:.3f}, "
            f"matched={observation_transition['matched']}/{observation_transition['ground_truth']}, "
            f"predicted={observation_transition['predicted']}",
            flush=True,
        )
        threshold_results[f"{threshold:.3f}"] = {
            "static": static,
            "stable_false_positive_rate": stable_fpr,
            "all_negative_false_positive_rate": all_negative_fpr,
            "stateful": stateful,
        }

    payload = {
        "split": args.split,
        "model": str(model_path),
        "metadata": str(metadata_path),
        "samples": str(args.samples.resolve()),
        "labels_dir": str(args.labels_dir.resolve()),
        "prediction_horizons_ms": list(horizons),
        "control_horizon_ms": control_horizon,
        "transition_tolerance_ms": tolerance_ms,
        "auxiliary_future_action": auxiliary,
        "thresholds": threshold_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Evaluation: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
