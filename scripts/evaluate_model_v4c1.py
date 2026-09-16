#!/usr/bin/env python3
"""Evaluate v4-C1 control and held-out road-axis prediction."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_model_v3 import (  # noqa: E402
    binary_metrics,
    chatter_summary,
    evaluation_tolerance,
    false_positive_rate,
    load_label_data,
    load_yaml_mapping,
    points_from_decisions,
    select_device,
    StatefulDecision,
)
from karting_agent.model.axis_supervised import build_axis_supervised_model  # noqa: E402
from karting_agent.train.axis_labels import (  # noqa: E402
    AxisSupervisedVideoDataset,
    load_axis_pseudo_labels,
)
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.sequence_evaluator import (  # noqa: E402
    combine_evaluations,
    evaluate_sequence,
)
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402
from karting_agent.train.trainer import (  # noqa: E402
    load_train_loop_config,
    load_video_split,
    partition_samples,
)
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v4-C1 axis-supervised policy.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "train_v4c1.yaml"
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
    parser.add_argument("--axis-labels", type=Path, default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--artifact-name",
        type=str,
        default=None,
        help="Artifact directory name under artifacts/models/.",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.5, 0.6, 0.7, 0.8, 0.9),
    )
    parser.add_argument("--tolerance-ms", type=float, default=None)
    return parser.parse_args()


def artifact_dir_from_config(raw: dict[str, object]) -> Path:
    artifact = raw.get("artifact", {})
    if not isinstance(artifact, dict):
        raise ValueError("artifact config must be a mapping")
    return ROOT / "artifacts" / "models" / str(
        artifact.get("name", "mobilenet_v3_small_v4c1")
    )


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
    decisions = []
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


def stateful_sequence_summary(
    samples,
    switch_if_release: np.ndarray,
    switch_if_press: np.ndarray,
    *,
    horizon_index: int,
    horizon_ms: float,
    threshold: float,
    tolerance_ms: float,
    label_data,
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


def axis_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    selected = weights > 0
    if not np.any(selected):
        return {
            "samples": 0,
            "weighted_mean_error_deg": 0.0,
            "mean_error_deg": 0.0,
        }
    pred = predictions[selected]
    target = targets[selected]
    pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    dots = np.clip(np.sum(pred * target, axis=1), -1.0, 1.0)
    errors = 0.5 * np.degrees(np.arccos(dots))
    selected_weights = weights[selected]
    return {
        "samples": int(selected.sum()),
        "mean_error_deg": float(errors.mean()),
        "weighted_mean_error_deg": float(
            np.sum(errors * selected_weights) / np.sum(selected_weights)
        ),
    }


def main() -> int:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader

    raw = load_yaml_mapping(args.config.resolve())
    split_config = load_video_split(args.config.resolve())
    loop_config = load_train_loop_config(args.config.resolve())
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)
    tolerance_ms = evaluation_tolerance(raw, args.tolerance_ms)
    artifact_dir = (
        ROOT / "artifacts" / "models" / args.artifact_name
        if args.artifact_name
        else artifact_dir_from_config(raw)
    )
    model_path = args.model.resolve() if args.model else artifact_dir / "model.pt"
    metadata_path = (
        args.metadata.resolve() if args.metadata else artifact_dir / "metadata.json"
    )
    output_path = (
        args.output.resolve()
        if args.output
        else artifact_dir / "evaluation" / f"{args.split}_v4c1.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_family") != "state_conditioned_axis_v4c1":
        raise ValueError("metadata is not a v4-C1 artifact")
    horizons = tuple(float(value) for value in metadata["prediction_horizons_ms"])
    control_horizon = float(metadata["control_horizon_ms"])
    control_index = horizons.index(control_horizon)
    axis_labels_path = (
        args.axis_labels.resolve()
        if args.axis_labels
        else Path(str(metadata["axis_labels"])).resolve()
    )
    axis_labels = load_axis_pseudo_labels(axis_labels_path)

    model = build_axis_supervised_model(
        str(metadata["architecture"]),
        frame_stack=int(metadata["frame_stack"]),
        pretrained=False,
        horizon_count=len(horizons),
        visual_feature_dim=int(metadata["visual_feature_dim"]),
        state_embedding_dim=int(metadata["state_embedding_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
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
    dataset = AxisSupervisedVideoDataset(
        samples,
        axis_labels=axis_labels,
        project_root=ROOT,
        preprocess_config=preprocess_config,
        cache_root=cache_root,
        require_cache=args.require_cache,
        counterfactual_states=False,
    )
    num_workers = loop_config.num_workers if args.num_workers is None else args.num_workers
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
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

    release_batches = []
    press_batches = []
    future_batches = []
    axis_batches = []
    axis_target_batches = []
    axis_weight_batches = []
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["input"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                batch_size = int(inputs.shape[0])
                release_states = torch.zeros(
                    batch_size, device=device, dtype=torch.long
                )
                press_states = torch.ones(batch_size, device=device, dtype=torch.long)
                release_logits, future_logits, axis_vector = model(
                    inputs, release_states
                )
                press_logits, _, _ = model(inputs, press_states)
                release_batches.append(torch.sigmoid(release_logits).cpu().numpy())
                press_batches.append(torch.sigmoid(press_logits).cpu().numpy())
                future_batches.append(torch.sigmoid(future_logits).cpu().numpy())
                axis_batches.append(axis_vector.cpu().numpy())
                axis_target_batches.append(batch["axis_target"].numpy())
                axis_weight_batches.append(batch["axis_weight"].numpy())
    finally:
        dataset.close()

    switch_if_release = np.concatenate(release_batches, axis=0)
    switch_if_press = np.concatenate(press_batches, axis=0)
    future_probabilities = np.concatenate(future_batches, axis=0)
    axis_prediction = np.concatenate(axis_batches, axis=0)
    axis_target = np.concatenate(axis_target_batches, axis=0)
    axis_weight = np.concatenate(axis_weight_batches, axis=0).reshape(-1)
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
    axis_result = axis_metrics(axis_prediction, axis_target, axis_weight)
    auxiliary = {
        f"{h:g}ms": binary_metrics(
            future_probabilities[:, index], future_targets[:, index], 0.5
        )
        for index, h in enumerate(horizons)
    }
    label_data = {
        video: load_label_data(args.labels_dir.resolve(), video)
        for video in getattr(split_config, args.split)
    }
    print(
        f"Device: {device}; split={args.split}; samples={len(samples)}; "
        f"axis_n={axis_result['samples']}; "
        f"axis_err={axis_result['weighted_mean_error_deg']:.2f}deg",
        flush=True,
    )
    print(
        "Aux future-action F1: "
        + ", ".join(
            f"h{h:g}={auxiliary[f'{h:g}ms']['f1']:.3f}" for h in horizons
        ),
        flush=True,
    )

    threshold_results = {}
    primary_targets = switch_targets[:, control_index]
    primary_probabilities = expert_switch_probabilities[:, control_index]
    stable_mask = ~near_transition[:, control_index]
    for threshold in (float(value) for value in args.thresholds):
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
        observation_transition = stateful["observation_timeline"]["transition"][
            "all"
        ]
        target_short = stateful["target_timeline"]["release_segment_recall"][
            "short_100_300ms"
        ]
        chatter = stateful["chatter"]
        print(
            f"threshold={threshold:.2f}/static: p={static['precision']:.3f}, "
            f"r={static['recall']:.3f}, f1={static['f1']:.3f}, "
            f"pred_pos={static['predicted_positive_rate']:.3f}, "
            f"stable_fpr={stable_fpr:.3f}, "
            f"all_negative_fpr={all_negative_fpr:.3f}",
            flush=True,
        )
        print(
            f"threshold={threshold:.2f}/stateful-target: "
            f"transition_f1={target_transition['f1']:.3f}, "
            f"matched={target_transition['matched']}/{target_transition['ground_truth']}, "
            f"predicted={target_transition['predicted']}, "
            f"short_recall={target_short['recall']:.3f}, "
            f"chatter_lt100={chatter['lt_100ms']}, "
            f"chatter_lt200={chatter['lt_200ms']}",
            flush=True,
        )
        print(
            f"threshold={threshold:.2f}/stateful-observation: "
            f"transition_f1={observation_transition['f1']:.3f}, "
            f"matched={observation_transition['matched']}/"
            f"{observation_transition['ground_truth']}, "
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
        "axis": axis_result,
        "auxiliary_future_action": auxiliary,
        "thresholds": threshold_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Evaluation: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
