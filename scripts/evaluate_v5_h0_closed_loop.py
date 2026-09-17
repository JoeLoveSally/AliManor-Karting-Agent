#!/usr/bin/env python3
"""Evaluate V5-H0 as an immediate closed-loop controller on validation visuals.

The frozen test split is intentionally unavailable. The evaluator runs the model
for both possible physical action states at every expert observation, then replays
H0 against the controller's own simulated state. This separates H0 representation
quality from the previously used H200/H300 scheduler semantics.
"""

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

from karting_agent.model.kart_relative_supervised import (  # noqa: E402
    build_kart_relative_model,
)
from karting_agent.train.evaluator import BinaryMetricAccumulator  # noqa: E402
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.h0_closed_loop import (  # noqa: E402
    H0Decision,
    simulate_h0_closed_loop,
)
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

EXPECTED_HORIZONS_MS = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V5-H0 validation replay.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "train_v5_h0.yaml"
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "processed" / "v5_h0" / "samples.jsonl",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "v5_h0" / "labels",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--min-state-hold-ms", type=float, default=100.0)
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


def artifact_dir(raw: dict[str, object]) -> Path:
    section = raw.get("artifact", {})
    if not isinstance(section, dict):
        raise ValueError("artifact config must be a mapping")
    return ROOT / "artifacts" / "models" / str(section["name"])


def evaluation_tolerance(raw: dict[str, object], override: float | None) -> float:
    if override is not None:
        value = float(override)
    else:
        section = raw.get("evaluation", {})
        if not isinstance(section, dict):
            raise ValueError("evaluation config must be a mapping")
        value = float(section.get("transition_tolerance_ms", 100.0))
    if value < 0:
        raise ValueError("transition tolerance must be >= 0")
    return value


def load_label_data(
    labels_dir: Path,
    video: str,
) -> tuple[list[Transition], list[ReleaseSegment]]:
    payload = json.loads(
        (labels_dir / f"{Path(video).stem}.json").read_text(encoding="utf-8")
    )
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"label file has no events for {video}")
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


def points_from_decisions(video: str, decisions: list[H0Decision]) -> list[SequencePoint]:
    if not decisions:
        return []
    first = decisions[0]
    points = [
        SequencePoint(
            video=video,
            timestamp_ms=first.timestamp_ms - 1e-3,
            probability=1.0 if first.state_before else 0.0,
        )
    ]
    points.extend(
        SequencePoint(
            video=video,
            timestamp_ms=decision.timestamp_ms,
            probability=1.0 if decision.state_after else 0.0,
        )
        for decision in decisions
    )
    return points


def replay_summary(
    decisions_by_video: dict[str, list[H0Decision]],
    expert_states_by_video: dict[str, list[bool]],
    label_data: dict[str, tuple[list[Transition], list[ReleaseSegment]]],
    *,
    tolerance_ms: float,
) -> dict[str, object]:
    evaluations = []
    transitions = 0
    chatter_lt100 = 0
    chatter_lt200 = 0
    blocked = 0
    agreement_n = 0
    agreement_correct = 0

    for video, decisions in decisions_by_video.items():
        truth_transitions, releases = label_data[video]
        evaluations.append(
            evaluate_sequence(
                points_from_decisions(video, decisions),
                truth_transitions,
                releases,
                threshold=0.5,
                tolerance_ms=tolerance_ms,
            )
        )
        switch_times = np.asarray(
            [decision.timestamp_ms for decision in decisions if decision.switched],
            dtype=np.float64,
        )
        intervals = np.diff(switch_times)
        transitions += int(switch_times.size)
        chatter_lt100 += int((intervals < 100.0).sum()) if intervals.size else 0
        chatter_lt200 += int((intervals < 200.0).sum()) if intervals.size else 0
        blocked += sum(decision.blocked_by_min_hold for decision in decisions)

        expert_states = expert_states_by_video[video]
        if len(expert_states) != len(decisions):
            raise RuntimeError("replay/expert state length mismatch")
        agreement_n += len(decisions)
        agreement_correct += sum(
            decision.state_after == expert
            for decision, expert in zip(decisions, expert_states, strict=True)
        )

    summary = combine_evaluations(evaluations).summary()
    return {
        "sequence": summary,
        "chatter": {
            "transitions": transitions,
            "lt_100ms": chatter_lt100,
            "lt_200ms": chatter_lt200,
            "blocked_by_min_hold": blocked,
        },
        "observation_state_accuracy": (
            agreement_correct / agreement_n if agreement_n else 0.0
        ),
    }


def print_replay_summary(name: str, summary: dict[str, object]) -> None:
    sequence = summary["sequence"]
    transition = sequence["transition"]["all"]
    short = sequence["release_segment_recall"]["short_100_300ms"]
    press = sequence["onset_timing_ms"]["press"]
    release = sequence["onset_timing_ms"]["release"]
    chatter = summary["chatter"]
    print(
        f"{name}: target_f1={transition['f1']:.3f} "
        f"matched={transition['matched']}/{transition['ground_truth']} "
        f"pred={transition['predicted']} short_recall={short['recall']:.3f} "
        f"press_mae={press['mae_ms']:.1f}ms release_mae={release['mae_ms']:.1f}ms "
        f"state_acc={summary['observation_state_accuracy']:.3f} "
        f"chatter_lt100={chatter['lt_100ms']} lt200={chatter['lt_200ms']} "
        f"blocked={chatter['blocked_by_min_hold']}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    if args.min_state_hold_ms < 0:
        raise ValueError("--min-state-hold-ms must be >= 0")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    import torch
    from torch.utils.data import DataLoader

    config_path = args.config.resolve()
    raw = load_yaml_mapping(config_path)
    split_config = load_video_split(config_path)
    loop_config = load_train_loop_config(config_path)
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)
    tolerance_ms = evaluation_tolerance(raw, args.tolerance_ms)

    artifact = artifact_dir(raw)
    model_path = args.model.resolve() if args.model else artifact / "model.pt"
    metadata_path = args.metadata.resolve() if args.metadata else artifact / "metadata.json"
    output_path = (
        args.output.resolve()
        if args.output
        else artifact / "evaluation" / "validation_h0_closed_loop.json"
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("model_family", "")) != "state_conditioned_kart_relative_v5_h0":
        raise ValueError("expected a state_conditioned_kart_relative_v5_h0 artifact")
    horizons = tuple(float(value) for value in metadata["prediction_horizons_ms"])
    if horizons != EXPECTED_HORIZONS_MS:
        raise ValueError(f"unexpected V5-H0 horizons: {horizons}")

    model = build_kart_relative_model(
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
    samples = partition_samples(all_samples, split_config)["validation"]
    dataset = StateConditionedVideoDataset(
        samples,
        project_root=ROOT,
        preprocess_config=preprocess_config,
        cache_root=cache_root,
        require_cache=args.require_cache,
        counterfactual_states=False,
    )
    num_workers = loop_config.num_workers if args.num_workers is None else args.num_workers
    loader_kwargs: dict[str, object] = {
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
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["input"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                features = model.encode_visual(inputs)
                batch_size = int(inputs.shape[0])
                release = torch.zeros(batch_size, device=device, dtype=torch.long)
                press = torch.ones(batch_size, device=device, dtype=torch.long)
                release_logits = model.forward_from_features(features, release)[0]
                press_logits = model.forward_from_features(features, press)[0]
                release_batches.append(torch.sigmoid(release_logits).cpu().numpy())
                press_batches.append(torch.sigmoid(press_logits).cpu().numpy())
    finally:
        dataset.close()

    switch_if_release = np.concatenate(release_batches, axis=0)
    switch_if_press = np.concatenate(press_batches, axis=0)
    expected_shape = (len(samples), len(horizons))
    if switch_if_release.shape != expected_shape or switch_if_press.shape != expected_shape:
        raise RuntimeError("switch prediction shape mismatch")

    future_states = np.asarray(
        [sample.target_pressed_by_horizon for sample in samples], dtype=np.bool_
    )
    if future_states.shape != expected_shape:
        raise RuntimeError("future target shape mismatch")

    horizon_metrics: list[dict[str, object]] = []
    complement_error = np.abs(switch_if_release + switch_if_press - 1.0)
    for index, horizon_ms in enumerate(horizons):
        probabilities = np.concatenate(
            (switch_if_release[:, index], switch_if_press[:, index])
        )
        targets = np.concatenate(
            (future_states[:, index], np.logical_not(future_states[:, index]))
        ).astype(np.float32)
        accumulator = BinaryMetricAccumulator(args.threshold)
        accumulator.update(probabilities, targets)
        errors = complement_error[:, index]
        horizon_metrics.append(
            {
                "horizon_ms": horizon_ms,
                "switch": accumulator.result().to_dict(),
                "complement_error_mean": float(errors.mean()),
                "complement_error_p95": float(np.percentile(errors, 95)),
                "complement_error_max": float(errors.max()),
            }
        )

    expert_h0 = np.asarray(
        [
            switch_if_press[index, 0]
            if bool(sample.current_pressed)
            else switch_if_release[index, 0]
            for index, sample in enumerate(samples)
        ],
        dtype=np.float64,
    )
    expert_h0_summary = {
        "false_positive_rate": float((expert_h0 >= args.threshold).mean()),
        "mean": float(expert_h0.mean()),
        "p50": float(np.percentile(expert_h0, 50)),
        "p95": float(np.percentile(expert_h0, 95)),
        "p99": float(np.percentile(expert_h0, 99)),
        "max": float(expert_h0.max()),
    }

    indices_by_video: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_video[sample.video].append(index)

    labels_dir = args.labels_dir.resolve()
    label_data = {
        video: load_label_data(labels_dir, video) for video in indices_by_video
    }
    expert_states_by_video: dict[str, list[bool]] = {}
    no_hold_by_video: dict[str, list[H0Decision]] = {}
    guarded_by_video: dict[str, list[H0Decision]] = {}

    for video, raw_indices in indices_by_video.items():
        indices = sorted(
            raw_indices,
            key=lambda index: float(samples[index].input_timestamps_ms[-1]),
        )
        timestamps = [float(samples[index].input_timestamps_ms[-1]) for index in indices]
        release_h0 = [float(switch_if_release[index, 0]) for index in indices]
        press_h0 = [float(switch_if_press[index, 0]) for index in indices]
        initial = samples[indices[0]].current_pressed
        if initial is None:
            raise ValueError("validation sample missing current_pressed")
        expert_states_by_video[video] = [
            bool(samples[index].current_pressed) for index in indices
        ]
        no_hold_by_video[video] = simulate_h0_closed_loop(
            timestamps,
            release_h0,
            press_h0,
            initial_pressed=bool(initial),
            threshold=args.threshold,
            min_state_hold_ms=0.0,
        )
        guarded_by_video[video] = simulate_h0_closed_loop(
            timestamps,
            release_h0,
            press_h0,
            initial_pressed=bool(initial),
            threshold=args.threshold,
            min_state_hold_ms=args.min_state_hold_ms,
        )

    no_hold_summary = replay_summary(
        no_hold_by_video,
        expert_states_by_video,
        label_data,
        tolerance_ms=tolerance_ms,
    )
    guarded_summary = replay_summary(
        guarded_by_video,
        expert_states_by_video,
        label_data,
        tolerance_ms=tolerance_ms,
    )

    print(
        f"split=validation samples={len(samples)} device={device} "
        f"threshold={args.threshold:.2f} min_hold={args.min_state_hold_ms:g}ms",
        flush=True,
    )
    print("counterfactual per-horizon switch metrics:")
    for item in horizon_metrics:
        metric = item["switch"]
        print(
            f"  h{item['horizon_ms']:g}: p={metric['precision']:.3f} "
            f"r={metric['recall']:.3f} f1={metric['f1']:.3f} "
            f"comp_err_mean={item['complement_error_mean']:.3f} "
            f"p95={item['complement_error_p95']:.3f}",
            flush=True,
        )
    print(
        "expert-state h0: "
        f"false_positive_rate={expert_h0_summary['false_positive_rate']:.4f} "
        f"mean={expert_h0_summary['mean']:.3f} p95={expert_h0_summary['p95']:.3f} "
        f"p99={expert_h0_summary['p99']:.3f} max={expert_h0_summary['max']:.3f}",
        flush=True,
    )
    print_replay_summary("h0_no_hold", no_hold_summary)
    print_replay_summary(f"h0_min_hold_{args.min_state_hold_ms:g}ms", guarded_summary)

    payload = {
        "split": "validation",
        "samples": len(samples),
        "model": str(model_path),
        "metadata": str(metadata_path),
        "threshold": args.threshold,
        "min_state_hold_ms": args.min_state_hold_ms,
        "transition_tolerance_ms": tolerance_ms,
        "horizons": horizon_metrics,
        "expert_state_h0": expert_h0_summary,
        "h0_no_hold": no_hold_summary,
        "h0_min_hold": guarded_summary,
        "test_evaluated": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    print("Frozen test split was not evaluated.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
