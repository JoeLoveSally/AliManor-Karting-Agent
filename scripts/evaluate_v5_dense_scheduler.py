#!/usr/bin/env python3
"""Evaluate V5 dense-horizon scheduling on validation expert visuals only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
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

import evaluate_v5_h0_closed_loop as base  # noqa: E402

from karting_agent.model.kart_relative_supervised import (  # noqa: E402
    build_kart_relative_model,
)
from karting_agent.runtime.dense_horizon_scheduler import (  # noqa: E402
    DenseHorizonSchedulerConfig,
    DenseHorizonSwitchScheduler,
)
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.h0_closed_loop import simulate_h0_closed_loop  # noqa: E402
from karting_agent.train.sequence_evaluator import (  # noqa: E402
    SequencePoint,
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
        description="Evaluate H0-anchored dense future scheduling on validation."
    )
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


def _event_summary(
    points_by_video: dict[str, list[SequencePoint]],
    switch_times_by_video: dict[str, list[float]],
    observation_accuracy: tuple[int, int],
    label_data,
    *,
    tolerance_ms: float,
) -> dict[str, object]:
    evaluations = []
    chatter_lt100 = 0
    chatter_lt200 = 0
    transitions = 0
    for video, points in points_by_video.items():
        truth_transitions, releases = label_data[video]
        evaluations.append(
            evaluate_sequence(
                points,
                truth_transitions,
                releases,
                threshold=0.5,
                tolerance_ms=tolerance_ms,
            )
        )
        times = np.asarray(switch_times_by_video[video], dtype=np.float64)
        intervals = np.diff(times)
        transitions += int(times.size)
        if intervals.size:
            chatter_lt100 += int((intervals < 100.0).sum())
            chatter_lt200 += int((intervals < 200.0).sum())
    correct, total = observation_accuracy
    return {
        "sequence": combine_evaluations(evaluations).summary(),
        "chatter": {
            "transitions": transitions,
            "lt_100ms": chatter_lt100,
            "lt_200ms": chatter_lt200,
        },
        "observation_state_accuracy": correct / total if total else 0.0,
    }


def _print_summary(name: str, summary: dict[str, object]) -> None:
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
        f"press_mae={press['mae_ms']:.1f}ms "
        f"release_mae={release['mae_ms']:.1f}ms "
        f"state_acc={summary['observation_state_accuracy']:.3f} "
        f"chatter_lt100={chatter['lt_100ms']} lt200={chatter['lt_200ms']}",
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
    raw = base.load_yaml_mapping(config_path)
    split_config = load_video_split(config_path)
    loop_config = load_train_loop_config(config_path)
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)
    tolerance_ms = base.evaluation_tolerance(raw, args.tolerance_ms)

    artifact = base.artifact_dir(raw)
    model_path = args.model.resolve() if args.model else artifact / "model.pt"
    metadata_path = (
        args.metadata.resolve() if args.metadata else artifact / "metadata.json"
    )
    output_path = (
        args.output.resolve()
        if args.output
        else artifact / "evaluation" / "validation_dense_horizon_scheduler.json"
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("model_family", "")) != "state_conditioned_kart_relative_v5_h0":
        raise ValueError("expected a state_conditioned_kart_relative_v5_h0 artifact")
    horizons = tuple(float(value) for value in metadata["prediction_horizons_ms"])
    if horizons != base.EXPECTED_HORIZONS_MS:
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
    device = base.select_device(torch, args.device)
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
                release_state = torch.zeros(batch_size, device=device, dtype=torch.long)
                press_state = torch.ones(batch_size, device=device, dtype=torch.long)
                release_logits = model.forward_from_features(features, release_state)[0]
                press_logits = model.forward_from_features(features, press_state)[0]
                release_batches.append(torch.sigmoid(release_logits).cpu().numpy())
                press_batches.append(torch.sigmoid(press_logits).cpu().numpy())
    finally:
        dataset.close()

    switch_if_release = np.concatenate(release_batches, axis=0)
    switch_if_press = np.concatenate(press_batches, axis=0)
    expected_shape = (len(samples), len(horizons))
    if switch_if_release.shape != expected_shape or switch_if_press.shape != expected_shape:
        raise RuntimeError("switch prediction shape mismatch")

    indices_by_video: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_video[sample.video].append(index)
    labels_dir = args.labels_dir.resolve()
    label_data = {
        video: base.load_label_data(labels_dir, video) for video in indices_by_video
    }

    dense_points_by_video: dict[str, list[SequencePoint]] = {}
    dense_switch_times: dict[str, list[float]] = {}
    dense_reason_counts: Counter[str] = Counter()
    dense_deadline_executes = 0
    dense_armed_delays: list[float] = []
    dense_correct = 0
    dense_total = 0

    h0_decisions_by_video = {}
    expert_states_by_video: dict[str, list[bool]] = {}

    scheduler_config = DenseHorizonSchedulerConfig(
        horizons_ms=horizons,
        threshold=args.threshold,
        min_state_hold_ms=args.min_state_hold_ms,
    )

    for video, raw_indices in indices_by_video.items():
        indices = sorted(
            raw_indices,
            key=lambda index: float(samples[index].input_timestamps_ms[-1]),
        )
        timestamps = [float(samples[index].input_timestamps_ms[-1]) for index in indices]
        initial = samples[indices[0]].current_pressed
        if initial is None:
            raise ValueError("validation sample missing current_pressed")
        state = bool(initial)
        scheduler = DenseHorizonSwitchScheduler(scheduler_config)
        first_timestamp = timestamps[0]
        points = [
            SequencePoint(
                video=video,
                timestamp_ms=first_timestamp - 1e-3,
                probability=1.0 if state else 0.0,
            )
        ]
        switch_times: list[float] = []

        expert_states = [bool(samples[index].current_pressed) for index in indices]
        expert_states_by_video[video] = expert_states
        release_h0 = [float(switch_if_release[index, 0]) for index in indices]
        press_h0 = [float(switch_if_press[index, 0]) for index in indices]
        h0_decisions_by_video[video] = simulate_h0_closed_loop(
            timestamps,
            release_h0,
            press_h0,
            initial_pressed=state,
            threshold=args.threshold,
            min_state_hold_ms=args.min_state_hold_ms,
        )

        for local_index, sample_index in enumerate(indices):
            timestamp_ms = timestamps[local_index]
            due_ms = scheduler.pending_due_ms
            if due_ms is not None and due_ms <= timestamp_ms + 1e-6:
                execution = scheduler.execute_pending_if_due(
                    timestamp_ms=due_ms,
                    current_pressed=state,
                )
                if execution is not None:
                    state = not state
                    switch_times.append(due_ms)
                    points.append(
                        SequencePoint(
                            video=video,
                            timestamp_ms=due_ms,
                            probability=1.0 if state else 0.0,
                        )
                    )
                    dense_reason_counts["pending_execute_deadline"] += 1
                    dense_deadline_executes += 1

            probabilities = (
                switch_if_press[sample_index]
                if state
                else switch_if_release[sample_index]
            )
            decision = scheduler.update(
                timestamp_ms=timestamp_ms,
                probabilities=probabilities,
                current_pressed=state,
            )
            dense_reason_counts[decision.reason] += 1
            if decision.reason == "pending_armed" and decision.pending_delay_ms is not None:
                dense_armed_delays.append(float(decision.pending_delay_ms))
            if decision.switch:
                state = not state
                switch_times.append(timestamp_ms)
                points.append(
                    SequencePoint(
                        video=video,
                        timestamp_ms=timestamp_ms,
                        probability=1.0 if state else 0.0,
                    )
                )

            dense_total += 1
            dense_correct += int(state == expert_states[local_index])

        final_timestamp_ms = timestamps[-1]
        if points[-1].timestamp_ms < final_timestamp_ms - 1e-6:
            points.append(
                SequencePoint(
                    video=video,
                    timestamp_ms=final_timestamp_ms,
                    probability=1.0 if state else 0.0,
                )
            )
        dense_points_by_video[video] = points
        dense_switch_times[video] = switch_times

    h0_summary = base.replay_summary(
        h0_decisions_by_video,
        expert_states_by_video,
        label_data,
        tolerance_ms=tolerance_ms,
    )
    dense_summary = _event_summary(
        dense_points_by_video,
        dense_switch_times,
        (dense_correct, dense_total),
        label_data,
        tolerance_ms=tolerance_ms,
    )

    delays = np.asarray(dense_armed_delays, dtype=np.float64)
    delay_summary = {
        "count": int(delays.size),
        "mean_ms": float(delays.mean()) if delays.size else 0.0,
        "p50_ms": float(np.percentile(delays, 50)) if delays.size else 0.0,
        "p95_ms": float(np.percentile(delays, 95)) if delays.size else 0.0,
        "min_ms": float(delays.min()) if delays.size else 0.0,
        "max_ms": float(delays.max()) if delays.size else 0.0,
    }

    print(
        f"split=validation samples={len(samples)} device={device} "
        f"threshold={args.threshold:.2f} min_hold={args.min_state_hold_ms:g}ms",
        flush=True,
    )
    _print_summary(f"h0_min_hold_{args.min_state_hold_ms:g}ms", h0_summary)
    _print_summary("dense_horizon_scheduler", dense_summary)
    print(
        "dense events: "
        + ", ".join(
            f"{key}={value}" for key, value in sorted(dense_reason_counts.items())
        ),
        flush=True,
    )
    print(
        f"dense armed delay: n={delay_summary['count']} "
        f"mean={delay_summary['mean_ms']:.1f}ms "
        f"p50={delay_summary['p50_ms']:.1f}ms "
        f"p95={delay_summary['p95_ms']:.1f}ms "
        f"range={delay_summary['min_ms']:.1f}..{delay_summary['max_ms']:.1f}ms",
        flush=True,
    )

    payload = {
        "split": "validation",
        "samples": len(samples),
        "model": str(model_path),
        "metadata": str(metadata_path),
        "scheduler_config": asdict(scheduler_config),
        "transition_tolerance_ms": tolerance_ms,
        "h0_min_hold": h0_summary,
        "dense_horizon_scheduler": dense_summary,
        "dense_events": dict(sorted(dense_reason_counts.items())),
        "dense_deadline_executes": dense_deadline_executes,
        "dense_armed_delay_ms": delay_summary,
        "test_evaluated": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    print("Frozen test split was not evaluated.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
