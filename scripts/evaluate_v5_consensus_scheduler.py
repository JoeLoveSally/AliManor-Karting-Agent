#!/usr/bin/env python3
"""Evaluate consensus dense-horizon scheduling on V5 validation visuals."""

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

import evaluate_v5_dense_scheduler as dense_base  # noqa: E402
import evaluate_v5_h0_closed_loop as base  # noqa: E402

from karting_agent.model.kart_relative_supervised import (  # noqa: E402
    build_kart_relative_model,
)
from karting_agent.runtime.consensus_dense_scheduler import (  # noqa: E402
    ConsensusDenseConfig,
    ConsensusDenseHorizonScheduler,
)
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.h0_closed_loop import simulate_h0_closed_loop  # noqa: E402
from karting_agent.train.sequence_evaluator import (  # noqa: E402
    SequencePoint,
    evaluate_sequence,
    transitions_from_points,
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
        description="Evaluate V5 future-head transition-time consensus on validation."
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
    parser.add_argument("--min-consensus-heads", type=int, default=3)
    parser.add_argument("--consensus-window-ms", type=float, default=60.0)
    parser.add_argument("--tolerance-ms", type=float, default=None)
    return parser.parse_args()


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
        else artifact / "evaluation" / "validation_consensus_dense_scheduler.json"
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
    num_workers = (
        loop_config.num_workers if args.num_workers is None else args.num_workers
    )
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
    if (
        switch_if_release.shape != expected_shape
        or switch_if_press.shape != expected_shape
    ):
        raise RuntimeError("switch prediction shape mismatch")

    indices_by_video: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_video[sample.video].append(index)
    labels_dir = args.labels_dir.resolve()
    label_data = {
        video: base.load_label_data(labels_dir, video) for video in indices_by_video
    }

    scheduler_config = ConsensusDenseConfig(
        horizons_ms=horizons,
        threshold=args.threshold,
        min_state_hold_ms=args.min_state_hold_ms,
        min_consensus_heads=args.min_consensus_heads,
        consensus_window_ms=args.consensus_window_ms,
    )
    scheduler_config.validate()

    consensus_points_by_video: dict[str, list[SequencePoint]] = {}
    consensus_switch_times: dict[str, list[float]] = {}
    transition_events_by_video: dict[str, list[dict[str, object]]] = {}
    event_counts: Counter[str] = Counter()
    armed_delays: list[float] = []
    armed_spreads: list[float] = []
    armed_support_sizes: list[int] = []
    state_correct = 0
    state_total = 0

    h0_decisions_by_video = {}
    expert_states_by_video: dict[str, list[bool]] = {}

    for video, raw_indices in indices_by_video.items():
        indices = sorted(
            raw_indices,
            key=lambda index: float(samples[index].input_timestamps_ms[-1]),
        )
        timestamps = [
            float(samples[index].input_timestamps_ms[-1]) for index in indices
        ]
        initial = samples[indices[0]].current_pressed
        if initial is None:
            raise ValueError("validation sample missing current_pressed")
        state = bool(initial)
        scheduler = ConsensusDenseHorizonScheduler(scheduler_config)
        points = [
            SequencePoint(
                video=video,
                timestamp_ms=timestamps[0] - 1e-3,
                probability=1.0 if state else 0.0,
            )
        ]
        switch_times: list[float] = []
        transition_events: list[dict[str, object]] = []

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
                    event_counts["consensus_execute_deadline"] += 1
                    transition_events.append(
                        {
                            "timestamp_ms": due_ms,
                            "pressed": state,
                            "reason": "consensus_execute_deadline",
                            "support_heads_ms": list(execution.support_heads_ms),
                            "due_spread_ms": execution.due_spread_ms,
                            "armed_at_ms": execution.armed_at_ms,
                            "delay_ms": execution.delay_ms,
                        }
                    )

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
            event_counts[decision.reason] += 1
            if decision.reason == "consensus_armed":
                if decision.pending_delay_ms is not None:
                    armed_delays.append(float(decision.pending_delay_ms))
                if decision.consensus_due_spread_ms is not None:
                    armed_spreads.append(float(decision.consensus_due_spread_ms))
                armed_support_sizes.append(len(decision.consensus_heads_ms))
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
                transition_events.append(
                    {
                        "timestamp_ms": timestamp_ms,
                        "pressed": state,
                        "reason": decision.reason,
                        "support_heads_ms": list(decision.consensus_heads_ms),
                        "due_spread_ms": decision.consensus_due_spread_ms,
                        "armed_at_ms": None,
                        "delay_ms": None,
                    }
                )

            state_total += 1
            state_correct += int(state == expert_states[local_index])

        final_timestamp_ms = timestamps[-1]
        if points[-1].timestamp_ms < final_timestamp_ms - 1e-6:
            points.append(
                SequencePoint(
                    video=video,
                    timestamp_ms=final_timestamp_ms,
                    probability=1.0 if state else 0.0,
                )
            )
        consensus_points_by_video[video] = points
        consensus_switch_times[video] = switch_times
        transition_events_by_video[video] = transition_events

    h0_summary = base.replay_summary(
        h0_decisions_by_video,
        expert_states_by_video,
        label_data,
        tolerance_ms=tolerance_ms,
    )
    consensus_summary = dense_base._event_summary(
        consensus_points_by_video,
        consensus_switch_times,
        (state_correct, state_total),
        label_data,
        tolerance_ms=tolerance_ms,
    )

    def unmatched_diagnostics(
        points_by_video: dict[str, list[SequencePoint]],
        event_metadata_by_video: dict[str, list[dict[str, object]]] | None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        false_positives: list[dict[str, object]] = []
        false_negatives: list[dict[str, object]] = []

        for video, points in points_by_video.items():
            truth_transitions, releases = label_data[video]
            evaluation = evaluate_sequence(
                points,
                truth_transitions,
                releases,
                threshold=0.5,
                tolerance_ms=tolerance_ms,
            )
            matched_predicted = {match.predicted for match in evaluation.matches}
            matched_expected = {match.expected for match in evaluation.matches}
            metadata = (
                event_metadata_by_video.get(video, [])
                if event_metadata_by_video
                else []
            )
            metadata_by_key = {
                (float(item["timestamp_ms"]), bool(item["pressed"])): item
                for item in metadata
            }
            predicted = list(evaluation.predicted_transitions)

            for index, event in enumerate(predicted):
                if event in matched_predicted:
                    continue
                same_direction = [
                    expected
                    for expected in evaluation.expected_transitions
                    if expected.pressed == event.pressed
                ]
                nearest_delta = (
                    min(
                        (
                            event.timestamp_ms - expected.timestamp_ms
                            for expected in same_direction
                        ),
                        key=abs,
                    )
                    if same_direction
                    else None
                )
                details = dict(
                    metadata_by_key.get(
                        (float(event.timestamp_ms), bool(event.pressed)),
                        {},
                    )
                )
                details.update(
                    {
                        "video": video,
                        "timestamp_ms": event.timestamp_ms,
                        "pressed": event.pressed,
                        "nearest_same_direction_gt_delta_ms": nearest_delta,
                        "previous_transition_gap_ms": (
                            event.timestamp_ms - predicted[index - 1].timestamp_ms
                            if index > 0
                            else None
                        ),
                        "next_transition_gap_ms": (
                            predicted[index + 1].timestamp_ms - event.timestamp_ms
                            if index + 1 < len(predicted)
                            else None
                        ),
                    }
                )
                false_positives.append(details)

            for event in evaluation.expected_transitions:
                if event in matched_expected:
                    continue
                false_negatives.append(
                    {
                        "video": video,
                        "timestamp_ms": event.timestamp_ms,
                        "pressed": event.pressed,
                    }
                )

        return false_positives, false_negatives

    h0_points_by_video = {
        video: base.points_from_decisions(video, decisions)
        for video, decisions in h0_decisions_by_video.items()
    }
    h0_transitions_by_video = {
        video: transitions_from_points(points, threshold=0.5)
        for video, points in h0_points_by_video.items()
    }
    h0_false_positives, h0_false_negatives = unmatched_diagnostics(
        h0_points_by_video,
        None,
    )
    consensus_false_positives, consensus_false_negatives = unmatched_diagnostics(
        consensus_points_by_video,
        transition_events_by_video,
    )
    for item in consensus_false_positives:
        h0_same_direction = [
            event
            for event in h0_transitions_by_video[str(item["video"])]
            if event.pressed == bool(item["pressed"])
        ]
        item["nearest_same_direction_h0_delta_ms"] = (
            min(
                (
                    float(item["timestamp_ms"]) - event.timestamp_ms
                    for event in h0_same_direction
                ),
                key=abs,
            )
            if h0_same_direction
            else None
        )

    delay_array = np.asarray(armed_delays, dtype=np.float64)
    spread_array = np.asarray(armed_spreads, dtype=np.float64)
    support_array = np.asarray(armed_support_sizes, dtype=np.float64)

    def summary(values: np.ndarray) -> dict[str, float | int]:
        return {
            "count": int(values.size),
            "mean": float(values.mean()) if values.size else 0.0,
            "p50": float(np.percentile(values, 50)) if values.size else 0.0,
            "p95": float(np.percentile(values, 95)) if values.size else 0.0,
            "min": float(values.min()) if values.size else 0.0,
            "max": float(values.max()) if values.size else 0.0,
        }

    delay_summary = summary(delay_array)
    spread_summary = summary(spread_array)
    support_summary = summary(support_array)

    print(
        f"split=validation samples={len(samples)} device={device} "
        f"threshold={args.threshold:.2f} min_hold={args.min_state_hold_ms:g}ms "
        f"consensus_heads={args.min_consensus_heads} "
        f"consensus_window={args.consensus_window_ms:g}ms",
        flush=True,
    )
    dense_base._print_summary(f"h0_min_hold_{args.min_state_hold_ms:g}ms", h0_summary)
    dense_base._print_summary("consensus_dense_scheduler", consensus_summary)
    print(
        "consensus events: "
        + ", ".join(f"{key}={value}" for key, value in sorted(event_counts.items())),
        flush=True,
    )
    print(
        f"consensus armed delay: n={delay_summary['count']} "
        f"mean={delay_summary['mean']:.1f}ms p50={delay_summary['p50']:.1f}ms "
        f"p95={delay_summary['p95']:.1f}ms "
        f"range={delay_summary['min']:.1f}..{delay_summary['max']:.1f}ms",
        flush=True,
    )
    print(
        f"consensus spread: n={spread_summary['count']} "
        f"mean={spread_summary['mean']:.1f}ms p95={spread_summary['p95']:.1f}ms "
        f"support_mean={support_summary['mean']:.2f} "
        f"support_range={support_summary['min']:.0f}..{support_summary['max']:.0f}",
        flush=True,
    )

    print(
        f"unmatched transitions: h0_fp={len(h0_false_positives)} "
        f"h0_fn={len(h0_false_negatives)} "
        f"consensus_fp={len(consensus_false_positives)} "
        f"consensus_fn={len(consensus_false_negatives)}",
        flush=True,
    )
    print("consensus false positives:", flush=True)
    for item in consensus_false_positives:
        heads = ",".join(f"h{value:g}" for value in item.get("support_heads_ms", []))
        nearest = item["nearest_same_direction_gt_delta_ms"]
        nearest_h0 = item["nearest_same_direction_h0_delta_ms"]
        previous_gap = item["previous_transition_gap_ms"]
        next_gap = item["next_transition_gap_ms"]
        print(
            f"  {Path(str(item['video'])).name} "
            f"t={float(item['timestamp_ms']):.1f}ms "
            f"{'PRESS' if item['pressed'] else 'RELEASE'} "
            f"reason={item.get('reason', '<unknown>')} "
            f"heads=[{heads}] spread={item.get('due_spread_ms')} "
            f"nearest_gt_delta={nearest} nearest_h0_delta={nearest_h0} "
            f"prev_gap={previous_gap} next_gap={next_gap}",
            flush=True,
        )
    if consensus_false_negatives:
        print("consensus false negatives:", flush=True)
        for item in consensus_false_negatives:
            print(
                f"  {Path(str(item['video'])).name} "
                f"t={float(item['timestamp_ms']):.1f}ms "
                f"{'PRESS' if item['pressed'] else 'RELEASE'}",
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
        "consensus_dense_scheduler": consensus_summary,
        "consensus_events": dict(sorted(event_counts.items())),
        "consensus_armed_delay_ms": delay_summary,
        "consensus_due_spread_ms": spread_summary,
        "consensus_support_size": support_summary,
        "diagnostics": {
            "h0_false_positives": h0_false_positives,
            "h0_false_negatives": h0_false_negatives,
            "consensus_false_positives": consensus_false_positives,
            "consensus_false_negatives": consensus_false_negatives,
        },
        "test_evaluated": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    print("Frozen test split was not evaluated.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
