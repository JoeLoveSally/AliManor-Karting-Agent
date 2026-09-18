#!/usr/bin/env python3
"""Evaluate V5 consensus scheduling with a coherent future-action H0.

The candidate keeps the already validated native conditioned switch heads for
H50..H300 transition-time consensus, but replaces only H0 with the visual-only
future-action H0 projected back to SWITCH for the current physical state.

Lifecycle confirmation uses the same coherent H0 source. The frozen test split
is intentionally not evaluated.
"""

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
from karting_agent.runtime.consensus_transition_lifecycle import (  # noqa: E402
    ConsensusTransitionLifecycle,
)
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.h0_closed_loop import simulate_h0_closed_loop  # noqa: E402
from karting_agent.train.sequence_evaluator import SequencePoint  # noqa: E402
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
        description="Evaluate native future consensus with coherent future-action H0."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_v5_h0.yaml",
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


def _project_h0(desired_press: float, current_pressed: bool) -> float:
    return 1.0 - desired_press if current_pressed else desired_press


def main() -> int:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    if args.min_state_hold_ms < 0:
        raise ValueError("--min-state-hold-ms must be >= 0")
    if args.min_consensus_heads < 2:
        raise ValueError("--min-consensus-heads must be >= 2")
    if args.consensus_window_ms < 0:
        raise ValueError("--consensus-window-ms must be >= 0")
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
        args.metadata.resolve()
        if args.metadata
        else artifact / "metadata.json"
    )
    output_path = (
        args.output.resolve()
        if args.output
        else artifact
        / "evaluation"
        / "validation_consensus_future_h0.json"
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("model_family", "")) != "state_conditioned_kart_relative_v5_h0":
        raise ValueError("expected a state_conditioned_kart_relative_v5_h0 artifact")
    horizons = tuple(float(value) for value in metadata["prediction_horizons_ms"])
    if horizons != base.EXPECTED_HORIZONS_MS:
        raise ValueError(f"unexpected V5-H0 horizons: {horizons}")
    h0_index = horizons.index(0.0)

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
        loop_config.num_workers
        if args.num_workers is None
        else args.num_workers
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
    future_batches: list[np.ndarray] = []
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
                release_state = torch.zeros(
                    batch_size,
                    device=device,
                    dtype=torch.long,
                )
                press_state = torch.ones(
                    batch_size,
                    device=device,
                    dtype=torch.long,
                )
                release_outputs = model.forward_from_features(
                    features,
                    release_state,
                )
                press_outputs = model.forward_from_features(
                    features,
                    press_state,
                )
                release_batches.append(
                    torch.sigmoid(release_outputs[0]).cpu().numpy()
                )
                press_batches.append(
                    torch.sigmoid(press_outputs[0]).cpu().numpy()
                )
                future_batches.append(
                    torch.sigmoid(release_outputs[1]).cpu().numpy()
                )
    finally:
        dataset.close()

    switch_if_release = np.concatenate(release_batches, axis=0)
    switch_if_press = np.concatenate(press_batches, axis=0)
    future_press = np.concatenate(future_batches, axis=0)
    expected_shape = (len(samples), len(horizons))
    for name, values in (
        ("switch_if_release", switch_if_release),
        ("switch_if_press", switch_if_press),
        ("future_press", future_press),
    ):
        if values.shape != expected_shape:
            raise RuntimeError(f"{name} shape mismatch: {values.shape}")

    indices_by_video: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_video[sample.video].append(index)

    labels_dir = args.labels_dir.resolve()
    label_data = {
        video: base.load_label_data(labels_dir, video)
        for video in indices_by_video
    }

    scheduler_config = ConsensusDenseConfig(
        horizons_ms=horizons,
        threshold=args.threshold,
        min_state_hold_ms=args.min_state_hold_ms,
        min_consensus_heads=args.min_consensus_heads,
        consensus_window_ms=args.consensus_window_ms,
    )
    scheduler_config.validate()

    direct_decisions_by_video = {}
    expert_states_by_video: dict[str, list[bool]] = {}
    points_by_video: dict[str, list[SequencePoint]] = {}
    switch_times_by_video: dict[str, list[float]] = {}
    scheduler_events: Counter[str] = Counter()
    lifecycle_events: Counter[str] = Counter()
    state_correct = 0
    state_total = 0
    armed_delays: list[float] = []
    armed_spreads: list[float] = []

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
        expert_states = [
            bool(samples[index].current_pressed) for index in indices
        ]
        expert_states_by_video[video] = expert_states

        desired_press = [
            float(future_press[index, h0_index]) for index in indices
        ]
        direct_release = desired_press
        direct_press = [1.0 - value for value in desired_press]
        direct_decisions_by_video[video] = simulate_h0_closed_loop(
            timestamps,
            direct_release,
            direct_press,
            initial_pressed=bool(initial),
            threshold=args.threshold,
            min_state_hold_ms=args.min_state_hold_ms,
        )

        state = bool(initial)
        scheduler = ConsensusDenseHorizonScheduler(scheduler_config)
        lifecycle = ConsensusTransitionLifecycle(
            threshold=args.threshold,
            min_state_hold_ms=args.min_state_hold_ms,
        )
        points = [
            SequencePoint(
                video=video,
                timestamp_ms=timestamps[0] - 1e-3,
                probability=1.0 if state else 0.0,
            )
        ]
        switch_times: list[float] = []

        for local_index, sample_index in enumerate(indices):
            timestamp_ms = timestamps[local_index]
            due_ms = scheduler.pending_due_ms
            if due_ms is not None and due_ms <= timestamp_ms + 1e-6:
                previous_state = state
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
                    lifecycle.begin(
                        executed_at_ms=due_ms,
                        previous_pressed=previous_state,
                        current_pressed=state,
                        evidence_lead_ms=execution.delay_ms,
                    )
                    scheduler_events["consensus_execute_deadline"] += 1

            native_probabilities = (
                switch_if_press[sample_index]
                if state
                else switch_if_release[sample_index]
            )
            effective_probabilities = np.asarray(
                native_probabilities,
                dtype=np.float64,
            ).copy()
            desired_press_h0 = float(future_press[sample_index, h0_index])
            effective_probabilities[h0_index] = _project_h0(
                desired_press_h0,
                state,
            )

            if lifecycle.active:
                pending = lifecycle.pending
                assert pending is not None
                previous_state_h0 = _project_h0(
                    desired_press_h0,
                    pending.previous_pressed,
                )
                lifecycle_status = lifecycle.observe(
                    timestamp_ms=timestamp_ms,
                    previous_state_h0_probability=previous_state_h0,
                )
                lifecycle_events[lifecycle_status] += 1
                if lifecycle_status in {"waiting", "confirmed"}:
                    effective_probabilities[h0_index] = 0.0

            decision = scheduler.update(
                timestamp_ms=timestamp_ms,
                probabilities=effective_probabilities,
                current_pressed=state,
            )
            scheduler_events[decision.reason] += 1
            if decision.reason == "consensus_armed":
                if decision.pending_delay_ms is not None:
                    armed_delays.append(float(decision.pending_delay_ms))
                if decision.consensus_due_spread_ms is not None:
                    armed_spreads.append(
                        float(decision.consensus_due_spread_ms)
                    )

            if decision.switch:
                previous_state = state
                state = not state
                switch_times.append(timestamp_ms)
                points.append(
                    SequencePoint(
                        video=video,
                        timestamp_ms=timestamp_ms,
                        probability=1.0 if state else 0.0,
                    )
                )
                if decision.reason == "consensus_execute":
                    lifecycle.begin(
                        executed_at_ms=timestamp_ms,
                        previous_pressed=previous_state,
                        current_pressed=state,
                        evidence_lead_ms=float(
                            decision.evidence_lead_ms or 0.0
                        ),
                    )
                else:
                    lifecycle.clear()

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
        points_by_video[video] = points
        switch_times_by_video[video] = switch_times

    direct_summary = base.replay_summary(
        direct_decisions_by_video,
        expert_states_by_video,
        label_data,
        tolerance_ms=tolerance_ms,
    )
    hybrid_summary = dense_base._event_summary(
        points_by_video,
        switch_times_by_video,
        (state_correct, state_total),
        label_data,
        tolerance_ms=tolerance_ms,
    )

    def numeric_summary(values: list[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean": float(array.mean()) if array.size else 0.0,
            "p50": float(np.percentile(array, 50)) if array.size else 0.0,
            "p95": float(np.percentile(array, 95)) if array.size else 0.0,
            "min": float(array.min()) if array.size else 0.0,
            "max": float(array.max()) if array.size else 0.0,
        }

    delay_summary = numeric_summary(armed_delays)
    spread_summary = numeric_summary(armed_spreads)

    print(
        f"split=validation samples={len(samples)} device={device} "
        f"threshold={args.threshold:.2f} "
        f"min_hold={args.min_state_hold_ms:g}ms "
        f"consensus_heads={args.min_consensus_heads} "
        f"consensus_window={args.consensus_window_ms:g}ms",
        flush=True,
    )
    dense_base._print_summary(
        "future_action_h0_direct",
        direct_summary,
    )
    dense_base._print_summary(
        "consensus_future_h0_lifecycle",
        hybrid_summary,
    )
    print(
        "scheduler events: "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(scheduler_events.items())
        ),
        flush=True,
    )
    print(
        "lifecycle events: "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(lifecycle_events.items())
        ),
        flush=True,
    )
    print(
        f"armed delay: n={delay_summary['count']} "
        f"mean={delay_summary['mean']:.1f}ms "
        f"p50={delay_summary['p50']:.1f}ms "
        f"p95={delay_summary['p95']:.1f}ms",
        flush=True,
    )
    print(
        f"consensus spread: n={spread_summary['count']} "
        f"mean={spread_summary['mean']:.1f}ms "
        f"p95={spread_summary['p95']:.1f}ms",
        flush=True,
    )

    payload = {
        "split": "validation",
        "samples": len(samples),
        "model": str(model_path),
        "metadata": str(metadata_path),
        "scheduler_config": asdict(scheduler_config),
        "control_h0_source": "future_action_h0_projection",
        "future_heads_source": "native_switch",
        "transition_tolerance_ms": tolerance_ms,
        "future_action_h0_direct": direct_summary,
        "consensus_future_h0_lifecycle": hybrid_summary,
        "scheduler_events": dict(sorted(scheduler_events.items())),
        "lifecycle_events": dict(sorted(lifecycle_events.items())),
        "armed_delay_ms": delay_summary,
        "consensus_spread_ms": spread_summary,
        "test_evaluated": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    print("Frozen test split was not evaluated.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
