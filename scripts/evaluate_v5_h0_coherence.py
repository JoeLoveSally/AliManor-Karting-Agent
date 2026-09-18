#!/usr/bin/env python3
"""Evaluate coherent V5-H0 control sources on validation visuals.

This evaluator keeps the validation split, threshold, and minimum state hold
fixed. It compares the native state-conditioned H0 controller with two
model-only coherence projections:

1. paired_switch_projection:
   Interpret p(switch | RELEASE) and 1 - p(switch | PRESS) as two estimates of
   the same latent P(desired PRESS), average them, then project back to a
   state-conditioned switch probability.

2. future_action_projection:
   Use the visual-only future-action H0 head as P(desired PRESS) and project it
   back to a state-conditioned switch probability.

The frozen test split is intentionally not evaluated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from karting_agent.runtime.h0_coherence import (  # noqa: E402
    desired_press_from_switch_pair,
    project_desired_press_to_switch,
)
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.h0_closed_loop import simulate_h0_closed_loop  # noqa: E402
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
        description="Evaluate coherent V5-H0 control sources on validation."
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
    parser.add_argument("--tolerance-ms", type=float, default=None)
    return parser.parse_args()


def _conflict_summary(
    release_h0: np.ndarray,
    press_h0: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    release_switch = release_h0 >= threshold
    press_switch = press_h0 >= threshold
    both_switch = release_switch & press_switch
    both_keep = (~release_switch) & (~press_switch)
    complement_error = np.abs(release_h0 + press_h0 - 1.0)
    return {
        "samples": int(release_h0.size),
        "both_switch": int(both_switch.sum()),
        "both_switch_rate": float(both_switch.mean()),
        "both_keep": int(both_keep.sum()),
        "both_keep_rate": float(both_keep.mean()),
        "complement_error_mean": float(complement_error.mean()),
        "complement_error_p95": float(np.percentile(complement_error, 95)),
        "complement_error_p99": float(np.percentile(complement_error, 99)),
        "complement_error_max": float(complement_error.max()),
    }


def _replay(
    *,
    samples,
    indices_by_video: dict[str, list[int]],
    release_h0: np.ndarray,
    press_h0: np.ndarray,
    threshold: float,
    min_state_hold_ms: float,
):
    decisions_by_video = {}
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
        expert_states_by_video[video] = [
            bool(samples[index].current_pressed) for index in indices
        ]
        decisions_by_video[video] = simulate_h0_closed_loop(
            timestamps,
            [float(release_h0[index]) for index in indices],
            [float(press_h0[index]) for index in indices],
            initial_pressed=bool(initial),
            threshold=threshold,
            min_state_hold_ms=min_state_hold_ms,
        )

    return decisions_by_video, expert_states_by_video


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
        args.metadata.resolve()
        if args.metadata
        else artifact / "metadata.json"
    )
    output_path = (
        args.output.resolve()
        if args.output
        else artifact / "evaluation" / "validation_h0_coherence.json"
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
                release_logits, future_logits, *_ = model.forward_from_features(
                    features,
                    release_state,
                )
                press_logits = model.forward_from_features(
                    features,
                    press_state,
                )[0]
                release_batches.append(torch.sigmoid(release_logits).cpu().numpy())
                press_batches.append(torch.sigmoid(press_logits).cpu().numpy())
                future_batches.append(torch.sigmoid(future_logits).cpu().numpy())
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

    native_release = switch_if_release[:, h0_index].astype(np.float64)
    native_press = switch_if_press[:, h0_index].astype(np.float64)

    paired_desired_press = desired_press_from_switch_pair(
        native_release,
        native_press,
    )
    paired_release, paired_press = project_desired_press_to_switch(
        paired_desired_press
    )

    future_desired_press = future_press[:, h0_index].astype(np.float64)
    future_release, future_press_switch = project_desired_press_to_switch(
        future_desired_press
    )

    indices_by_video: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_video[sample.video].append(index)

    labels_dir = args.labels_dir.resolve()
    label_data = {
        video: base.load_label_data(labels_dir, video)
        for video in indices_by_video
    }

    candidates = {
        "native_h0": (native_release, native_press),
        "paired_switch_projection": (paired_release, paired_press),
        "future_action_h0_projection": (
            future_release,
            future_press_switch,
        ),
    }
    summaries: dict[str, object] = {}
    for name, (release_values, press_values) in candidates.items():
        decisions, expert_states = _replay(
            samples=samples,
            indices_by_video=indices_by_video,
            release_h0=release_values,
            press_h0=press_values,
            threshold=args.threshold,
            min_state_hold_ms=args.min_state_hold_ms,
        )
        summaries[name] = base.replay_summary(
            decisions,
            expert_states,
            label_data,
            tolerance_ms=tolerance_ms,
        )

    native_conflicts = _conflict_summary(
        native_release,
        native_press,
        threshold=args.threshold,
    )

    print(
        f"split=validation samples={len(samples)} device={device} "
        f"threshold={args.threshold:.2f} "
        f"min_hold={args.min_state_hold_ms:g}ms",
        flush=True,
    )
    print(
        "native h0 state-consistency: "
        f"both_switch={native_conflicts['both_switch']} "
        f"({native_conflicts['both_switch_rate']:.4f}) "
        f"both_keep={native_conflicts['both_keep']} "
        f"({native_conflicts['both_keep_rate']:.4f}) "
        f"comp_err_mean={native_conflicts['complement_error_mean']:.3f} "
        f"p95={native_conflicts['complement_error_p95']:.3f} "
        f"p99={native_conflicts['complement_error_p99']:.3f}",
        flush=True,
    )
    for name in (
        "native_h0",
        "paired_switch_projection",
        "future_action_h0_projection",
    ):
        base.print_replay_summary(name, summaries[name])

    payload = {
        "split": "validation",
        "samples": len(samples),
        "model": str(model_path),
        "metadata": str(metadata_path),
        "threshold": args.threshold,
        "min_state_hold_ms": args.min_state_hold_ms,
        "transition_tolerance_ms": tolerance_ms,
        "native_h0_state_consistency": native_conflicts,
        "summaries": summaries,
        "test_evaluated": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    print("Frozen test split was not evaluated.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
