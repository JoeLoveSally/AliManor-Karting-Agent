#!/usr/bin/env python3
"""Compare native switch-head replay with future-action-derived switch replay.

This is an offline validation diagnostic. It keeps the scheduler semantics fixed
and changes only the probability source:

* native: the learned state-conditioned KEEP/SWITCH head;
* future_action: the visual-only absolute future-action head projected into
  P(future action != current physical state).

The projection enforces exact complement symmetry between RELEASE and PRESS and
requires no model retraining.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_multi_horizon_scheduler import (  # noqa: E402
    artifact_dir,
    evaluation_tolerance,
    load_label_data,
    load_yaml_mapping,
    print_replay_summary,
    replay_summary,
    scheduler_diagnostics,
    select_device,
    simulate_fixed_horizon,
    simulate_scheduler,
)
from karting_agent.model.kart_relative_supervised import (  # noqa: E402
    build_kart_relative_model,
)
from karting_agent.runtime.multi_horizon_scheduler import (  # noqa: E402
    MultiHorizonSchedulerConfig,
)
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
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
        description=(
            "Compare native switch-head scheduling with switch probabilities "
            "derived from the absolute future-action head."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_v4c2_temporal_v2.yaml",
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
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--control-horizon-ms", type=float, default=200.0)
    parser.add_argument("--anticipation-horizon-ms", type=float, default=300.0)
    parser.add_argument("--min-state-hold-ms", type=float, default=100.0)
    parser.add_argument("--pending-advance-ms", type=float, default=0.0)
    parser.add_argument("--arm-pending-during-min-hold", action="store_true")
    parser.add_argument("--tolerance-ms", type=float, default=None)
    return parser.parse_args()


def summarize_source(
    *,
    source_name: str,
    samples,
    indices_by_video: dict[str, list[int]],
    label_data,
    switch_if_release: np.ndarray,
    switch_if_press: np.ndarray,
    scheduler_config: MultiHorizonSchedulerConfig,
    tolerance_ms: float,
) -> dict[str, object]:
    control_index = scheduler_config.horizons_ms.index(
        scheduler_config.control_horizon_ms
    )
    baseline_by_video = {}
    scheduler_by_video = {}
    for video, indices in indices_by_video.items():
        baseline_by_video[video] = simulate_fixed_horizon(
            video,
            samples,
            indices,
            switch_if_release,
            switch_if_press,
            horizon_index=control_index,
            threshold=scheduler_config.threshold,
        )
        scheduler_by_video[video] = simulate_scheduler(
            video,
            samples,
            indices,
            switch_if_release,
            switch_if_press,
            scheduler_config=scheduler_config,
        )

    baseline = replay_summary(
        baseline_by_video,
        label_data,
        control_horizon_ms=scheduler_config.control_horizon_ms,
        tolerance_ms=tolerance_ms,
    )
    scheduler = replay_summary(
        scheduler_by_video,
        label_data,
        control_horizon_ms=scheduler_config.control_horizon_ms,
        tolerance_ms=tolerance_ms,
    )
    diagnostics = scheduler_diagnostics(scheduler_by_video)

    print(f"[{source_name}]", flush=True)
    print_replay_summary(
        f"baseline_h{scheduler_config.control_horizon_ms:g}", baseline
    )
    print_replay_summary("scheduler", scheduler)
    print(
        "scheduler events: "
        + ", ".join(
            f"{key}={value}" for key, value in diagnostics["reasons"].items()
        ),
        flush=True,
    )
    delays = diagnostics["armed_delay_ms"]
    print(
        f"armed delay: n={delays['count']} mean={delays['mean']:.1f}ms "
        f"p50={delays['p50']:.1f}ms p95={delays['p95']:.1f}ms "
        f"range={delays['min']:.1f}..{delays['max']:.1f}ms",
        flush=True,
    )
    return {
        "baseline": baseline,
        "scheduler": scheduler,
        "diagnostics": diagnostics,
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

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
    artifact = artifact_dir(raw)
    model_path = args.model.resolve() if args.model else artifact / "model.pt"
    metadata_path = args.metadata.resolve() if args.metadata else artifact / "metadata.json"
    output_path = (
        args.output.resolve()
        if args.output
        else artifact / "evaluation" / f"{args.split}_future_action_scheduler.json"
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("model_family", "")) != "state_conditioned_kart_relative_v4c2":
        raise ValueError("future-action scheduler evaluator expects v4-C2 model")
    horizons = tuple(float(value) for value in metadata["prediction_horizons_ms"])
    scheduler_config = MultiHorizonSchedulerConfig(
        horizons_ms=horizons,
        control_horizon_ms=float(args.control_horizon_ms),
        anticipation_horizon_ms=float(args.anticipation_horizon_ms),
        threshold=float(args.threshold),
        min_state_hold_ms=float(args.min_state_hold_ms),
        pending_advance_ms=float(args.pending_advance_ms),
        arm_pending_during_min_hold=bool(args.arm_pending_during_min_hold),
        execute_short_horizon_overdue=False,
    )
    scheduler_config.validate()

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
    samples = partition_samples(all_samples, split_config)[args.split]
    dataset = StateConditionedVideoDataset(
        samples,
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

    native_release_batches: list[np.ndarray] = []
    native_press_batches: list[np.ndarray] = []
    future_action_batches: list[np.ndarray] = []
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
                release_states = torch.zeros(batch_size, device=device, dtype=torch.long)
                press_states = torch.ones(batch_size, device=device, dtype=torch.long)
                release_outputs = model.forward_from_features(features, release_states)
                press_outputs = model.forward_from_features(features, press_states)
                native_release_batches.append(
                    torch.sigmoid(release_outputs[0]).cpu().numpy()
                )
                native_press_batches.append(
                    torch.sigmoid(press_outputs[0]).cpu().numpy()
                )
                future_action_batches.append(
                    torch.sigmoid(release_outputs[1]).cpu().numpy()
                )
    finally:
        dataset.close()

    native_release = np.concatenate(native_release_batches, axis=0)
    native_press = np.concatenate(native_press_batches, axis=0)
    future_action = np.concatenate(future_action_batches, axis=0)
    expected_shape = (len(samples), len(horizons))
    for name, values in (
        ("native_release", native_release),
        ("native_press", native_press),
        ("future_action", future_action),
    ):
        if values.shape != expected_shape:
            raise RuntimeError(f"{name} prediction shape mismatch: {values.shape}")

    projected_release = future_action
    projected_press = 1.0 - future_action
    complement_error = float(
        np.max(np.abs(projected_release + projected_press - 1.0))
    )

    indices_by_video: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        indices_by_video.setdefault(sample.video, []).append(index)
    labels_dir = args.labels_dir.resolve()
    label_data = {
        video: load_label_data(labels_dir, video) for video in indices_by_video
    }

    print(
        f"split={args.split} samples={len(samples)} device={device} "
        f"threshold={args.threshold:.2f} "
        f"control_h={scheduler_config.control_horizon_ms:g}ms "
        f"anticipation_h={scheduler_config.anticipation_horizon_ms:g}ms "
        f"min_hold={scheduler_config.min_state_hold_ms:g}ms "
        f"pending_advance={scheduler_config.pending_advance_ms:g}ms "
        f"hold_arm={scheduler_config.arm_pending_during_min_hold}",
        flush=True,
    )
    print(
        f"future-action projection complement_error={complement_error:.3e}",
        flush=True,
    )

    native_summary = summarize_source(
        source_name="native_switch_head",
        samples=samples,
        indices_by_video=indices_by_video,
        label_data=label_data,
        switch_if_release=native_release,
        switch_if_press=native_press,
        scheduler_config=scheduler_config,
        tolerance_ms=tolerance_ms,
    )
    projected_summary = summarize_source(
        source_name="future_action_projection",
        samples=samples,
        indices_by_video=indices_by_video,
        label_data=label_data,
        switch_if_release=projected_release,
        switch_if_press=projected_press,
        scheduler_config=scheduler_config,
        tolerance_ms=tolerance_ms,
    )

    payload = {
        "split": args.split,
        "samples": len(samples),
        "model": str(model_path),
        "metadata": str(metadata_path),
        "scheduler_config": asdict(scheduler_config),
        "transition_tolerance_ms": tolerance_ms,
        "future_action_projection": {
            "complement_error": complement_error,
            "semantics": "P(switch@h)=P(action@h) for RELEASE; 1-P(action@h) for PRESS",
        },
        "native_switch_head": native_summary,
        "projected_future_action_head": projected_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
