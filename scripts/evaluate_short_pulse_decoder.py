#!/usr/bin/env python3
"""Evaluate whether multi-horizon heads reliably encode short action pulses.

This is an expert-conditioned diagnostic, not a closed-loop controller replay.
It asks whether the learned ``H100 high -> H200 low -> H300 low`` shape aligns
with the recorded future action returning to its current state by H200/H300.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from karting_agent.runtime.short_pulse_decoder import (  # noqa: E402
    PulseMode,
    ShortPulseCandidate,
    decode_short_pulse,
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


MODES: tuple[PulseMode, ...] = ("strict_min_hold", "reject_too_short")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate expert-conditioned short-pulse decoding on held-out data."
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
    parser.add_argument("--short-horizon-ms", type=float, default=100.0)
    parser.add_argument("--control-horizon-ms", type=float, default=200.0)
    parser.add_argument("--anticipation-horizon-ms", type=float, default=300.0)
    parser.add_argument("--min-pulse-width-ms", type=float, default=100.0)
    return parser.parse_args()


def load_yaml_mapping(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def artifact_dir(raw: dict[str, object]) -> Path:
    artifact = raw.get("artifact", {})
    if not isinstance(artifact, dict):
        raise ValueError("artifact config must be a mapping")
    return ROOT / "artifacts" / "models" / str(artifact["name"])


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


def load_action_events(labels_dir: Path, video: str) -> list[tuple[float, bool]]:
    path = labels_dir / f"{Path(video).stem}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"label file has no events: {path}")
    return [
        (float(raw["timestamp_ms"]), bool(raw["pressed"])) for raw in raw_events
    ]


def resolved_ground_truth_timing(
    *,
    events: list[tuple[float, bool]],
    observation_ms: float,
    current_pressed: bool,
    short_horizon_ms: float,
    control_horizon_ms: float,
) -> tuple[float, float] | None:
    """Return absolute away/back transition times for a horizon-confirmed pulse."""

    away_deadline = observation_ms + short_horizon_ms
    back_deadline = observation_ms + control_horizon_ms
    away_at: float | None = None
    for timestamp_ms, pressed in events:
        if timestamp_ms <= observation_ms + 1e-6:
            continue
        if away_at is None:
            if timestamp_ms > away_deadline + 1e-6:
                return None
            if pressed != current_pressed:
                away_at = timestamp_ms
            continue
        if timestamp_ms > back_deadline + 1e-6:
            return None
        if pressed == current_pressed:
            return away_at, timestamp_ms
    return None


def binary_summary(*, predicted: np.ndarray, target: np.ndarray) -> dict[str, object]:
    predicted = predicted.astype(np.bool_, copy=False)
    target = target.astype(np.bool_, copy=False)
    tp = int(np.logical_and(predicted, target).sum())
    fp = int(np.logical_and(predicted, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(predicted), target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted": int(predicted.sum()),
        "ground_truth": int(target.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def mae(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.abs(array).mean()) if array.size else 0.0


def print_mode(name: str, summary: dict[str, object]) -> None:
    classification = summary["classification"]
    widths = summary["candidate_width_ms"]
    timing = summary["timing"]
    event_recall = summary["unique_event_recall"]
    print(
        f"{name}: candidates={classification['predicted']} "
        f"tp={classification['tp']} fp={classification['fp']} "
        f"gt={classification['ground_truth']} "
        f"precision={classification['precision']:.3f} "
        f"recall={classification['recall']:.3f} "
        f"f1={classification['f1']:.3f} "
        f"event_recall={event_recall['detected']}/{event_recall['ground_truth']} "
        f"({event_recall['recall']:.3f})",
        flush=True,
    )
    print(
        f"  width raw_mean={widths['raw']['mean']:.1f}ms "
        f"raw_p50={widths['raw']['p50']:.1f}ms "
        f"raw_p95={widths['raw']['p95']:.1f}ms "
        f"effective_mean={widths['effective']['mean']:.1f}ms",
        flush=True,
    )
    print(
        f"  timing n={timing['count']} start_mae={timing['start_mae_ms']:.1f}ms "
        f"end_mae={timing['end_mae_ms']:.1f}ms "
        f"width_mae={timing['width_mae_ms']:.1f}ms",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    if args.min_pulse_width_ms < 0.0:
        raise ValueError("--min-pulse-width-ms must be >= 0")
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
    artifact = artifact_dir(raw)
    model_path = args.model.resolve() if args.model else artifact / "model.pt"
    metadata_path = args.metadata.resolve() if args.metadata else artifact / "metadata.json"
    output_path = (
        args.output.resolve()
        if args.output
        else artifact / "evaluation" / f"{args.split}_short_pulse_decoder.json"
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("model_family", "")) != "state_conditioned_kart_relative_v4c2":
        raise ValueError("pulse evaluator expects state_conditioned_kart_relative_v4c2")
    horizons = tuple(float(value) for value in metadata["prediction_horizons_ms"])
    for value in (
        args.short_horizon_ms,
        args.control_horizon_ms,
        args.anticipation_horizon_ms,
    ):
        if float(value) not in horizons:
            raise ValueError(f"requested horizon {value:g}ms is not present in metadata")
    short_index = horizons.index(float(args.short_horizon_ms))
    control_index = horizons.index(float(args.control_horizon_ms))
    anticipation_index = horizons.index(float(args.anticipation_horizon_ms))

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

    prediction_batches: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["input"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                states = batch["current_pressed"].to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=device.type == "cuda",
                )
                features = model.encode_visual(inputs)
                logits = model.forward_from_features(features, states)[0]
                prediction_batches.append(torch.sigmoid(logits).cpu().numpy())
    finally:
        dataset.close()

    predictions = np.concatenate(prediction_batches, axis=0)
    if predictions.shape != (len(samples), len(horizons)):
        raise RuntimeError("switch prediction shape mismatch")

    labels_dir = args.labels_dir.resolve()
    events_by_video = {
        video: load_action_events(labels_dir, video)
        for video in sorted({sample.video for sample in samples})
    }

    gt_shape = np.zeros(len(samples), dtype=np.bool_)
    gt_timing: list[tuple[float, float] | None] = [None] * len(samples)
    gt_event_keys: list[tuple[str, float, float] | None] = [None] * len(samples)
    for index, sample in enumerate(samples):
        current_pressed = sample.current_pressed
        if current_pressed is None:
            raise ValueError("pulse evaluator requires current_pressed")
        targets = sample.target_pressed_by_horizon
        if len(targets) != len(horizons):
            raise ValueError("pulse evaluator requires per-horizon future action targets")
        switches = tuple(bool(value) != bool(current_pressed) for value in targets)
        is_pulse = (
            switches[short_index]
            and not switches[control_index]
            and not switches[anticipation_index]
        )
        gt_shape[index] = is_pulse
        if not is_pulse:
            continue
        observation_ms = float(sample.input_timestamps_ms[-1])
        timing = resolved_ground_truth_timing(
            events=events_by_video[sample.video],
            observation_ms=observation_ms,
            current_pressed=bool(current_pressed),
            short_horizon_ms=float(args.short_horizon_ms),
            control_horizon_ms=float(args.control_horizon_ms),
        )
        gt_timing[index] = timing
        if timing is not None:
            gt_event_keys[index] = (sample.video, timing[0], timing[1])

    mode_summaries: dict[str, object] = {}
    examples: list[dict[str, object]] = []
    all_gt_events = {key for key in gt_event_keys if key is not None}
    for mode in MODES:
        predicted = np.zeros(len(samples), dtype=np.bool_)
        candidates: list[ShortPulseCandidate | None] = [None] * len(samples)
        detected_gt_events: set[tuple[str, float, float]] = set()
        raw_widths: list[float] = []
        effective_widths: list[float] = []
        start_errors: list[float] = []
        end_errors: list[float] = []
        width_errors: list[float] = []

        for index, (sample, probabilities) in enumerate(zip(samples, predictions)):
            candidate = decode_short_pulse(
                horizons_ms=horizons,
                probabilities=probabilities,
                short_horizon_ms=float(args.short_horizon_ms),
                control_horizon_ms=float(args.control_horizon_ms),
                anticipation_horizon_ms=float(args.anticipation_horizon_ms),
                threshold=float(args.threshold),
                min_pulse_width_ms=float(args.min_pulse_width_ms),
                mode=mode,
            )
            candidates[index] = candidate
            if candidate is None:
                continue
            predicted[index] = True
            raw_widths.append(candidate.raw_width_ms)
            effective_widths.append(candidate.effective_width_ms)
            if not gt_shape[index] or gt_timing[index] is None:
                continue

            observation_ms = float(sample.input_timestamps_ms[-1])
            true_start, true_end = gt_timing[index]
            predicted_start = observation_ms + candidate.start_horizon_ms
            predicted_end = observation_ms + candidate.effective_end_horizon_ms
            start_errors.append(predicted_start - true_start)
            end_errors.append(predicted_end - true_end)
            predicted_width = candidate.effective_width_ms
            true_width = true_end - true_start
            width_errors.append(predicted_width - true_width)
            event_key = gt_event_keys[index]
            if event_key is not None:
                detected_gt_events.add(event_key)

        classification = binary_summary(predicted=predicted, target=gt_shape)
        event_count = len(all_gt_events)
        detected_count = len(detected_gt_events)
        summary = {
            "classification": classification,
            "candidate_width_ms": {
                "raw": distribution(raw_widths),
                "effective": distribution(effective_widths),
            },
            "timing": {
                "count": len(start_errors),
                "start_mae_ms": mae(start_errors),
                "end_mae_ms": mae(end_errors),
                "width_mae_ms": mae(width_errors),
                "start_error_ms": distribution(start_errors),
                "end_error_ms": distribution(end_errors),
                "width_error_ms": distribution(width_errors),
            },
            "unique_event_recall": {
                "ground_truth": event_count,
                "detected": detected_count,
                "recall": detected_count / event_count if event_count else 0.0,
            },
        }
        mode_summaries[mode] = summary

        if mode == "strict_min_hold":
            for index, candidate in enumerate(candidates):
                if candidate is None or len(examples) >= 30:
                    continue
                sample = samples[index]
                timing = gt_timing[index]
                examples.append(
                    {
                        "video": sample.video,
                        "observation_timestamp_ms": float(sample.input_timestamps_ms[-1]),
                        "probabilities": [float(value) for value in predictions[index]],
                        "ground_truth_shape": bool(gt_shape[index]),
                        "ground_truth_timing_ms": (
                            None
                            if timing is None
                            else {
                                "start": timing[0],
                                "end": timing[1],
                                "width": timing[1] - timing[0],
                            }
                        ),
                        "decoded": asdict(candidate),
                    }
                )

    resolved_gt_widths = [
        timing[1] - timing[0] for timing in gt_timing if timing is not None
    ]
    print(
        f"split={args.split} samples={len(samples)} device={device} "
        f"threshold={args.threshold:.2f} horizons="
        f"{args.short_horizon_ms:g}/{args.control_horizon_ms:g}/"
        f"{args.anticipation_horizon_ms:g}ms min_width={args.min_pulse_width_ms:g}ms",
        flush=True,
    )
    print(
        f"ground truth: shape_samples={int(gt_shape.sum())} "
        f"timing_resolved={len(resolved_gt_widths)} unique_events={len(all_gt_events)} "
        f"width_mean={distribution(resolved_gt_widths)['mean']:.1f}ms",
        flush=True,
    )
    for mode in MODES:
        print_mode(mode, mode_summaries[mode])

    payload = {
        "split": args.split,
        "samples": len(samples),
        "model": str(model_path),
        "metadata": str(metadata_path),
        "decoder": {
            "horizons_ms": list(horizons),
            "short_horizon_ms": float(args.short_horizon_ms),
            "control_horizon_ms": float(args.control_horizon_ms),
            "anticipation_horizon_ms": float(args.anticipation_horizon_ms),
            "threshold": float(args.threshold),
            "min_pulse_width_ms": float(args.min_pulse_width_ms),
            "implicit_h0_probability": 0.0,
        },
        "ground_truth": {
            "shape_samples": int(gt_shape.sum()),
            "timing_resolved_samples": len(resolved_gt_widths),
            "unique_events": len(all_gt_events),
            "resolved_width_ms": distribution(resolved_gt_widths),
        },
        "modes": mode_summaries,
        "examples": examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
