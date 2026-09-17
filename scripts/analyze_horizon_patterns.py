#!/usr/bin/env python3
"""Diagnose ground-truth and learned multi-horizon KEEP/SWITCH patterns.

This script is deliberately diagnostic only. It does not replay a controller or
change runtime semantics. It is intended to answer whether rare non-monotonic
patterns such as ``100`` are represented often enough in training and whether the
current model predicts them reliably on held-out expert-conditioned samples.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.horizon_pattern_analysis import (  # noqa: E402
    confusion_counts,
    one_vs_rest_metrics,
    pattern_counts,
    per_horizon_metrics,
    switch_pattern,
    threshold_pattern,
    weighted_pattern_counts,
)
from karting_agent.train.state_conditioned_dataset import (  # noqa: E402
    StateConditionedVideoDataset,
    load_v3_samples,
)
from karting_agent.train.trainer import (  # noqa: E402
    build_sample_weights,
    load_sampling_config,
    load_train_loop_config,
    load_video_split,
    partition_samples,
)
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze discrete multi-horizon switch-pattern learning."
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
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--ground-truth-only", action="store_true")
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


def complement_pattern(pattern: str) -> str:
    return "".join("0" if bit == "1" else "1" for bit in pattern)


def shares(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {key: value / total if total else 0.0 for key, value in counts.items()}


def weighted_shares(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    return {key: value / total if total else 0.0 for key, value in counts.items()}


def print_distribution(title: str, counts: dict[str, int], weighted=None) -> None:
    raw_share = shares(counts)
    weighted_share = weighted_shares(weighted) if weighted is not None else {}
    print(title, flush=True)
    for pattern, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        suffix = ""
        if weighted is not None:
            suffix = f" weighted_share={weighted_share.get(pattern, 0.0):.4f}"
        print(
            f"  {pattern}: n={count} share={raw_share[pattern]:.4f}{suffix}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    config_path = args.config.resolve()
    raw = load_yaml_mapping(config_path)
    split_config = load_video_split(config_path)
    sampling_config = load_sampling_config(config_path)
    loop_config = load_train_loop_config(config_path)
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)
    artifact = artifact_dir(raw)
    model_path = args.model.resolve() if args.model else artifact / "model.pt"
    metadata_path = args.metadata.resolve() if args.metadata else artifact / "metadata.json"
    output_path = (
        args.output.resolve()
        if args.output
        else artifact / "evaluation" / f"{args.split}_horizon_patterns.json"
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    horizons = tuple(float(value) for value in metadata["prediction_horizons_ms"])
    if not horizons:
        raise ValueError("metadata prediction_horizons_ms must not be empty")

    all_samples = load_v3_samples(args.samples.resolve())
    samples = partition_samples(all_samples, split_config)[args.split]
    if not samples:
        raise ValueError(f"{args.split} split is empty")

    expert_patterns: list[str] = []
    for sample in samples:
        if sample.current_pressed is None:
            raise ValueError("pattern analysis requires current_pressed")
        targets = sample.target_pressed_by_horizon
        if len(targets) != len(horizons):
            raise ValueError("per-horizon future-action width does not match metadata")
        expert_patterns.append(switch_pattern(bool(sample.current_pressed), targets))

    sample_weights = build_sample_weights(samples, sampling_config)
    expert_counts = pattern_counts(expert_patterns)
    expert_weighted = weighted_pattern_counts(expert_patterns, sample_weights)
    near_short_count = sum(sample.near_short_correction for sample in samples)
    near_transition_count = sum(sample.near_transition for sample in samples)

    dataset_config = raw.get("dataset", {})
    if not isinstance(dataset_config, dict):
        raise ValueError("dataset config must be a mapping")
    counterfactual_train = bool(dataset_config.get("counterfactual_train_states", True))
    train_conditioned_patterns: list[str] = []
    train_conditioned_weights: list[float] = []
    if args.split == "train" and counterfactual_train:
        for pattern, weight in zip(expert_patterns, sample_weights, strict=True):
            train_conditioned_patterns.extend((pattern, complement_pattern(pattern)))
            train_conditioned_weights.extend((weight, weight))
    else:
        train_conditioned_patterns = list(expert_patterns)
        train_conditioned_weights = list(sample_weights)
    conditioned_counts = pattern_counts(train_conditioned_patterns)
    conditioned_weighted = weighted_pattern_counts(
        train_conditioned_patterns,
        train_conditioned_weights,
    )

    print(
        f"split={args.split} samples={len(samples)} horizons="
        + "/".join(f"{value:g}" for value in horizons)
        + f"ms threshold={args.threshold:.2f} gt_only={args.ground_truth_only}",
        flush=True,
    )
    print(
        f"labels: near_transition={near_transition_count} "
        f"near_short_correction={near_short_count} "
        f"counterfactual_train={counterfactual_train}",
        flush=True,
    )
    print_distribution("expert-state ground truth patterns:", expert_counts, expert_weighted)
    if args.split == "train" and counterfactual_train:
        print_distribution(
            "actual train conditioned patterns (recorded + counterfactual):",
            conditioned_counts,
            conditioned_weighted,
        )

    payload: dict[str, object] = {
        "split": args.split,
        "samples": len(samples),
        "horizons_ms": horizons,
        "threshold": args.threshold,
        "near_transition_samples": near_transition_count,
        "near_short_correction_samples": near_short_count,
        "counterfactual_train_states": counterfactual_train,
        "expert_ground_truth": {
            "counts": expert_counts,
            "raw_share": shares(expert_counts),
            "weighted_count": expert_weighted,
            "weighted_share": weighted_shares(expert_weighted),
        },
        "train_conditioned_ground_truth": {
            "counts": conditioned_counts,
            "raw_share": shares(conditioned_counts),
            "weighted_count": conditioned_weighted,
            "weighted_share": weighted_shares(conditioned_weighted),
        },
    }

    if not args.ground_truth_only:
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise RuntimeError(
                'PyTorch is required; install with: python -m pip install -e ".[train]"'
            ) from exc

        if str(metadata.get("model_family", "")) != "state_conditioned_kart_relative_v4c2":
            raise ValueError("model analysis expects state_conditioned_kart_relative_v4c2")
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

        probability_batches: list[np.ndarray] = []
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
                    probability_batches.append(torch.sigmoid(logits).cpu().numpy())
        finally:
            dataset.close()

        probabilities = np.concatenate(probability_batches, axis=0)
        if probabilities.shape != (len(samples), len(horizons)):
            raise RuntimeError("switch prediction shape mismatch")
        predicted_patterns = [
            threshold_pattern(row, args.threshold) for row in probabilities
        ]
        predicted_counts = pattern_counts(predicted_patterns)
        confusion = confusion_counts(expert_patterns, predicted_patterns)
        horizon_metrics = per_horizon_metrics(expert_patterns, predicted_patterns)
        pattern_metrics = {
            pattern: one_vs_rest_metrics(expert_patterns, predicted_patterns, pattern)
            for pattern in sorted(set(expert_patterns) | set(predicted_patterns))
        }

        print_distribution("predicted patterns:", predicted_counts)
        print("per-horizon switch metrics:", flush=True)
        for horizon, metrics in zip(horizons, horizon_metrics, strict=True):
            print(
                f"  h{horizon:g}: p={metrics['precision']:.3f} "
                f"r={metrics['recall']:.3f} f1={metrics['f1']:.3f}",
                flush=True,
            )
        print("pattern one-vs-rest metrics:", flush=True)
        for pattern, metrics in sorted(pattern_metrics.items()):
            if expert_counts.get(pattern, 0) or predicted_counts.get(pattern, 0):
                print(
                    f"  {pattern}: gt={expert_counts.get(pattern, 0)} "
                    f"pred={predicted_counts.get(pattern, 0)} "
                    f"p={metrics['precision']:.3f} r={metrics['recall']:.3f} "
                    f"f1={metrics['f1']:.3f}",
                    flush=True,
                )
        print("confusion for non-monotonic target patterns:", flush=True)
        for truth, row in sorted(confusion.items()):
            transitions = sum(left != right for left, right in zip(truth, truth[1:]))
            if transitions == 0:
                continue
            top = Counter(row).most_common(5)
            print(
                f"  {truth}: " + ", ".join(f"{pred}={count}" for pred, count in top),
                flush=True,
            )

        payload["model"] = {
            "path": str(model_path),
            "device": str(device),
            "predicted_counts": predicted_counts,
            "confusion": confusion,
            "per_horizon": horizon_metrics,
            "per_pattern": pattern_metrics,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
