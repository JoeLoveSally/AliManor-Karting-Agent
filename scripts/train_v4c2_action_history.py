#!/usr/bin/env python3
"""Train an OFFLINE-only factual action-history adapter on frozen V4-C2 weights.

The original V4-C2 model and its Armed runtime are never modified. This script
requires human-video touch timeline labels; recorded failed agent runs are NOT
used as behavioral-cloning ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from karting_agent.model.action_history import history_feature_width  # noqa: E402
from karting_agent.model.action_history_residual import ActionHistoryResidualPolicy  # noqa: E402
from karting_agent.model.state_conditioned_runner import StateConditionedModelRunner  # noqa: E402
from karting_agent.train.action_history_dataset import ActionHistoryVideoDataset  # noqa: E402
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402
from karting_agent.train.trainer import (  # noqa: E402
    build_sample_weights,
    load_sampling_config,
    load_video_split,
    partition_samples,
)
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=ROOT / "artifacts/models/mobilenet_v3_small_v4c2_temporal_v2/model.pt")
    parser.add_argument("--metadata", type=Path, default=ROOT / "artifacts/models/mobilenet_v3_small_v4c2_temporal_v2/metadata_h200.json")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/processed/v3/samples.jsonl")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "data/processed/v3/labels")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_v4c2.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/models/mobilenet_v3_small_v4c2_history_residual_v1")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-age-ms", type=float, default=500.0)
    parser.add_argument("--smoke", action="store_true", help="one train/validation batch; write no checkpoint")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def run_epoch(model, loader, device, *, control_index: int, optimizer=None, max_batches=None):
    import torch
    from torch.nn import functional as F

    model.train(optimizer is not None)
    totals = {"examples": 0, "control_loss": 0., "base_control_loss": 0.,
              "all_horizon_loss": 0., "transient_examples": 0,
              "transient_negatives": 0, "transient_fp": 0, "base_transient_fp": 0}
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images = batch["input"].to(device=device, dtype=torch.float32)
            states = batch["current_pressed"].to(device=device)
            history = batch["action_history"].to(device=device, dtype=torch.float32)
            targets = batch["switch_target"].to(device=device, dtype=torch.float32)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            corrected, baseline = model(images, states, history)
            loss = F.binary_cross_entropy_with_logits(corrected, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            bs = int(images.shape[0])
            loss_control = F.binary_cross_entropy_with_logits(
                corrected[:, control_index], targets[:, control_index]
            )
            base_control = F.binary_cross_entropy_with_logits(
                baseline[:, control_index], targets[:, control_index]
            )
            totals["examples"] += bs
            totals["control_loss"] += float(loss_control.detach()) * bs
            totals["base_control_loss"] += float(base_control.detach()) * bs
            totals["all_horizon_loss"] += float(loss.detach()) * bs
            # A known, recent transition at observation means age < 100 ms / max_age.
            age = history[:, -2]
            known = history[:, -1] > .5
            transient = known & (age < (100.0 / model.max_age_ms))
            negative = transient & (targets[:, control_index] < .5)
            predicted = torch.sigmoid(corrected[:, control_index]) >= .60
            baseline_pred = torch.sigmoid(baseline[:, control_index]) >= .60
            totals["transient_examples"] += int(transient.sum())
            totals["transient_negatives"] += int(negative.sum())
            totals["transient_fp"] += int((negative & predicted).sum())
            totals["base_transient_fp"] += int((negative & baseline_pred).sum())
    if not totals["examples"]:
        raise ValueError("empty data loader")
    result = dict(totals)
    for metric in ("control_loss", "base_control_loss", "all_horizon_loss"):
        result[metric] /= totals["examples"]
    return result


def main() -> int:
    args = parse_args()
    import torch
    import yaml
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if (args.epochs < 1 or args.patience < 0 or args.batch_size < 1
            or args.num_workers < 0 or args.learning_rate <= 0 or args.max_age_ms <= 0):
        raise ValueError("invalid training arguments")
    base_path = args.base_model.resolve()
    meta_path = args.metadata.resolve()
    out = args.output_dir.resolve()
    if out == base_path.parent or out == (ROOT / "artifacts/models/mobilenet_v3_small_v4c2_temporal_v2").resolve():
        raise ValueError("refuse to save experiment into baseline model directory")
    if not args.smoke and (out / "history_adapter.pt").exists():
        raise FileExistsError(f"experimental adapter already exists: {out}")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("training config must be a mapping")
    print("INFO: factual-only action history: counterfactual state augmentation disabled")
    seed = int(raw.get("train", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base_runner = StateConditionedModelRunner(base_path, metadata_path=meta_path, device=args.device)
    if base_runner.model_family != "state_conditioned_kart_relative_v4c2":
        raise ValueError("this experiment requires the V4-C2 checkpoint family")
    if base_runner.spec.frame_stack != 5:
        raise ValueError("this experiment currently requires five input frames")
    expected_horizons = tuple(float(x) for x in raw["dataset"]["prediction_horizons_ms"])
    if tuple(base_runner.spec.prediction_horizons_ms) != expected_horizons:
        raise ValueError("dataset horizon list disagrees with loaded checkpoint")
    preprocess = preprocess_config_from_mapping(raw)
    if preprocess != base_runner.spec.preprocess_config:
        raise ValueError("training preprocess differs from deployed V4-C2 metadata")

    samples = load_v3_samples(args.samples.resolve())
    split = load_video_split(args.config.resolve())
    partitions = partition_samples(samples, split)
    cache_root = frame_cache_root_from_config(raw, ROOT)
    datasets = {}
    try:
        for name in ("train", "validation", "test"):
            datasets[name] = ActionHistoryVideoDataset(
                partitions[name], project_root=ROOT, labels_dir=args.labels_dir.resolve(),
                preprocess_config=preprocess, cache_root=cache_root,
                max_age_ms=args.max_age_ms,
            )
        sampling_config = load_sampling_config(args.config.resolve())
        weights = build_sample_weights(partitions["train"], sampling_config)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=sampling_config.replacement,
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(datasets["train"], batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers)
        val_loader = DataLoader(datasets["validation"], batch_size=args.batch_size,
                                num_workers=args.num_workers)
        test_loader = DataLoader(datasets["test"], batch_size=args.batch_size,
                                 num_workers=args.num_workers)
        model = ActionHistoryResidualPolicy(
            base_runner.model, history_width=history_feature_width(5)
        ).to(base_runner.device)
        model.max_age_ms = float(args.max_age_ms)
        optimizer = torch.optim.AdamW(model.residual.parameters(), lr=args.learning_rate)
        control_index = base_runner.spec.control_output_index
        print(f"Base checkpoint (frozen): {base_path}; model={base_runner.model_family}; "
              f"control H{base_runner.spec.prediction_horizon_ms:g}; "
              f"train/validation/test={len(datasets['train'])}/{len(datasets['validation'])}/{len(datasets['test'])}")
        best_loss = float("inf")
        best_adapter = None
        for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
            train = run_epoch(model, train_loader, base_runner.device, control_index=control_index,
                              optimizer=optimizer, max_batches=1 if args.smoke else None)
            validation = run_epoch(model, val_loader, base_runner.device, control_index=control_index,
                                   max_batches=1 if args.smoke else None)
            print(json.dumps({"epoch": epoch, "train": train, "validation": validation},
                             ensure_ascii=False), flush=True)
            if validation["control_loss"] < best_loss:
                best_loss = validation["control_loss"]
                best_adapter = {name: value.detach().cpu().clone()
                                for name, value in model.residual.state_dict().items()}
                no_improvement = 0
            else:
                no_improvement += 1
            if not args.smoke and args.patience and no_improvement >= args.patience:
                break
        if args.smoke:
            print("SMOKE PASSED: model forward/backward and factual data loading; no weights saved")
            return 0
        assert best_adapter is not None
        model.residual.load_state_dict(best_adapter)
        report = {
            "validation": run_epoch(model, val_loader, base_runner.device, control_index=control_index),
            "test": run_epoch(model, test_loader, base_runner.device, control_index=control_index),
        }
        out.mkdir(parents=True, exist_ok=True)
        torch.save(best_adapter, out / "history_adapter.pt")
        manifest = {
            "model_family": "experimental_v4c2_action_history_residual_v1",
            "deployment_ready": False,
            "base_model_path": str(base_path),
            "base_model_sha256": sha256_file(base_path),
            "base_metadata_path": str(meta_path),
            "base_metadata_sha256": sha256_file(meta_path),
            "max_age_ms": args.max_age_ms,
            "history_feature_width": history_feature_width(5),
            "frame_stack": 5,
            "prediction_horizons_ms": list(base_runner.spec.prediction_horizons_ms),
            "control_horizon_ms": base_runner.spec.prediction_horizon_ms,
            "factual_only": True,
            "train_samples": str(args.samples.resolve()),
            "labels_dir": str(args.labels_dir.resolve()),
            "video_split": {name: list(getattr(split, name)) for name in ("train", "validation", "test")},
            "result": report,
            "limitations": "Offline video imitation only; runtime command-to-pixel latency unmeasured; no Armed validation.",
        }
        (out / "history_adapter_metadata.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"artifact": str(out / "history_adapter.pt"), "report": report},
                         ensure_ascii=False, indent=2))
        return 0
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
