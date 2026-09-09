#!/usr/bin/env python3
"""Train the temporal karting classifier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.model.base import build_model
from karting_agent.train.frame_cache import frame_cache_root_from_config
from karting_agent.train.trainer import (
    load_samples,
    load_sampling_config,
    load_train_loop_config,
    load_video_split,
    make_weighted_sampler,
    partition_samples,
    run_epoch,
)
from karting_agent.train.video_dataset import TemporalVideoDataset
from karting_agent.vision.preprocess import preprocess_config_from_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the temporal karting model.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one train batch and one validation batch without writing artifacts.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable torchvision pretrained weights for this run.",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Fail instead of falling back to MP4 when a frame cache is missing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override train.num_workers for this machine.",
    )
    return parser.parse_args()


def load_raw_config(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("train config root must be a mapping")
    return raw


def select_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_summary(prefix: str, summary: dict[str, object]) -> None:
    all_metrics = summary["all"]
    transition = summary["transition"]
    short = summary["short_correction"]
    print(
        f"{prefix}: loss={summary['loss']:.4f}, "
        f"acc={all_metrics['accuracy']:.3f}, "
        f"f1={all_metrics['f1']:.3f}, "
        f"gt_press={all_metrics['target_positive_rate']:.3f}, "
        f"pred_press={all_metrics['predicted_positive_rate']:.3f}, "
        f"transition_f1={transition['f1']:.3f}, "
        f"short_f1={short['f1']:.3f}"
    )


def make_loader(
    DataLoader,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    sampler=None,
    shuffle: bool = False,
):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": shuffle,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def main() -> int:
    args = parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: python -m pip install -e ".[train]"'
        ) from exc

    raw = load_raw_config(args.config)
    loop_config = load_train_loop_config(args.config)
    sampling_config = load_sampling_config(args.config)
    split = load_video_split(args.config)
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)

    num_workers = loop_config.num_workers if args.num_workers is None else args.num_workers
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    samples = load_samples(ROOT / "data" / "processed" / "samples.jsonl")
    partitions = partition_samples(samples, split)

    random.seed(loop_config.seed)
    np.random.seed(loop_config.seed)
    torch.manual_seed(loop_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(loop_config.seed)

    device = select_device(torch)
    print(f"Device: {device}")
    print(
        f"Split {split.name}: train={len(partitions['train'])}, "
        f"validation={len(partitions['validation'])}, test={len(partitions['test'])}"
    )
    print(
        "Frame cache: "
        + (
            f"{cache_root} (required={args.require_cache})"
            if cache_root is not None
            else "disabled"
        )
    )
    print(f"DataLoader workers: {num_workers}")

    datasets = {
        name: TemporalVideoDataset(
            partition,
            project_root=ROOT,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=args.require_cache,
        )
        for name, partition in partitions.items()
    }

    generator = torch.Generator()
    generator.manual_seed(loop_config.seed)
    train_sampler = make_weighted_sampler(
        partitions["train"], sampling_config, generator=generator
    )
    pin_memory = device.type == "cuda"

    loaders = {
        "train": make_loader(
            DataLoader,
            datasets["train"],
            batch_size=loop_config.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
        ),
        "validation": make_loader(
            DataLoader,
            datasets["validation"],
            batch_size=loop_config.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": make_loader(
            DataLoader,
            datasets["test"],
            batch_size=loop_config.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    model_config = raw.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a mapping")
    architecture = str(model_config.get("architecture", "mobilenet_v3_small"))
    frame_stack = int(model_config.get("frame_stack", 3))
    pretrained = bool(model_config.get("pretrained", True)) and not args.no_pretrained

    model = build_model(
        architecture,
        frame_stack=frame_stack,
        pretrained=pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=loop_config.learning_rate,
        weight_decay=loop_config.weight_decay,
    )

    try:
        if args.smoke:
            train_summary = run_epoch(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                max_batches=1,
            )
            validation_summary = run_epoch(
                model,
                loaders["validation"],
                device,
                max_batches=1,
            )
            print_summary("smoke/train", train_summary)
            print_summary("smoke/validation", validation_summary)
            print("Smoke test passed.")
            return 0

        artifact_config = raw.get("artifact", {})
        if not isinstance(artifact_config, dict):
            raise ValueError("artifact config must be a mapping")
        artifact_name = str(
            artifact_config.get("name", f"{architecture}_{split.name}")
        )
        artifact_dir = ROOT / "artifacts" / "models" / artifact_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "model.pt"

        best_val_loss = float("inf")
        best_epoch = 0
        history: list[dict[str, object]] = []

        for epoch in range(1, loop_config.epochs + 1):
            train_summary = run_epoch(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
            )
            validation_summary = run_epoch(
                model,
                loaders["validation"],
                device,
            )

            print_summary(f"epoch {epoch:02d}/train", train_summary)
            print_summary(f"epoch {epoch:02d}/validation", validation_summary)
            history.append(
                {
                    "epoch": epoch,
                    "train": train_summary,
                    "validation": validation_summary,
                }
            )

            val_loss = float(validation_summary["loss"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)

        model.load_state_dict(torch.load(model_path, map_location=device))
        test_summary = run_epoch(model, loaders["test"], device)
        print_summary("test", test_summary)

        metadata = {
            "architecture": architecture,
            "frame_stack": frame_stack,
            "input_size": int(model_config.get("input_size", 224)),
            "pretrained": pretrained,
            "split": split.name,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "test": test_summary,
            "config": raw,
        }
        (artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        (artifact_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(f"Artifact: {artifact_dir}")
        return 0
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
