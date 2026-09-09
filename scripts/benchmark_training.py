#!/usr/bin/env python3
"""Benchmark temporal video loading plus one training step."""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.model.base import build_model
from karting_agent.train.trainer import (
    load_samples,
    load_sampling_config,
    load_train_loop_config,
    load_video_split,
    make_weighted_sampler,
    partition_samples,
)
from karting_agent.train.video_dataset import TemporalVideoDataset
from karting_agent.vision.preprocess import PreprocessConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark training throughput.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train.yaml",
    )
    parser.add_argument(
        "--workers",
        default="0,2,4",
        help="Comma-separated DataLoader worker counts to benchmark.",
    )
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    return parser.parse_args()


def load_raw_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("train config root must be a mapping")
    return raw


def build_preprocess_config(raw: dict) -> PreprocessConfig:
    model = raw.get("model", {})
    preprocess = raw.get("preprocess", {})
    size = int(model.get("input_size", 224))
    roi = tuple(
        float(value)
        for value in preprocess.get("touch_roi", (0.78, 0.82, 0.98, 0.98))
    )
    if len(roi) != 4:
        raise ValueError("preprocess.touch_roi must contain four values")
    return PreprocessConfig(
        input_width=size,
        input_height=size,
        mask_touch_area=bool(preprocess.get("mask_touch_area", True)),
        touch_roi=roi,
    )


def select_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_workers(value: str) -> tuple[int, ...]:
    workers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not workers or any(item < 0 for item in workers):
        raise ValueError("workers must be non-negative comma-separated integers")
    return workers


def synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_worker_count(
    torch,
    *,
    worker_count: int,
    batches: int,
    warmup: int,
    train_samples,
    sampling_config,
    loop_config,
    preprocess_config,
    raw_config: dict,
    device,
) -> dict[str, float]:
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    dataset = TemporalVideoDataset(
        train_samples,
        project_root=ROOT,
        preprocess_config=preprocess_config,
    )
    generator = torch.Generator().manual_seed(loop_config.seed)
    sampler = make_weighted_sampler(
        train_samples,
        sampling_config,
        generator=generator,
    )

    loader_kwargs = {
        "batch_size": loop_config.batch_size,
        "sampler": sampler,
        "num_workers": worker_count,
        "pin_memory": device.type == "cuda",
    }
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    model_cfg = raw_config.get("model", {})
    model = build_model(
        str(model_cfg.get("architecture", "mobilenet_v3_small")),
        frame_stack=int(model_cfg.get("frame_stack", 3)),
        pretrained=False,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=loop_config.learning_rate,
        weight_decay=loop_config.weight_decay,
    )
    model.train()

    measured_data = 0.0
    measured_step = 0.0
    measured_total = 0.0
    measured_samples = 0
    iterator = iter(loader)

    try:
        for batch_index in range(warmup + batches):
            cycle_start = time.perf_counter()
            batch = next(iterator)
            data_done = time.perf_counter()

            inputs = batch["input"].to(device=device, dtype=torch.float32)
            targets = batch["target"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            synchronize(torch, device)
            step_done = time.perf_counter()

            if batch_index >= warmup:
                measured_data += data_done - cycle_start
                measured_step += step_done - data_done
                measured_total += step_done - cycle_start
                measured_samples += int(targets.numel())
    finally:
        dataset.close()
        del iterator
        del loader
        del model
        del optimizer
        gc.collect()

    if measured_samples == 0:
        raise RuntimeError("benchmark produced no measured samples")

    measured_batches = float(batches)
    return {
        "data_ms_per_batch": 1000.0 * measured_data / measured_batches,
        "step_ms_per_batch": 1000.0 * measured_step / measured_batches,
        "total_ms_per_batch": 1000.0 * measured_total / measured_batches,
        "samples_per_second": measured_samples / measured_total,
    }


def main() -> int:
    args = parse_args()
    if args.batches < 1 or args.warmup < 0:
        raise ValueError("batches must be >= 1 and warmup must be >= 0")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: python -m pip install -e ".[train]"'
        ) from exc

    raw = load_raw_config(args.config)
    loop_config = load_train_loop_config(args.config)
    sampling_config = load_sampling_config(args.config)
    split = load_video_split(args.config)
    preprocess_config = build_preprocess_config(raw)
    samples = load_samples(ROOT / "data" / "processed" / "samples.jsonl")
    train_samples = partition_samples(samples, split)["train"]
    workers = parse_workers(args.workers)
    device = select_device(torch)

    epoch_batches = math.ceil(len(train_samples) / loop_config.batch_size)
    print(
        f"Device: {device}; train_samples={len(train_samples)}; "
        f"batch_size={loop_config.batch_size}; batches_per_epoch={epoch_batches}"
    )
    print(f"Benchmark: warmup={args.warmup}, measured_batches={args.batches}")

    results: list[tuple[int, dict[str, float]]] = []
    for worker_count in workers:
        result = benchmark_worker_count(
            torch,
            worker_count=worker_count,
            batches=args.batches,
            warmup=args.warmup,
            train_samples=train_samples,
            sampling_config=sampling_config,
            loop_config=loop_config,
            preprocess_config=preprocess_config,
            raw_config=raw,
            device=device,
        )
        results.append((worker_count, result))
        epoch_minutes = result["total_ms_per_batch"] * epoch_batches / 60_000.0
        print(
            f"workers={worker_count}: "
            f"data_wait={result['data_ms_per_batch']:.1f} ms/batch, "
            f"train_step={result['step_ms_per_batch']:.1f} ms/batch, "
            f"total={result['total_ms_per_batch']:.1f} ms/batch, "
            f"throughput={result['samples_per_second']:.1f} samples/s, "
            f"train_epoch≈{epoch_minutes:.1f} min"
        )

    best_workers, best = min(results, key=lambda item: item[1]["total_ms_per_batch"])
    train_epoch_minutes = best["total_ms_per_batch"] * epoch_batches / 60_000.0
    print(
        f"Recommended num_workers from this benchmark: {best_workers} "
        f"(train-only epoch≈{train_epoch_minutes:.1f} min; validation time not included)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
