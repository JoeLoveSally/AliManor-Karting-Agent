#!/usr/bin/env python3
"""Train matched Tiny RGB control-feedback ablations; expert videos only.

Full / vision_only / state_only reuse the SAME RGB samples, class sampler,
CNN+GRU parameterization and causal future PRESS label. No held-out test,
Android access, policy deployment, or checkpoint overwrite is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiment_tiny_temporal_policy import (  # noqa: E402
    TemporalExpertDataset, cache_required_frames, evaluate_sequences,
    label_events, run_epoch, validate_sample,
)
from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from karting_agent.model.tiny_control_ablation import (  # noqa: E402
    CONTROL_MASKS, TinyControlAblation,
)
from karting_agent.train.gpu_budget import select_training_device  # noqa: E402
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402
from karting_agent.train.trainer import (  # noqa: E402
    build_sample_weights, load_sampling_config, load_video_split, partition_samples,
)
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


EPOCHS_DEFAULT = 8
BATCH_DEFAULT = 16
SEED = 42


def fresh_sampler(weights):
    """Reset the same training sample sequence for each variant."""
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )


def variant_paths(output_dir: Path, variant: str, *, smoke: bool) -> tuple[Path, Path]:
    if variant not in CONTROL_MASKS:
        raise ValueError(f"unknown variant: {variant}")
    prefix = "tiny_control_smoke_" if smoke else "tiny_control_"
    return (output_dir / f"{prefix}{variant}.pt",
            output_dir / f"{prefix}{variant}.json")


def preflight_inputs(args):
    """Validate splits, labels, and causal horizons before decoding/GPU use."""
    if not args.output_dir.is_dir():
        raise ValueError(f"output directory must already exist: {args.output_dir}")
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError("epochs/batch-size/cpu-threads must be positive")
    if not 0 < args.cuda_memory_fraction <= 1 or args.min_cuda_free_mib < 0:
        raise ValueError("invalid CUDA budget")
    if not args.variants or len(set(args.variants)) != len(args.variants):
        raise ValueError("variants must be nonempty and unique")
    for variant in args.variants:
        checkpoint, report = variant_paths(args.output_dir, variant, smoke=args.smoke)
        if checkpoint.exists() or report.exists():
            raise FileExistsError(f"refusing to overwrite {checkpoint} or {report}")

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if tuple(raw["dataset"]["prediction_horizons_ms"]) != (100, 200, 300):
        raise ValueError("unexpected source horizon labels")
    if int(raw["model"]["frame_stack"]) != 5:
        raise ValueError("expected five input frames")
    if not raw["preprocess"]["mask_touch_area"]:
        raise ValueError("touch masking must be enabled")
    split = load_video_split(args.config)
    partitions = partition_samples(load_v3_samples(args.samples), split)
    if args.smoke:
        partitions = {
            "train": partitions["train"][:128],
            "validation": partitions["validation"][:64],
        }
    else:
        partitions = {
            "train": partitions["train"],
            "validation": partitions["validation"],
        }
    if not partitions["train"] or not partitions["validation"]:
        raise ValueError("empty train or validation partition")
    selected = partitions["train"] + partitions["validation"]
    videos = {sample.video for sample in selected}
    events = {video: label_events(args.labels_dir, video) for video in videos}
    for sample in selected:
        validate_sample(sample, events[sample.video])
    return raw, split, partitions, selected, events


def train_variant(variant, *, partitions, features, events, split, device, budget,
                  epochs, batch_size, smoke, output_dir, cpu_threads, config):
    """Independently reset weights, optimizer and sampling for fair comparisons."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = TinyControlAblation(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    train_ds = TemporalExpertDataset(partitions["train"], features, events, mode="rgb")
    val_ds = TemporalExpertDataset(partitions["validation"], features, events, mode="rgb")
    sample_weights = build_sample_weights(
        partitions["train"], load_sampling_config(config)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=fresh_sampler(sample_weights), num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    best_loss, best_state = float("inf"), None
    history = []
    for epoch in range(1, (2 if smoke else epochs + 1)):
        train = run_epoch(model, train_loader, device, optimizer)
        val = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": train, "validation": val})
        print(f"{variant} epoch={epoch} train_BCE={train['bce']:.4f} "
              f"validation_BCE={val['bce']:.4f} val_acc={val['accuracy']:.4f}",
              flush=True)
        if val["bce"] < best_loss:
            best_loss = val["bce"]
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("no successful model epoch")
    model.load_state_dict(best_state, strict=True)
    validation_videos = ([partitions["validation"][0].video] if smoke
                         else list(split.validation))
    sequence = evaluate_sequences(
        model, val_ds, batch_size=batch_size, device=device,
        event_map=events, split_videos=validation_videos,
    )
    report = {
        "kind": "tiny_rgb_control_input_ablation_training",
        "variant": variant,
        "masked_control_channels": [i for i, v in enumerate(CONTROL_MASKS[variant]) if not v],
        "control_mask": list(CONTROL_MASKS[variant]),
        "frame_delta_preserved": True,
        "architecture": "TinyTemporalPolicy RGB CNN32 GRU64 control_dim3",
        "identical_model_parameter_count": sum(p.numel() for p in model.parameters()),
        "training_device": str(device), "device_budget": budget,
        "seed": SEED, "epochs_requested": epochs, "batch_size": batch_size,
        "smoke_unrepresentative": smoke,
        "split": split.name,
        "train_videos": list(split.train),
        "validation_videos": list(split.validation),
        "test_videos_not_evaluated": list(split.test),
        "source_input": "five masked RGB frames + causal expert state/age/frame_dt",
        "target": "expert_PRESS_at_observation_plus_100ms",
        "input_control_semantics": ["expert_pressed", "age_since_action_change_500ms", "frame_dt_200ms"],
        "selection": "lowest validation BCE at fixed threshold 0.5",
        "cpu_threads_if_cpu": cpu_threads,
        "history": history,
        "validation_sequence": sequence,
        "limitations": [
            "Teacher-forced original expert pixels and control history only.",
            "No model was tested on held-out test videos or live closed-loop control.",
            "For vision_only, frame delta remains; action state and age are masked.",
            "Mask is stored in JSON and is not encoded in state_dict; never load with bare TinyTemporalPolicy.",
        ],
    }
    checkpoint_path, report_path = variant_paths(output_dir, variant, smoke=smoke)
    torch.save(best_state, checkpoint_path)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    metrics = sequence["candidate"]["transition"]["all"]
    short = sequence["candidate"]["release_segment_recall"]["short_100_300ms"]
    print(f"{variant}: validation_f1={metrics['f1']} "
          f"matched={metrics['matched']}/{metrics['ground_truth']} "
          f"short_release={short['detected']}/{short['segments']} "
          f"checkpoint={checkpoint_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/train_v4c2_temporal_v2.yaml")
    parser.add_argument("--samples", type=Path,
                        default=ROOT / "data/processed/v3/samples.jsonl")
    parser.add_argument("--labels-dir", type=Path,
                        default=ROOT / "data/processed/v3/labels")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(CONTROL_MASKS),
                        default=list(CONTROL_MASKS))
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_DEFAULT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.03)
    parser.add_argument("--min-cuda-free-mib", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    raw, split, partitions, selected, events = preflight_inputs(args)
    print(f"preflight: train={len(partitions['train'])}, "
          f"validation={len(partitions['validation'])}, "
          f"videos={len(events)}, variants={args.variants}, "
          f"smoke={args.smoke}", flush=True)
    if args.preflight_only:
        print("Preflight passed: no video decoding, CUDA initialization or writes.")
        return

    device, budget = select_training_device(
        args.device, cuda_memory_fraction=args.cuda_memory_fraction,
        min_cuda_free_mib=args.min_cuda_free_mib,
    )
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    print(f"selected_device={device} budget={json.dumps(budget)}", flush=True)
    prep = preprocess_config_from_mapping(raw)
    road_cfg, _, _ = load_road_config(ROOT / "configs/geometry_pseudo_labels.yaml")
    features = cache_required_frames(selected, preprocess=prep, road_config=road_cfg)

    for variant in args.variants:
        train_variant(
            variant, partitions=partitions, features=features, events=events,
            split=split, device=device, budget=budget, epochs=args.epochs,
            batch_size=args.batch_size, smoke=args.smoke,
            output_dir=args.output_dir, cpu_threads=args.cpu_threads,
            config=args.config,
        )
    print("Ablation reports saved to", args.output_dir)


if __name__ == "__main__":
    main()
