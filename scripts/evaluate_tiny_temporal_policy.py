#!/usr/bin/env python3
"""One-shot, read-only held-out test of frozen tiny temporal checkpoints.

Does NOT train, retune a threshold, select an epoch or call the live game.
The output is an aggregate JSON with per-video metrics and checkpoint hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from experiment_tiny_temporal_policy import (  # noqa: E402
    SIZE, TemporalExpertDataset, cache_required_frames, evaluate_sequences,
    label_events, run_epoch, validate_sample,
)
from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from karting_agent.model.tiny_temporal_policy import TinyTemporalPolicy  # noqa: E402
from karting_agent.train.gpu_budget import select_training_device  # noqa: E402
from karting_agent.train.heldout_guard import require_frozen_checkpoint  # noqa: E402
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402
from karting_agent.train.trainer import load_video_split, partition_samples  # noqa: E402
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402

MODES = ("action_only", "rgb", "rgb_hsv")
OUTPUT_NAME = "tiny_policy_heldout_test.json"


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def frozen_paths(folder: Path, mode: str) -> tuple[Path, Path]:
    if mode not in MODES:
        raise ValueError(f"unknown model mode: {mode}")
    return folder / f"tiny_policy_{mode}.pt", folder / f"tiny_policy_{mode}.json"


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/train_v4c2_temporal_v2.yaml")
    parser.add_argument("--samples", type=Path,
                        default=ROOT / "data/processed/v3/samples.jsonl")
    parser.add_argument("--labels-dir", type=Path,
                        default=ROOT / "data/processed/v3/labels")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        default=list(MODES))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.03)
    parser.add_argument("--min-cuda-free-mib", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive")
    if len(set(args.modes)) != len(args.modes):
        parser.error("duplicate model modes")
    if not args.output_dir.is_dir() or not args.checkpoint_dir.is_dir():
        parser.error("output-dir and checkpoint-dir must exist")
    out = args.output_dir / OUTPUT_NAME
    if out.exists():
        parser.error(f"refusing to overwrite held-out result: {out}")

    # Freeze protocol and fail before expensive decoding or loading CUDA.
    split = load_video_split(args.config)
    if not split.test:
        parser.error("no held-out test videos")
    records = {}
    for mode in args.modes:
        checkpoint, report_path = frozen_paths(args.checkpoint_dir, mode)
        if not checkpoint.is_file() or not report_path.is_file():
            parser.error(f"missing frozen {mode} model/report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require_frozen_checkpoint(report, mode=mode, split=split)
        records[mode] = {
            "checkpoint": checkpoint, "report": report_path,
            "checkpoint_sha256": sha256(checkpoint),
            "training_report_sha256": sha256(report_path),
        }

    all_samples = load_v3_samples(args.samples)
    partitions = partition_samples(all_samples, split)
    samples = partitions["test"]
    assert samples
    expected_videos = set(split.test)
    assert {sample.video for sample in samples} == expected_videos
    event_map = {
        video: label_events(args.labels_dir, video)
        for video in split.test
    }
    for sample in samples:
        validate_sample(sample, event_map[sample.video])
    print(f"Frozen held-out test: {len(samples)} expert samples, "
          f"{len(split.test)} unseen videos; modes={args.modes}", flush=True)

    device, device_budget = select_training_device(
        args.device,
        cuda_memory_fraction=args.cuda_memory_fraction,
        min_cuda_free_mib=args.min_cuda_free_mib,
    )
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    print(f"Inference device={device}; {json.dumps(device_budget)}", flush=True)

    # The zero-image ablation does not need video. Decoding occurs once for
    # BOTH visual models and never accesses train/validation video frames.
    feature_cache = {}
    if any(mode != "action_only" for mode in args.modes):
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        prep = preprocess_config_from_mapping(raw)
        if not prep.mask_touch_area:
            raise ValueError("touch input must be masked")
        road_config, _, _ = load_road_config(
            ROOT / "configs/geometry_pseudo_labels.yaml"
        )
        feature_cache = cache_required_frames(
            samples, preprocess=prep, road_config=road_config,
        )
    result = {
        "kind": "frozen_tiny_policy_heldout_test",
        "read_only_evaluation": True,
        "expert_teacher_forced": True,
        "not_closed_loop": True,
        "split": split.name,
        "test_videos": list(split.test),
        "test_samples": len(samples),
        "prediction_horizon_ms": 100.0,
        "threshold": 0.5,
        "transition_tolerance_ms": 100.0,
        "device": str(device),
        "device_budget": device_budget,
        "results": {},
    }
    for mode in args.modes:
        channels = 4 if mode == "rgb_hsv" else 3
        model = TinyTemporalPolicy(image_channels=channels)
        state = torch.load(records[mode]["checkpoint"], map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        dataset = TemporalExpertDataset(
            samples, feature_cache, event_map, mode=mode,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
        aggregate_loss = run_epoch(model, loader, device)
        aggregate_seq = evaluate_sequences(
            model, dataset, batch_size=args.batch_size,
            device=device, event_map=event_map, split_videos=split.test,
        )
        per_video = {}
        for video in split.test:
            video_rows = [sample for sample in samples if sample.video == video]
            video_ds = TemporalExpertDataset(
                video_rows, feature_cache, event_map, mode=mode,
            )
            video_loss = run_epoch(
                model,
                DataLoader(video_ds, batch_size=args.batch_size,
                           shuffle=False, num_workers=0),
                device,
            )
            video_seq = evaluate_sequences(
                model, video_ds, batch_size=args.batch_size, device=device,
                event_map=event_map, split_videos=[video],
            )
            per_video[video] = {
                "sample_metrics": video_loss,
                "sequence": video_seq,
            }
        result["results"][mode] = {
            "checkpoint_sha256": records[mode]["checkpoint_sha256"],
            "training_report_sha256": records[mode]["training_report_sha256"],
            "sample_metrics": aggregate_loss,
            "sequence": aggregate_seq,
            "per_video": per_video,
        }
        events = aggregate_seq["candidate"]["transition"]["all"]
        short = aggregate_seq["candidate"]["release_segment_recall"]["short_100_300ms"]
        print(f"{mode}: test_F1={events['f1']:.4f} "
              f"matched={events['matched']}/{events['ground_truth']} "
              f"FP={events['false_positive']} FN={events['false_negative']} "
              f"short_RELEASE={short['detected']}/{short['segments']}",
              flush=True)
        del model
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Saved held-out evaluation: {out}", flush=True)


if __name__ == "__main__":
    run()
