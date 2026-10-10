#!/usr/bin/env python3
"""Read-only RGB/HSV frozen Validation parity probe: CPU vs CUDA.

Not a training run, threshold sweep, model change, shadow-control patch,
Armed session or Test-set evaluation. Does NOT write any results or weights.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiment_tiny_temporal_policy import (  # noqa: E402
    TemporalExpertDataset, cache_required_frames, label_events, validate_sample,
)
from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from karting_agent.model.tiny_temporal_policy import TinyTemporalPolicy  # noqa: E402
from karting_agent.train.gpu_budget import select_training_device  # noqa: E402
from karting_agent.train.heldout_guard import require_frozen_checkpoint  # noqa: E402
from karting_agent.train.sequence_evaluator import (  # noqa: E402
    SequencePoint, Transition, ReleaseSegment, evaluate_sequence,
    combine_evaluations,
)
from karting_agent.train.shadow_backend_parity import compare_backend_probabilities  # noqa: E402
from karting_agent.train.shadow_teacher import canonical_teacher_probabilities  # noqa: E402
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402
from karting_agent.train.trainer import load_video_split, partition_samples  # noqa: E402
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


def score_original_protocol(samples, values, events, videos):
    """Same sequence evaluator, target timestamps and per-video GT as training."""
    by_video = defaultdict(list)
    for sample in samples:
        by_video[sample.video].append(sample)
    evaluations = []
    for video in videos:
        rows = sorted(by_video[video], key=lambda sample: sample.target_timestamp_ms)
        points = [
            SequencePoint(video, sample.target_timestamp_ms,
                          values[(video, float(sample.target_timestamp_ms))])
            for sample in rows
        ]
        timeline = events[video]
        transitions = [
            Transition(video, item.timestamp_ms, item.pressed)
            for item in timeline[1:]
        ]
        releases = [
            ReleaseSegment(video, a.timestamp_ms, b.timestamp_ms)
            for a, b in zip(timeline, timeline[1:])
            if not a.pressed and b.pressed
        ]
        evaluations.append(evaluate_sequence(
            points, transitions, releases, threshold=0.5, tolerance_ms=100.0
        ))
    return combine_evaluations(evaluations).summary()


def counts(summary):
    return {
        direction: {
            k: summary["transition"][direction][k]
            for k in ("ground_truth", "predicted", "matched", "false_positive", "false_negative")
        }
        for direction in ("all", "press", "release")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT/"configs/train_v4c2_temporal_v2.yaml")
    parser.add_argument("--samples", type=Path,
                        default=ROOT/"data/processed/v3/samples.jsonl")
    parser.add_argument("--labels-dir", type=Path,
                        default=ROOT/"data/processed/v3/labels")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("rgb", "rgb_hsv"), default="rgb")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.03)
    parser.add_argument("--min-cuda-free-mib", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be >= 1")
    if not args.checkpoint_dir.is_dir():
        parser.error("checkpoint-dir must exist")

    split = load_video_split(args.config)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (not split.validation or
            set(split.test) & (set(split.validation) | set(split.train)) or
            int(config["model"]["frame_stack"]) != 5 or
            not config["preprocess"]["mask_touch_area"]):
        raise ValueError("invalid frozen Validation-only protocol")
    report = json.loads(
        (args.checkpoint_dir/f"tiny_policy_{args.mode}.json").read_text(encoding="utf-8")
    )
    require_frozen_checkpoint(report, mode=args.mode, split=split)
    if report.get("training_device") != "cuda:0":
        raise ValueError("expected frozen original CUDA Validation training")
    batch_size = report.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("missing frozen reference batch size")

    # Strict CUDA preflight BEFORE decoding video; never stop co-resident vLLM.
    device, gpu_budget = select_training_device(
        "cuda", cuda_memory_fraction=args.cuda_memory_fraction,
        min_cuda_free_mib=args.min_cuda_free_mib,
        allow_cpu_fallback=False,
    )
    torch.set_num_threads(args.cpu_threads)
    partitions = partition_samples(load_v3_samples(args.samples), split)
    validation = partitions["validation"]
    if {s.video for s in validation} != set(split.validation):
        raise ValueError("validation source videos incomplete")
    events = {video: label_events(args.labels_dir, video) for video in split.validation}
    for sample in validation:
        validate_sample(sample, events[sample.video])
    if len(validation) != report["validation_sequence"]["candidate"]["samples"]:
        raise ValueError("validation sample count differs from frozen report")
    preprocess = preprocess_config_from_mapping(config)
    road_cfg, _, _ = load_road_config(ROOT/"configs/geometry_pseudo_labels.yaml")
    print(f"Frozen {args.mode}: {len(validation)} Validation samples, batch={batch_size}; "
          f"GPU budget={json.dumps(gpu_budget)}; no Test/Armed",flush=True)
    frames = cache_required_frames(validation, preprocess=preprocess, road_config=road_cfg)
    data = TemporalExpertDataset(validation, frames, events, mode=args.mode)
    channels = 3 if args.mode == "rgb" else 4
    weights = torch.load(args.checkpoint_dir/f"tiny_policy_{args.mode}.pt",
                         map_location="cpu",weights_only=True)
    cpu_model = TinyTemporalPolicy(image_channels=channels)
    cpu_model.load_state_dict(weights, strict=True)
    cpu = canonical_teacher_probabilities(
        cpu_model, data, batch_size=batch_size, device="cpu"
    )
    gpu_model = TinyTemporalPolicy(image_channels=channels)
    gpu_model.load_state_dict(weights, strict=True)
    gpu_model = gpu_model.to(device)
    cuda = canonical_teacher_probabilities(
        gpu_model, data, batch_size=batch_size, device=device
    )
    expected = report["validation_sequence"]["candidate"]
    cpu_summary = score_original_protocol(validation, cpu, events, split.validation)
    cuda_summary = score_original_protocol(validation, cuda, events, split.validation)
    difference = compare_backend_probabilities(cpu, cuda, threshold=0.5)
    results = {
        "kind": "read_only_frozen_validation_cpu_cuda_parity_probe",
        "model": args.mode,
        "train_device_from_report": report["training_device"],
        "frozen_decision_threshold": 0.5,
        "test_videos_used": False,
        "expected": counts(expected),
        "cpu": counts(cpu_summary),
        "cuda": counts(cuda_summary),
        "cpu_matches_frozen_counts": counts(cpu_summary) == counts(expected),
        "cuda_matches_frozen_counts": counts(cuda_summary) == counts(expected),
        "cpu_cuda_comparison": difference,
    }
    print(json.dumps(results, ensure_ascii=False, indent=2),flush=True)
    # Diagnostic only: do not modify original reports or relax failed parity.
    print("Read-only diagnosis finished; original shadow strict check unchanged.",flush=True)


if __name__ == "__main__":
    main()
