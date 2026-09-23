#!/usr/bin/env python3
"""Paired, read-only evaluation of frozen V4-C2 and its history residual.

Uses human-held-out videos only. This is an offline imitation audit, not a
counterfactual replay of the two policies in a physical closed loop.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model_dir = ROOT / "artifacts/models/mobilenet_v3_small_v4c2_temporal_v2"
    adapter_dir = ROOT / "artifacts/models/mobilenet_v3_small_v4c2_history_residual_v1"
    parser.add_argument("--base-model", type=Path, default=model_dir / "model.pt")
    parser.add_argument("--metadata", type=Path, default=model_dir / "metadata_h200.json")
    parser.add_argument("--adapter", type=Path, default=adapter_dir / "history_adapter.pt")
    parser.add_argument("--adapter-metadata", type=Path, default=adapter_dir / "history_adapter_metadata.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_v4c2.yaml")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/processed/v3/samples.jsonl")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "data/processed/v3/labels")
    parser.add_argument("--split", choices=("validation", "test", "both"), default="both")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--output", type=Path, default=adapter_dir / "paired_eval.json")
    return parser.parse_args()


class Counts:
    def __init__(self):
        self.n = self.positive = self.negative = 0
        self.base_tp = self.base_fp = self.base_fn = self.base_tn = 0
        self.new_tp = self.new_fp = self.new_fn = self.new_tn = 0
        self.fixed_fp = self.added_fp = self.fixed_fn = self.added_fn = 0
        self.changed = 0
        self.brier_base = self.brier_new = 0.0
        self.logloss_base = self.logloss_new = 0.0

    def add(self, label, pb, pn, threshold):
        import math
        self.n += 1
        truth = bool(label)
        b, n = pb >= threshold, pn >= threshold
        self.positive += int(truth)
        self.negative += int(not truth)
        for prefix, decision in (("base", b), ("new", n)):
            outcome = "tp" if truth and decision else "fn" if truth else "fp" if decision else "tn"
            setattr(self, prefix + "_" + outcome, getattr(self, prefix + "_" + outcome) + 1)
        self.fixed_fp += int(not truth and b and not n)
        self.added_fp += int(not truth and not b and n)
        self.fixed_fn += int(truth and not b and n)
        self.added_fn += int(truth and b and not n)
        self.changed += int(b != n)
        self.brier_base += (pb - label) ** 2
        self.brier_new += (pn - label) ** 2
        eps = 1e-7
        self.logloss_base += -(label * math.log(max(pb, eps)) + (1 - label) * math.log(max(1 - pb, eps)))
        self.logloss_new += -(label * math.log(max(pn, eps)) + (1 - label) * math.log(max(1 - pn, eps)))

    def as_dict(self):
        result = dict(vars(self))
        if self.n:
            result["base_recall"] = self.base_tp / self.positive if self.positive else None
            result["new_recall"] = self.new_tp / self.positive if self.positive else None
            result["base_precision"] = self.base_tp / (self.base_tp + self.base_fp) if self.base_tp + self.base_fp else None
            result["new_precision"] = self.new_tp / (self.new_tp + self.new_fp) if self.new_tp + self.new_fp else None
            for key in ("brier_base", "brier_new", "logloss_base", "logloss_new"):
                result[key] /= self.n
        return result


def evaluate_split(dataset, model, *, device, batch_size, threshold, horizon_index):
    import torch
    from torch.utils.data import DataLoader

    metrics = defaultdict(Counts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sample_offset = 0
    max_age_ms = model.max_age_ms
    with torch.inference_mode():
        for batch in loader:
            images = batch["input"].to(device=device, dtype=torch.float32)
            current = batch["current_pressed"].to(device=device)
            history = batch["action_history"].to(device=device, dtype=torch.float32)
            truth = batch["switch_target"][:, horizon_index].detach().cpu().numpy()
            corrected, baseline = model.model(images, current, history)
            pn = torch.sigmoid(corrected[:, horizon_index]).detach().cpu().numpy()
            pb = torch.sigmoid(baseline[:, horizon_index]).detach().cpu().numpy()
            hist = batch["action_history"].numpy()
            current_np = batch["current_pressed"].numpy()
            for j, (label, base_prob, new_prob) in enumerate(zip(truth, pb, pn)):
                age_known = hist[j, -1] > .5
                age_ms = float(hist[j, -2] * max_age_ms)
                if not age_known:
                    phase = "age_unknown"
                elif age_ms < 100.0 - 1e-5:
                    phase = "post_switch_0_100ms"
                elif age_ms < 200.0 - 1e-5:
                    phase = "post_switch_100_200ms"
                else:
                    phase = "post_switch_200ms_plus"
                mixed = bool((hist[j, :-2:3] != float(current_np[j])).any())
                video = dataset.samples[sample_offset + j].video
                groups = ("all", phase, "mixed_history" if mixed else "uniform_history",
                          "pressed" if int(current_np[j]) else "released",
                          "video:" + video)
                for group in groups:
                    metrics[group].add(float(label), float(base_prob), float(new_prob), threshold)
            sample_offset += len(truth)
    assert sample_offset == len(dataset)
    return {key: value.as_dict() for key, value in sorted(metrics.items())}


def main():
    args = parse_args()
    if args.batch_size < 1 or not 0 < args.threshold < 1:
        raise ValueError("invalid batch size or probability threshold")
    import torch
    import yaml
    from karting_agent.model.action_history_runner import ActionHistoryAdapterRunner
    from karting_agent.train.action_history_dataset import ActionHistoryVideoDataset
    from karting_agent.train.state_conditioned_dataset import load_v3_samples
    from karting_agent.train.trainer import load_video_split, partition_samples
    from karting_agent.train.frame_cache import frame_cache_root_from_config

    model = ActionHistoryAdapterRunner(
        base_model_path=args.base_model, base_metadata_path=args.metadata,
        adapter_path=args.adapter, adapter_metadata_path=args.adapter_metadata,
        device=args.device,
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    samples = load_v3_samples(args.samples)
    partitions = partition_samples(samples, load_video_split(args.config))
    cache_root = frame_cache_root_from_config(config, ROOT)
    horizon_index = model.baseline.spec.control_output_index
    output = {"threshold": args.threshold, "horizon_ms": model.baseline.spec.control_horizon_ms,
              "caution": "Offline human-video labels, NOT verified closed-loop progress; adjacent frames are correlated.",
              "splits": {}}
    for split_name in (("validation", "test") if args.split == "both" else (args.split,)):
        dataset = ActionHistoryVideoDataset(
            partitions[split_name], project_root=ROOT, labels_dir=args.labels_dir,
            preprocess_config=model.baseline.spec.preprocess_config,
            cache_root=cache_root, max_age_ms=model.max_age_ms,
        )
        try:
            result = evaluate_split(dataset, model, device=model.baseline.device,
                                    batch_size=args.batch_size, threshold=args.threshold,
                                    horizon_index=horizon_index)
        finally:
            dataset.close()
        output["splits"][split_name] = result
        print(json.dumps({"split": split_name,
                          "all": result["all"],
                          "post_switch_0_100ms": result.get("post_switch_0_100ms", {}),
                          "mixed_history": result.get("mixed_history", {})},
                         ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"evaluation report exists, choose another --output: {args.output}")
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved paired audit: {args.output}")


if __name__ == "__main__":
    main()
