#!/usr/bin/env python3
"""Validation-gated, read-only H200 threshold sweep for V4-C2 history adapter.

The original controller's threshold remains fixed at 0.60 (or --baseline-threshold).
This script never trains a model or issues ADB commands. Previously inspected
project test videos are not a pristine holdout; --mode test is exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = ROOT / "artifacts/models/mobilenet_v3_small_v4c2_temporal_v2"
    adapter = ROOT / "artifacts/models/mobilenet_v3_small_v4c2_history_residual_v1"
    parser.add_argument("--mode", choices=("validation", "test"), default="validation")
    parser.add_argument("--base-model", type=Path, default=baseline / "model.pt")
    parser.add_argument("--metadata", type=Path, default=baseline / "metadata_h200.json")
    parser.add_argument("--adapter", type=Path, default=adapter / "history_adapter.pt")
    parser.add_argument("--adapter-metadata", type=Path, default=adapter / "history_adapter_metadata.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_v4c2.yaml")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/processed/v3/samples.jsonl")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "data/processed/v3/labels")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--baseline-threshold", type=float, default=0.60)
    parser.add_argument("--threshold-min", type=float, default=0.55)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--validation-report", type=Path, default=adapter / "threshold_validation.json")
    parser.add_argument("--test-report", type=Path, default=adapter / "threshold_test.json")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(labels: np.ndarray, baseline_probs: np.ndarray, adapter_probs: np.ndarray,
          baseline_threshold: float, adapter_threshold: float) -> dict[str, int | float | None]:
    """Compare decisions on matching human frames; keep baseline threshold fixed."""
    truth = np.asarray(labels, dtype=np.bool_)
    pb = np.asarray(baseline_probs, dtype=np.float64)
    pn = np.asarray(adapter_probs, dtype=np.float64)
    if truth.ndim != 1 or pb.shape != truth.shape or pn.shape != truth.shape:
        raise ValueError("paired label/probability vectors must be one-dimensional and equal length")
    if not np.all(np.isfinite(pb)) or not np.all(np.isfinite(pn)):
        raise ValueError("non-finite probability")
    if np.any((pb < 0) | (pb > 1)) or np.any((pn < 0) | (pn > 1)):
        raise ValueError("probability must lie in [0,1]")
    b = pb >= baseline_threshold
    n = pn >= adapter_threshold
    positive = int(truth.sum())
    result: dict[str, int | float | None] = {
        "n": len(truth), "positive": positive, "negative": len(truth) - positive,
        "base_tp": int((truth & b).sum()), "base_fp": int((~truth & b).sum()),
        "base_fn": int((truth & ~b).sum()), "base_tn": int((~truth & ~b).sum()),
        "new_tp": int((truth & n).sum()), "new_fp": int((~truth & n).sum()),
        "new_fn": int((truth & ~n).sum()), "new_tn": int((~truth & ~n).sum()),
        "fixed_fp": int((~truth & b & ~n).sum()),
        "added_fp": int((~truth & ~b & n).sum()),
        "fixed_fn": int((truth & ~b & n).sum()),
        "added_fn": int((truth & b & ~n).sum()),
    }
    result["base_recall"] = result["base_tp"] / positive if positive else None
    result["new_recall"] = result["new_tp"] / positive if positive else None
    return result


def scan(rows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
         *, baseline_threshold: float, thresholds: list[float]):
    """Require no FP increase and no TP decrease overall AND in the early window."""
    early = rows["post_switch_0_100ms"][0]
    if len(rows["all"][0]) == 0 or len(early) == 0 or int(early.sum()) == 0:
        raise ValueError("validation must contain positive post-switch samples")
    results = []
    for threshold in thresholds:
        groups = {name: score(*values, baseline_threshold, threshold)
                  for name, values in rows.items()}
        overall = groups["all"]
        early_counts = groups["post_switch_0_100ms"]
        gates = {
            "overall_fp_not_higher": overall["new_fp"] <= overall["base_fp"],
            "overall_tp_not_lower": overall["new_tp"] >= overall["base_tp"],
            "early_fp_not_higher": early_counts["new_fp"] <= early_counts["base_fp"],
            "early_tp_not_lower": early_counts["new_tp"] >= early_counts["base_tp"],
        }
        results.append({"threshold": threshold, "eligible": all(gates.values()),
                        "gates": gates, "groups": groups})
    eligible = [row for row in results if row["eligible"]]
    # A candidate for exploratory offline verification, NEVER an Armed setting.
    best = max(eligible, key=lambda item: (
        item["groups"]["post_switch_0_100ms"]["new_tp"],
        item["groups"]["all"]["new_tp"],
        -item["groups"]["post_switch_0_100ms"]["new_fp"],
        -item["groups"]["all"]["new_fp"], item["threshold"],
    )) if eligible else None
    return results, best


def threshold_grid(low: float, high: float, step: float) -> list[float]:
    if not (0 < low <= high < 1) or not step > 0:
        raise ValueError("invalid threshold interval")
    length = int(round((high - low) / step))
    if length < 0 or length > 200:
        raise ValueError("at most 201 thresholds are permitted")
    if abs(low + length * step - high) > 1e-7:
        raise ValueError("threshold interval must be divisible by step")
    return [round(low + index * step, 8) for index in range(length + 1)]


def collect(dataset, model, *, device, batch_size: int, horizon_index: int):
    import torch
    from torch.utils.data import DataLoader

    collected: dict[str, list[tuple[float, float, float]]] = {
        name: [] for name in ("all", "post_switch_0_100ms", "post_switch_100_200ms",
                            "post_switch_200ms_plus", "age_unknown", "mixed_history", "uniform_history")
    }
    model.model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            images = batch["input"].to(device=device, dtype=torch.float32)
            current = batch["current_pressed"].to(device=device)
            history = batch["action_history"].to(device=device, dtype=torch.float32)
            truth = batch["switch_target"][:, horizon_index].detach().cpu().numpy()
            corrected, original = model.model(images, current, history)
            pn = torch.sigmoid(corrected[:, horizon_index]).detach().cpu().numpy()
            pb = torch.sigmoid(original[:, horizon_index]).detach().cpu().numpy()
            hist = batch["action_history"].numpy()
            states = batch["current_pressed"].numpy()
            for j, (label, base_prob, new_prob) in enumerate(zip(truth, pb, pn)):
                if hist[j, -1] <= .5:
                    phase = "age_unknown"
                else:
                    age_ms = float(hist[j, -2] * model.max_age_ms)
                    phase = ("post_switch_0_100ms" if age_ms < 100.0 - 1e-5 else
                             "post_switch_100_200ms" if age_ms < 200.0 - 1e-5 else
                             "post_switch_200ms_plus")
                mixed = bool((hist[j, :-2:3] != float(states[j])).any())
                record = (float(label), float(base_prob), float(new_prob))
                for name in ("all", phase, "mixed_history" if mixed else "uniform_history"):
                    collected[name].append(record)
    return {name: tuple(np.array([row[i] for row in records], dtype=np.float64)
                        for i in range(3))
            for name, records in collected.items() if records}


def evaluate_rows(partitions, split_name, args, model, raw_config):
    from karting_agent.train.action_history_dataset import ActionHistoryVideoDataset
    from karting_agent.train.frame_cache import frame_cache_root_from_config
    dataset = ActionHistoryVideoDataset(
        partitions[split_name], project_root=ROOT, labels_dir=args.labels_dir,
        preprocess_config=model.baseline.spec.preprocess_config,
        cache_root=frame_cache_root_from_config(raw_config, ROOT), max_age_ms=model.max_age_ms)
    try:
        return collect(dataset, model, device=model.baseline.device,
                       batch_size=args.batch_size,
                       horizon_index=model.baseline.spec.control_output_index)
    finally:
        dataset.close()


def main():
    args = parse_args()
    if args.batch_size < 1 or not 0 < args.baseline_threshold < 1:
        raise ValueError("invalid batch size or baseline threshold")
    import yaml
    from karting_agent.model.action_history_runner import ActionHistoryAdapterRunner
    from karting_agent.train.state_conditioned_dataset import load_v3_samples
    from karting_agent.train.trainer import load_video_split, partition_samples

    paths = {name: Path(path).resolve() for name, path in (
        ("base_model", args.base_model), ("base_metadata", args.metadata),
        ("adapter", args.adapter), ("adapter_metadata", args.adapter_metadata),
        ("config", args.config), ("samples", args.samples))}
    if args.mode == "validation":
        thresholds = threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
        output_path = args.validation_report.resolve()
        if output_path.exists():
            raise FileExistsError(f"validation report exists; choose a new path: {output_path}")
    else:
        validated = json.loads(args.validation_report.read_text(encoding="utf-8"))
        selected = validated.get("selected")
        if validated.get("mode") != "validation" or not isinstance(selected, dict):
            raise ValueError("validation has no eligible candidate; test skipped")
        if float(validated["baseline_threshold"]) != args.baseline_threshold:
            raise ValueError("baseline threshold differs from validation")
        if validated["sha256"] != {name: sha256_file(path) for name, path in paths.items()}:
            raise ValueError("model/data/config differ from validation; test skipped")
        candidate = float(selected["threshold"])
        if not any(row["eligible"] and row["threshold"] == candidate for row in validated["sweep"]):
            raise ValueError("selected candidate not eligible in validation sweep")
        output_path = args.test_report.resolve()
        if output_path.exists():
            raise FileExistsError(f"test report exists; choose a new path: {output_path}")

    model = ActionHistoryAdapterRunner(
        base_model_path=paths["base_model"], base_metadata_path=paths["base_metadata"],
        adapter_path=paths["adapter"], adapter_metadata_path=paths["adapter_metadata"],
        device=args.device)
    raw_config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("config must be a mapping")
    samples = load_v3_samples(paths["samples"])
    partitions = partition_samples(samples, load_video_split(paths["config"]))
    if args.mode == "validation":
        rows = evaluate_rows(partitions, "validation", args, model, raw_config)
        results, best = scan(rows, baseline_threshold=args.baseline_threshold, thresholds=thresholds)
        report = {"mode": "validation", "baseline_threshold": args.baseline_threshold,
                  "criteria": "new FP <= old FP and new TP >= old TP overall AND within first 100ms; fixed original threshold",
                  "sha256": {name: sha256_file(path) for name, path in paths.items()},
                  "selected": ({"threshold": best["threshold"], "groups": best["groups"]} if best else None),
                  "sweep": results,
                  "caution": "Validation-only selection. Test was already inspected in earlier experiments; not an untouched holdout or Armed validation."}
        print(json.dumps({"mode": "validation", "eligible": sum(row["eligible"] for row in results),
                          "selected": report["selected"],
                          "baseline_0_60": next((r["groups"] for r in results if r["threshold"] == .6), None)},
                         ensure_ascii=False), flush=True)
    else:
        rows = evaluate_rows(partitions, "test", args, model, raw_config)
        groups = {name: score(*values, args.baseline_threshold, candidate)
                  for name, values in rows.items()}
        report = {"mode": "test", "baseline_threshold": args.baseline_threshold,
                  "candidate_threshold": candidate, "validation_report": str(args.validation_report.resolve()),
                  "sha256": validated["sha256"], "groups": groups,
                  "caution": "Earlier experiments already inspected test. Exploratory comparison, not a pristine holdout or closed-loop evidence."}
        print(json.dumps({"mode": "test", "threshold": candidate, "all": groups["all"],
                          "post_switch_0_100ms": groups.get("post_switch_0_100ms")},
                         ensure_ascii=False), flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Saved read-only {args.mode} threshold audit: {output_path}")


if __name__ == "__main__":
    main()
