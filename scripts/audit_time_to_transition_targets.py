#!/usr/bin/env python3
"""Audit first-transition timing targets for a state-conditioned event-time policy."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402
from karting_agent.train.trainer import (  # noqa: E402
    load_video_split,
    partition_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit time-to-first-switch targets under recorded and "
            "counterfactual action states."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_v4c3_temporal_delta_dense.yaml",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "v4c3_temporal_delta_dense"
            / "samples.jsonl"
        ),
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "v4c3_temporal_delta_dense"
            / "labels"
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,validation",
        help="Comma-separated dataset splits to audit.",
    )
    parser.add_argument("--max-horizon-ms", type=float, default=300.0)
    parser.add_argument("--bin-ms", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load_transition_times(labels_dir: Path, video: str) -> np.ndarray:
    path = labels_dir / f"{Path(video).stem}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError(f"label file has no events: {path}")
    return np.asarray(
        [float(event["timestamp_ms"]) for event in events[1:]],
        dtype=np.float64,
    )


def _first_transition_after(
    transition_times: np.ndarray,
    observation_ms: float,
) -> float | None:
    index = int(np.searchsorted(transition_times, observation_ms, side="right"))
    if index >= transition_times.size:
        return None
    return float(transition_times[index])


def _bin_label(
    tte_ms: float | None,
    *,
    max_horizon_ms: float,
    bin_ms: float,
) -> str:
    if tte_ms is None or tte_ms > max_horizon_ms:
        return f">{max_horizon_ms:g}"
    if tte_ms <= 1e-6:
        return "immediate"
    upper = min(
        max_horizon_ms,
        np.ceil(tte_ms / bin_ms) * bin_ms,
    )
    lower = max(0.0, upper - bin_ms)
    return f"({lower:g},{upper:g}]"


def _quantiles(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
    }


def _audit_split(
    samples,
    *,
    labels_dir: Path,
    max_horizon_ms: float,
    bin_ms: float,
) -> dict[str, object]:
    transitions_by_video = {
        sample.video: _load_transition_times(labels_dir, sample.video)
        for sample in samples
    }

    aligned_bins: Counter[str] = Counter()
    counterfactual_bins: Counter[str] = Counter()
    aligned_ttes: list[float] = []
    future_transition_counts: Counter[int] = Counter()

    for sample in samples:
        observation_ms = float(sample.input_timestamps_ms[-1])
        transition_times = transitions_by_video[sample.video]
        future_count = int(
            (
                (transition_times > observation_ms)
                & (transition_times <= observation_ms + max_horizon_ms)
            ).sum()
        )
        future_transition_counts[future_count] += 1

        first_transition_ms = _first_transition_after(
            transition_times,
            observation_ms,
        )
        aligned_tte = (
            None
            if first_transition_ms is None
            else first_transition_ms - observation_ms
        )
        aligned_bins[
            _bin_label(
                aligned_tte,
                max_horizon_ms=max_horizon_ms,
                bin_ms=bin_ms,
            )
        ] += 1
        if aligned_tte is not None and aligned_tte <= max_horizon_ms:
            aligned_ttes.append(aligned_tte)

        # With two counterfactual conditioning states, the state opposite to
        # the expert action at observation time should switch immediately.
        counterfactual_bins["immediate"] += 1
        counterfactual_bins[
            _bin_label(
                aligned_tte,
                max_horizon_ms=max_horizon_ms,
                bin_ms=bin_ms,
            )
        ] += 1

    total = len(samples)
    multiple = sum(
        count
        for transition_count, count in future_transition_counts.items()
        if transition_count >= 2
    )
    return {
        "samples": total,
        "aligned_bins": dict(sorted(aligned_bins.items())),
        "counterfactual_bins": dict(sorted(counterfactual_bins.items())),
        "aligned_first_transition_ms": _quantiles(aligned_ttes),
        "future_transition_count_within_horizon": dict(
            sorted(future_transition_counts.items())
        ),
        "multiple_transition_samples": multiple,
        "multiple_transition_fraction": multiple / total if total else 0.0,
    }


def main() -> int:
    args = parse_args()
    if args.max_horizon_ms <= 0:
        raise ValueError("--max-horizon-ms must be > 0")
    if args.bin_ms <= 0 or args.bin_ms > args.max_horizon_ms:
        raise ValueError("--bin-ms must be in (0, max-horizon-ms]")

    requested_splits = tuple(
        part.strip() for part in args.splits.split(",") if part.strip()
    )
    if not requested_splits:
        raise ValueError("--splits must contain at least one split")

    split_config = load_video_split(args.config.resolve())
    samples = load_v3_samples(args.samples.resolve())
    partitions = partition_samples(samples, split_config)
    unknown = [name for name in requested_splits if name not in partitions]
    if unknown:
        raise ValueError(f"unknown splits: {unknown}")

    results: dict[str, object] = {}
    for name in requested_splits:
        summary = _audit_split(
            partitions[name],
            labels_dir=args.labels_dir.resolve(),
            max_horizon_ms=float(args.max_horizon_ms),
            bin_ms=float(args.bin_ms),
        )
        results[name] = summary
        print(f"\n{name}: samples={summary['samples']}")
        print(f"  aligned bins: {summary['aligned_bins']}")
        print(f"  counterfactual bins: {summary['counterfactual_bins']}")
        print(
            "  aligned first transition: "
            f"{summary['aligned_first_transition_ms']}"
        )
        print(
            "  future transition counts: "
            f"{summary['future_transition_count_within_horizon']}"
        )
        print(
            "  multiple-transition fraction: "
            f"{summary['multiple_transition_fraction']:.4f}"
        )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            ROOT
            / "artifacts"
            / "analysis"
            / "time_to_transition_targets.json"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "config": str(args.config.resolve()),
                "samples": str(args.samples.resolve()),
                "labels_dir": str(args.labels_dir.resolve()),
                "max_horizon_ms": args.max_horizon_ms,
                "bin_ms": args.bin_ms,
                "splits": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nOutput: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
