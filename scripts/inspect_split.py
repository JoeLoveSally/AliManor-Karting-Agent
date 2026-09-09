#!/usr/bin/env python3
"""Inspect configured train/validation/test sample partitions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.train.trainer import (
    load_samples,
    load_sampling_config,
    load_video_split,
    partition_samples,
    sampling_summary,
)


def main() -> int:
    config_path = ROOT / "configs" / "train.yaml"
    samples = load_samples(ROOT / "data" / "processed" / "samples.jsonl")
    split = load_video_split(config_path)
    sampling = load_sampling_config(config_path)
    partitions = partition_samples(samples, split)

    print(f"Split: {split.name} ({split.strategy})")
    for name in ("train", "validation", "test"):
        items = partitions[name]
        videos = sorted({sample.video for sample in items})
        print(f"\n{name}: videos={len(videos)}, samples={len(items)}")
        for video in videos:
            print(f"  {video}")

        summary = sampling_summary(items, sampling)
        print(
            "  raw mix: "
            f"stable={summary['stable_raw_share']:.1%}, "
            f"transition={summary['transition_raw_share']:.1%}, "
            f"short={summary['short_correction_raw_share']:.1%}"
        )
        if name == "train":
            print(
                "  weighted mix: "
                f"stable={summary['stable_weighted_share']:.1%}, "
                f"transition={summary['transition_weighted_share']:.1%}, "
                f"short={summary['short_correction_weighted_share']:.1%}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
