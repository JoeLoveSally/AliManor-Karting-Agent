#!/usr/bin/env python3
"""Build labeled temporal dataset manifests from gameplay videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.train.dataset import DatasetConfig, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build labeled temporal dataset manifests.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    return parser.parse_args()


def load_config(path: Path) -> DatasetConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = raw.get("model", {})
    dataset = raw.get("dataset", {})
    return DatasetConfig(
        sample_fps=float(dataset.get("sample_fps", 30)),
        frame_stack=int(model.get("frame_stack", 3)),
        history_ms=float(dataset.get("history_ms", 100)),
        frame_interval_ms=float(dataset.get("frame_interval_ms", 50)),
        prediction_horizon_ms=float(dataset.get("prediction_horizon_ms", 100)),
        transition_window_ms=float(dataset.get("transition_window_ms", 200)),
        short_correction_min_ms=float(dataset.get("short_correction_min_ms", 100)),
        short_correction_max_ms=float(dataset.get("short_correction_max_ms", 300)),
        clean_isolated_one_frame_glitches=bool(
            dataset.get("clean_isolated_one_frame_glitches", True)
        ),
    )


def main() -> int:
    args = parse_args()
    videos = sorted(args.input.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no MP4 videos found in {args.input}")

    config = load_config(args.config)
    print(f"Building dataset from {len(videos)} video(s)...")
    manifest = build_dataset(
        videos,
        args.output,
        config,
        project_root=ROOT,
        progress=lambda index, total, path: print(
            f"[{index}/{total}] {path.name}", flush=True
        ),
    )
    summary = manifest["summary"]
    print(
        f"Done: videos={summary['videos']}, samples={summary['samples']}, "
        f"cleaned_frames={summary['cleaned_frames']}"
    )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
