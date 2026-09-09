#!/usr/bin/env python3
"""Build memory-mapped RGB uint8 frame caches for temporal training samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.train.frame_cache import build_frame_cache, frame_cache_root_from_config
from karting_agent.train.trainer import load_samples
from karting_agent.vision.preprocess import preprocess_config_from_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal frame cache.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "processed" / "samples.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override frame_cache.root from config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild caches even when current files are valid.",
    )
    return parser.parse_args()


def load_raw_config(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("train config root must be a mapping")
    return raw


def main() -> int:
    args = parse_args()
    raw = load_raw_config(args.config)
    preprocess_config = preprocess_config_from_mapping(raw)
    samples = load_samples(args.samples)

    cache_root = (
        args.output.resolve()
        if args.output is not None
        else frame_cache_root_from_config(raw, ROOT)
    )
    if cache_root is None:
        raise ValueError("frame cache is disabled in config; enable it or pass --output")

    print(f"Building frame cache from {len(samples)} samples into {cache_root}")
    summary = build_frame_cache(
        samples,
        project_root=ROOT,
        cache_root=cache_root,
        preprocess_config=preprocess_config,
        force=args.force,
        progress=lambda index, total, video, frames: print(
            f"[{index}/{total}] {video}: unique_frames={frames}",
            flush=True,
        ),
    )

    manifest = {
        "cache_root": str(cache_root),
        "preprocess": {
            "input_width": preprocess_config.input_width,
            "input_height": preprocess_config.input_height,
            "mask_touch_area": preprocess_config.mask_touch_area,
            "touch_roi": preprocess_config.touch_roi,
        },
        "summary": {
            "videos": summary["videos"],
            "frames": summary["frames"],
            "built": summary["built"],
            "cached": summary["cached"],
        },
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(
        "Done: "
        f"videos={summary['videos']}, "
        f"unique_frames={summary['frames']}, "
        f"built={summary['built']}, "
        f"cached={summary['cached']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
