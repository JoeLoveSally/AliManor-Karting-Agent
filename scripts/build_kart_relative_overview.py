#!/usr/bin/env python3
"""Combine kart-relative per-video contact sheets into one audit overview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from inspect_geometry_pseudo_labels import write_overview_sheet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one overview image from kart-relative contact sheets."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "artifacts" / "kart_relative_geometry" / "expert_audit",
    )
    parser.add_argument("--columns", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.columns < 1:
        raise ValueError("--columns must be >= 1")
    input_dir = args.input_dir.resolve()
    index_path = input_dir / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    summaries = payload.get("videos")
    if not isinstance(summaries, list) or not summaries:
        raise ValueError(f"index contains no videos: {index_path}")
    if not all(isinstance(item, dict) for item in summaries):
        raise ValueError("index videos must contain mappings")

    output = input_dir / "overview_contact_sheet.jpg"
    write_overview_sheet(summaries, output, columns=args.columns)
    print(f"Overview: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
