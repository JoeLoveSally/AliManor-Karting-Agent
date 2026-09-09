#!/usr/bin/env python3
"""Create a contact sheet of representative frames for map grouping."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a map-inspection contact sheet.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "map_contact_sheet.jpg",
    )
    parser.add_argument(
        "--positions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="Relative video positions to sample, each in (0, 1).",
    )
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=180)
    return parser.parse_args()


def read_frame_at(path: Path, position: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"invalid frame count: {path}")
        frame_index = min(
            frame_count - 1,
            max(0, round((frame_count - 1) * position)),
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read frame {frame_index}: {path}")
        return frame
    finally:
        capture.release()


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, round(source_w * scale))
    resized_h = max(1, round(source_h * scale))
    resized = cv2.resize(
        frame,
        (resized_w, resized_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def build_contact_sheet(
    videos: list[Path],
    positions: list[float],
    *,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    if not videos:
        raise ValueError("videos must not be empty")
    if not positions or any(not 0.0 < position < 1.0 for position in positions):
        raise ValueError("positions must be inside (0, 1)")
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile size must be positive")

    label_height = 32
    rows: list[np.ndarray] = []
    for index, video in enumerate(videos, start=1):
        tiles = [
            letterbox(read_frame_at(video, position), tile_width, tile_height)
            for position in positions
        ]
        row = np.concatenate(tiles, axis=1)
        labeled = np.zeros(
            (label_height + tile_height, row.shape[1], 3),
            dtype=np.uint8,
        )
        labeled[label_height:] = row
        label = f"{index:02d}  {video.name}"
        cv2.putText(
            labeled,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rows.append(labeled)
    return np.concatenate(rows, axis=0)


def main() -> int:
    args = parse_args()
    videos = sorted(args.input.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no MP4 videos found in {args.input}")

    sheet = build_contact_sheet(
        videos,
        list(args.positions),
        tile_width=args.tile_width,
        tile_height=args.tile_height,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(args.output),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    ):
        raise RuntimeError(f"failed to write: {args.output}")

    print(
        f"Wrote {args.output} "
        f"({len(videos)} videos x {len(args.positions)} frames)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
