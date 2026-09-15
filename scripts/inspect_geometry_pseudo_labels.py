#!/usr/bin/env python3
"""Inspect v4 road pseudo-labels on representative video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.train.geometry_pseudo_labels import (  # noqa: E402
    RoadMaskConfig,
    extract_road_mask,
    overlay_road_geometry,
    road_mask_metrics,
    row_centerline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sparse road-mask/centerline previews for v4 pseudo-label inspection."
    )
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "geometry_pseudo_labels.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "geometry_pseudo_labels",
    )
    parser.add_argument("--sample-every-frames", type=int, default=None)
    parser.add_argument("--max-previews-per-video", type=int, default=20)
    return parser.parse_args()


def load_config(path: Path) -> tuple[RoadMaskConfig, dict[str, object]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("geometry pseudo-label config root must be a mapping")
    road = raw.get("road_mask", {})
    preview = raw.get("preview", {})
    if not isinstance(road, dict) or not isinstance(preview, dict):
        raise ValueError("road_mask and preview config values must be mappings")

    config = RoadMaskConfig(
        hsv_lower=tuple(int(value) for value in road.get("hsv_lower", [103, 90, 50])),
        hsv_upper=tuple(int(value) for value in road.get("hsv_upper", [130, 255, 220])),
        close_kernel=int(road.get("close_kernel", 9)),
        open_kernel=int(road.get("open_kernel", 3)),
        min_component_area_fraction=float(road.get("min_component_area_fraction", 0.02)),
    )
    config.validate()
    return config, preview


def write_contact_sheet(images: list[np.ndarray], path: Path, columns: int = 4) -> None:
    if not images:
        return
    thumb_width = 180
    thumb_height = int(round(images[0].shape[0] * thumb_width / images[0].shape[1]))
    thumbs = [cv2.resize(image, (thumb_width, thumb_height)) for image in images]
    rows = (len(thumbs) + columns - 1) // columns
    canvas = np.zeros((rows * thumb_height, columns * thumb_width, 3), dtype=np.uint8)
    for index, image in enumerate(thumbs):
        row, column = divmod(index, columns)
        y0 = row * thumb_height
        x0 = column * thumb_width
        canvas[y0 : y0 + thumb_height, x0 : x0 + thumb_width] = image
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write contact sheet: {path}")


def write_overview_sheet(
    summaries: list[dict[str, object]],
    path: Path,
    *,
    columns: int = 3,
) -> None:
    """Create one audit image containing every per-video contact sheet."""
    tiles: list[np.ndarray] = []
    for summary in summaries:
        contact_sheet = Path(str(summary["contact_sheet"]))
        image = cv2.imread(str(contact_sheet))
        if image is None:
            continue
        target_width = 720
        target_height = max(1, round(image.shape[0] * target_width / image.shape[1]))
        tile = cv2.resize(image, (target_width, target_height))
        label_height = 42
        labeled = np.zeros(
            (tile.shape[0] + label_height, tile.shape[1], 3), dtype=np.uint8
        )
        labeled[label_height:, :] = tile
        video_name = Path(str(summary["video"])).name
        cv2.putText(
            labeled,
            video_name,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(labeled)

    if not tiles:
        return

    tile_width = max(tile.shape[1] for tile in tiles)
    tile_height = max(tile.shape[0] for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    canvas = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        y0 = row * tile_height
        x0 = column * tile_width
        canvas[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile

    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write overview sheet: {path}")


def inspect_video(
    video: Path,
    *,
    output_dir: Path,
    road_config: RoadMaskConfig,
    preview_config: dict[str, object],
    sample_every_frames: int,
    max_previews: int,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    stem_dir = output_dir / video.stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    row_step = int(preview_config.get("row_step", 16))
    min_run_width = int(preview_config.get("min_run_width", 12))
    rows: list[dict[str, object]] = []
    preview_images: list[np.ndarray] = []
    written = 0
    frame_index = 0
    try:
        while written < max_previews:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                break

            mask = extract_road_mask(frame, road_config)
            centerline = row_centerline(
                mask,
                row_step=row_step,
                min_run_width=min_run_width,
            )
            metrics = road_mask_metrics(mask, centerline)
            overlay = overlay_road_geometry(frame, mask, centerline)
            cv2.putText(
                overlay,
                f"frame={frame_index} road={metrics['road_area_fraction']:.2f}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            preview_path = stem_dir / f"frame_{frame_index:06d}.jpg"
            mask_path = stem_dir / f"frame_{frame_index:06d}_mask.png"
            if not cv2.imwrite(str(preview_path), overlay):
                raise RuntimeError(f"failed to write preview: {preview_path}")
            if not cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"failed to write mask: {mask_path}")
            preview_images.append(overlay)

            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": (frame_index / fps * 1000.0) if fps > 0 else None,
                    "preview": str(preview_path),
                    "mask": str(mask_path),
                    "centerline": [
                        {"y": int(y), "x_norm": float(x_norm)}
                        for y, x_norm in centerline
                    ],
                    **metrics,
                }
            )
            written += 1
            frame_index += sample_every_frames
            if frame_index >= frame_count:
                break
    finally:
        capture.release()

    contact_sheet = stem_dir / "contact_sheet.jpg"
    write_contact_sheet(preview_images, contact_sheet)
    summary = {
        "video": str(video),
        "fps": fps,
        "frame_count": frame_count,
        "sample_every_frames": sample_every_frames,
        "contact_sheet": str(contact_sheet),
        "previews": rows,
    }
    summary_path = stem_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    if args.max_previews_per_video < 1:
        raise ValueError("--max-previews-per-video must be >= 1")

    road_config, preview_config = load_config(args.config.resolve())
    sample_every_frames = (
        int(args.sample_every_frames)
        if args.sample_every_frames is not None
        else int(preview_config.get("sample_every_frames", 120))
    )
    if sample_every_frames < 1:
        raise ValueError("sample_every_frames must be >= 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for video in args.videos:
        summary = inspect_video(
            video.resolve(),
            output_dir=output_dir,
            road_config=road_config,
            preview_config=preview_config,
            sample_every_frames=sample_every_frames,
            max_previews=args.max_previews_per_video,
        )
        summaries.append(summary)
        areas = [row["road_area_fraction"] for row in summary["previews"]]
        centerline = [row["centerline_valid_fraction"] for row in summary["previews"]]
        mean_area = sum(areas) / len(areas) if areas else 0.0
        mean_centerline = sum(centerline) / len(centerline) if centerline else 0.0
        print(
            f"{video}: previews={len(areas)} mean_road_area={mean_area:.3f} "
            f"mean_centerline_valid={mean_centerline:.3f} "
            f"sheet={summary['contact_sheet']}",
            flush=True,
        )

    overview_path = output_dir / "overview_contact_sheet.jpg"
    write_overview_sheet(summaries, overview_path)
    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "overview_contact_sheet": str(overview_path),
                "videos": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Overview: {overview_path}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
