#!/usr/bin/env python3
"""Read-only road-occupancy visualization and quantization POC.

Reuses the existing broad HSV road-mask teacher and warm-color kart detector.
The outputs are UNREVIEWED segmentation candidates, not road ground truth.
Never imports ADB/Armed executors or emits any game control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from inspect_kart_relative_geometry import load_kart_configs  # noqa: E402
from audit_visual_state_observability import load_run_timestamps  # noqa: E402
from karting_agent.train.geometry_pseudo_labels import extract_road_mask  # noqa: E402
from karting_agent.train.kart_pose_pseudo_labels import estimate_kart_pose  # noqa: E402


def select_kart_local_component(
    mask: np.ndarray, center_xy: tuple[float, float] | None,
    *, radius_px: int = 28, min_seed_pixels: int = 5,
) -> tuple[np.ndarray, dict]:
    """Choose a connected mask component supported in a disk around the kart.

    A support disk is NOT proof of correct road identity (large background-color
    components can include the kart). Preserve the raw mask for manual review.
    """
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("mask must be uint8 HxW")
    empty = np.zeros_like(mask)
    if center_xy is None:
        return empty, {"seed_found": False, "support_pixels": 0,
                       "reason": "kart_not_detected"}
    if radius_px <= 0 or min_seed_pixels <= 0:
        raise ValueError("radius_px and min_seed_pixels must be positive")
    x, y = center_xy
    if not (np.isfinite(x) and np.isfinite(y)):
        return empty, {"seed_found": False, "support_pixels": 0,
                       "reason": "invalid_kart_position"}
    height, width = mask.shape
    yy, xx = np.ogrid[:height, :width]
    disk = ((xx - x) ** 2 + (yy - y) ** 2 <= radius_px ** 2)
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    if count <= 1:
        return empty, {"seed_found": False, "support_pixels": 0,
                       "reason": "road_mask_empty"}
    supported = labels[disk & (labels > 0)]
    if supported.size == 0:
        return empty, {"seed_found": False, "support_pixels": 0,
                       "reason": "no_local_road_support"}
    counts = np.bincount(supported, minlength=count)
    counts[0] = 0
    chosen = int(np.argmax(counts))
    pixel_count = int(counts[chosen])
    if pixel_count < min_seed_pixels:
        return empty, {"seed_found": False, "support_pixels": pixel_count,
                       "reason": "too_few_local_road_pixels"}
    chosen_mask = np.where(labels == chosen, 255, 0).astype(np.uint8)
    return chosen_mask, {
        "seed_found": True, "support_pixels": pixel_count,
        "reason": "local_component_candidate",
        "component_area_fraction": float(np.count_nonzero(chosen_mask) / mask.size),
    }


def crop_and_downsample(
    image: np.ndarray, center_xy: tuple[float, float], *,
    patch_width: int, patch_height: int, out_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel-centered crop and area-averaged occupancy/known fractions."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 HxW")
    if patch_width <= 0 or patch_height <= 0 or out_size <= 0:
        raise ValueError("invalid patch or output size")
    cx, cy = center_xy
    shift = np.float32([[1, 0, patch_width / 2 - cx],
                        [0, 1, patch_height / 2 - cy]])
    cropped = cv2.warpAffine(
        image, shift, (patch_width, patch_height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    in_bounds = cv2.warpAffine(
        np.ones(image.shape, dtype=np.uint8) * 255,
        shift, (patch_width, patch_height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    occupancy = cv2.resize(
        cropped, (out_size, out_size), interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0
    known = cv2.resize(
        in_bounds, (out_size, out_size), interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0
    return cropped, occupancy, known


def quantize_frame(
    frame: np.ndarray, raw_mask: np.ndarray,
    kart_xy: tuple[float, float] | None, *,
    nominal_center: tuple[float, float],
    local_size: tuple[int, int] = (240, 240),
    wide_size: tuple[int, int] = (360, 480),
    grid_size: int = 32,
    seed_radius: int = 28,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Four planes: local road, local known, wide road, wide known.

    When no local seed exists, the occupancy planes remain zeros but metadata
    marks the sample as INVALID. Zero occupancy alone must never be treated as
    confidently observed off-road or used for policy training.
    """
    selected, info = select_kart_local_component(
        raw_mask, kart_xy, radius_px=seed_radius
    )
    center = kart_xy if kart_xy is not None else nominal_center
    cropped, local_road, local_known = crop_and_downsample(
        selected, center, patch_width=local_size[0],
        patch_height=local_size[1], out_size=grid_size,
    )
    _, wide_road, wide_known = crop_and_downsample(
        selected, center, patch_width=wide_size[0],
        patch_height=wide_size[1], out_size=grid_size,
    )
    planes = np.stack((local_road, local_known, wide_road, wide_known)).astype(np.float32)
    info = dict(info)
    info.update({
        "kart_detected": kart_xy is not None,
        "anchor_x_px": float(center[0]), "anchor_y_px": float(center[1]),
        "quantizer_valid_unreviewed": kart_xy is not None and info["seed_found"],
        "raw_road_pixel_fraction": float(np.count_nonzero(raw_mask) / raw_mask.size),
        "selected_road_pixel_fraction": float(np.count_nonzero(selected) / selected.size),
        "local_observed_coverage": float(local_known.mean()),
        "wide_observed_coverage": float(wide_known.mean()),
    })
    return planes, info, {"selected_mask": selected, "local_crop": cropped}


def tile_preview(frame: np.ndarray, raw: np.ndarray, selected: np.ndarray,
                 occupancy: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    size = (256, 256)
    original = frame.copy()
    cv2.circle(original, (int(center[0]), int(center[1])), 7,
               (0, 0, 255), 2)
    original = cv2.resize(original, size)
    raw_vis = cv2.cvtColor(cv2.resize(raw, size), cv2.COLOR_GRAY2BGR)
    selected_vis = cv2.cvtColor(cv2.resize(selected, size), cv2.COLOR_GRAY2BGR)
    q = cv2.resize((occupancy * 255).astype(np.uint8), size,
                   interpolation=cv2.INTER_NEAREST)
    quant_vis = cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)
    panels = [original, raw_vis, selected_vis, quant_vis]
    for name, panel in zip(("RGB", "HSV mask", "near-kart component", "32x32 local"),
                           panels):
        cv2.putText(panel, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 0, 200), 1, cv2.LINE_AA)
    return np.concatenate(panels, axis=1)


def inspect_video(video: Path, json_path: Path, *,
                  output_root: Path, start: int, end: int | None,
                  stride: int, max_samples: int,
                  road_cfg, kart_cfg, grid_size: int = 32) -> dict:
    output_dir = output_root / video.stem
    output_dir.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    timestamps = load_run_timestamps(json_path, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    assert timestamps is not None
    rows = []
    masks = []
    cards = []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (end is not None and i > end) or len(rows) >= max_samples:
                break
            if i < start or (i - start) % stride:
                i += 1
                continue
            if i >= len(timestamps):
                raise ValueError("decoded source frame index out of timestamp bounds")
            road = extract_road_mask(frame, road_cfg)
            pose, _ = estimate_kart_pose(frame, kart_cfg)
            xy = (pose.center_x, pose.center_y) if pose else None
            default_center = (frame.shape[1] * .55, frame.shape[0] * .50)
            planes, audit, images = quantize_frame(
                frame, road, xy, nominal_center=default_center,
                grid_size=grid_size,
            )
            t = timestamps[i]
            rows.append({"frame_index": i, "timestamp_ms": t, **audit})
            masks.append(planes.astype(np.float16))
            if len(rows) <= 20 or len(rows) % 8 == 0:
                preview = tile_preview(
                    frame, road, images["selected_mask"], planes[0],
                    xy if xy is not None else default_center,
                )
                p = output_dir / f"preview_{i:06d}.jpg"
                if not cv2.imwrite(str(p), preview):
                    raise RuntimeError(f"cannot write preview: {p}")
                cards.append(preview)
            i += 1
    finally:
        cap.release()
    if not rows:
        raise ValueError("no sampled frames")
    np.savez_compressed(
        output_dir / "occupancy.npz",
        grids=np.stack(masks), frame_indices=np.asarray(
            [x["frame_index"] for x in rows], dtype=np.int32
        ),
        timestamps_ms=np.asarray(
            [x["timestamp_ms"] for x in rows], dtype=np.float64
        ),
        quantizer_valid=np.asarray(
            [x["quantizer_valid_unreviewed"] for x in rows], dtype=np.bool_
        ),
    )
    with (output_dir / "metadata.jsonl").open("x", encoding="utf-8") as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "video": str(video), "samples": len(rows),
        "candidate_valid_rate_unreviewed": sum(
            r["quantizer_valid_unreviewed"] for r in rows
        ) / len(rows),
        "grid_shape": [4, grid_size, grid_size],
        "channel_order": ["local_road", "local_known", "wide_road", "wide_known"],
        "grid_dtype": "float16",
        "mask_quality_verified": False,
        "warning": (
            "This candidate inherits HSV segmentation limitations; near-kart "
            "component association can still select background or a wrong road. "
            "Do not use for real control before visual validation."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # A compact overview, preserving the full-size per-frame diagnostic panels.
    if cards:
        thumb_w, thumb_h = 512, 128
        columns = 2
        rows_count = (len(cards) + columns - 1) // columns
        sheet = np.zeros((rows_count * thumb_h, columns * thumb_w, 3),
                         dtype=np.uint8)
        for k, card in enumerate(cards):
            small = cv2.resize(card, (thumb_w, thumb_h),
                               interpolation=cv2.INTER_AREA)
            r, c = divmod(k, columns)
            sheet[r * thumb_h:(r + 1) * thumb_h,
                  c * thumb_w:(c + 1) * thumb_w] = small
        cv2.imwrite(str(output_dir / "contact_sheet.jpg"), sheet)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path,
                        default=ROOT / "configs/geometry_pseudo_labels.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--sample-every-frames", type=int, default=9)
    parser.add_argument("--max-samples", type=int, default=120)
    args = parser.parse_args()
    if (args.frame_start < 0 or args.sample_every_frames < 1 or args.max_samples < 1
            or (args.frame_end is not None and args.frame_end < args.frame_start)):
        parser.error("invalid frame range/sampling")
    road_cfg, _, _ = load_road_config(args.geometry_config)
    kart_cfg, _ = load_kart_configs(args.geometry_config)
    report = inspect_video(
        args.video, args.run_json, output_root=args.output_root,
        start=args.frame_start, end=args.frame_end,
        stride=args.sample_every_frames, max_samples=args.max_samples,
        road_cfg=road_cfg, kart_cfg=kart_cfg,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
