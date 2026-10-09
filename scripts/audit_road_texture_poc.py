#!/usr/bin/env python3
"""Compare HSV-v2 against kart-anchored texture/geometry v3 on real game MP4s.

Read-only diagnostic outputs only; no labels, training, actuators or deployment.
Exports are flat, video-prefixed files into the *explicit* output directory.
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
from audit_road_occupancy_poc import (  # noqa: E402
    crop_and_downsample, normalize_game_frame, roi_redaction_mask,
)
from karting_agent.train.geometry_pseudo_labels import extract_road_mask  # noqa: E402
from karting_agent.train.kart_pose_pseudo_labels import estimate_kart_pose  # noqa: E402
from karting_agent.train.road_texture_poc import quantify_texture_frame  # noqa: E402

CHANNELS = (
    "local_candidate", "local_texture", "local_geometry", "local_known",
    "wide_candidate", "wide_texture", "wide_geometry", "wide_known",
)


def prepare_feature_grids(
    evidence: dict[str, np.ndarray],
    anchor_xy: tuple[float, float],
    *, grid_size: int = 32,
    local_size: tuple[int, int] = (240, 240),
    wide_size: tuple[int, int] = (360, 480),
) -> np.ndarray:
    """Create 8 explicit, consistently ordered soft feature planes."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    mask = evidence["unknown"]
    observed = np.where(mask > 0, 0, 255).astype(np.uint8)
    grids = []
    for size in (local_size, wide_size):
        for key in ("candidate", "texture", "geometry"):
            image = (np.clip(evidence[key], 0, 1) * 255).astype(np.uint8)
            _, plane, _ = crop_and_downsample(
                image, anchor_xy,
                patch_width=size[0], patch_height=size[1],
                out_size=grid_size, observed_mask=observed,
            )
            grids.append(plane)
        _, _, known = crop_and_downsample(
            observed, anchor_xy,
            patch_width=size[0], patch_height=size[1],
            out_size=grid_size,
        )
        grids.append(known)
    return np.stack(grids).astype(np.float32)


def _aligned_crop(
    image: np.ndarray, center: tuple[float, float],
    *, width=240, height=240,
) -> np.ndarray:
    transform = np.float32([
        [1, 0, width / 2 - center[0]],
        [0, 1, height / 2 - center[1]],
    ])
    return cv2.warpAffine(
        image, transform, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def preview_panel(
    frame: np.ndarray, hsv_mask: np.ndarray,
    maps: dict[str, np.ndarray], grid: np.ndarray,
    anchor: tuple[float, float],
) -> np.ndarray:
    size = (192, 192)
    color = _aligned_crop(frame, anchor)
    cv2.circle(color, (120, 120), 6, (0, 0, 255), 2)
    panels = [cv2.resize(color, size, interpolation=cv2.INTER_AREA)]
    gray_sources = [
        _aligned_crop(hsv_mask, anchor),
        _aligned_crop((maps["texture"] * 255).astype(np.uint8), anchor),
        _aligned_crop((maps["geometry"] * 255).astype(np.uint8), anchor),
        _aligned_crop((maps["candidate"] * 255).astype(np.uint8), anchor),
        cv2.resize((grid[0] * 255).astype(np.uint8), (240, 240),
                   interpolation=cv2.INTER_NEAREST),
        cv2.resize((grid[4] * 255).astype(np.uint8), (240, 240),
                   interpolation=cv2.INTER_NEAREST),
        cv2.resize((grid[3] * 255).astype(np.uint8), (240, 240),
                   interpolation=cv2.INTER_NEAREST),
    ]
    panels.extend(
        cv2.cvtColor(cv2.resize(p, size, interpolation=cv2.INTER_AREA),
                     cv2.COLOR_GRAY2BGR)
        for p in gray_sources
    )
    names = ("REAL RGB", "HSV-v2", "TEXTURE", "LINE EVIDENCE",
             "V3 CANDIDATE", "32x32 LOCAL", "32x32 WIDE", "LOCAL KNOWN")
    for name, p in zip(names, panels):
        cv2.rectangle(p, (0, 0), (192, 21), (20, 20, 20), -1)
        cv2.putText(
            p, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
            .39, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return np.concatenate(panels, axis=1)


def contact_sheet(cards: list[np.ndarray], output: Path) -> None:
    if not cards:
        raise ValueError("empty contact sheet")
    # Keep original per-frame panel files only if explicitly requested.
    # A compact 1536px-wide comparison is easier to review than many folders.
    thumb_height = 120
    rows = []
    for card in cards:
        rows.append(
            cv2.resize(card, (card.shape[1] * thumb_height // card.shape[0],
                              thumb_height),
                       interpolation=cv2.INTER_AREA)
        )
    all_rows = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(output), all_rows):
        raise RuntimeError(f"could not write {output}")


def analyze_video(
    video: Path, output_dir: Path, *, config_path: Path,
    start_frame: int, stride: int, max_samples: int,
    touch_roi: tuple[float, float, float, float] | None,
    run_json: Path | None = None,
) -> dict:
    if start_frame < 0 or stride < 1 or max_samples < 1:
        raise ValueError("invalid sampling controls")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output_dir}")
    paths = {
        "contact_sheet": output_dir / f"{video.stem}_v3_contact.jpg",
        "grids": output_dir / f"{video.stem}_v3_features.npz",
        "metadata": output_dir / f"{video.stem}_v3_metadata.jsonl",
        "summary": output_dir / f"{video.stem}_v3_summary.json",
    }
    exists = [str(v) for v in paths.values() if v.exists()]
    if exists:
        raise FileExistsError(f"refusing to overwrite existing results: {exists}")
    road_cfg, _, _ = load_road_config(config_path)
    pose_cfg, _ = load_kart_configs(config_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    timestamps = (
        load_run_timestamps(run_json, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if run_json is not None else None
    )
    if timestamps is None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("raw expert video requires nominal FPS")
    rows = []
    grids = []
    cards = []
    index = 0
    try:
        while len(rows) < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if index < start_frame or (index - start_frame) % stride:
                index += 1
                continue
            frame = normalize_game_frame(frame, target_size=(360, 800))
            pose, _ = estimate_kart_pose(frame, pose_cfg)
            xy = ((pose.center_x, pose.center_y) if pose else None)
            anchor = xy if xy is not None else (198.0, 400.0)
            redaction = (roi_redaction_mask(frame.shape[:2], touch_roi)
                         if touch_roi is not None else None)
            maps, properties = quantify_texture_frame(
                frame, xy, redaction_mask=redaction,
            )
            grid = prepare_feature_grids(maps, anchor)
            baseline = extract_road_mask(frame, road_cfg)
            # Redact only after calculating the baseline, to avoid changing the
            # comparison algorithm itself. No action UI is exported to the
            # v3 grids.
            if redaction is not None:
                baseline[redaction > 0] = 0
            cards.append(preview_panel(frame, baseline, maps, grid, anchor))
            if timestamps is not None:
                if index >= len(timestamps):
                    raise ValueError("source frame exceeds recorded timestamps")
                ts = float(timestamps[index])
                timing = "source_frame_timestamp_ms_from_adb_json"
            else:
                ts = index * 1000.0 / fps
                timing = "nominal_mp4_fps_unverified"
            rows.append({
                "frame_index": index, "timestamp_ms": ts,
                "timestamp_source": timing,
                "kart_xy": list(xy) if xy is not None else None,
                **properties,
            })
            grids.append(grid.astype(np.float16))
            index += 1
    finally:
        cap.release()
    if not rows:
        raise ValueError("no gameplay samples decoded")
    np.savez_compressed(
        paths["grids"], grids=np.stack(grids),
        source_frame_indices=np.asarray(
            [r["frame_index"] for r in rows], dtype=np.int32
        ),
        source_timestamp_ms=np.asarray(
            [r["timestamp_ms"] for r in rows], dtype=np.float64
        ),
        candidate_valid_unreviewed=np.asarray(
            [r["candidate_valid_unreviewed"] for r in rows],
            dtype=np.bool_,
        ),
    )
    with paths["metadata"].open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    contact_sheet(cards, paths["contact_sheet"])
    summary = {
        "video": str(video), "samples": len(rows),
        "video_size": [width, height],
        "canonical_size": [360, 800],
        "model_input_not_approved": True,
        "label_supervision_available": False,
        "action_control_executed": False,
        "contact_sheet": str(paths["contact_sheet"]),
        "channel_order": list(CHANNELS),
        "shape": [len(rows), 8, 32, 32],
        "valid_candidates_unreviewed": sum(
            int(r["candidate_valid_unreviewed"]) for r in rows
        ),
        "mean_texture_gt_0_2": float(np.mean(
            [r["texture_fraction_gt_0_2"] for r in rows]
        )),
        "mean_candidate_gt_0_2": float(np.mean(
            [r["candidate_fraction_gt_0_2"] for r in rows]
        )),
        "timestamp_source": rows[0]["timestamp_source"],
        "redaction_roi": list(touch_roi) if touch_roi else None,
        "note": (
            "Compare shapes in the actual RGB frames. Texture and geometry "
            "are heuristic scores, not calibrated road probabilities. "
            "Candidate-valid measures a seed only, NOT mask accuracy. "
            "No automated steering and no policy training are performed."
        ),
    }
    with paths["summary"].open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--geometry-config", type=Path,
                    default=ROOT / "configs/geometry_pseudo_labels.yaml")
    ap.add_argument("--frame-start", type=int, default=240)
    ap.add_argument("--sample-every-frames", type=int, default=120)
    ap.add_argument("--max-samples", type=int, default=12)
    ap.add_argument("--touch-roi", nargs=4, type=float, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--run-json", type=Path, default=None,
                    help="Only valid with one ADB MP4")
    args = ap.parse_args()
    if args.run_json is not None and len(args.videos) != 1:
        ap.error("--run-json requires exactly one video")
    for video in args.videos:
        report = analyze_video(
            video, args.output_dir, config_path=args.geometry_config,
            start_frame=args.frame_start,
            stride=args.sample_every_frames, max_samples=args.max_samples,
            touch_roi=(tuple(args.touch_roi) if args.touch_roi else None),
            run_json=args.run_json,
        )
        print(json.dumps({
            "video": video.name, "samples": report["samples"],
            "valid_candidates_unreviewed": report["valid_candidates_unreviewed"],
            "mean_texture_gt_0_2": report["mean_texture_gt_0_2"],
            "mean_candidate_gt_0_2": report["mean_candidate_gt_0_2"],
            "contact_sheet": report["contact_sheet"],
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
