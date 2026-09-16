#!/usr/bin/env python3
"""Audit kart center/heading and local road-relative geometry on videos."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from inspect_geometry_pseudo_labels import (  # noqa: E402
    load_config as load_road_config,
    write_contact_sheet,
)
from karting_agent.train.geometry_pseudo_labels import (  # noqa: E402
    estimate_rectilinear_geometry,
    extract_road_mask,
    overlay_road_geometry,
)
from karting_agent.train.kart_pose_pseudo_labels import (  # noqa: E402
    KartPoseConfig,
    KartRoadRelationConfig,
    estimate_kart_pose,
    estimate_kart_road_relation,
    overlay_kart_relative_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate offline kart-center/heading and local road-relative overlays. "
            "This is a diagnostic teacher only; it never controls the game."
        )
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
        default=ROOT / "artifacts" / "kart_relative_geometry",
    )
    parser.add_argument("--sample-every-frames", type=int, default=15)
    parser.add_argument("--max-previews-per-video", type=int, default=60)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument(
        "--run-json",
        type=Path,
        default=None,
        help="Optional matching ADB run JSON; supported for a single video.",
    )
    return parser.parse_args()


def _triplet(values: object, *, name: str) -> tuple[int, int, int]:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def _hsv_ranges(values: object, *, name: str) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    parsed = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"{name}[{index}] must be a mapping")
        parsed.append(
            (
                _triplet(item.get("lower"), name=f"{name}[{index}].lower"),
                _triplet(item.get("upper"), name=f"{name}[{index}].upper"),
            )
        )
    return tuple(parsed)


def load_kart_configs(path: Path) -> tuple[KartPoseConfig, KartRoadRelationConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("geometry pseudo-label config root must be a mapping")
    pose_raw = raw.get("kart_pose", {})
    relation_raw = raw.get("kart_relation", {})
    if not isinstance(pose_raw, dict) or not isinstance(relation_raw, dict):
        raise ValueError("kart_pose/kart_relation config must be mappings")

    defaults = KartPoseConfig()
    warm_ranges = (
        _hsv_ranges(pose_raw["warm_hsv_ranges"], name="kart_pose.warm_hsv_ranges")
        if "warm_hsv_ranges" in pose_raw
        else defaults.warm_hsv_ranges
    )
    red_ranges = (
        _hsv_ranges(pose_raw["red_hsv_ranges"], name="kart_pose.red_hsv_ranges")
        if "red_hsv_ranges" in pose_raw
        else defaults.red_hsv_ranges
    )
    pose = KartPoseConfig(
        warm_hsv_ranges=warm_ranges,
        red_hsv_ranges=red_ranges,
        search_x_min_norm=float(pose_raw.get("search_x_min_norm", defaults.search_x_min_norm)),
        search_x_max_norm=float(pose_raw.get("search_x_max_norm", defaults.search_x_max_norm)),
        search_y_min_norm=float(pose_raw.get("search_y_min_norm", defaults.search_y_min_norm)),
        search_y_max_norm=float(pose_raw.get("search_y_max_norm", defaults.search_y_max_norm)),
        close_kernel=int(pose_raw.get("close_kernel", defaults.close_kernel)),
        min_component_area_fraction=float(
            pose_raw.get("min_component_area_fraction", defaults.min_component_area_fraction)
        ),
        max_component_area_fraction=float(
            pose_raw.get("max_component_area_fraction", defaults.max_component_area_fraction)
        ),
        min_red_fraction=float(pose_raw.get("min_red_fraction", defaults.min_red_fraction)),
        anchor_x_norm=float(pose_raw.get("anchor_x_norm", defaults.anchor_x_norm)),
        anchor_y_norm=float(pose_raw.get("anchor_y_norm", defaults.anchor_y_norm)),
        anchor_distance_weight=float(
            pose_raw.get("anchor_distance_weight", defaults.anchor_distance_weight)
        ),
        join_distance_fraction=float(
            pose_raw.get("join_distance_fraction", defaults.join_distance_fraction)
        ),
        min_heading_quality=float(
            pose_raw.get("min_heading_quality", defaults.min_heading_quality)
        ),
    )
    pose.validate()

    relation_defaults = KartRoadRelationConfig()
    offsets = relation_raw.get(
        "tangent_offsets_fraction", relation_defaults.tangent_offsets_fraction
    )
    if not isinstance(offsets, (list, tuple)):
        raise ValueError("kart_relation.tangent_offsets_fraction must be a list")
    relation = KartRoadRelationConfig(
        tangent_offsets_fraction=tuple(float(value) for value in offsets),
        max_normal_distance_fraction=float(
            relation_raw.get(
                "max_normal_distance_fraction",
                relation_defaults.max_normal_distance_fraction,
            )
        ),
        min_road_width_fraction=float(
            relation_raw.get("min_road_width_fraction", relation_defaults.min_road_width_fraction)
        ),
        max_road_width_fraction=float(
            relation_raw.get("max_road_width_fraction", relation_defaults.max_road_width_fraction)
        ),
    )
    relation.validate()
    return pose, relation


def load_run_steps(path: Path | None) -> tuple[list[int], list[dict[str, object]]]:
    if path is None:
        return [], []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
        raise ValueError("run JSON must contain a steps list")
    rows: list[tuple[int, dict[str, object]]] = []
    for step in raw["steps"]:
        if not isinstance(step, dict):
            continue
        indices = step.get("input_frame_indices")
        if not isinstance(indices, list) or not indices:
            continue
        rows.append((int(indices[-1]), step))
    rows.sort(key=lambda item: item[0])
    return [row[0] for row in rows], [row[1] for row in rows]


def nearest_run_step(
    frame_index: int,
    frames: list[int],
    steps: list[dict[str, object]],
    *,
    max_delta: int = 2,
) -> dict[str, object] | None:
    if not frames:
        return None
    position = bisect_left(frames, frame_index)
    choices = []
    if position < len(frames):
        choices.append(position)
    if position > 0:
        choices.append(position - 1)
    if not choices:
        return None
    best = min(choices, key=lambda index: abs(frames[index] - frame_index))
    return steps[best] if abs(frames[best] - frame_index) <= max_delta else None


def _text(
    image: np.ndarray,
    value: str,
    *,
    y: int,
) -> None:
    cv2.putText(
        image,
        value,
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        value,
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def inspect_video(
    video: Path,
    *,
    output_dir: Path,
    road_config,
    geometry_config,
    kart_config: KartPoseConfig,
    relation_config: KartRoadRelationConfig,
    sample_every_frames: int,
    max_previews: int,
    frame_start: int,
    frame_end: int | None,
    run_frames: list[int],
    run_steps: list[dict[str, object]],
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    last_frame = frame_count - 1 if frame_end is None else min(frame_end, frame_count - 1)
    if frame_start < 0 or frame_start > last_frame:
        raise ValueError(f"invalid frame range for {video}: {frame_start}..{last_frame}")

    stem_dir = output_dir / video.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    preview_images: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    try:
        for frame_index in range(frame_start, last_frame + 1, sample_every_frames):
            if len(rows) >= max_previews:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                break

            road_mask = extract_road_mask(frame, road_config)
            geometry = estimate_rectilinear_geometry(road_mask, geometry_config)
            pose, kart_mask = estimate_kart_pose(frame, kart_config)
            relation = (
                estimate_kart_road_relation(
                    road_mask,
                    geometry,
                    pose,
                    relation_config,
                )
                if pose is not None
                else None
            )
            overlay = overlay_road_geometry(
                frame,
                road_mask,
                geometry=geometry,
                geometry_config=geometry_config,
            )
            overlay = overlay_kart_relative_geometry(overlay, kart_mask, pose, relation)

            pose_text = "kart=NA"
            if pose is not None:
                heading = (
                    f"{pose.heading_angle_deg:.0f}deg"
                    if pose.heading_angle_deg is not None
                    else "NA"
                )
                pose_text = (
                    f"kart=({pose.center_x:.0f},{pose.center_y:.0f}) "
                    f"head={heading} hq={pose.heading_quality:.2f}"
                )
            relation_text = "relation=NA"
            if relation is not None:
                heading_error = (
                    f"{relation.heading_error_deg:+.0f}deg"
                    if relation.heading_error_deg is not None
                    else "NA"
                )
                lateral = (
                    f"{relation.lateral_offset_norm:+.2f}w"
                    if relation.lateral_offset_norm is not None
                    else "NA"
                )
                relation_text = (
                    f"road={relation.road_angle_deg:.0f} herr={heading_error} "
                    f"lat={lateral} relq={relation.confidence:.2f}"
                )
            _text(overlay, f"f={frame_index} {pose_text}", y=22)
            _text(overlay, relation_text, y=42)

            run_step = nearest_run_step(frame_index, run_frames, run_steps)
            run_payload = None
            if run_step is not None:
                probability = float(run_step.get("probability", 0.0))
                action = str(run_step.get("action", "?"))
                pressed = bool(run_step.get("pressed", False))
                _text(
                    overlay,
                    f"policy p={probability:.3f} action={action} state={'PRESS' if pressed else 'RELEASE'}",
                    y=62,
                )
                run_payload = {
                    "probability": probability,
                    "action": action,
                    "pressed": pressed,
                }

            preview_path = stem_dir / f"frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(preview_path), overlay):
                raise RuntimeError(f"failed to write preview: {preview_path}")
            preview_images.append(overlay)
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": frame_index / fps * 1000.0 if fps > 0 else None,
                    "preview": str(preview_path),
                    "geometry": geometry.to_dict(),
                    "kart_pose": pose.to_dict() if pose is not None else None,
                    "kart_road_relation": relation.to_dict() if relation is not None else None,
                    "policy": run_payload,
                }
            )
    finally:
        capture.release()

    contact_sheet = stem_dir / "contact_sheet.jpg"
    write_contact_sheet(preview_images, contact_sheet, columns=4)
    summary = {
        "video": str(video),
        "fps": fps,
        "frame_count": frame_count,
        "frame_start": frame_start,
        "frame_end": last_frame,
        "sample_every_frames": sample_every_frames,
        "contact_sheet": str(contact_sheet),
        "previews": rows,
    }
    (stem_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def summarize(summary: dict[str, object]) -> str:
    previews = summary["previews"]
    assert isinstance(previews, list)
    count = len(previews)
    poses = [row["kart_pose"] for row in previews if row["kart_pose"] is not None]
    headings = [pose for pose in poses if pose["heading_usable"]]
    relations = [
        row["kart_road_relation"]
        for row in previews
        if row["kart_road_relation"] is not None
    ]
    local = [relation for relation in relations if relation["local_road_usable"]]
    heading_errors = [
        float(relation["abs_heading_error_deg"])
        for relation in relations
        if relation["abs_heading_error_deg"] is not None
    ]
    lateral = [
        abs(float(relation["lateral_offset_norm"]))
        for relation in local
        if relation["lateral_offset_norm"] is not None
    ]
    offroad = [relation for relation in local if relation["inside_road"] is False]

    def ratio(value: int) -> float:
        return value / count if count else 0.0

    mean_heading = float(np.mean(heading_errors)) if heading_errors else 0.0
    mean_lateral = float(np.mean(lateral)) if lateral else 0.0
    return (
        f"previews={count} kart={ratio(len(poses)):.3f} "
        f"heading={ratio(len(headings)):.3f} relation={ratio(len(local)):.3f} "
        f"offroad={len(offroad) / len(local) if local else 0.0:.3f} "
        f"mean_abs_herr={mean_heading:.1f}deg mean_abs_lat={mean_lateral:.2f}w"
    )


def main() -> int:
    args = parse_args()
    if args.sample_every_frames < 1:
        raise ValueError("--sample-every-frames must be >= 1")
    if args.max_previews_per_video < 1:
        raise ValueError("--max-previews-per-video must be >= 1")
    if args.frame_start < 0:
        raise ValueError("--frame-start must be >= 0")
    if args.frame_end is not None and args.frame_end < args.frame_start:
        raise ValueError("--frame-end must be >= --frame-start")
    if args.run_json is not None and len(args.videos) != 1:
        raise ValueError("--run-json currently supports exactly one video")

    road_config, geometry_config, _ = load_road_config(args.config.resolve())
    kart_config, relation_config = load_kart_configs(args.config.resolve())
    run_frames, run_steps = load_run_steps(args.run_json.resolve() if args.run_json else None)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for video in args.videos:
        summary = inspect_video(
            video.resolve(),
            output_dir=output_dir,
            road_config=road_config,
            geometry_config=geometry_config,
            kart_config=kart_config,
            relation_config=relation_config,
            sample_every_frames=args.sample_every_frames,
            max_previews=args.max_previews_per_video,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            run_frames=run_frames,
            run_steps=run_steps,
        )
        summaries.append(summary)
        print(
            f"{video}: {summarize(summary)} sheet={summary['contact_sheet']}",
            flush=True,
        )

    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "teacher_semantics": {
                    "kart_center": "centroid of merged red/orange chassis evidence",
                    "kart_heading": "axial PCA direction of merged chassis evidence",
                    "road_orientation": "dominant road-edge axial orientation",
                    "lateral_offset": "kart displacement from local road cross-section center, normalized by half road width",
                    "heading_error": "signed axial kart-heading minus road-axis difference",
                    "training_status": "diagnostic_only_not_approved_for_supervision",
                },
                "run_json": str(args.run_json.resolve()) if args.run_json else None,
                "videos": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
