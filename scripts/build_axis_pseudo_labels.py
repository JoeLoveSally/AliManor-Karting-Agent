#!/usr/bin/env python3
"""Build high-confidence current-frame road-axis pseudo-labels for v4-C1."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
import math
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
    RectilinearGeometryConfig,
    RoadMaskConfig,
    estimate_rectilinear_geometry,
    extract_road_mask,
    road_mask_metrics,
)
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build straight-road axial-orientation labels for v4-C1."
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "processed" / "v3" / "samples.jsonl",
    )
    parser.add_argument(
        "--geometry-config",
        type=Path,
        default=ROOT / "configs" / "geometry_pseudo_labels.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "v4c1" / "axis_labels.jsonl",
    )
    return parser.parse_args()


def _triplet(values: object, *, name: str) -> tuple[int, int, int]:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def load_teacher_config(
    path: Path,
) -> tuple[RoadMaskConfig, RectilinearGeometryConfig, dict[str, float]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("geometry config root must be a mapping")
    road = raw.get("road_mask", {})
    rect = raw.get("rectilinear", {})
    axis = raw.get("axis_label", {})
    if not isinstance(road, dict) or not isinstance(rect, dict) or not isinstance(axis, dict):
        raise ValueError("road_mask, rectilinear and axis_label must be mappings")

    ranges = []
    for index, item in enumerate(road.get("hsv_ranges", [])):
        if not isinstance(item, dict):
            raise ValueError(f"road_mask.hsv_ranges[{index}] must be a mapping")
        ranges.append(
            (
                _triplet(item.get("lower"), name=f"hsv_ranges[{index}].lower"),
                _triplet(item.get("upper"), name=f"hsv_ranges[{index}].upper"),
            )
        )
    road_config = RoadMaskConfig(
        hsv_lower=_triplet(road.get("hsv_lower", [103, 90, 50]), name="road_mask.hsv_lower"),
        hsv_upper=_triplet(road.get("hsv_upper", [130, 255, 220]), name="road_mask.hsv_upper"),
        hsv_ranges=tuple(ranges),
        close_kernel=int(road.get("close_kernel", 9)),
        open_kernel=int(road.get("open_kernel", 3)),
        min_component_area_fraction=float(road.get("min_component_area_fraction", 0.01)),
        max_components=int(road.get("max_components", 4)),
    )
    geometry_config = RectilinearGeometryConfig(
        canny_low=int(rect.get("canny_low", 40)),
        canny_high=int(rect.get("canny_high", 120)),
        hough_threshold=int(rect.get("hough_threshold", 24)),
        min_line_length_fraction=float(rect.get("min_line_length_fraction", 0.16)),
        max_line_gap_fraction=float(rect.get("max_line_gap_fraction", 0.025)),
        axis_tolerance_deg=float(rect.get("axis_tolerance_deg", 10.0)),
        min_axis_separation_deg=float(rect.get("min_axis_separation_deg", 25.0)),
        min_road_width_fraction=float(rect.get("min_road_width_fraction", 0.025)),
        max_road_width_fraction=float(rect.get("max_road_width_fraction", 0.22)),
        min_axis_overlap_fraction=float(rect.get("min_axis_overlap_fraction", 0.06)),
        corner_extension_fraction=float(rect.get("corner_extension_fraction", 0.08)),
        anchor_x_norm=float(rect.get("anchor_x_norm", 0.50)),
        anchor_y_norm=float(rect.get("anchor_y_norm", 0.58)),
    )
    road_config.validate()
    geometry_config.validate()

    gate = {
        "min_straight_confidence": float(axis.get("min_straight_confidence", 0.65)),
        "max_corner_score": float(axis.get("max_corner_score", 0.20)),
        "min_road_area_fraction": float(axis.get("min_road_area_fraction", 0.15)),
        "max_road_area_fraction": float(axis.get("max_road_area_fraction", 0.65)),
    }
    if not 0.0 <= gate["min_straight_confidence"] <= 1.0:
        raise ValueError("min_straight_confidence must be in [0,1]")
    if not 0.0 <= gate["max_corner_score"] <= 1.0:
        raise ValueError("max_corner_score must be in [0,1]")
    if not 0.0 <= gate["min_road_area_fraction"] < gate["max_road_area_fraction"] <= 1.0:
        raise ValueError("invalid road-area gate")
    return road_config, geometry_config, gate


def axis_target(angle_deg: float) -> tuple[float, float]:
    theta2 = math.radians(2.0 * angle_deg)
    return math.cos(theta2), math.sin(theta2)


def label_weight(
    *,
    primary_angle_deg: float | None,
    straight_confidence: float,
    corner_score: float,
    road_area_fraction: float,
    gate: dict[str, float],
) -> float:
    if primary_angle_deg is None:
        return 0.0
    if straight_confidence < gate["min_straight_confidence"]:
        return 0.0
    if corner_score > gate["max_corner_score"]:
        return 0.0
    if not gate["min_road_area_fraction"] <= road_area_fraction <= gate["max_road_area_fraction"]:
        return 0.0
    return float(np.clip(straight_confidence * (1.0 - corner_score), 0.0, 1.0))


def main() -> int:
    args = parse_args()
    samples = load_v3_samples(args.samples.resolve())
    road_config, geometry_config, gate = load_teacher_config(args.geometry_config.resolve())

    requested: dict[str, set[int]] = defaultdict(set)
    for sample in samples:
        requested[sample.video].add(int(sample.input_frame_indices[-1]))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    video_summaries: dict[str, dict[str, float | int]] = {}

    for video, desired in sorted(requested.items()):
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = ROOT / video_path
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")

        max_frame = max(desired)
        found = 0
        accepted = 0
        detected = 0
        weight_sum = 0.0
        frame_index = 0
        try:
            while frame_index <= max_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in desired:
                    mask = extract_road_mask(frame, road_config)
                    geometry = estimate_rectilinear_geometry(mask, geometry_config)
                    metrics = road_mask_metrics(mask, geometry=geometry)
                    area = float(metrics["road_area_fraction"])
                    angle = geometry.primary_angle_deg
                    if angle is not None:
                        detected += 1
                        target_x, target_y = axis_target(angle)
                    else:
                        target_x, target_y = 0.0, 0.0
                    weight = label_weight(
                        primary_angle_deg=angle,
                        straight_confidence=geometry.straight_confidence,
                        corner_score=geometry.corner_score,
                        road_area_fraction=area,
                        gate=gate,
                    )
                    if weight > 0.0:
                        accepted += 1
                        weight_sum += weight
                    rows.append(
                        {
                            "video": video,
                            "frame_index": frame_index,
                            "angle_deg": angle,
                            "target_x": target_x,
                            "target_y": target_y,
                            "weight": weight,
                            "straight_confidence": geometry.straight_confidence,
                            "corner_score": geometry.corner_score,
                            "road_area_fraction": area,
                        }
                    )
                    found += 1
                frame_index += 1
        finally:
            capture.release()

        if found != len(desired):
            raise RuntimeError(
                f"video ended before all requested frames were labeled: {video} "
                f"found={found} expected={len(desired)}"
            )
        summary = {
            "frames": found,
            "axis_detected": detected,
            "axis_accepted": accepted,
            "detected_rate": detected / found if found else 0.0,
            "accepted_rate": accepted / found if found else 0.0,
            "mean_accepted_weight": weight_sum / accepted if accepted else 0.0,
        }
        video_summaries[video] = summary
        print(
            f"{video}: frames={found} detected={summary['detected_rate']:.3f} "
            f"accepted={summary['accepted_rate']:.3f} "
            f"mean_weight={summary['mean_accepted_weight']:.3f}",
            flush=True,
        )

    rows.sort(key=lambda row: (str(row["video"]), int(row["frame_index"])))
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    total = len(rows)
    accepted_total = sum(int(summary["axis_accepted"]) for summary in video_summaries.values())
    detected_total = sum(int(summary["axis_detected"]) for summary in video_summaries.values())
    manifest = {
        "samples": str(args.samples.resolve()),
        "geometry_config": str(args.geometry_config.resolve()),
        "axis_encoding": "cos_2theta_sin_2theta",
        "gate": gate,
        "summary": {
            "frames": total,
            "axis_detected": detected_total,
            "axis_accepted": accepted_total,
            "detected_rate": detected_total / total if total else 0.0,
            "accepted_rate": accepted_total / total if total else 0.0,
        },
        "videos": video_summaries,
    }
    manifest_path = output.with_name("axis_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Overall: frames={total} detected={manifest['summary']['detected_rate']:.3f} "
        f"accepted={manifest['summary']['accepted_rate']:.3f}",
        flush=True,
    )
    print(f"Labels: {output}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
