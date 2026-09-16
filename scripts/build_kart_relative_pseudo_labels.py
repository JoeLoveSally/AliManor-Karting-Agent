#!/usr/bin/env python3
"""Build current-frame kart-relative pseudo-labels for v4-C2."""

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
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from inspect_kart_relative_geometry import load_kart_configs  # noqa: E402
from karting_agent.train.geometry_pseudo_labels import (  # noqa: E402
    estimate_rectilinear_geometry,
    extract_road_mask,
)
from karting_agent.train.kart_pose_pseudo_labels import (  # noqa: E402
    estimate_kart_pose,
    estimate_kart_road_relation,
)
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build kart-relative lateral/heading weak labels for v4-C2."
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
        default=ROOT / "data" / "processed" / "v4c2" / "kart_relative_labels.jsonl",
    )
    return parser.parse_args()


def heading_error_target(error_deg: float) -> tuple[float, float]:
    """Encode signed axial heading error without a discontinuity at +/-90 deg."""

    doubled = math.radians(2.0 * error_deg)
    return math.cos(doubled), math.sin(doubled)


def load_label_gate(path: Path) -> dict[str, float]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("geometry config root must be a mapping")
    values = raw.get("kart_relative_label", {})
    if not isinstance(values, dict):
        raise ValueError("kart_relative_label must be a mapping")
    gate = {
        "min_confidence": float(values.get("min_confidence", 0.05)),
        "lateral_clip_abs": float(values.get("lateral_clip_abs", 1.25)),
        "edge_risk_threshold": float(values.get("edge_risk_threshold", 0.70)),
    }
    if not 0.0 <= gate["min_confidence"] <= 1.0:
        raise ValueError("min_confidence must be in [0,1]")
    if gate["lateral_clip_abs"] <= 0.0:
        raise ValueError("lateral_clip_abs must be > 0")
    if gate["edge_risk_threshold"] <= 0.0:
        raise ValueError("edge_risk_threshold must be > 0")
    return gate


def main() -> int:
    args = parse_args()
    samples = load_v3_samples(args.samples.resolve())
    config_path = args.geometry_config.resolve()
    road_config, geometry_config, _ = load_road_config(config_path)
    kart_config, relation_config = load_kart_configs(config_path)
    gate = load_label_gate(config_path)

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
        pose_found = 0
        heading_found = 0
        relation_found = 0
        accepted = 0
        edge_risk_positive = 0
        offroad = 0
        accepted_weight_sum = 0.0
        accepted_abs_lateral_sum = 0.0
        accepted_abs_heading_sum = 0.0
        frame_index = 0
        try:
            while frame_index <= max_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in desired:
                    pose, _ = estimate_kart_pose(frame, kart_config)
                    if pose is not None:
                        pose_found += 1
                    if pose is not None and pose.heading_angle_deg is not None:
                        heading_found += 1

                    relation = None
                    if pose is not None and pose.heading_angle_deg is not None:
                        road_mask = extract_road_mask(frame, road_config)
                        geometry = estimate_rectilinear_geometry(
                            road_mask, geometry_config
                        )
                        relation = estimate_kart_road_relation(
                            road_mask,
                            geometry,
                            pose,
                            relation_config,
                        )
                    if relation is not None:
                        relation_found += 1

                    raw_lateral = (
                        None if relation is None else float(relation.lateral_offset_norm)
                    )
                    heading_error = (
                        None if relation is None else relation.heading_error_deg
                    )
                    confidence = 0.0 if relation is None else float(relation.confidence)
                    weight = (
                        confidence
                        if relation is not None
                        and heading_error is not None
                        and confidence >= gate["min_confidence"]
                        else 0.0
                    )
                    if weight > 0.0:
                        assert raw_lateral is not None
                        assert heading_error is not None
                        accepted += 1
                        accepted_weight_sum += weight
                        accepted_abs_lateral_sum += abs(raw_lateral)
                        accepted_abs_heading_sum += abs(float(heading_error))

                    lateral_target = (
                        0.0
                        if raw_lateral is None
                        else float(
                            np.clip(
                                raw_lateral,
                                -gate["lateral_clip_abs"],
                                gate["lateral_clip_abs"],
                            )
                        )
                    )
                    if heading_error is None:
                        heading_x, heading_y = 0.0, 0.0
                    else:
                        heading_x, heading_y = heading_error_target(float(heading_error))
                    edge_risk = (
                        1.0
                        if raw_lateral is not None
                        and abs(raw_lateral) >= gate["edge_risk_threshold"]
                        else 0.0
                    )
                    if weight > 0.0 and edge_risk > 0.5:
                        edge_risk_positive += 1
                    if weight > 0.0 and relation is not None and not relation.inside_road:
                        offroad += 1

                    rows.append(
                        {
                            "video": video,
                            "frame_index": frame_index,
                            "lateral_target": lateral_target,
                            "heading_target_x": heading_x,
                            "heading_target_y": heading_y,
                            "edge_risk_target": edge_risk,
                            "weight": weight,
                            "raw_lateral_offset_norm": raw_lateral,
                            "heading_error_deg": heading_error,
                            "inside_road": (
                                None if relation is None else relation.inside_road
                            ),
                            "confidence": confidence,
                            "heading_quality": (
                                0.0 if pose is None else pose.heading_quality
                            ),
                            "road_support_fraction": (
                                0.0
                                if relation is None
                                else relation.road_support_fraction
                            ),
                            "valid_cross_sections": (
                                0 if relation is None else relation.valid_cross_sections
                            ),
                            "cross_section_count": (
                                0 if relation is None else relation.cross_section_count
                            ),
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
            "pose_found": pose_found,
            "heading_found": heading_found,
            "relation_found": relation_found,
            "accepted": accepted,
            "pose_rate": pose_found / found if found else 0.0,
            "heading_rate": heading_found / found if found else 0.0,
            "relation_rate": relation_found / found if found else 0.0,
            "accepted_rate": accepted / found if found else 0.0,
            "edge_risk_positive_rate": (
                edge_risk_positive / accepted if accepted else 0.0
            ),
            "offroad_rate": offroad / accepted if accepted else 0.0,
            "mean_accepted_weight": (
                accepted_weight_sum / accepted if accepted else 0.0
            ),
            "mean_abs_lateral": (
                accepted_abs_lateral_sum / accepted if accepted else 0.0
            ),
            "mean_abs_heading_error_deg": (
                accepted_abs_heading_sum / accepted if accepted else 0.0
            ),
        }
        video_summaries[video] = summary
        print(
            f"{video}: frames={found} pose={summary['pose_rate']:.3f} "
            f"heading={summary['heading_rate']:.3f} "
            f"relation={summary['relation_rate']:.3f} "
            f"accepted={summary['accepted_rate']:.3f} "
            f"risk+={summary['edge_risk_positive_rate']:.3f} "
            f"offroad={summary['offroad_rate']:.3f} "
            f"weight={summary['mean_accepted_weight']:.3f} "
            f"|lat|={summary['mean_abs_lateral']:.3f} "
            f"|herr|={summary['mean_abs_heading_error_deg']:.1f}deg",
            flush=True,
        )

    rows.sort(key=lambda row: (str(row["video"]), int(row["frame_index"])))
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    total = len(rows)
    totals = {
        key: sum(int(summary[key]) for summary in video_summaries.values())
        for key in ("pose_found", "heading_found", "relation_found", "accepted")
    }
    accepted_total = totals["accepted"]
    risk_total = sum(
        int(round(float(summary["edge_risk_positive_rate"]) * int(summary["accepted"])))
        for summary in video_summaries.values()
    )
    manifest = {
        "samples": str(args.samples.resolve()),
        "geometry_config": str(config_path),
        "heading_encoding": "cos_2error_sin_2error",
        "lateral_normalization": "offset_div_half_road_width",
        "gate": gate,
        "summary": {
            "frames": total,
            **totals,
            "pose_rate": totals["pose_found"] / total if total else 0.0,
            "heading_rate": totals["heading_found"] / total if total else 0.0,
            "relation_rate": totals["relation_found"] / total if total else 0.0,
            "accepted_rate": accepted_total / total if total else 0.0,
            "edge_risk_positive_rate": (
                risk_total / accepted_total if accepted_total else 0.0
            ),
        },
        "videos": video_summaries,
    }
    manifest_path = output.with_name("kart_relative_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = manifest["summary"]
    print(
        f"Overall: frames={total} pose={summary['pose_rate']:.3f} "
        f"heading={summary['heading_rate']:.3f} "
        f"relation={summary['relation_rate']:.3f} "
        f"accepted={summary['accepted_rate']:.3f} "
        f"risk+={summary['edge_risk_positive_rate']:.3f}",
        flush=True,
    )
    print(f"Labels: {output}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
