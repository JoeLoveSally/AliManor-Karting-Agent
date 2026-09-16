#!/usr/bin/env python3
"""Build temporally tracked kart-relative pseudo-labels for the v4-C2 follow-up."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_kart_relative_pseudo_labels import (  # noqa: E402
    heading_error_target,
    load_label_gate,
)
from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from inspect_kart_relative_geometry import load_kart_configs  # noqa: E402
from inspect_kart_relative_geometry_v2 import load_temporal_config  # noqa: E402
from karting_agent.train.geometry_pseudo_labels import (  # noqa: E402
    estimate_rectilinear_geometry,
    extract_road_mask,
)
from karting_agent.train.kart_pose_pseudo_labels import estimate_kart_pose  # noqa: E402
from karting_agent.train.kart_road_temporal import TemporalKartRoadTracker  # noqa: E402
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v4-C2 temporal-corridor kart-relative pseudo-labels."
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
        "--temporal-config",
        type=Path,
        default=ROOT / "configs" / "kart_relative_temporal_v2.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "v4c2_temporal_v2"
            / "kart_relative_labels.jsonl"
        ),
    )
    return parser.parse_args()


def _row_from_result(
    *,
    video: str,
    frame_index: int,
    pose,
    tracked,
    gate: dict[str, float],
) -> tuple[dict[str, object], bool, bool, bool]:
    relation = tracked.relation
    raw_lateral = None if relation is None else float(relation.lateral_offset_norm)
    heading_error = None if relation is None else relation.heading_error_deg
    confidence = 0.0 if relation is None else float(relation.confidence)
    weight = (
        confidence
        if relation is not None
        and heading_error is not None
        and confidence >= gate["min_confidence"]
        else 0.0
    )

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
    accepted = weight > 0.0
    risk_positive = accepted and edge_risk > 0.5
    offroad = accepted and relation is not None and not relation.inside_road

    return (
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
            "inside_road": None if relation is None else relation.inside_road,
            "confidence": confidence,
            "heading_quality": 0.0 if pose is None else pose.heading_quality,
            "road_support_fraction": (
                0.0 if relation is None else relation.road_support_fraction
            ),
            "road_angle_deg": None if relation is None else relation.road_angle_deg,
            "valid_cross_sections": (
                0 if relation is None else relation.valid_cross_sections
            ),
            "cross_section_count": (
                0 if relation is None else relation.cross_section_count
            ),
            "tracker_mode": tracked.mode,
            "tracker_stable_angle_deg": tracked.stable_angle_deg,
            "tracker_pending_angle_deg": tracked.pending_angle_deg,
            "tracker_pending_count": tracked.pending_count,
        },
        accepted,
        risk_positive,
        offroad,
    )


def main() -> int:
    args = parse_args()
    samples = load_v3_samples(args.samples.resolve())
    geometry_path = args.geometry_config.resolve()
    temporal_path = args.temporal_config.resolve()
    road_config, geometry_config, _ = load_road_config(geometry_path)
    kart_config, relation_config = load_kart_configs(geometry_path)
    temporal_config = load_temporal_config(temporal_path)
    gate = load_label_gate(geometry_path)

    requested: dict[str, set[int]] = defaultdict(set)
    for sample in samples:
        requested[sample.video].add(int(sample.input_frame_indices[-1]))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    video_summaries: dict[str, dict[str, object]] = {}

    for video, desired in sorted(requested.items()):
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = ROOT / video_path
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")

        tracker = TemporalKartRoadTracker(temporal_config)
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
        modes: Counter[str] = Counter()
        frame_index = 0
        try:
            while frame_index <= max_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index not in desired:
                    frame_index += 1
                    continue

                pose, _ = estimate_kart_pose(frame, kart_config)
                if pose is not None:
                    pose_found += 1
                if pose is not None and pose.heading_angle_deg is not None:
                    heading_found += 1

                if pose is None or pose.heading_angle_deg is None:
                    tracked = tracker.miss()
                else:
                    road_mask = extract_road_mask(frame, road_config)
                    geometry = estimate_rectilinear_geometry(
                        road_mask,
                        geometry_config,
                    )
                    tracked = tracker.update(
                        road_mask,
                        geometry,
                        pose,
                        relation_config,
                    )
                modes[tracked.mode] += 1
                relation = tracked.relation
                if relation is not None:
                    relation_found += 1

                row, row_accepted, risk_positive, row_offroad = _row_from_result(
                    video=video,
                    frame_index=frame_index,
                    pose=pose,
                    tracked=tracked,
                    gate=gate,
                )
                rows.append(row)
                found += 1
                if row_accepted:
                    accepted += 1
                    weight = float(row["weight"])
                    lateral = float(row["raw_lateral_offset_norm"])
                    heading_error = float(row["heading_error_deg"])
                    accepted_weight_sum += weight
                    accepted_abs_lateral_sum += abs(lateral)
                    accepted_abs_heading_sum += abs(heading_error)
                if risk_positive:
                    edge_risk_positive += 1
                if row_offroad:
                    offroad += 1
                frame_index += 1
        finally:
            capture.release()

        if found != len(desired):
            raise RuntimeError(
                f"video ended before all requested frames were labeled: {video} "
                f"found={found} expected={len(desired)}"
            )

        summary: dict[str, object] = {
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
            "tracker_modes": dict(sorted(modes.items())),
            "switches": int(modes.get("switch_confirmed", 0)),
            "pending_no_current": int(modes.get("pending_no_current", 0)),
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
            f"|herr|={summary['mean_abs_heading_error_deg']:.1f}deg "
            f"switches={summary['switches']} "
            f"pending_na={summary['pending_no_current']}",
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
        int(
            round(
                float(summary["edge_risk_positive_rate"])
                * int(summary["accepted"])
            )
        )
        for summary in video_summaries.values()
    )
    mode_totals: Counter[str] = Counter()
    for summary in video_summaries.values():
        mode_totals.update(summary["tracker_modes"])

    manifest = {
        "teacher_version": "temporal_corridor_v2",
        "samples": str(args.samples.resolve()),
        "geometry_config": str(geometry_path),
        "temporal_config_path": str(temporal_path),
        "temporal_config": temporal_config.__dict__,
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
            "tracker_modes": dict(sorted(mode_totals.items())),
            "switches": int(mode_totals.get("switch_confirmed", 0)),
            "pending_no_current": int(mode_totals.get("pending_no_current", 0)),
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
        f"risk+={summary['edge_risk_positive_rate']:.3f} "
        f"switches={summary['switches']} "
        f"pending_na={summary['pending_no_current']}",
        flush=True,
    )
    print(f"Labels: {output}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
