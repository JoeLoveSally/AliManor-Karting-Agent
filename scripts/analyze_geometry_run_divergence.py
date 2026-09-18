#!/usr/bin/env python3
"""Compare temporal kart-road teacher trajectories around aligned live anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_OFFSETS_MS = (
    0.0,
    50.0,
    100.0,
    150.0,
    200.0,
    250.0,
    300.0,
    350.0,
    400.0,
    500.0,
    600.0,
    700.0,
    800.0,
    900.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align two kart-road teacher summaries by source-frame anchors and "
            "compare geometry over a common relative-time grid."
        )
    )
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--reference-anchor-frame", type=int, required=True)
    parser.add_argument("--candidate-anchor-frame", type=int, required=True)
    parser.add_argument(
        "--offsets-ms",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_OFFSETS_MS),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary root must be a mapping: {path}")
    previews = payload.get("previews")
    if not isinstance(previews, list) or not previews:
        raise ValueError(f"summary has no previews: {path}")
    fps = float(payload.get("fps", 0.0))
    if fps <= 0:
        raise ValueError(f"summary has invalid fps: {path}")
    return payload


def _parse_offsets(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("--offsets-ms must contain at least one value")
    if tuple(sorted(values)) != values:
        raise ValueError("--offsets-ms must be sorted")
    if len(set(values)) != len(values):
        raise ValueError("--offsets-ms must be unique")
    return values


def _nearest_preview(
    previews: list[dict[str, Any]],
    *,
    target_frame: float,
) -> dict[str, Any]:
    return min(
        previews,
        key=lambda row: abs(float(row["frame_index"]) - target_frame),
    )


def _relation_values(row: dict[str, Any]) -> dict[str, Any]:
    relation = row.get("kart_road_relation")
    tracker = row.get("tracker")
    pose = row.get("kart_pose")
    relation_map = relation if isinstance(relation, dict) else {}
    tracker_map = tracker if isinstance(tracker, dict) else {}
    pose_map = pose if isinstance(pose, dict) else {}
    return {
        "frame_index": int(row["frame_index"]),
        "timestamp_ms": row.get("timestamp_ms"),
        "inside_road": relation_map.get("inside_road"),
        "lateral_offset_norm": relation_map.get("lateral_offset_norm"),
        "heading_error_deg": relation_map.get("heading_error_deg"),
        "road_angle_deg": relation_map.get("road_angle_deg"),
        "confidence": relation_map.get("confidence"),
        "kart_center_x": pose_map.get("center_x"),
        "kart_center_y": pose_map.get("center_y"),
        "kart_heading_deg": pose_map.get("heading_angle_deg"),
        "kart_heading_quality": pose_map.get("heading_quality"),
        "tracker_mode": tracker_map.get("mode"),
        "stable_angle_deg": tracker_map.get("stable_angle_deg"),
        "pending_angle_deg": tracker_map.get("pending_angle_deg"),
        "pending_count": tracker_map.get("pending_count"),
    }


def _fmt(value: Any, *, width: int = 7, decimals: int = 2) -> str:
    if value is None:
        return "NA".rjust(width)
    if isinstance(value, bool):
        return ("in" if value else "OUT").rjust(width)
    return f"{float(value):{width}.{decimals}f}"


def main() -> int:
    args = parse_args()
    offsets_ms = _parse_offsets(args.offsets_ms)
    reference = _load_summary(args.reference_summary.resolve())
    candidate = _load_summary(args.candidate_summary.resolve())

    reference_fps = float(reference["fps"])
    candidate_fps = float(candidate["fps"])
    reference_previews = reference["previews"]
    candidate_previews = candidate["previews"]
    assert isinstance(reference_previews, list)
    assert isinstance(candidate_previews, list)

    rows: list[dict[str, Any]] = []
    for offset_ms in offsets_ms:
        reference_target = (
            args.reference_anchor_frame + offset_ms * reference_fps / 1000.0
        )
        candidate_target = (
            args.candidate_anchor_frame + offset_ms * candidate_fps / 1000.0
        )
        ref_row = _nearest_preview(
            reference_previews,
            target_frame=reference_target,
        )
        cand_row = _nearest_preview(
            candidate_previews,
            target_frame=candidate_target,
        )
        ref_values = _relation_values(ref_row)
        cand_values = _relation_values(cand_row)

        def delta(key: str) -> float | None:
            left = ref_values.get(key)
            right = cand_values.get(key)
            if left is None or right is None:
                return None
            return float(right) - float(left)

        rows.append(
            {
                "offset_ms": offset_ms,
                "reference": ref_values,
                "candidate": cand_values,
                "delta_lateral_offset_norm": delta("lateral_offset_norm"),
                "delta_heading_error_deg": delta("heading_error_deg"),
                "delta_road_angle_deg": delta("road_angle_deg"),
                "delta_kart_center_x": delta("kart_center_x"),
                "delta_kart_center_y": delta("kart_center_y"),
                "delta_kart_heading_deg": delta("kart_heading_deg"),
                "delta_confidence": delta("confidence"),
            }
        )

    print(
        f"reference={Path(args.reference_summary).name} "
        f"candidate={Path(args.candidate_summary).name}",
        flush=True,
    )
    print(
        f"anchors: ref=f{args.reference_anchor_frame} "
        f"candidate=f{args.candidate_anchor_frame}",
        flush=True,
    )
    print(
        " offset | "
        " ref_frame ref_x ref_y ref_kart ref_road ref_herr ref_mode "
        "ref_stable ref_pending:n | "
        " cur_frame cur_x cur_y cur_kart cur_road cur_herr cur_mode "
        "cur_stable cur_pending:n | "
        " d_x d_y d_lat d_herr",
        flush=True,
    )
    for row in rows:
        ref = row["reference"]
        cand = row["candidate"]
        assert isinstance(ref, dict)
        assert isinstance(cand, dict)
        print(
            f"{float(row['offset_ms']):+7.0f} | "
            f"{int(ref['frame_index']):9d} "
            f"{_fmt(ref['kart_center_x'], width=5, decimals=0)} "
            f"{_fmt(ref['kart_center_y'], width=5, decimals=0)} "
            f"{_fmt(ref['kart_heading_deg'], width=8, decimals=1)} "
            f"{_fmt(ref['road_angle_deg'], width=8, decimals=1)} "
            f"{_fmt(ref['heading_error_deg'], width=8, decimals=1)} "
            f"{str(ref['tracker_mode']):>9} "
            f"{_fmt(ref['stable_angle_deg'], width=8, decimals=1)} "
            f"{_fmt(ref['pending_angle_deg'], width=8, decimals=1)}:"
            f"{str(ref['pending_count']):>1} | "
            f"{int(cand['frame_index']):9d} "
            f"{_fmt(cand['kart_center_x'], width=5, decimals=0)} "
            f"{_fmt(cand['kart_center_y'], width=5, decimals=0)} "
            f"{_fmt(cand['kart_heading_deg'], width=8, decimals=1)} "
            f"{_fmt(cand['road_angle_deg'], width=8, decimals=1)} "
            f"{_fmt(cand['heading_error_deg'], width=8, decimals=1)} "
            f"{str(cand['tracker_mode']):>9} "
            f"{_fmt(cand['stable_angle_deg'], width=8, decimals=1)} "
            f"{_fmt(cand['pending_angle_deg'], width=8, decimals=1)}:"
            f"{str(cand['pending_count']):>1} | "
            f"{_fmt(row['delta_kart_center_x'], width=5, decimals=0)} "
            f"{_fmt(row['delta_kart_center_y'], width=5, decimals=0)} "
            f"{_fmt(row['delta_lateral_offset_norm'], width=6)} "
            f"{_fmt(row['delta_heading_error_deg'], width=7)}",
            flush=True,
        )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else Path(args.candidate_summary).resolve().with_name(
            "aligned_geometry_divergence.json"
        )
    )
    output_path.write_text(
        json.dumps(
            {
                "reference_summary": str(args.reference_summary.resolve()),
                "candidate_summary": str(args.candidate_summary.resolve()),
                "reference_anchor_frame": args.reference_anchor_frame,
                "candidate_anchor_frame": args.candidate_anchor_frame,
                "offsets_ms": list(offsets_ms),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
