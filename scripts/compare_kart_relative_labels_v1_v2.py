#!/usr/bin/env python3
"""Compare v4-C2 single-frame and temporal kart-relative pseudo-labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare v4-C2 teacher-v1 and temporal teacher-v2 labels."
    )
    parser.add_argument(
        "--v1",
        type=Path,
        default=ROOT / "data" / "processed" / "v4c2" / "kart_relative_labels.jsonl",
    )
    parser.add_argument(
        "--v2",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "v4c2_temporal_v2"
            / "kart_relative_labels.jsonl"
        ),
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=ROOT / "configs" / "train_v4c2.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "kart_relative_geometry"
            / "teacher_v1_v2_comparison.json"
        ),
    )
    return parser.parse_args()


def _load(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    rows: dict[tuple[str, int], dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["video"]), int(row["frame_index"]))
            if key in rows:
                raise ValueError(f"duplicate key in {path}:{line_number}: {key}")
            rows[key] = row
    return rows


def _video_id(video: str) -> int | None:
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(video).name)
    return None if match is None else int(match.group(1))


def _split_map(path: Path) -> dict[int, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    split = raw.get("split", {})
    mapping: dict[int, str] = {}
    for name in ("train", "validation", "test"):
        for video_id in split.get(name, []):
            mapping[int(video_id)] = name
    return mapping


def _accepted(row: dict[str, object]) -> bool:
    return float(row.get("weight", 0.0)) > 0.0


def _relation(row: dict[str, object]) -> bool:
    return row.get("raw_lateral_offset_norm") is not None


def _axial_distance_deg(left: float, right: float) -> float:
    return abs(float((left - right + 90.0) % 180.0 - 90.0))


def _version_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    accepted = [row for row in rows if _accepted(row)]
    relations = [row for row in rows if _relation(row)]
    risk = [row for row in accepted if float(row.get("edge_risk_target", 0.0)) > 0.5]
    offroad = [row for row in accepted if row.get("inside_road") is False]

    adjacent_pairs = []
    for previous, current in zip(rows, rows[1:]):
        prev_frame = int(previous["frame_index"])
        curr_frame = int(current["frame_index"])
        if curr_frame - prev_frame > 4:
            continue
        if not (_accepted(previous) and _accepted(current)):
            continue
        adjacent_pairs.append((previous, current))

    lateral_jumps = []
    heading_jumps = []
    for previous, current in adjacent_pairs:
        prev_lat = previous.get("raw_lateral_offset_norm")
        curr_lat = current.get("raw_lateral_offset_norm")
        prev_heading = previous.get("heading_error_deg")
        curr_heading = current.get("heading_error_deg")
        if prev_lat is not None and curr_lat is not None:
            lateral_jumps.append(abs(float(curr_lat) - float(prev_lat)))
        if prev_heading is not None and curr_heading is not None:
            heading_jumps.append(
                _axial_distance_deg(float(curr_heading), float(prev_heading))
            )

    return {
        "frames": len(rows),
        "relation": len(relations),
        "relation_rate": len(relations) / len(rows) if rows else 0.0,
        "accepted": len(accepted),
        "accepted_rate": len(accepted) / len(rows) if rows else 0.0,
        "edge_risk_positive_rate": len(risk) / len(accepted) if accepted else 0.0,
        "offroad_rate": len(offroad) / len(accepted) if accepted else 0.0,
        "mean_weight": mean(float(row["weight"]) for row in accepted) if accepted else 0.0,
        "mean_abs_lateral": (
            mean(abs(float(row["raw_lateral_offset_norm"])) for row in accepted)
            if accepted
            else 0.0
        ),
        "mean_abs_heading_error_deg": (
            mean(abs(float(row["heading_error_deg"])) for row in accepted)
            if accepted
            else 0.0
        ),
        "adjacent_accepted_pairs": len(adjacent_pairs),
        "lateral_jump_gt_0p5_rate": (
            sum(value > 0.5 for value in lateral_jumps) / len(lateral_jumps)
            if lateral_jumps
            else 0.0
        ),
        "heading_jump_gt_25deg_rate": (
            sum(value > 25.0 for value in heading_jumps) / len(heading_jumps)
            if heading_jumps
            else 0.0
        ),
    }


def _pair_summary(
    v1_rows: list[dict[str, object]],
    v2_rows: list[dict[str, object]],
) -> dict[str, object]:
    v1 = {int(row["frame_index"]): row for row in v1_rows}
    v2 = {int(row["frame_index"]): row for row in v2_rows}
    if set(v1) != set(v2):
        missing_v1 = sorted(set(v2) - set(v1))[:5]
        missing_v2 = sorted(set(v1) - set(v2))[:5]
        raise ValueError(
            f"frame-key mismatch: missing_v1={missing_v1}, missing_v2={missing_v2}"
        )

    both = []
    gained = 0
    lost = 0
    risk_disagreements = 0
    lateral_deltas = []
    heading_deltas = []
    for frame in sorted(v1):
        left = v1[frame]
        right = v2[frame]
        left_ok = _accepted(left)
        right_ok = _accepted(right)
        if right_ok and not left_ok:
            gained += 1
        elif left_ok and not right_ok:
            lost += 1
        if not (left_ok and right_ok):
            continue
        both.append(frame)
        if bool(float(left.get("edge_risk_target", 0.0)) > 0.5) != bool(
            float(right.get("edge_risk_target", 0.0)) > 0.5
        ):
            risk_disagreements += 1
        left_lat = left.get("raw_lateral_offset_norm")
        right_lat = right.get("raw_lateral_offset_norm")
        if left_lat is not None and right_lat is not None:
            lateral_deltas.append(abs(float(right_lat) - float(left_lat)))
        left_heading = left.get("heading_error_deg")
        right_heading = right.get("heading_error_deg")
        if left_heading is not None and right_heading is not None:
            heading_deltas.append(
                _axial_distance_deg(float(right_heading), float(left_heading))
            )

    return {
        "accepted_both": len(both),
        "accepted_gained_v2": gained,
        "accepted_lost_v2": lost,
        "edge_risk_disagreement_rate_common": (
            risk_disagreements / len(both) if both else 0.0
        ),
        "mean_abs_lateral_delta_common": (
            mean(lateral_deltas) if lateral_deltas else 0.0
        ),
        "mean_axial_heading_delta_deg_common": (
            mean(heading_deltas) if heading_deltas else 0.0
        ),
    }


def _print_row(name: str, split: str, v1: dict[str, object], v2: dict[str, object], pair: dict[str, object]) -> None:
    print(
        f"{name:>6} {split:>10} "
        f"accept={v1['accepted_rate']:.3f}->{v2['accepted_rate']:.3f} "
        f"relation={v1['relation_rate']:.3f}->{v2['relation_rate']:.3f} "
        f"risk+={v1['edge_risk_positive_rate']:.3f}->{v2['edge_risk_positive_rate']:.3f} "
        f"jumpH={v1['heading_jump_gt_25deg_rate']:.3f}->{v2['heading_jump_gt_25deg_rate']:.3f} "
        f"jumpL={v1['lateral_jump_gt_0p5_rate']:.3f}->{v2['lateral_jump_gt_0p5_rate']:.3f} "
        f"gain/loss={pair['accepted_gained_v2']}/{pair['accepted_lost_v2']}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    v1 = _load(args.v1.resolve())
    v2 = _load(args.v2.resolve())
    if set(v1) != set(v2):
        raise ValueError(
            f"label-key mismatch: v1={len(v1)} v2={len(v2)} "
            f"v1_only={len(set(v1)-set(v2))} v2_only={len(set(v2)-set(v1))}"
        )

    split_by_id = _split_map(args.train_config.resolve())
    grouped_v1: dict[str, list[dict[str, object]]] = defaultdict(list)
    grouped_v2: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (video, _), row in v1.items():
        grouped_v1[video].append(row)
    for (video, _), row in v2.items():
        grouped_v2[video].append(row)
    for rows in grouped_v1.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    for rows in grouped_v2.values():
        rows.sort(key=lambda row: int(row["frame_index"]))

    videos: dict[str, object] = {}
    for video in sorted(grouped_v1):
        video_id = _video_id(video)
        split = split_by_id.get(video_id, "unknown") if video_id is not None else "unknown"
        left = _version_summary(grouped_v1[video])
        right = _version_summary(grouped_v2[video])
        pair = _pair_summary(grouped_v1[video], grouped_v2[video])
        videos[video] = {
            "video_id": video_id,
            "split": split,
            "v1": left,
            "v2": right,
            "comparison": pair,
        }
        _print_row(str(video_id), split, left, right, pair)

    all_v1 = [v1[key] for key in sorted(v1)]
    all_v2 = [v2[key] for key in sorted(v2)]
    overall_v1 = _version_summary(all_v1)
    overall_v2 = _version_summary(all_v2)
    overall_pair = _pair_summary(all_v1, all_v2)
    print("-" * 120)
    _print_row("ALL", "all", overall_v1, overall_v2, overall_pair)

    split_summaries: dict[str, object] = {}
    for split in ("train", "validation", "test"):
        split_videos = [
            video
            for video, payload in videos.items()
            if payload["split"] == split
        ]
        left_rows = [row for video in split_videos for row in grouped_v1[video]]
        right_rows = [row for video in split_videos for row in grouped_v2[video]]
        if not left_rows:
            continue
        split_summaries[split] = {
            "v1": _version_summary(left_rows),
            "v2": _version_summary(right_rows),
            "comparison": _pair_summary(left_rows, right_rows),
        }

    payload = {
        "v1": str(args.v1.resolve()),
        "v2": str(args.v2.resolve()),
        "overall": {
            "v1": overall_v1,
            "v2": overall_v2,
            "comparison": overall_pair,
        },
        "splits": split_summaries,
        "videos": videos,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Comparison: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
