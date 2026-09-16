#!/usr/bin/env python3
"""Audit temporally tracked kart-relative road geometry on videos."""

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
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from inspect_geometry_pseudo_labels import (  # noqa: E402
    load_config as load_road_config,
    write_contact_sheet,
)
from inspect_kart_relative_geometry import (  # noqa: E402
    _text,
    load_kart_configs,
    load_run_steps,
    nearest_run_step,
)
from karting_agent.train.geometry_pseudo_labels import (  # noqa: E402
    estimate_rectilinear_geometry,
    extract_road_mask,
    overlay_road_geometry,
)
from karting_agent.train.kart_pose_pseudo_labels import (  # noqa: E402
    estimate_kart_pose,
    overlay_kart_relative_geometry,
)
from karting_agent.train.kart_road_temporal import (  # noqa: E402
    TemporalKartRoadTracker,
    TemporalRoadTrackerConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the v4-C2 temporal corridor teacher on videos."
    )
    parser.add_argument("videos", type=Path, nargs="+")
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
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "kart_relative_geometry_v2",
    )
    parser.add_argument("--sample-every-frames", type=int, default=3)
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


def load_temporal_config(path: Path) -> TemporalRoadTrackerConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("temporal config root must be a mapping")
    values = raw.get("temporal_road_tracker", {})
    if not isinstance(values, dict):
        raise ValueError("temporal_road_tracker must be a mapping")
    defaults = TemporalRoadTrackerConfig()
    config = TemporalRoadTrackerConfig(
        continuity_weight=float(
            values.get("continuity_weight", defaults.continuity_weight)
        ),
        switch_angle_deg=float(values.get("switch_angle_deg", defaults.switch_angle_deg)),
        switch_confirm_samples=int(
            values.get("switch_confirm_samples", defaults.switch_confirm_samples)
        ),
        switch_score_margin=float(
            values.get("switch_score_margin", defaults.switch_score_margin)
        ),
        reset_gap_samples=int(
            values.get("reset_gap_samples", defaults.reset_gap_samples)
        ),
        stable_angle_alpha=float(
            values.get("stable_angle_alpha", defaults.stable_angle_alpha)
        ),
    )
    config.validate()
    return config


def _angle_text(value: float | None) -> str:
    return "NA" if value is None else f"{value:.0f}"


def inspect_video(
    video: Path,
    *,
    output_dir: Path,
    road_config,
    geometry_config,
    kart_config,
    relation_config,
    temporal_config: TemporalRoadTrackerConfig,
    sample_every_frames: int,
    max_previews: int,
    frame_start: int,
    frame_end: int | None,
    run_frames: list[int],
    run_steps: list[dict[str, object]],
) -> dict[str, object]:
    if sample_every_frames < 1:
        raise ValueError("sample_every_frames must be >= 1")
    if max_previews < 1:
        raise ValueError("max_previews must be >= 1")

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
    tracker = TemporalKartRoadTracker(temporal_config)
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
            tracked = tracker.update(road_mask, geometry, pose, relation_config)
            relation = tracked.relation

            overlay = overlay_road_geometry(
                frame,
                road_mask,
                geometry=geometry,
                geometry_config=geometry_config,
            )
            overlay = overlay_kart_relative_geometry(
                overlay,
                kart_mask,
                pose,
                relation,
            )

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
                relation_text = (
                    f"road={relation.road_angle_deg:.0f} herr={heading_error} "
                    f"lat={relation.lateral_offset_norm:+.2f}w "
                    f"relq={relation.confidence:.2f}"
                )
            tracker_text = (
                f"track={tracked.mode} stable={_angle_text(tracked.stable_angle_deg)} "
                f"pending={_angle_text(tracked.pending_angle_deg)}:{tracked.pending_count}"
            )
            _text(overlay, f"f={frame_index} {pose_text}", y=22)
            _text(overlay, relation_text, y=42)
            _text(overlay, tracker_text, y=62)

            run_step = nearest_run_step(frame_index, run_frames, run_steps)
            run_payload = None
            if run_step is not None:
                probability = float(run_step.get("probability", 0.0))
                action = str(run_step.get("action", "?"))
                pressed = bool(run_step.get("pressed", False))
                state = "PRESS" if pressed else "RELEASE"
                _text(
                    overlay,
                    f"policy p={probability:.3f} action={action} state={state}",
                    y=82,
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
                    "kart_road_relation": (
                        relation.to_dict() if relation is not None else None
                    ),
                    "tracker": {
                        "mode": tracked.mode,
                        "stable_angle_deg": tracked.stable_angle_deg,
                        "pending_angle_deg": tracked.pending_angle_deg,
                        "pending_count": tracked.pending_count,
                        "candidate_angles_deg": list(tracked.candidate_angles_deg),
                    },
                    "policy": run_payload,
                }
            )
    finally:
        capture.release()

    contact_sheet = stem_dir / "contact_sheet.jpg"
    write_contact_sheet(preview_images, contact_sheet, columns=4)
    relations = [
        row["kart_road_relation"]
        for row in rows
        if row["kart_road_relation"] is not None
    ]
    switches = sum(
        1 for row in rows if row["tracker"]["mode"] == "switch_confirmed"
    )
    pending_holds = sum(
        1 for row in rows if row["tracker"]["mode"] == "hold_pending"
    )
    summary = {
        "video": str(video),
        "fps": fps,
        "frame_count": frame_count,
        "frame_start": frame_start,
        "frame_end": last_frame,
        "sample_every_frames": sample_every_frames,
        "temporal_config": temporal_config.__dict__,
        "contact_sheet": str(contact_sheet),
        "tracker_summary": {
            "relations": len(relations),
            "relation_rate": len(relations) / len(rows) if rows else 0.0,
            "switches": switches,
            "pending_holds": pending_holds,
            "offroad": sum(1 for relation in relations if not relation["inside_road"]),
            "mean_abs_lateral": (
                float(np.mean([abs(float(r["lateral_offset_norm"])) for r in relations]))
                if relations
                else 0.0
            ),
            "mean_abs_heading_error_deg": (
                float(
                    np.mean(
                        [
                            abs(float(r["heading_error_deg"]))
                            for r in relations
                            if r["heading_error_deg"] is not None
                        ]
                    )
                )
                if any(r["heading_error_deg"] is not None for r in relations)
                else 0.0
            ),
        },
        "previews": rows,
    }
    (stem_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    if args.run_json is not None and len(args.videos) != 1:
        raise ValueError("--run-json is supported only with one video")

    geometry_path = args.geometry_config.resolve()
    road_config, geometry_config, _ = load_road_config(geometry_path)
    kart_config, relation_config = load_kart_configs(geometry_path)
    temporal_config = load_temporal_config(args.temporal_config.resolve())
    run_frames, run_steps = load_run_steps(
        args.run_json.resolve() if args.run_json is not None else None
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_arg in args.videos:
        video = video_arg.resolve()
        summary = inspect_video(
            video,
            output_dir=output_dir,
            road_config=road_config,
            geometry_config=geometry_config,
            kart_config=kart_config,
            relation_config=relation_config,
            temporal_config=temporal_config,
            sample_every_frames=args.sample_every_frames,
            max_previews=args.max_previews_per_video,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            run_frames=run_frames,
            run_steps=run_steps,
        )
        stats = summary["tracker_summary"]
        print(
            f"{video}: previews={len(summary['previews'])} "
            f"relation={stats['relation_rate']:.3f} "
            f"switches={stats['switches']} pending_holds={stats['pending_holds']} "
            f"offroad={stats['offroad']} |lat|={stats['mean_abs_lateral']:.3f} "
            f"|herr|={stats['mean_abs_heading_error_deg']:.1f}deg "
            f"sheet={summary['contact_sheet']}",
            flush=True,
        )
    print(f"Output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
