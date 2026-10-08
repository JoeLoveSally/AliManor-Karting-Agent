#!/usr/bin/env python3
"""Read-only kart/road visual-state observability audit.

Extract CV teacher states from a sequential video, not switch labels or actions.
Output a JSONL measurement timeline, preview overlays, a blank human-review CSV,
and availability/temporal-continuity statistics. Never execute game controls.

An axial pose has an inherent 180-degree ambiguity. Signed lateral offset is
only relative to the chosen road-axis normal; it is not guaranteed to be a
consistent left/right steering coordinate across corners. Visible axis crossings
are NOT automatically the next corner.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def axial_difference_deg(angle: float, reference: float) -> float:
    """Smallest signed difference between 180-degree periodic orientations."""
    return float((angle - reference + 90.0) % 180.0 - 90.0)


def quantize_measurement(
    *,
    frame_index: int,
    timestamp_ms: float,
    timestamp_source: str,
    pose: dict | None,
    relation: dict | None,
    geometry: dict,
    tracker_mode: str,
    previous: dict | None = None,
    min_confidence: float = 0.05,
    max_pair_gap_ms: float = 250.0,
    max_axis_jump_deg: float = 15.0,
) -> dict:
    """Turn auditable CV teacher outputs into explicitly nullable observations."""
    if not math.isfinite(timestamp_ms):
        raise ValueError("invalid timestamp")
    if not 0 <= min_confidence <= 1 or max_pair_gap_ms <= 0 or max_axis_jump_deg <= 0:
        raise ValueError("invalid observability thresholds")

    confidence = float(relation.get("confidence", 0.0)) if relation else 0.0
    usable = bool(
        pose is not None
        and relation is not None
        and relation.get("heading_error_deg") is not None
        and math.isfinite(confidence)
        and confidence >= min_confidence
        and tracker_mode not in ("missing", "pending_no_current")
    )
    lateral = float(relation["lateral_offset_norm"]) if usable else None
    heading = float(relation["heading_error_deg"]) if usable else None
    axis = float(relation["road_angle_deg"]) if usable else None
    valid = usable and all(math.isfinite(v) for v in (lateral, heading, axis))
    if not valid:
        lateral = heading = axis = None

    # These are screen-space changes, not calibrated world-space kinematics.
    lateral_rate = None
    heading_rate = None
    pair_dt_ms = None
    if valid and previous and previous.get("state_valid"):
        dt = timestamp_ms - float(previous["timestamp_ms"])
        prior_axis = previous.get("road_axis_axial_deg")
        if (0 < dt <= max_pair_gap_ms
                and prior_axis is not None
                and abs(axial_difference_deg(axis, float(prior_axis)))
                    <= max_axis_jump_deg):
            lateral_rate = (lateral - float(previous["lateral_offset_norm"])) * 1000.0 / dt
            heading_rate = axial_difference_deg(
                heading, float(previous["heading_error_axial_deg"])
            ) * 1000.0 / dt
            pair_dt_ms = dt

    corner_visible = bool(geometry.get("corner_visible", False))
    corner_distance = (geometry.get("corner_distance_norm")
                       if corner_visible else None)
    return {
        "frame_index": int(frame_index),
        "timestamp_ms": float(timestamp_ms),
        "timestamp_source": timestamp_source,
        "state_valid": bool(valid),
        "kart_detected": pose is not None,
        "heading_detected": pose is not None and pose.get("heading_angle_deg") is not None,
        "kart_center_xy_norm": (
            [float(pose["center_x_norm"]), float(pose["center_y_norm"])]
            if pose is not None else None
        ),
        "kart_heading_axial_deg": (
            float(pose["heading_angle_deg"])
            if pose is not None and pose.get("heading_angle_deg") is not None else None
        ),
        "pose_heading_quality": float(pose["heading_quality"]) if pose else None,
        "road_axis_axial_deg": axis,
        "lateral_offset_norm": lateral,
        "heading_error_axial_deg": heading,
        "lateral_rate_per_s": lateral_rate,
        "heading_rate_deg_per_s": heading_rate,
        "temporal_pair_dt_ms": pair_dt_ms,
        "relation_confidence": confidence if relation else None,
        "road_support_fraction": (
            float(relation["road_support_fraction"]) if relation else None
        ),
        "inside_road_estimate": (
            bool(relation["inside_road"]) if valid else None
        ),
        "tracker_mode": tracker_mode,
        "geometry_class": geometry.get("geometry_class"),
        "corner_score": geometry.get("corner_score"),
        "visible_axis_intersection_distance_norm": corner_distance,
        "next_corner_direction": None,
        "next_corner_distance": None,
        "next_corner_valid": False,
        "raw_relation": relation,
        "raw_pose": pose,
    }


def summarize_rows(rows: list[dict]) -> dict:
    total = len(rows)
    valid = [r for r in rows if r["state_valid"]]
    rates = [r for r in rows if r["lateral_rate_per_s"] is not None]
    def frac(n):
        return (n / total) if total else 0.0
    def rms(values):
        return math.sqrt(sum(x*x for x in values) / len(values)) if values else None

    return {
        "sample_count": total,
        "kart_detected_rate": frac(sum(bool(r["kart_detected"]) for r in rows)),
        "heading_detected_rate": frac(sum(bool(r["heading_detected"]) for r in rows)),
        "valid_relation_rate": frac(len(valid)),
        "temporal_rate_coverage": frac(len(rates)),
        "axis_intersection_visible_rate": frac(sum(
            r["visible_axis_intersection_distance_norm"] is not None for r in rows
        )),
        "candidate_next_corner_coverage": 0.0,
        "confidence_mean_on_valid": (
            sum(r["relation_confidence"] for r in valid) / len(valid) if valid else None
        ),
        "rms_observed_lateral_rate_per_s": rms([r["lateral_rate_per_s"] for r in rates]),
        "rms_observed_heading_rate_deg_per_s": rms([r["heading_rate_deg_per_s"] for r in rates]),
        "note": (
            "Coverage is NOT accuracy. RMS rates measure variability, not error "
            "against independent ground truth. A human review is required."
        ),
    }


def load_run_timestamps(run_json: Path | None, frame_count_hint: int):
    if run_json is None:
        return None
    run = json.loads(run_json.read_text(encoding="utf-8"))
    recording = run.get("recording", {})
    if (recording.get("frame_mapping_valid") is not True
            or recording.get("dropped_frames") != 0
            or recording.get("timing") != "synthetic_cfr_stream_copy"):
        raise ValueError("ADB recording lacks validated source-frame mapping")
    if int(recording["frame_count"]) != len(run["capture"]["decoded_frame_timestamps_ms"]):
        raise ValueError("run JSON decoded timestamps do not match recorded frame count")
    # Fragmented MP4s can report an unreliable CAP_PROP_FRAME_COUNT. Do not
    # reject on the container hint; validate sampled indices and, when decoded
    # to EOF, compare actual decoded frame count against the JSON timeline.
    timestamps = list(map(float, run["capture"]["decoded_frame_timestamps_ms"]))
    if any(not math.isfinite(t) for t in timestamps):
        raise ValueError("non-finite logged timestamps")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("non-monotonic run JSON source-frame timestamps")
    return timestamps


def inspect_video(args, video: Path, *, output_dir: Path):
    import cv2

    from inspect_geometry_pseudo_labels import (
        load_config as load_road_config, write_contact_sheet,
    )
    from inspect_kart_relative_geometry import load_kart_configs
    from inspect_kart_relative_geometry_v2 import load_temporal_config
    from karting_agent.train.geometry_pseudo_labels import (
        estimate_rectilinear_geometry, extract_road_mask,
        overlay_road_geometry,
    )
    from karting_agent.train.kart_pose_pseudo_labels import (
        estimate_kart_pose, overlay_kart_relative_geometry,
    )
    from karting_agent.train.kart_road_temporal import TemporalKartRoadTracker

    road_cfg, geom_cfg, _ = load_road_config(args.geometry_config)
    pose_cfg, relation_cfg = load_kart_configs(args.geometry_config)
    tracker = TemporalKartRoadTracker(load_temporal_config(args.temporal_config))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    count_hint = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    run_times = load_run_timestamps(args.run_json, count_hint)
    output_dir.mkdir(parents=True, exist_ok=False)

    all_rows = []
    review = []
    overlays = []
    preview_dir = output_dir / "previews"
    preview_dir.mkdir()
    last_row = None
    frame_index = 0
    decoded_to_eof = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                decoded_to_eof = True
                break
            if args.frame_end is not None and frame_index > args.frame_end:
                break
            take = (frame_index >= args.frame_start and
                    (frame_index - args.frame_start) % args.sample_every_frames == 0)
            if not take:
                frame_index += 1
                continue
            if len(all_rows) >= args.max_samples:
                break
            if run_times is not None:
                if frame_index >= len(run_times):
                    raise ValueError("decoded more frames than run timestamps")
                t_ms = run_times[frame_index]
                time_source = "adb_json_host_decoded_timestamp"
            else:
                # CAP_PROP_POS_MSEC often tracks container PTS, but may be
                # synthetic. This is explicitly not a measured host timestamp.
                pts = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                last_time = last_row["timestamp_ms"] if last_row else float("-inf")
                if math.isfinite(pts) and pts > max(0.0, last_time):
                    t_ms, time_source = pts, "opencv_container_pts_unverified"
                elif math.isfinite(fps) and fps > 0:
                    t_ms = frame_index * 1000.0 / fps
                    time_source = "frame_index_fps_approx"
                else:
                    raise ValueError("neither timestamp nor valid FPS available")
            road_mask = extract_road_mask(frame, road_cfg)
            geometry = estimate_rectilinear_geometry(road_mask, geom_cfg)
            pose, mask = estimate_kart_pose(frame, pose_cfg)
            tracked = tracker.update(road_mask, geometry, pose, relation_cfg)
            relation = tracked.relation
            row = quantize_measurement(
                frame_index=frame_index, timestamp_ms=t_ms,
                timestamp_source=time_source,
                pose=pose.to_dict() if pose else None,
                relation=relation.to_dict() if relation else None,
                geometry=geometry.to_dict(),
                tracker_mode=tracked.mode, previous=last_row,
                min_confidence=args.min_confidence,
                max_pair_gap_ms=args.max_pair_gap_ms,
                max_axis_jump_deg=args.max_axis_jump_deg,
            )
            all_rows.append(row)
            last_row = row
            if (len(all_rows) - 1) % args.preview_every_samples == 0:
                overlay = overlay_road_geometry(
                    frame, road_mask, geometry=geometry, geometry_config=geom_cfg,
                )
                overlay = overlay_kart_relative_geometry(
                    overlay, mask, pose, relation,
                )
                label = (
                    f"f={frame_index} valid={int(row['state_valid'])}"
                    f" lat={row['lateral_offset_norm']} "
                    f"heading={row['heading_error_axial_deg']}"
                )
                cv2.putText(overlay, label[:110], (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                            1, cv2.LINE_AA)
                preview = preview_dir / f"frame_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(preview), overlay):
                    raise RuntimeError(f"could not write {preview}")
                overlays.append(overlay)
                review.append({
                    "frame_index": frame_index,
                    "timestamp_ms": round(t_ms, 3),
                    "preview": str(preview),
                    "kart_pose_correct": "",
                    "road_corridor_correct": "",
                    "lateral_sign_consistent": "",
                    "heading_axis_correct": "",
                    "visible_corner_correct": "",
                    "is_next_corner": "",
                    "recoverable_deviation": "",
                    "notes": "",
                })
            frame_index += 1
    finally:
        capture.release()

    if not all_rows:
        raise ValueError("no sampled frames; check --frame-start/--frame-end")
    if decoded_to_eof and run_times is not None and frame_index != len(run_times):
        raise ValueError("decoded source frame count differs from run JSON")
    jsonl = output_dir / "measurements.jsonl"
    with jsonl.open("x", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    review_path = output_dir / "manual_review.csv"
    with review_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    sheet = output_dir / "contact_sheet.jpg"
    write_contact_sheet(overlays, sheet, columns=4)
    summary = {
        "video": str(video),
        "run_json": str(args.run_json) if args.run_json else None,
        "sampled_frame_range": [all_rows[0]["frame_index"], all_rows[-1]["frame_index"]],
        "frame_step": args.sample_every_frames,
        "sampling_is_sequential": True,
        "source_timestamp_policy": (
            "ADB host-decoder timing, source frame index"
            if run_times is not None else
            "container PTS / nominal FPS; time derivatives approximate"
        ),
        "manual_review_required": True,
        "metrics": summarize_rows(all_rows),
        "outputs": {
            "measurements": str(jsonl),
            "manual_review": str(review_path),
            "contact_sheet": str(sheet),
        },
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--run-json", type=Path, default=None,
                    help="Matching run JSON; only one video supported")
    ap.add_argument("--geometry-config", type=Path,
                    default=ROOT / "configs/geometry_pseudo_labels.yaml")
    ap.add_argument("--temporal-config", type=Path,
                    default=ROOT / "configs/kart_relative_temporal_v2.yaml")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--sample-every-frames", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=180)
    ap.add_argument("--preview-every-samples", type=int, default=5)
    ap.add_argument("--min-confidence", type=float, default=0.05)
    ap.add_argument("--max-pair-gap-ms", type=float, default=250.0)
    ap.add_argument("--max-axis-jump-deg", type=float, default=15.0)
    args = ap.parse_args()
    if (args.frame_start < 0 or args.sample_every_frames < 1
            or args.max_samples < 1 or args.preview_every_samples < 1):
        ap.error("frame start, sampling stride and sample limits must be valid")
    if args.frame_end is not None and args.frame_end < args.frame_start:
        ap.error("frame end must be >= start")
    if args.run_json is not None and len(args.videos) != 1:
        ap.error("run JSON is valid only with one video")
    # Separate output directory per video ensures no collisions on duplicates.
    for video in args.videos:
        summary = inspect_video(
            args, video, output_dir=args.output_dir / video.stem,
        )
        print(json.dumps({"video": video.name,
                          "metrics": summary["metrics"],
                          "contact_sheet": summary["outputs"]["contact_sheet"],
                          "manual_review": summary["outputs"]["manual_review"]},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
