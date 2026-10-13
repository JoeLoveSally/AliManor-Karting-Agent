#!/usr/bin/env python3
"""Read-only original-video scene sheets for frozen Tiny CNN+GRU Test errors.

Displays raw source frames, NOT the model's masked/resized image. Does not
load models, train, change matching tolerances, or control the game.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

MODEL_NAMES = ("rgb", "rgb_hsv")
DELTA_MS = (-200.0, 0.0, 200.0)


def candidate_cases(events_by_mode: dict, *, max_cases: int = 8,
                    cluster_ms: float = 260.) -> list[dict]:
    """Prioritize GT short-release misses; deduplicate bursts of extra switches."""
    if max_cases < 1 or cluster_ms < 0:
        raise ValueError("invalid event selection parameters")
    if set(events_by_mode) != set(MODEL_NAMES):
        raise ValueError("must have both frozen visual models")
    proposals = {}
    for mode in MODEL_NAMES:
        record = events_by_mode[mode]
        for release in record["short_releases"]:
            if release["detected"]:
                continue
            start, end = float(release["start_ms"]), float(release["end_ms"])
            key = (start, end)
            item = proposals.setdefault(key, {
                "time_ms": (start+end)/2, "start_ms": start,
                "end_ms": end, "kind": "missed_short_release", "models": [],
            })
            item["models"].append(mode)
    cases = sorted(proposals.values(),
                   key=lambda item: (-len(item["models"]), item["time_ms"]))
    isolated, nearby = [], []
    for mode in MODEL_NAMES:
        record = events_by_mode[mode]
        for event in record["predicted"]:
            if event["status"] != "false_positive":
                continue
            t = float(event["time_ms"])
            same_direction = [
                abs(t - float(gt["time_ms"])) for gt in record["expected"]
                if bool(gt["pressed"]) == bool(event["pressed"])
            ]
            nearest = min(same_direction, default=float("inf"))
            case = {"time_ms": t, "kind": "extra_switch",
                    "models": [mode], "pressed": bool(event["pressed"])}
            (isolated if nearest > 150.0 else nearby).append(case)
    isolated.sort(key=lambda item: item["time_ms"])
    nearby.sort(key=lambda item: item["time_ms"])
    # Keep every unique missed GT short-release before considering FPs.
    for proposal in isolated + nearby:
        if any(abs(proposal["time_ms"] - case["time_ms"]) <= cluster_ms
               for case in cases):
            continue
        cases.append(proposal)
    return cases[:max_cases]


def get_frames_for_cases(video_path: Path, cases: list[dict],
                         *, delta_ms=DELTA_MS):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open original video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not 0 < fps < 1000 or count < 1:
        capture.release()
        raise ValueError(f"invalid video metadata: {video_path}")
    requested = {}
    for case_idx, case in enumerate(cases):
        for offset_idx, delta in enumerate(delta_ms):
            t = max(0., min(float(case["time_ms"]) + delta,
                            (count-1)*1000/fps))
            index = max(0, min(count-1, round(t * fps / 1000)))
            requested[(case_idx, offset_idx)] = (index, index*1000/fps)
    decoded = {}
    try:
        for index in sorted({entry[0] for entry in requested.values()}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise IOError(f"cannot read {video_path} at frame {index}")
            decoded[index] = frame
    finally:
        capture.release()
    return fps, count, requested, decoded


def make_contact_sheet(cases, requested, decoded, *, view_width=225):
    if not cases or view_width < 100:
        raise ValueError("empty cases or invalid view width")
    source = next(iter(decoded.values()))
    height, width = source.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("invalid source frame")
    view_height = round(view_width*height/width)
    row_height = view_height+82
    sheet_width = 330+3*(view_width+10)+10
    sheet = np.full((row_height*len(cases),sheet_width,3),247,np.uint8)
    for i, case in enumerate(cases):
        y=i*row_height
        cv2.putText(sheet, f"{i+1}. {case['kind']}"[:34],
                    (8,y+24),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,30,30),1,cv2.LINE_AA)
        cv2.putText(sheet, f"t={case['time_ms']/1000:.3f}s",
                    (8,y+53),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1,cv2.LINE_AA)
        cv2.putText(sheet, ", ".join(case["models"]),
                    (8,y+80),cv2.FONT_HERSHEY_SIMPLEX,.44,(80,80,80),1,cv2.LINE_AA)
        if case["kind"] == "missed_short_release":
            cv2.putText(sheet,
                        f"GT {case['start_ms']/1000:.3f}-{case['end_ms']/1000:.3f}s",
                        (8,y+110),cv2.FONT_HERSHEY_SIMPLEX,.39,(40,40,130),1)
        for j, delta in enumerate(DELTA_MS):
            index,timestamp=requested[(i,j)]
            frame=cv2.resize(decoded[index],(view_width,view_height),
                             interpolation=cv2.INTER_AREA)
            x=330+j*(view_width+10)
            sheet[y+35:y+35+view_height,x:x+view_width]=frame
            cv2.putText(sheet,f"{delta:+.0f}ms | f={index}",
                        (x,y+24),cv2.FONT_HERSHEY_SIMPLEX,.42,(40,40,40),1,cv2.LINE_AA)
            cv2.putText(sheet,f"source {timestamp/1000:.3f}s",
                        (x,y+view_height+58),cv2.FONT_HERSHEY_SIMPLEX,
                        .4,(60,60,60),1,cv2.LINE_AA)
    return sheet


def validate_trace(trace_path: Path, expected_samples: int) -> None:
    with trace_path.open("r",encoding="utf-8",newline="") as fp:
        reader=csv.DictReader(fp)
        required={"target_ms","gt_future_pressed",
                  "rgb_probability","rgb_hsv_probability"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("trace CSV missing required columns")
        times=[float(row["target_ms"]) for row in reader]
    if len(times)!=expected_samples or any(
        later<=earlier for earlier,later in zip(times,times[1:])
    ):
        raise ValueError("trace CSV disagrees with frozen audit sample count or time sequence")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit",type=Path,required=True)
    ap.add_argument("--trace-dir",type=Path,required=True)
    ap.add_argument("--video-root",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--max-cases",type=int,default=8)
    args=ap.parse_args()
    if args.max_cases<1 or not args.output_dir.is_dir():
        ap.error("max-cases must be positive and output-dir must exist")
    audit=json.loads(args.audit.read_text(encoding="utf-8"))
    if (audit.get("kind")!="frozen_tiny_policy_timeline_audit" or
        audit.get("threshold")!=.5 or audit.get("tolerance_ms")!=100. or
        not audit.get("diagnostic_only_no_tuning") or
        set(audit["test_videos"])!=set(audit["videos"])):
        raise ValueError("not the frozen Test diagnostic audit")
    todo=[]
    for video in audit["test_videos"]:
        relative=Path(video)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe video path in audit: {video}")
        stem=relative.stem
        output=args.output_dir/f"tiny_scenes_{stem}.jpg"
        if output.exists():
            ap.error(f"refusing to overwrite {output}")
        trace=args.trace_dir/f"tiny_trace_{stem}.csv"
        validate_trace(trace,int(audit["videos"][video]["samples"]))
        cases=candidate_cases(
            audit["videos"][video]["events_by_mode"],max_cases=args.max_cases)
        if cases:
            todo.append((args.video_root/relative,output,cases))
    for video_path,output,cases in todo:
        fps,_,requested,decoded=get_frames_for_cases(video_path,cases)
        sheet=make_contact_sheet(cases,requested,decoded)
        if not cv2.imwrite(str(output),sheet,[cv2.IMWRITE_JPEG_QUALITY,90]):
            raise IOError(f"cannot write {output}")
        print(f"{output.name}: {len(cases)} unique windows, {fps:.3f} fps; "
              "raw source frames; no inference",flush=True)


if __name__=="__main__":
    main()
