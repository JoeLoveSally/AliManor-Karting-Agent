#!/usr/bin/env python3
"""Passive manual-tap timing trace with decoded-frame and raw kernel timestamps.

Records a single screenrecord H264 stream as fragmented MP4, complete host
BGR-decode timestamps, host consumer-read timestamps, and raw Android getevent.
No model weights, inference, training, ADB input injection or Armed controls.
The MP4 uses synthetic CFR PTS; NOT measured Android capture timestamps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def output_paths(directory: Path, run_name: str) -> dict[str, Path]:
    if not directory.is_dir() or not run_name.startswith("touch_calibration_"):
        raise ValueError("expected an existing output directory and touch_calibration run name")
    paths = {
        "video": directory / f"{run_name}.mp4",
        "frames": directory / f"{run_name}_frames.jsonl",
        "touch": directory / f"{run_name}_touch.jsonl",
        "reads": directory / f"{run_name}_reads.jsonl",
        "summary": directory / f"{run_name}_summary.json",
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
    return paths


def consumption_row(frame, host_read_perf_counter_ms: float,
                    origin_perf_counter_ms: float) -> dict:
    decoded_perf_ms = origin_perf_counter_ms + float(frame.timestamp_ms)
    delay = host_read_perf_counter_ms - decoded_perf_ms
    if delay < -0.01:
        raise ValueError("consumer read predates decoded frame: clock mismatch")
    return {
        "frame_index": int(frame.frame_index),
        "host_read_perf_counter_ms": host_read_perf_counter_ms,
        "host_decoded_perf_counter_ms": decoded_perf_ms,
        "decode_to_read_ms": max(0.0, delay),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=45.0,
                        help="Seconds of passive recording (default 45)")
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--touch-device", type=str, default=None,
                        help="Select /dev/input/eventN if touchscreen autodetection is ambiguous")
    parser.add_argument("--decode-width", type=int, default=360,
                        help="Default 360px; optical heuristic expects 360x800 on a 1440x3200 display")
    args = parser.parse_args()
    if not 0 < args.duration <= 180 or args.decode_width < 200:
        parser.error("duration must be 0..180 s and decode width >=200")
    run_name = ("touch_calibration_" +
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    paths = output_paths(args.output_dir, run_name)

    for folder in (ROOT / "src",):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    from karting_agent.data_flow.adb import AdbClient, AdbConfig
    from karting_agent.data_flow.input.adb_video import AdbVideoInput, AdbVideoConfig
    from karting_agent.train.labels.live_touch_events import (
        LiveTouchRecorder, choose_touch_device, decoded_frame_rows,
    )

    adb = AdbClient(AdbConfig(serial=args.serial))
    adb.require_device()
    selected = args.touch_device
    if selected is None:
        listing = adb.run("shell", "getevent", "-lp").decode("utf-8", errors="replace")
        selected = choose_touch_device(listing)
    video = AdbVideoInput(
        adb, screen_size=adb.screen_size(),
        config=AdbVideoConfig(decode_width=args.decode_width),
        record_path=paths["video"],
    )
    touch = LiveTouchRecorder(adb, paths["touch"], device=selected)
    count = 0
    error = None
    capture_start_perf_ms = time.perf_counter() * 1000.0
    print("Read-only capture. Play manually; no automated phone input.", flush=True)
    try:
        touch.start()
        started = time.monotonic()
        with paths["reads"].open("x", encoding="utf-8") as out:
            while time.monotonic() - started < args.duration:
                frame = video.read()
                now_ms = time.perf_counter() * 1000.0
                out.write(json.dumps(consumption_row(
                    frame, now_ms, video.origin_perf_counter_ms,
                )) + "\n")
                count += 1
                if count == 1:
                    print("Video and touch capture active. Perform manual PRESS/RELEASE pulses.", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, KeyboardInterrupt):
            print("Stopped manually; finalizing trace.", flush=True)
        else:
            print(f"Capture warning: {error}", flush=True)
    finally:
        touch.close()
        video.close()
        if video.started:
            with paths["frames"].open("x", encoding="utf-8") as stream:
                for row in decoded_frame_rows(
                    video.decoded_frame_timestamps_ms, video.origin_perf_counter_ms,
                ):
                    stream.write(json.dumps(row) + "\n")
        summary = {
            "kind": "passive_touch_timing_calibration_trace",
            "read_only": True,
            "capture_error": error,
            "capture_start_perf_counter_ms": capture_start_perf_ms,
            "host_decoder_origin_perf_counter_ms": (
                video.origin_perf_counter_ms if video.started else None
            ),
            "duration_requested_seconds": args.duration,
            "decoded_frames": video.decoded_frames,
            "consumed_frames": count,
            "dropped_latest_queue_frames": video.dropped_frames,
            "decode_resolution": [video.decode_width, video.decode_height],
            "touch": touch.summary(),
            "recording_error": video.recording_error,
            "recording_frame_mapping_valid": video.recording_frame_mapping_valid,
            "recorded_frames": video.recorded_frames,
            "mp4_time_base": "synthetic_60fps_NOT_device_capture_time",
            "decoded_frame_time_base": "host perf_counter at BGR output availability",
            "getevent_time_base": "Android kernel timestamp + host stdout arrival; offset unknown",
            "files": {key: str(value) for key, value in paths.items()},
            "limitations": [
                "No Android capture/encode timestamp is exposed by this recording path.",
                "Host getevent receipt is not the physical touch event time.",
                "Visibility of the touch indicator is delayed UI feedback.",
                "A debug MP4 exit error makes recording validity uncertain; verify frame index parity.",
                "No automatic or externally commanded tap is sent to Android.",
            ],
        }
        with paths["summary"].open("x", encoding="utf-8") as out:
            json.dump(summary, out, ensure_ascii=False, indent=2)
            out.write("\n")
        print(f"Summary: {paths['summary']}", flush=True)
        print(f"Raw traces: {paths['frames']}, {paths['touch']}, {paths['reads']}", flush=True)
        print(f"Video: {paths['video']}", flush=True)
        print(f"Touch events: {touch.transitions}; decoded frames: {video.decoded_frames}", flush=True)
        if video.recording_error:
            print("Recording warning: " + video.recording_error, flush=True)
    if error and not error.startswith("KeyboardInterrupt"):
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
