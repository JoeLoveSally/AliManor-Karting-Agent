#!/usr/bin/env python3
"""Inspect stamp pixels via ADB screencap and the existing decoded video stream.

Keep the old browser probe page (black/white bars) in the foreground. The HTTP
stamp server need not be running: the last painted canvas persists. This tool
sends NO touch commands, performs NO game inference, and saves NO image files.
Only numeric brightness samples and bit strings are written to its JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_adb_closed_loop as legacy  # noqa: E402
from diagnose_adb_video_frame_age import BITS, SYNC, decode_stamp  # noqa: E402
from karting_agent.data_flow.adb import AdbClient, AdbConfig  # noqa: E402
from karting_agent.data_flow.input.adb_video import AdbVideoInput  # noqa: E402

ROWS = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95)


def inspect_pixels(image: np.ndarray) -> dict[str, object]:
    """Return only 24 mean-brightness samples per row; never return image bytes."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected HxWx3 BGR image")
    height, width = image.shape[:2]
    rows = []
    for fraction in ROWS:
        y = min(height - 1, int(height * fraction))
        pixels = np.stack([
            image[y, min(width - 1, int((bit + 0.5) * width / BITS))]
            for bit in range(BITS)
        ]).astype(np.int16)
        luma = np.rint(pixels.mean(axis=1)).astype(np.int16)
        brightness = [int(value) for value in luma]
        bits = "".join("0" if value < 65 else "1" if value > 190 else "?"
                       for value in brightness)
        rows.append({
            "y_fraction": fraction,
            "y_pixel": y,
            "brightness_24": brightness,
            "threshold_bits_24": bits,
            "sync_bits": bits[:8],
            "sync_matches": bits[:8] == f"{SYNC:08b}",
            "neutral_samples": bits.count("?"),
        })
    return {
        "height": height,
        "width": width,
        "existing_decoder_sequence": decode_stamp(image),
        "rows": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Numeric ADB stamp-pixel diagnostic; no saved screenshots")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs/hardware.yaml")
    parser.add_argument("--adb", default=None)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--decode-width", type=int, default=360)
    parser.add_argument("--video-bit-rate", type=int, default=None)
    parser.add_argument("--video-warmup-seconds", type=float, default=None)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/adb_runs/video_stamp_pixels_360.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw = legacy.load_mapping(args.hardware_config)
    adb = legacy.resolve_adb_config(raw, args)
    client = AdbClient(AdbConfig(executable=adb.executable, serial=adb.serial,
                                 command_timeout_seconds=15.0))
    client.require_device()
    payload: dict[str, object] = {
        "purpose": "Compare numeric stamp pixels in Android screencap and ADB decoded video",
        "note": "No screenshot or video file saved. Source screenshot is held only in memory.",
        "expected_sync_bits": f"{SYNC:08b}",
    }
    video = None
    try:
        screen_size = client.screen_size()
        video = AdbVideoInput(
            client, screen_size=screen_size,
            config=legacy.resolve_adb_video_config(raw, args),
        )
        # First read uses AdbVideoInput's longer startup timeout. A static
        # browser may emit only one frame; subsequent frames are optional.
        first = video.read()
        frames = [first]
        for _ in range(3):
            try:
                frames.append(video.read(timeout_seconds=0.4))
            except RuntimeError as exc:
                if "timed out waiting for Android video frame" in str(exc):
                    break
                raise
        payload["video_samples"] = [
            {"source_frame_index": frame.frame_index,
             "timestamp_ms": frame.timestamp_ms,
             **inspect_pixels(frame.image)}
            for frame in ([frames[0], frames[-1]] if len(frames) > 1 else frames)
        ]
        # ADB screencap bypasses screenrecord/H264/FFmpeg. It is decoded only
        # in memory and reduced to 24 numeric brightness values per sample row.
        png = client.run("exec-out", "screencap", "-p")
        screenshot = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if screenshot is None:
            raise RuntimeError("cannot decode in-memory adb screencap PNG")
        payload["android_screencap_sample"] = inspect_pixels(screenshot)
        payload["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        payload["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        payload["error"] = str(exc) or type(exc).__name__
    finally:
        if video is not None:
            payload["video_health"] = {
                "decoded_frames": video.decoded_frames,
                "dropped_frames": video.dropped_frames,
                "reader_error": video._error,
            }
            try:
                video.close()
            except Exception as exc:
                payload["video_close_error"] = str(exc)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        print(f"status={payload['status']} output={output}", flush=True)
        for name in ("video_samples", "android_screencap_sample"):
            sample = payload.get(name)
            if isinstance(sample, list):
                sample = sample[-1] if sample else None
            if isinstance(sample, dict):
                print(f"{name}: {sample['width']}x{sample['height']} "
                      f"decoder={sample['existing_decoder_sequence']} "
                      f"sync_rows={[row['y_fraction'] for row in sample['rows'] if row['sync_matches']]}",
                      flush=True)
        if "error" in payload:
            print(f"error: {payload['error']}", file=sys.stderr, flush=True)
    return 0 if payload["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
