#!/usr/bin/env python3
"""Measure host-command -> decoded-screen-change delay on a benign Android screen.

Run with the phone UNLOCKED and showing the static Android Settings main page.
This program alternates HOME and opening Settings; it never touches the game,
uses the existing live screenrecord+FFmpeg video path, and saves NO screenshots.
The measurement includes adb dispatch, Android UI/display response, screenrecord,
transport, decode, and host polling. It is NOT pure screenrecord latency.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from threading import Event, Thread
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_adb_closed_loop as legacy  # noqa: E402
from karting_agent.data_flow.adb import AdbClient, AdbConfig  # noqa: E402
from karting_agent.data_flow.input.adb_video import AdbVideoConfig, AdbVideoInput  # noqa: E402
from karting_agent.runtime.video_latency_probe import (  # noqa: E402
    ScreenChangeGate,
    changed_pixel_fraction,
)


MARKERS = (
    ("home", ("shell", "input", "keyevent", "KEYCODE_HOME")),
    ("settings", ("shell", "am", "start", "-a", "android.settings.SETTINGS")),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-game ADB video responsiveness probe")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs/hardware.yaml")
    parser.add_argument("--adb", type=str, default=None)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--ffmpeg", type=str, default=None)
    parser.add_argument("--decode-width", type=int, default=360)
    parser.add_argument("--video-bit-rate", type=int, default=None)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--pixel-delta", type=int, default=32)
    parser.add_argument("--changed-fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def baseline_frame(video: AdbVideoInput, *, settle_seconds: float,
                   pixel_delta: int) -> tuple[object, float]:
    """Consume recent decoded frames before arming a marker; reject moving screens."""
    deadline = time.perf_counter() + settle_seconds
    previous = None
    last = None
    seen = 0
    while time.perf_counter() < deadline or seen < 3:
        current = video.read()
        if previous is not None:
            last = changed_pixel_fraction(previous.image, current.image,
                                          pixel_delta=pixel_delta)
        previous = current
        seen += 1
    assert previous is not None
    if last is None or last > 0.08:
        raise RuntimeError(
            f"baseline changed too much (fraction={last!r}); "
            "open Android Settings main page and wait for animations to stop"
        )
    return previous, last


def measure_marker(video: AdbVideoInput, client: AdbClient, *,
                   baseline, target: str, command: tuple[str, ...],
                   timeout_seconds: float, pixel_delta: int,
                   changed_fraction: float) -> dict[str, object]:
    """Watch video while a SECOND thread issues the screen change command."""
    started = Event()
    timing: dict[str, object] = {}

    def invoke() -> None:
        timing["command_started_monotonic_ms"] = time.perf_counter() * 1000.0
        started.set()
        try:
            client.run(*command)
        except Exception as exc:
            timing["command_error"] = str(exc)
        finally:
            timing["command_returned_monotonic_ms"] = time.perf_counter() * 1000.0

    gate = ScreenChangeGate(
        baseline.image,
        changed_fraction_threshold=changed_fraction,
        pixel_delta=pixel_delta,
        consecutive_frames=2,
    )
    worker = Thread(target=invoke, name="adb-screen-marker", daemon=True)
    worker.start()
    if not started.wait(timeout=1.0):
        raise RuntimeError("ADB marker command did not begin")
    issued_ms = float(timing["command_started_monotonic_ms"])
    deadline = time.perf_counter() + timeout_seconds
    onset: tuple[float, int, float] | None = None
    first_decode_ms: float | None = None
    while time.perf_counter() < deadline:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        # read() consumes the same latest-frame queue as the closed-loop runner.
        frame = video.read(timeout_seconds=min(remaining, 2.0))
        read_ms = time.perf_counter() * 1000.0
        if frame.frame_index <= baseline.frame_index:
            continue
        sample = gate.observe(
            image=frame.image,
            observed_monotonic_ms=read_ms,
            frame_index=frame.frame_index,
        )
        if sample is not None:
            onset = sample
            first_decode_ms = frame.timestamp_ms
            break
    worker.join(timeout=client.config.command_timeout_seconds + 1.0)
    if worker.is_alive():
        raise RuntimeError("ADB marker thread did not finish")
    if "command_error" in timing:
        raise RuntimeError(f"ADB marker failed: {timing['command_error']}")
    completed_ms = float(timing["command_returned_monotonic_ms"])
    if onset is None:
        raise RuntimeError(
            f"no confirmed {target} screen transition within {timeout_seconds:g}s; "
            "make sure the phone is unlocked and initially on Android Settings"
        )
    observed_ms, onset_index, fraction = onset
    delay_ms = observed_ms - issued_ms
    if delay_ms < 0:
        raise RuntimeError("screen marker detected before the ADB command began")
    return {
        "target": target,
        "baseline_frame_index": baseline.frame_index,
        "marker_first_frame_index": onset_index,
        "marker_confirmation_frame_timestamp_ms": first_decode_ms,
        "changed_fraction_at_confirmation": fraction,
        "command_started_monotonic_ms": issued_ms,
        "command_returned_monotonic_ms": completed_ms,
        "marker_first_observed_monotonic_ms": observed_ms,
        "command_duration_ms": completed_ms - issued_ms,
        "command_start_to_screen_observed_ms": delay_ms,
        "command_return_to_screen_observed_ms": observed_ms - completed_ms,
    }


def summarize(samples: list[dict[str, object]]) -> dict[str, float]:
    values = np.asarray(
        [float(sample["command_start_to_screen_observed_ms"]) for sample in samples],
        dtype=np.float64,
    )
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
        "min_ms": float(np.min(values)),
        "command_duration_mean_ms": statistics.mean(
            float(sample["command_duration_ms"]) for sample in samples
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.trials < 1 or args.trials > 30:
        raise ValueError("--trials must be between 1 and 30")
    if args.settle_seconds < 0.5 or args.timeout_seconds <= 0:
        raise ValueError("settle >= 0.5s and timeout > 0s are required")
    if not 0 < args.changed_fraction <= 1:
        raise ValueError("--changed-fraction must be in (0, 1]")
    raw = legacy.load_mapping(args.hardware_config)
    adb_conf = legacy.resolve_adb_config(raw, args)
    # Launching Settings may legitimately exceed the normal 3s query timeout.
    client = AdbClient(AdbConfig(
        executable=adb_conf.executable,
        serial=adb_conf.serial,
        command_timeout_seconds=max(8.0, adb_conf.command_timeout_seconds),
    ))
    client.require_device()
    screen = client.screen_size()
    video_conf = legacy.resolve_adb_video_config(raw, args)
    video = AdbVideoInput(
        client,
        screen_size=screen,
        config=AdbVideoConfig(
            ffmpeg_executable=video_conf.ffmpeg_executable,
            decode_width=args.decode_width,
            bit_rate=video_conf.bit_rate,
            startup_timeout_seconds=video_conf.startup_timeout_seconds,
            frame_timeout_seconds=video_conf.frame_timeout_seconds,
            warmup_seconds=video_conf.warmup_seconds,
        ),
    )
    output = (args.output or ROOT / "artifacts/adb_runs" /
              f"adb_video_latency_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json").resolve()
    payload: dict[str, object] = {
        "measurement": "host ADB command start -> first confirmed screen change observed by WSL",
        "limitations": (
            "Includes ADB command dispatch, Android UI/display response, "
            "screenrecord encode/transport, FFmpeg decode and host polling. "
            "Not pure camera-to-host latency, not game-control latency; no screen images saved."
        ),
        "trials_requested": args.trials,
        "decode_width": video.decode_width,
        "decode_height": video.decode_height,
        "samples": [],
    }
    last_target: str | None = None
    try:
        print("Probe only: phone must be unlocked and showing static Android SETTINGS. "
              "Will alternate HOME / SETTINGS; no game touches or screenshots.", flush=True)
        video.read()  # start the normal screenrecord+FFmpeg pipeline
        for index in range(args.trials):
            base, baseline_noise = baseline_frame(
                video, settle_seconds=args.settle_seconds, pixel_delta=args.pixel_delta
            )
            target, command = MARKERS[index % len(MARKERS)]
            last_target = target
            result = measure_marker(
                video, client, baseline=base, target=target, command=command,
                timeout_seconds=args.timeout_seconds,
                pixel_delta=args.pixel_delta,
                changed_fraction=args.changed_fraction,
            )
            result["trial"] = index + 1
            result["baseline_changed_fraction"] = baseline_noise
            payload["samples"].append(result)
            print(f"trial={index + 1} target={target} "
                  f"delay={result['command_start_to_screen_observed_ms']:.1f}ms "
                  f"command={result['command_duration_ms']:.1f}ms", flush=True)
        payload["summary"] = summarize(payload["samples"])
        payload["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        payload["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        payload["error"] = str(exc) or type(exc).__name__
        print(f"Probe stopped: {payload['error']}", file=sys.stderr, flush=True)
    finally:
        try:
            video.close()
        except Exception as exc:
            payload["video_close_error"] = str(exc)
        if last_target == "home":
            # Leave the phone on the benign Settings page, not in the game.
            try:
                client.run("shell", "am", "start", "-a", "android.settings.SETTINGS")
            except Exception as exc:
                payload["restore_settings_error"] = str(exc)
        payload["decoded_frames"] = video.decoded_frames
        payload["dropped_frames"] = video.dropped_frames
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"status={payload['status']} output={output}", flush=True)
        if "summary" in payload:
            print(json.dumps(payload["summary"], indent=2), flush=True)
    return 0 if payload["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
