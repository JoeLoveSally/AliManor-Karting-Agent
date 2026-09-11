#!/usr/bin/env python3
"""Run the development-only ADB video/screenshot closed-loop POC."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.control.controller import (  # noqa: E402
    ControlAction,
    HysteresisConfig,
    HysteresisController,
)
from karting_agent.data_flow.adb import AdbClient, AdbConfig  # noqa: E402
from karting_agent.data_flow.execute.adb import AdbExecutor  # noqa: E402
from karting_agent.data_flow.execute.mock import MockExecutor  # noqa: E402
from karting_agent.data_flow.input.adb import AdbInput  # noqa: E402
from karting_agent.data_flow.input.adb_video import (  # noqa: E402
    AdbVideoConfig,
    AdbVideoInput,
)
from karting_agent.model.runner import ModelRunner  # noqa: E402
from karting_agent.runtime.engine import RuntimeEngine, RuntimeEngineConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADB closed-loop POC.")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=ROOT / "configs" / "runtime.yaml",
    )
    parser.add_argument(
        "--hardware-config",
        type=Path,
        default=ROOT / "configs" / "hardware.yaml",
    )
    parser.add_argument(
        "--execute-config",
        type=Path,
        default=ROOT / "configs" / "execute.yaml",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--input", choices=("video", "screencap"), default="video")
    parser.add_argument("--adb", type=str, default=None)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--ffmpeg", type=str, default=None)
    parser.add_argument("--decode-width", type=int, default=None)
    parser.add_argument("--video-bit-rate", type=int, default=None)
    parser.add_argument("--video-warmup-seconds", type=float, default=None)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--arm", action="store_true", help="Enable real ADB DOWN/UP events.")
    parser.add_argument(
        "--wait-for-start",
        action="store_true",
        help="Warm the realtime pipeline, then wait for Enter before starting control.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def nested_mapping(raw: dict[str, object], key: str) -> dict[str, object]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} config must be a mapping")
    return value


def resolve_adb_config(raw: dict[str, object], args: argparse.Namespace) -> AdbConfig:
    adb = nested_mapping(raw, "adb")
    config = AdbConfig(
        executable=args.adb or str(adb.get("executable", "adb")),
        serial=args.serial if args.serial is not None else adb.get("serial"),
        command_timeout_seconds=float(adb.get("command_timeout_seconds", 3.0)),
    )
    config.validate()
    return config


def resolve_adb_video_config(
    raw: dict[str, object],
    args: argparse.Namespace,
) -> AdbVideoConfig:
    video = nested_mapping(raw, "adb_video")
    config = AdbVideoConfig(
        ffmpeg_executable=args.ffmpeg or str(video.get("ffmpeg_executable", "ffmpeg")),
        decode_width=(
            args.decode_width
            if args.decode_width is not None
            else int(video.get("decode_width", 360))
        ),
        bit_rate=(
            args.video_bit_rate
            if args.video_bit_rate is not None
            else int(video.get("bit_rate", 8_000_000))
        ),
        startup_timeout_seconds=float(video.get("startup_timeout_seconds", 15.0)),
        frame_timeout_seconds=float(video.get("frame_timeout_seconds", 3.0)),
        warmup_seconds=(
            args.video_warmup_seconds
            if args.video_warmup_seconds is not None
            else float(video.get("warmup_seconds", 0.5))
        ),
    )
    config.validate()
    return config


def resolve_controller_config(raw: dict[str, object]) -> HysteresisConfig:
    control = nested_mapping(raw, "control")
    config = HysteresisConfig(
        press_threshold=float(control.get("press_threshold", 0.55)),
        release_threshold=float(control.get("release_threshold", 0.45)),
    )
    config.validate()
    return config


def resolve_coordinates(
    raw: dict[str, object],
    args: argparse.Namespace,
) -> tuple[int, int] | None:
    adb = nested_mapping(raw, "adb")
    x = args.x if args.x is not None else adb.get("x")
    y = args.y if args.y is not None else adb.get("y")
    if x is None and y is None:
        return None
    if x is None or y is None:
        raise ValueError("ADB touch x/y must be provided together")
    x_int, y_int = int(x), int(y)
    if x_int < 0 or y_int < 0:
        raise ValueError("ADB touch x/y must be >= 0")
    return x_int, y_int


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(array.max()),
    }


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "artifacts" / "adb_runs" / f"adb_{stamp}.json"


def main() -> int:
    args = parse_args()
    if args.max_seconds <= 0:
        raise ValueError("--max-seconds must be > 0")
    if args.wait_for_start and not args.arm:
        raise ValueError("--wait-for-start requires --arm")

    runtime_raw = load_mapping(args.runtime_config)
    hardware_raw = load_mapping(args.hardware_config)
    execute_raw = load_mapping(args.execute_config)

    client = AdbClient(resolve_adb_config(hardware_raw, args))
    client.require_device()
    screen_width, screen_height = client.screen_size()

    coordinates = resolve_coordinates(execute_raw, args)
    if args.arm and coordinates is None:
        raise ValueError(
            "--arm requires explicit ADB touch coordinates via --x/--y or execute.yaml"
        )
    if coordinates is not None:
        x, y = coordinates
        if x >= screen_width or y >= screen_height:
            raise ValueError(
                f"touch coordinate ({x}, {y}) is outside screen "
                f"{screen_width}x{screen_height}"
            )

    model = ModelRunner(args.model, metadata_path=args.metadata, device=args.device)
    warmup = model.warmup()
    controller_config = resolve_controller_config(runtime_raw)
    controller = HysteresisController(controller_config, initial_pressed=False)
    if args.arm:
        assert coordinates is not None
        executor = AdbExecutor(client, x=coordinates[0], y=coordinates[1])
    else:
        executor = MockExecutor()

    target_fps = float(runtime_raw.get("target_fps", 30.0))
    engine = RuntimeEngine(
        model=model,
        preprocess_config=model.spec.preprocess_config,
        controller=controller,
        executor=executor,
        config=RuntimeEngineConfig(
            target_fps=target_fps,
            frame_offsets_ms=model.spec.frame_offsets_ms,
            prediction_horizon_ms=model.spec.prediction_horizon_ms,
        ),
    )

    video_input: AdbVideoInput | None = None
    if args.input == "video":
        video_config = resolve_adb_video_config(hardware_raw, args)
        video_input = AdbVideoInput(
            client,
            screen_size=(screen_width, screen_height),
            config=video_config,
        )
        adb_input = video_input
    else:
        adb_input = AdbInput(client)

    mode = "ARMED" if args.arm else "DRY-RUN"
    print(f"Mode: {mode}; input={args.input}", flush=True)
    print(
        f"ADB: serial={client.config.serial or '<default>'}; "
        f"screen={screen_width}x{screen_height}",
        flush=True,
    )
    if coordinates is not None:
        print(f"Touch: x={coordinates[0]}, y={coordinates[1]}", flush=True)
    print(
        f"Model: {model.model_path} ({model.spec.architecture}, device={model.device}); "
        f"warmup first={warmup[0]:.2f}ms last={warmup[-1]:.2f}ms",
        flush=True,
    )
    print(
        f"Runtime: target_fps={target_fps:g}, offsets={model.spec.frame_offsets_ms}, "
        f"horizon={model.spec.prediction_horizon_ms:g}ms, "
        f"press>={controller_config.press_threshold:.2f}, "
        f"release<={controller_config.release_threshold:.2f}",
        flush=True,
    )

    steps = []
    input_frames = 0
    read_frame_timestamps_ms: list[float] = []
    stop_reason = "duration"
    shutdown_error: str | None = None
    safety_release = False
    started: float | None = None
    ended: float | None = None
    decoded_start = 0
    decoded_end = 0
    dropped_start = 0
    dropped_end = 0
    interval_start = 0
    interval_end = 0

    try:
        pending_frame = None
        if video_input is not None:
            pending_frame = video_input.read()
            if video_input.config.warmup_seconds:
                time.sleep(video_input.config.warmup_seconds)
                pending_frame = video_input.read()
            print(
                f"Video input: {video_input.decode_width}x{video_input.decode_height}, "
                f"bit_rate={video_input.config.bit_rate}, "
                f"startup={video_input.startup_ms:.1f}ms, "
                f"warmup={video_input.config.warmup_seconds:g}s",
                flush=True,
            )

        if args.arm:
            executor.start()

        if args.wait_for_start:
            print(
                "READY: realtime input, model and ADB executor are warm. "
                "Enter the race, then press Enter when the 3-second countdown ends.",
                flush=True,
            )
            input()
            if video_input is not None:
                pending_frame = video_input.read()
            print("START: closed-loop control enabled.", flush=True)

        if video_input is not None:
            decoded_start = video_input.decoded_frames
            dropped_start = video_input.dropped_frames
            interval_start = len(video_input.decode_intervals_ms)

        started = time.perf_counter()
        while time.perf_counter() - started < args.max_seconds:
            if pending_frame is None:
                frame = adb_input.read()
            else:
                frame = pending_frame
                pending_frame = None

            input_frames += 1
            read_frame_timestamps_ms.append(frame.timestamp_ms)
            step = engine.ingest(frame)
            if step is None:
                continue
            steps.append(step)
            if args.verbose or step.action is not ControlAction.HOLD:
                if video_input is None:
                    input_detail = f"capture={adb_input.last_capture_ms:.1f}ms "
                else:
                    input_detail = (
                        f"source_frame={frame.frame_index} "
                        f"dropped={video_input.dropped_frames - dropped_start} "
                    )
                suffix = ""
                if args.arm and step.action is not ControlAction.HOLD:
                    suffix = f" enqueue={executor.last_execute_ms:.2f}ms"
                print(
                    f"t={step.observation_timestamp_ms:8.1f}ms "
                    f"p={step.probability:.3f} action={step.action.value:<7} "
                    f"state={'PRESS' if step.pressed else 'RELEASE':<7} "
                    f"{input_detail}"
                    f"infer={step.inference_ms:.2f}ms{suffix}",
                    flush=True,
                )
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("Interrupted; requesting safety RELEASE...", flush=True)
    finally:
        ended = time.perf_counter() if started is not None else None
        if video_input is not None:
            decoded_end = video_input.decoded_frames
            dropped_end = video_input.dropped_frames
            interval_end = len(video_input.decode_intervals_ms)
        try:
            safety_release = engine.shutdown()
        except Exception as exc:
            shutdown_error = str(exc)
            print(f"WARNING: safety RELEASE failed: {exc}", file=sys.stderr, flush=True)
        adb_input.close()
        if args.arm:
            executor.close()

    elapsed = (
        ended - started
        if started is not None and ended is not None
        else 0.0
    )
    inference_stats = stats([step.inference_ms for step in steps])
    execute_stats = stats(executor.execute_latencies_ms) if args.arm else stats([])
    state_changes = sum(step.action is not ControlAction.HOLD for step in steps)
    control_fps = len(steps) / elapsed if elapsed > 0 else 0.0

    if video_input is not None:
        decode_intervals = video_input.decode_intervals_ms[interval_start:interval_end]
        frame_interval_stats = stats(decode_intervals)
        input_fps = (
            1000.0 / frame_interval_stats["mean_ms"]
            if frame_interval_stats["mean_ms"] > 0
            else 0.0
        )
        decoded_frames = max(0, decoded_end - decoded_start)
        dropped_frames = max(0, dropped_end - dropped_start)
        capture_payload = {
            "mode": "video",
            "frames_read": input_frames,
            "decoded_frames": decoded_frames,
            "dropped_frames": dropped_frames,
            "fps": input_fps,
            "startup_ms": video_input.startup_ms,
            "resolution": [video_input.decode_width, video_input.decode_height],
            "frame_interval_ms": frame_interval_stats,
        }
        print(
            "Summary: "
            f"elapsed={elapsed:.2f}s frames_read={input_frames} "
            f"decoded={decoded_frames} dropped={dropped_frames} "
            f"input_fps={input_fps:.1f} "
            f"frame_p95={frame_interval_stats['p95_ms']:.1f}ms "
            f"steps={len(steps)} control_fps={control_fps:.1f} "
            f"state_changes={state_changes} "
            f"infer_mean={inference_stats['mean_ms']:.2f}ms "
            f"infer_p95={inference_stats['p95_ms']:.2f}ms "
            f"safety_release={safety_release}",
            flush=True,
        )
    else:
        capture_stats = stats(adb_input.capture_latencies_ms)
        capture_fps = input_frames / elapsed if elapsed > 0 else 0.0
        capture_payload = {
            "mode": "screencap",
            "frames": input_frames,
            "fps": capture_fps,
            **capture_stats,
        }
        print(
            "Summary: "
            f"elapsed={elapsed:.2f}s frames={input_frames} "
            f"steps={len(steps)} state_changes={state_changes} "
            f"capture_fps={capture_fps:.1f} control_fps={control_fps:.1f} "
            f"capture_mean={capture_stats['mean_ms']:.1f}ms "
            f"capture_p95={capture_stats['p95_ms']:.1f}ms "
            f"infer_mean={inference_stats['mean_ms']:.2f}ms "
            f"infer_p95={inference_stats['p95_ms']:.2f}ms "
            f"safety_release={safety_release}",
            flush=True,
        )

    if args.arm:
        print(
            f"Execute enqueue: mean={execute_stats['mean_ms']:.2f}ms "
            f"p95={execute_stats['p95_ms']:.2f}ms "
            f"max={execute_stats['max_ms']:.2f}ms",
            flush=True,
        )

    output_path = args.output.resolve() if args.output is not None else default_output()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "input_mode": args.input,
        "wait_for_start": args.wait_for_start,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed,
        "screen_size": [screen_width, screen_height],
        "touch": (
            None
            if coordinates is None
            else {"x": coordinates[0], "y": coordinates[1]}
        ),
        "runtime": {
            "target_fps": target_fps,
            "press_threshold": controller_config.press_threshold,
            "release_threshold": controller_config.release_threshold,
            "prediction_horizon_ms": model.spec.prediction_horizon_ms,
            "frame_offsets_ms": model.spec.frame_offsets_ms,
        },
        "capture": capture_payload,
        "inference": {
            "steps": len(steps),
            "fps": control_fps,
            **inference_stats,
        },
        "execute": (
            {"semantics": "persistent_shell_enqueue", **execute_stats}
            if args.arm
            else None
        ),
        "state_changes": state_changes,
        "safety_release": safety_release,
        "shutdown_error": shutdown_error,
        "read_frame_timestamps_ms": read_frame_timestamps_ms,
        "steps": [
            {**asdict(step), "action": step.action.value}
            for step in steps
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Run: {output_path}", flush=True)
    return 0 if shutdown_error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
