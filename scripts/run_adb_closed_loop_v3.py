#!/usr/bin/env python3
"""Run the v3 state-conditioned KEEP/SWITCH policy over realtime ADB video."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_adb_closed_loop as legacy  # noqa: E402

from karting_agent.control.controller import ControlAction  # noqa: E402
from karting_agent.data_flow.adb import AdbClient  # noqa: E402
from karting_agent.data_flow.execute.adb import AdbExecutor  # noqa: E402
from karting_agent.data_flow.execute.mock import MockExecutor  # noqa: E402
from karting_agent.data_flow.input.adb import AdbInput  # noqa: E402
from karting_agent.data_flow.input.adb_video import AdbVideoInput  # noqa: E402
from karting_agent.model.state_conditioned_runner import (  # noqa: E402
    StateConditionedModelRunner,
)
from karting_agent.runtime.state_conditioned_engine import (  # noqa: E402
    StateConditionedRuntimeConfig,
    StateConditionedRuntimeEngine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 ADB closed-loop POC.")
    parser.add_argument(
        "--runtime-config", type=Path, default=ROOT / "configs" / "runtime.yaml"
    )
    parser.add_argument(
        "--hardware-config", type=Path, default=ROOT / "configs" / "hardware.yaml"
    )
    parser.add_argument(
        "--execute-config", type=Path, default=ROOT / "configs" / "execute.yaml"
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
    parser.add_argument("--switch-threshold", type=float, default=0.6)
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--wait-for-start", action="store_true")
    parser.add_argument(
        "--record-debug",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_seconds <= 0:
        raise ValueError("--max-seconds must be > 0")
    if not 0.0 < args.switch_threshold < 1.0:
        raise ValueError("--switch-threshold must be in (0, 1)")
    if args.wait_for_start and not args.arm:
        raise ValueError("--wait-for-start requires --arm")

    record_debug = (
        args.record_debug
        if args.record_debug is not None
        else bool(args.arm and args.input == "video")
    )
    if record_debug and args.input != "video":
        raise ValueError("--record-debug requires --input video")

    output_path = args.output.resolve() if args.output else legacy.default_output()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recording_path = output_path.with_suffix(".mp4") if record_debug else None

    runtime_raw = legacy.load_mapping(args.runtime_config)
    hardware_raw = legacy.load_mapping(args.hardware_config)
    execute_raw = legacy.load_mapping(args.execute_config)

    client = AdbClient(legacy.resolve_adb_config(hardware_raw, args))
    client.require_device()
    screen_width, screen_height = client.screen_size()
    coordinates = legacy.resolve_coordinates(execute_raw, args)
    if args.arm and coordinates is None:
        raise ValueError("--arm requires explicit ADB touch coordinates via --x/--y")
    if coordinates is not None:
        x, y = coordinates
        if x >= screen_width or y >= screen_height:
            raise ValueError(
                f"touch coordinate ({x}, {y}) is outside screen "
                f"{screen_width}x{screen_height}"
            )

    model = StateConditionedModelRunner(
        args.model,
        metadata_path=args.metadata,
        device=args.device,
    )
    warmup = model.warmup()
    if args.arm:
        assert coordinates is not None
        executor = AdbExecutor(client, x=coordinates[0], y=coordinates[1])
    else:
        executor = MockExecutor()

    target_fps = float(runtime_raw.get("target_fps", 30.0))
    engine = StateConditionedRuntimeEngine(
        model=model,
        preprocess_config=model.spec.preprocess_config,
        executor=executor,
        config=StateConditionedRuntimeConfig(
            target_fps=target_fps,
            frame_offsets_ms=model.spec.frame_offsets_ms,
            prediction_horizon_ms=model.spec.prediction_horizon_ms,
            switch_threshold=args.switch_threshold,
        ),
        initial_pressed=False,
    )

    video_input: AdbVideoInput | None = None
    if args.input == "video":
        video_config = legacy.resolve_adb_video_config(hardware_raw, args)
        video_input = AdbVideoInput(
            client,
            screen_size=(screen_width, screen_height),
            config=video_config,
            record_path=recording_path,
        )
        adb_input = video_input
    else:
        adb_input = AdbInput(client)

    mode = "ARMED" if args.arm else "DRY-RUN"
    print(f"Mode: {mode}; policy=v3-state-conditioned; input={args.input}", flush=True)
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
        f"switch_horizon={model.spec.prediction_horizon_ms:g}ms, "
        f"switch_threshold={args.switch_threshold:.2f}",
        flush=True,
    )
    if recording_path is not None:
        print(f"Debug recording: {recording_path}", flush=True)

    steps = []
    input_frames = 0
    read_frame_timestamps_ms: list[float] = []
    stop_reason = "duration"
    shutdown_error: str | None = None
    safety_release = False
    started: float | None = None
    ended: float | None = None
    decoded_start = dropped_start = interval_start = 0
    decoded_end = dropped_end = interval_end = 0
    control_start_source_frame: int | None = None
    control_start_timestamp_ms: float | None = None

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
                "READY: realtime input, v3 model and ADB executor are warm. "
                "Enter the race, then press Enter when the 3-second countdown ends.",
                flush=True,
            )
            input()
            if video_input is not None:
                pending_frame = video_input.read()
            print("START: v3 closed-loop control enabled.", flush=True)

        if video_input is not None:
            if pending_frame is None:
                pending_frame = video_input.read()
            control_start_source_frame = pending_frame.frame_index
            control_start_timestamp_ms = pending_frame.timestamp_ms
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
                    f"p_switch={step.probability:.3f} action={step.action.value:<7} "
                    f"state={'PRESS' if step.pressed else 'RELEASE':<7} "
                    f"{input_detail}infer={step.inference_ms:.2f}ms{suffix}",
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

    elapsed = ended - started if started is not None and ended is not None else 0.0
    inference_stats = legacy.stats([step.inference_ms for step in steps])
    execute_stats = legacy.stats(executor.execute_latencies_ms) if args.arm else legacy.stats([])
    state_changes = sum(step.action is not ControlAction.HOLD for step in steps)
    control_fps = len(steps) / elapsed if elapsed > 0 else 0.0

    if video_input is not None:
        decode_intervals = video_input.decode_intervals_ms[interval_start:interval_end]
        frame_stats = legacy.stats(decode_intervals)
        input_fps = 1000.0 / frame_stats["mean_ms"] if frame_stats["mean_ms"] > 0 else 0.0
        capture_payload = {
            "mode": "video",
            "frames_read": input_frames,
            "decoded_frames": max(0, decoded_end - decoded_start),
            "decoded_frame_count_total": len(video_input.decoded_frame_timestamps_ms),
            "decoded_frame_timestamps_ms": list(video_input.decoded_frame_timestamps_ms),
            "dropped_frames": max(0, dropped_end - dropped_start),
            "fps": input_fps,
            "startup_ms": video_input.startup_ms,
            "resolution": [video_input.decode_width, video_input.decode_height],
            "frame_interval_ms": frame_stats,
        }
        print(
            "Summary: "
            f"elapsed={elapsed:.2f}s frames_read={input_frames} "
            f"decoded={capture_payload['decoded_frames']} "
            f"dropped={capture_payload['dropped_frames']} input_fps={input_fps:.1f} "
            f"steps={len(steps)} control_fps={control_fps:.1f} "
            f"state_changes={state_changes} infer_mean={inference_stats['mean_ms']:.2f}ms "
            f"infer_p95={inference_stats['p95_ms']:.2f}ms safety_release={safety_release}",
            flush=True,
        )
    else:
        capture_stats = legacy.stats(adb_input.capture_latencies_ms)
        capture_payload = {
            "mode": "screencap",
            "frames": input_frames,
            "fps": input_frames / elapsed if elapsed > 0 else 0.0,
            **capture_stats,
        }
        print(
            "Summary: "
            f"elapsed={elapsed:.2f}s frames={input_frames} steps={len(steps)} "
            f"state_changes={state_changes} control_fps={control_fps:.1f} "
            f"infer_mean={inference_stats['mean_ms']:.2f}ms "
            f"infer_p95={inference_stats['p95_ms']:.2f}ms safety_release={safety_release}",
            flush=True,
        )

    if args.arm:
        print(
            f"Execute enqueue: mean={execute_stats['mean_ms']:.2f}ms "
            f"p95={execute_stats['p95_ms']:.2f}ms max={execute_stats['max_ms']:.2f}ms",
            flush=True,
        )

    recording_payload = (
        {
            "path": str(recording_path),
            "format": "mp4",
            "codec": "mp4v",
            "resolution": [video_input.decode_width, video_input.decode_height]
            if video_input is not None
            else None,
            "timing": "cfr_visual_only",
            "fps": video_input.recording_fps if video_input is not None else None,
            "frame_count": video_input.recorded_frames if video_input is not None else None,
            "source_frame_mapping": "mp4_frame_index_equals_decoded_frame_index",
            "frame_mapping_valid": (
                video_input.recording_frame_mapping_valid
                if video_input is not None
                else False
            ),
            "dropped_frames": (
                video_input.recording_dropped_frames if video_input is not None else None
            ),
            "dropped_frame_indices": (
                list(video_input.recording_dropped_frame_indices)
                if video_input is not None
                else None
            ),
            "error": video_input.recording_error if video_input is not None else None,
            "includes_pre_control": True,
            "control_start_source_frame": control_start_source_frame,
            "control_start_timestamp_ms": control_start_timestamp_ms,
        }
        if recording_path is not None
        else None
    )
    if recording_payload is not None and not recording_payload["frame_mapping_valid"]:
        print(
            "WARNING: debug MP4/source-frame correspondence is invalid: "
            f"dropped={recording_payload['dropped_frames']} "
            f"error={recording_payload['error']}",
            file=sys.stderr,
            flush=True,
        )

    payload = {
        "mode": mode,
        "policy": "state_conditioned_transition_v3",
        "input_mode": args.input,
        "wait_for_start": args.wait_for_start,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed,
        "screen_size": [screen_width, screen_height],
        "touch": None
        if coordinates is None
        else {"x": coordinates[0], "y": coordinates[1]},
        "runtime": {
            "target_fps": target_fps,
            "switch_threshold": args.switch_threshold,
            "prediction_horizon_ms": model.spec.prediction_horizon_ms,
            "frame_offsets_ms": model.spec.frame_offsets_ms,
            "initial_pressed": False,
        },
        "capture": capture_payload,
        "recording": recording_payload,
        "inference": {"steps": len(steps), "fps": control_fps, **inference_stats},
        "execute": {"semantics": "persistent_shell_enqueue", **execute_stats}
        if args.arm
        else None,
        "state_changes": state_changes,
        "safety_release": safety_release,
        "shutdown_error": shutdown_error,
        "read_frame_timestamps_ms": read_frame_timestamps_ms,
        "steps": [{**asdict(step), "action": step.action.value} for step in steps],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Run: {output_path}", flush=True)
    if recording_path is not None:
        print(f"Recording: {recording_path}", flush=True)
    return 0 if shutdown_error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
