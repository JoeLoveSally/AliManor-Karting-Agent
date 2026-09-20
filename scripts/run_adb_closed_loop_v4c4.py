#!/usr/bin/env python3
"""Run V4-C4 current-action-only control over local ADB H.264 video.

Default is a no-touch dry run. Armed execution requires --arm --x --y.
The existing EventTimePolicyDecoder is used with event_class=no_event_class.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_adb_closed_loop as legacy  # noqa: E402
from karting_agent.data_flow.adb import AdbClient  # noqa: E402
from karting_agent.data_flow.execute.adb import AdbExecutor  # noqa: E402
from karting_agent.data_flow.execute.mock import MockExecutor  # noqa: E402
from karting_agent.data_flow.input.adb_video import AdbVideoInput  # noqa: E402
from karting_agent.model.event_time_runner import EventTimeActionRunner  # noqa: E402
from karting_agent.runtime.event_time_action_engine import (  # noqa: E402
    ActionRuntimeConfig,
    EventTimeActionEngine,
)
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


DEFAULT_ARTIFACT = ROOT / "artifacts/models/mobilenet_v3_small_v4c4_event_time/model.pt"
DEFAULT_TRAIN_CONFIG = ROOT / "configs/train_v4c4_event_time.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V4-C4 local-ADB action-only closed loop")
    parser.add_argument("--model", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--runtime-config", type=Path, default=ROOT / "configs/runtime.yaml")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs/hardware.yaml")
    parser.add_argument("--adb", type=str, default=None)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--ffmpeg", type=str, default=None)
    parser.add_argument("--decode-width", type=int, default=None)
    parser.add_argument("--video-bit-rate", type=int, default=None)
    parser.add_argument("--video-warmup-seconds", type=float, default=None)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--record-debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--arm", action="store_true", help="Enable real Android touch commands")
    parser.add_argument("--wait-for-start", action="store_true")
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def runtime_config_from_training(raw: dict[str, object], *, target_fps: float) -> ActionRuntimeConfig:
    model = legacy.nested_mapping(raw, "model")
    data = legacy.nested_mapping(raw, "dataset")
    event = legacy.nested_mapping(raw, "event_time")
    stack = int(model["frame_stack"])
    history = float(data["history_ms"])
    interval = float(data["frame_interval_ms"])
    offsets = tuple(-history + interval * index for index in range(stack))
    if len(offsets) != 5 or abs(offsets[-1]) > 1e-6:
        raise ValueError("training history/frame interval must produce five frames ending now")
    if str(model["input_representation"]) != "current_rgb_plus_adjacent_deltas":
        raise ValueError("unexpected training input representation")
    bin_ms = float(event["bin_ms"])
    max_horizon = float(event["max_horizon_ms"])
    bins = round(max_horizon / bin_ms)
    if bins < 1 or abs(bins * bin_ms - max_horizon) > 1e-6:
        raise ValueError("invalid event-time bins in training config")
    config = ActionRuntimeConfig(
        target_fps=target_fps,
        frame_offsets_ms=offsets,
        action_threshold=0.5,
        min_state_hold_ms=100.0,
        event_time_bin_ms=bin_ms,
        no_event_class=bins,
    )
    config.validate()
    return config


def validate_artifact_training_match(model: EventTimeActionRunner, raw: dict[str, object], config: ActionRuntimeConfig) -> None:
    trained = legacy.nested_mapping(raw, "model")
    metadata = model.metadata
    for key in ("architecture", "frame_stack", "input_representation", "visual_feature_dim", "hidden_dim"):
        if metadata[key] != trained[key]:
            raise ValueError(f"checkpoint/train-config mismatch: {key}")
    if metadata["event_time_classes"] != config.no_event_class + 1:
        raise ValueError("checkpoint/train-config mismatch: event_time_classes")
    if metadata["no_event_class"] != config.no_event_class:
        raise ValueError("checkpoint/train-config mismatch: no_event_class")
    if float(metadata["event_time_bin_ms"]) != config.event_time_bin_ms:
        raise ValueError("checkpoint/train-config mismatch: event_time_bin_ms")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_seconds <= 0:
        raise ValueError("--max-seconds must be > 0")
    if args.wait_for_start and not args.arm:
        raise ValueError("--wait-for-start requires --arm")
    if args.arm and (args.x is None or args.y is None):
        raise ValueError("--arm requires explicit --x and --y (configuration defaults do not count)")
    if (args.x is None) != (args.y is None):
        raise ValueError("--x and --y must be provided together")
    if args.x is not None and (args.x < 0 or args.y < 0):
        raise ValueError("touch coordinates must be nonnegative")
    record_debug = args.record_debug if args.record_debug is not None else args.arm
    output = args.output.resolve() if args.output else legacy.default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    recording = output.with_suffix(".mp4") if record_debug else None
    train_raw = legacy.load_mapping(args.train_config)
    runtime_raw = legacy.load_mapping(args.runtime_config)
    hardware_raw = legacy.load_mapping(args.hardware_config)
    config = runtime_config_from_training(
        train_raw, target_fps=float(runtime_raw.get("target_fps", 30.0))
    )
    preprocess = preprocess_config_from_mapping(train_raw)
    if preprocess.input_height != 224 or preprocess.input_width != 224:
        raise ValueError("V4-C4 checkpoint expects 224x224 preprocessing")
    model = EventTimeActionRunner(
        args.model, metadata_path=args.metadata, device=args.device,
        torch_num_threads=args.torch_num_threads,
    )
    validate_artifact_training_match(model, train_raw, config)
    warmup = model.warmup()
    client = AdbClient(legacy.resolve_adb_config(hardware_raw, args))
    client.require_device()
    screen_width, screen_height = client.screen_size()
    if args.x is not None and (args.x >= screen_width or args.y >= screen_height):
        raise ValueError("touch coordinates outside Android screen bounds")
    executor = (
        AdbExecutor(client, x=args.x, y=args.y)
        if args.arm else MockExecutor()
    )
    engine = EventTimeActionEngine(
        model=model, executor=executor, preprocess_config=preprocess, config=config
    )
    video = AdbVideoInput(
        client, screen_size=(screen_width, screen_height),
        config=legacy.resolve_adb_video_config(hardware_raw, args),
        record_path=recording,
    )
    print(
        f"Mode={'ARMED' if args.arm else 'DRY-RUN'} model=V4-C4 action-only "
        f"device={model.device} serial={client.config.serial or '<default>'} "
        f"offsets={config.frame_offsets_ms} threshold={config.action_threshold} "
        f"min_hold={config.min_state_hold_ms}ms warmup={warmup[-1]:.2f}ms",
        flush=True,
    )
    if recording:
        print(f"Debug video: {recording}", flush=True)
    steps: list[dict[str, object]] = []
    read_timestamps: list[float] = []
    stop_reason = "duration"
    errors: list[str] = []
    safety_release = False
    start: float | None = None
    elapsed = 0.0
    capture_start = (0, 0, 0)
    try:
        frame = video.read()
        if video.config.warmup_seconds:
            time.sleep(video.config.warmup_seconds)
            frame = video.read()
        if args.arm:
            executor.start()
        if args.wait_for_start:
            print("READY: enter race, then press Enter after countdown finishes.", flush=True)
            input()
            frame = video.read()
        capture_start = (
            video.decoded_frames, video.dropped_frames, len(video.decode_intervals_ms)
        )
        start = time.perf_counter()
        while time.perf_counter() - start < args.max_seconds:
            read_timestamps.append(frame.timestamp_ms)
            step = engine.ingest(frame)
            if step is not None:
                steps.append(step)
                if args.verbose or step["events"] or step["decoder_reason"] == "action_hold":
                    print(
                        f"t={step['observation_timestamp_ms']:.1f}ms "
                        f"p_press={step['action_probability']:.3f} "
                        f"state={'PRESS' if step['pressed'] else 'RELEASE'} "
                        f"reason={step['decoder_reason']} "
                        f"pending={step['pending_due_ms']} "
                        f"frame={step['source_frame_index']} "
                        f"infer={step['inference_ms']:.2f}ms "
                        f"events={step['events']}", flush=True,
                    )
            frame = video.read()
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    except Exception as exc:
        stop_reason = "error"
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if start is not None:
            elapsed = time.perf_counter() - start
        try:
            safety_release = engine.shutdown()
        except Exception as exc:
            errors.append(f"safety release failed: {type(exc).__name__}: {exc}")
        try:
            video.close()
        except Exception as exc:
            errors.append(f"video close failed: {type(exc).__name__}: {exc}")
        if args.arm:
            try:
                executor.close()
            except Exception as exc:
                errors.append(f"ADB executor close failed: {type(exc).__name__}: {exc}")
    decoded = video.decoded_frames - capture_start[0]
    dropped = video.dropped_frames - capture_start[1]
    intervals = video.decode_intervals_ms[capture_start[2]:]
    input_stats = legacy.stats(intervals)
    infer_stats = legacy.stats([float(step["inference_ms"]) for step in steps])
    transitions = [event for step in steps for event in step["events"]]
    payload = {
        "model": str(model.model_path), "metadata": str(model.metadata_path),
        "policy": "v4c4_current_action_only", "mode": "armed" if args.arm else "dry_run",
        "device": str(model.device), "action_threshold": config.action_threshold,
        "min_state_hold_ms": config.min_state_hold_ms,
        "frame_offsets_ms": config.frame_offsets_ms, "elapsed_seconds": elapsed,
        "stop_reason": stop_reason, "errors": errors,
        "control_fps": len(steps) / elapsed if elapsed else 0.0,
        "read_frames": len(read_timestamps), "read_frame_timestamps_ms": read_timestamps,
        "decoded_frames": decoded, "dropped_frames": dropped,
        "decoded_frame_timestamps_ms": video.decoded_frame_timestamps_ms,
        "recording": str(recording) if recording else None,
        "recording_frame_mapping_valid": video.recording_frame_mapping_valid,
        "input_interval_ms": input_stats, "inference_ms": infer_stats,
        "transitions": transitions, "transition_reasons": dict(Counter(event["reason"] for event in transitions)),
        "steps": steps, "safety_release": safety_release,
        "host_enqueue_latency_ms": legacy.stats(executor.execute_latencies_ms) if args.arm else None,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Summary: control_fps={payload['control_fps']:.1f} "
        f"decoded={decoded} dropped={dropped} transitions={len(transitions)} "
        f"infer_p95={infer_stats['p95_ms']:.2f}ms "
        f"safety_release={safety_release} status={stop_reason} output={output}",
        flush=True,
    )
    if errors:
        print("; ".join(errors), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
