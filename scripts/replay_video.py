#!/usr/bin/env python3
"""Replay a recorded gameplay video through the runtime pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.control.controller import HysteresisConfig, HysteresisController
from karting_agent.data_flow.execute.mock import MockExecutor
from karting_agent.data_flow.input.video import VideoInput
from karting_agent.model.runner import ModelRunner
from karting_agent.runtime.engine import RuntimeEngine, RuntimeEngineConfig
from karting_agent.runtime.replay import ReplayRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a video through the runtime pipeline."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=ROOT / "configs" / "runtime.yaml",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def resolve_model_path(
    runtime_raw: dict[str, object], override: Path | None
) -> Path:
    if override is not None:
        return override.resolve()
    model = runtime_raw.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("runtime model config must be a mapping")
    configured_value = model.get("path")
    if configured_value is None or not str(configured_value).strip():
        raise ValueError("runtime model.path is required when --model is omitted")
    configured = Path(str(configured_value))
    resolved = configured if configured.is_absolute() else ROOT / configured
    return resolved if resolved.suffix == ".pt" else resolved / "model.pt"


def controller_config(runtime_raw: dict[str, object]) -> HysteresisConfig:
    control = runtime_raw.get("control", {})
    if not isinstance(control, dict):
        raise ValueError("runtime control config must be a mapping")
    defaults = HysteresisConfig()
    config = HysteresisConfig(
        press_threshold=float(
            control.get("press_threshold", defaults.press_threshold)
        ),
        release_threshold=float(
            control.get("release_threshold", defaults.release_threshold)
        ),
    )
    config.validate()
    return config


def default_output(model_path: Path, video_path: Path) -> Path:
    artifact_name = model_path.parent.name
    return (
        ROOT
        / "artifacts"
        / "replays"
        / artifact_name
        / f"{video_path.stem}.json"
    )


def main() -> int:
    args = parse_args()
    runtime_raw = load_mapping(args.runtime_config)
    target_fps = float(runtime_raw.get("target_fps", 30.0))
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    model_path = resolve_model_path(runtime_raw, args.model)
    model = ModelRunner(
        model_path,
        metadata_path=args.metadata,
        device=args.device,
    )
    hysteresis = controller_config(runtime_raw)
    controller = HysteresisController(hysteresis, initial_pressed=False)
    executor = MockExecutor()
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

    print(f"Video: {args.video.resolve()}", flush=True)
    print(
        f"Model: {model_path} ({model.spec.architecture}, device={model.device})",
        flush=True,
    )
    print(
        "Temporal: "
        f"offsets={model.spec.frame_offsets_ms}, "
        f"horizon={model.spec.prediction_horizon_ms:g}ms, "
        f"target_fps={target_fps:g}",
        flush=True,
    )
    print(
        "Controller: "
        f"press>={hysteresis.press_threshold:.3f}, "
        f"release<={hysteresis.release_threshold:.3f}, startup=RELEASE",
        flush=True,
    )

    with VideoInput(args.video) as video_input:
        result = ReplayRunner(engine).run(
            video_input,
            max_seconds=args.max_seconds,
        )

    for step in result.steps:
        if args.verbose or step.action.value != "HOLD":
            print(
                f"t={step.observation_timestamp_ms:8.1f}ms "
                f"target={step.prediction_target_timestamp_ms:8.1f}ms "
                f"p={step.probability:.3f} "
                f"action={step.action.value:<7} "
                f"state={'PRESS' if step.pressed else 'RELEASE':<7} "
                f"infer={step.inference_ms:.2f}ms",
                flush=True,
            )

    summary = result.summary()
    print(
        "Summary: "
        f"frames={result.source_frames_read}, steps={summary['steps']}, "
        f"state_changes={summary['state_changes']}, "
        f"mean_infer={summary['mean_inference_ms']:.2f}ms, "
        f"p95_infer={summary['p95_inference_ms']:.2f}ms, "
        f"processing_fps={summary['processing_fps']:.1f}, "
        f"safety_release={result.safety_release}",
        flush=True,
    )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else default_output(model_path, args.video)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(model_path),
        "controller": {
            "press_threshold": hysteresis.press_threshold,
            "release_threshold": hysteresis.release_threshold,
            "startup_state": "RELEASE",
        },
        "temporal": {
            "frame_offsets_ms": model.spec.frame_offsets_ms,
            "prediction_horizon_ms": model.spec.prediction_horizon_ms,
            "target_fps": target_fps,
            "selection": (
                "latest_decoded_frame_at_or_before_each_requested_timestamp"
            ),
        },
        "replay": result.to_dict(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Replay: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
