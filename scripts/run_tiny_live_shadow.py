#!/usr/bin/env python3
"""Read-only WSL ADB screenrecord -> frozen Tiny RGB + RGB/HSV CPU shadow.

This module does NOT import AdbExecutor or accept an Armed flag. No touch,
input motionevent, training, checkpoint writes, or Test-set evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from experiment_tiny_temporal_policy import prepare_video_feature  # noqa: E402
from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from karting_agent.data_flow.adb import AdbClient, AdbConfig  # noqa: E402
from karting_agent.data_flow.input.adb_video import AdbVideoInput, AdbVideoConfig  # noqa: E402
from karting_agent.data_flow.input.video import VideoInput  # noqa: E402
from karting_agent.model.tiny_temporal_policy import TinyTemporalPolicy  # noqa: E402
from karting_agent.runtime.tiny_live_shadow import TinyLiveShadow, LiveShadowConfig  # noqa: E402
from karting_agent.train.heldout_guard import require_frozen_checkpoint  # noqa: E402
from karting_agent.train.trainer import load_video_split  # noqa: E402
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


def checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_frozen_models(checkpoint_dir: Path, config_path: Path):
    split = load_video_split(config_path)
    models, provenance = {}, {}
    for mode, channels in (("rgb", 3), ("rgb_hsv", 4)):
        weights_path = checkpoint_dir / f"tiny_policy_{mode}.pt"
        report_path = checkpoint_dir / f"tiny_policy_{mode}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require_frozen_checkpoint(report, mode=mode, split=split)
        model = TinyTemporalPolicy(image_channels=channels)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        if any(not torch.isfinite(param).all() for param in model.parameters()):
            raise ValueError(f"nonfinite frozen weights: {mode}")
        models[mode] = model
        provenance[mode] = {
            "weights_sha256": checksum(weights_path),
            "metadata_sha256": checksum(report_path),
            "source_training_device": report.get("training_device"),
            "inference_device": "cpu",
        }
    return models, provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=ROOT/"configs/train_v4c2_temporal_v2.yaml")
    ap.add_argument("--road-config", type=Path,
                    default=ROOT/"configs/geometry_pseudo_labels.yaml")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--serial", type=str, default=None)
    ap.add_argument("--video", type=Path, default=None,
                    help="Offline MP4 smoke; omit for live read-only ADB")
    args = ap.parse_args()
    if not 0 < args.duration <= 3600 or args.cpu_threads < 1:
        ap.error("duration must be 0..3600 sec, threads >= 1")
    if not args.output_dir.is_dir() or not args.checkpoint_dir.is_dir():
        ap.error("output-dir and checkpoint-dir must already exist")
    torch.set_num_threads(args.cpu_threads)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (not raw["preprocess"]["mask_touch_area"] or
            tuple(raw["dataset"]["prediction_horizons_ms"]) != (100, 200, 300) or
            int(raw["model"]["frame_stack"]) != 5):
        raise ValueError("frozen preprocessing/timing config mismatch")
    preprocess = preprocess_config_from_mapping(raw)
    road_config, _, _ = load_road_config(args.road_config)
    models, provenance = load_frozen_models(args.checkpoint_dir, args.config)
    for mode, model in models.items():
        channels = 3 if mode == "rgb" else 4
        with torch.inference_mode():
            for _ in range(5):
                model(torch.zeros(1, 5, channels, 96, 96), torch.zeros(1, 5, 3))

    def feature_builder(frame):
        # Exactly the same function as the frozen training/Validation dataset.
        return prepare_video_feature(frame, preprocess=preprocess,
                                     road_config=road_config)

    source = None
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"tiny_live_shadow_{run_stamp}"
    log_path = args.output_dir / f"{name}.jsonl"
    summary_path = args.output_dir / f"{name}_summary.json"
    if log_path.exists() or summary_path.exists():
        ap.error("refusing to overwrite an existing shadow run")
    started = time.perf_counter()
    errors, rows, shadow = None, 0, None
    try:
        if args.video is None:
            adb = AdbClient(AdbConfig(serial=args.serial))
            adb.require_device()
            source = AdbVideoInput(
                adb, screen_size=adb.screen_size(),
                config=AdbVideoConfig(decode_width=360))
            clock = source.monotonic_now_ms
            mode = "adb_screenrecord_read_only"
        else:
            source = VideoInput(args.video)
            clock = None
            mode = "file_replay_smoke_not_live_latency"
        shadow = TinyLiveShadow(models, feature_builder, clock_ms=clock,
                                config=LiveShadowConfig())
        print(f"Shadow {mode}; CPU threads={args.cpu_threads}; "
              "no ADB touch, no Armed; Ctrl+C to stop", flush=True)
        with log_path.open("x", encoding="utf-8") as stream:
            while time.perf_counter() - started < args.duration:
                frame = source.read()
                if frame is None:
                    break
                record = shadow.ingest(frame)
                if record is not None:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows += 1
                    if record["kind"] == "prediction" and shadow.decisions % 30 == 0:
                        values = record["models"]
                        print(f"t={record['observation_ms']:.0f}ms "
                              f"rgb={values['rgb']['probability']:.3f} "
                              f"hsv={values['rgb_hsv']['probability']:.3f} "
                              f"steps={shadow.decisions} "
                              f"decode_to_read={record['decoded_to_read_ms']:.1f}ms",
                              flush=True)
    except KeyboardInterrupt:
        errors = "keyboard_interrupt"
    except Exception as exc:
        errors = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if source is not None:
            source.close()
        if shadow is not None:
            output = {
                "kind": "tiny_live_shadow_read_only",
                "source": mode,
                "duration_wall_seconds": time.perf_counter() - started,
                "output_log": str(log_path),
                "log_rows": rows,
                "status": errors,
                "models": provenance,
                "threshold": 0.5,
                "forecast_horizon_ms": 100.0,
                "virtual_not_human_control": True,
                "no_touch_or_armed": True,
                "capture_timestamps": "host decoded BGR time, not Android capture/encode time",
                "summary": shadow.summary(),
            }
            if isinstance(source, AdbVideoInput):
                output["capture"] = {
                    "decoded_frames": source.decoded_frames,
                    "dropped_frames": source.dropped_frames,
                    "startup_ms": source.startup_ms,
                    "resolution": [source.decode_width, source.decode_height],
                    "decoded_intervals_ms_p95": float(np.percentile(
                        source.decode_intervals_ms, 95))
                    if source.decode_intervals_ms else None,
                }
            with summary_path.open("x", encoding="utf-8") as stream:
                json.dump(output, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(f"Shadow records: {log_path}\nSummary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
