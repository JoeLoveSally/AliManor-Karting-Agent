"""CLI smoke test: dry run must never construct the real Android touch executor."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from karting_agent.data_flow.input.common import Frame


def test_cli_dry_run_never_uses_adb_touch(monkeypatch, tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/run_adb_closed_loop_v4c4.py"
    spec = importlib.util.spec_from_file_location("v4c4_dry_run_module", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script.parent))

    raw = module.legacy.load_mapping(module.DEFAULT_TRAIN_CONFIG)
    trained = module.legacy.nested_mapping(raw, "model")

    class FakeRunner:
        def __init__(self, model_path, **kwargs):
            self.model_path = Path(model_path)
            self.metadata_path = self.model_path.with_name("metadata.json")
            self.device = "cpu"
            self.metadata = {
                key: trained[key]
                for key in (
                    "architecture", "frame_stack", "input_representation",
                    "visual_feature_dim", "hidden_dim",
                )
            }
            self.metadata.update(
                event_time_classes=7, no_event_class=6, event_time_bin_ms=50.0
            )

        def warmup(self):
            return (0.1,)

        def predict_action(self, normalized_rgb_stack):
            return 0.1

    class FakeAdbClient:
        def __init__(self, config):
            self.config = config

        def require_device(self):
            return None

        def screen_size(self):
            return (720, 1600)

    class FakeVideo:
        def __init__(self, *args, **kwargs):
            self.config = type("Config", (), {"warmup_seconds": 0.0})()
            self.decoded_frames = 0
            self.dropped_frames = 0
            self.decode_intervals_ms = []
            self.decoded_frame_timestamps_ms = []
            self.recording_frame_mapping_valid = False
            self.closed = False

        def read(self):
            time.sleep(0.001)
            index = self.decoded_frames
            self.decoded_frames += 1
            timestamp = index * 33.3334
            self.decoded_frame_timestamps_ms.append(timestamp)
            self.decode_intervals_ms.append(33.3334)
            return Frame(
                image=np.zeros((32, 32, 3), dtype=np.uint8),
                frame_index=index,
                timestamp_ms=timestamp,
            )

        def close(self):
            self.closed = True

    def forbidden_touch_executor(*args, **kwargs):
        raise AssertionError("real ADB touch executor created during dry run")

    monkeypatch.setattr(module, "EventTimeActionRunner", FakeRunner)
    monkeypatch.setattr(module, "AdbClient", FakeAdbClient)
    monkeypatch.setattr(module, "AdbVideoInput", FakeVideo)
    monkeypatch.setattr(module, "AdbExecutor", forbidden_touch_executor)
    output = tmp_path / "dry_run.json"
    result = module.main(["--max-seconds", "0.01", "--output", str(output)])
    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "dry_run"
    assert payload["stop_reason"] == "duration"
    assert payload["transitions"] == []
    assert payload["host_enqueue_latency_ms"] is None
