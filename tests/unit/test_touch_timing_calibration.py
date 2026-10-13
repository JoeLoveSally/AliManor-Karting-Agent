"""Targeted CPU-only tests for passive manual-touch calibration utilities."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def import_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


record = import_script("record_touch_timing_calibration")
analysis = import_script("analyze_touch_timing_calibration")


def transition(host, kernel, pressed):
    return {"kind": "touch_transition", "host_received_perf_counter_ms": host,
            "device_timestamp_ms": kernel, "pressed": pressed}


def fake_data():
    # 65ms short correction and 400ms isolated hold, neither is filtered.
    return [transition(1000, 200, True), transition(1063, 265, False),
            transition(1200, 400, True), transition(1601, 800, False)]


def test_output_manifest_preserves_existing_files(tmp_path):
    paths = record.output_paths(tmp_path, "touch_calibration_20261013T050000Z")
    assert set(paths) == {"video", "frames", "touch", "reads", "summary"}
    paths["video"].touch()
    with pytest.raises(FileExistsError, match="overwrite"):
        record.output_paths(tmp_path, "touch_calibration_20261013T050000Z")
    with pytest.raises(ValueError, match="touch_calibration"):
        record.output_paths(tmp_path, "tiny_live_shadow_test")


def test_consumer_clock_preserves_decoder_origin_and_delay():
    frame = SimpleNamespace(frame_index=12, timestamp_ms=1530.0)
    row = record.consumption_row(frame, host_read_perf_counter_ms=103560.,
                                 origin_perf_counter_ms=102000.)
    assert row["host_decoded_perf_counter_ms"] == 103530.
    assert row["decode_to_read_ms"] == 30.
    with pytest.raises(ValueError, match="predates"):
        record.consumption_row(frame, host_read_perf_counter_ms=103500.,
                               origin_perf_counter_ms=102000.)


def test_kernel_durations_keep_rapid_corrective_pulses():
    result = analysis.physical_episodes(fake_data())
    assert result["complete_press_release_episodes"] == 2
    assert result["kernel_duration_bins"]["lt_100ms"]["count"] == 1
    assert result["kernel_duration_bins"]["300_600ms"]["count"] == 1
    assert result["kernel_hold_ms"]["median_ms"] == pytest.approx(232.5)
    assert result["kernel_episodes"][0]["kernel_duration_ms"] == 65
    assert result["kernel_release_to_next_press_ms"]["median_ms"] == 135


def test_kernel_transition_order_or_clock_reversal_raises():
    events = fake_data()
    with pytest.raises(ValueError, match="alternate"):
        analysis.physical_episodes([events[0], events[0]])
    misordered = [events[0], {**events[1], "device_timestamp_ms": 199}]
    with pytest.raises(ValueError, match="nonmonotonic"):
        analysis.physical_episodes(misordered)
    with pytest.raises(ValueError, match="no touch_transition"):
        analysis.physical_episodes([{"kind": "kernel_event"}])


def test_frame_index_and_consumer_read_join_and_drops():
    frames = [
        {"frame_index": i, "host_decoded_perf_counter_ms": float(1000 + i * 50)}
        for i in range(4)
    ]
    reads = [
        {"frame_index": i, "host_decoded_perf_counter_ms": 1000 + i * 50,
         "host_read_perf_counter_ms": 1007 + i * 50}
        for i in (0, 2, 3)
    ]
    summary = {"decoded_frames": 4, "recording_frame_mapping_valid": True,
               "recording_error": None}
    result = analysis.validate_frames(frames, reads, summary)
    assert result["decoder_to_consumer_read_ms"]["median_ms"] == 7
    assert result["consumer_skipped_decoded_frames"] == 1
    with pytest.raises(ValueError, match="strictly increasing"):
        analysis.validate_frames(frames, reads + [reads[1]], summary)
    with pytest.raises(ValueError, match="summary"):
        analysis.validate_frames(frames, reads, {**summary, "decoded_frames": 3})


def test_incomplete_end_press_is_counted_not_invented():
    rows = fake_data() + [transition(1800, 1000, True)]
    result = analysis.physical_episodes(rows)
    assert result["complete_press_release_episodes"] == 2
    assert result["incomplete_or_boundary_transitions"] == 1


def test_full_offline_analysis_without_mp4_keeps_kernel_clock_unknown(tmp_path, monkeypatch):
    stem = tmp_path / "touch_calibration_fixture"
    summary = {"kind": "passive_touch_timing_calibration_trace",
               "read_only": True, "decoded_frames": 2,
               "capture_error": None, "recording_error": "ffmpeg code 255",
               "recording_frame_mapping_valid": False,
               "touch": {"transitions": 4}}
    Path(f"{stem}_summary.json").write_text(json.dumps(summary))
    rows = fake_data() + [{"kind": "kernel_event", "device_timestamp_ms": 200,
                           "host_received_perf_counter_ms": 1000}]
    Path(f"{stem}_touch.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    frames = [{"frame_index": i, "host_decoded_perf_counter_ms": 900 + 50 * i}
              for i in range(2)]
    reads = [{"frame_index": i, "host_decoded_perf_counter_ms": 900 + i * 50,
              "host_read_perf_counter_ms": 902 + i * 50} for i in range(2)]
    Path(f"{stem}_frames.jsonl").write_text("".join(json.dumps(x) + "\n" for x in frames))
    Path(f"{stem}_reads.jsonl").write_text("".join(json.dumps(x) + "\n" for x in reads))
    fake = ModuleType("diagnose_release_timing")
    fake.clock_offset_diagnostics = lambda events: {"clock_offset_unknown": True}
    fake.detect_visual = lambda *args: (_ for _ in ()).throw(AssertionError("should not decode"))
    fake.pair_visual_with_kernel = lambda *args: None
    monkeypatch.setitem(sys.modules, "diagnose_release_timing", fake)
    report = analysis.analyze(stem, with_video=False)
    assert report["raw_physical_episodes"]["complete_press_release_episodes"] == 2
    assert report["optical_touch_indicator"] is None
    assert report["kernel_host_clock"]["clock_offset_unknown"]
    assert report["frame_timing"]["debug_mp4_recorder_error"] == "ffmpeg code 255"


def test_record_main_writes_sidecars_without_automated_touch(tmp_path, monkeypatch):
    """Use fake ADB/video/touch to exercise capture setup, cleanup and outputs."""
    adb_mod = ModuleType("karting_agent.data_flow.adb")
    video_mod = ModuleType("karting_agent.data_flow.input.adb_video")
    touch_mod = ModuleType("karting_agent.train.labels.live_touch_events")

    class FakeAdb:
        def __init__(self, config):
            self.config = config

        def require_device(self):
            return None

        def run(self, *args):
            assert args == ("shell", "getevent", "-lp")
            return b"touchscreen"

        def screen_size(self):
            return 1440, 3200

    class FakeVideo:
        def __init__(self, client, *, screen_size, config, record_path):
            self.origin_perf_counter_ms = 1000.
            self.decoded_frame_timestamps_ms = []
            self.decoded_frames = 0
            self.dropped_frames = 0
            self.decode_width = 360
            self.decode_height = 800
            self.recorded_frames = 0
            self.recording_error = None
            self.recording_frame_mapping_valid = True
            self.started = False
            record_path.write_bytes(b"fake MP4")

        def read(self):
            if self.decoded_frames == 2:
                raise KeyboardInterrupt()
            self.started = True
            i = self.decoded_frames
            self.decoded_frames += 1
            self.recorded_frames += 1
            self.decoded_frame_timestamps_ms.append(float(i + 5))
            return SimpleNamespace(frame_index=i, timestamp_ms=float(i + 5))

        def close(self):
            pass

    class FakeTouch:
        def __init__(self, client, path, *, device):
            assert device == "/dev/input/event6"
            self.path = path
            self.transitions = 2

        def start(self):
            self.path.write_text('{"kind":"kernel_event"}\n')

        def close(self):
            pass

        def summary(self):
            return {"transitions": self.transitions}

    adb_mod.AdbClient = FakeAdb
    adb_mod.AdbConfig = lambda serial: serial
    video_mod.AdbVideoInput = FakeVideo
    video_mod.AdbVideoConfig = lambda decode_width: decode_width
    touch_mod.LiveTouchRecorder = FakeTouch
    touch_mod.choose_touch_device = lambda listing: "/dev/input/event6"
    touch_mod.decoded_frame_rows = lambda stamps, origin: [
        {"frame_index": i, "host_decoded_ms": stamp,
         "host_decoded_perf_counter_ms": origin + stamp}
        for i, stamp in enumerate(stamps)
    ]
    monkeypatch.setitem(sys.modules, "karting_agent.data_flow.adb", adb_mod)
    monkeypatch.setitem(sys.modules, "karting_agent.data_flow.input.adb_video", video_mod)
    monkeypatch.setitem(sys.modules, "karting_agent.train.labels.live_touch_events", touch_mod)
    monkeypatch.setattr(sys, "argv", ["record_touch_timing_calibration.py",
                                      "--output-dir", str(tmp_path), "--duration", "0.1"])
    record.main()
    summary_path = next(tmp_path.glob("*_summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["decoded_frames"] == summary["consumed_frames"] == 2
    assert len(list(tmp_path.glob("*_frames.jsonl"))) == 1
    assert len(list(tmp_path.glob("*_touch.jsonl"))) == 1
    assert len(list(tmp_path.glob("*_reads.jsonl"))) == 1
    assert summary["read_only"] is True
