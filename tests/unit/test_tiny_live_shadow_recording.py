"""Read-only live Shadow recording contract; no ADB or hardware required."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts" / "run_tiny_live_shadow.py"
SOURCE = CLI.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def test_recording_is_explicit_and_cannot_arm_touch():
    flags = [
        node.value
        for node in ast.walk(TREE)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value.startswith("--")
    ]
    assert "--record-debug" in flags
    assert "--arm" not in flags
    assert "--x" not in flags and "--y" not in flags
    assert "AdbExecutor(" not in SOURCE
    assert 'ap.error("--record-debug is available only for live ADB video input")' in SOURCE


def test_recording_reuses_same_stream_and_has_no_second_encoder():
    assert "source = AdbVideoInput(" in SOURCE
    assert "record_path=recording_path" in SOURCE
    assert "args.output_dir / f\"{name}.mp4\" if args.record_debug else None" in SOURCE
    assert "if any(path.exists() for path in (log_path, summary_path, recording_path)" in SOURCE
    assert "screenrecord" not in SOURCE.split("source = AdbVideoInput(", 1)[1]
    assert '"format": "fragmented_mp4_h264_stream_copy"' in SOURCE


def test_logs_expose_source_frame_join_and_finalization_status():
    assert '"timeline_join": "JSONL frame_index joins MP4 decoded-frame index"' in SOURCE
    assert '"mp4_time_base": "synthetic_60fps_not_host_wall_clock"' in SOURCE
    assert '"frame_mapping_valid": source.recording_frame_mapping_valid' in SOURCE
    assert '"recording_error": source.recording_error' in SOURCE
    assert SOURCE.index("source.close()") < SOURCE.index('"frame_mapping_valid":')
    assert 'except KeyboardInterrupt:' in SOURCE
    assert 'errors = "keyboard_interrupt"' in SOURCE


def test_existing_decoder_supports_nonduplicating_stream_copy():
    source = (ROOT / "src/karting_agent/data_flow/input/adb_video.py").read_text(
        encoding="utf-8"
    )
    assert '"-c:v",' in source
    assert '"copy",' in source
    assert '"-movflags",' in source
    assert "recording_frame_mapping_valid" in source
