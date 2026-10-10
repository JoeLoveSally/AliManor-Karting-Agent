"""Static safety/regression checks for the WSL-only Tiny live Shadow runner."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_tiny_live_shadow.py"
RUNTIME = ROOT / "src/karting_agent/runtime/tiny_live_shadow.py"
ADB_VIDEO = ROOT / "src/karting_agent/data_flow/input/adb_video.py"


def test_no_armed_or_android_touch_path_is_imported_or_exposed():
    source = RUNNER.read_text(encoding="utf-8")
    runtime = RUNTIME.read_text(encoding="utf-8")
    runner_tree = ast.parse(source)
    runtime_tree = ast.parse(runtime)
    for tree in (runner_tree, runtime_tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("karting_agent.data_flow.execute")
            if isinstance(node, ast.Import):
                assert all(not name.name.startswith("karting_agent.data_flow.execute")
                           for name in node.names)
    parser_flags = [
        constant.value
        for constant in ast.walk(runner_tree)
        if isinstance(constant, ast.Constant) and isinstance(constant.value, str)
        and constant.value.startswith("--")
    ]
    assert "--arm" not in parser_flags
    assert "--x" not in parser_flags
    assert "--y" not in parser_flags
    assert "AdbExecutor(" not in source
    assert "client.run(\"shell\", \"input\"" not in source


def test_existing_h264_decoder_and_frozen_preprocessor_reused():
    source = RUNNER.read_text(encoding="utf-8")
    assert "AdbVideoInput(" in source
    assert "AdbVideoConfig(decode_width=360)" in source
    assert "prepare_video_feature(frame, preprocess=preprocess," in source
    assert "road_config=road_config" in source
    assert "load_frozen_models(args.checkpoint_dir, args.config)" in source
    assert "weights_only=True" in source
    assert "require_frozen_checkpoint(report, mode=mode, split=split)" in source
    assert "device" not in source.split("torch.load(weights_path", 1)[1].split(")",1)[0]


def test_decoder_host_monotonic_clock_and_virtual_results_are_explicit():
    source = ADB_VIDEO.read_text(encoding="utf-8")
    assert "def monotonic_now_ms(self)" in source
    assert "(time.perf_counter() - self._origin) * 1000.0" in source
    runner = RUNNER.read_text(encoding="utf-8")
    assert "clock = source.monotonic_now_ms" in runner
    assert '"virtual_not_human_control": True' in runner
    assert '"no_touch_or_armed": True' in runner
    assert '"capture_timestamps": "host decoded BGR time' in runner


def test_live_metrics_track_both_models_and_full_preprocessing():
    runner = RUNNER.read_text(encoding="utf-8")
    runtime = RUNTIME.read_text(encoding="utf-8")
    for key in (
        "feature_preprocess_ms", "total_dual_model_processing_ms",
        "decoded_to_read_ms", "decoded_to_completion_ms",
    ):
        assert key in runner
        assert key in runtime
    assert "rgb_hsv" in runtime and "rgb" in runtime
