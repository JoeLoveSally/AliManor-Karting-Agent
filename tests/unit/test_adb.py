import io
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from karting_agent.data_flow.adb import AdbClient, AdbConfig, AdbError, parse_screen_size
from karting_agent.data_flow.execute.adb import AdbExecutor
from karting_agent.data_flow.input.adb import AdbInput
from karting_agent.data_flow.input.adb_video import AdbVideoConfig, AdbVideoInput, _read_exact
from karting_agent.data_flow.input.common import Frame


def test_parse_screen_size_prefers_override() -> None:
    assert parse_screen_size("Physical size: 1440x3200\nOverride size: 1080x2400\n") == (
        1080,
        2400,
    )


def test_adb_client_adds_serial_and_reports_failure() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=b"device\n", stderr=b"")

    client = AdbClient(AdbConfig(serial="SERIAL123"), runner=runner)
    assert client.run("get-state") == b"device\n"
    assert calls[0][0] == ("adb", "-s", "SERIAL123", "get-state")

    def failing_runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"offline")

    failing = AdbClient(runner=failing_runner)
    with pytest.raises(AdbError, match="offline"):
        failing.run("get-state")


def test_adb_input_decodes_png_into_bgr_frame() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[..., 0] = 17
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    class FakeClient:
        def run(self, *args: str) -> bytes:
            assert args == ("exec-out", "screencap", "-p")
            return encoded.tobytes()

    adb_input = AdbInput(FakeClient())  # type: ignore[arg-type]
    frame = adb_input.read()

    assert frame.frame_index == 0
    assert frame.timestamp_ms >= 0
    assert frame.image.shape == (10, 20, 3)
    assert np.all(frame.image[..., 0] == 17)
    assert len(adb_input.capture_latencies_ms) == 1


def test_adb_executor_uses_one_persistent_shell_and_sends_state_changes_only() -> None:
    class FakeStdin:
        def __init__(self) -> None:
            self.writes: list[bytes] = []

        def write(self, data: bytes) -> int:
            self.writes.append(data)
            return len(data)

        def flush(self) -> None:
            return None

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = FakeStdin()
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    calls = []
    process = FakeProcess()

    def popen_factory(command, **kwargs):
        calls.append((command, kwargs))
        return process

    client = AdbClient(AdbConfig(serial="SERIAL123"))
    executor = AdbExecutor(
        client,
        x=100,
        y=200,
        popen_factory=popen_factory,  # type: ignore[arg-type]
    )

    executor.start()
    executor.set_pressed(True)
    executor.set_pressed(True)
    executor.set_pressed(False)
    executor.close()

    assert len(calls) == 1
    assert calls[0][0] == ("adb", "-s", "SERIAL123", "shell")
    assert process.stdin.writes == [
        b"input motionevent DOWN 100 200\n",
        b"input motionevent UP 100 200\n",
        b"exit\n",
    ]
    assert executor.pressed is False
    assert len(executor.execute_latencies_ms) == 2


def test_adb_video_input_builds_scaled_screenrecord_command() -> None:
    client = AdbClient(AdbConfig(serial="SERIAL123"))
    video = AdbVideoInput(
        client,
        screen_size=(1440, 3200),
        config=AdbVideoConfig(decode_width=720, bit_rate=4_000_000),
    )

    assert video.decode_width == 720
    assert video.decode_height == 1600
    assert video._recorder_command() == (
        "adb",
        "-s",
        "SERIAL123",
        "exec-out",
        "screenrecord",
        "--output-format=h264",
        "--size=720x1600",
        "--bit-rate=4000000",
        "--time-limit=0",
        "-",
    )


def test_adb_video_input_records_same_stream_to_fragmented_mp4(tmp_path: Path) -> None:
    record_path = tmp_path / "run.mp4"
    video = AdbVideoInput(
        AdbClient(),
        screen_size=(1440, 3200),
        record_path=record_path,
    )

    command = video._ffmpeg_command()
    assert command[:11] == (
        "ffmpeg",
        "-loglevel",
        "error",
        "-use_wallclock_as_timestamps",
        "1",
        "-f",
        "h264",
        "-flags",
        "low_delay",
        "-i",
        "pipe:0",
    )
    assert command[11:20] == (
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-movflags",
        "+frag_keyframe+empty_moov+default_base_moof",
        "-y",
        str(record_path.resolve()),
    )
    assert command[-8:] == (
        "-map",
        "0:v:0",
        "-an",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    )


def test_adb_video_input_rejects_non_mp4_recording_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must use .mp4"):
        AdbVideoInput(
            AdbClient(),
            screen_size=(20, 40),
            record_path=tmp_path / "run.mkv",
        )


def test_adb_video_input_replaces_stale_frame() -> None:
    client = AdbClient()
    video = AdbVideoInput(client, screen_size=(20, 40))
    image = np.zeros((4, 2, 3), dtype=np.uint8)
    first = Frame(image=image, frame_index=0, timestamp_ms=0.0)
    second = Frame(image=image, frame_index=1, timestamp_ms=10.0)

    video._offer_latest(first)
    video._offer_latest(second)

    assert video.dropped_frames == 1
    assert video._frames.get_nowait().frame_index == 1


def test_read_exact_collects_partial_chunks() -> None:
    class PartialReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(min(size, 2))

    assert _read_exact(PartialReader(b"abcdef"), 6) == b"abcdef"
