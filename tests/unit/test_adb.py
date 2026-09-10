import subprocess

import cv2
import numpy as np
import pytest

from karting_agent.data_flow.adb import AdbClient, AdbConfig, AdbError, parse_screen_size
from karting_agent.data_flow.execute.adb import AdbExecutor
from karting_agent.data_flow.input.adb import AdbInput


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


def test_adb_executor_sends_state_changes_only() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.commands: list[tuple[str, ...]] = []

        def run(self, *args: str) -> bytes:
            self.commands.append(args)
            return b""

    client = FakeClient()
    executor = AdbExecutor(client, x=100, y=200)  # type: ignore[arg-type]

    executor.set_pressed(True)
    executor.set_pressed(True)
    executor.set_pressed(False)

    assert client.commands == [
        ("shell", "input", "motionevent", "DOWN", "100", "200"),
        ("shell", "input", "motionevent", "UP", "100", "200"),
    ]
    assert executor.pressed is False
    assert len(executor.execute_latencies_ms) == 2
