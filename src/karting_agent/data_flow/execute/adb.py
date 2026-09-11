"""ADB touch executor for development-only closed-loop POC."""

from __future__ import annotations

import subprocess
import time
from typing import Callable

from karting_agent.data_flow.adb import AdbClient, AdbError


PopenFactory = Callable[..., subprocess.Popen[bytes]]


class AdbExecutor:
    """Translate controller state changes into a persistent Android shell.

    A single ``adb shell`` process is kept alive while the POC is armed so
    short PRESS/RELEASE corrections do not pay ADB process startup cost on
    every state transition.  ``execute_latencies_ms`` measures host-side
    command enqueue/flush latency, not physical Android touch latency.
    """

    def __init__(
        self,
        client: AdbClient,
        *,
        x: int,
        y: int,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        if x < 0 or y < 0:
            raise ValueError("ADB touch coordinates must be >= 0")
        self.client = client
        self.x = int(x)
        self.y = int(y)
        self._popen_factory = popen_factory
        self._process: subprocess.Popen[bytes] | None = None
        self.pressed = False
        self.execute_latencies_ms: list[float] = []
        self.last_execute_ms = 0.0

    @property
    def started(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.started:
            return
        try:
            self._process = self._popen_factory(
                self.client.command("shell"),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise AdbError(f"ADB executable not found: {self.client.config.executable}") from exc
        if self._process.stdin is None:
            self._process = None
            raise AdbError("persistent ADB shell has no stdin")

    def set_pressed(self, pressed: bool) -> None:
        pressed = bool(pressed)
        if pressed == self.pressed:
            return

        self.start()
        assert self._process is not None and self._process.stdin is not None
        if self._process.poll() is not None:
            raise AdbError(
                f"persistent ADB shell exited with code {self._process.returncode}"
            )

        event = "DOWN" if pressed else "UP"
        command = f"input motionevent {event} {self.x} {self.y}\n".encode()
        started = time.perf_counter()
        try:
            self._process.stdin.write(command)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise AdbError("failed to write to persistent ADB shell") from exc
        self.last_execute_ms = (time.perf_counter() - started) * 1000.0
        self.execute_latencies_ms.append(self.last_execute_ms)
        self.pressed = pressed

    def close(self) -> None:
        process = self._process
        if process is None:
            return

        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(b"exit\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

        if process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        self._process = None
