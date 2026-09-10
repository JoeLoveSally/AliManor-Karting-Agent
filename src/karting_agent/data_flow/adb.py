"""Shared ADB command client for development-only input/output paths."""

from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
from typing import Callable


class AdbError(RuntimeError):
    """Raised when an ADB command cannot be executed successfully."""


@dataclass(frozen=True)
class AdbConfig:
    executable: str = "adb"
    serial: str | None = None
    command_timeout_seconds: float = 3.0

    def validate(self) -> None:
        if not self.executable.strip():
            raise ValueError("ADB executable must not be empty")
        if self.serial is not None and not self.serial.strip():
            raise ValueError("ADB serial must be non-empty when provided")
        if self.command_timeout_seconds <= 0:
            raise ValueError("ADB command timeout must be > 0")


Runner = Callable[..., subprocess.CompletedProcess]


class AdbClient:
    def __init__(
        self,
        config: AdbConfig | None = None,
        *,
        runner: Runner = subprocess.run,
    ) -> None:
        self.config = config or AdbConfig()
        self.config.validate()
        self._runner = runner

    def command(self, *args: str) -> tuple[str, ...]:
        command: list[str] = [self.config.executable]
        if self.config.serial is not None:
            command.extend(["-s", self.config.serial])
        command.extend(args)
        return tuple(command)

    def run(self, *args: str) -> bytes:
        command = self.command(*args)
        try:
            completed = self._runner(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.config.command_timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise AdbError(f"ADB executable not found: {self.config.executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AdbError(
                f"ADB command timed out after {self.config.command_timeout_seconds:g}s: "
                + " ".join(command)
            ) from exc

        if completed.returncode != 0:
            stderr = bytes(completed.stderr or b"").decode("utf-8", errors="replace").strip()
            message = f"ADB command failed ({completed.returncode}): {' '.join(command)}"
            if stderr:
                message += f"\n{stderr}"
            raise AdbError(message)
        return bytes(completed.stdout or b"")

    def require_device(self) -> None:
        state = self.run("get-state").decode("utf-8", errors="replace").strip()
        if state != "device":
            raise AdbError(f"ADB device is not ready: state={state!r}")

    def screen_size(self) -> tuple[int, int]:
        return parse_screen_size(
            self.run("shell", "wm", "size").decode("utf-8", errors="replace")
        )


def parse_screen_size(output: str) -> tuple[int, int]:
    """Parse effective Android screen size, preferring the last reported size."""
    matches = re.findall(r"(?:Physical|Override) size:\s*(\d+)x(\d+)", output)
    if not matches:
        matches = re.findall(r"\b(\d+)x(\d+)\b", output)
    if not matches:
        raise ValueError(f"unable to parse screen size from: {output!r}")
    width, height = matches[-1]
    return int(width), int(height)
