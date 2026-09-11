"""Low-latency Android screen video input for development-only closed-loop POC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
import shutil
import subprocess
from threading import Event, Thread
import time
from typing import BinaryIO, Callable

import numpy as np

from karting_agent.data_flow.adb import AdbClient, AdbError
from karting_agent.data_flow.input.common import Frame


PopenFactory = Callable[..., subprocess.Popen[bytes]]


@dataclass(frozen=True)
class AdbVideoConfig:
    ffmpeg_executable: str = "ffmpeg"
    decode_width: int = 360
    bit_rate: int = 8_000_000
    startup_timeout_seconds: float = 15.0
    frame_timeout_seconds: float = 3.0
    warmup_seconds: float = 0.5

    def validate(self) -> None:
        if not self.ffmpeg_executable.strip():
            raise ValueError("FFmpeg executable must not be empty")
        if self.decode_width <= 0:
            raise ValueError("decode_width must be > 0")
        if self.bit_rate <= 0:
            raise ValueError("bit_rate must be > 0")
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be > 0")
        if self.frame_timeout_seconds <= 0:
            raise ValueError("frame_timeout_seconds must be > 0")
        if self.warmup_seconds < 0:
            raise ValueError("warmup_seconds must be >= 0")


class AdbVideoInput:
    """Decode Android ``screenrecord`` H.264 into a latest-frame BGR stream.

    When ``record_path`` is provided, the same H.264 stream is copied into a
    fragmented MP4 while it is decoded for Runtime.  Recording therefore adds
    no second Android encoder and no CPU video re-encode.
    """

    def __init__(
        self,
        client: AdbClient,
        *,
        screen_size: tuple[int, int],
        config: AdbVideoConfig | None = None,
        record_path: Path | None = None,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        screen_width, screen_height = screen_size
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("screen_size dimensions must be > 0")

        self.client = client
        self.config = config or AdbVideoConfig()
        self.config.validate()
        self.decode_width = min(self.config.decode_width, screen_width)
        scaled_height = screen_height * self.decode_width / screen_width
        self.decode_height = max(2, int(round(scaled_height / 2.0) * 2))
        self.record_path = Path(record_path).resolve() if record_path is not None else None
        if self.record_path is not None and self.record_path.suffix.lower() != ".mp4":
            raise ValueError("ADB debug recording path must use .mp4")
        self._popen_factory = popen_factory

        self._recorder: subprocess.Popen[bytes] | None = None
        self._ffmpeg: subprocess.Popen[bytes] | None = None
        self._reader: Thread | None = None
        self._stop = Event()
        self._frames: Queue[Frame] = Queue(maxsize=1)
        self._origin: float | None = None
        self._first_frame_at: float | None = None
        self._last_frame_at: float | None = None
        self._error: str | None = None

        self.decoded_frames = 0
        self.dropped_frames = 0
        self.decode_intervals_ms: list[float] = []
        self.startup_ms = 0.0

    @property
    def started(self) -> bool:
        return self._reader is not None

    def _recorder_command(self) -> tuple[str, ...]:
        return self.client.command(
            "exec-out",
            "screenrecord",
            "--output-format=h264",
            f"--size={self.decode_width}x{self.decode_height}",
            f"--bit-rate={self.config.bit_rate}",
            "--time-limit=0",
            "-",
        )

    def _ffmpeg_command(self) -> tuple[str, ...]:
        command: list[str] = [
            self.config.ffmpeg_executable,
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
        ]
        if self.record_path is not None:
            command.extend(
                [
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "copy",
                    "-movflags",
                    "+frag_keyframe+empty_moov+default_base_moof",
                    "-y",
                    str(self.record_path),
                ]
            )
        command.extend(
            [
                "-map",
                "0:v:0",
                "-an",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
        )
        return tuple(command)

    def _start(self) -> None:
        if self.started:
            return
        if shutil.which(self.config.ffmpeg_executable) is None:
            raise RuntimeError(f"FFmpeg executable not found: {self.config.ffmpeg_executable}")
        if self.record_path is not None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)

        self._origin = time.perf_counter()
        try:
            self._recorder = self._popen_factory(
                self._recorder_command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise AdbError(f"ADB executable not found: {self.client.config.executable}") from exc

        assert self._recorder.stdout is not None
        try:
            self._ffmpeg = self._popen_factory(
                self._ffmpeg_command(),
                stdin=self._recorder.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            self._recorder.terminate()
            raise RuntimeError(
                f"FFmpeg executable not found: {self.config.ffmpeg_executable}"
            ) from exc

        self._recorder.stdout.close()
        self._reader = Thread(
            target=self._read_frames,
            name="adb-video-reader",
            daemon=True,
        )
        self._reader.start()

    def read(self, timeout_seconds: float | None = None) -> Frame:
        first_read = not self.started
        timeout = (
            self.config.startup_timeout_seconds
            if timeout_seconds is None and first_read
            else self.config.frame_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        if timeout <= 0:
            raise ValueError("timeout_seconds must be > 0")

        self._start()
        deadline = time.monotonic() + timeout
        while True:
            if self._error:
                raise RuntimeError(self._error)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = self._process_error()
                raise RuntimeError(detail or "timed out waiting for Android video frame")
            try:
                return self._frames.get(timeout=min(remaining, 0.1))
            except Empty:
                continue

    def _read_frames(self) -> None:
        assert self._ffmpeg is not None and self._ffmpeg.stdout is not None
        assert self._origin is not None

        frame_size = self.decode_width * self.decode_height * 3
        try:
            while not self._stop.is_set():
                content = _read_exact(self._ffmpeg.stdout, frame_size)
                if len(content) != frame_size:
                    if not self._stop.is_set():
                        self._error = self._process_error() or "Android video stream ended"
                    return

                now = time.perf_counter()
                if self._first_frame_at is None:
                    self._first_frame_at = now
                    self.startup_ms = (now - self._origin) * 1000.0
                if self._last_frame_at is not None:
                    self.decode_intervals_ms.append(
                        (now - self._last_frame_at) * 1000.0
                    )
                self._last_frame_at = now

                image = np.frombuffer(content, dtype=np.uint8).reshape(
                    self.decode_height,
                    self.decode_width,
                    3,
                ).copy()
                frame = Frame(
                    image=image,
                    frame_index=self.decoded_frames,
                    timestamp_ms=(now - self._origin) * 1000.0,
                )
                self.decoded_frames += 1
                self._offer_latest(frame)
        except Exception as exc:
            if not self._stop.is_set():
                self._error = f"ADB video frame reader failed: {exc}"

    def _offer_latest(self, frame: Frame) -> None:
        try:
            self._frames.put_nowait(frame)
            return
        except Full:
            pass

        try:
            self._frames.get_nowait()
        except Empty:
            pass
        self.dropped_frames += 1
        self._frames.put_nowait(frame)

    def _process_error(self) -> str | None:
        for label, process in (("screenrecord", self._recorder), ("ffmpeg", self._ffmpeg)):
            if process is None or process.poll() is None or process.stderr is None:
                continue
            detail = process.stderr.read().decode("utf-8", errors="replace").strip()
            return f"{label} exited with code {process.returncode}: {detail}".rstrip()
        return None

    def close(self) -> None:
        self._stop.set()
        for process in (self._recorder, self._ffmpeg):
            if process is not None and process.poll() is None:
                process.terminate()
        for process in (self._recorder, self._ffmpeg):
            if process is None:
                continue
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self._reader is not None:
            self._reader.join(timeout=2)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
