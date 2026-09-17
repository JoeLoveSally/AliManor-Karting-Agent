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

import cv2
import numpy as np

from karting_agent.data_flow.adb import AdbClient, AdbError
from karting_agent.data_flow.input.common import Frame


PopenFactory = Callable[..., subprocess.Popen[bytes]]
DEBUG_RECORDING_FPS = 60.0
DEBUG_RECORDING_QUEUE_SIZE = 128


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

    Runtime timestamps are assigned when decoded BGR frames arrive. When
    ``record_path`` is provided, those exact decoded frames are also written to
    a constant-frame-rate debug MP4 on a separate thread. The MP4 is therefore
    a visual frame-index copy; authoritative timing remains in
    ``decoded_frame_timestamps_ms``.
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
        self._closing = Event()
        self._frames: Queue[Frame] = Queue(maxsize=1)
        self._origin: float | None = None
        self._first_frame_at: float | None = None
        self._last_frame_at: float | None = None
        self._error: str | None = None

        self.recording_fps = DEBUG_RECORDING_FPS
        self._record_queue: Queue[Frame | None] | None = (
            Queue(maxsize=DEBUG_RECORDING_QUEUE_SIZE)
            if self.record_path is not None
            else None
        )
        self._record_writer: cv2.VideoWriter | None = None
        self._record_thread: Thread | None = None
        self.recorded_frames = 0
        self.recording_dropped_frames = 0
        self.recording_dropped_frame_indices: list[int] = []
        self.recording_error: str | None = None

        self.decoded_frames = 0
        self.dropped_frames = 0
        self.decode_intervals_ms: list[float] = []
        self.decoded_frame_timestamps_ms: list[float] = []
        self.startup_ms = 0.0

    @property
    def started(self) -> bool:
        return self._reader is not None

    @property
    def recording_frame_mapping_valid(self) -> bool:
        if self.record_path is None:
            return False
        return (
            self.recording_error is None
            and self.recording_dropped_frames == 0
            and self.recorded_frames == len(self.decoded_frame_timestamps_ms)
        )

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
        return (
            self.config.ffmpeg_executable,
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-flags",
            "low_delay",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-an",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        )

    def _start_debug_recorder(self) -> None:
        if self.record_path is None:
            return
        if self._record_writer is not None or self._record_thread is not None:
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.record_path),
            fourcc,
            self.recording_fps,
            (self.decode_width, self.decode_height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"failed to open debug MP4 writer: {self.record_path}")

        self._record_writer = writer
        self._record_thread = Thread(
            target=self._record_frames,
            name="adb-video-debug-recorder",
            daemon=True,
        )
        self._record_thread.start()

    def _start(self) -> None:
        if self.started:
            return
        if shutil.which(self.config.ffmpeg_executable) is None:
            raise RuntimeError(f"FFmpeg executable not found: {self.config.ffmpeg_executable}")
        if self.record_path is not None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self._start_debug_recorder()

        self._origin = time.perf_counter()
        try:
            self._recorder = self._popen_factory(
                self._recorder_command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            self._close_debug_recorder()
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
            self._close_debug_recorder()
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
                    if not self._closing.is_set() and not self._stop.is_set():
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
                self.decoded_frame_timestamps_ms.append(frame.timestamp_ms)
                self._offer_recording(frame)
                self.decoded_frames += 1
                self._offer_latest(frame)
        except Exception as exc:
            if not self._closing.is_set() and not self._stop.is_set():
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

    def _offer_recording(self, frame: Frame) -> None:
        queue = self._record_queue
        if queue is None:
            return
        try:
            queue.put_nowait(frame)
        except Full:
            self.recording_dropped_frames += 1
            self.recording_dropped_frame_indices.append(frame.frame_index)
            if self.recording_error is None:
                self.recording_error = (
                    "debug recording queue overflow; MP4/source-frame mapping is invalid"
                )

    def _record_frames(self) -> None:
        queue = self._record_queue
        writer = self._record_writer
        assert queue is not None and writer is not None
        try:
            while True:
                frame = queue.get()
                if frame is None:
                    return
                writer.write(frame.image)
                self.recorded_frames += 1
        except Exception as exc:
            if self.recording_error is None:
                self.recording_error = f"debug MP4 recorder failed: {exc}"

    def _close_debug_recorder(self) -> None:
        queue = self._record_queue
        thread = self._record_thread
        writer = self._record_writer
        if queue is None or writer is None:
            return

        if thread is not None and thread.is_alive():
            try:
                queue.put(None, timeout=5.0)
            except Full:
                if self.recording_error is None:
                    self.recording_error = "timed out draining debug recording queue"
            thread.join(timeout=10.0)
            if thread.is_alive() and self.recording_error is None:
                self.recording_error = "debug recording thread did not stop"

        writer.release()
        self._record_writer = None
        self._record_thread = None

    def _process_error(self) -> str | None:
        for label, process in (("screenrecord", self._recorder), ("ffmpeg", self._ffmpeg)):
            if process is None or process.poll() is None or process.stderr is None:
                continue
            detail = process.stderr.read().decode("utf-8", errors="replace").strip()
            return f"{label} exited with code {process.returncode}: {detail}".rstrip()
        return None

    @staticmethod
    def _wait_or_kill(process: subprocess.Popen[bytes], timeout: float) -> None:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    def close(self) -> None:
        """Stop screen capture, drain decoded frames, and finalize debug recording."""
        self._closing.set()

        recorder = self._recorder
        ffmpeg = self._ffmpeg

        # Stop the producer first. Once the ADB stdout pipe closes, FFmpeg sees
        # EOF and drains decoded raw frames. The reader must stay alive during
        # this phase or FFmpeg can block on its rawvideo stdout pipe.
        if recorder is not None and recorder.poll() is None:
            recorder.terminate()
            self._wait_or_kill(recorder, timeout=2)

        if ffmpeg is not None and ffmpeg.poll() is None:
            try:
                ffmpeg.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ffmpeg.terminate()
                self._wait_or_kill(ffmpeg, timeout=2)

        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=2)

        self._close_debug_recorder()


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
