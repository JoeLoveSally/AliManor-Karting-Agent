"""Passive Android touchscreen event capture for live Shadow diagnostics.

This module never injects touch events. Kernel event timestamps and host pipe
arrival timestamps are different clocks; raw events are preserved for later
synchronization instead of pretending they are video capture timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from threading import Event, Thread
import time
from typing import Callable

from karting_agent.data_flow.adb import AdbClient


_EVENT = re.compile(
    # Explicit 'getevent -lt /dev/input/eventN' omits the device prefix.
    # 'getevent -lt' across multiple devices retains it.
    r"^\[\s*(\d+(?:\.\d+)?)\]\s+(?:(/dev/input/event\d+):\s+)?"
    r"(EV_[A-Z]+)\s+([A-Z0-9_]+)\s+([0-9a-fA-F]+|DOWN|UP)\b"
)
_DEVICE = re.compile(r"(?m)^add device \d+: (/dev/input/event\d+)\s*$")


def choose_touch_device(getevent_listing: str) -> str:
    """Accept exactly one multitouch-capable device, never guess between two."""
    blocks = _DEVICE.split(getevent_listing)
    candidates = []
    for index in range(1, len(blocks), 2):
        path, content = blocks[index:index + 2]
        if all(key in content for key in (
            "ABS_MT_POSITION_X", "ABS_MT_POSITION_Y", "ABS_MT_TRACKING_ID"
        )):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            f"expected one multitouch device, found {candidates}; "
            "pass --touch-device /dev/input/eventN from 'adb shell getevent -lp'"
        )
    return candidates[0]


@dataclass(frozen=True)
class KernelEvent:
    device_timestamp_ms: float
    device: str
    kind: str
    code: str
    value: str


def parse_kernel_event(
    line: str, *, default_device: str | None = None,
) -> KernelEvent | None:
    """Decode getevent -lt with or without the explicit event-device prefix.

    Never infer the device for an unprefixed line unless the caller supplied
    the device used to start the single-device getevent process.
    """
    match = _EVENT.match(line.strip())
    if not match:
        return None
    timestamp, reported_device, kind, code, value = match.groups()
    device = reported_device or default_device
    if device is None:
        return None
    return KernelEvent(float(timestamp) * 1000.0, device, kind, code, value)


def _signed_hex(value: str) -> int:
    value_int = int(value, 16)
    return value_int - (1 << 32) if value_int >= (1 << 31) else value_int


class TouchTransitionParser:
    """Convert Protocol B tracking IDs (or BTN_TOUCH fallback) at SYN_REPORT.

    State transitions are based on *kernel input* events, not video frames.
    All multitouch slots count toward whether any finger is pressed.
    """

    def __init__(self) -> None:
        self.slot = 0
        self.active_slots: set[int] = set()
        self.has_tracking = False
        self.btn_pressed = False
        self.pressed = False
        self.transitions = 0

    def ingest(self, event: KernelEvent) -> bool | None:
        if event.kind == "EV_ABS":
            if event.code == "ABS_MT_SLOT":
                self.slot = _signed_hex(event.value)
            elif event.code == "ABS_MT_TRACKING_ID":
                self.has_tracking = True
                if _signed_hex(event.value) < 0:
                    self.active_slots.discard(self.slot)
                else:
                    self.active_slots.add(self.slot)
        elif event.kind == "EV_KEY" and event.code == "BTN_TOUCH":
            self.btn_pressed = event.value.upper() in {"DOWN", "00000001"}
        elif event.kind == "EV_SYN" and event.code == "SYN_REPORT":
            next_state = bool(self.active_slots) if self.has_tracking else self.btn_pressed
            if next_state != self.pressed:
                self.pressed = next_state
                self.transitions += 1
                return self.pressed
        return None


def decoded_frame_rows(timestamps_ms: list[float], origin_perf_counter_ms: float):
    """Frame index to host monotonic timestamps; never synthetic MP4 PTS."""
    previous = -1.0
    for index, stamp in enumerate(timestamps_ms):
        if not previous < stamp:
            raise ValueError("decoded frame timestamps must strictly increase")
        previous = stamp
        yield {
            "frame_index": index,
            "host_decoded_ms": stamp,
            "host_decoded_perf_counter_ms": origin_perf_counter_ms + stamp,
        }


class LiveTouchRecorder:
    """Read-only background ``adb shell getevent -lt`` recorder.

    Raw events and derived transitions are saved separately in one JSONL.
    Host timestamps are at stdout *arrival*, not at physical touch time.
    """

    def __init__(
        self,
        client: AdbClient,
        path: Path,
        *,
        device: str,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        if not re.fullmatch(r"/dev/input/event\d+", device):
            raise ValueError("touch device must be /dev/input/eventN")
        self.client = client
        self.path = Path(path)
        self.device = device
        self._popen_factory = popen_factory
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: Thread | None = None
        self._stopping = Event()
        self.error: str | None = None
        self.events = 0
        self.transitions = 0
        self.parser = TouchTransitionParser()

    def start(self) -> None:
        if self._reader is not None:
            raise RuntimeError("touch recorder already started")
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite: {self.path}")
        self._process = self._popen_factory(
            self.client.command("shell", "getevent", "-lt", self.device),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        self._reader = Thread(target=self._read_loop, name="touch-event-reader", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            with self.path.open("x", encoding="utf-8") as out:
                for line in self._process.stdout:
                    received_ms = time.perf_counter() * 1000.0
                    event = parse_kernel_event(
                        line.decode("utf-8", errors="replace"),
                        default_device=self.device,
                    )
                    if event is None or event.device != self.device:
                        continue
                    out.write(json.dumps({
                        "kind": "kernel_event",
                        "device_timestamp_ms": event.device_timestamp_ms,
                        "host_received_perf_counter_ms": received_ms,
                        "device": event.device,
                        "event_type": event.kind,
                        "code": event.code,
                        "value": event.value,
                    }) + "\n")
                    self.events += 1
                    pressed = self.parser.ingest(event)
                    if pressed is not None:
                        out.write(json.dumps({
                            "kind": "touch_transition",
                            "device_timestamp_ms": event.device_timestamp_ms,
                            "host_received_perf_counter_ms": received_ms,
                            "pressed": pressed,
                        }) + "\n")
                        self.transitions += 1
        except Exception as exc:
            self.error = f"touch recorder failed: {type(exc).__name__}: {exc}"
        finally:
            if (not self._stopping.is_set() and self._process is not None
                    and self._process.poll() is not None):
                stderr = self._process.stderr.read().decode(errors="replace").strip() if self._process.stderr else ""
                self.error = self.error or (
                    f"getevent exited with code {self._process.returncode}: {stderr}"
                )

    def close(self) -> None:
        self._stopping.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        if self._reader is not None:
            self._reader.join(timeout=3)
            if self._reader.is_alive():
                self.error = self.error or "touch event reader did not terminate"

    def summary(self) -> dict:
        return {
            "path": str(self.path), "device": self.device,
            "kernel_events": self.events, "transitions": self.transitions,
            "event_stream_status": (
                "error" if self.error else
                "no_recognized_events" if self.events == 0 else
                "events_without_transitions" if self.transitions == 0 else
                "transitions_recorded"
            ),
            "recording_error": self.error,
            "timing_note": "kernel device clock vs host stdout arrival clock; not yet video-aligned",
        }
