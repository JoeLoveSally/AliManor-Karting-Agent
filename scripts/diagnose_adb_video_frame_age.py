#!/usr/bin/env python3
"""Measure host-stamped browser frames through the game ADB video capture path.

The phone browser renders bit bars carrying host-generated sequence IDs. Both
stamp generation and observation use WSL's perf_counter; Android clock sync is
unnecessary. Age includes HTTP over adb reverse, browser/rendering, screenrecord,
transport, FFmpeg and queue polling. It is NOT isolated screenrecord latency.
Never sends game touches, accesses the model, or stores screenshots.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import statistics
import sys
from threading import Lock, Thread
import time
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_adb_closed_loop as legacy  # noqa: E402
from karting_agent.data_flow.adb import AdbClient  # noqa: E402
from karting_agent.data_flow.input.adb_video import AdbVideoInput  # noqa: E402

SYNC = 0xCA
BITS = 24

HTML = r'''<!doctype html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>ADB video freshness probe</title>
<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#000}
canvas{position:fixed;left:0;top:0;width:100vw;height:100vh;image-rendering:pixelated}</style>
</head><body><canvas id="stamp" width="24" height="1"></canvas><script>
const c=document.getElementById('stamp');const ctx=c.getContext('2d',{alpha:false});
function paint(seq){const bits=(0xCAn<<16n)|BigInt(seq);for(let i=0;i<24;i++){
 ctx.fillStyle=((bits>>BigInt(23-i))&1n)?'#ffffff':'#000000';
 ctx.fillRect(i,0,1,1);
}}
(async function poll(){for(;;){try{const response=await fetch('/stamp?x='+Math.random(),
 {cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);
 const data=await response.json();paint(data.seq);
 }catch(error){console.error(error);}await new Promise(resolve=>setTimeout(resolve,50));}})();
</script></body></html>'''


class StampBook:
    def __init__(self) -> None:
        self.lock = Lock()
        self.sequence = 0
        self.timestamps_ms: dict[int, float] = {}

    def stamp(self) -> int:
        with self.lock:
            self.sequence = (self.sequence + 1) & 0xFFFF
            self.timestamps_ms[self.sequence] = time.perf_counter() * 1000.0
            return self.sequence

    def get(self, sequence: int) -> float | None:
        with self.lock:
            return self.timestamps_ms.get(sequence)

    def request_count(self) -> int:
        with self.lock:
            return len(self.timestamps_ms)


def make_handler(book: StampBook):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == '/':
                body = HTML.encode('utf-8')
                mime = 'text/html; charset=utf-8'
            elif path == '/stamp':
                body = json.dumps({'seq': book.stamp()}).encode('ascii')
                mime = 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def decode_stamp(image: np.ndarray) -> int | None:
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[1] < 240:
        return None
    height, width = image.shape[:2]
    y = min(height - 1, int(height * 0.55))
    value = 0
    for bit in range(BITS):
        x = min(width - 1, int((bit + 0.5) * width / BITS))
        pixel = image[y, x].astype(np.int16)
        if np.max(pixel) - np.min(pixel) > 45:
            return None
        intensity = float(np.mean(pixel))
        if intensity < 65:
            next_bit = 0
        elif intensity > 190:
            next_bit = 1
        else:
            return None
        value = (value << 1) | next_bit
    if (value >> 16) != SYNC:
        return None
    return value & 0xFFFF


def summarize_ages(ages_ms: list[float]) -> dict[str, float | int]:
    if not ages_ms:
        raise ValueError('no unique timestamps observed in ADB video')
    values = np.asarray(ages_ms, dtype=np.float64)
    return {
        'unique_stamps': len(ages_ms),
        'min_ms': float(values.min()),
        'median_ms': float(np.median(values)),
        'p95_ms': float(np.percentile(values, 95)),
        'max_ms': float(values.max()),
        'mean_ms': statistics.mean(ages_ms),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='No-game-touch browser-stamp ADB video age estimate')
    parser.add_argument('--hardware-config', type=Path, default=ROOT / 'configs/hardware.yaml')
    parser.add_argument('--adb', default=None)
    parser.add_argument('--serial', default=None)
    parser.add_argument('--ffmpeg', default=None)
    parser.add_argument('--decode-width', type=int, default=360)
    parser.add_argument('--video-bit-rate', type=int, default=None)
    parser.add_argument('--video-warmup-seconds', type=float, default=None)
    parser.add_argument('--port', type=int, default=18765)
    parser.add_argument('--seconds', type=float, default=8.0)
    parser.add_argument('--browser-ready-timeout', type=float, default=8.0)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/adb_runs/video_frame_age_360.json')
    return parser.parse_args(argv)


def wait_for_browser_stamps(book: StampBook, *, timeout_seconds: float) -> int:
    """Separate browser/adb-reverse failure from screenrecord/FFmpeg failure."""
    if timeout_seconds <= 0:
        raise ValueError('browser-ready-timeout must be > 0')
    deadline = time.perf_counter() + timeout_seconds
    while time.perf_counter() < deadline:
        count = book.request_count()
        if count >= 3:
            return count
        time.sleep(0.05)
    raise RuntimeError(
        f'phone browser requested only {book.request_count()} time stamps; '
        'check that the binary bars are visible on the PHONE browser, '
        'adb reverse is connected, and JavaScript is running'
    )


def video_health(video: AdbVideoInput) -> dict[str, object]:
    """Avoid blocking on child stderr while a process remains alive."""
    details: dict[str, object] = {
        'video_started': video.started,
        'decoded_frames': video.decoded_frames,
        'dropped_frames': video.dropped_frames,
        'reader_error': getattr(video, '_error', None),
    }
    for name in ('_recorder', '_ffmpeg'):
        process = getattr(video, name, None)
        details[name.removeprefix('_') + '_returncode'] = None if process is None else process.poll()
    return details


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seconds <= 0 or not 1024 <= args.port <= 65535:
        raise ValueError('--seconds must be positive and --port must be in [1024, 65535]')
    if args.browser_ready_timeout <= 0:
        raise ValueError('--browser-ready-timeout must be > 0')
    raw = legacy.load_mapping(args.hardware_config)
    client = AdbClient(legacy.resolve_adb_config(raw, args))
    client.require_device()
    book = StampBook()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(book))
    server.daemon_threads = True
    server_thread = Thread(target=server.serve_forever, daemon=True, name='stamp-http-server')
    server_thread.start()
    reverse_created = False
    video = None
    output = args.output.resolve()
    payload: dict[str, object] = {
        'measurement': 'WSL HTTP timestamp generation -> first ADB-video observation of the rendered stamp',
        'limitations': 'Includes adb reverse HTTP, browser scheduling/rendering, screenrecord, USB video, FFmpeg and host dequeue; NOT isolated screenrecord or game/touch latency.',
        'decode_width_requested': args.decode_width,
        'seconds_requested': args.seconds,
        'samples': [],
    }
    status = 2
    try:
        client.run('reverse', f'tcp:{args.port}', f'tcp:{args.port}')
        reverse_created = True
        print(f'Phone (unlocked) browser URL: http://127.0.0.1:{args.port}/', flush=True)
        input('Open this URL on the PHONE, keep its binary bars visible, then press Enter here: ')
        stamp_requests = wait_for_browser_stamps(book, timeout_seconds=args.browser_ready_timeout)
        print(f'Browser active: received {stamp_requests} stamp requests; starting ADB video...', flush=True)
        screen = client.screen_size()
        video = AdbVideoInput(
            client,
            screen_size=screen,
            config=legacy.resolve_adb_video_config(raw, args),
        )
        # Crucial: do not replace the configured 15s video-startup timeout with
        # a 3s regular frame timeout or charge startup against the sample window.
        try:
            first = video.read(timeout_seconds=video.config.startup_timeout_seconds)
        except RuntimeError as exc:
            raise RuntimeError(
                f'ADB video did not deliver its first decoded frame within '
                f'{video.config.startup_timeout_seconds:g}s; '
                f'browser_stamp_requests={book.request_count()}, '
                f'video_health={video_health(video)}; original_error={exc}'
            ) from exc
        print(
            f'First decoded frame: index={first.frame_index}, '
            f'video_startup_ms={video.startup_ms:.1f}',
            flush=True,
        )
        observed: set[int] = set()
        capture_started_ms = time.perf_counter() * 1000.0
        deadline = time.perf_counter() + args.seconds
        decoded = 0
        invalid = 0
        duplicate = 0
        frame_timeouts = 0
        frame = first
        while True:
            if frame is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    frame = video.read(timeout_seconds=min(video.config.frame_timeout_seconds, remaining))
                except RuntimeError as exc:
                    if str(exc) == 'timed out waiting for Android video frame':
                        frame_timeouts += 1
                        frame = None
                        continue
                    raise
            read_at_ms = time.perf_counter() * 1000.0
            decoded += 1
            sequence = decode_stamp(frame.image)
            frame = None
            if sequence is None:
                invalid += 1
                continue
            if sequence in observed:
                duplicate += 1
                continue
            produced_at_ms = book.get(sequence)
            if produced_at_ms is None or produced_at_ms > read_at_ms:
                invalid += 1
                continue
            # A stale warmup frame must not bias the measured steady-state age.
            if produced_at_ms < capture_started_ms:
                continue
            observed.add(sequence)
            payload['samples'].append({
                'sequence': sequence,
                'source_frame_index': decoded,
                'stamp_generated_monotonic_ms': produced_at_ms,
                'first_video_observed_monotonic_ms': read_at_ms,
                'stamp_to_video_observed_ms': read_at_ms - produced_at_ms,
            })
        payload['decoded_frames_read'] = decoded
        payload['invalid_or_nonprobe_frames'] = invalid
        payload['duplicate_stamp_frames'] = duplicate
        payload['frame_read_timeouts'] = frame_timeouts
        ages = [float(s['stamp_to_video_observed_ms']) for s in payload['samples']]
        if len(ages) < 15:
            raise RuntimeError(
                f'only {len(ages)} unique stamps, {decoded} frames read, '
                f'{invalid} undecodable frames, {frame_timeouts} frame gaps; '
                f'browser_stamp_requests={book.request_count()}, '
                f'video_health={video_health(video)}'
            )
        payload['summary'] = summarize_ages(ages)
        payload['status'] = 'completed'
        status = 0
    except (Exception, KeyboardInterrupt, EOFError) as exc:
        payload['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
        payload['error'] = str(exc) or type(exc).__name__
        print(f"Probe stopped: {payload['error']}", file=sys.stderr, flush=True)
    finally:
        payload['browser_stamp_requests'] = book.request_count()
        if video is not None:
            payload['video_health_before_close'] = video_health(video)
            try:
                video.close()
            except Exception as exc:
                payload['video_close_error'] = str(exc)
            payload['decoded_frames'] = video.decoded_frames
            payload['dropped_frames'] = video.dropped_frames
        if reverse_created:
            try:
                client.run('reverse', '--remove', f'tcp:{args.port}')
            except Exception as exc:
                payload['reverse_cleanup_error'] = str(exc)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f"status={payload['status']} output={output}", flush=True)
        if 'summary' in payload:
            print(json.dumps(payload['summary'], indent=2), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
