#!/usr/bin/env python3
"""Estimate host-stamp-to-video-observation age without game controls.

A local WSL HTTP server returns a 16-bit sequence number stamped with the
host's perf_counter. An unlocked Android browser (connected via `adb reverse`)
draws the sequence as large black/white bars. We decode it from the exact
AdbVideoInput capture pipeline used by the karting runtime. Since the producer
and observer share a host clock, no phone/host clock sync or OCR is required.

Measured age INCLUDES USB HTTP response, browser scheduling/rendering,
Android screen capture, USB video transport, FFmpeg, and host dequeue. It is
NOT an isolated screenrecord latency or Android touch latency measurement.
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
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_adb_closed_loop as legacy  # noqa: E402
from karting_agent.data_flow.adb import AdbClient  # noqa: E402
from karting_agent.data_flow.input.adb_video import AdbVideoInput  # noqa: E402

SYNC = 0xCA
BITS = 24

# 8 sync bits followed by a 16-bit sequence counter. Each bit occupies a full
# vertical bar, wide enough to survive a 360px H.264 capture.
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
    """Decode a binary stamp from the center row of the fullscreen browser canvas."""
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
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/adb_runs/video_frame_age_360.json')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seconds <= 0 or not 1024 <= args.port <= 65535:
        raise ValueError('--seconds must be positive and --port must be in [1024, 65535]')
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
        'measurement': 'WSL HTTP timestamp generation -> WSL first ADB-video observation of the rendered stamp',
        'limitations': 'Includes HTTP over adb reverse, browser scheduling/rendering, screenrecord, video transport, FFmpeg and host dequeue; NOT pure camera, touch or game-frame latency.',
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
        screen = client.screen_size()
        video = AdbVideoInput(client, screen_size=screen,
                              config=legacy.resolve_adb_video_config(raw, args))
        observed: set[int] = set()
        start = time.perf_counter()
        deadline = start + args.seconds
        decoded = 0
        invalid = 0
        duplicate = 0
        while time.perf_counter() < deadline:
            frame = video.read(timeout_seconds=min(3.0, max(0.1, deadline-time.perf_counter()+0.1)))
            read_at_ms = time.perf_counter() * 1000.0
            decoded += 1
            sequence = decode_stamp(frame.image)
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
            observed.add(sequence)
            payload['samples'].append({
                'sequence': sequence,
                'source_frame_index': frame.frame_index,
                'stamp_generated_monotonic_ms': produced_at_ms,
                'first_video_observed_monotonic_ms': read_at_ms,
                'stamp_to_video_observed_ms': read_at_ms-produced_at_ms,
            })
        payload['decoded_frames_read'] = decoded
        payload['invalid_or_nonprobe_frames'] = invalid
        payload['duplicate_stamp_frames'] = duplicate
        ages = [float(s['stamp_to_video_observed_ms']) for s in payload['samples']]
        if len(ages) < 15:
            raise RuntimeError(f'only {len(ages)} valid unique stamps; keep the phone browser foreground with all 24 bars visible and retry')
        payload['summary'] = summarize_ages(ages)
        payload['status'] = 'completed'
        status = 0
    except (Exception, KeyboardInterrupt, EOFError) as exc:
        payload['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
        payload['error'] = str(exc) or type(exc).__name__
        print(f"Probe stopped: {payload['error']}", file=sys.stderr, flush=True)
    finally:
        if video is not None:
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
