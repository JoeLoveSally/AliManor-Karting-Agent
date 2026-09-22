# Diagnose 0/171 decodable browser-stamp frames

The initial browser frame-age probe received 225 HTTP requests and 171 decoded video frames, but no valid 24-bar stamp. Those are **zero valid measurement samples**, not zero latency. The next step is checking rendered pixels, not another Armed game attempt or another full 8-second latency test.

## Run

Leave the phone unlocked, with the browser's **previously opened black/white bar page** visible in the foreground. The old HTTP server may already be stopped: the last rendered bars can remain visible without new requests. If the bars are no longer visible, reopen the browser's existing tab; navigating afresh to `127.0.0.1:18765` will not work while its server is stopped. This diagnostic does not start a server.

```bash
python -m pytest tests/unit/test_adb_stamp_pixels.py -q

python scripts/diagnose_adb_stamp_pixels.py \
  --decode-width 360 \
  --output artifacts/adb_runs/video_stamp_pixels_360.json
```

It reads the first frame and up to three optional subsequent frames using the same `AdbVideoInput` path as V4-C4. A completely static page may emit only one frame; that is sufficient for this pixel-layout check. It also performs one `adb exec-out screencap -p` **in memory** for comparison. It saves **only 24 numeric brightness samples at seven relative row positions**, bit strings, dimensions and decoder outcomes. No screenshots or videos are saved, and no touches or model inference are performed.

Compare `android_screencap_sample` with `video_samples`. If screencap has matching `sync_bits=11001010` at some row but video has none, inspect video orientation, capture crop, scaling or color transform. If both show sync at rows other than 0.55, adjust the stamp decoder sampling location. If neither shows sync, check that the browser bars are actually visible in the foreground. `brightness_24` can distinguish neutral-colored non-probe screens from bar spacing or browser viewport mismatches. None of these results measures video latency; an unchanged stamp is sufficient for this **pixel-only** check.
