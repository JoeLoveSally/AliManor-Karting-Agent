# Diagnose 0/171 decodable browser-stamp frames

The initial browser frame-age probe received 225 HTTP requests and 171 decoded
video frames, but no valid 24-bar stamp. Those are **zero valid measurement
samples**, not zero latency. The next step is checking the rendered pixels
without another Armed game attempt or a repeated full 8-second age test.

## Run

Leave the phone unlocked, with the browser's **previously opened black/white
bar page** visible in the foreground. The old local HTTP server may already be
stopped: the last successfully rendered bars can remain on the page without
new requests. If the bars are no longer visible, first reopen the browser's
existing tab; a new navigation to `127.0.0.1:18765` will not work while the
previous stamp server is stopped. The diagnostic does not start a server.

```bash
python -m pytest tests/unit/test_adb_stamp_pixels.py -q

python scripts/diagnose_adb_stamp_pixels.py \
  --decode-width 360 \
  --output artifacts/adb_runs/video_stamp_pixels_360.json
```

It reads 10 frames using exactly the same `AdbVideoInput` video path as the
V4-C4 runtime, and performs one `adb exec-out screencap -p` in memory as a
comparison. It writes **only 24 numeric brightness samples at seven relative
row positions**, threshold bit strings, dimensions and decoder outcomes. No
screenshots or videos are saved; no touches or model inference are performed.

Compare `android_screencap_sample` with `video_samples`. If screencap has a
matching `sync_bits=11001010` at some row but video has none, inspect video
orientation, capture crop, scaling and color transform. If both show matching
sync at rows other than 0.55, adjust the stamp decoder's sampling location.
If neither shows sync, check that the black/white browser bars are truly visible
in the foreground; inspect `brightness_24` to distinguish neutral-colored
non-probe screens from incorrect bar spacing or browser viewport margins.
Do not interpret any such diagnosis as a video latency estimate. If the phone
page is stale but still visibly displays the old bars, an unchanged bar sequence
is sufficient for this *pixel-layout-only* check.
