# Browser-stamped ADB video freshness diagnostic

The earlier Settings/HOME probe returned hundreds of milliseconds for a **UI command-to-screen-change** response. That result contains Android app startup and animation. It is not a clean measurement of camera/video freshness.

This independent experiment measures how long it takes for a host-originated timestamp to become visible in the *same* `AdbVideoInput` stream used by V4-C4. It does not load the model, play the game, or issue touch events.

## Run (WSL, phone attached to ADB)

Close the game; leave the phone unlocked. The probe creates a local HTTP server bound to WSL `127.0.0.1` and an `adb reverse` tunnel for port 18765. It never needs Internet access. It prints the local URL and waits for you to open it on the **phone's browser**. Keep the page foregrounded and its black/white vertical bars filling the screen; then press Enter in WSL.

```bash
python -m pytest tests/unit/test_adb_video_frame_age.py -q

python scripts/diagnose_adb_video_frame_age.py \
  --decode-width 360 --seconds 8 \
  --output artifacts/adb_runs/video_frame_age_360.json
```

If the page is not reachable, check `adb devices -l` and ensure WSL is using the same ADB server/device as the earlier successful probe. If another service owns port 18765, specify a different free `--port`; the URL printed by the script changes accordingly. The script removes its own `adb reverse` port mapping on exit. **Do not use a port with an existing ADB reverse mapping you want to preserve.**

The host HTTP server assigns a sequence number and records `time.perf_counter()` at `/stamp`. Android's browser polls the local endpoint approximately every 50 ms and renders the sequence using 8 sync bits and 16 data bits in 24 black/white bars. WSL decodes those bits from raw `AdbVideoInput` frames, recording only the **first frame showing each unique sequence**. Both timestamp creation and observation are measured with the same WSL monotonic clock. No OCR, phone clock synchronization, or video timestamp reconstruction is involved. The probe only writes numeric measurements to JSON; it does not save screen content.

`summary.median_ms` and `summary.p95_ms` are **server timestamp generation -> first video-observation ages**. Their components include HTTP transport through ADB reverse, Android browser event-loop/rendering/vsync, screenrecord encoding, ADB video transport, FFmpeg decoding, and host dequeue. They are **not** isolated screenrecord latency, Android touch latency, nor guaranteed in-game capture-to-action latency. They are especially useful for checking whether hundreds of milliseconds remain when Settings/HOME application startup is removed from the measurement. Displaying the diagnostic page itself imposes work that can differ from rendering the game.

`status=error` due to fewer than 15 unique stamps indicates invalid/inadequate detection, **not zero lag**. Keep the mobile browser foreground, do not zoom, scroll or lock the screen during collection. If this probe reports consistently high age, investigate the capture/encode/transport/decode path before drawing conclusions about the model. If it reports low age but the Settings probe was high, the Settings/HOME response was likely dominated by UI command/transition work, though behavior may differ inside the game.

Run a second 720px capture **only if 360px succeeds** and you want to test resolution sensitivity, with `--decode-width 720 --output artifacts/adb_runs/video_frame_age_720.json`. Do not run an Armed game test on the basis of these measurements alone.
