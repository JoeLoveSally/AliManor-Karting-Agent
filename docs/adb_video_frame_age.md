# Browser-stamped ADB video freshness diagnostic

The Settings/HOME probe measured command-to-screen-change response, including Android page startup/animation, not pure screenrecord latency. This experiment instead measures the age of a timestamp generated on the WSL host when it first appears in the same ADB video capture pipeline used by the karting controller. It does not start the game, issue game touches or load a model.

## Run on WSL

Exit the game, unlock the phone and keep its browser in the foreground. The script creates an HTTP server bound to WSL `127.0.0.1` and `adb reverse tcp:18765 tcp:18765`; no external Internet access is required. Open the printed `http://127.0.0.1:18765/` URL **in the phone browser**, confirm that its black-and-white vertical bars fill the page and press Enter at the WSL terminal.

```bash
git pull --ff-only origin feat/v4c4-adb-runtime
python -m pytest tests/unit/test_adb_video_frame_age.py tests/unit/test_adb_video_frame_age_startup.py -q
python scripts/diagnose_adb_video_frame_age.py \
  --decode-width 360 --seconds 8 \
  --output artifacts/adb_runs/video_frame_age_360.json
```

**Startup fix:** the first video frame is now allowed the `startup_timeout_seconds` in `configs/hardware.yaml` (15 seconds by default), rather than an accidental 3-second timeout. The 8-second sampling window starts **after** the first video frame is received; video startup does not consume measurement time. Occasional gaps between subsequent frames are counted and handled until the sampling deadline.

**Diagnostic stages:** before video starts, the terminal prints `Browser active: received N stamp requests`. If the browser makes no requests, check that the URL is open in the PHONE browser rather than WSL, the phone is unlocked, its browser can execute JavaScript and the `adb reverse` mapping is active. This failure is separate from video startup. After the first frame, it prints `First decoded frame: ...`. If startup times out, the JSON and terminal include HTTP request count, decoded-frame count, video-reader error and screenrecord/FFmpeg process return codes. A running process with no decoded frames is different from a terminated process; inspect those diagnostics before choosing a fix. If frames arrive but stamps cannot be decoded, keep the browser foreground, do not zoom/scroll and make the bars fill the view.

The producer logs WSL `time.perf_counter()` at `/stamp`. The Android browser polls at approximately 50 ms intervals and paints 24 black/white bars (8 sync bits and 16 sequence bits). The probe decodes the bars in WSL video frames, using the same host clock for stamp generation and first observation. It stores numeric observations only, not screenshots. `summary.median_ms` / `p95_ms` include `adb reverse` HTTP, browser scheduling/rendering, screenrecord encoding, USB video transfer, FFmpeg decode and host dequeue; these figures **are not isolated screenrecord latency or guaranteed game capture-to-control latency**.

`status=error` or fewer than 15 decoded unique stamps does not mean zero delay: consult `browser_stamp_requests`, `video_health_before_close`, `decoded_frames_read`, `invalid_or_nonprobe_frames` and `frame_read_timeouts`. If the 360px run succeeds and you need to compare resolution sensitivity, repeat at `--decode-width 720 --output artifacts/adb_runs/video_frame_age_720.json`. Do not retry Armed driving solely on the strength of this probe.

The script removes its own `adb reverse` mapping on exit. Do not use port 18765 if you have an existing reverse mapping you wish to preserve; choose another unused `--port`.
