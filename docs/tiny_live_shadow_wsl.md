# Tiny CNN+GRU live read-only Shadow on WSL2

## Boundary and source-of-truth

This is a **live-video, virtual-action Shadow**, not autonomous driving:
- The **phone** is reachable from WSL2 only. Spark is training-only.
- Reuse the already-validated `AdbVideoInput`: Android screenrecord H.264
  via USB ADB, FFmpeg BGR24, dedicated decoder thread, latest-frame queue
  of size **1**. Scrcpy may remain the independent human viewing tool.
- **Never create an AdbExecutor**, never call `input motionevent`, never
  change Armed/runtime control code or phone state. There is no `--arm` flag.
- The two frozen Tiny checkpoints run on WSL **CPU**, separately, using
  identical five images but **separate virtual executed-action histories**.
  No expert/manual PRESS labels are available from the screen stream.
- Read-only virtual PRESS predictions **do not affect video pixels**.
  No Transition F1 against the human is computed from this run.

The original realtime ADB stream has already been measured around 56 FPS
decoded and ~29.8Hz Runtime throughput at 360×800. That benchmark used
an older model; it is NOT a Tiny CPU benchmark.

## Exact image/temporal semantics

`run_tiny_live_shadow.py` imports the **existing frozen training function**
`experiment_tiny_temporal_policy.prepare_video_feature()`.
It does NOT silently substitute the 224×224 normalized older Runtime path.

For each selected decoded BGR frame:
1. `normalize_game_frame(..., 360×800)` (strict aspect check)
2. Mask touch ROI and two HUD ROIs from the frozen config
3. Resize masked RGB to 96×96 and compute frozen HSV road mask channel
4. Select five causal decoded frames at offsets `[-200,-150,-100,-50,0]ms`
5. Convert cached uint8 to float32/255, identical to `make_mode_images`
6. Feed controls `[virtual_pressed, age_since_last_virtual_transition/500,
   selected_frame_dt/200]`; different virtual history per model
7. Each model predicts the absolute PRESS at `observation+100ms`
8. Virtual scheduled execution time is
   `max(observation+100ms, actual_model_output_ready_ms)`.
   A pending action is not visible to the five-frame history prematurely.

The host timestamp originates from **FFmpeg decoded BGR availability**.
`AdbVideoInput.monotonic_now_ms()` exposes the same host monotonic clock
base, enabling measured *decode-to-read* and *decode-to-completion* lag.
Neither is the true Android capture-to-photon latency; encoder and USB
transport time occur before the host decode timestamp.

The loop follows an approximate 30Hz schedule, drops missed ticks rather
than performing burst catch-up, and skips images older than the 150ms
decode-to-read limit. Missing historical input within 80ms of the desired
frame timestamp is reported instead of silently substituting grossly
stale history. Video preprocessing is done once per selected frame and
shared by both policies. No second Android capture/encoder is spawned.

## WSL sync

```bash
cd /home/jianqiao/workspace/AliManor-Karting-Agent-v4c2-history
git pull --ff-only origin exp/v4c2-action-history-residual

# The following transfer is needed only if WSL is not running directly
# from the local git branch (rspark uploads to Spark; NOT needed for WSL run).
# No rspark for this phase.

# Copy only the FOUR frozen checkpoint files to existing downloads,
# without creating any new output directory.
rsync -av \
  --include='tiny_policy_rgb.pt' \
  --include='tiny_policy_rgb.json' \
  --include='tiny_policy_rgb_hsv.pt' \
  --include='tiny_policy_rgb_hsv.json' \
  --exclude='*' \
  grg@10.1.48.26:/tmp/tiny_policy_export/ \
  /home/jianqiao/downloads/
```

The WSL Python environment needs `torch`, `numpy`, `opencv-python`,
`PyYAML`, and `pytest`; `ffmpeg` and `adb` executables must be
discoverable. Verify `adb devices -l` shows exactly your intended Android
device, and explicitly set `--serial` if multiple devices are connected.

## Test before phone connection

```bash
cd /home/jianqiao/workspace/AliManor-Karting-Agent-v4c2-history
python -m pytest -q \
  tests/unit/test_tiny_live_shadow.py \
  tests/unit/test_shadow_control.py \
  tests/unit/test_shadow_replay.py

python -m compileall -q \
  scripts/run_tiny_live_shadow.py \
  src/karting_agent/runtime/tiny_live_shadow.py

adb devices -l
command -v ffmpeg
```

An **optional offline MP4 smoke** (does not touch ADB) is:

```bash
python scripts/run_tiny_live_shadow.py \
  --checkpoint-dir /home/jianqiao/downloads \
  --output-dir /home/jianqiao/downloads \
  --video data/raw/video_20260130_173545.mp4 \
  --duration 15 \
  --cpu-threads 4
```

That mode uses synthetic MP4 frame-time timestamps. Its timing is NOT live
host latency, and it cannot be used for a real-time performance claim.
The MP4 must be present in the local WSL clone, otherwise skip this step.

## Live-only, read-only run

```bash
python scripts/run_tiny_live_shadow.py \
  --checkpoint-dir /home/jianqiao/downloads \
  --output-dir /home/jianqiao/downloads \
  --duration 20 \
  --cpu-threads 4
```

No `--arm`, no click coordinates, no `AdbExecutor`.
Optionally pass `--serial DEVICE_ID` to select the USB device.
Run while the Android screen is showing moving gameplay; scrcpy can be
open for human viewing if the device supports both encoders simultaneously.
If concurrent use causes device-side screenrecord errors, stop scrcpy
temporarily to isolate the capture source, rather than rewriting the model.

Every execution uses a unique UTC-based flat filename and exclusive
file creation in existing `/home/jianqiao/downloads`:
- `tiny_live_shadow_<utc>.jsonl`: every 30Hz prediction or skipped tick,
  frame timestamps, RGB/HSV probability, virtual scheduled event time,
  and per-stage timing.
- `tiny_live_shadow_<utc>_summary.json`: model SHA-256, frame counts,
  decoder drop count, decode FPS interval P95, and P50/P95/MAX for
  decode-to-read, preprocessing, model forwards, and full CPU cycle.

Interpret `decoded_to_read_ms` as **host-only** stream backlog, not the
full capture-to-model latency. The virtual controls are not human actions;
do not calculate per-video model imitation accuracy from this run.

## Validation and constraints

Local CPU-only regression tests checked five-frame selection, streaming
history causality, identical RGB inputs, independent virtual controllers,
ready-time scheduling, skip-on-stale, missing-history, monotonicity and
no incorrect feature shape. The full **actual WSL USB/ADB integration
and CPU performance remain to be tested by the user**, including
whether both models fit 30Hz when preprocessing HSV on a real game screen.
