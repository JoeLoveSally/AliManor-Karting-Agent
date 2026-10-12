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

For an **entire manually driven race** rather than the original 20-second
Home-screen performance smoke, use a longer cap and explicit debug MP4
recording. Start capture on the pre-race screen (before the countdown),
then drive normally and press Ctrl+C **once** when the result screen
appears. Wait for the printed `Shadow records` and `Summary` paths,
because FFmpeg needs time to finalize its fragmented MP4.

```bash
python scripts/run_tiny_live_shadow.py \
  --checkpoint-dir /home/jianqiao/downloads \
  --output-dir /home/jianqiao/downloads \
  --duration 180 \
  --cpu-threads 4 \
  --record-debug
```

`--duration 180` is a **safety maximum** of 180 seconds, not a
requirement to keep recording after the finish. If a full race lasts
longer, raise the cap within the supported 3600s limit before starting.
`--record-debug` is accepted only in live ADB mode; MP4 offline
replay has no additional recording. The script cannot know when the
Android race starts or finishes, so it is important to start before
countdown and end on the results screen.

Three files with the exact same UTC timestamp stem are created flat
under `/home/jianqiao/downloads`:

- `tiny_live_shadow_<utc>.mp4`: stream-copy of the already decoded ADB
  H.264 source, including the pre-race countdown and finished screen.
- `tiny_live_shadow_<utc>.jsonl`: model samples at ~30Hz, with
  `frame_index` pointing to the matching input decoder frame.
- `tiny_live_shadow_<utc>_summary.json`: inference/capture statistics,
  recording finalization error and whether the decoder-to-MP4
  frame-index mapping appears valid.

**Sync limitation:** the MP4 uses synthetic 60fps H.264 time stamps
to keep the raw screenrecord stream playable. The JSONL's monotonic
host timestamps are independent. Align by decoded frame *index*,
provided `recording.frame_mapping_valid` is true; do not align by
equating MP4 seconds to JSONL `observation_ms` seconds. Inspect
`recording_error` and playback/ffprobe after capture—internal metadata
alone cannot guarantee MP4 decodes properly. Sending just JSONL/summary
is insufficient for visual decision analysis.


No `--arm`, no click coordinates, no `AdbExecutor`.
Optionally pass `--serial DEVICE_ID` to select the USB device.
Run while the Android screen is showing moving gameplay; scrcpy can be
open for human viewing if the device supports both encoders simultaneously.
If concurrent use causes device-side screenrecord errors, stop scrcpy
temporarily to isolate the capture source, rather than rewriting the model.

Outputs retain their common UTC name and exclusive file creation. Model
probabilities, virtual scheduled events, source frame indices and processing
times are in JSONL; SHA-256, dropped frames and timing P50/P95/MAX are in
the summary. The optional MP4 shares that UTC stem.

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

## Optional read-only kernel touch capture

The Hough touch-circle detector used in historical training is deliberately
unchanged. Static round app icons can cause false detections, and overlay
disappearance can split a long physical hold. The next live test can capture
kernel touch data as a separate, read-only signal.

Run `adb shell getevent -lp` to inspect the available input devices, then
run the existing Shadow command with both `--record-debug` and
`--record-touch-events`. For ambiguous device discovery, add
`--touch-device /dev/input/eventN` using the actual touchscreen entry.
The app must be showing a stable *in-game* screen; disable double-tap-to-lock
temporarily if it interferes.

Additional files use the same timestamp stem:
- `*_frames.jsonl` contains *every* decoded frame's source index and host
  decoder arrival time in both stream-relative and absolute perf_counter ms.
  It includes frames discarded by the one-frame latest-frame queue.
- `*_touch.jsonl` contains raw passive `getevent -lt` records and transitions
  reconstructed from multitouch tracking-ID or BTN_TOUCH input.
  Movement during one hold does not produce repeated PRESS.
- `*_summary.json` records kernel event counts and any recording errors.

**Timing limitation:** Device kernel timestamps and host stdout arrival times
have different clock bases. The arrival time includes ADB transport/buffering,
and decoder availability excludes Android capture/encode latency. Collecting
these timestamps does not yet establish their precise synchronization.
Do not calculate action Transition F1 against video frames until alignment is
calibrated. If `getevent` access is denied or reports zero events while
touching, the event log is not valid ground truth. The system touch-circle
overlay can be retained as an independent calibration signal.

## Offline paired-race assessment (read-only)

After recording a complete race with \`--record-debug --record-touch-events\`,
use \`scripts/evaluate_tiny_live_recording.py\`. The run stem is the common
path prefix of the five paired MP4/JSONL/Summary files.

\`\`\`bash
python -m pytest -q tests/unit/test_evaluate_tiny_live_recording.py

python scripts/evaluate_tiny_live_recording.py \
  --run-stem /home/jianqiao/downloads/tiny_live_shadow_20261012T041903848106Z \
  --checkpoint-dir /home/jianqiao/downloads \
  --output-dir /home/jianqiao/downloads \
  --race-start-ms 5000 --race-end-ms 56800 \
  --batch-size 48 --cpu-threads 4
\`\`\`

Remove \`--checkpoint-dir\` for a fast stored-log diagnosis without re-running
the frozen models. The script never accesses ADB and never writes game control.

Checkpoint replay compares:
1. RGB / RGB+HSV models with expert kernel-touch history vs recorded virtual
   history, each kept causal at every selected frame;
2. the *frozen training* touch mask vs expanded
   \`[0.68, 0.76, 1.0, 1.0]\` mask, on **identical recorded source frames**;
3. PRESS and RELEASE events, state accuracy at observation+100ms, and
   kernel-clock hold duration differences.

Outputs: \`*_offline_eval_summary.json\` and, when replay is enabled,
\`*_offline_eval_predictions.jsonl\`. Files are not overwritten.

**Interpretation safeguards:** input video uses synthetic container FPS; frame
indices and \`*_frames.jsonl\` are authoritative. A frame-count check alone
does not prove decoded frame content matches the original live pixels. The
recorded-virtual/original-mask re-evaluation must reproduce live probabilities:
see \`recorded_live_probability_replay_parity\`, and do not interpret ablations
if its \`passes_0p03_p95\` is false. The FFmpeg code 255 on Ctrl+C is retained as
a warning. Changes to the touch mask are OOD perturbations, not causal proof
of label leakage. Expert state persistence is an oracle-input reference, not
a deployable policy.

The getevent \`host_received_perf_counter_ms\` and decoder availability time
share a host clock but have distinct/unknown upstream delays. The default
reference shift is **uncalibrated**. Inspect \`shift_sensitivity\` and optionally
specify \`--touch-shift-ms\` for what-if analyses. Never interpret signed
transition lag or event F1 as calibrated end-to-end latency, and do not apply
a hard-coded action timing compensation on the basis of this single race.
