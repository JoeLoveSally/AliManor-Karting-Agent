# ADB video latency probe (no model, no game touches)

The first-bend failure recorded in `v4c4_armed_01` has **234/234 identical**
recorded-MP4 replay probabilities. This establishes replay consistency but not
screen freshness. The following experiment uses the **same** `AdbVideoInput`
(screenrecord H.264, FFmpeg, latest-frame queue) as the game runtime.

## Preparation and run

1. Disconnect any other program that consumes the phone's screenrecord stream.
2. Unlock the phone, **leave the game**, and open Android **Settings main page**.
   Wait until the page is static. Avoid sensitive pages: nothing is recorded to
   disk, but the tool processes live screen pixels in memory.
3. Run from the WSL repository root with the existing ADB + FFmpeg connection:

```bash
python -m pytest tests/unit/test_adb_video_latency_probe.py -q

python scripts/diagnose_adb_video_latency.py \
  --decode-width 360 --trials 6 \
  --output artifacts/adb_runs/video_latency_360.json
```

The probe alternates Android HOME and reopening Settings, without tapping the
game or invoking the model. Each marker is run from a settled static page. It
compares the central 80% of successive BGR images against the pre-command
baseline and requires at least two consecutive frames with a large change.
After each trial it prints the measured interval; JSON contains each event's
host monotonic command-start and completion times, first confirmed frame,
measured interval, frame counts and summary. It **never records screenshots**.

If it reports `unstable baseline` or `no confirmed ... screen change`, this
probe was **inconclusive**, not an indication of zero latency; ensure Settings
was initially open, the phone is unlocked, and animations have settled. The
script attempts to leave the phone on Settings after an odd number of trials.

## What is and is not measured

`command_start_to_screen_observed_ms` is measured on the **same WSL host clock**:
from starting an ADB command that switches the phone's visible screen to the
first confirmed screen-change frame returned by `AdbVideoInput.read()`.
It includes **ADB dispatch, Android app/home switch and rendering, display
refresh, screenrecord encoding, USB transport, FFmpeg decode, and host frame
polling**. It is an end-to-end upper-bound-style responsiveness probe and
**cannot isolate screenrecord-only frame age**; Settings and the racing game's
rendering paths may also differ. `command_duration_ms` helps distinguish a slow
ADB command from a slow observed visual response. A negative
`command_return_to_screen_observed_ms` is possible if the image changes before
the shell command returns and is not itself an error.

The metric should be compared to the control-loop period (~33 ms) and the
short corrective actions (100–300 ms); hundreds of milliseconds would merit
investigating input delay, but the Android HOME/Settings switch itself may
also be slow. A small result does **not** prove a small racing-frame latency,
and does **not** establish Android motion-event/touch application time.

Only after the 360px probe is working, optionally repeat at a higher live
decode resolution (also without starting a race) to compare runtime responsiveness:

```bash
python scripts/diagnose_adb_video_latency.py \
  --decode-width 720 --trials 6 \
  --output artifacts/adb_runs/video_latency_720.json
```

Do not use these values to infer that the model has learned closed-loop driving:
if video responsiveness is acceptable, the recorded first-bend failure remains
an actual policy/generalization failure to address separately.
