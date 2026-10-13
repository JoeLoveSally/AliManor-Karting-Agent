# Passive manual-touch timing calibration

## Scope

This workflow captures human-operated karting input independently of Tiny model inference, training and Android controller execution. It saves raw kernel events alongside host decoder and consumer-read timestamps. **No new training labels or deployment timing corrections are created.**

Android screenrecord H264 does not expose a measured device capture timestamp through this pipeline. The debug MP4 uses synthetic 60 fps packet timestamps, while frame sidecars record host BGR decoder availability. Kernel getevent timestamps and host perf_counter do not have a calibrated shared origin.

## Collection on WSL (no Spark)

Connect the phone to WSL ADB. Enable Developer Options / Show taps, open the karting game, and keep the scene stable.

    cd ~/workspace/AliManor-Karting-Agent
    git pull --ff-only origin integration/karting-policy
    python -m pytest -q tests/unit/test_touch_timing_calibration.py
    python -m ruff check scripts/record_touch_timing_calibration.py scripts/analyze_touch_timing_calibration.py tests/unit/test_touch_timing_calibration.py

    python scripts/record_touch_timing_calibration.py --output-dir "$HOME/downloads" --duration 45

If more than one touchscreen is discovered, inspect the output of adb shell getevent -lp and pass --touch-device /dev/input/eventN explicitly, rather than guessing.

After the message "Video and touch capture active", **manually** perform approximately 100 ms, 200 ms, 400 ms and 800 ms presses, several times each. Leave at least 500 ms between isolated trials. In a separate part, include a few 100–200 ms RELEASE-to-repress corrections. Durations need not be exact; raw kernel events supply the measured intervals. The recorder stops after the specified duration.

Each touch_calibration_TIMESTAMP stem generates:

- .mp4 — source recording, synthetic 60fps timestamps (not physical capture time)
- _frames.jsonl — all decoded frame indices and host BGR decoder timestamps
- _reads.jsonl — frames read by the consumer and decoder-to-read delay
- _touch.jsonl — unsmoothed getevent raw events and transitions
- _summary.json — recorder health, capture errors, mapping status

## Analysis on WSL

    SUMMARY=$(ls -t "$HOME"/downloads/touch_calibration_*_summary.json | head -1)
    RUN=${SUMMARY%_summary.json}
    python scripts/analyze_touch_timing_calibration.py --run-stem "$RUN" --output-dir "$HOME/downloads"

The analysis produces a _analysis.json report. The default optical path decodes MP4 and requires its decoded frame count to match the sidecar; if it fails, rerun with --no-video to analyze physical kernel durations and timing without optical claims. The output is never overwritten: only remove a failed analysis result, not the raw recording, before rerunning.

## Interpretation limits

- Kernel PRESS/RELEASE duration is determined using Android event-time *differences*, not video-frame labels. Pulses shorter than 100ms are preserved.
- Host event receipt minus kernel event time contains unknown offset and ADB delay.
- Consumer-read minus BGR decoder-availability time measures queue delay, not device capture latency.
- Optical circle onset/offset lags include UI display persistence, rendering, encoding and decoding. The existing video-specific heuristic matches only long isolated holds, deliberately excluding rapid corrections from optical comparison.
- A nonzero FFmpeg exit code makes recording health uncertain. Require frame-index consistency and inspect warnings before trusting optical results.
- A synchronized capture timestamp or independent physical sync signal is still needed for absolute capture-to-actuation calibration. Do not infer it from these reports.

Upload only the _summary.json and _analysis.json for initial review; video and raw event traces may remain local. Never optimize a real control scheduler or retrain physical labels directly from this exploratory calibration.
