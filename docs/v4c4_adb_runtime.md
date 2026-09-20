# V4-C4 action-only ADB runtime

This entrypoint deploys the **existing** V4-C4 checkpoint in a local WSL session
that owns the Android ADB connection. It does not require Spark at runtime,
retrain the model, or use the event-time head for control.

## Prerequisites

Run from the repository root in WSL. The connected phone must appear as a
`device` in `adb devices -l`. Install project dependencies including PyTorch,
`pytest` and FFmpeg; the exact checkpoint and sibling `metadata.json` must exist
at `artifacts/models/mobilenet_v3_small_v4c4_event_time/`.

The runner takes image masking and temporal frame spacing from
`configs/train_v4c4_event_time.yaml`. The checkpoint's model family,
architecture, representation, dimensions and event-time bins must agree with
that config; this check prevents accidentally running another artifact.

```bash
python -m pytest \
  tests/unit/test_event_time_policy.py \
  tests/unit/test_event_time_actual_execution.py \
  tests/unit/test_v4c4_adb_runtime.py \
  tests/unit/test_v4c4_adb_dry_run.py -q

python scripts/run_adb_closed_loop_v4c4.py \
  --device cpu --max-seconds 8 --record-debug \
  --output artifacts/adb_runs/v4c4_dry_run.json
```

**Dry Run is the default.** Without `--arm`, the program only creates a
`MockExecutor`; it cannot issue Android DOWN/UP control commands. Dry Run still
reads the live Android video stream, so keep the game visible on the phone.
The JSON file includes full observation probabilities, history-frame indices,
planned deadlines, host command timestamps, input/drop-frame counters and FPS.
The optional MP4 file has a synthetic presentation frame rate: use
`decoded_frame_timestamps_ms` and frame-index mappings in JSON for timing,
not MP4 presentation timestamps.

## Armed test (after dry-run review)

Start the phone's karting race, verify the touch coordinates against its actual
screen resolution and use a short test. The following is an **example only**:
replace `<X>` and `<Y>` with the actual control-button coordinates.

```bash
python scripts/run_adb_closed_loop_v4c4.py \
  --device cpu --max-seconds 5 --record-debug \
  --arm --x <X> --y <Y> --wait-for-start \
  --output artifacts/adb_runs/v4c4_armed_01.json
```

`--wait-for-start` warms the stream then waits for Enter. Press Enter when the
in-game countdown has ended. Allow 200 ms of camera history before the first
model decision. Ctrl+C and normal shutdown request RELEASE if currently
pressed, then close the video stream and persistent ADB shell. This is
best-effort cleanup; it cannot compensate for loss of the USB/ADB connection.

The action-only decoder is configured to `action_threshold=0.5`,
`min_state_hold_ms=100` and `event_class=no_event_class`. Pending transitions
have **no independent timer**: they are considered when the next frame arrives
at or after the hold deadline. The artifact records the planned due time,
observation time and host command enqueue/return monotonic timestamps
separately. The runtime starts the next minimum hold at the actual *observation*
that triggered the pending execution, not retroactively at the planned time.
Android's physical touch-application time remains unmeasured.

The default configuration is a diagnostic development POC, not a safety-critical
controller. Preserve the original validation/test splits; do not use live test
observations to tune the validation metric without recording that change.
