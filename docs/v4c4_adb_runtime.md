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
  tests/unit/test_v4c4_adb_dry_run.py \
  tests/unit/test_v4c4_recorded_input_audit.py -q

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

## Investigate an existing failure without starting another race

Use the previous run JSON and its matching debug MP4, together with the exact
checkpoint used in that run. This diagnostic **does not call ADB or issue touch**.

```bash
python scripts/diagnose_v4c4_recorded_input.py \
  --run-json artifacts/adb_runs/v4c4_armed_01.json \
  --video artifacts/adb_runs/v4c4_armed_01.mp4 \
  --device cpu \
  --output artifacts/adb_runs/v4c4_armed_01_replay_audit.json
```

The diagnostic takes the five *recorded* source-frame indices for each model
step, decodes those frames again, runs the training-compatible preprocessing and
V4-C4 current-action head, and compares the probabilities against those logged
during the live run. It reports the largest probability errors and how often
replayed and logged probabilities disagree on the 0.5 PRESS threshold. The
`--max-acceptable-probability-error` option (default 0.05) controls only the
**audit alarm**; it does not modify the game control threshold. MP4 versus live
FFmpeg pixel conversion can create small numerical differences. A failed audit
needs investigation; a passing audit confirms **recorded-frame replay
consistency only**, not video freshness, training-domain equivalence or correct
closed-loop behavior. An older run JSON does not contain a cryptographic
checkpoint identity: the operator must ensure this is the same model.pt file.

The live video defaults to a 360-pixel decode width (`hardware.yaml`); several
training recordings use a 720-pixel width. Compare the training and live game
appearance, including player skin, course geometry and overlays. A safe way to
check the runtime throughput at a higher capture resolution is another **Dry
Run**, explicitly without `--arm`:

```bash
python scripts/run_adb_closed_loop_v4c4.py \
  --device cpu --decode-width 720 --max-seconds 8 --record-debug \
  --output artifacts/adb_runs/v4c4_dry_run_720.json
```

This changes the camera-resolution experiment, not the trained checkpoint or
control policy. It cannot prove that the model would successfully steer: no
real steering occurs during Dry Run.

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
Android's physical touch-application time remains unmeasured. FPS alone does
not measure screen-capture-to-model frame age or phone touch latency.

The default configuration is a diagnostic development POC, not a safety-critical
controller. Preserve the original validation/test splits; do not use live test
observations to tune the validation metric without recording that change.
