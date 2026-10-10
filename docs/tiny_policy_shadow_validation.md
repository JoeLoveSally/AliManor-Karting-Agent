# Tiny CNN+GRU frozen Validation-only shadow control replay

## Purpose / hard limits

This is a deterministic **fixed-expert-video counterfactual action-history**
experiment, **not closed-loop kart driving** and not a live Shadow Mode
connection. It uses the same original V4-C2 2 Validation videos with frozen
RGB and RGB+HSV Tiny CNN+GRU checkpoints. No training or test-video decoding.

The original video pixels always come from a HUMAN expert's trajectory.
Once a simulated action disagrees with the expert, those pixels do **not**
represent where a model-driven kart would actually be. The resulting
self-fed metrics are diagnostics for decision consistency and timing
under a mismatched action history, NOT estimators of lap completion,
collision risk, recovery behavior or real closed-loop Transition F1.

No changes to Armed, ADB, real touch input, model weights, labels, original
Test report, selection epoch, threshold, or validation split.

## Input and timing semantics

Each example uses 5 masked 96×96 frames at observation timestamps
`[t-200,t-150,t-100,t-50,t]`. The trained network outputs the desired
PRESS probability at `t+100ms`, not at `t`.

For each video, the simulator starts at **t=0** with only the expert's
initial PRESS/RELEASE state. No later expert action transitions are fed
to the simulated-action branch.

At an observation time `t`:

1. Apply queued actions whose simulated execution time is at or before
   `t`. Do not apply pending future actions.
2. Build the complete 5-frame control history from **actually simulated
   executed** action states: `[pressed, last_transition_age / 500ms,
   frame_delta / 200ms]`.
3. Compare two inferences with **identical frame pixels**:
   - teacher-forced: actual expert action history (the training input);
   - self-fed: only the simulated executed-action history.
4. Schedule the self-fed prediction at:
   `execution_ms = max(t + 100, t + simulated_latency_ms)`.
   A proposal never acts retroactively. Consecutive identical target
   states create no fake switch event.
5. Advance the clock and repeat, then flush remaining proposals at end.

The script initializes each video separately and resets the controller.
Its teacher-forced Transition counts and short-release counts **must
exactly reproduce the original saved Validation report**, or it refuses
to save any output. Checkpoint metadata is validated for source split,
training selection and touch masking; model weights are loaded read-only.
The evaluator always uses threshold `0.5` and tolerance `±100ms`.

This model inference runs on CPU and records **measured CPU wall-clock
forward time** separately from the `--simulated-latency-ms` value.
The default simulated latency is 0ms (idealized), NOT a claim of zero
inference cost. A later latency experiment can use measured deployment
latency, but per-frame CPU benchmark time is not a substitute for
live Spark+vLLM or phone control latency. The timeline assumes new
inferences can be evaluated at each recorded observation timestamp;
it does not simulate a backed-up inference server.

## Files

- `src/karting_agent/train/shadow_control.py`: causal future-action
  queue, applied transitions and control-feature history
- `src/karting_agent/train/shadow_replay.py`: paired teacher-forced/self-fed
  inference and per-observation records on fixed expert visual frames
- `scripts/audit_tiny_policy_shadow.py`: Validation-only runner,
  provenance checks, frozen-metric parity and JSON/CSV outputs
- `tests/unit/test_shadow_control.py`, `test_shadow_replay.py`,
  `test_shadow_cli.py`: causal timing, control history, no-future-state,
  delayed execution and output protocol regression tests

## Run

**WSL, in the existing experimental branch:**

```bash
cd /home/jianqiao/workspace/AliManor-Karting-Agent-v4c2-history
git pull --ff-only origin exp/v4c2-action-history-residual

rspark \
  ./src/karting_agent/train/shadow_control.py \
  ./src/karting_agent/train/shadow_replay.py \
  ./scripts/audit_tiny_policy_shadow.py \
  ./tests/unit/test_shadow_control.py \
  ./tests/unit/test_shadow_replay.py \
  ./tests/unit/test_shadow_cli.py
```

**Spark, in the existing project virtual environment:**

```bash
cd /home/grg/Workspace/AliManor-Karting-Agent

python -m pytest -q \
  tests/unit/test_shadow_control.py \
  tests/unit/test_shadow_replay.py \
  tests/unit/test_shadow_cli.py

python scripts/audit_tiny_policy_shadow.py \
  --checkpoint-dir /tmp/tiny_policy_export \
  --output-dir /tmp/tiny_policy_export \
  --simulated-latency-ms 0 \
  --cpu-threads 4
```

Expected files inside existing Spark directory:
- `tiny_policy_shadow_validation.json` — aggregate and per-video
  Transition/short-release comparisons, each simulated action event,
  CPU inference timing, first divergence time, self/expert action
  disagreement, checkpoint SHA-256 and protocol assumptions.
- `tiny_shadow_rgb_video_20260130_173545.csv`
- `tiny_shadow_rgb_video_20260130_174012.csv`
- `tiny_shadow_rgb_hsv_video_20260130_173545.csv`
- `tiny_shadow_rgb_hsv_video_20260130_174012.csv`

No original Test video is decoded. No existing output name is overwritten.
The script verifies all teacher-forced Validation event/short-correction
counts against the frozen checkpoint's training metadata before writing.

**WSL, copy directly into the existing downloads directory:**

```bash
rsync -av \
  --include='tiny_policy_shadow_validation.json' \
  --include='tiny_shadow_*.csv' \
  --exclude='*' \
  grg@10.1.48.26:/tmp/tiny_policy_export/ \
  /home/jianqiao/downloads/
```

## Interpret results

Read three separate, clearly qualified sequences:

1. `teacher_forced`: frozen reference policy, expert control history;
   must match previously saved Validation events.
2. `self_fed_desired`: same frozen policy, its own *past executed*
   control state, with decisions projected onto **target** timestamps.
3. `simulated_executed`: actual applied virtual action states at target
   timestamps, including delayed ready-time execution if requested.

Inspect:
- first simulated/expert disagreement on an observation
- disagreement fraction of observed action states (fixed-video proxy)
- desired vs actually simulated-executed Transition F1
- extra switches and 100–300ms release detection
- 50th/95th percentile of paired-batch CPU forward time and any
  `late_deadlines`

Do NOT compare self-fed scores to Test and select a checkpoint.
No real closed-loop performance claim is supported. The next step, if
the simulator is stable, is a distinct explicitly authorized live
read-only Shadow Mode, not Armed or autonomous control.

## Verification

Before submission, pure CPU local tests: **20 passed**. The two
user-provided frozen RGB/RGB+HSV weight files were independently
SHA-256 checked against their saved Test report and each completed a
CPU batched forward/replay smoke. The six corresponding GitHub source
and test files were verified to have the *same Git blob hashes* as the
locally tested copies. Spark validation video end-to-end has not
been executed here.

## Frozen Validation parity fix (2026-10-10)

The first Spark replay passed its **20 tests** and decoded both Validation
videos, but stopped at `rgb: teacher replay diverged from frozen validation
all.predicted`. **No result CSV or JSON was saved**. The original experiment
evaluated the complete Validation DataLoader in unshuffled batches of **16**,
whereas the initial shadow replay ran expert/self-fed pairs in batches of
**2** per observation. Different inference batching can introduce numerical
differences near the threshold (the original training validation also used
CUDA, while shadow uses CPU). The specific magnitude of the first Spark
mismatch has not yet been established.

The correction:
- `shadow_teacher.py` now recomputes the frozen teacher reference from
  the **entire Validation dataset in original manifest order**, using the
  original batch size read from each model's training report, retaining
  batch boundaries even across the two videos.
- This teacher prediction is mapped to `(video, target_timestamp_ms)`.
  The causal self-fed shadow path performs single-sample inference and
  never substitutes its own control inputs into the reference.
- `compare_frozen_validation` is **still strict** for every PRESS/RELEASE
  event count and short-release recall: it does not change threshold,
  tolerance, predictions or checkpoints. If the original
  CUDA-vs-CPU numerical difference still crosses the threshold, the
  error now prints both expected and actual event counts. **Do not**
  force a metric match by adjusting thresholds.
- Two additional CPU test files cover canonical batch boundaries and
  prove a deliberately batch-sensitive model cannot contaminate the
  teacher reference.

For WSL to sync the fix, after `git pull --ff-only` add these files to
the existing `rspark` transfer:

```bash
rspark \
  ./src/karting_agent/train/shadow_teacher.py \
  ./src/karting_agent/train/shadow_replay.py \
  ./scripts/audit_tiny_policy_shadow.py \
  ./tests/unit/test_shadow_teacher.py \
  ./tests/unit/test_shadow_teacher_replay.py
```

On Spark run the earlier 20 tests **plus** the new teacher tests:

```bash
python -m pytest -q \
  tests/unit/test_shadow_control.py \
  tests/unit/test_shadow_replay.py \
  tests/unit/test_shadow_cli.py \
  tests/unit/test_shadow_teacher.py \
  tests/unit/test_shadow_teacher_replay.py
```

Then rerun the same `python scripts/audit_tiny_policy_shadow.py` command.
The previous failed attempt did not create results, so it should not require
deleting any original artifact. If a new parity exception persists, share
the **expected/got values** before running any other experiment.

