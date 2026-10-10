# Frozen Tiny CNN+GRU — offline event timeline audit

**Purpose:** explain the already-completed held-out Test without selecting
another model, epoch, threshold, smoothing filter, or controller setting.

Frozen inputs:
- `tiny_policy_rgb.pt` and `tiny_policy_rgb_hsv.pt`, as actually evaluated
  in `tiny_policy_heldout_test.json`;
- their paired training metadata JSON;
- the original `v4c2_temporal_v2` expert samples, native-FPS action timelines,
  and the two untouched-by-training test videos;
- decision threshold **0.5**, horizon **100ms**, event tolerance **±100ms**.

The replay checks SHA-256 of both weights and training metadata against the
previous held-out report, verifies original train/validation/test assignments,
and **rejects any per-video event counts that disagree with the stored Test**.
It cannot silently change an event match; it uses the same
`sequence_evaluator.evaluate_sequence()`.

The tool uses **CPU** to avoid disturbing vLLM-Omni running on the Spark GPU,
and writes reports/plots only. It never sends ADB/robot controls or modifies
`Armed`, checkpoints, training examples, or the original Test results.

## Interpretation to investigate

Held-out Test contained 2,737 expert samples / 88 GT transitions:

| Test video | RGB Transition F1 | RGB+HSV F1 | RGB short RELEASE | HSV short RELEASE |
| --- | ---: | ---: | ---: | ---: |
| `173426` | 0.9286 | 0.8780 | 7/9 | 4/9 |
| `173835` | 0.8000 | 0.9184 | 8/9 | 9/9 |

- `173426`: examine the 100–200ms short releases that HSV misses
- `173835`: examine the RGB-only extra PRESS/RELEASE transitions
- Treat causal explanations as **hypotheses** until corresponding original
  expert video frames are visually inspected; probability curves alone do
  not reveal why the decision diverged

This tool deliberately does **not** change a model in response to these Test
mistakes. Any future model changes must be based on the development set
and then assessed using **new independent test data**, not reused Test as
an optimization target.

## Sync to Spark

WSL:

```bash
cd /home/jianqiao/workspace/AliManor-Karting-Agent-v4c2-history
git pull --ff-only origin exp/v4c2-action-history-residual

rspark \
  ./src/karting_agent/train/tiny_policy_timeline.py \
  ./scripts/audit_tiny_policy_timeline.py \
  ./tests/unit/test_tiny_policy_timeline.py \
  ./tests/unit/test_tiny_policy_timeline_runner.py
```

Spark:

```bash
cd /home/grg/Workspace/AliManor-Karting-Agent

python -m pytest -q \
  tests/unit/test_tiny_policy_timeline.py \
  tests/unit/test_tiny_policy_timeline_runner.py

python scripts/audit_tiny_policy_timeline.py \
  --checkpoint-dir /tmp/tiny_policy_export \
  --output-dir /tmp/tiny_policy_export \
  --batch-size 16 \
  --cpu-threads 4
```

The script checks for existing names and refuses to overwrite prior results.
The two test videos are decoded once and reused for both models; no training
or validation video is decoded.

## Outputs

All outputs go directly into **`/tmp/tiny_policy_export`**:

- `tiny_policy_timeline_audit.json` — exact matched/missed/spurious events,
  matched signed time errors, full 100–300ms RELEASE detection details,
  prioritized focus times, and frozen checkpoint identifiers
- `tiny_timeline_<video_stem>.png` — full expert timeline (GT step state,
  RGB/HSV probability, 0.5 threshold, red missed GT dots and blue spurious
  prediction crosses)
- `tiny_focus_<video_stem>.jpg` — at most 16 zoomed event windows,
  prioritizing missed short releases, then extra predicted switches
- `tiny_trace_<video_stem>.csv` — each target timestamp, actual
  expert future PRESS target, observed expert action state at time `t`,
  RGB/HSV probability and their binary decisions

In particular, `gt_future_pressed` denotes the **future** state at
the row's target timestamp, while `actual_pressed_at_observation` is the
historically executed expert action from **100ms earlier**. This distinction
is essential when diagnosing switch anticipation vs action persistence.

The PNG/JPG are probability timelines, not a substitute for visual inspection
of the original expert game video. The JSON contains **all** matched/missed
events, while the JPG limits zoomed panels to 16 for readability.

On WSL, transfer only the new read-only artifacts to the existing
`/home/jianqiao/downloads` directory, with no new download subfolder:

```bash
rsync -av \
  --include='tiny_policy_timeline_audit.json' \
  --include='tiny_timeline_*.png' \
  --include='tiny_focus_*.jpg' \
  --include='tiny_trace_*.csv' \
  --exclude='*' \
  grg@10.1.48.26:/tmp/tiny_policy_export/ \
  /home/jianqiao/downloads/
```

## Local verification

Before commit, the isolated CPU environment completed 12/12 tests
covering exact matching projection, short-release flag consistency,
per-model timestamp agreement, plots, immutable model-hash comparisons,
refusal of protocol changes and replay/event count parity. Syntax
compilation succeeded, and the three uploaded checkpoints were verified
against their stored SHA-256 and loaded for a CPU forward pass.

The complete video replay on Spark has **not** been executed in this
environment; user-side Spark execution remains the final integration check.
