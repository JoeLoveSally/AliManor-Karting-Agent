# Tiny RGB fixed-video visual-phase findings (2026-10-13)

## Decision

**Do not promote the Vision-only ablation or optimize deployment timing from
this recording. Do not launch another input-mask training sweep yet.**

The six nonzero future-RGB shifts (+50 to +300 ms) cause near-identical
changes to simply reading the original zero-shift probability curve at a
later target time. This is strong evidence that the oracle result is a
**temporal-translation effect**, not new evidence of forecast competence.

Temporal translation equivariance by itself **cannot distinguish a genuinely
predictive stationary model from a reactive model**. The current fixed-video
results show RELEASE late relative to *host-received kernel touch events*,
but capture-to-decode latency and label semantics are not calibrated.

## Frozen visual-phase experiment

Input:
- `tiny_live_shadow_20261012T041903848106Z_visual_phase_oracle.json`
  and `.jsonl` (1,574 common prediction rows).
- Frozen `tiny_control_vision_only.pt`, same weights, RGB preprocessing,
  five-frame temporal stack, original frame-`dt`, and original H100 target.
- Only decoded RGB frames move forward +0, 50, 100, 150, 200, 250, 300 ms.
  Positive shifts beyond H100 use data from **after** the scored target and
  are deliberately noncausal.

Input integrity:
- 1,574/1,574 predictions included; no missing nearest-frame windows.
- Nonzero-shift frame mapping error p95 is approximately 10–12 ms.
- Zero-shift frozen replay parity: p95 4.47e-8; max 1.79e-7.
- The original MP4 recording reported FFmpeg exit code 255 even though
  sampled decoded frames and frozen replay probabilities were consistent.

Raw report (same ±150 ms event matching used before):

| RGB shift | PRESS matched | RELEASE matched | Median P(PRESS) at human RELEASE+150ms |
|---|---:|---:|---:|
| 0 ms | 23/23 | 0/22 | 0.982 |
| +50 ms | 23/23 | 0/22 | 0.939 |
| +100 ms | 23/23 | 5/22 | 0.484 |
| +150 ms | 18/23 | 18/22 | 0.062 |
| +200 ms | 9/23 | 22/22 | 0.015 |
| +250 ms | 3/23 | 22/22 | 0.028 |
| +300 ms | 0/23 | 22/22 | 0.243 |

This is **NOT** a viable production lookahead. PRESS recall collapses as
release recall improves, while the 0 ms Vision-only predictions match
their original recorded replay.

## Temporal-translation diagnostic

For the same recorded target timestamps `t`, compare:

```
A(t, delta) = P(model | RGB(t + delta), original frame-dt)
B(t, delta) = P(model | RGB(t), original frame-dt) evaluated at (t + delta)
```

`B` is computed by linear interpolation of zero-shift frozen probabilities.
Use only paired rows with `t >= 5000ms` and `t + delta <= 56800ms`;
the original time series is strictly increasing. The original H100 scored
target `t` is never changed for `A`.

| Future RGB shift | Correlation of A/B | A/B binary-action disagreement |
|---|---:|---:|
| +50 ms | 0.9977 | 0.90% |
| +100 ms | 0.9974 | 0.84% |
| +150 ms | 0.9975 | 0.78% |
| +200 ms | 0.9975 | 0.97% |
| +250 ms | 0.9975 | 0.91% |
| +300 ms | 0.9977 | 0.65% |

The optimal temporal offset within ±40ms (5ms steps) was the requested
shift itself for each of the six treatments. This comparison tests
**phase equivariance**; it does NOT establish physical cause or an
action-forecasting horizon.

## Critical label-provenance difference

1. `src/karting_agent/train/labels/touch_marker.py` derives expert action
   events from a **visual circle in the native recorded video**, assigning
   timestamps from the video frame index and FPS.
2. `scripts/experiment_tiny_temporal_policy.py` reads the resulting
   `data/processed/v3/labels/<video>.json` as the action history and the
   future absolute PRESS/RELEASE supervision at observation +100ms.
3. `scripts/evaluate_tiny_live_recording.py` instead uses **host-received
   Android getevent transitions**, mapped to the host's decoded-video origin.
4. `src/karting_agent/data_flow/input/adb_video.py` timestamps frames on
   **host decode availability**. Its debug MP4 uses a synthetic 60fps input
   time base: MP4 packet timestamps are not measured Android capture times.

Consequently, earlier expert-video Validation and the recent live-recording
replay do NOT use physically equivalent reference event timestamps.
The prior visual-circle alignment heuristic for this live recording estimated
approximately +72ms UI PRESS lag and +127ms UI RELEASE lag versus host touch
arrival, a ~55ms **UI hold-duration inflation**. These values are not a
calibration for other videos, and are insufficient by themselves to explain
the ~209–226ms model-versus-kernel hold-duration differences.

## Next high-value experiment (before architecture changes)

Create a **new short synchronized data-collection trace**, rather than
training on the existing fixed video as if it were a physical rollout:

1. Simultaneously collect encoded Android frames, per-frame host decoder
   availability, raw getevent device timestamps and host receipt timestamps,
   and the visual touch indicator. Keep all raw observations.
2. Distinguish **physical actuation**, **on-screen touch indicator
   visibility**, **game visual response**, **screen capture/encode**, and
   **host decode availability**. Do not substitute synthetic MP4 CFR PTS for
   device capture PTS.
3. Record a repeated sequence of PRESS, RELEASE and short corrective
   pulses at several different durations, with intentional quiet gaps.
   No model must control the phone during this calibration.
4. Measure PRESS and RELEASE separately and independently. Require event
   matching at the episode level; flag chatter/ambiguous pairs.
5. Establish a common clock and uncertainty intervals before deriving
   causal future-action labels for training. Compare physical-event labels
   with optical-circle-derived labels.
6. On the resulting independent sessions, decide whether earlier RELEASE
   prediction is even observable in the current RGB stack. If not, prioritize
   road/vehicle-state estimation and closed-loop control over a larger
   absolute-future-action imitation model.

A high Validation F1 against optical action markers is not proof of
accurate, low-latency physical RELEASE actuation. This record is not a
suitable source for threshold optimization, timing correction or Test split
model selection.
