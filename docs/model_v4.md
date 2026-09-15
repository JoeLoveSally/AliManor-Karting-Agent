# Model v4: Structured Temporal Policy

## 1. Goal

v4 keeps the project model-driven while separating spatial perception, temporal representation, and control:

```text
RGB frame sequence
      ↓
shared per-frame CNN
      ↓
ordered latent sequence z(t)
      ↓
temporal fusion
      +
current PRESS / RELEASE
      ↓
state-conditioned policy
      ↓
KEEP / SWITCH
```

Traditional computer vision is allowed only as an offline pseudo-label teacher and shadow debugger. It is not part of the deployed control path.

The final runtime remains learned:

```text
Camera / ADB video
      ↓
CNN
      ↓
learned temporal fusion
      ↓
learned policy
      ↓
PRESS / RELEASE
```

## 2. What v3 actually proved

v3 concatenates five RGB frames into a 15-channel tensor. The pretrained first convolution is initialized by repeating the RGB filter for each frame and dividing by `frame_stack`.

At initialization only, this is equivalent to applying the original RGB filter to the average of the input frames. The expanded convolution is trainable, so the trained model is not constrained to remain a temporal average.

The temporal counterfactual diagnostic confirmed that trained v3 uses channel order:

```text
repeat_latest: flip_rate = 0.351
reverse:       flip_rate = 0.622

adjacent replacement:
-200ms: 0.000
-150ms: 0.081
-100ms: 0.081
 -50ms: 0.297
    0ms: 0.649
```

So the v4 question is not "can the model use time at all?". It is whether a cleaner explicit time representation improves generalization and closed-loop control.

## 3. Invariants across temporal ablations

For the v3 -> v4-A -> v4-A2 comparison, keep these fixed unless a later experiment explicitly changes them:

- same 15 source videos;
- same train / validation / test video split;
- same v3 sample manifest;
- same five observation times `[-200, -150, -100, -50, 0] ms`;
- same `224×224` preprocessing;
- same touch/HUD masks;
- same MobileNetV3-Small backbone family;
- same 128-d per-frame visual feature in explicit-time variants;
- same current-action conditioning;
- same counterfactual train states;
- same KEEP/SWITCH targets;
- same future-action auxiliary targets and weight `0.5`;
- same `100/200/300ms` prediction horizons and `100ms` control horizon;
- same sampling weights, optimizer, learning rate, weight decay, batch size, epoch count, and seed;
- same checkpoint rule: lowest validation `switch_loss`;
- no geometry supervision during the temporal ablation.

The existing `data/processed/frame_cache_v2` is reused. It already stores individual prepared RGB frames, so explicit-time models only reshape the model-facing tensor from `(15,H,W)` to `(5,3,H,W)`.

## 4. v4-A: per-frame CNN + GRU

### Architecture

```text
5 RGB frames
   ↓ shared MobileNetV3-Small, independently per frame
5 × 128-d z(t)
   ↓ GRU(hidden=128, 1 layer)
h(t)
   + 8-d current-action embedding
   ↓ nonlinear policy head
switch@100 / 200 / 300ms

h(t)
   ↓ auxiliary head
future-action@100 / 200 / 300ms
```

### Result

v4-A trained successfully but overfit very quickly. Lowest validation `switch_loss` occurred at epoch 1; training metrics then approached saturation while validation loss worsened.

Held-out test result of the selected checkpoint:

```text
static switch@100 F1 = 0.776
aux action F1:
  h100 = 0.960
  h200 = 0.959
  h300 = 0.962
```

Stateful test replay:

```text
threshold 0.50:
  target transition F1 = 0.892
  matched              = 74 / 88
  predicted            = 78
  short recall         = 0.722
  chatter <100ms       = 0
  chatter <200ms       = 1
  observation F1       = 0.554

threshold 0.60:
  target transition F1 = 0.867
  matched              = 72 / 88
  predicted            = 78
  short recall         = 0.722
  chatter <100ms       = 0
  chatter <200ms       = 1
  observation F1       = 0.590
```

The relevant v3 reference at threshold `0.60` is:

```text
target transition F1 = 0.918
matched              = 78 / 88
predicted            = 82
short recall         = 0.778
chatter <100ms       = 1
chatter <200ms       = 9
observation F1       = 0.753
```

### Interpretation

v4-A is much smoother but misses useful short corrections and does not beat v3 on the main stateful metrics. Increasing threshold mostly delays switches; it does not recover the lost target-timeline quality.

This does **not** prove that an explicit time axis is unhelpful, because v4-A changed two things together:

```text
1. 15-channel joint CNN → shared per-frame CNN
2. implicit channel-time fusion → recurrent GRU fusion
```

Therefore the next experiment removes recurrence while keeping the explicit per-frame representation.

Do not spend the next iteration on GRU threshold tuning, longer GRU history, or closed-loop ADB testing. The offline gate is already worse than v3.

## 5. v4-A2: per-frame CNN + ordered temporal MLP

### Question

Does the explicit per-frame representation help when time is preserved without recurrent smoothing?

### Architecture

```text
frame -200 → shared CNN → z1
frame -150 → shared CNN → z2
frame -100 → shared CNN → z3
frame  -50 → shared CNN → z4
frame    0 → shared CNN → z5

[z1, z2, z3, z4, z5]
        ↓ ordered concatenation
      640-d vector
        ↓ temporal MLP
      128-d h(t)
        + current PRESS / RELEASE embedding
        ↓ policy head
      KEEP / SWITCH
```

The temporal MLP deliberately receives frame features in fixed chronological positions. It does not average them and does not use a recurrent hidden state.

The auxiliary future-action head remains attached to the fused temporal feature so the comparison keeps the same secondary supervision as v3/v4-A.

### Controlled difference from v4-A

Only temporal fusion changes:

```text
v4-A:
5 × 128 feature sequence
→ GRU(128)
→ temporal feature

v4-A2:
5 × 128 feature sequence
→ ordered concat (640)
→ Linear(640,128) + Hardswish + Dropout
→ temporal feature
```

Everything listed in section 3 remains fixed.

### Acceptance criteria

First compare offline metrics. Do not run ADB closed loop unless v4-A2 is at least competitive with v3.

Primary reference:

```text
v3 @ threshold 0.60
stateful target transition F1 = 0.918
short correction recall       = 0.778
```

Interpretation rule:

```text
v4-A2 >= v3 on meaningful sequence metrics
→ explicit per-frame representation remains promising;
  GRU was the likely bad temporal inductive bias.

v4-A2 still clearly below v3
→ stop temporal-architecture redesign for now;
  return to the best v3 control path and move to structured geometry supervision.
```

Do not infer success from static sample F1 alone.

## 6. v4-A2 implementation

Implemented files:

```text
configs/train_v4a2.yaml
src/karting_agent/model/temporal_mlp_state_conditioned.py
scripts/train_model_v4a2.py
scripts/evaluate_model_v4a2.py
tests/unit/test_v4a2_training.py
```

The training script intentionally reuses the v4-A epoch/loss/metric helpers so optimizer and metric semantics do not drift between the two ablations.

No v4-A2 runtime is added yet. Runtime implementation is deferred until the offline gate justifies real-device testing.

## 7. Later temporal-history ablation

The previous v4-B `200/400/800ms` history experiment is paused, not deleted.

It should only resume if an explicit-time architecture first proves competitive at the existing 200ms window. Otherwise longer history mixes another variable into a temporal representation that has not yet justified itself.

If resumed, keep five frames:

```text
200ms: [-200, -150, -100,  -50, 0]
400ms: [-400, -300, -200, -100, 0]
800ms: [-800, -600, -400, -200, 0]
```

Spatial lookahead and temporal history remain different concepts; road visibility hundreds of milliseconds ahead does not directly imply the temporal history must be equally long.

## 8. Geometry auxiliary supervision

If temporal redesign fails to beat v3, the next model change is structured visual supervision rather than a larger temporal model.

First candidate:

```text
road segmentation
```

Later candidates may include centerline, curvature, bend distance, kart center, or kart heading only after pseudo-label quality is audited.

The existing OpenCV geometry pipeline is an offline teacher, not the deployed controller.

Before pseudo-labels enter a loss:

1. sample representative frames from all 15 videos;
2. inspect mask continuity through bends and around the kart;
3. mark unreliable intervals/themes;
4. exclude or down-weight unreliable pseudo-labels;
5. measure valid-label coverage.

The geometry loss remains a real hyperparameter:

```text
L = L_switch
  + λ_action * L_future_action
  + λ_road * L_road
```

`λ_road` must be tuned rather than fixed by convention.

## 9. Representation vs. covariate shift

Architecture and data distribution are separate axes.

A stronger visual/temporal representation may reduce the probability of the first action error and improve mild-deviation generalization, but expert-only behavior cloning still lacks many states created by the model's own earlier mistakes.

Therefore later closed-loop iterations must preserve failure/deviation runs and eventually add valid pre-failure correction/recovery supervision through iterative behavior cloning or a DAgger-like process.

Do not rely on impossible post-failure recovery labels after the kart is already irreversibly off track.

## 10. Shadow teacher

Keep the analytic geometry teacher for post-run diagnosis even when deployed control is model-only:

```text
recorded MP4
   ├── learned visual/geometry prediction
   ├── offline analytic teacher geometry
   └── policy KEEP/SWITCH output
```

This helps separate:

```text
perception wrong
→ visual representation problem

perception reasonable, policy wrong
→ temporal/control problem

both reasonable but observation is off expert distribution
→ data/recovery problem
```

The teacher never sends runtime control actions.

## 11. Evaluation matrix

Every v4 variant is compared against the same held-out videos and v3 reference.

```text
Policy quality
- static switch precision / recall / F1
- stateful target transition F1
- stateful observation transition F1
- short-correction recall
- chatter <100ms / <200ms
- PRESS / RELEASE timing errors

Runtime quality (only after offline gate)
- input fps
- control fps
- inference mean / p95 / max
- frame gap p95 / max
- CNN feature-cache hit / miss ratio

Closed-loop quality (only after offline gate)
- bends completed
- time / distance before failure
- first irreversible trajectory deviation
- recovery success on small deviations

Later representation quality
- road Dice / IoU
- geometry teacher/student disagreement
```

Offline metrics are gates and diagnostics, not proof of closed-loop success.

## 12. Current implementation order

```text
1. v4-A2 local tests + smoke
2. v4-A2 Spark CUDA/cache smoke
3. v4-A2 full training
4. v4-A2 static + stateful held-out evaluation
5. compare v3 vs v4-A vs v4-A2
6. only if v4-A2 is competitive: implement runtime and ADB A/B
7. if v4-A2 is not competitive: stop temporal redesign and move to audited geometry supervision
8. keep pseudo-label audit running in parallel
9. later collect valid closed-loop deviation/recovery data
```

## 13. What v4 is not

v4 is not a hand-written geometric controller.

v4 is not based on the claim that v3 had no temporal information.

v4 is not committed to GRU. The invariant is a controlled, evidence-driven search for a representation that preserves useful spatial/temporal information without degrading short corrective control.

v4 is also not expected to solve behavior-cloning covariate shift by architecture alone.
