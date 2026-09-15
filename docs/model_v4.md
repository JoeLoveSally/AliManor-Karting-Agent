# Model v4: Structured CNN + GRU Policy

## 1. Goal

v4 keeps the project model-driven, but separates three jobs that v1-v3 mixed together:

```text
RGB frame sequence
      ↓
shared per-frame CNN
      ↓
ordered latent sequence z(t)
      ↓
GRU temporal state
      +
current PRESS / RELEASE
      ↓
state-conditioned policy
      ↓
KEEP / SWITCH
```

Traditional computer vision is allowed only as an offline pseudo-label teacher and shadow debugger. It is not part of the deployed control path.

The final runtime remains:

```text
Camera / ADB video
      ↓
CNN
      ↓
GRU
      ↓
learned policy
      ↓
PRESS / RELEASE
```

## 2. What v3 actually proved

v3 concatenates five RGB frames into a 15-channel tensor. The pretrained first convolution is initialized by repeating the RGB filter for each frame and dividing by `frame_stack`.

At initialization only, this first convolution is therefore equivalent to applying the original RGB filter to the average of the input frames. This does **not** mean the trained v3 model remains a temporal average: the expanded convolution is trainable, so the per-frame channel weights may diverge during optimization.

The temporal counterfactual diagnostic also showed that the trained v3 model uses channel order:

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

So v3 did learn temporal information, but it represents time implicitly through channel position. v4 changes this to an explicit sequence representation because that is a cleaner temporal inductive bias and easier to inspect, ablate, and run incrementally.

## 3. Responsibilities

### CNN — spatial perception

The CNN processes one RGB frame at a time with weights shared across time. Its primary output is a compact latent feature `z(t)`.

Later geometry supervision may require the same encoder to predict:

- drivable-road segmentation;
- local road direction / centerline;
- curvature or distance-to-bend when pseudo-label quality is sufficient;
- kart position / heading only when reliable labels are available.

The CNN is the spatial perception model, not the temporal controller.

### GRU — temporal state

The GRU receives the ordered latent sequence rather than an image stack collapsed into channels:

```text
z(t-200ms)
z(t-150ms)
z(t-100ms)
z(t-50ms)
z(t)
    ↓
   GRU
    ↓
temporal state h(t)
```

Its job is to encode motion trend: whether the kart is rotating into a bend, continuing to drift, recovering, or moving toward/away from a road boundary.

GRU is the first temporal model to test, not a permanent architectural requirement. A temporal 1D convolution or attention model can replace it later if evidence justifies that change. The invariant is that the time axis remains explicit until temporal modeling occurs.

### Policy head — control decision

The policy receives:

- temporal state `h(t)`;
- the physical current state (`PRESS` or `RELEASE`).

It predicts `KEEP / SWITCH`. The v3 counterfactual current-state training remains part of v4.

## 4. Important separation: representation vs. covariate shift

v4 addresses representation quality. It does **not** claim to solve behavior-cloning covariate shift by architecture alone.

Expert-only training data mostly contains states on expert trajectories. Closed-loop execution can create states such as:

```text
small timing error
→ different lateral position / heading
→ observation is less represented in expert data
→ another action error
→ larger deviation
```

A better CNN/GRU can reduce the probability of entering this failure chain and may generalize better to small deviations, but recovery from off-expert states ultimately requires closed-loop correction data if those states are absent from training.

Therefore architecture work and on-policy data work are tracked as two independent axes.

## 5. Experimental plan

The central rule is: change one source of capability at a time.

### v4-A — explicit temporal representation, 200ms

First test only the temporal representation change.

Keep from v3:

- same train / validation / test video split;
- same five observation times: `[-200, -150, -100, -50, 0] ms`;
- same image resolution initially (`224×224`);
- same current-action conditioning;
- same counterfactual train states;
- same KEEP/SWITCH targets;
- same future-action auxiliary target and weight;
- no geometry supervision.

Change only:

```text
v3:
5 RGB frames → 15-channel CNN → visual feature

v4-A:
each RGB frame → shared 3-channel CNN → ordered latent sequence → GRU
```

This is the strictest test of whether explicit temporal modeling improves offline sequence metrics and closed-loop behavior.

Acceptance is based on more than sample F1:

- stateful transition F1;
- short-correction recall;
- transition timing error;
- chatter;
- held-out replay;
- ADB closed-loop trajectory and survival time.

If v4-A does not improve meaningful sequence/closed-loop behavior, do not continue increasing temporal-model complexity by default.

### v4-B — history-window ablation

Only after v4-A is working, test how much temporal history is useful. Do not infer required history directly from how far ahead the road is visible: spatial lookahead and temporal history answer different questions.

Initial controlled variants keep five frames to avoid mixing history length and compute:

```text
200ms: [-200, -150, -100,  -50, 0]
400ms: [-400, -300, -200, -100, 0]
800ms: [-800, -600, -400, -200, 0]
```

The 200/400/800ms values are experiment points, not assumptions that longer must be better.

### v4-C — geometry auxiliary supervision

Only after the temporal ablation is understood, add explicit geometry supervision to the CNN.

Initial target:

```text
road segmentation
```

Later targets may include centerline, curvature, bend distance, kart center, or kart heading only after their pseudo-labels pass quality audit.

This sequencing allows attribution:

```text
v3 → v4-A : effect of explicit temporal representation
v4-A → v4-B : effect of longer temporal context
v4-B → v4-C : effect of structured geometry supervision
```

### v4-D — closed-loop correction data

In parallel with v4-A/B/C, preserve every closed-loop failure run. Once the structured model is functional, add recovery / deviation states through iterative behavior cloning or a DAgger-like collection loop.

The key new data is not more copies of expert-centerline driving. It is states such as:

- kart too far toward one boundary;
- heading already over-rotated;
- late release / late press situations;
- valid recovery after a small policy error.

## 6. v4-A implementation

The first controlled variant is implemented with:

```text
configs/train_v4a.yaml
src/karting_agent/model/sequential_state_conditioned.py
src/karting_agent/model/sequential_state_conditioned_runner.py
src/karting_agent/train/state_conditioned_dataset.py
src/karting_agent/runtime/sequential_state_conditioned_engine.py
scripts/train_model_v4a.py
scripts/evaluate_model_v4a.py
tests/unit/test_v4a_training.py
```

The v4-A dataset deliberately reuses the exact v3 manifest and the existing v2/v3 frame cache. Spatial preprocessing is unchanged. The only model-facing data change is:

```text
v3 dataset output:   (15, H, W)
v4-A dataset output: (5, 3, H, W)
```

The same five cached RGB frames are therefore used without rebuilding the 5.6 GB cache.

The model structure is:

```text
5 RGB frames
   ↓ shared MobileNetV3-Small, applied independently
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

Checkpoint selection remains lowest validation `switch_loss`, matching v3.

## 7. Runtime compute design

Training encodes all five frames. Realtime inference must not rerun the CNN for all five overlapping frames on every control step.

The initial v4-A runtime therefore uses an exact-window feature cache:

```text
new temporal window indices
      ↓
look up z(frame_index)
      ↓ miss only
shared CNN once for unseen frame
      ↓
assemble the exact five latent features used by training
      ↓
small GRU over 5 latent vectors
      ↓
policy
```

This preserves the exact sampled-window semantics while making steady-state CNN cost approach one new-frame encode per observation. The GRU is tiny, so recomputing five latent steps is intentionally preferred over introducing a recurrent hidden-state semantic mismatch at this stage.

A later streaming-GRU optimization is allowed only if it reproduces the trained sampling semantics or is trained explicitly for streaming state.

For CPU runtime, benchmark and pin a small PyTorch thread count (`1`, `2`, `4`). Do not assume the host default is optimal.

## 8. Pseudo-label teacher and quality audit

The existing geometry teacher remains useful, but its role changes from "next thing to train" to a parallel data-quality track.

First teacher:

```text
expert video frame
      ↓
OpenCV HSV threshold + morphology
      ↓
road mask
      ↓
optional centerline / geometry diagnostics
```

Before pseudo-labels enter a loss, audit all 15 videos by visual theme:

1. sample representative frames from every video;
2. inspect mask continuity through bends and around the kart;
3. mark unreliable intervals or themes;
4. exclude or down-weight unreliable pseudo-labels;
5. measure coverage instead of assuming all teacher outputs are valid.

A teacher error must not silently become a student target.

## 9. Geometry loss is a hyperparameter, not a constant

When v4-C begins, the loss is conceptually:

```text
L = L_switch
  + λ_action * L_future_action
  + λ_road * L_road
```

`λ_road` is not fixed by convention. It must be tuned because two failure modes are possible:

```text
λ_road too small  → visual shortcut may remain
λ_road too large  → representation over-optimizes segmentation and hurts policy
```

The geometry loss is accepted only if it improves held-out geometry quality without degrading stateful sequence/control metrics. Additional geometry heads get their own independently tuned weights.

## 10. Shadow teacher for post-run diagnosis

Keep the analytic geometry teacher even after the deployed runtime becomes model-only.

After each recorded closed-loop run:

```text
recorded MP4
   ├── learned CNN geometry prediction
   ├── offline analytic teacher geometry
   └── policy KEEP/SWITCH output
```

This separates failure classes:

```text
CNN geometry wrong
→ perception failure

CNN geometry reasonable, policy wrong
→ temporal / policy failure

both reasonable but state is outside training distribution
→ data / recovery failure
```

The shadow teacher never sends runtime control actions.

## 11. Evaluation matrix

Every v4 variant must be compared against the same v3 reference using the same held-out videos and runtime semantics.

Track separately:

```text
Policy quality
- static switch precision / recall / F1
- stateful target transition F1
- stateful observation transition F1
- short-correction recall
- chatter <100ms / <200ms
- PRESS / RELEASE timing errors

Runtime quality
- input fps
- control fps
- inference mean / p95 / max
- frame gap p95 / max
- CNN feature-cache hit / miss ratio

Closed-loop quality
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
1. v4-A train/smoke using the exact v3 manifest and frame cache
2. v4-A static + stateful held-out evaluation
3. strict v3 vs v4-A replay/ADB comparison
4. v4-B: 200 / 400 / 800ms history ablation
5. continue pseudo-label quality audit in parallel
6. v4-C: add audited road supervision and tune λ_road
7. shadow-teacher diagnostics on closed-loop runs
8. collect recovery/on-policy data and iterate
```

The existing geometry pseudo-label POC is retained; it is not discarded. It simply does not enter the policy training loss until v4-A/B isolate the temporal contribution and the teacher quality has been audited.

## 13. What v4 is not

v4 is not a hand-written geometric controller.

v4 is not based on the claim that v3 had no temporal information. v3 used time implicitly through trainable channel positions; v4 makes the sequence explicit and gives the architecture a better temporal inductive bias.

v4 is also not expected to solve expert-data covariate shift by architecture alone. Closed-loop correction data remains a separate requirement if recovery states are missing from expert demonstrations.
