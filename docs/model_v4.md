# Model v4: Structured CNN + GRU Policy

## 1. Why v4

v1-v3 proved that the realtime capture/ADB path is fast enough and that a lightweight CNN can learn strong offline action metrics. They also exposed the main limitation of direct behavior cloning: the visual backbone is only supervised by action labels, so it is free to use shortcuts instead of learning the spatial state that actually determines steering.

v4 keeps the project model-driven, but gives the model a more structured job:

```text
single frames
    ↓
shared CNN visual encoder
    ├── road / geometry auxiliary heads
    └── latent feature z(t)
             ↓
     sequence of z(t)
             ↓
            GRU
             +
      current PRESS/RELEASE
             ↓
      state-conditioned policy
             ↓
         KEEP / SWITCH
```

Traditional computer vision is allowed only as an **offline pseudo-label teacher**. It is not part of the final runtime path.

## 2. Responsibilities

### CNN: understand one frame spatially

The CNN should learn information such as:

- drivable-road segmentation;
- road centerline / local road direction;
- optional kart position and heading when reliable pseudo-labels become available;
- a compact latent feature vector for the temporal model.

The first v4 milestone requires only road-mask supervision plus the latent feature. Centerline, curvature, kart center, and kart heading are added only after their pseudo-labels are shown to be stable enough.

### GRU: understand how the scene is changing

The GRU receives one CNN latent vector per frame rather than a 15-channel image stack. It models short-term motion and temporal trend:

```text
z(t-200ms)
z(t-150ms)
z(t-100ms)
z(t-50ms)
z(t)
    ↓
   GRU
    ↓
temporal driving state
```

This lets the policy distinguish states that look similar in a single frame but are moving differently, for example a kart that is still rotating into a bend versus one that has already started recovering.

### Policy head: decide KEEP / SWITCH

The policy receives:

- the GRU temporal state;
- the physical current action (`PRESS` or `RELEASE`).

It predicts whether the current physical state should be kept or switched. The v3 counterfactual state-conditioning idea is retained.

## 3. Pseudo-label teacher

Pseudo-labels are generated offline from the recorded expert videos. The first teacher is a theme-specific HSV road segmenter because the current blue-track theme has strong color separation between the road and background.

```text
expert video frame
      ↓
OpenCV HSV threshold + morphology
      ↓
road mask
      ↓
optional row-wise centerline diagnostic
```

The teacher output is used only during training and evaluation. Deployment remains:

```text
camera / ADB video
      ↓
CNN
      ↓
GRU
      ↓
policy
      ↓
PRESS / RELEASE
```

The initial HSV thresholds are deliberately treated as a feasibility POC, not a cross-theme solution. If later recordings use different visual themes, the teacher can become adaptive or use per-theme threshold profiles without changing the learned runtime architecture.

## 4. Proposed model structure

The intended v4 model is:

```text
                       ┌─ road segmentation head
frame(t) → CNN encoder ┤
                       └─ latent z(t)

z(t-N ... t) → GRU → temporal feature ─┐
                                       ├→ nonlinear switch head
current action → state embedding ──────┘
```

The CNN weights are shared across time. Unlike v1-v3, the five RGB frames are **not** concatenated into a 15-channel image before the CNN. Each frame is encoded independently and the GRU receives the ordered feature sequence.

This separation is intentional:

- CNN = spatial representation;
- GRU = temporal representation;
- policy head = control decision.

## 5. Training losses

The first implementation target is:

```text
loss = L_switch
     + λ_road * L_road_segmentation
     + λ_action * L_future_action
```

Where:

- `L_switch` keeps the v3 state-conditioned KEEP/SWITCH objective;
- `L_road_segmentation` forces the CNN to encode drivable-road structure;
- `L_future_action` is retained as a weak auxiliary target for continuity with v2/v3.

Later geometry heads can add losses for centerline, curvature, kart position, or heading only after pseudo-label quality has been measured.

## 6. Data policy

The existing video-level split remains unchanged. Pseudo-labels must be generated independently for train/validation/test videos; they must not change the split.

The first geometry POC does **not** replace the current action labels. It adds a new supervision source on top of them.

## 7. Implementation phases

### Phase 0 — road pseudo-label feasibility

Before building the v4 network, run the road-mask teacher on representative frames from all 15 videos and inspect overlays.

Acceptance criteria:

1. the road region is selected rather than the cyan background;
2. the mask remains connected through bends and around the kart;
3. UI elements do not become the dominant component;
4. the row-wise centerline diagnostic follows the visible road where it is defined;
5. failures can be grouped by visual theme so thresholds can be adjusted systematically.

### Phase 1 — CNN geometry supervision

Train a shared per-frame MobileNetV3-Small encoder with a lightweight road segmentation decoder and latent feature output. Verify segmentation IoU/Dice on held-out videos and inspect predicted masks.

### Phase 2 — GRU state-conditioned policy

Feed the ordered latent sequence into a GRU and combine its final state with the current-action embedding. Train the counterfactual KEEP/SWITCH objective from v3.

### Phase 3 — joint fine-tuning and closed-loop

Jointly fine-tune the geometry and policy objectives, then repeat held-out replay and ADB closed-loop evaluation. If expert-only data still causes covariate shift, add closed-loop correction data after the structured model is working.

## 8. What v4 is not

v4 is not a hand-written geometric controller. The pseudo-label generator is a training tool, not the deployed policy.

v4 is also not simply "CNN + GRU" with the same weak action-only supervision. The important change is that the CNN is explicitly supervised to represent the road before the GRU learns temporal control.
