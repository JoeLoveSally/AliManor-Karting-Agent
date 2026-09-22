# V4-C4 first-bend failure: visual evidence and next iteration gate

This note is based on the **existing** `v4c4_armed_01.mp4`, `v4c4_armed_01.json`, and the user's `v4c4_armed_01_multitask.json` console excerpt. It introduces no gameplay, new labels for actions, new model weights, or control-policy changes.

## Confirmed evidence

The 234 logged current-action probabilities reproduced exactly from the recorded five-frame inputs (`max_action_replay_error=0`; no threshold disagreements). This validates replay consistency, **not** image capture freshness or steering correctness. The prior Settings/HOME responsiveness test included Android page switching; the browser timestamp experiment produced zero decodable samples; true game-frame age is **unknown**.

Visually inspect the actual MP4 source frames below (frame indices, **not** synchronized Android capture timestamps):

| MP4 source frame | Visible road-relative state (manual review) | V4-C4 action probability from original/multi-head replay | Observed runtime state | Auxiliary-head diagnostic (where available) |
|---:|---|---:|---|---|
| 376 | Kart on roadway, first PRESS transition | 0.62542 | PRESS | lateral −0.0874, edge 0.2207 |
| 389–390 | Kart on roadway, corner entry | ≈1.0 | PRESS | at 389: lateral −0.3902, edge 0.1678 |
| 399–400 | Kart near visible roadway boundary | ≈1.0 | PRESS | at 399: lateral −0.4133, edge 0.2029 |
| 404–405 | Kart crossing/near visible boundary; exact contact not labeled | ≈0.9999 | PRESS | do not create an automatic action label |
| 410 | Kart visibly outside roadway (among reviewed samples) | 0.99973 | PRESS | lateral −0.2951, edge 0.2745 |
| 420 | Kart visibly outside roadway | 0.94491 | PRESS | lateral −0.7010, edge 0.2201 |
| 430 | Kart visibly outside roadway | 0.97018 | PRESS | lateral −0.5317, edge 0.1969 |
| 440 | Kart visibly outside roadway | 0.92663 | PRESS | lateral −0.7713, edge 0.1782 |
| 444 | Game's failure overlay appears; first RELEASE follows action probability dropping below threshold | 0.34166 | RELEASE | lateral −0.0058, edge 0.3234; overlay predictions have no valid driving interpretation |

On the **host observation clock** in the original JSON, frame 410 is about 669 ms earlier than first RELEASE at frame 444. This is NOT a calibrated capture-to-action lag and should not be construed as the time for which the *physical vehicle* was off the road. Source MP4 playback uses nominal 60 fps and cannot independently recover the phone's frame-capture timestamps.

The high PRESS probability continues even after the kart is visibly off-road. Because the model's edge-risk target is `abs(lateral_target) >= 0.70`, an edge probability of 0.178 when the *separate predicted* lateral value is −0.771 indicates that the two heads' predictions are not reliably coherent on this live recording; it does **not** establish ground-truth road geometry. The heading head's raw `(cos(2θ), sin(2θ))` vector has a non-unit norm on live frames: `atan2` still yields an angle modulo 180°, but calibration against live geometry has not been established.

## What NOT to infer

- The failure does not identify the correct control action from any single paused image; do not auto-label all off-road frames as RELEASE or align the first bend to an expert's prerecorded action timeline.
- Do not claim `counterfactual_train_states=true` synthesizes off-road observations. It only duplicates an existing *visual* sample with each hypothetical current button state.
- Do not turn lateral/heading/edge auxiliary predictions into forced action rules or tune `action_threshold`/`min_state_hold_ms` solely to make this one failed rollout look better.
- No valid Android capture-to-decoded-video age has been established. Treat visual-domain shift and capture latency as **separate unresolved hypotheses**, not as proven causes.

## Next dataset, limited first-bend collection

Collect a small **actual-phone, same-game-appearance** demonstration set with human control. Include ordinary entry, intentional short pulse corrections **before** departure, and voluntary recovery from near-edge states where possible. Record both visible game frames and real held-button state/transitions in a single verifiable timeline; never derive PRESS/RELEASE labels from the failed autonomous run. Native phone recording with explicit visible tap indicators can be used for independent annotation only if the control-indicator region is removed from every model input and the indicator's actual temporal semantics are checked. Independently logged touch events are preferable where accessible, but Android restrictions and clock synchronization must be measured rather than assumed.

For every demonstration, store `session_id`, native source video, visual theme/kart appearance, resolution/FPS, action transition times and timestamp provenance, and per-interval `on_road / near_edge / off_road / unknown` *observations*. Visually inspect ambiguous examples; keep the autonomous failure MP4 in a **diagnostic-only/holdout** pool, never as automatic expert action supervision. Split by full recording/session rather than adjacent frames to avoid validation leakage.

Before another Armed rollout: verify label/frame synchronization on a few visibly short pulses, show action-head behavior on same-phone held-out human trajectories, and compare geometry auxiliary heads against independently reviewed live-frame road occupancy. A passing offline audit still does not prove closed-loop recovery; subsequent measured criteria include first departure time, actual recovery, and completed runs. Preserve the original V4-C4 model/runtime as baseline and PR draft until these checks pass.
