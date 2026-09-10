# AliManor Karting Agent Dataset Design

## 1. 数据目标

Dataset 的目标不是学习“当前是否正在按下”，而是根据短时间视觉历史预测未来控制状态，并重点学习 Transition 和 Short Correction。

当前原始数据：

- 15 个游戏录屏
- 总时长约 668.72 秒
- 原视频 FPS 约 65~87
- 完整 PRESS / RELEASE 段各 301 个
- RELEASE `<=300ms` 的 P-R-P 修正 115 / 301，约 38.2%
- RELEASE `100~300ms` 共 102 个

因此 100~300ms 的短时 RELEASE 是有效控制行为，不做时间平滑。

---

## 2. Label 生成

Touch Marker 作为玩家操作标签来源：

```text
Raw Video
    ↓
Touch Marker Detection @ native FPS
    ↓
Raw Action Timeline
    ↓
1-frame Glitch Cleaning
    ↓
Training Action Timeline
```

Action Timeline 保存 Transition Timestamp，不提前降采样。

第一版 Cleaning 规则：

- 只合并明确的 1-frame isolated glitch
- 不使用 `<100ms` 固定阈值删除动作
- 2-frame 及以上短动作保留
- 100~300ms Short Correction 完整保留

全量构建结果中只清理了 10 个 frame，说明该规则对原始操作序列影响很小。

---

## 3. Temporal Sample

不同地图车速不同，因此模型需要时序信息估计运动速度和趋势。

第一版输入：

```text
frame(t-100ms)
frame(t-50ms)
frame(t)
      ↓
Channel Stack
      ↓
CNN
      ↓
action(t + Δt)
```

当前基线：

```text
frame_stack           = 3
history_ms            = 100
frame_interval_ms     = 50
sample_fps            = 30
prediction_horizon_ms = 100
```

3 个 RGB Frame 在 Channel 维拼接为 9-channel 输入。

`prediction_horizon_ms=100` 是第一组实验值，后续对比 50~100ms。

3 帧是否足够通过 Ablation 判断：

- 1 frame
- 3 frames / 100ms
- 5 frames 或更长 history

---

## 4. Frame Cache

原始实现会针对每个训练 sample 对 MP4 做随机 seek。CPU benchmark 已显示 DataLoader 等待占据明显训练时间，而 GPU 上模型计算更快后，这个 I/O 瓶颈会进一步放大。

因此正式训练使用可重建的 Frame Cache：

```text
samples.jsonl
    ↓
按 Video 收集 input_frame_indices 的并集
    ↓
每个 Raw MP4 顺序 decode 一次
    ↓
Fixed Touch Mask
    ↓
Resize 224×224
    ↓
BGR → RGB
    ↓
uint8 Frame Cache
```

缓存结构：

```text
data/processed/frame_cache/
└── <video_stem>/
    ├── frames.npy
    └── index.json
```

其中：

- `frames.npy`：`[N, H, W, 3]` RGB `uint8`，使用 NumPy mmap 随机读取
- `index.json`：保存原始 Video Frame Index 与 Cache Row 的对应关系，以及预处理元数据
- 只缓存当前 `samples.jsonl` 实际访问的 unique input frames
- Normalize 不写入 Cache，读取时再执行，因此调整 mean/std 不需要重建 Cache
- Input Size、Touch Mask ROI 或 Temporal Sampling 改变后需要重建 Cache
- Cache 是 Derived Data，位于 `data/processed`，不提交 Git

Dataset 读取优先级：

```text
Frame Cache available
    → mmap cache

Frame Cache missing
    → MP4 fallback
```

正式 GPU 训练使用 `require_cache`，避免 Cache 不完整时无意间退回慢速 MP4 Random Seek。

---

## 5. Sampling

全量 Dataset Build 生成：

```text
samples                 = 19,979
near_transition         = 6,381  (~31.9%)
near_short_correction   = 1,862  (~9.3%)
```

Short Correction 样本是 Transition 样本中的高优先级子集。

第一版使用 Weighted Sampling，而不是同时叠加 Weighted Loss：

```text
Stable            : 1.0
Transition        : 2.0
Short Correction  : 3.0
```

Train Split 的实际原始构成为：

```text
Stable                 68.1%
Transition(non-short)  23.0%
Short Correction        8.9%
```

加权后的期望采样构成约为：

```text
Stable                 48.3%
Transition(non-short)  32.6%
Short Correction       19.0%
```

第一版不额外做 PRESS / RELEASE class balancing；Train / Validation / Test 的 PRESS 比例分别约为 55.8% / 54.2% / 57.9%，不存在明显类别失衡。

---

## 6. Split 与防泄漏

禁止 Frame-level random split。

第一版固定：

```text
Train       11 videos / 14,678 samples
Validation   2 videos /  2,546 samples
Test         2 videos /  2,755 samples
```

策略：

```text
video_holdout_visual_theme_stratified
```

Contact Sheet 没有提供足够证据证明存在“同一赛道重复录制”；因此不强行把相似路面主题视为同地图。Validation / Test 保持完整视频隔离，同时尽量选择 Train 中存在相近视觉主题的地图，优先评价新赛道几何泛化。

Touch Marker 只能用于 Label，必须从模型输入固定 Mask。

训练目标使用 Future Action：

```text
frames(..., t) → action(t + Δt)
```

避免模型只利用已经发生动作产生的视觉特征。

---

## 7. Evaluation

训练阶段保留逐 sample 分类指标：

- Accuracy
- Precision
- Recall
- F1
- GT PRESS ratio
- Predicted PRESS ratio

这些指标用于观察训练稳定性，但不能直接代表闭环驾驶能力。

### 7.1 Sequence-level Transition

正式离线评价按每个完整 Video 的时间顺序执行推理。第一版先将 PRESS probability 以 `threshold=0.5` 转为离散状态，再从状态变化提取预测 Transition。

Ground Truth Transition 直接来自 native-FPS Action Timeline，不从 30Hz sample 标签重新生成，以保留原始操作时序精度。

预测 Transition 与 Ground Truth Transition 必须满足：

- 同一 Video
- 同一方向（`PRESS onset` 或 `RELEASE onset`）
- 时间差绝对值不超过 `transition_tolerance_ms`
- 一对一匹配

第一版：

```text
threshold               = 0.5
transition_tolerance_ms = 100
```

未匹配预测 Transition 记为 False Positive；未匹配 Ground Truth Transition 记为 False Negative。因此短时间内反复抖动产生的额外切换会降低 Transition Precision / F1。

输出：

- Transition Precision / Recall / F1
- PRESS Transition Precision / Recall / F1
- RELEASE Transition Precision / Recall / F1
- PRESS onset signed error / MAE / P50 abs / P95 abs / max abs
- RELEASE onset signed error / MAE / P50 abs / P95 abs / max abs

`mean_error_ms = predicted_timestamp - ground_truth_timestamp`：负值表示提前，正值表示滞后。

### 7.2 Short Correction Recall

Short Correction 仍按 native Action Timeline 中 `100~300ms` 的 RELEASE Segment 定义。

一个 Ground Truth Short Correction 只有在以下两个边界都成功匹配时才算 detected：

```text
PRESS → RELEASE onset matched
RELEASE → PRESS onset matched
```

因此它衡量的是模型是否完整复现一次短时修正，而不是“短修正附近的 frame 分类是否正确”。这与训练日志中的 `short_f1` 不同；后者只是 `near_short_correction` sample subset 的分类 F1。

Release Segment 同时按 duration 输出：

```text
<100ms        # 仅监控，不作为有效 Short Correction 目标
100~200ms
200~300ms
>300ms
```

其中 `100~300ms` 另外汇总为 `Short Correction Recall`。

### 7.3 Action Persistence Baseline

为了确认 CNN 是否真的学到了未来控制信号，而不是只识别当前操作状态并假设其继续保持，Evaluation 需要一个不使用图像、不训练模型的 label-only 对照：

```text
Ground Truth action(t)
        ↓ copy state
Prediction action(t + prediction_horizon_ms)
```

当前 `prediction_horizon_ms=100`，因此该基线定义为：

```text
action(t) → action(t+100ms)
```

这个 baseline 故意使用输入时刻 `t` 的真实动作标签，相当于一个拥有“当前动作 oracle”的 persistence predictor。它不是 Runtime 实现方案，而是用于量化数据本身的动作持续性：如果 CNN 只是在视觉上恢复 `action(t)`，其表现不应明显超过该基线。

Persistence Baseline 与 CNN 使用完全相同的 Video Split、Sequence Transition Matching 和 Short Correction Recall。重点比较：

- sample accuracy：说明单纯保持当前动作本身能拿到多高的逐帧准确率
- Transition Precision / Recall / F1
- PRESS / RELEASE signed timing error 与 MAE
- Short Correction Recall

如果 persistence baseline 的 transition signed error 接近 `+prediction_horizon_ms`，而 CNN 的误差显著更接近 0，说明 CNN 确实利用视觉信息提前预测了未来切换；反之则需要警惕 current-action leakage / action persistence shortcut。

当前 Sequence Evaluator 评价的是 raw classifier state；Runtime Hysteresis 的影响将在 Replay / Controller-aware Evaluation 阶段单独评价。

最终评价仍以 Replay / 实机闭环为准：

- 过弯成功率
- 位置修正效果
- 完整赛程完成率
- Short Correction 是否延迟或遗漏

---

## 8. 当前基线

```text
Label extraction       : native video FPS
Glitch cleaning        : isolated 1-frame only
Valid short correction : preserve 100~300ms
Frame stack            : 3
History window         : 100ms
Frame interval         : 50ms
Sample rate            : 30Hz
Prediction horizon     : 100ms baseline
Transition window      : ±200ms training sample tagging
Sampling weights       : 1 / 2 / 3
Split                  : v1 video-level holdout
Training input         : mmap frame cache preferred
Cache representation   : prepared RGB uint8
Sequence threshold     : 0.5
Transition tolerance   : ±100ms
Control baseline       : action(t) → action(t+100ms) persistence
```

当前不做：

- minimum action duration
- `<100ms` 全量删除
- Frame-level random split
- Weighted Sampling 与 Weighted Loss 同时叠加
- 把相似视觉主题直接当作同一地图
- 用训练期 `short_f1` 代替 Sequence Short Correction Recall
