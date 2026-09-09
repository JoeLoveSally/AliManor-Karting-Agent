# AliManor Karting Agent Dataset Design

## 1. 数据目标

Dataset 的目标不是简单学习“当前画面是否正在按下”，而是让模型根据短时间视觉历史预测未来控制状态，并重点学习动作切换和短时位置修正。

当前原始数据：

- 15 个游戏录屏
- 总时长约 668.72 秒
- 原视频 FPS 约 65~87
- 完整 PRESS 段 301 个
- 完整 RELEASE 段 301 个

触控统计显示：

- RELEASE `<=300ms` 的 `P-R-P` 修正：115 / 301，约 38.2%
- RELEASE `100~200ms`：40 个
- RELEASE `200~300ms`：62 个

因此 100~300ms 的短时 RELEASE 不是边缘噪声，而是需要保留和重点学习的 **Short Correction**。

---

## 2. Label 生成

录屏中的 Touch Marker 作为玩家真实操作标签来源。

```text
Raw Video
    ↓
Touch Marker Detection @ native FPS
    ↓
Raw Action Timeline
    ↓
Glitch Cleaning
    ↓
Training Action Timeline
```

Action Timeline 优先保存 Transition Timestamp，而不是先降采样为固定 FPS 的 0/1 序列，例如：

```text
12.351s PRESS
12.487s RELEASE
12.676s PRESS
```

这样后续可以自由调整：

- Training FPS
- Frame Stack
- Frame Interval
- Prediction Horizon

而无需重新检测 Touch Marker。

### Glitch Cleaning

当前检测结果存在少量约 14~30ms 的段，其中部分表现为：

```text
PRESS ─── RELEASE(1 frame) ─── PRESS
```

或：

```text
RELEASE ─── PRESS(1 frame) ─── RELEASE
```

这类 1-frame 孤立状态高度可能是 Touch Marker 漏检 / 误检。

第一版规则：

- 只自动合并明确的 **1-frame isolated glitch**
- 不使用固定 `<100ms` 时长阈值删除动作
- 2-frame 及以上短动作暂时保留，必要时人工抽查
- 100~300ms 动作必须完整保留

---

## 3. 模型样本

不同地图存在不同车速，因此单帧不能可靠表达速度、横向运动和运动趋势。

第一版样本使用 3 个 RGB Frame：

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

初始参数：

```text
frame_stack            = 3
history_ms             ≈ 100
frame_interval_ms      ≈ 50
prediction_horizon_ms  = 50~100（实验确定）
```

3 帧直接在 Channel 维拼接，形成 9-channel 输入。第一版优先使用 ResNet18 或 MobileNetV3-Small。

3 帧是否足够不预先假定，后续通过 Ablation 对比：

- 1 frame
- 3 frames / ~100ms
- 更长 history 或 5 frames

比较重点是 Transition Timing 和实机控制效果，而非仅比较 Accuracy。

---

## 4. Sampling 与训练权重

数据可按控制意义分为：

```text
Stable PRESS
Stable RELEASE
PRESS Transition
RELEASE Transition
Short Correction
```

真实数据中 Stable Frame 数量远多于 Transition，但游戏效果主要取决于：

```text
什么时候 PRESS
什么时候 RELEASE
什么时候快速 RELEASE 后重新 PRESS
```

因此 Dataset 不应简单均匀抽帧。

第一版采用 Transition-aware Sampling：

- Transition 附近样本提高采样概率或 Loss Weight
- Short Correction 的两个边界都视为高价值 Transition
- Stable 区域保留，但不允许数量优势淹没 Transition

Transition Window 初始可使用约 `±200ms`，后续根据实验调整。

对于 100~300ms Short Correction，两个 Transition Window 可能重叠，这是预期行为，表示整个短动作区域都属于高价值决策区域。

---

## 5. Dataset Split 与防泄漏

禁止 Frame-level random split。

相邻视频帧高度相似，如果随机拆分：

```text
frame(t)       → Train
frame(t+15ms)  → Validation
```

会造成严重 Data Leakage。

必须按完整 Video 划分：

```text
Train Videos
Validation Videos
Test Videos
```

具体 15 个视频如何分配在第一次完整训练前确定，并固定记录，避免实验间随意变化。

### Action Leakage

Touch Marker 是标签来源，必须从模型输入中 Mask / Crop 掉。

同时需要关注已经发生动作产生的视觉特征，例如漂移烟雾、车辆姿态等，因此训练目标使用 Future Action：

```text
frames(..., t) → action(t + Δt)
```

而不是只学习：

```text
frame(t) → action(t)
```

---

## 6. Evaluation

普通分类指标保留：

- Accuracy
- Precision
- Recall
- F1

但它们不是主要评价标准。

核心指标：

### Transition Timing

分别计算预测和 GT 的：

- PRESS onset timing error
- RELEASE onset timing error
- Transition F1

### Short Correction Recall

按 RELEASE duration 分桶：

```text
100~200ms
200~300ms
>=300ms
```

对于 GT Short Correction，检查模型是否：

1. 成功进入 RELEASE
2. 在合理时间内重新 PRESS

`<100ms` 区域单独观察，不与 100~300ms 有效 Short Correction 混在一起，因为其中可能仍包含检测噪声。

### Closed-loop Evaluation

最终以 Replay 和实机为准：

- 过弯成功率
- 位置修正效果
- 完整赛程完成率
- 是否出现延迟或遗漏 Short Correction

---

## 7. 当前基线

第一版 Dataset / Model 基线固定为：

```text
Label extraction       : native video FPS
Glitch cleaning        : only isolated 1-frame segments
Valid short correction : preserve 100~300ms actions
Frame stack            : 3
History window         : ~100ms
Frame interval         : ~50ms
Prediction horizon     : 50~100ms experiment range
Runtime target         : 30Hz
Sampling               : transition-aware
Split                  : video-level
```

当前不做：

- 固定 minimum action duration
- `<100ms` 全量删除
- Frame-level random split
- 只按 PRESS/RELEASE 类别比例做简单 class balancing
