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

## 4. Sampling

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

按当前数据分布，期望采样构成大致从：

```text
Stable                 ~68.1%
Transition(non-short)  ~22.6%
Short Correction        ~9.3%
```

调整为：

```text
Stable                 ~48.2%
Transition(non-short)  ~32.0%
Short Correction       ~19.8%
```

这样高价值决策区域约占一半训练样本，同时仍保留足够 Stable 样本。

第一版不额外做 PRESS / RELEASE class balancing；原始 pressed ratio 约 55.6%，不存在明显类别失衡。

---

## 5. Split 与防泄漏

禁止 Frame-level random split。

必须按完整 Video 划分：

```text
Train Videos
Validation Videos
Test Videos
```

具体 15 个视频的 Split 在第一次正式训练前固定。

Touch Marker 只能用于 Label，必须从模型输入 Mask / Crop。

训练目标使用 Future Action：

```text
frames(..., t) → action(t + Δt)
```

避免模型只利用已经发生动作产生的视觉特征。

---

## 6. Evaluation

普通分类指标保留：

- Accuracy
- Precision
- Recall
- F1

核心指标：

- PRESS onset timing error
- RELEASE onset timing error
- Transition F1
- Short Correction Recall

Short Correction 按 RELEASE duration 分桶：

```text
100~200ms
200~300ms
>=300ms
```

最终评价以 Replay / 实机闭环为准：

- 过弯成功率
- 位置修正效果
- 完整赛程完成率
- Short Correction 是否延迟或遗漏

---

## 7. 当前基线

```text
Label extraction       : native video FPS
Glitch cleaning        : isolated 1-frame only
Valid short correction : preserve 100~300ms
Frame stack            : 3
History window         : 100ms
Frame interval         : 50ms
Sample rate            : 30Hz
Prediction horizon     : 100ms baseline
Transition window      : ±200ms
Sampling weights       : 1 / 2 / 3
Split                  : video-level
```

当前不做：

- minimum action duration
- `<100ms` 全量删除
- Frame-level random split
- Weighted Sampling 与 Weighted Loss 同时叠加
