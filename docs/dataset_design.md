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

普通分类指标保留：

- Accuracy
- Precision
- Recall
- F1
- GT PRESS ratio
- Predicted PRESS ratio

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
Transition window      : ±200ms
Sampling weights       : 1 / 2 / 3
Split                  : v1 video-level holdout
Training input         : mmap frame cache preferred
Cache representation   : prepared RGB uint8
```

当前不做：

- minimum action duration
- `<100ms` 全量删除
- Frame-level random split
- Weighted Sampling 与 Weighted Loss 同时叠加
- 把相似视觉主题直接当作同一地图
