# AliManor Karting Agent System Design

## 1. 项目目标

AliManor Karting Agent 用于自动控制支付宝小鸡卡丁车小游戏。

游戏只有两种实际控制状态：

- `PRESS`
- `RELEASE`

最终链路：

```text
Phone Screen
    ↓
Camera
    ↓
Input
    ↓
Vision
    ↓
Model
    ↓
Control
    ↓
Execute
    ↓
bleOTG
    ↓
Android Phone
```

开发阶段允许使用 ADB 和历史视频替代真实硬件。

---

## 2. 总体架构

系统分为离线训练与在线 Runtime：

```text
Raw Videos → Label → Dataset Manifest → Frame Cache → Trainer → Model Artifact
                                                     ↓
                                             Sequence Evaluator
```

```text
Input → Vision → Model → Control → Execute
           ↑        ↑
        RuntimeEngine orchestration
```

核心原则：

- Train 与 Runtime 分离
- Input 与 Execute 可替换
- Control 与 Execute 分离
- Train 与 Runtime 共享 Vision Preprocess
- Frame Cache 只是训练侧 Derived Data，不改变模型输入语义
- Sequence Evaluation 与训练期逐 sample 指标分离
- Replay 不依赖 Dataset Manifest、Frame Cache 或 Label
- Logging / Replay 不参与核心控制决策
- 高频 Short Correction 不默认平滑掉

---

## 3. 核心模块

### Input

统一输出 `Frame`：

```text
Frame
├── image          # BGR ndarray
├── frame_index
└── timestamp_ms
```

实现：

- ADB：开发 / POC
- Camera：最终实时输入
- Video：历史视频 / Replay

`VideoInput` 使用顺序 Decode，不依赖训练 Dataset 的 `input_frame_indices`，也不做 MP4 Random Seek。

### Vision

在线处理链路：

```text
Raw Frame
    ↓
Perspective Correction
    ↓
Fixed Touch Area Mask
    ↓
Resize / RGB
    ↓
Normalize
    ↓
Temporal Frame Stack
    ↓
Model Input
```

Touch Marker 是 Label 来源，但不能进入模型输入。

Mask 必须对 PRESS / RELEASE 所有帧无条件应用同一个固定区域。只在检测到 Touch Marker 时 Mask 会让“是否出现 Mask”本身成为标签泄漏。

第一版标准输入大小为 `224×224`。

训练侧 Frame Cache 会预计算其中确定性的 `Fixed Touch Area Mask + Resize + RGB`，以 `uint8` 保存；Normalize 和 Temporal Stack 仍在 Dataset 读取时执行。这样训练与 Runtime 保持同一预处理语义，同时避免每个 epoch 重复随机 seek MP4 和重复做空间预处理。

Runtime Temporal Buffer 保存最近的原始输入帧，并在推理时构建：

```text
t-100ms / t-50ms / t
```

Replay 中每个目标历史时刻只选择“已解码且时间戳不晚于该时刻”的最新帧，避免为了匹配训练样本而偷看未来帧。

### Model

第一版正式 baseline：

```text
frame(t-100ms)
frame(t-50ms)
frame(t)
      ↓
9-channel CNN
      ↓
PRESS logit / probability
```

参数：

- 3 个 RGB Frame
- History Window：100ms
- Frame Interval：50ms
- 主模型：MobileNetV3-Small
- 对照模型：ResNet18
- ImageNet pretrained weights
- `prediction_horizon_ms=100` 作为第一组实验值

预训练模型第一层从 3 channels 扩展为 9 channels；RGB filter 按 frame stack 重复并除以 3，以尽量保持原始激活尺度。

Runtime 不从 `train.yaml` 重建模型输入约束，而是从 Artifact `metadata.json` 读取：

- architecture
- frame_stack
- frame_interval_ms
- history_ms
- prediction_horizon_ms
- input_size
- preprocess / touch mask

3 帧是否足够通过 Ablation 和实机结果判断。

### Control

```text
turn_probability
    ↓
PRESS / RELEASE / HOLD
```

第一版使用 Hysteresis：

```text
当前 RELEASE 且 P >= press_threshold   → PRESS
当前 PRESS   且 P <= release_threshold → RELEASE
否则                                   → HOLD
```

Validation Threshold Sweep 后冻结第一版 Runtime 参数：

```text
press_threshold   = 0.55
release_threshold = 0.45
```

`0.45 < P < 0.55` 是 deadband，控制状态保持不变，用于抑制 raw probability 在 0.5 附近波动造成的额外 Transition。

本轮 Validation Sweep 的关键结果：

| Controller | ±100ms F1 | ±50ms F1 | ±50ms Recall | ±50ms Short Recall | Predicted Transition |
|---|---:|---:|---:|---:|---:|
| Raw 0.5 | 0.885 | 0.782 | 0.847 | 0.750 | 84 |
| 0.55 / 0.45 | 0.895 | 0.803 | 0.847 | 0.750 | 80 |
| 0.60 / 0.40 | 0.908 | 0.789 | 0.833 | 0.667 | 80 |
| 0.65 / 0.35 | 0.920 | 0.787 | 0.819 | 0.667 | 78 |
| 0.70 / 0.30 | 0.920 | 0.760 | 0.792 | 0.583 | 78 |

选择 `0.55 / 0.45` 的理由是：相对 Raw 0.5 减少额外 Transition，同时严格 `±50ms` 下不牺牲 Recall 和 Short Correction Recall。

当前不设置 `min_press_ms` / `min_release_ms`，避免删除真实 100~300ms 高频修正。

Runtime 启动时使用安全状态 `RELEASE`。离线 Controller-aware Evaluation 为避免把“视频从中间开始时的初始状态”误计为控制器性能，会仅对第一个 sample 用上下阈值中点推断初始状态，之后完全按同一 Hysteresis 状态机运行。

### Execute

统一语义：

```python
executor.set_pressed(True)
executor.set_pressed(False)
```

实现：

- `MockExecutor`：Replay / 自动测试，只记录状态变化，不产生外部副作用
- `AdbExecutor`：下一阶段开发 / POC
- `BleOtgExecutor`：最终硬件

最终：

```text
BleOtgExecutor → USB Serial → bleOTG → Bluetooth HID → Android
```

---

## 4. Runtime

第一版 `RuntimeEngine` 是单输入、单模型、单 Controller、单 Executor 的同步控制 Loop。

对于每个输入 Frame：

```text
Sequential Frame
    ↓ Temporal Buffer
Inference slot due?
    ↓ yes
Causal temporal stack
    ↓ Shared Vision Preprocess
Model.predict
    ↓ probability
HysteresisController.update
    ↓ PRESS / RELEASE / HOLD
Executor
```

第一版目标频率：**30Hz**。

Runtime 不做“漏掉几个 tick 后立即补跑多次推理”。当新 Frame 到达时，如果推理槽已到期，只对最新可用 Frame 推理一次，然后推进到下一个未来槽位，避免积压后 burst inference。

Model 的 `prediction_horizon_ms=100` 表示模型输出语义是预测 `t+100ms` 的目标动作。Runtime Step 同时记录：

```text
observation_timestamp_ms
prediction_target_timestamp_ms = observation_timestamp_ms + 100ms
```

第一版 Replay 的 Mock Execute 仍在当前 Runtime Step 立即应用 Controller 决策；因此 Replay 主要用于验证真实 Runtime 数据流、控制稳定性与性能，不把 Mock Execute 时间直接解释成最终物理触控时间。Camera / bleOTG 接入后需要实测端到端延迟，再决定是否调整 `prediction_horizon_ms`。

状态：

```text
IDLE → READY → RUNNING → PAUSED
                    └──→ ERROR
```

任何从 `RUNNING` 离开的软件路径都必须尝试 `RELEASE`。

---

## 5. 数据与训练

当前数据：

- 15 个录屏
- 总时长约 668.72 秒
- PRESS / RELEASE 完整段各 301 个
- RELEASE `<=300ms` 的 P-R-P 修正：115 / 301
- 100~300ms RELEASE：102 个

因此 100~300ms 定义为有效 **Short Correction**。

Label：

```text
Native FPS Video
    ↓
Touch Marker Detection
    ↓
Raw Action Timeline
    ↓
1-frame Isolated Glitch Cleaning
    ↓
Training Timeline
```

全量 Dataset Build：

```text
samples                 = 19,979
cleaned_frames          = 10
near_transition         = 6,381
near_short_correction   = 1,862
```

第一版 Weighted Sampling：

```text
Stable            1.0
Transition        2.0
Short Correction  3.0
```

不同时叠加 Weighted Loss。

### Frame Cache

本机 CPU benchmark 显示，直接从 MP4 随机 seek 3 帧时 DataLoader I/O 是主要瓶颈之一；GPU 训练时该问题会更明显。因此正式训练优先使用 Memory-mapped Frame Cache：

```text
samples.jsonl
    ↓
收集每个视频实际需要的 unique frame indices
    ↓
顺序 decode Raw MP4 一次
    ↓
Fixed Touch Mask + Resize + RGB
    ↓
data/processed/frame_cache/<video>/
    ├── frames.npy   # uint8, mmap
    └── index.json   # original frame index → cache row
```

原则：

- 只缓存当前 Dataset Manifest 实际使用的输入帧
- Cache 为 Derived Data，不提交 Git
- Cache 缺失时 Dataset 可回退 MP4，正式训练可要求 `require_cache`
- Input Size、Touch Mask 或 Dataset Temporal Sampling 改变后需要重建 Cache
- Mean / Std Normalization 改变不需要重建 Cache

### Split

第一版固定 Video-level Split：

```text
Train       11 videos / 14,678 samples
Validation   2 videos /  2,546 samples
Test         2 videos /  2,755 samples
```

策略：`video_holdout_visual_theme_stratified`。

验证和测试使用未见过的赛道，同时尽量保证其视觉主题在 Train 中有相近代表，先测新赛道几何上的控制泛化，再单独做 unseen-style stress test。

---

## 6. Replay 与 Evaluation

训练期 `Accuracy / Precision / Recall / F1 / transition_f1 / short_f1` 是逐 sample 分类指标，只用于观察优化过程。

正式离线 Sequence Evaluator 按每个完整 Video 的时间顺序推理，并使用 native-FPS Action Timeline 作为 Ground Truth。

### Raw classifier evaluation

```text
Ordered Model Probabilities
    ↓ threshold=0.5
Predicted PRESS/RELEASE States
    ↓
Predicted Transitions
    ↓ same direction + one-to-one + tolerance
Native Ground Truth Transitions
```

Raw classifier evaluation 用于直接观察模型输出本身，包括 probability chatter，不代表最终 Runtime Control 输出。

### Action persistence baseline

为了排除“模型只是识别当前动作并假设未来仍保持”的 shortcut，使用 label-only baseline：

```text
Ground Truth action(t)
        ↓ copy
Prediction action(t + 100ms)
```

Validation 结果：

```text
sample accuracy          = 0.915
Transition F1 @ ±100ms  = 0.042
Transition F1 @ ±50ms   = 0.000
Short Recall             = 0.000
```

CNN 明显优于该 persistence baseline，因此第一版 CNN 已表现出真实的 future-action prediction 信号，而不是仅靠 action persistence 获得高 Accuracy。

### Controller-aware evaluation

同一 ordered probability sequence 送入正式 Hysteresis Controller：

```text
Ordered Model Probabilities
    ↓ press_threshold=0.55 / release_threshold=0.45
Hysteresis Controller
    ↓ PRESS / RELEASE / HOLD
Controller State Sequence
    ↓
Predicted Transitions
    ↓ same direction + one-to-one + tolerance
Native Ground Truth Transitions
```

Controller-aware Evaluation 的目标不是提高 sample accuracy，而是验证 deadband 能否减少额外 Transition，同时尽量保持真正 Transition 和 100~300ms Short Correction。

重点比较：

- Transition predicted count / False Positive
- Transition Precision / Recall / F1
- PRESS onset timing error
- RELEASE onset timing error
- Short Correction Recall

Sequence Matching 第一版默认 `±100ms`，同时使用 `±50ms` 作为更严格的 timing stress check。

Short Correction Recall 要求一个 100~300ms Ground Truth RELEASE 的 `RELEASE onset` 与随后 `PRESS onset` 两个边界都被模型匹配到。额外抖动 Transition 会作为 False Positive 降低 Precision / F1。

### Replay Runtime

Replay 复用正式 Runtime 组件，但不依赖训练期数据结构：

```text
Recorded MP4
    ↓ sequential native-FPS decode
VideoInput
    ↓
TemporalFrameBuffer
    ↓ causal t-100 / t-50 / t selection
Shared Vision Preprocess
    ↓
ModelRunner
    ↓
HysteresisController
    ↓
MockExecutor
    ↓
Replay JSON
```

Replay 第一版原则：

- 不读取 `samples.jsonl`
- 不读取 Frame Cache
- 不读取 Touch Marker Label
- 原视频顺序 Decode
- 30Hz 为最大推理频率，不在掉帧后 burst 补算
- Temporal History 只使用当前时刻已经观察到的 Frame
- 使用 Artifact Metadata 恢复模型与预处理约束
- Controller 从安全状态 `RELEASE` 启动
- 视频结束或异常退出时尝试安全 `RELEASE`
- 默认尽快运行，不主动 sleep 到实时速度
- 每个 Step 记录 observation time、prediction target time、输入帧时间戳、probability、decision、controller state 和 inference latency

第一版 Replay 的目标是验证“脱离 Dataset / Cache 后正式 Runtime 数据流仍成立”。后续再加入 Ground Truth overlay / 视频可视化，以及 ADB 实机闭环。

最终核心闭环指标：

- Replay / 实机过弯成功率
- 位置修正效果
- 完整赛程完成率
- Short Correction 是否延迟或遗漏

---

## 7. 配置与 Artifact

配置：

```text
hardware.yaml
execute.yaml
runtime.yaml
train.yaml
```

`train.yaml` 记录 Frame Cache 开关和路径、Sequence Evaluation threshold / tolerance；`runtime.yaml` 记录 Runtime 频率和冻结后的 Hysteresis Controller 阈值。不同训练机可通过 CLI 覆盖 `num_workers`，避免把机器相关的吞吐参数写死到公共配置。

Model Artifact：

```text
artifacts/models/<model_name>/
├── model.pt
├── metadata.json
├── history.json
└── evaluation/
    ├── <split>_sequence.json
    └── <split>_hysteresis_sequence.json
```

Replay Artifact：

```text
artifacts/replays/<model_name>/<video_stem>.json
```

Metadata 至少记录：

```text
architecture
input_size
frame_stack
frame_interval_ms
history_ms
prediction_horizon_ms
normalization
touch_mask_roi
split
```

Runtime 从 Artifact Metadata 读取模型输入约束。

---

## 8. 当前约束与开发顺序

当前约束：

- 单进程
- 单 Input / Model / Controller / Executor
- 单实时控制 Loop
- Runtime 目标 30Hz
- Train 与 Runtime 使用同一 Vision Preprocess
- Touch Marker 区域始终固定 Mask
- 100~300ms Short Correction 保留
- Controller v1 无 minimum duration
- Controller v1 阈值冻结为 `0.55 / 0.45`
- Dataset 按 Video 隔离
- 正式训练优先使用 Frame Cache，避免 MP4 Random Seek 成为 GPU 饥饿瓶颈
- Sequence Evaluation 按 Video 顺序运行，不使用 Weighted Sampler
- Replay 不读取 Dataset / Frame Cache / Label
- Test Split 不用于日常超参数迭代
- 离开 RUNNING 尝试 RELEASE

开发顺序：

```text
Dataset / Label
    ↓
Temporal Preprocess + CNN
    ↓
Video-level Split
    ↓
Frame Cache + Trainer
    ↓
GPU Baseline Training
    ↓
Raw Sequence Evaluator + Persistence Baseline
    ↓
Controller-aware Hysteresis Evaluation
    ↓
Replay Runtime
    ↓
ADB Closed Loop
    ↓
Camera
    ↓
bleOTG
```
