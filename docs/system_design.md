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
Raw Videos → Label → Dataset → Trainer → Model Artifact
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
- Logging / Replay 不参与核心控制决策
- 高频 Short Correction 不默认平滑掉

---

## 3. 核心模块

### Input

统一输出 `Frame`：

- ADB：开发 / POC
- Camera：最终实时输入
- Video：历史视频 / Replay

### Vision

处理链路：

```text
Raw Frame
    ↓
Perspective Correction
    ↓
Fixed Touch Area Mask
    ↓
Resize / Normalize
    ↓
Temporal Frame Stack
    ↓
Model Input
```

Touch Marker 是 Label 来源，但不能进入模型输入。

Mask 必须对 PRESS / RELEASE 所有帧无条件应用同一个固定区域。只在检测到 Touch Marker 时 Mask 会让“是否出现 Mask”本身成为标签泄漏。

第一版标准输入大小为 `224×224`。

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

3 帧是否足够通过 Ablation 和实机结果判断。

### Control

```text
turn_probability
    ↓
PRESS / RELEASE / HOLD
```

第一版使用 Hysteresis：

```text
P >= press_threshold   → PRESS
P <= release_threshold → RELEASE
otherwise              → HOLD
```

当前不设置 `min_press_ms` / `min_release_ms`，避免删除真实 100~300ms 高频修正。

### Execute

统一语义：

```python
executor.set_pressed(True)
executor.set_pressed(False)
```

实现：

- `AdbExecutor`
- `BleOtgExecutor`

最终：

```text
BleOtgExecutor → USB Serial → bleOTG → Bluetooth HID → Android
```

---

## 4. Runtime

RuntimeEngine 核心循环：

```python
while state == RUNNING:
    frame = input.read()
    model_input = vision.process(frame)
    prediction = model.predict(model_input)
    decision = controller.control(prediction)
    executor.execute(decision)
```

第一版目标频率：**30Hz**。

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

Dataset 必须按完整 Video 划分 Train / Validation / Test，禁止 Frame-level random split。

---

## 6. Replay 与 Evaluation

Replay 复用正式 Runtime：

```text
Recorded Video
    ↓
VideoInput
    ↓
Vision
    ↓
Model
    ↓
Control
    ↓
No-op / Mock Execute
```

核心评价指标：

- PRESS onset timing error
- RELEASE onset timing error
- Transition F1
- Short Correction Recall
- Replay / 实机过弯成功率
- 完整赛程完成率

普通 Accuracy / Precision / Recall / F1 仅作为辅助指标。

---

## 7. 配置与 Artifact

配置：

```text
hardware.yaml
execute.yaml
runtime.yaml
train.yaml
```

Model Artifact：

```text
artifacts/models/<model_name>/
├── model.pt
└── metadata.json
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
- Dataset 按 Video 隔离
- 离开 RUNNING 尝试 RELEASE

开发顺序：

```text
Dataset / Label
    ↓
Temporal Preprocess + CNN
    ↓
Video-level Split
    ↓
Trainer / Evaluator
    ↓
Replay Runtime
    ↓
ADB Closed Loop
    ↓
Camera
    ↓
bleOTG
```
