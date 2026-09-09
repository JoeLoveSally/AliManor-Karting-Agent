# AliManor Karting Agent System Design

## 1. 项目目标

AliManor Karting Agent 用于自动控制支付宝小鸡卡丁车小游戏。

游戏实际控制状态只有两种：

- `PRESS`：按住屏幕
- `RELEASE`：松开屏幕

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

系统分为离线训练和在线运行两条链路。

### Offline Train

```text
Raw Videos
    ↓
Label Extraction
    ↓
Dataset
    ↓
Trainer
    ↓
Evaluator
    ↓
Model Artifact
```

### Online Runtime

```text
                         RuntimeEngine
                              │
                              ▼
Input ───▶ Vision ───▶ Model ───▶ Control ───▶ Execute
  │                                              │
  ├── ADB                                        ├── ADB
  ├── Camera                                     └── bleOTG
  └── Video
```

核心原则：

- Train 与 Runtime 分离
- Input 与 Execute 可替换
- Control 与 Execute 分离
- Runtime 只负责编排和生命周期
- Logging / Replay 不参与核心控制决策
- 高频短时修正属于有效控制行为，不在 Runtime 中默认平滑掉

---

## 3. 核心模块

### Input

负责获取画面并输出统一 `Frame`。

- `ADB`：开发 / POC
- `Camera`：最终实时输入
- `Video`：历史视频与 Replay

Input 不负责图像预处理。

### Vision

负责将原始画面转换为模型输入：

```text
Raw Frame
    ↓
Perspective Correction
    ↓
ROI / Touch Marker Mask
    ↓
Resize / Normalize
    ↓
Temporal Frame Stack
    ↓
Model Input
```

Camera 模式需要透视校正；ADB 和原始录屏通常可以跳过。Train 与 Runtime 必须共享同一套核心预处理规则。

### Model

模型输出：

```text
turn_probability ∈ [0, 1]
```

不同地图车速不同，单帧无法可靠表达速度和运动趋势，因此第一版直接采用时序输入，而不是单帧 baseline 作为正式方案：

```text
frame(t-100ms)
frame(t-50ms)
frame(t)
      ↓
    CNN
      ↓
action(t + Δt)
```

初始设计：

- 3 个 RGB Frame
- History Window 约 `100 ms`
- Frame Interval 约 `50 ms`
- 通过 Channel Stack 输入 CNN
- `prediction_horizon_ms` 初始在 `50~100 ms` 范围实验确定
- 第一阶段模型候选：ResNet18、MobileNetV3-Small

3 帧是否足够仍需通过 Replay 和实机结果验证。后续可以对比更长 History Window、5-frame stack 或其他时序模型。

### Control

负责：

```text
turn_probability
    ↓
PRESS / RELEASE / HOLD
```

第一版使用双阈值 Hysteresis：

```text
P >= press_threshold   → PRESS
P <= release_threshold → RELEASE
otherwise              → HOLD
```

当前**不设置** `min_press_ms` / `min_release_ms`。已有数据表明 100~300ms 的短时 RELEASE 是常见的人工位置修正行为，不能通过 minimum duration 强行过滤。

如果后续出现输出抖动，优先从 Label Noise、模型预测、概率校准和 Hysteresis 阈值处理。

### Execute

负责把 ControlDecision 转换为真实设备操作。

统一语义：

```python
executor.set_pressed(True)
executor.set_pressed(False)
```

实现：

- `AdbExecutor`
- `BleOtgExecutor`

最终链路：

```text
BleOtgExecutor
    ↓
USB Serial
    ↓
bleOTG
    ↓
Bluetooth HID
    ↓
Android Phone
```

Control 不感知 ADB、Serial 或 Bluetooth 的具体实现。

---

## 4. Runtime 与状态机

RuntimeEngine 核心循环：

```python
while state == RUNNING:
    frame = input.read()
    model_input = vision.process(frame)
    prediction = model.predict(model_input)
    decision = controller.control(prediction)
    executor.execute(decision)
```

第一版 Runtime 目标控制频率为 **30 Hz**，主要原因是需要表达 100~300ms 的短时修正。30Hz 下 100ms 操作约有 3 个控制 Tick，20Hz 只有约 2 个。

状态机初始状态：

```text
IDLE
  ↓
READY
  ↓
RUNNING ───▶ ERROR
  │
  ▼
PAUSED
```

Camera 需要独立校准流程时增加 `CALIBRATING`。

### Safe Release

任何从 `RUNNING` 离开的软件路径都必须尝试执行 `RELEASE`，包括 pause、stop、error、shutdown 和 runtime exception。

PC 掉电、USB 断开或硬件自身失效属于硬件故障边界。

---

## 5. 数据与训练

当前 `data/raw` 包含 15 个游戏录屏，总时长约 668.72 秒。

当前触控统计：

- 完整 PRESS 段：301
- 完整 RELEASE 段：301
- `P-R-P` 中 RELEASE `<=300ms`：115 / 301，约 38.2%
- RELEASE `100~200ms`：40
- RELEASE `200~300ms`：62

因此 100~300ms 的 `PRESS → RELEASE → PRESS` 被定义为有效 **Short Correction**，属于模型需要重点学习的控制行为。

Label Pipeline：

```text
Raw Video (native FPS)
    ↓
Touch Marker Detection
    ↓
Raw Action Timeline
    ↓
1-frame Isolated Glitch Cleaning
    ↓
Training Action Timeline
```

原则：

- Label 必须先按原视频 FPS 提取，再生成训练采样
- 不使用固定 `<100ms` 阈值粗暴删除短动作
- 第一版只自动合并明确的 1-frame 孤立状态毛刺
- Touch Marker 只用于生成标签，不能直接暴露给模型输入
- Dataset 按完整 Video 划分 Train / Validation / Test，禁止 Frame-level random split
- Transition 和 Short Correction 区域需要提高采样权重

训练目标：

```text
frames(t-history ... t) → action(t + Δt)
```

重点不是总体 Frame Accuracy，而是：

- PRESS transition timing error
- RELEASE transition timing error
- Transition F1
- Short Correction Recall（100~200ms / 200~300ms / >=300ms）
- Replay / 实机过弯表现

更详细规则见 `dataset_design.md`。

---

## 6. Replay 与 Logging

### Replay

每次真实 Runtime Session 可保存：

```text
artifacts/replays/<session_id>/
├── metadata.json
├── video.mp4
└── events.jsonl
```

- `video.mp4`：Runtime 原始输入画面
- `events.jsonl`：Prediction、Control、Execute、Metrics 时间线
- `metadata.json`：模型、配置、运行模式、时间等信息

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

用于 Runtime Debug、Regression Test、Model Comparison、Controller Comparison 以及 Transition Timing 分析。

### Logging

Logging 与 Replay 独立，通过 `session_id` 关联。

日志主要记录 Runtime State、Input / Execute 生命周期、Model Load、PRESS / RELEASE、Hardware Error 和 Runtime Exception。

Metrics 主要记录：

```text
capture_ms
vision_ms
inference_ms
control_ms
execute_ms
loop_ms
fps
```

高频指标默认聚合为 mean / P50 / P95 / P99 / max，而不是逐帧输出 INFO 日志。

---

## 7. 配置与 Model Artifact

配置分为：

```text
hardware.yaml  # 物理设备和连接
execute.yaml   # 执行动作方式
runtime.yaml   # 实时运行参数
train.yaml     # 数据与训练参数
```

训练结果保存为：

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
```

Runtime 应读取 Model Artifact Metadata，不重复硬编码训练参数。

---

## 8. 开发顺序与系统约束

开发顺序：

```text
Dataset / Label Cleaning
    ↓
3-frame CNN Baseline
    ↓
Replay Runtime + Timing Evaluation
    ↓
ADB Closed Loop
    ↓
Camera Input
    ↓
bleOTG Execute
```

当前约束：

- 单进程、单 Input、单 Model、单 Controller、单 Executor、单实时控制 Loop
- Runtime 目标 30Hz
- Model 不直接调用 Execute
- Control 不感知具体执行设备
- Train 与 Runtime 共享视觉预处理
- 100~300ms Short Correction 必须保留并重点评估
- Controller v1 不设置 minimum state duration
- Dataset 按 Video 隔离
- 离开 `RUNNING` 必须尝试 `RELEASE`
- Replay 不依赖普通日志作为输入
- Logging / Metrics 不参与模型决策
