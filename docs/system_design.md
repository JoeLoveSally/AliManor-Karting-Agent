# AliManor Karting Agent System Design

## 1. 项目目标

AliManor Karting Agent 用于自动控制支付宝小鸡卡丁车小游戏。

游戏控制只有两种实际状态：

* `PRESS`：按住屏幕
* `RELEASE`：松开屏幕

最终系统通过外接摄像头获取手机游戏画面，视觉模型预测是否需要转向，由 Control 转换为稳定控制状态，再通过 bleOTG 控制手机。

```text id="jfm8dg"
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

```text id="zvu2xt"
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

```text id="03covh"
Input
    ↓
Vision
    ↓
Model
    ↓
Control
    ↓
Execute
```

`RuntimeEngine` 负责在线链路编排：

```text id="4a5hx8"
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

* Train 与 Runtime 分离
* Input 与 Execute 可替换
* Control 与 Execute 分离
* Runtime 只负责编排和生命周期
* Logging / Replay 不参与核心控制决策

---

## 3. 核心模块

### Input

负责获取画面并输出统一 `Frame`。

实现：

* `ADB`：开发 / POC
* `Camera`：最终实时输入
* `Video`：历史视频与 Replay

Input 不负责图像预处理。

### Vision

负责将原始画面转换为模型输入：

```text id="hqc00d"
Raw Frame
    ↓
Perspective Correction
    ↓
ROI / Mask
    ↓
Resize
    ↓
Normalize
    ↓
Model Input
```

Camera 模式需要透视校正；ADB 和原始录屏通常可以跳过。

Train 和 Runtime 必须共享相同的核心预处理逻辑。

### Model

负责预测：

```text id="5iiqv1"
turn_probability ∈ [0, 1]
```

模型不直接控制设备。

第一阶段候选：

* ResNet18
* MobileNetV3-Small

若单帧信息不足，再考虑 Frame Stack 或时序模型。

### Control

负责：

```text id="t2p7co"
turn_probability
    ↓
PRESS / RELEASE / HOLD
```

Controller 是有状态组件。

初始采用双阈值 Hysteresis：

```text id="yf3tlg"
P >= press_threshold
    → PRESS

P <= release_threshold
    → RELEASE

otherwise
    → HOLD
```

后续可加入：

```text id="y0pwe1"
min_press_ms
min_release_ms
```

避免高频抖动。

### Execute

负责把 ControlDecision 转换成真实设备操作。

统一语义：

```python id="8zf91p"
executor.set_pressed(True)
executor.set_pressed(False)
```

实现：

```text id="30m00t"
AdbExecutor
BleOtgExecutor
```

最终链路：

```text id="j4usrt"
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

```python id="rbamky"
while state == RUNNING:
    frame = input.read()
    model_input = vision.process(frame)
    prediction = model.predict(model_input)
    decision = controller.control(prediction)
    executor.execute(decision)
```

状态机初始状态：

```text id="vky9ba"
IDLE
  ↓
READY
  ↓
RUNNING ───▶ ERROR
  │
  ▼
PAUSED
```

Camera 需要独立校准流程时增加：

```text id="egswzk"
CALIBRATING
```

### Safe Release

系统强制约束：

> 任何从 `RUNNING` 离开的软件路径都必须尝试执行 `RELEASE`。

包括：

* pause
* stop
* error
* shutdown
* runtime exception

PC 掉电、USB 断开或硬件自身失效属于硬件故障边界。

---

## 5. 数据与训练

当前 `data/raw` 包含 **15 个游戏录屏**。

录屏中的触控标记用于自动生成玩家操作标签：

```text id="yz3o1j"
Raw Video
    ↓
Touch Marker Detection
    ↓
Action Timeline
```

Touch Marker 只用于生成标签，不能直接暴露给模型输入。

训练目标优先采用：

```text id="t30w22"
frame(t) → action(t + Δt)
```

而不是：

```text id="ygfeye"
frame(t) → action(t)
```

原因包括：

* Runtime 存在端到端控制延迟
* 当前画面可能已经包含动作产生的视觉结果
* 实时控制需要提前进入转向状态

`Δt` 根据模型效果和 Runtime 延迟实测确定。

数据必须按完整 Video 划分：

```text id="q4laoc"
Train Videos
Validation Videos
Test Videos
```

禁止随机按 Frame 划分，避免相邻帧导致 Data Leakage。

---

## 6. Replay 与 Logging

### Replay

Replay 用于：

* Runtime Debug
* Regression Test
* Model Comparison
* Controller Comparison

真实 Runtime Session 可以保存：

```text id="dq2xb9"
artifacts/replays/<session_id>/
├── metadata.json
├── video.mp4
└── events.jsonl
```

其中：

* `video.mp4`：Runtime 原始输入画面
* `events.jsonl`：Prediction、Control、Execute、Metrics 时间线
* `metadata.json`：模型、配置、运行模式、时间等信息

Replay 复用正式 Runtime：

```text id="21kfch"
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

不实现第二套推理链路。

### Logging

Logging 与 Replay 独立，通过 `session_id` 关联。

日志主要记录：

* Runtime State
* Input / Execute open / close
* Model load
* PRESS / RELEASE
* Hardware Error
* Runtime Exception

Metrics 主要记录：

```text id="hxsv8e"
capture_ms
vision_ms
inference_ms
control_ms
execute_ms
loop_ms
fps
```

高频指标默认进行聚合，例如：

```text id="izf8kn"
mean
P50
P95
P99
max
```

而不是逐帧打印 INFO 日志。

---

## 7. 配置

配置分为：

```text id="he9cbu"
hardware.yaml
execute.yaml
runtime.yaml
train.yaml
```

### hardware.yaml

负责物理设备和连接参数：

```yaml id="6ifs2g"
camera:
  device: 0
  width: 1920
  height: 1080
  fps: 60

ble_otg:
  port: /dev/ttyUSB0
  baudrate: 115200
```

### execute.yaml

负责执行行为：

```yaml id="t9c1q4"
type: ble_otg

ble_otg:
  touch_x: 500
  touch_y: 1200
```

边界：

```text id="iz15p7"
hardware.yaml = 怎么连接设备
execute.yaml  = 怎么使用设备执行动作
```

`runtime.yaml` 负责实时运行参数，`train.yaml` 负责训练参数。

---

## 8. Model Artifact

训练结果保存为：

```text id="504ptp"
artifacts/models/<model_name>/
├── model.pt
└── metadata.json
```

Metadata 至少记录：

```text id="3j9801"
architecture
input_size
frame_stack
prediction_horizon_ms
normalization
```

Runtime 应读取 Model Artifact Metadata，不重复硬编码训练参数。

---

## 9. 开发顺序

### Phase 1：Dataset

```text id="v45q2i"
15 Raw Videos
    ↓
Touch Marker Detection
    ↓
Dataset
```

### Phase 2：Model

```text id="j8dyqe"
Dataset
    ↓
CNN
    ↓
Offline Evaluation
```

### Phase 3：Replay Runtime

```text id="m9z8v3"
VideoInput
    ↓
RuntimeEngine
    ↓
Logging / Replay
```

### Phase 4：ADB POC

```text id="k0dtgf"
Real Game
    ↓
ADB / Camera Input
    ↓
Model
    ↓
ADB Execute
```

### Phase 5：Camera

替换为真实 Camera Input，重点解决：

* Perspective
* Exposure
* Reflection
* Moiré
* Domain Shift
* Latency

### Phase 6：bleOTG

```text id="l5nzx2"
ADB Execute
    ↓
BleOtgExecutor
```

形成最终硬件闭环。

---

## 10. 系统约束

当前版本确认：

* 单进程
* 单 Input
* 单 Model
* 单 Controller
* 单 Executor
* 单实时控制 Loop
* Model 不直接调用 Execute
* Control 不感知具体执行设备
* Train 与 Runtime 共享视觉预处理
* Dataset 按 Video 隔离
* 离开 `RUNNING` 必须尝试 `RELEASE`
* Replay 不依赖普通日志作为输入
* Logging / Metrics 不参与模型决策
