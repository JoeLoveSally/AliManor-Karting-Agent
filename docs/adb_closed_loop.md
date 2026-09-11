# ADB Closed-Loop POC

## 1. 目标

ADB Closed Loop 用于在接入 External Camera 与 bleOTG 之前，先验证真实 Android 手机上的闭环控制。

实时 POC 主链路改为：

```text
Android Screen
    ↓ screenrecord raw H.264
AdbVideoInput
    ↓ FFmpeg decode + latest-frame queue
RuntimeEngine
    ↓
ModelRunner
    ↓
HysteresisController
    ↓
AdbExecutor
    ↓ ADB input motionevent
Android Touch
```

它是开发 / POC 链路，不是最终部署路径。最终仍使用：

```text
External Camera → Runtime → bleOTG → Android
```

旧的逐帧截图输入仍保留为诊断模式：

```text
adb exec-out screencap -p → AdbInput
```

实机已验证该路径只有约 1 FPS，因此不再作为实时闭环默认输入。

---

## 2. 为什么使用连续 H.264 流

逐帧 `screencap -p` 每次都需要生成 PNG 并完成一次独立 ADB 命令，Capture 延迟远大于模型推理延迟。

实时输入改用一个持续存在的 Android `screenrecord`：

```text
adb exec-out screenrecord --output-format=h264 ... -
    ↓
FFmpeg
    ↓ bgr24 rawvideo
background reader
    ↓
Queue(maxsize=1)
    ↓
Runtime
```

设计原则：

- `screenrecord` 与 FFmpeg 在一次运行内持续存在，不逐帧启动子进程
- Reader Thread 持续解码，不让模型推理阻塞编码器读取
- 队列只保留最新一帧；Runtime 来不及消费时主动丢弃旧帧
- 丢帧是低延迟策略，不是错误；闭环控制优先使用最新视觉状态
- Runtime 仍以 `target_fps` 限频，不对错过的 tick 做 burst catch-up
- 默认 H.264 解码宽度 720，保持设备宽高比并约束为偶数高度

`scrcpy` 仍可用于人工观察手机实时画面，但 Runtime 不依赖 scrcpy GUI、V4L2 或 scrcpy 内部协议。

---

## 3. 时间戳与延迟语义

`AdbVideoInput` 在 FFmpeg 输出一张完整 BGR Frame 时记录 Host Monotonic Clock：

```text
Frame.timestamp_ms = frame available to Runtime
```

它不是 Android 屏幕像素的真实曝光时间。因此：

```text
真实 frame age
= Android capture/encode
+ USB/ADB transport
+ FFmpeg decode
+ queue waiting
```

其中前三项无法仅从 Host timestamp 精确分离。

模型训练语义保持不变：

```text
frame(t-100ms), frame(t-50ms), frame(t)
                 ↓
        predict action(t+100ms)
```

ADB POC 主要验证闭环可用性与链路吞吐，不用它最终标定 `prediction_horizon_ms`。External Camera + bleOTG 接入后必须重新测量端到端视觉年龄和执行延迟。

---

## 4. 可观测指标

实时 Video Input 至少记录：

- `startup_ms`：启动 screenrecord / FFmpeg 到首帧可用的时间
- `decoded_frames`：FFmpeg 完整解码帧数
- `input_fps`：由连续 decoded frame interval 估算的输入帧率
- `frame_interval_ms`：decoded frame interval 的 mean / P50 / P95 / max
- `dropped_frames`：latest-frame queue 主动丢弃的旧帧数
- `control_fps`：Runtime 实际完成的推理 / 控制频率
- `inference_ms`：模型 mean / P50 / P95 / max

`dropped_frames > 0` 本身不代表失败。只要输入持续、Runtime 能稳定拿到新帧且 frame age 不持续积压，丢弃旧帧符合实时控制目标。

---

## 5. ADB Execute

`AdbExecutor` 将 Controller 状态变化映射为：

```text
PRESS   → adb shell input motionevent DOWN x y
RELEASE → adb shell input motionevent UP   x y
```

触控坐标不提供隐式默认值。只有 `--arm` 时才允许发送真实触控；默认 Dry Run 使用 `MockExecutor`。

第一轮实时 Video Input 验证仍沿用当前一次一条 ADB 命令的 Executor。只有 Video Dry Run 达到目标吞吐后，再把 Executor 改成 persistent ADB shell，避免把输入链路和执行链路的改动混在同一次实机验证中。

Runtime 启动状态固定为 `RELEASE`。正常结束、Ctrl+C 和异常退出都必须调用 `RuntimeEngine.shutdown()`；如果 Controller 当时处于 `PRESS`，必须尝试发送 `UP`。

---

## 6. 运行模式

默认实时模式：

```text
--input video
Android H.264 → Runtime → MockExecutor
```

保留诊断模式：

```text
--input screencap
ADB PNG screenshot → Runtime → MockExecutor
```

真实控制必须额外传入：

```text
--arm --x <X> --y <Y>
```

首次 Video Dry Run 不进入游戏也可以先在持续动态页面测吞吐；真正送入模型时必须关闭 Android Developer Options 中的“显示点按”和“指针位置”，避免动作相关视觉标记泄漏给模型。

---

## 7. 分阶段实机流程

阶段 A：Video Dry Run

```text
AdbVideoInput → Runtime → Model → MockExecutor
```

先验证连续流、Temporal Stack 与 CPU inference 能同时维持实时运行。

阶段 B：Persistent Execute

Video Dry Run 通过后，再把 ADB Execute 改成 persistent shell，并单独测量 PRESS / RELEASE 行为。

阶段 C：Armed Closed Loop

```text
AdbVideoInput → Runtime → Model → Controller → AdbExecutor → Android
```

第一次只运行数秒，并准备 Ctrl+C。确认触控坐标、DOWN / UP 语义和安全 RELEASE 后再进入完整赛道。

阶段 D：External Camera + bleOTG

ADB 闭环只用于开发验证。最终切换真实摄像头输入和 bleOTG 后重新做延迟标定。

---

## 8. 第一阶段通过标准

Video Dry Run 的目标值：

```text
input_fps    >= 30
control_fps  ≈ 30
infer_p95    < 15 ms
```

同时要求：

- 连续运行无 decoder / pipe 中断
- Temporal history 能稳定形成，不因 burst timestamps 失真
- Runtime 不积压旧帧
- Ctrl+C / 正常退出能够清理 screenrecord 与 FFmpeg 子进程
- 输出 JSON 包含输入、推理和 dropped-frame 指标

达到这些条件后才进入真实 ADB PRESS / RELEASE 优化与 Armed 闭环。
