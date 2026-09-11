# ADB Closed-Loop POC

## 1. 目标

ADB Closed Loop 用于在接入 External Camera 与 bleOTG 之前，先验证真实 Android 手机上的闭环控制。

实时 POC 主链路：

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
    ↓ persistent ADB shell + input motionevent
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
- 默认 H.264 解码宽度为 360，保持设备宽高比并约束为偶数高度

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

Runtime benchmark 的 `elapsed` 只计算实际控制循环，不包含 `RuntimeEngine.shutdown()`、`screenrecord` / FFmpeg 终止和 persistent ADB shell 清理时间。Video counters 也在进入 shutdown 前截断，避免清理阶段污染 FPS / dropped-frame 指标。

---

## 5. ADB Execute

`AdbExecutor` 将 Controller 状态变化映射为：

```text
PRESS   → input motionevent DOWN x y
RELEASE → input motionevent UP   x y
```

Armed 模式启动一个持续存在的：

```text
adb shell
```

之后通过 stdin 写入 PRESS / RELEASE 命令，避免每个 100~300ms Short Correction 都重新启动 ADB 子进程。

触控坐标不提供隐式默认值。只有 `--arm` 时才允许发送真实触控；默认 Dry Run 使用 `MockExecutor`。

`execute_latencies_ms` 当前测量的是 Host 向 persistent shell 写入并 `flush()` 的 enqueue latency，不等价于 Android 物理触控生效延迟。真实触控延迟需要通过 Developer Options 指针显示或后续外部观测单独测量。

Runtime 启动状态固定为 `RELEASE`。正常结束、Ctrl+C 和异常退出都必须调用 `RuntimeEngine.shutdown()`；如果 Controller 当时处于 `PRESS`，必须先尝试发送 `UP`，再关闭 persistent shell。

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

## 7. 实机性能结果

设备：V2227A，1440×3200；本地模型使用 CPU；Runtime 目标 30 Hz。

10 秒 Dry Run A/B：

| Decode | Decoded FPS | Frame P95 | Steps / 10s | Effective Control FPS | Inference Mean | Inference P95 | Dropped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 720×1600 | 56.2 | 30.6 ms | 290 | 29.0 | 11.99 ms | 19.90 ms | 101 |
| 360×800 | 56.1 | 27.6 ms | 298 | 29.8 | 7.93 ms | 11.89 ms | 4 |

测试时旧脚本把约 3 秒 shutdown 等待计入 `elapsed`，因此当时 Summary 显示的 `control_fps=22.2/22.9` 偏低；控制循环本身固定运行 10 秒，实际完成 290 / 298 个 step。该统计问题现已修复。

360 宽模式明显降低 FFmpeg raw BGR pipe、NumPy copy 和 Runtime CPU / memory bandwidth 压力，同时仍高于模型最终 224×224 输入分辨率，因此作为 ADB POC 默认值。

当前 Video Dry Run 已满足进入 Persistent Execute 阶段的条件：

```text
input_fps    > 30
control_fps  ≈ 30
infer_p95    < 15 ms   # 360-wide mode
```

---

## 8. 分阶段实机流程

阶段 A：Video Dry Run

```text
AdbVideoInput → Runtime → Model → MockExecutor
```

已通过：连续 H.264 输入与 CPU inference 可同时维持约 30 Hz 控制频率。

阶段 B：Persistent Execute

```text
AdbVideoInput → Runtime → Model → Controller → persistent ADB shell
```

代码已接入。下一步先用安全坐标或 Developer Options 指针显示验证 DOWN / UP 与安全 RELEASE，再进入游戏闭环。

阶段 C：Armed Closed Loop

```text
AdbVideoInput → Runtime → Model → Controller → AdbExecutor → Android
```

第一次只运行数秒，并准备 Ctrl+C。确认触控坐标、DOWN / UP 语义和安全 RELEASE 后再进入完整赛道。

阶段 D：External Camera + bleOTG

ADB 闭环只用于开发验证。最终切换真实摄像头输入和 bleOTG 后重新做延迟标定。

---

## 9. 第一阶段通过标准

Video Dry Run 的目标值：

```text
input_fps    >= 30
control_fps  ≈ 30
infer_p95    < 15 ms
```

360-wide A/B 结果已满足该目标。

同时要求：

- 连续运行无 decoder / pipe 中断
- Temporal history 能稳定形成，不因 burst timestamps 失真
- Runtime 不积压旧帧
- Ctrl+C / 正常退出能够清理 screenrecord 与 FFmpeg 子进程
- 输出 JSON 包含输入、推理和 dropped-frame 指标

下一验收点是 persistent ADB PRESS / RELEASE 的真机行为与 Armed 闭环。