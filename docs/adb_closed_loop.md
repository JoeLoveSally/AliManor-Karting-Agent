# ADB Closed-Loop POC

## 1. 目标

ADB Closed Loop 用于在接入 External Camera 与 bleOTG 之前，先验证真实 Android 手机上的闭环控制：

```text
Android Screen
    ↓ ADB screencap
AdbInput
    ↓
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

ADB POC 的价值是把“离线视频能预测”推进到“真实手机画面能够驱动真实触控”，同时暴露 Capture / Execute 延迟、画面差异和闭环累积误差。

---

## 2. 安全策略

真实触控必须显式启用。

默认运行模式只采集手机截图并执行模型与 Controller，输出动作使用 `MockExecutor`，不会触摸手机。

只有传入 `--arm` 时才创建 `AdbExecutor` 并发送真实触控事件。

Runtime 启动状态固定为 `RELEASE`。正常结束、Ctrl+C 和异常退出都必须调用 `RuntimeEngine.shutdown()`；如果 Controller 当时处于 `PRESS`，必须尝试发送 `UP`。

触控坐标不提供隐式默认值。首次实机测试必须显式通过 CLI 或 `configs/execute.yaml` 配置 `x/y`，避免在未知屏幕布局上发送触控。

---

## 3. ADB Input

第一版 `AdbInput` 使用：

```text
adb [-s SERIAL] exec-out screencap -p
```

每次调用获得一张 PNG，解码为 BGR `Frame`。

设计原则：

- 顺序采集，不缓存未来帧
- 每张截图完成后才进入 Runtime
- `Frame.timestamp_ms` 使用 Host Monotonic Clock 的“截图可用时刻”
- 单独记录每次 `capture_ms`
- Runtime 仍保持最大 30Hz，不在采集变慢后 burst 补算

ADB screencap 启动子进程和截图本身可能远慢于最终 UVC Camera，因此这一阶段的 Capture FPS / Capture Latency 只用于 POC，不代表最终系统性能。

由于 `Frame.timestamp_ms` 是截图完成时刻而不是屏幕像素曝光时刻，ADB POC 的真实视觉链路还包含截图生成延迟。该延迟必须单独观察，不能把 Model `inference_ms` 当成端到端延迟。

---

## 4. ADB Execute

第一版 `AdbExecutor` 使用 Android `input motionevent`：

```text
PRESS   → adb shell input motionevent DOWN x y
RELEASE → adb shell input motionevent UP   x y
```

要求目标 Android 的 `input` 命令支持 `motionevent`。首次连接设备时应先人工检查：

```bash
adb devices
adb shell wm size
adb shell input help
```

如果设备不支持独立的 `DOWN / UP`，不能用 `tap` 或固定时长 `swipe` 替代正式状态控制，因为游戏中 PRESS 持续时间由模型动态决定。

Executor 只负责把 `set_pressed(bool)` 转换为 ADB 命令，不拥有 Controller 策略。

---

## 5. 时序语义

模型训练语义保持不变：

```text
frame(t-100ms), frame(t-50ms), frame(t)
                 ↓
        predict action(t+100ms)
```

在真实 ADB POC 中，当前版本仍在模型推理完成后立即把 Controller 决策交给 Executor。因此物理触控相对预测目标时刻的近似关系为：

```text
T_apply - T_target
≈ T_capture_age + T_vision + T_model + T_adb_execute - prediction_horizon
```

其中 ADB screenshot 的像素年龄无法仅靠 Host Clock 精确测量，所以 ADB POC 重点验证闭环可用性，而不是据此最终标定 `prediction_horizon_ms`。

最终 Camera + bleOTG 链路需要重新测量端到端延迟。

---

## 6. 分阶段实机流程

阶段 A：设备与命令检查

```text
adb devices
wm size
input help
```

阶段 B：Dry Run

```text
ADB screenshot → Runtime → MockExecutor
```

观察：

- Capture FPS / Capture latency
- Inference latency
- probability
- PRESS / RELEASE 决策
- 手机画面方向和分辨率是否符合预期

阶段 C：Armed POC

```text
ADB screenshot → Runtime → AdbExecutor
```

第一次只运行数秒，并准备 Ctrl+C。确认触控坐标和 PRESS / RELEASE 语义正确后，再进行完整赛道闭环。

阶段 D：记录问题

重点记录：

- 是否能起跑
- 是否在正确位置转向
- 是否出现持续 PRESS / RELEASE 锁死
- Short Correction 是否实际生效
- Capture 是否成为控制频率瓶颈
- 闭环偏离后模型能否恢复

---

## 7. 通过标准

ADB Closed Loop 不以离线 F1 为通过标准。第一阶段通过条件是：

- Dry Run 能稳定持续采集和推理
- Runtime 在异常 / Ctrl+C 时安全 RELEASE
- Armed 模式能产生正确的 DOWN / UP
- 真实游戏能够完成至少一段连续控制而无失控锁死
- Capture / Inference / Execute 延迟均有可观测数据

如果 ADB screencap 明显限制控制频率，不继续围绕 ADB 做性能优化；直接进入 Camera Input，因为 ADB 只是 POC 链路。
