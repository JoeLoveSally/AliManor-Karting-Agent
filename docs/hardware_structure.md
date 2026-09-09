# AliManor Karting Agent Hardware Structure

## 1. 硬件目标

最终系统使用独立视觉输入和独立控制输出：

```text
             ┌────────────── PC ──────────────┐
             │                                 │
Phone Screen ──▶ Camera ──▶ Karting Agent     │
             │                    │            │
             │                    ▼            │
             │               USB Serial        │
             └────────────────────┼────────────┘
                                  ▼
                               bleOTG
                                  │
                           Bluetooth HID
                                  │
                                  ▼
                            Android Phone
```

最终运行不依赖手机内部截图和 ADB。ADB 仅用于开发与 POC。

---

## 2. 硬件组成

| 设备 | 作用 |
| --- | --- |
| Android Phone | 运行支付宝卡丁车小游戏 |
| External Camera | 获取手机屏幕实时画面 |
| PC | Vision、Model、Control 和 Runtime |
| bleOTG | 将 PC 控制命令转换为手机 HID 触控 |
| USB Connection | PC 与 Camera、bleOTG 的物理连接 |

Camera 和 bleOTG 分别构成 Visual Input Path 与 Control Output Path，两条链路相互独立。

---

## 3. Camera 链路

```text
Android Phone
     │
     │ screen image
     ▼
External Camera
     │
     │ USB / UVC
     ▼
PC
     │
     ▼
CameraInput
```

摄像头型号当前未确定。

选型重点：

- USB / UVC 兼容性
- 1080p 或更高分辨率
- 优先 60 FPS 或更高稳定帧率
- 低采集延迟
- 可锁定曝光和白平衡
- 对屏幕摩尔纹、PWM 和反光的表现
- 固定安装后的稳定性

Runtime 第一版目标控制频率为 **30 Hz**。Camera 原始采集帧率应高于控制频率，并为 3-frame temporal input 提供足够的时间采样余量。

模型不直接依赖摄像头原始坐标。Camera Input 获取原始画面后，由 Vision 进行：

```text
Camera Frame
    ↓
Perspective Correction
    ↓
Canonical Game Frame
```

摄像头和手机应尽量固定安装，减少运行过程中相对位置变化。

---

## 4. bleOTG 链路与协议

最终控制路径：

```text
PC
 │
 │ USB Serial
 ▼
bleOTG Controller
 │
 │ Bluetooth HID
 ▼
Android Phone
```

bleOTG PC 侧通过串口发送文本指令，手机侧以 HID 触控形式接收。

当前项目只需要：

```text
PRESS
RELEASE
```

参考协议：

```text
1,x,y          # press
2,x,y          # release
3,x,y          # click
10,width,height # screen size
```

项目 Runtime 统一暴露：

```python
set_pressed(True)
set_pressed(False)
```

由 `BleOtgExecutor` 转换为对应串口命令。

当前参考串口波特率为 `115200`，实际以实物和固件测试结果为准。

---

## 5. 硬件与软件边界

系统软件直接管理：

```text
Camera → Frame
bleOTG ← PRESS / RELEASE
```

内部边界：

```text
Camera Hardware
      ↓
CameraInput
      ↓
Vision
```

以及：

```text
Control
      ↓
BleOtgExecutor
      ↓
Serial
      ↓
bleOTG Hardware
```

以下内容不进入 Control：

- 串口号和波特率
- Bluetooth 配对状态
- HID 协议
- 摄像头设备号
- 摄像头曝光参数

---

## 6. 延迟与故障边界

端到端延迟：

```text
T_total
=
T_camera
+ T_capture
+ T_vision
+ T_model
+ T_control
+ T_serial
+ T_ble_otg
+ T_phone
```

软件可直接测量：

```text
capture_ms
vision_ms
inference_ms
control_ms
serial_write_ms
loop_ms
```

由于有效 Short Correction 大量集中在 100~300ms，硬件链路延迟必须显著低于该时间尺度，否则模型即使预测正确也可能错过控制窗口。

`Serial write completed` 不等价于 `Android touch applied`，因此 bleOTG 到手机实际触控的端到端延迟必须通过实机实验测量，并用于确定 `prediction_horizon_ms`。

任何正常软件停止流程都必须尝试 `RELEASE`。

以下情况属于 Hardware Failure Boundary：

- PC 突然掉电
- USB 连接断开
- bleOTG 掉电
- Bluetooth 连接中断
- 手机系统异常

---

## 7. 当前待确定项

- Camera 型号
- Camera 最终分辨率和 FPS
- Camera 固定位置与距离
- 手机与 Camera 安装结构
- bleOTG 实际串口设备名
- 固定触控坐标
- Camera 实际采集延迟
- bleOTG 实际端到端执行延迟
- 真实 Runtime 是否能稳定达到 30 Hz

具体连接参数记录于 `configs/hardware.yaml`，动作参数记录于 `configs/execute.yaml`。
