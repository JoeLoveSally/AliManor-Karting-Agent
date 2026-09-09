# AliManor Karting Agent Hardware Structure

## 1. 硬件目标

最终系统使用独立视觉输入和独立控制输出：

```text id="jhwll9"
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

系统不依赖手机内部截图和 ADB 完成最终运行。

ADB 仅作为开发与 POC 方案。

---

## 2. 硬件组成

最终系统主要包含：

| 设备              | 作用                             |
| --------------- | ------------------------------ |
| Android Phone   | 运行支付宝卡丁车小游戏                    |
| External Camera | 获取手机屏幕实时画面                     |
| PC              | Vision、Model、Control 和 Runtime |
| bleOTG          | 将 PC 控制命令转换为手机 HID 触控          |
| USB Connection  | PC 与 Camera、bleOTG 的物理连接       |

其中 Camera 和 bleOTG 分别构成：

```text id="n20dzc"
Visual Input Path
```

和：

```text id="v7g31v"
Control Output Path
```

两条链路相互独立。

---

## 3. Camera 链路

```text id="7wg1jl"
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

选型主要关注：

* USB / UVC 兼容性
* 1080p 或更高分辨率
* 30 FPS 以上，优先 60 FPS
* 较低采集延迟
* 可锁定曝光和白平衡
* 对屏幕摩尔纹、PWM 和反光的表现
* 固定安装后的稳定性

模型不直接依赖摄像头原始坐标。

Camera Input 获取原始画面后，由 Vision 进行：

```text id="33l7k6"
Camera Frame
    ↓
Perspective Correction
    ↓
Canonical Game Frame
```

因此摄像头和手机应尽量采用固定支架安装，减少运行过程中手机与相机的相对移动。

---

## 4. bleOTG 链路

最终控制路径：

```text id="z7sx4v"
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

bleOTG 官方示例要求主控板通过 USB 接入电脑，并通过 CH340 USB-to-Serial 与 PC 通信；手机再与 bleOTG 蓝牙设备配对。

PC 侧参考实现使用 Python `serial` 打开串口，示例波特率为：

```text id="sy4p82"
115200
```

并在发送后执行 `flush()`。

因此项目侧的：

```text id="yxt9yy"
BleOtgExecutor
```

直接将 bleOTG 视为一个串口控制设备，不需要感知其内部 Bluetooth HID 实现。

---

## 5. bleOTG 控制协议

当前项目只需要：

```text id="a5ulpa"
PRESS
RELEASE
```

不需要使用 bleOTG 的完整触控能力。

参考实现采用换行分隔的文本指令，并定义：

```text id="2uioou"
1,x,y
```

表示在 `(x, y)` 按下，

```text id="p9lfsq"
2,x,y
```

表示松开，

```text id="333bu5"
3,x,y
```

表示一次完整点击，

```text id="ty0yp4"
10,width,height
```

用于设置屏幕尺寸。

本项目 Runtime 只对外暴露：

```python id="zhn3pw"
set_pressed(True)
set_pressed(False)
```

由 `BleOtgExecutor` 转换成对应的 bleOTG 指令。

例如：

```text id="m4j7q1"
set_pressed(True)
        ↓
1,x,y\n
```

```text id="lrgn0h"
set_pressed(False)
        ↓
2,x,y\n
```

具体触控坐标由 `execute.yaml` 配置。

---

## 6. 硬件与软件边界

系统软件只直接管理两类硬件接口：

```text id="iu53ol"
Camera
    → Frame

bleOTG
    ← PRESS / RELEASE
```

内部边界为：

```text id="23kyjh"
Camera Hardware
      ↓
CameraInput
      ↓
Vision
```

以及：

```text id="73m3ss"
Control
      ↓
BleOtgExecutor
      ↓
Serial
      ↓
bleOTG Hardware
```

以下内容不进入 Control：

* 串口号
* 波特率
* Bluetooth 配对状态
* HID 协议
* 摄像头设备号
* 摄像头曝光参数

---

## 7. 配置边界

硬件连接参数放在：

```text id="saxuy3"
configs/hardware.yaml
```

例如：

```yaml id="09c4w1"
camera:
  device: 0
  width: 1920
  height: 1080
  fps: 60

ble_otg:
  port: /dev/ttyUSB0
  baudrate: 115200
```

动作相关配置放在：

```text id="hb0g9l"
configs/execute.yaml
```

例如：

```yaml id="xiy2oj"
type: ble_otg

ble_otg:
  touch_x: 500
  touch_y: 1200
```

即：

```text id="we867h"
hardware.yaml
= Hardware Connection

execute.yaml
= Control Usage
```

---

## 8. 延迟与故障边界

最终端到端延迟：

```text id="w13fwu"
T_total
=
T_camera
+
T_capture
+
T_vision
+
T_model
+
T_control
+
T_serial
+
T_ble_otg
+
T_phone
```

其中软件可以直接测量：

```text id="rhrlg7"
capture_ms
vision_ms
inference_ms
control_ms
serial_write_ms
loop_ms
```

但：

```text id="6t077u"
Serial write completed
```

并不等价于：

```text id="bnn65p"
Android touch applied
```

因此 bleOTG 到手机实际触控的端到端延迟需要通过实机实验测量。

任何正常的软件停止流程都必须尝试：

```text id="4rpx1e"
RELEASE
```

但以下故障无法依靠 Runtime 完全保证恢复：

* PC 突然掉电
* USB 连接断开
* bleOTG 掉电
* Bluetooth 连接中断
* 手机系统异常

这些情况属于 Hardware Failure Boundary。

---

## 9. 当前待确定项

以下硬件参数在实物测试前保持未定：

* Camera 型号
* Camera 最终分辨率和 FPS
* Camera 固定位置与距离
* 手机与 Camera 安装结构
* bleOTG 实际串口设备名
* 固定触控坐标
* Camera 实际采集延迟
* bleOTG 实际端到端执行延迟

bleOTG 仓库同时说明其主控板固件针对作者提供的硬件设计，因此实际使用时应以对应硬件和固件组合为准。
