# K230 YOLOv8 Detection Deployment

本项目用于在 K230 / CanMV 环境中部署 YOLOv8 目标检测模型。当前模型由本地训练得到的 YOLOv8 `.pt` 转换为 `.onnx`，再转换为 `.kmodel`，相比直接在勘智开发者社区官网训练得到的模型，检测效果更好，也能绕开官网训练平台对数据集压缩包大小和单文件大小的限制。

## 项目背景

最初使用勘智开发者社区官网训练 kmodel 模型，但官网训练对数据集上传限制较严格：

- 数据集需要打包为 `.zip`
- 压缩包大小需要小于 200 MB
- 单个文件大小需要小于 10 MB
- 图片数量越多，训练时间越长

这些限制不利于上传大量较高分辨率的原始图像。后续改为本地显卡训练 YOLOv8 模型，再走以下转换流程：

```text
YOLOv8 .pt -> ONNX .onnx -> K230 .kmodel
```

该路线已经验证可以在 K230 上运行，实际检测效果优于官网训练模型。

## 模型转换流程

本项目的模型不是直接在官网训练得到的 kmodel，而是先在本地显卡环境中训练 YOLOv8，得到 `best.pt`，再分两步转换：

```text
best.pt -> best.onnx -> best.kmodel
```

转换相关脚本保存在：

```text
code/
```

主要文件：

| 文件 | 说明 |
| --- | --- |
| `code/to_kmodel.py` | 使用 nncase 将 ONNX 编译为 K230 可运行的 kmodel |
| `code/test_det_onnx.py` | 在电脑端使用 onnxruntime 验证 ONNX 检测效果 |
| `code/test_det_kmodel.py` | 在电脑端使用 nncase Simulator 验证 kmodel 检测效果 |
| `code/augment_noise_brightness.py` | 对 PCB 数据集做噪声和亮度增强 |

### 1. 数据增强

为了提升模型对工业相机噪声、光照变化的鲁棒性，使用了噪声增强和亮度增强：

```text
高斯噪声 sigma=0.01~0.05
椒盐噪声 p=0.02
亮度随机变化 0.8~1.2
```

脚本：

```bash
python code/augment_noise_brightness.py
```

该脚本会按缺陷类别生成增强图片，输出结构与原始数据集类别目录保持一致。使用前需要根据本机数据集位置修改脚本中的 `base_dir`。

### 2. PT 导出 ONNX

本地训练 YOLOv8 得到 `best.pt` 后，使用 Ultralytics 导出 ONNX：

```bash
yolo export model=best.pt format=onnx dynamic=False opset=12
```

关键点：

- `dynamic=False`：导出固定输入尺寸，便于后续转换 kmodel
- `opset=12`：使用较稳定的 ONNX 算子版本
- 当前模型输入尺寸为 `640x640`
- YOLOv8 输出通常为单输出 `output0`

导出后可以用电脑端脚本验证 ONNX：

```bash
python code/test_det_onnx.py --model best.onnx --image test.jpg --input_width 640 --input_height 640
```

### 3. ONNX 转 KModel

ONNX 转 kmodel 使用 `code/to_kmodel.py`：

```bash
python code/to_kmodel.py --target k230 --model best.onnx --dataset ../test --input_width 640 --input_height 640 --ptq_option 0
```

当前转换脚本的主要处理流程：

```text
读取 ONNX
-> shape inference
-> onnxsim simplify
-> nncase import_onnx
-> 使用校准图片做 PTQ 量化
-> compile
-> 生成 best.kmodel
```

主要 nncase 配置：

```text
target: k230
preprocess: True
input_shape: [1, 3, 640, 640]
input_type: uint8
input_layout: NCHW
quant_type: uint8
calibrate_method: NoClip
PTQ samples_count: 5
```

`--dataset` 指向 PTQ 校准图片目录。脚本默认使用前 5 张图片做量化校准，因此该目录至少需要包含 5 张可读取图片。校准图片应尽量覆盖真实场景中的亮度、背景、缺陷类型和拍摄角度，否则量化后的 kmodel 效果可能下降。

`--ptq_option` 可选值：

| 参数 | 校准方式 | 权重量化 | 激活量化 |
| --- | --- | --- | --- |
| `0` | `NoClip` | `uint8` | `uint8` |
| `1` | `NoClip` | `int16` | 默认 |
| `2` | `NoClip` | 默认 | `int16` |
| `3` | `Kld` | `uint8` | `uint8` |
| `4` | `Kld` | `int16` | 默认 |
| `5` | `Kld` | 默认 | `int16` |

当前使用的是：

```text
ptq_option = 0
```

也就是 `NoClip + uint8` 量化。

生成 kmodel 后，可以先在电脑端用 nncase Simulator 验证：

```bash
python code/test_det_kmodel.py --model best.kmodel --image test.jpg --input_width 640 --input_height 640
```

确认 kmodel 在电脑端输出正常后，再复制到 K230 的 `/sdcard/mp_deployment_source/best.kmodel`。

## 目录结构

```text
.
├── code
│   ├── augment_noise_brightness.py
│   ├── test_det_kmodel.py
│   ├── test_det_onnx.py
│   └── to_kmodel.py
├── det_video_yolov8.py
├── mp_deployment_source
│   ├── best.kmodel
│   └── deploy_config.json
├── README.md
├── README.pdf
└── test.jpg
```

核心文件说明：

| 文件 | 说明 |
| --- | --- |
| `code/` | 数据增强、ONNX/KModel 转换和电脑端验证脚本 |
| `det_video_yolov8.py` | K230 视频实时检测脚本，适配 YOLOv8 单输出后处理 |
| `mp_deployment_source/best.kmodel` | 转换后的 K230 推理模型 |
| `mp_deployment_source/deploy_config.json` | 模型输入尺寸、类别、阈值等部署配置 |
| `README.pdf` | 原始参考说明文档 |
| `test.jpg` | 测试图片 |

## 当前模型配置

当前部署配置位于：

```text
mp_deployment_source/deploy_config.json
```

主要参数：

```json
{
    "model_type": "YOLOv8",
    "img_size": [640, 640],
    "confidence_threshold": 0.4,
    "nms_threshold": 0.5,
    "kmodel_path": "best.kmodel",
    "num_classes": 6
}
```

类别列表：

```text
Missing_hole
Mouse_bite
Open_circuit
Short
Spur
Spurious_copper
```

## 部署到 K230

将以下文件和目录复制到 K230 的 `/sdcard`：

```text
det_video_yolov8.py
mp_deployment_source/
```

K230 上的目标路径应为：

```text
/sdcard/det_video_yolov8.py
/sdcard/mp_deployment_source/best.kmodel
/sdcard/mp_deployment_source/deploy_config.json
```

然后在 K230 / CanMV 环境中运行：

```bash
python det_video_yolov8.py
```

运行后，屏幕左上角会显示实时 FPS。

## 为什么不能直接用原来的 det_video.py

原始 `det_video.py` 使用的是 Canaan 示例里的 `DetectionApp`，它会按 `AnchorBaseDet` 的三输出层方式进行后处理：

```text
results[0], results[1], results[2]
```

而 YOLOv8 导出的 ONNX / kmodel 通常是单输出：

```text
output0
```

因此直接把 YOLOv8 的 `best.kmodel` 替换旧模型，会出现类似错误：

```text
IndexError: list index out of range
```

当前 `det_video_yolov8.py` 已经改为 YOLOv8 单输出后处理，不再使用 `AnchorBaseDet` 后处理。

## 刷新率优化

`det_video_yolov8.py` 中有几个影响速度的参数：

```python
max_candidates = 80
max_detections = 20
gc_interval = 10
```

含义：

| 参数 | 说明 |
| --- | --- |
| `max_candidates` | NMS 前最多保留多少个候选框 |
| `max_detections` | 每帧最多显示多少个检测框 |
| `gc_interval` | 每隔多少帧执行一次垃圾回收 |

如果想进一步提升速度，可以尝试：

```python
max_candidates = 40
max_detections = 10
gc_interval = 20
```

注意：参数调得越小，速度可能越快，但在目标较多时可能出现漏检。

更大的速度提升通常需要重新导出更小输入尺寸的模型，例如：

```text
640x640 -> 416x416 或 320x320
```

然后同步修改 `deploy_config.json` 中的 `img_size`。

## 常见问题

### 1. `IndexError: list index out of range`

原因通常是模型输出结构和后处理方式不匹配。YOLOv8 不能直接使用 `AnchorBaseDet` 后处理。

解决方法：使用当前项目中的 `det_video_yolov8.py`。

### 2. `NameError: name 'ALIGN_UP' isn't defined`

部分 CanMV 固件环境没有将 `ALIGN_UP` 暴露到脚本作用域。当前脚本已经内置 fallback 实现，正常不会再出现该问题。

### 3. 运行速度慢

可优先检查：

- 是否打开了 `debug_mode = 1`
- 是否每帧都在打印输出
- 是否每帧都执行 `gc.collect()`
- `max_candidates` 是否设置过大
- 模型输入尺寸是否过大

当前脚本默认：

```python
debug_mode = 0
gc_interval = 10
```

### 4. 检测框位置不准

可能原因：

- ONNX 转 kmodel 时输入尺寸和 `deploy_config.json` 不一致
- 模型导出时使用的输入尺寸不是 `640x640`
- 摄像头输入分辨率和脚本中的 `rgb888p_size` 不一致

当前脚本默认摄像头输入：

```python
rgb888p_size = [1280, 720]
```

模型输入：

```python
model_input_size = [640, 640]
```

## 后续建议

如果需要继续提高 K230 实时检测性能，建议按优先级尝试：

1. 使用更小的 YOLOv8 模型，例如 `yolov8n`
2. 降低输入尺寸，例如 `320x320` 或 `416x416`
3. 减少检测类别或优化数据集标注质量
4. 调低 `max_candidates` 和 `max_detections`
5. 在转换 kmodel 时确认量化配置和输入尺寸正确
