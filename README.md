# 基于 YOLOv8 与 K230 的 PCB 缺陷检测项目

项目成员：lyf、pl
## 项目简介

本项目围绕 PCB 板表面缺陷检测展开，完成了从数据集整理、数据增强、YOLOv8 模型训练、PC 端可视化检测，到 K230 / CanMV 边缘端部署的完整流程。项目目标是识别 PCB 图像中的 6 类常见缺陷，并将训练得到的模型部署到 K230 开发板上进行实时检测。

说明：GitHub 仓库版本主要保存项目代码、配置文件和已训练模型文件；完整图片数据集体积较大，未直接提交到仓库中。

支持识别的缺陷类型如下：

| 编号 | 英文类别 | 中文含义 |
| --- | --- | --- |
| 0 | `missing_hole` | 缺孔 |
| 1 | `mouse_bite` | 鼠咬 |
| 2 | `open_circuit` | 开路 |
| 3 | `short` | 短路 |
| 4 | `spur` | 毛刺 |
| 5 | `spurious_copper` | 多余铜 / 铜渣 |

整个项目的主要技术路线为：

```text
PCB 数据集
-> XML 标注整理
-> 亮度 / 噪声 / 旋转数据增强
-> 转换为 YOLOv8 数据格式
-> 训练 YOLOv8 目标检测模型
-> PC 端 PyQt5 可视化检测与评估
-> best.pt 导出 ONNX
-> ONNX 转换为 K230 可运行的 best.kmodel
-> K230 / CanMV 实时检测
```

## 项目目录结构

```text
.
├── 数据集_增广后/
│   ├── PCB_USED/                 # 原始 PCB 图片
│   ├── images/                   # 按缺陷类别整理后的原始样本
│   ├── Annotations/              # Pascal VOC XML 标注文件
│   ├── brightness/               # 亮度增强后的图片
│   ├── noise/                    # 噪声增强后的图片
│   ├── rotation/                 # 旋转增强后的图片和角度记录
│   └── README.md
│
├── yolov8/
│   ├── train.py                  # YOLOv8 训练脚本
│   ├── diagnose_pcb_model.py     # 模型诊断脚本
│   ├── yolov8n.pt                # YOLOv8n 预训练权重
│   ├── yolov8_pcb_dataset/       # YOLO 格式数据集
│   │   ├── pcb.yaml              # 数据集配置文件
│   │   ├── images/
│   │   └── labels/
│   ├── runs/                     # 训练输出目录
│   └── README.md
│
├── pc_pcb/
│   ├── code/MainProgram.py       # PC 端 PyQt5 检测系统
│   ├── PCB_DATASET/              # PC 端评估使用的数据集
│   ├── runs/best.pt              # 当前训练得到的最佳模型
│   ├── evaluation_results/       # 混淆矩阵等评估结果
│   └── save_data/                # 检测结果保存目录
│
├── K230/
│   └── MAX/
│       ├── det_video_yolov8.py   # K230 YOLOv8 实时检测脚本
│       ├── mp_deployment_source/
│       │   ├── best.kmodel       # K230 推理模型
│       │   └── deploy_config.json
│       ├── code/
│       │   ├── augment_noise_brightness.py
│       │   ├── test_det_onnx.py
│       │   ├── test_det_kmodel.py
│       │   └── to_kmodel.py
│       └── README.md
│
└── PCB缺陷检测汇报.pptx
```

## 数据集说明

原始数据位于 `数据集_增广后/`，其中 `images/` 保存按缺陷类别划分的原始图片，`Annotations/` 保存对应的 XML 标注文件。为了提高模型对光照变化、噪声干扰和角度变化的鲁棒性，项目中额外生成了亮度增强、噪声增强和旋转增强数据。

原始图片与标注数量：

| 类别 | 图片数量 | 标注数量 |
| --- | ---: | ---: |
| `Missing_hole` | 115 | 115 |
| `Mouse_bite` | 115 | 115 |
| `Open_circuit` | 116 | 116 |
| `Short` | 116 | 116 |
| `Spur` | 115 | 115 |
| `Spurious_copper` | 116 | 116 |
| **合计** | **693** | **693** |

增广数据规模：

| 增广方式 | 目录 | 图片数量 |
| --- | --- | ---: |
| 亮度增强 | `brightness/` | 693 |
| 噪声增强 | `noise/` | 693 |
| 旋转增强 | `rotation/` | 693 |

YOLOv8 训练使用的数据集位于 `yolov8/yolov8_pcb_dataset/`，已经划分为训练集、验证集和测试集：

| 子集 | 图片数量 | 标签数量 |
| --- | ---: | ---: |
| train | 2216 | 2216 |
| val | 276 | 276 |
| test | 280 | 280 |

数据集配置文件为 `yolov8/yolov8_pcb_dataset/pcb.yaml`：

```yaml
path: D:/数字图像/yolov8/yolov8_pcb_dataset
train: images/train
val: images/val
test: images/test

names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
```

如果在其他电脑上运行，需要把 `path` 改成自己本机的数据集绝对路径。

## 环境配置

建议使用 Python 3.10 或 Python 3.11。PC 端训练和检测主要依赖如下：

```powershell
pip install ultralytics torch torchvision torchaudio
pip install opencv-python numpy pillow matplotlib pyqt5
```

如果需要运行模型诊断脚本，还需要确保能够正常导入 `cv2`、`numpy` 和 `ultralytics`。

如果使用 NVIDIA GPU 训练，请先安装与本机 CUDA 版本匹配的 PyTorch。可以使用下面的命令检查 GPU 是否可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## YOLOv8 模型训练

进入 `yolov8/` 目录后运行：

```powershell
python train.py
```

训练脚本核心参数如下：

```python
model = YOLO("yolov8n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=200,
    imgsz=640,
    batch=32,
    device=0,
    patience=20,
    project=str(RUNS_DIR),
    name="yolov8n_pcb",
    optimizer="AdamW",
    seed=42,
    cos_lr=True,
    plots=True,
    workers=4,
)
```

参数说明：

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `model` | `yolov8n.pt` | 使用 YOLOv8 nano 预训练权重 |
| `epochs` | 200 | 最大训练轮数 |
| `imgsz` | 640 | 输入图片尺寸 |
| `batch` | 32 | 批大小 |
| `device` | 0 | 使用第 0 张 GPU |
| `patience` | 20 | 20 轮无提升则提前停止 |
| `optimizer` | `AdamW` | 优化器 |
| `cos_lr` | `True` | 使用余弦学习率 |

训练完成后，结果会保存在：

```text
yolov8/runs/yolov8n_pcb/
```

常见输出文件包括：

| 文件 | 说明 |
| --- | --- |
| `weights/best.pt` | 验证集表现最好的模型 |
| `weights/last.pt` | 最后一轮模型 |
| `results.png` | 训练指标曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `args.yaml` | 本次训练参数 |

当前 PC 端项目中使用的最佳模型位于：

```text
pc_pcb/runs/best.pt
```

## 模型预测

可以直接使用 Ultralytics 命令进行测试集预测：

```powershell
yolo detect predict model=runs/yolov8n_pcb/weights/best.pt source=yolov8_pcb_dataset/images/test
```

也可以指定当前项目中已经保存的模型：

```powershell
yolo detect predict model=../pc_pcb/runs/best.pt source=yolov8_pcb_dataset/images/test
```

预测结果默认会保存到 Ultralytics 自动生成的 `runs/detect/predict` 目录中。

## PC 端可视化检测系统

PC 端检测界面位于：

```text
pc_pcb/code/MainProgram.py
```

运行方式：

```powershell
cd pc_pcb
python code/MainProgram.py
```

该程序基于 PyQt5 实现，主要功能包括：

- 自动加载 `pc_pcb/runs/best.pt` 模型；
- 支持打开单张 PCB 图片并进行缺陷检测；
- 支持批量选择文件夹检测；
- 原图与检测结果并排显示；
- 支持图片缩放、拖拽和平移；
- 表格展示缺陷类别、英文名称、置信度；
- 点击表格中的缺陷项，可自动定位到对应检测框；
- 支持保存检测后的图片和检测信息；
- 支持模型评估并生成混淆矩阵。

检测类别在界面中会显示为中文：

| 英文类别 | 中文显示 |
| --- | --- |
| `missing_hole` | 缺失孔 |
| `mouse_bite` | 老鼠咬痕 |
| `open_circuit` | 开路 |
| `short` | 短路 |
| `spur` | 毛刺 |
| `spurious_copper` | 铜渣 |

评估结果会保存到：

```text
pc_pcb/evaluation_results/
```

目前该目录中包含：

```text
confusion_matrix.png
confusion_matrix_normalized.png
```

## 模型诊断

`yolov8/diagnose_pcb_model.py` 用于在不同数据来源和不同置信度阈值下统计模型表现，支持分别评估：

- 原始图片：`images`
- 亮度增强图片：`brightness`
- 噪声增强图片：`noise`
- 旋转增强图片：`rotation`

运行示例：

```powershell
cd yolov8
python diagnose_pcb_model.py
```

默认使用：

```text
模型: pc_pcb/runs/best.pt
数据集: pc_pcb/PCB_DATASET
输出: yolov8/diagnosis_results.csv
```

脚本会在多个置信度阈值下统计 `gt`、`tp`、`fp`、`fn`、`precision` 和 `recall`。当前诊断结果整体统计如下：

| 置信度阈值 | Precision | Recall | FP |
| ---: | ---: | ---: | ---: |
| 0.05 | 0.8785 | 0.9442 | 1543 |
| 0.10 | 0.9138 | 0.9377 | 1045 |
| 0.15 | 0.9291 | 0.9325 | 840 |
| 0.20 | 0.9471 | 0.9285 | 612 |
| 0.25 | 0.9570 | 0.9237 | 490 |
| 0.30 | 0.9687 | 0.9193 | 351 |
| 0.40 | 0.9784 | 0.9094 | 237 |
| 0.50 | 0.9851 | 0.8933 | 160 |

可以看到，置信度阈值越高，误检数量越少，Precision 越高，但 Recall 会有所下降。实际部署时需要根据业务需求权衡漏检和误检。

## K230 部署流程

K230 部署相关文件位于：

```text
K230/MAX/
```

本项目采用的模型转换路线为：

```text
best.pt -> best.onnx -> best.kmodel
```

没有直接使用在线训练平台生成 kmodel，主要原因是在线平台通常存在数据集压缩包大小、单文件大小和训练时间等限制。本项目选择先在本地训练 YOLOv8，再转换为 K230 可运行的 kmodel，便于使用更完整的数据集和更灵活的训练参数。

### 1. PT 导出 ONNX

训练得到 `best.pt` 后，使用 Ultralytics 导出 ONNX：

```powershell
yolo export model=best.pt format=onnx dynamic=False opset=12
```

关键参数：

| 参数 | 说明 |
| --- | --- |
| `dynamic=False` | 固定输入尺寸，方便后续转换 kmodel |
| `opset=12` | 使用较稳定的 ONNX 算子版本 |
| 输入尺寸 | 当前项目使用 `640x640` |

可以先用 ONNX Runtime 在 PC 上验证：

```powershell
python code/test_det_onnx.py --model best.onnx --image test.jpg --input_width 640 --input_height 640
```

### 2. ONNX 转 KModel

使用 `K230/MAX/code/to_kmodel.py` 将 ONNX 编译为 K230 可运行的 kmodel：

```powershell
python code/to_kmodel.py --target k230 --model best.onnx --dataset ../test --input_width 640 --input_height 640 --ptq_option 0
```

转换过程大致为：

```text
读取 ONNX
-> shape inference
-> onnxsim simplify
-> nncase import_onnx
-> 使用校准图片做 PTQ 量化
-> compile
-> 生成 best.kmodel
```

当前部署配置位于：

```text
K230/MAX/mp_deployment_source/deploy_config.json
```

核心配置如下：

```json
{
  "chip_type": "k230",
  "model_type": "YOLOv8",
  "img_size": [640, 640],
  "confidence_threshold": 0.4,
  "nms_threshold": 0.5,
  "kmodel_path": "best.kmodel",
  "num_classes": 6
}
```

转换完成后，可以先在 PC 上使用 nncase Simulator 验证：

```powershell
python code/test_det_kmodel.py --model best.kmodel --image test.jpg --input_width 640 --input_height 640
```

### 3. 部署到 K230

将以下文件复制到 K230 的 `/sdcard`：

```text
K230/MAX/det_video_yolov8.py
K230/MAX/mp_deployment_source/
```

K230 上最终路径应为：

```text
/sdcard/det_video_yolov8.py
/sdcard/mp_deployment_source/best.kmodel
/sdcard/mp_deployment_source/deploy_config.json
```

在 K230 / CanMV 环境中运行：

```python
python det_video_yolov8.py
```

程序启动后会调用摄像头进行实时检测，并在屏幕上绘制检测框、类别和置信度，同时显示 FPS。

## K230 部署中的关键问题

### 1. YOLOv8 不能直接套用旧的三输出后处理

很多 K230 示例中的 `det_video.py` 使用的是 AnchorBaseDet 风格后处理，默认模型有三个输出层：

```text
results[0], results[1], results[2]
```

而 YOLOv8 导出的 ONNX / kmodel 通常是单输出：

```text
output0
```

如果直接替换模型，很容易出现：

```text
IndexError: list index out of range
```

因此本项目单独编写了：

```text
K230/MAX/det_video_yolov8.py
```

该脚本针对 YOLOv8 单输出结构实现了解码和 NMS 后处理。

### 2. K230 实时速度优化

`det_video_yolov8.py` 中几个影响速度的参数：

```python
max_candidates = 80
max_detections = 20
gc_interval = 10
```

含义如下：

| 参数 | 说明 |
| --- | --- |
| `max_candidates` | NMS 前最多保留的候选框数量 |
| `max_detections` | 每帧最多显示的检测框数量 |
| `gc_interval` | 每隔多少帧执行一次垃圾回收 |

如果追求更高帧率，可以尝试：

```python
max_candidates = 40
max_detections = 10
gc_interval = 20
```

不过参数调得越小，目标较多时越可能漏检。更根本的提速方式是重新训练或导出更小输入尺寸的模型，例如 `416x416` 或 `320x320`，然后同步修改 `deploy_config.json` 中的 `img_size`。

## 常见问题

### 1. 训练时报 `Dataset images not found`

通常是 `pcb.yaml` 中的 `path` 不正确。需要确认它指向本机真实存在的数据集目录：

```yaml
path: D:/数字图像/yolov8/yolov8_pcb_dataset
```

如果项目移动到其他路径，需要手动修改该字段。

### 2. 显存不足

可以降低 `train.py` 中的 `batch`：

```python
batch=16
```

或者：

```python
batch=8
```

如果没有 GPU，可以改成：

```python
device="cpu"
```

### 3. PC 端程序无法加载模型

`MainProgram.py` 默认按下面顺序查找模型：

```text
pc_pcb/runs/best.pt
pc_pcb/best.pt
yolov8n.pt
```

建议把训练好的模型放在：

```text
pc_pcb/runs/best.pt
```

### 4. K230 上检测框位置不准

需要检查以下配置是否一致：

- 导出 ONNX 时的输入尺寸；
- 转换 kmodel 时的 `--input_width` 和 `--input_height`；
- `deploy_config.json` 中的 `img_size`；
- K230 摄像头输入分辨率；
- 后处理中的缩放和坐标映射逻辑。

当前项目默认模型输入尺寸为：

```text
640x640
```

### 5. K230 运行速度慢

可以优先检查：

- 是否开启了 debug 输出；
- 是否每帧打印过多日志；
- 是否每帧都执行 `gc.collect()`；
- `max_candidates` 和 `max_detections` 是否过大；
- 模型输入尺寸是否过大。

## 项目总结

本项目完成了 PCB 缺陷检测从算法训练到边缘端部署的完整闭环。PC 端部分能够完成模型训练、图像检测、批量检测和模型评估；K230 端部分完成了 YOLOv8 模型从 `.pt` 到 `.onnx` 再到 `.kmodel` 的转换，并针对 YOLOv8 单输出结构实现了可运行的实时检测脚本。

项目的核心价值在于：

- 使用 YOLOv8 完成 6 类 PCB 缺陷检测；
- 通过亮度、噪声、旋转增强提升模型鲁棒性；
- 提供 PyQt5 可视化检测系统，方便演示和结果分析；
- 提供模型诊断脚本，便于比较不同阈值下的 Precision 和 Recall；
- 打通 K230 部署流程，能够在边缘端进行实时检测；
- 解决 YOLOv8 单输出模型与 K230 示例三输出后处理不兼容的问题。

后续可以继续优化的方向：

- 增加更多真实工业场景数据；
- 提高小目标缺陷的标注质量；
- 尝试 YOLOv8s、YOLOv8m 或更轻量化模型进行对比；
- 重新训练低分辨率输入模型以提升 K230 FPS；
- 引入 INT8 量化校准集优化，提高 kmodel 精度；
- 增加检测结果导出为 Excel 或数据库记录的功能。
