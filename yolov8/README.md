# YOLOv8 PCB Defect Detection

本项目使用 Ultralytics YOLOv8 对 PCB 缺陷数据集进行目标检测训练，支持识别 6 类常见 PCB 缺陷。

## 项目结构

```text
yolov8/
├── train.py
├── yolov8n.pt
├── yolo26n.pt
├── yolov8_pcb_dataset/
│   ├── pcb.yaml
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── runs/
```

## 数据集说明

数据集配置文件位于：

```text
yolov8_pcb_dataset/pcb.yaml
```

当前配置如下：

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

类别含义：

| 编号 | 类别名 | 含义 |
| --- | --- | --- |
| 0 | missing_hole | 缺孔 |
| 1 | mouse_bite | 鼠咬 |
| 2 | open_circuit | 断路 |
| 3 | short | 短路 |
| 4 | spur | 毛刺 |
| 5 | spurious_copper | 多余铜 |

## 环境要求

建议使用 Python 3.11，并安装以下依赖：

```powershell
pip install ultralytics torch torchvision torchaudio
```

如果需要使用 NVIDIA GPU 训练，请确保已经正确安装 CUDA 版本对应的 PyTorch。

可以用下面的命令检查 GPU 是否可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 开始训练

在项目根目录运行：

```powershell
python train.py
```

`train.py` 中的主要训练参数：

```python
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
| `epochs` | `200` | 最大训练轮数 |
| `imgsz` | `640` | 输入图像尺寸 |
| `batch` | `32` | 批大小 |
| `device` | `0` | 使用第 0 张 GPU |
| `patience` | `20` | 20 轮无提升则提前停止 |
| `optimizer` | `AdamW` | 优化器 |
| `cos_lr` | `True` | 使用余弦学习率调度 |

## 训练结果

训练结果会保存到：

```text
runs/yolov8n_pcb/
```

常见输出文件包括：

| 文件 | 说明 |
| --- | --- |
| `weights/best.pt` | 验证集表现最好的模型 |
| `weights/last.pt` | 最后一轮模型 |
| `results.png` | 训练指标曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `args.yaml` | 本次训练参数 |

## 使用模型预测

训练完成后，可以使用最佳权重进行预测：

```powershell
yolo detect predict model=runs/yolov8n_pcb/weights/best.pt source=yolov8_pcb_dataset/images/test
```

预测结果默认保存到 Ultralytics 自动生成的 `runs/detect/predict` 目录中。

## 常见问题

### 1. 报错 `images not found`

如果出现类似错误：

```text
Dataset images not found, missing path ...
```

通常是 `yolov8_pcb_dataset/pcb.yaml` 里的 `path` 写错了。请确认 `path` 指向当前数据集目录，例如：

```yaml
path: D:/数字图像/yolov8/yolov8_pcb_dataset
```

并确认下面这些目录真实存在：

```text
yolov8_pcb_dataset/images/train
yolov8_pcb_dataset/images/val
yolov8_pcb_dataset/images/test
yolov8_pcb_dataset/labels/train
yolov8_pcb_dataset/labels/val
yolov8_pcb_dataset/labels/test
```

### 2. 提示 Ultralytics 有新版本

如果运行时出现：

```text
New https://pypi.org/project/ultralytics/... available
```

这只是版本更新提示，不影响训练。需要更新时可以运行：

```powershell
pip install -U ultralytics
```

### 3. 显存不足

如果出现 CUDA out of memory，可以降低 `train.py` 中的 `batch`，例如：

```python
batch=16
```

或者：

```python
batch=8
```

## 备注

当前项目默认使用 GPU 训练：

```python
device=0
```

如果没有可用 GPU，可以改为 CPU：

```python
device="cpu"
```
