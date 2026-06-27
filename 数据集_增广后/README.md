# PCB 缺陷数据集（增广后）

本数据集用于 PCB 缺陷检测/识别任务。`PCB_USED` 为原始数据集图片，`Annotations` 为标注文件，其他目录为基于原始数据通过不同方式生成的增广数据。

## 目录结构

```text
.
├── PCB_USED/          # 原始 PCB 图片
├── images/            # 原始样本图片，按缺陷类别划分
├── Annotations/       # 标注文件，按缺陷类别划分，XML 格式
├── brightness/        # 亮度增广后的图片
├── noise/             # 噪声增广后的图片
└── rotation/          # 旋转增广后的图片，以及旋转角度记录文件
```

## 数据类别

数据集包含 6 类 PCB 缺陷：

| 类别目录名 | 含义 |
| --- | --- |
| `Missing_hole` | 缺孔 |
| `Mouse_bite` | 鼠咬 |
| `Open_circuit` | 开路 |
| `Short` | 短路 |
| `Spur` | 毛刺 |
| `Spurious_copper` | 多余铜 |

## 数据统计

### 原始图片与标注

| 类别 | 图片数量 | 标注数量 |
| --- | ---: | ---: |
| `Missing_hole` | 115 | 115 |
| `Mouse_bite` | 115 | 115 |
| `Open_circuit` | 116 | 116 |
| `Short` | 116 | 116 |
| `Spur` | 115 | 115 |
| `Spurious_copper` | 116 | 116 |
| **合计** | **693** | **693** |

### 增广数据

| 增广方式 | 目录 | 图片数量 |
| --- | --- | ---: |
| 亮度增广 | `brightness/` | 693 |
| 噪声增广 | `noise/` | 693 |
| 旋转增广 | `rotation/` | 693 |

说明：`rotation/` 目录下还包含各类别对应的旋转角度记录文件，例如 `Missing_hole_angles.txt`，这些文件用于记录旋转增广参数，不属于训练图片。

## 标注说明

- 标注文件位于 `Annotations/` 目录下，格式为 `.xml`。
- `Annotations/` 与 `images/` 采用相同的类别目录结构。
- 图片与标注文件应按文件名对应使用。
- 增广后的图片位于 `brightness/`、`noise/`、`rotation/` 目录下，分别对应亮度变化、加噪声和旋转变换。

## 使用建议

1. 训练检测模型时，可将 `images/` 作为原始训练图片来源，将 `Annotations/` 作为对应标注来源。
2. 需要扩大训练集时，可加入 `brightness/`、`noise/`、`rotation/` 中的增广图片。
3. 划分训练集、验证集和测试集时，建议按原始样本维度划分，再加入对应增广样本，避免同一原始样本的增广版本同时出现在训练集和测试集中。
4. 如果训练脚本自动扫描图片，请注意排除 `rotation/` 下的 `*_angles.txt` 文件。

## 注意事项

- `PCB_USED/` 中保存的是原始 PCB 图片。
- `images/` 为整理后的原始样本图片目录。
- `Annotations/` 为标注目录，不包含图片。
- `brightness/`、`noise/`、`rotation/` 均为数据增广结果，可根据实验需求选择使用。
