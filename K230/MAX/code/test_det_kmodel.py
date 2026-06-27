import os
import sys
import argparse
import cv2
import numpy as np
import shutil
from pathlib import Path

# nncase-kpu 插件将 k230 的 DLL 和可执行程序安装在 site-packages 目录下，
# 需同时注册 DLL 搜索路径和 PATH 环境变量
_site_packages = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages')
if os.path.isdir(_site_packages):
    os.add_dll_directory(_site_packages)
    os.environ['PATH'] = _site_packages + ';' + os.environ.get('PATH', '')

import nncase
import getcolors

# ================== PCB 缺陷检测配置 ==================
CLASS_NAMES = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# 支持的图片后缀
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def imread_cn(filepath):
    """cv2.imread 的替代，支持中文路径。"""
    data = np.fromfile(filepath, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_model_file(model_file):
    """读取 kmodel 二进制文件。"""
    with open(model_file, 'rb') as f:
        return f.read()


def preprocess(image, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT):
    """
    预处理输入图像：letterbox 缩放 + 填充、BGR→RGB、HWC→CHW、添加批次维度。
    返回 uint8 格式（kmodel 内置预处理负责归一化）。
    """
    orig_h, orig_w = image.shape[:2]
    scale = min(input_width / orig_w, input_height / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    canvas = np.ones((input_height, input_width, 3), dtype=np.uint8) * 128
    canvas[0:new_h, 0:new_w, :] = resized_image
    img = canvas[:, :, ::-1]               # BGR → RGB
    img = np.transpose(img, (2, 0, 1))       # HWC → CHW
    img = np.expand_dims(img, axis=0)        # 添加 batch 维度
    return img.copy(), scale


def postprocess(predictions, scale, conf_threshold=CONF_THRESHOLD,
                iou_threshold=IOU_THRESHOLD):
    """
    后处理：解析边界框、NMS、坐标映射回原图。
    """
    predictions = predictions[0]
    predictions = np.transpose(predictions, (1, 0))

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    mask = scores > conf_threshold
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

    if len(boxes) == 0:
        return []

    boxes_xy = boxes[:, :2]
    boxes_wh = boxes[:, 2:4]
    boxes_xy -= boxes_wh / 2
    boxes_xy /= scale
    boxes_wh /= scale
    boxes = np.concatenate([boxes_xy, boxes_xy + boxes_wh], axis=1).astype(np.float32)
    scores = scores.astype(np.float32)

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                conf_threshold, iou_threshold)
    if len(indices) == 0:
        return []
    indices = indices.flatten()

    return [{"box": boxes[i], "score": float(scores[i]), "class_id": int(class_ids[i])} for i in indices]


def draw_boxes(image, detections, class_names, colors):
    """在图像上绘制检测框和标签。"""
    for det in detections:
        box = det["box"]
        score = det["score"]
        class_id = det["class_id"]
        if class_id >= len(class_names):
            continue
        x1, y1, x2, y2 = map(int, box)
        color = colors[class_id]
        label = f"{class_names[class_id]}: {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


def process_single(image_path, sim, input_width, input_height, data_type,
                   conf, iou, class_names, colors, output_dir):
    """处理单张图片。"""
    print(f"\n{'='*60}")
    print(f"[IMAGE] {Path(image_path).name}")

    image = imread_cn(str(image_path))
    if image is None:
        print(f"[SKIP] 无法读取: {image_path}")
        return

    img_input, scale = preprocess(image, input_width, input_height)
    input_shape = [1, 3, input_height, input_width]
    input_tensor = img_input.astype(data_type).reshape(input_shape)

    sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input_tensor))
    sim.run()
    predictions = sim.get_output_tensor(0).to_numpy()

    detections = postprocess(predictions, scale, conf, iou)
    print(f"[RESULT] 检测到 {len(detections)} 个目标")

    for det in detections:
        cls_name = class_names[det['class_id']] if det['class_id'] < len(class_names) else 'Unknown'
        print(f"  - {cls_name}: {det['score']:.3f}")

    result_image = draw_boxes(image.copy(), detections, class_names, colors)
    out_name = f"kmodel_{Path(image_path).stem}.jpg"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, result_image)
    print(f"[SAVED] {out_path}")


def collect_images(paths):
    """收集所有图片路径：支持单个文件、多个文件、文件夹。"""
    images = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in IMG_EXTENSIONS:
                    images.append(f)
        elif p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
            images.append(p)
        else:
            print(f"[SKIP] 不是图片或目录: {p}")
    return images


def main():
    parser = argparse.ArgumentParser(description='KModel PCB 缺陷检测')
    parser.add_argument('--model', type=str,
                        default=r'D:\数字图像化处理\新模型\runs\yolov8n_pcb\weights\best.kmodel',
                        help='KModel 模型路径')
    parser.add_argument('--image', type=str, nargs='+',
                        default=['../test_images/test.jpg'],
                        help='测试图像路径（支持多个文件或文件夹）')
    parser.add_argument('--output', type=str, default='./output_kmodel',
                        help='输出目录')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=IOU_THRESHOLD, help='IoU 阈值')
    parser.add_argument('--input_width', type=int, default=INPUT_WIDTH, help='模型输入宽度')
    parser.add_argument('--input_height', type=int, default=INPUT_HEIGHT, help='模型输入高度')
    parser.add_argument('--classes', type=str, nargs='*', default=CLASS_NAMES, help='类别名称列表')
    args = parser.parse_args()

    class_names = args.classes
    colors = getcolors.get_colors(len(class_names))

    # 收集所有图片
    images = collect_images(args.image)
    if not images:
        print("[ERROR] 没有找到可处理的图片")
        return
    print(f"[INFO] 共找到 {len(images)} 张图片")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 创建 nncase Simulator（只加载一次模型）
    print("[INFO] 创建 nncase Simulator...")
    sim = nncase.Simulator()

    print(f"[INFO] 加载模型: {args.model}")
    kmodel = read_model_file(args.model)
    sim.load_model(kmodel)

    data_type = sim.get_input_desc(0).dtype
    print(f"[INFO] 模型输入 dtype: {data_type}")

    # 逐张处理
    for img_path in images:
        process_single(img_path, sim, args.input_width, args.input_height,
                       data_type, args.conf, args.iou, class_names, colors, args.output)

    # 清理临时文件
    if os.path.exists("./gmodel_dump_dir"):
        shutil.rmtree("./gmodel_dump_dir")

    print(f"\n[INFO] 全部完成! 结果保存在: {args.output}")


if __name__ == "__main__":
    main()
