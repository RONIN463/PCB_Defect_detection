from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

DEFECT_TYPES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
]

SOURCES = [
    ("images", ""),
    ("brightness", "_brightness"),
    ("noise", "_noise"),
    ("rotation", "_rotation"),
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Diagnose PCB YOLO model by source set and confidence threshold."
    )
    parser.add_argument("--model", type=Path, default=root / "pc_pcb" / "runs" / "best.pt")
    parser.add_argument("--dataset", type=Path, default=root / "pc_pcb" / "PCB_DATASET")
    parser.add_argument("--output", type=Path, default=root / "yolov8" / "diagnosis_results.csv")
    parser.add_argument("--conf-min", type=float, default=0.001)
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
        help="Comma-separated confidence thresholds to report.",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--rotation-boxes",
        choices=["adjusted", "original"],
        default="adjusted",
        help="Use adjusted rotated boxes or original XML boxes for rotation images.",
    )
    return parser.parse_args()


def iou_xyxy(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def parse_xml(xml_path: Path) -> tuple[list[list[float]], list[int], tuple[int, int] | None]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    image_size = None
    if not xml_path.exists():
        return boxes, classes, image_size

    root = ET.parse(xml_path).getroot()
    size_node = root.find("size")
    if size_node is not None:
        image_size = (
            int(float(size_node.findtext("width", "0"))),
            int(float(size_node.findtext("height", "0"))),
        )
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None or name_node.text not in CLASS_TO_ID:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        boxes.append([
            float(bnd.findtext("xmin", "0")),
            float(bnd.findtext("ymin", "0")),
            float(bnd.findtext("xmax", "0")),
            float(bnd.findtext("ymax", "0")),
        ])
        classes.append(CLASS_TO_ID[name_node.text])
    return boxes, classes, image_size


def load_rotation_angles(dataset: Path) -> dict[str, float]:
    angles: dict[str, float] = {}
    for path in (dataset / "rotation").glob("*_angles.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2:
                angles[parts[0]] = float(parts[1])
    return angles


def rotate_boxes(
    boxes: list[list[float]],
    angle_deg: float,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[list[float]]:
    if angle_deg == 0:
        return boxes
    matrix = cv2.getRotationMatrix2D((src_w / 2.0, src_h / 2.0), -angle_deg, 1.0)
    matrix[0, 2] += (dst_w / 2.0) - (src_w / 2.0)
    matrix[1, 2] += (dst_h / 2.0) - (src_h / 2.0)
    rotated = []
    for box in boxes:
        x1, y1, x2, y2 = box
        corners = np.array([
            [x1, y1, 1.0],
            [x2, y1, 1.0],
            [x2, y2, 1.0],
            [x1, y2, 1.0],
        ])
        transformed = corners @ matrix.T
        min_xy = transformed.min(axis=0)
        max_xy = transformed.max(axis=0)
        rotated.append([
            max(0.0, float(min_xy[0])),
            max(0.0, float(min_xy[1])),
            min(float(dst_w), float(max_xy[0])),
            min(float(dst_h), float(max_xy[1])),
        ])
    return rotated


def collect_files(dataset: Path) -> list[tuple[str, Path, Path]]:
    files: list[tuple[str, Path, Path]] = []
    xml_root = dataset / "Annotations"
    for source, suffix in SOURCES:
        source_root = dataset / source
        for defect in DEFECT_TYPES:
            image_dir = source_root / f"{defect}{suffix}"
            if not image_dir.exists():
                continue
            for image_path in sorted(image_dir.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                xml_path = xml_root / defect / f"{image_path.stem}.xml"
                files.append((source, image_path, xml_path))
    return files


def match_counts(
    pred_boxes: list[list[float]],
    pred_classes: list[int],
    gt_boxes: list[list[float]],
    gt_classes: list[int],
    iou_threshold: float,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    gt_total = defaultdict(int)
    tp = defaultdict(int)
    fn = defaultdict(int)
    fp = defaultdict(int)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for cls in gt_classes:
        gt_total[cls] += 1

    for pi, (p_box, p_cls) in enumerate(zip(pred_boxes, pred_classes)):
        best_iou = 0.0
        best_gi = -1
        for gi, g_box in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            current_iou = iou_xyxy(p_box, g_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gi = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            gt_cls = gt_classes[best_gi]
            if p_cls == gt_cls:
                tp[gt_cls] += 1
            else:
                fp[p_cls] += 1
                fn[gt_cls] += 1
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    for gi, gt_cls in enumerate(gt_classes):
        if gi not in matched_gt:
            fn[gt_cls] += 1
    for pi, p_cls in enumerate(pred_classes):
        if pi not in matched_pred:
            fp[p_cls] += 1

    return gt_total, tp, fp, fn


def add_counts(target: dict[str, dict[int, int]], source: str, counts: dict[int, int]) -> None:
    for cls, count in counts.items():
        target[source][cls] += count


def main() -> None:
    args = parse_args()
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    files = collect_files(args.dataset)
    rotation_angles = load_rotation_angles(args.dataset)
    model = YOLO(str(args.model))

    stats = {
        threshold: {
            "gt": defaultdict(lambda: defaultdict(int)),
            "tp": defaultdict(lambda: defaultdict(int)),
            "fp": defaultdict(lambda: defaultdict(int)),
            "fn": defaultdict(lambda: defaultdict(int)),
        }
        for threshold in thresholds
    }

    for idx, (source, image_path, xml_path) in enumerate(files, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gt_boxes, gt_classes, xml_size = parse_xml(xml_path)
        if source == "rotation" and args.rotation_boxes == "adjusted":
            angle = rotation_angles.get(image_path.stem, 0.0)
            src_w, src_h = xml_size if xml_size is not None else (image.shape[1], image.shape[0])
            gt_boxes = rotate_boxes(
                gt_boxes,
                angle,
                src_w,
                src_h,
                image.shape[1],
                image.shape[0],
            )

        result = model(image, conf=args.conf_min, device=args.device, verbose=False)[0]
        all_boxes: list[list[float]] = []
        all_classes: list[int] = []
        all_conf: list[float] = []
        if result.boxes is not None:
            for box in result.boxes:
                all_boxes.append(box.xyxy[0].cpu().numpy().tolist())
                all_classes.append(int(box.cls[0]))
                all_conf.append(float(box.conf[0]))

        for threshold in thresholds:
            pred_boxes = []
            pred_classes = []
            for box, cls, conf in zip(all_boxes, all_classes, all_conf):
                if conf >= threshold:
                    pred_boxes.append(box)
                    pred_classes.append(cls)
            gt_total, tp, fp, fn = match_counts(
                pred_boxes, pred_classes, gt_boxes, gt_classes, args.iou
            )
            add_counts(stats[threshold]["gt"], source, gt_total)
            add_counts(stats[threshold]["tp"], source, tp)
            add_counts(stats[threshold]["fp"], source, fp)
            add_counts(stats[threshold]["fn"], source, fn)

        if idx % 100 == 0 or idx == len(files):
            print(f"Processed {idx}/{len(files)}")

    rows = []
    for threshold in thresholds:
        for source, _suffix in SOURCES:
            gt_by_class = stats[threshold]["gt"][source]
            tp_by_class = stats[threshold]["tp"][source]
            fp_by_class = stats[threshold]["fp"][source]
            fn_by_class = stats[threshold]["fn"][source]
            for cls_id, cls_name in enumerate(CLASS_NAMES):
                gt = gt_by_class[cls_id]
                tp = tp_by_class[cls_id]
                fp = fp_by_class[cls_id]
                fn = fn_by_class[cls_id]
                recall = tp / gt if gt else 0.0
                precision = tp / (tp + fp) if tp + fp else 0.0
                rows.append({
                    "threshold": threshold,
                    "source": source,
                    "class": cls_name,
                    "gt": gt,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.output}")
    print("\nOverall by threshold:")
    print("conf   precision  recall  fp")
    for threshold in thresholds:
        gt = tp = fp = 0
        for source, _suffix in SOURCES:
            for cls_id in range(NUM_CLASSES):
                gt += stats[threshold]["gt"][source][cls_id]
                tp += stats[threshold]["tp"][source][cls_id]
                fp += stats[threshold]["fp"][source][cls_id]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt if gt else 0.0
        print(f"{threshold:0.2f}   {precision:0.4f}     {recall:0.4f}  {fp}")

    print("\nBy source at conf=0.25:")
    selected = 0.25 if 0.25 in thresholds else thresholds[0]
    print("source      precision  recall  gt    fp")
    for source, _suffix in SOURCES:
        gt = tp = fp = 0
        for cls_id in range(NUM_CLASSES):
            gt += stats[selected]["gt"][source][cls_id]
            tp += stats[selected]["tp"][source][cls_id]
            fp += stats[selected]["fp"][source][cls_id]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt if gt else 0.0
        print(f"{source:10}  {precision:0.4f}     {recall:0.4f}  {gt:<5} {fp}")


if __name__ == "__main__":
    main()
