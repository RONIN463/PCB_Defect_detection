import os
import sys

os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep)
    if p.lower().find("mingw") < 0 and p.lower().find("msys") < 0
)

import datetime
import math
import xml.etree.ElementTree as ET
import torch

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QGroupBox, QFrame,
    QSplitter, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QStatusBar, QProgressBar,
    QGraphicsView, QGraphicsScene, QShortcut, QTabWidget
)
from PyQt5.QtGui import (
    QPainter,
    QPixmap, QImage, QFont, QColor, QWheelEvent,
    QKeySequence
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from ultralytics import YOLO

CLASS_NAMES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper"
]

CLASS_CN = {
    "missing_hole": "缺失孔",
    "mouse_bite": "老鼠咬痕",
    "open_circuit": "开路",
    "short": "短路",
    "spur": "毛刺",
    "spurious_copper": "铜渣"
}

CLASS_COLORS = {
    "missing_hole": (255, 0, 0),
    "mouse_bite": (0, 255, 0),
    "open_circuit": (0, 0, 255),
    "short": (255, 255, 0),
    "spur": (255, 0, 255),
    "spurious_copper": (0, 255, 255)
}

DETECT_CONF = 0.25
ZOOM_MIN = 0.1
ZOOM_MAX = 5.0
ZOOM_STEP = 1.1
BOX_LINE_WIDTH = 3
FONT_SIZE_LABEL = 28
FONT_SIZE_LABEL_LARGE = 32
LARGE_IMAGE_THRESHOLD = 2000

def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def find_model_path():
    project_root = get_project_root()
    search_paths = [
        os.path.join(project_root, "runs", "best.pt"),
        os.path.join(project_root, "best.pt"),
        "yolov8n.pt"
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    return "yolov8n.pt"


def get_pil_font(size=28):
    font_path = "C:/Windows/Fonts/simsun.ttc"
    if os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_detection_boxes(image_bgr, boxes, font_size=None):
    img = image_bgr.copy()
    detections = []

    if boxes is None:
        return img, detections

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    if font_size is None:
        h, w = img.shape[:2]
        font_size = FONT_SIZE_LABEL_LARGE if max(w, h) > LARGE_IMAGE_THRESHOLD else FONT_SIZE_LABEL
    font = get_pil_font(font_size)

    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        conf = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        color = CLASS_COLORS.get(cls_name, (0, 255, 0))

        x1, y1, x2, y2 = map(int, xyxy)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=BOX_LINE_WIDTH)

        label_cn = CLASS_CN.get(cls_name, cls_name)
        label = f"{label_cn} {conf:.2f}"

        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        label_top = y1 - th - 10
        if label_top < 0:
            label_top = y1 + 4
            draw.rectangle([(x1, label_top), (x1 + tw + 8, label_top + th + 6)], fill=color)
            text_color = (0, 0, 0) if cls_name == "short" else (255, 255, 255)
            draw.text((x1 + 4, label_top + 2), label, font=font, fill=text_color)
        else:
            draw.rectangle([(x1, label_top), (x1 + tw + 8, y1)], fill=color)
            text_color = (0, 0, 0) if cls_name == "short" else (255, 255, 255)
            draw.text((x1 + 4, label_top + 2), label, font=font, fill=text_color)

        detections.append({
            "class": cls_name,
            "class_cn": label_cn,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2)
        })

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img, detections

class ZoomableImageView(QGraphicsView):
    zoom_changed = pyqtSignal(float)
    pan_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = self._scene.addPixmap(QPixmap())
        self._zoom_factor = 1.0
        self._linked_view = None
        self._updating_from_link = False
        self._placeholder_text = ""
        self._panning = False
        self._last_pan_pos = None
        self._resizing = False

        self.setRenderHints(
            QPainter.Antialiasing | QPainter.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setCursor(Qt.OpenHandCursor)
        self.setStyleSheet(
            "background-color: #f5f5f5;"
            " border: 1px solid #e0e0e0;"
        )
        self.setMinimumSize(400, 300)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap_item is not None:
            self._panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._last_pan_pos is not None:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()

            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )

            self._sync_scrollbars_to_linked()

            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() == Qt.LeftButton:
            self._panning = False
            self._last_pan_pos = None
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def set_placeholder(self, text):
        self._placeholder_text = text
        self._scene.clear()
        self._pixmap_item = None

    def set_image(self, image_rgb):
        h, w, ch = image_rgb.shape
        bytes_per_line = 3 * w
        q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img.copy())
        self._set_pixmap(pixmap)

    def set_image_bgr(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.set_image(rgb)

    def center_on(self, cx, cy):
        """以原始像素坐标居中，保持 zoom 不变。"""
        self.centerOn(QPointF(cx, cy))
        if self._linked_view is not None and not self._updating_from_link:
            self._linked_view._updating_from_link = True
            self._linked_view.centerOn(QPointF(cx, cy))
            self._linked_view._updating_from_link = False

    def set_image_preserve_zoom(self, image_rgb):
        """替换图片，保留当前 zoom 与 pan（不调用 fitInView）。"""
        h, w, _ = image_rgb.shape
        q_img = QImage(image_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img.copy())
        if self._pixmap_item is not None:
            # 就地替换 pixmap，QGraphicsView 的 transform 与滚动条保留
            self._pixmap_item.setPixmap(pixmap)
            self._scene.setSceneRect(QRectF(pixmap.rect()))
        else:
            self._set_pixmap(pixmap)

    def set_image_bgr_preserve_zoom(self, image_bgr):
        self.set_image_preserve_zoom(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    def _set_pixmap(self, pixmap):
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom_factor = 1.0

    def clear_image(self):
        self._scene.clear()
        self._pixmap_item = None
        if self._placeholder_text:
            text_item = self._scene.addText(self._placeholder_text)
            c = QColor("#666666")
            text_item.setDefaultTextColor(c)
        self._zoom_factor = 1.0

    def link_to(self, other_view):
        self._linked_view = other_view

    def zoom_in(self):
        self._apply_zoom(ZOOM_STEP)

    def zoom_out(self):
        self._apply_zoom(1.0 / ZOOM_STEP)

    def fit_to_window(self):
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
            self._zoom_factor = 1.0
            self.zoom_changed.emit(self._zoom_factor)

    def _apply_zoom(self, factor):
        new_zoom = self._zoom_factor * factor
        if new_zoom < ZOOM_MIN or new_zoom > ZOOM_MAX:
            return
        self._zoom_factor = new_zoom
        self.scale(factor, factor)
        self.zoom_changed.emit(self._zoom_factor)
        self._sync_to_linked()

    def _sync_to_linked(self):
        if self._linked_view is None or self._updating_from_link:
            return
        self._linked_view._updating_from_link = True
        self._linked_view._zoom_factor = self._zoom_factor
        self._linked_view.setTransform(self.transform())
        self._sync_scrollbars_to_linked()
        self._linked_view._updating_from_link = False

    def _sync_scrollbars_to_linked(self):
        if self._linked_view is None or self._updating_from_link:
            return
        if self._resizing or self._linked_view._resizing:
            return
        self._linked_view._updating_from_link = True
        self._linked_view.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value()
        )
        self._linked_view.verticalScrollBar().setValue(
            self.verticalScrollBar().value()
        )
        self._linked_view._updating_from_link = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self._apply_zoom(ZOOM_STEP)
        else:
            self._apply_zoom(1.0 / ZOOM_STEP)

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        if self._panning:
            return
        if self._updating_from_link:
            return
        if self._resizing:
            return
        self._sync_scrollbars_to_linked()

    def resizeEvent(self, event):
        if self._pixmap_item is not None:
            saved_h = self.horizontalScrollBar().value()
            saved_v = self.verticalScrollBar().value()
            self._resizing = True
            if self._linked_view is not None:
                self._linked_view._resizing = True
            old_anchor = self.resizeAnchor()
            self.setResizeAnchor(QGraphicsView.NoAnchor)
            super().resizeEvent(event)
            self.setResizeAnchor(old_anchor)
            self.horizontalScrollBar().setValue(saved_h)
            self.verticalScrollBar().setValue(saved_v)
            self._resizing = False
            if self._linked_view is not None:
                self._linked_view._resizing = False
            self._sync_scrollbars_to_linked()
        else:
            super().resizeEvent(event)

class DetectThread(QThread):
    finished = pyqtSignal(object, object)
    progress = pyqtSignal(str)

    def __init__(self, model, image):
        super().__init__()
        self.model = model
        self.image = image

    def run(self):
        try:
            self.progress.emit("正在检测...")
            results = self.model(self.image, conf=DETECT_CONF, verbose=False)
            self.progress.emit("检测完成")
            self.finished.emit(self.image, results)
        except Exception as e:
            self.progress.emit(f"检测出错: {str(e)}")
            self.finished.emit(None, None)


class BatchDetectThread(QThread):
    finished = pyqtSignal(list)
    progress = pyqtSignal(str, int, int)

    def __init__(self, model, image_paths, save_dir):
        super().__init__()
        self.model = model
        self.image_paths = image_paths
        self.save_dir = save_dir
        self._first_original = None
        self._first_result = None

    def run(self):
        results = []
        total = len(self.image_paths)
        first_saved = False

        for idx, img_path in enumerate(self.image_paths):
            try:
                self.progress.emit(f"正在检测第 {idx + 1} 张", idx + 1, total)
                img = cv2.imread(img_path)
                if img is None:
                    results.append({
                        "path": img_path, "detections": [],
                        "count": 0, "error": "无法读取图像"
                    })
                    continue

                yolo_results = self.model(img, conf=DETECT_CONF, verbose=False)
                boxes = yolo_results[0].boxes

                if boxes is not None:
                    result_img, detections = draw_detection_boxes(img, boxes)
                else:
                    result_img = img.copy()
                    detections = []

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(self.save_dir, f"{base_name}_detected.jpg")
                cv2.imwrite(save_path, result_img)

                info_path = os.path.join(self.save_dir, f"{base_name}_info.txt")
                with open(info_path, "w", encoding="utf-8") as f:
                    f.write(f"检测图像: {os.path.basename(img_path)}\n")
                    f.write(f"缺陷总数: {len(detections)}\n\n")
                    for i, d in enumerate(detections, 1):
                        f.write(
                            f"{i}. {d['class_cn']}({d['class']}) "
                            f"- 置信度: {d['confidence']:.2f}\n"
                        )

                if not first_saved:
                    self._first_original = img
                    self._first_result = result_img
                    first_saved = True

                results.append({
                    "path": img_path, "detections": detections,
                    "count": len(detections)
                })

            except Exception as e:
                results.append({
                    "path": img_path, "detections": [],
                    "count": 0, "error": str(e)
                })

        self.progress.emit("批量检测完成", total, total)
        self.finished.emit(results)

# --- 模型评估工具函数 ---

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def parse_xml_annotations(xml_path):
    gt_boxes = []
    gt_classes = []
    image_size = None
    if not os.path.exists(xml_path):
        return gt_boxes, gt_classes, image_size
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_node = root.find('size')
    if size_node is not None:
        try:
            image_size = (
                int(float(size_node.find('width').text)),
                int(float(size_node.find('height').text))
            )
        except Exception:
            image_size = None
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASS_NAME_TO_ID:
            continue
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        gt_boxes.append([xmin, ymin, xmax, ymax])
        gt_classes.append(CLASS_NAME_TO_ID[cls_name])
    return gt_boxes, gt_classes, image_size


def load_rotation_angles(project_root):
    angles = {}
    rotation_dir = os.path.join(project_root, 'PCB_DATASET', 'rotation')
    if not os.path.exists(rotation_dir):
        return angles
    for f in os.listdir(rotation_dir):
        if not f.endswith('_angles.txt'):
            continue
        path = os.path.join(rotation_dir, f)
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    angles[parts[0]] = float(parts[1])
    return angles


def rotate_boxes(boxes, angle_deg, src_w, src_h, dst_w=None, dst_h=None):
    if angle_deg == 0.0:
        return boxes
    if dst_w is None:
        dst_w = src_w
    if dst_h is None:
        dst_h = src_h

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
            max(0.0, min_xy[0]),
            max(0.0, min_xy[1]),
            min(float(dst_w), max_xy[0]),
            min(float(dst_h), max_xy[1])
        ])
    return rotated


def match_predictions(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5):
    y_true = []
    y_pred = []
    matched_gt = set()
    matched_pred = set()
    for pi, (p_box, p_cls) in enumerate(zip(pred_boxes, pred_classes)):
        best_iou = 0.0
        best_gi = -1
        for gi, g_box in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            current_iou = iou(p_box, g_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gi = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            gt_cls = gt_classes[best_gi]
            y_true.append(gt_cls)
            y_pred.append(p_cls)
            matched_gt.add(best_gi)
            matched_pred.add(pi)
    for gi in range(len(gt_classes)):
        if gi not in matched_gt:
            y_true.append(gt_classes[gi])
            y_pred.append(NUM_CLASSES)
    for pi in range(len(pred_classes)):
        if pi not in matched_pred:
            y_true.append(NUM_CLASSES)
            y_pred.append(pred_classes[pi])
    return y_true, y_pred

def confusion_matrix_np(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(t, -1)
        pi = label_to_idx.get(p, -1)
        if ti >= 0 and pi >= 0:
            cm[ti, pi] += 1
    return cm

def get_matplotlib_chinese_font():
    font_paths = [
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return FontProperties(fname=fp)
            except Exception:
                continue
    return None

def plot_confusion_matrix_chart(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=12, color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    font_prop = get_matplotlib_chinese_font()
    if font_prop:
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontproperties=font_prop)
        ax.set_yticklabels(class_names, fontproperties=font_prop)
        ax.set_xlabel('Predicted', fontsize=14, fontproperties=font_prop)
        ax.set_ylabel('True', fontsize=14, fontproperties=font_prop)
        ax.set_title('Confusion Matrix (IoU=0.5)', fontsize=16, fontproperties=font_prop)
    else:
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('True', fontsize=14)
        ax.set_title('Confusion Matrix (IoU=0.5)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_normalized_cm_chart(cm, class_names, save_path):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype('float') / row_sums
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=12, color='white' if val > 0.5 else 'black')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    font_prop = get_matplotlib_chinese_font()
    if font_prop:
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontproperties=font_prop)
        ax.set_yticklabels(class_names, fontproperties=font_prop)
        ax.set_xlabel('Predicted', fontsize=14, fontproperties=font_prop)
        ax.set_ylabel('True', fontsize=14, fontproperties=font_prop)
        ax.set_title('Normalized Confusion Matrix (IoU=0.5)', fontsize=16, fontproperties=font_prop)
    else:
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('True', fontsize=14)
        ax.set_title('Normalized Confusion Matrix (IoU=0.5)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class EvaluateThread(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str, int, int)

    def __init__(self, model):
        super().__init__()
        self.model = model

    def run(self):
        project_root = get_project_root()
        xml_dir = os.path.join(project_root, 'PCB_DATASET', 'Annotations')
        save_dir = os.path.join(project_root, 'evaluation_results')
        os.makedirs(save_dir, exist_ok=True)

        y_true_all = []
        y_pred_all = []
        defect_types = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']

        image_sources = [
            ('images', ''),
            ('brightness', '_brightness'),
            ('noise', '_noise'),
            ('rotation', '_rotation'),
        ]

        rotation_angles = load_rotation_angles(project_root)

        all_files = []
        for src_dir, suffix in image_sources:
            src_path = os.path.join(project_root, 'PCB_DATASET', src_dir)
            if not os.path.exists(src_path):
                continue
            for defect_type in defect_types:
                img_subdir = os.path.join(src_path, defect_type + suffix)
                if not os.path.exists(img_subdir):
                    continue
                image_files = sorted([
                    f for f in os.listdir(img_subdir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ])
                for img_file in image_files:
                    all_files.append((os.path.join(img_subdir, img_file),
                                      os.path.join(xml_dir, defect_type,
                                                   os.path.splitext(img_file)[0] + '.xml'),
                                      src_dir))

        total = len(all_files)
        for idx, (img_path, xml_path, src) in enumerate(all_files):
            self.progress.emit(f'评估中 ({idx+1}/{total})', idx + 1, total)
            gt_boxes, gt_classes, xml_size = parse_xml_annotations(xml_path)
            img = cv2.imread(img_path)
            if img is None:
                continue
            if src == 'rotation':
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                angle = rotation_angles.get(base_name, 0.0)
                if angle != 0.0:
                    h, w = img.shape[:2]
                    src_w, src_h = xml_size if xml_size is not None else (w, h)
                    gt_boxes = rotate_boxes(gt_boxes, angle, src_w, src_h, w, h)
            results = self.model(img, conf=DETECT_CONF, verbose=False)
            pred_boxes = []
            pred_classes = []
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    cls_id = int(boxes.cls[i])
                    pred_boxes.append(xyxy.tolist())
                    pred_classes.append(cls_id)
            yt, yp = match_predictions(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5)
            y_true_all.extend(yt)
            y_pred_all.extend(yp)

        self.progress.emit('评估完成', total, total)

        result = {
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'total_instances': len(y_true_all),
            'save_dir': save_dir
        }
        self.finished.emit(result)


class PCBDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.original_image = None
        self.result_image = None
        self.current_image_path = None
        self.detections = []
        self.folder_image_paths = []
        self.folder_save_dir = None

        self.init_ui()
        self.setup_shortcuts()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle("PCB板缺陷检测系统 - YOLOv8")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet("""
            /* ==================== 全局样式 ==================== */
            QMainWindow {
                background-color: #ECEFF1;
            }
            QWidget {
                font-family: "Microsoft YaHei", "Segoe UI", "SimSun";
                color: #263238;
            }

            /* ==================== 工具栏框架 ==================== */
            #toolbarFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0F0C29, stop:0.5 #1B1B2F, stop:1 #24243E);
                border-radius: 8px;
                padding: 8px 16px;
            }

            /* ==================== 按钮基础样式 ==================== */
            QPushButton {
                border: 1px solid rgba(255, 255, 255, 0.15);
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                min-width: 100px;
                font-family: "Microsoft YaHei", "Segoe UI", "SimSun";
            }

            /* 打开图片按钮 - 明亮蓝 */
            QPushButton#btnOpen {
                background-color: #4A90D9;
                color: #FFFFFF;
            }
            QPushButton#btnOpen:hover {
                background-color: #5BA0E8;
                border-color: rgba(255, 255, 255, 0.35);
            }
            QPushButton#btnOpen:pressed {
                background-color: #3A7BC8;
            }
            QPushButton#btnOpen:disabled {
                background-color: #3A5A7A;
                color: rgba(255, 255, 255, 0.4);
                border-color: rgba(255, 255, 255, 0.08);
            }

            /* 开始检测按钮 - 翠绿 */
            QPushButton#btnDetect {
                background-color: #52B86A;
                color: #FFFFFF;
            }
            QPushButton#btnDetect:hover {
                background-color: #66C87C;
                border-color: rgba(255, 255, 255, 0.35);
            }
            QPushButton#btnDetect:pressed {
                background-color: #3FA85A;
            }
            QPushButton#btnDetect:disabled {
                background-color: #3A6A44;
                color: rgba(255, 255, 255, 0.4);
                border-color: rgba(255, 255, 255, 0.08);
            }

            /* 批量检测按钮 - 青碧 */
            QPushButton#btnBatch {
                background-color: #3BA99E;
                color: #FFFFFF;
            }
            QPushButton#btnBatch:hover {
                background-color: #50BCB0;
                border-color: rgba(255, 255, 255, 0.35);
            }
            QPushButton#btnBatch:pressed {
                background-color: #2E968C;
            }
            QPushButton#btnBatch:disabled {
                background-color: #2D6B64;
                color: rgba(255, 255, 255, 0.4);
                border-color: rgba(255, 255, 255, 0.08);
            }

            /* 保存结果按钮 - 暖金 */
            QPushButton#btnSave {
                background-color: #E8A23A;
                color: #FFFFFF;
            }
            QPushButton#btnSave:hover {
                background-color: #F0B34E;
                border-color: rgba(255, 255, 255, 0.35);
            }
            QPushButton#btnSave:pressed {
                background-color: #D0912A;
            }
            QPushButton#btnSave:disabled {
                background-color: #7A643A;
                color: rgba(255, 255, 255, 0.4);
                border-color: rgba(255, 255, 255, 0.08);
            }

            /* 清空按钮 - 珊瑚红（重要操作警示色） */
            QPushButton#btnClear {
                background-color: #E06050;
                color: #FFFFFF;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            QPushButton#btnClear:hover {
                background-color: #EF7868;
                border-color: rgba(255, 255, 255, 0.4);
            }
            QPushButton#btnClear:pressed {
                background-color: #D04838;
            }
            QPushButton#btnClear:disabled {
                background-color: #6A3A35;
                color: rgba(255, 255, 255, 0.4);
                border-color: rgba(255, 255, 255, 0.08);
            }

            /* 评估模型按钮 - 淡紫 */
            QPushButton#btnEvaluate {
                background-color: #9B67D9;
                color: #FFFFFF;
            }
            QPushButton#btnEvaluate:hover {
                background-color: #AE7CE8;
                border-color: rgba(255, 255, 255, 0.35);
            }
            QPushButton#btnEvaluate:pressed {
                background-color: #8852C8;
            }
            QPushButton#btnEvaluate:disabled {
                background-color: #4A3A6A;
                color: rgba(255, 255, 255, 0.4);
                border-color: rgba(255, 255, 255, 0.08);
            }

            /* 缩放按钮 - 深色主题 */
            QPushButton#btnZoomIn, QPushButton#btnZoomOut {
                background-color: #37474F;
                color: #ECEFF1;
                border: 1px solid #546E7A;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 16px;
                min-width: 40px;
                font-weight: bold;
            }
            QPushButton#btnZoomIn:hover, QPushButton#btnZoomOut:hover {
                background-color: #4B5E68;
                border-color: #90A4AE;
            }
            QPushButton#btnZoomIn:pressed, QPushButton#btnZoomOut:pressed {
                background-color: #2A3740;
            }

            /* ==================== GroupBox 样式 ==================== */
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                font-family: "Microsoft YaHei", "Segoe UI", "SimSun";
                border: 1px solid #CFD8DC;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px 8px 8px 8px;
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0px 8px;
                color: #1976D2;
            }
            QGroupBox#imageGroup {
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                margin-top: 10px;
                background-color: #FFFFFF;
            }
            QGroupBox#imageGroup::title {
                color: #37474F;
                font-size: 13px;
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }

            /* ==================== 表格样式 ==================== */
            QTableWidget {
                font-size: 14px;
                font-family: "Microsoft YaHei", "Segoe UI", "SimSun";
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                gridline-color: #F0F0F0;
                selection-background-color: #E8EAF6;
                selection-color: #1A237E;
                alternate-background-color: #F8F9FB;
            }
            QTableWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid #F0F0F0;
            }
            QTableWidget::item:selected {
                background-color: #E8EAF6;
                color: #1A237E;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1A237E, stop:1 #3949AB);
                color: white;
                padding: 10px 12px;
                border: none;
                border-radius: 0px;
                font-weight: bold;
                font-size: 13px;
            }
            QHeaderView::section:first {
                border-top-left-radius: 8px;
            }
            QHeaderView::section:last {
                border-top-right-radius: 8px;
            }

            /* ==================== 文本编辑框 ==================== */
            QTextEdit {
                font-family: "Microsoft YaHei", "Segoe UI", "SimSun";
                font-size: 14px;
                background-color: #FAFBFC;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 10px 12px;
                selection-background-color: #E8EAF6;
            }
            QTextEdit:focus {
                border: 2px solid #3949AB;
            }

            /* ==================== Tab 样式 ==================== */
            QTabWidget::pane {
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                background-color: #FAFBFC;
                top: -1px;
            }
            QTabBar::tab {
                font-family: "Microsoft YaHei", "Segoe UI", "SimSun";
                font-size: 14px;
                font-weight: bold;
                padding: 10px 24px;
                margin-right: 4px;
                color: #78909C;
                background-color: #F0F2F5;
                border: 1px solid #E0E0E0;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background-color: #FAFBFC;
                color: #1A237E;
                border-color: #CFD8DC;
                border-bottom: 2px solid #3949AB;
            }
            QTabBar::tab:hover:!selected {
                background-color: #E8EAF0;
                color: #3949AB;
            }

            /* ==================== 分隔器 ==================== */
            QSplitter::handle:vertical {
                background-color: #E0E0E0;
                height: 3px;
            }
            QSplitter::handle:horizontal {
                background-color: #E0E0E0;
                width: 3px;
            }

            /* ==================== 状态栏 ==================== */
            QStatusBar {
                font-size: 13px;
                font-family: "Microsoft YaHei", "Segoe UI", "SimSun";
                background-color: #FFFFFF;
                border-top: 1px solid #E0E0E0;
                padding: 4px;
            }
            QStatusBar QLabel {
                color: #546E7A;
            }

            /* ==================== 进度条 ==================== */
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #E0E0E0;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1976D2, stop:1 #42A5F5);
                border-radius: 4px;
            }

            /* ==================== 消息框 ==================== */
            QMessageBox {
                background-color: #FAFAFA;
            }
            QMessageBox QLabel {
                font-family: "Microsoft YaHei", "Segoe UI", "SimSun";
                font-size: 14px;
                color: #263238;
                padding: 8px;
            }
            QMessageBox QPushButton {
                min-width: 80px;
                padding: 8px 16px;
                border-radius: 6px;
                background-color: #1976D2;
                color: white;
                border: none;
            }
            QMessageBox QPushButton:hover {
                background-color: #1E88E5;
            }

            /* ==================== 标签 ==================== */
            QLabel {
                font-family: "Microsoft YaHei", "Segoe UI", "SimSun";
                color: #263238;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # --- 工具栏 ---
        # 工具栏背景框架
        self.toolbar_frame = QFrame()
        self.toolbar_frame.setObjectName("toolbarFrame")
        toolbar_layout = QHBoxLayout(self.toolbar_frame)
        toolbar_layout.setContentsMargins(16, 12, 16, 12)
        toolbar_layout.setSpacing(12)

        self.btn_open = QPushButton("打开图片")
        self.btn_open.setObjectName("btnOpen")
        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.setObjectName("btnDetect")
        self.btn_batch_detect = QPushButton("批量检测")
        self.btn_batch_detect.setObjectName("btnBatch")
        self.btn_save = QPushButton("保存结果")
        self.btn_save.setObjectName("btnSave")
        self.btn_clear = QPushButton("清空")
        self.btn_clear.setObjectName("btnClear")
        self.btn_evaluate = QPushButton("评估模型")
        self.btn_evaluate.setObjectName("btnEvaluate")

        self.btn_detect.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.btn_open.clicked.connect(self.open_image)
        self.btn_detect.clicked.connect(self.detect)
        self.btn_batch_detect.clicked.connect(self.batch_detect)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_clear.clicked.connect(self.clear)
        self.btn_evaluate.clicked.connect(self.evaluate_model)

        toolbar_layout.addWidget(self.btn_open)
        toolbar_layout.addWidget(self.btn_detect)
        toolbar_layout.addWidget(self.btn_batch_detect)
        toolbar_layout.addWidget(self.btn_save)
        toolbar_layout.addWidget(self.btn_clear)
        toolbar_layout.addWidget(self.btn_evaluate)
        toolbar_layout.addStretch()

        model_info_label = QLabel("模型状态:")
        model_info_label.setStyleSheet("color: #78909C; font-weight: bold;")
        self.model_status_label = QLabel("未加载")
        self.model_status_label.setStyleSheet(
            "color: #FF8A65; font-weight: bold; padding: 4px 12px; "
            "background-color: rgba(255,138,101,0.12); border: 1px solid rgba(255,138,101,0.25);"
            " border-radius: 12px;"
        )
        toolbar_layout.addWidget(model_info_label)
        toolbar_layout.addWidget(self.model_status_label)

        main_layout.addWidget(self.toolbar_frame)

        # --- 图片面板 ---
        splitter = QSplitter(Qt.Horizontal)

        # 左侧
        left_group = QGroupBox("原始图像")
        left_group.setObjectName("imageGroup")
        left_layout = QVBoxLayout(left_group)
        left_layout.setContentsMargins(8, 8, 8, 8)

        left_zoom_bar = QHBoxLayout()
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setObjectName("btnZoomIn")
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setObjectName("btnZoomOut")
        self.btn_zoom_in.setMaximumWidth(40)
        self.btn_zoom_out.setMaximumWidth(40)
        left_zoom_bar.addWidget(self.btn_zoom_in)
        left_zoom_bar.addWidget(self.btn_zoom_out)
        left_zoom_bar.addSpacing(8)
        self.left_zoom_pct = QLabel("100%")
        self.left_zoom_pct.setStyleSheet(
            "color: #546E7A; font-size: 13px; font-weight: bold;"
        )
        left_zoom_bar.addWidget(self.left_zoom_pct)
        left_zoom_bar.addStretch()
        self.left_image_info = QLabel("")
        self.left_image_info.setStyleSheet(
            "color: #90A4AE; font-size: 12px;"
        )
        left_zoom_bar.addWidget(self.left_image_info)

        self.original_view = ZoomableImageView()
        self.original_view.set_placeholder("请打开一张PCB板图像")
        self.original_view.clear_image()

        left_layout.addLayout(left_zoom_bar)
        left_layout.addWidget(self.original_view)

        # 右侧
        right_group = QGroupBox("检测结果")
        right_group.setObjectName("imageGroup")
        right_layout = QVBoxLayout(right_group)
        right_layout.setContentsMargins(8, 8, 8, 8)

        right_zoom_bar = QHBoxLayout()
        self.btn_zoom_in2 = QPushButton("+")
        self.btn_zoom_in2.setObjectName("btnZoomIn")
        self.btn_zoom_out2 = QPushButton("-")
        self.btn_zoom_out2.setObjectName("btnZoomOut")
        self.btn_zoom_in2.setMaximumWidth(40)
        self.btn_zoom_out2.setMaximumWidth(40)
        right_zoom_bar.addWidget(self.btn_zoom_in2)
        right_zoom_bar.addWidget(self.btn_zoom_out2)
        right_zoom_bar.addSpacing(8)
        self.right_zoom_pct = QLabel("100%")
        self.right_zoom_pct.setStyleSheet(
            "color: #546E7A; font-size: 13px; font-weight: bold;"
        )
        right_zoom_bar.addWidget(self.right_zoom_pct)
        right_zoom_bar.addStretch()
        self.right_image_info = QLabel("")
        self.right_image_info.setStyleSheet(
            "color: #90A4AE; font-size: 12px;"
        )
        right_zoom_bar.addWidget(self.right_image_info)

        self.result_view = ZoomableImageView()
        self.result_view.set_placeholder("检测结果将显示在这里")
        self.result_view.clear_image()

        right_layout.addLayout(right_zoom_bar)
        right_layout.addWidget(self.result_view)

        # 联动
        self.original_view.link_to(self.result_view)
        self.result_view.link_to(self.original_view)

        # 缩放按钮连接
        self.btn_zoom_in.clicked.connect(self.zoom_all_in)
        self.btn_zoom_out.clicked.connect(self.zoom_all_out)
        self.btn_zoom_in2.clicked.connect(self.zoom_all_in)
        self.btn_zoom_out2.clicked.connect(self.zoom_all_out)

        # 缩放比例标签更新
        def make_zoom_updater(label):
            def update_zoom(factor):
                label.setText(f"{int(round(factor * 100))}%")
            return update_zoom

        self.original_view.zoom_changed.connect(
            make_zoom_updater(self.left_zoom_pct)
        )
        self.original_view.zoom_changed.connect(
            make_zoom_updater(self.right_zoom_pct)
        )

        splitter.addWidget(left_group)
        splitter.addWidget(right_group)
        # 垂直分割器：图片区域 vs 底部面板
        vsplitter = QSplitter(Qt.Vertical)
        vsplitter.addWidget(splitter)

        # --- 底部面板（Tab 切换） ---
        self.bottom_tabs = QTabWidget()

        # Tab 1: 检测结果
        detect_tab = QWidget()
        detect_tab_layout = QHBoxLayout(detect_tab)

        info_group = QGroupBox("检测信息")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(6, 6, 6, 6)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(200)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)

        table_group = QGroupBox("缺陷列表")
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(6, 6, 6, 6)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["序号", "缺陷类型", "英文名称", "置信度"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.cellClicked.connect(self.on_table_row_clicked)

        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)

        detect_tab_layout.addWidget(info_group, 1)
        detect_tab_layout.addWidget(table_group, 1)
        self.bottom_tabs.addTab(detect_tab, "检测结果")

        # Tab 2: 模型评估
        eval_tab = QWidget()
        eval_tab_layout = QHBoxLayout(eval_tab)

        # 左侧：评估信息
        eval_left = QVBoxLayout()
        eval_left.setContentsMargins(6, 6, 6, 6)
        self.eval_info_text = QTextEdit()
        self.eval_info_text.setReadOnly(True)
        self.eval_info_text.setPlaceholderText("点击「评估模型」按钮开始评估")
        eval_left.addWidget(self.eval_info_text)

        # 右侧：混淆矩阵图表（支持滚轮缩放 + 拖拽查看）
        eval_right_layout = QVBoxLayout()
        eval_right_layout.setContentsMargins(6, 6, 6, 6)
        self.eval_cm_view = ZoomableImageView()
        self.eval_cm_view.set_placeholder("混淆矩阵将显示在这里（评估完成后可滚轮缩放）")
        self.eval_cm_view.clear_image()
        eval_right_layout.addWidget(self.eval_cm_view)

        eval_tab_layout.addLayout(eval_left, 1)
        eval_tab_layout.addLayout(eval_right_layout, 1)
        self.bottom_tabs.addTab(eval_tab, "模型评估")

        vsplitter.addWidget(self.bottom_tabs)
        vsplitter.setStretchFactor(0, 3)
        vsplitter.setStretchFactor(1, 1)
        main_layout.addWidget(vsplitter)

        # --- 状态栏 ---
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "font-size: 13px; font-family: 'SimSun';"
        )
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请打开图像或加载模型")

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, self.open_image)
        QShortcut(QKeySequence("Ctrl+D"), self, self.detect)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_result)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

    def zoom_all_in(self):
        self.original_view.zoom_in()
        self.result_view.zoom_in()

    def zoom_all_out(self):
        self.original_view.zoom_out()
        self.result_view.zoom_out()

    def load_model(self):
        model_path = find_model_path()
        self.load_model_from_path(model_path)

    def load_model_from_path(self, model_path):
        try:
            self.model = YOLO(model_path)
            self.model_status_label.setText(
                f"已加载: {os.path.basename(model_path)}"
            )
            self.model_status_label.setStyleSheet(
                "color: #66BB6A; font-weight: bold; font-family: 'SimSun';"
                " padding: 4px 12px; border: 1px solid rgba(102,187,106,0.3);"
                " border-radius: 12px; background-color: rgba(102,187,106,0.1);"
            )
            self.status_bar.showMessage(f"模型已加载: {model_path}")
        except Exception as e:
            self.model_status_label.setText("加载失败")
            self.model_status_label.setStyleSheet(
                "color: #EF5350; font-weight: bold; font-family: 'SimSun';"
                " padding: 4px 12px; border: 1px solid rgba(239,83,80,0.3);"
                " border-radius: 12px; background-color: rgba(239,83,80,0.1);"
            )
            QMessageBox.warning(
                self, "警告",
                f"模型加载失败: {str(e)}\n请先运行 train.py 训练模型"
            )

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择PCB板图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if not file_path:
            return

        self.current_image_path = file_path
        self.original_image = cv2.imread(file_path)

        if self.original_image is None:
            QMessageBox.warning(self, "错误", "无法读取图像文件")
            return

        self.original_image = cv2.cvtColor(
            self.original_image, cv2.COLOR_BGR2RGB
        )
        self.original_view.set_image(self.original_image)
        h, w = self.original_image.shape[:2]
        self.left_image_info.setText(
            f"{os.path.basename(file_path)}  ({w}x{h})"
        )
        self.right_image_info.setText("")
        self.result_view.set_placeholder("检测结果将显示在这里")
        self.result_view.clear_image()
        self.btn_detect.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.detections = []
        self.info_text.clear()
        self.table.setRowCount(0)
        self.status_bar.showMessage(
            f"已加载图像: {os.path.basename(file_path)}"
        )

    def detect(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return

        bgr_image = cv2.cvtColor(
            self.original_image, cv2.COLOR_RGB2BGR
        )
        self.btn_detect.setEnabled(False)
        self.status_bar.showMessage("正在检测...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.detect_thread = DetectThread(self.model, bgr_image)
        self.detect_thread.finished.connect(self.on_detect_finished)
        self.detect_thread.progress.connect(self.status_bar.showMessage)
        self.detect_thread.start()

    def on_detect_finished(self, image, results):
        self.btn_detect.setEnabled(True)
        self.progress_bar.setVisible(False)

        if results is None:
            self.status_bar.showMessage("检测失败")
            return

        boxes = results[0].boxes
        result_bgr, self.detections = draw_detection_boxes(image, boxes)
        self.result_image = result_bgr
        self.result_view.set_image_bgr(result_bgr)
        h, w = result_bgr.shape[:2]
        self.right_image_info.setText(
            f"检测结果  ({w}x{h})  — {len(self.detections)} 个缺陷"
        )
        self.btn_save.setEnabled(True)

        self.update_info()
        self.update_table()
        self.status_bar.showMessage(
            f"检测完成 - 发现 {len(self.detections)} 个缺陷"
        )

    def batch_detect(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        # 弹出文件夹选择对话框
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择包含PCB图像的文件夹", ""
        )
        if not folder_path:
            return

        image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        self.folder_image_paths = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(image_extensions)
        ]

        if not self.folder_image_paths:
            QMessageBox.warning(self, "提示", "文件夹中没有找到图像文件")
            return

        project_root = get_project_root()
        self.folder_save_dir = os.path.join(
            project_root, "save_data", os.path.basename(folder_path)
        )
        os.makedirs(self.folder_save_dir, exist_ok=True)

        self.original_view.set_placeholder(
            f"已选择 {len(self.folder_image_paths)} 张图像\n"
            f"文件夹: {os.path.basename(folder_path)}"
        )
        self.original_view.clear_image()
        self.result_view.set_placeholder("点击「批量检测」开始检测")
        self.result_view.clear_image()
        self.btn_batch_detect.setEnabled(True)
        self.btn_detect.setEnabled(False)
        self.info_text.clear()
        self.info_text.append(f"已选择文件夹: {folder_path}")
        self.info_text.append(
            f"共 {len(self.folder_image_paths)} 张图像"
        )
        self.table.setRowCount(0)
        self.status_bar.showMessage(
            f"已加载文件夹: {folder_path}，"
            f"包含 {len(self.folder_image_paths)} 张图像"
        )

        self._do_batch_detect()

    def _do_batch_detect(self):
        self.btn_open.setEnabled(False)
        self.btn_batch_detect.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.status_bar.showMessage("开始批量检测...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.folder_image_paths))
        self.progress_bar.setValue(0)

        self.batch_thread = BatchDetectThread(
            self.model, self.folder_image_paths, self.folder_save_dir
        )
        self.batch_thread.finished.connect(self.on_batch_detect_finished)
        self.batch_thread.progress.connect(self.on_batch_progress)
        self.batch_thread.start()

    def on_batch_progress(self, text, current, total):
        self.status_bar.showMessage(f"{text} ({current}/{total})")
        self.progress_bar.setValue(current)

    def on_batch_detect_finished(self, results):
        self.btn_open.setEnabled(True)
        self.btn_batch_detect.setEnabled(True)
        self.progress_bar.setVisible(False)

        total_images = len(results)
        total_defects = sum(r["count"] for r in results)
        total_with_defects = sum(1 for r in results if r["count"] > 0)

        self.info_text.clear()

        # --- 标题区 ---
        html_parts = []
        html_parts.append(
            '<div style="font-family:SimSun;font-size:16px;font-weight:bold;'
            'color:#1976d2;margin-bottom:6px;">批量检测完成</div>'
        )
        html_parts.append(
            '<hr style="border:none;border-top:1px solid #e0e0e0;margin:4px 0 8px 0;"/>'
        )

        # --- 统计区 ---
        html_parts.append(
            f'<div style="font-family:SimSun;font-size:14px;color:#333;margin-bottom:2px;">'
            f'检测图像总数：<span style="color:#1976d2;font-weight:bold;">{total_images}</span></div>'
        )
        html_parts.append(
            f'<div style="font-family:SimSun;font-size:14px;color:#333;margin-bottom:2px;">'
            f'发现缺陷图像数：<span style="color:#d32f2f;font-weight:bold;">{total_with_defects}</span></div>'
        )
        html_parts.append(
            f'<div style="font-family:SimSun;font-size:14px;color:#333;margin-bottom:2px;">'
            f'缺陷总数：<span style="color:#d32f2f;font-weight:bold;">{total_defects}</span></div>'
        )
        html_parts.append(
            f'<div style="font-family:SimSun;font-size:14px;color:#555;margin-bottom:8px;">'
            f'结果保存位置：<span style="color:#1976d2;">{self.folder_save_dir}</span></div>'
        )

        html_parts.append(
            '<hr style="border:none;border-top:1px solid #e0e0e0;margin:4px 0 8px 0;"/>'
        )
        html_parts.append(
            '<div style="font-family:SimSun;font-size:14px;font-weight:bold;'
            'color:#333;margin-bottom:4px;">各图像检测结果</div>'
        )

        all_detections = []
        for idx, r in enumerate(results, 1):
            name = os.path.basename(r["path"])
            if "error" in r:
                html_parts.append(
                    f'<div style="font-family:SimSun;font-size:14px;color:#d32f2f;'
                    f'padding:2px 0 2px 4px;">'
                    f'<span style="color:#1976d2;font-weight:bold;">{idx:>3}.</span> '
                    f'{name} — 错误：{r["error"]}</div>'
                )
            elif r["count"] == 0:
                html_parts.append(
                    f'<div style="font-family:SimSun;font-size:14px;color:#555;'
                    f'padding:2px 0 2px 4px;">'
                    f'<span style="color:#1976d2;font-weight:bold;">{idx:>3}.</span> '
                    f'{name} — <span style="color:#2e7d32;">无缺陷</span></div>'
                )
            else:
                defect_list = "、".join(
                    [
                        f'{d["class_cn"]}<span style="color:#777;">({d["confidence"]:.2f})</span>'
                        for d in r["detections"]
                    ]
                )
                html_parts.append(
                    f'<div style="font-family:SimSun;font-size:14px;color:#333;'
                    f'padding:2px 0 2px 4px;">'
                    f'<span style="color:#1976d2;font-weight:bold;">{idx:>3}.</span> '
                    f'{name} — '
                    f'<span style="color:#d32f2f;font-weight:bold;">{r["count"]}个缺陷</span>：'
                    f'{defect_list}</div>'
                )
                # 给每条检测注入图片路径，便于点击表格行时定位到对应图片
                base_name = os.path.splitext(os.path.basename(r["path"]))[0]
                result_path = os.path.join(self.folder_save_dir, f"{base_name}_detected.jpg")
                for d in r["detections"]:
                    d["image_path"] = r["path"]
                    d["result_image_path"] = result_path
                all_detections.extend(r["detections"])

        self.info_text.setHtml("".join(html_parts))

        self.detections = all_detections
        self.update_table()

        # 显示第一张批量检测结果
        if self.batch_thread._first_original is not None:
            self.original_image = cv2.cvtColor(
                self.batch_thread._first_original, cv2.COLOR_BGR2RGB
            )
            self.original_view.set_image(self.original_image)
            self.result_image = self.batch_thread._first_result
            self.result_view.set_image_bgr(self.result_image)

        self.status_bar.showMessage(
            f"批量检测完成！检测了{total_images}张图像，"
            f"发现{total_defects}个缺陷，结果保存在: {self.folder_save_dir}"
        )

        QMessageBox.information(
            self, "批量检测完成",
            f"检测完成！\n\n"
            f"图像总数: {total_images}\n"
            f"发现缺陷图像数: {total_with_defects}\n"
            f"缺陷总数: {total_defects}\n\n"
            f"结果已保存到:\n{self.folder_save_dir}"
        )

    def update_info(self):
        self.info_text.clear()
        if not self.detections:
            self.info_text.setHtml(
                '<div style="color:#555;font-family:SimSun;font-size:14px;padding:4px;">'
                '未检测到任何缺陷</div>'
            )
            return

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 标题区
        html_parts = []
        html_parts.append(
            f'<div style="font-family:SimSun;font-size:15px;font-weight:bold;'
            f'color:#1976d2;margin-bottom:4px;">检测信息</div>'
        )
        html_parts.append(
            f'<div style="color:#333;font-family:SimSun;font-size:14px;'
            f'margin-bottom:2px;">检测时间：<span style="color:#1976d2;">{now_str}</span></div>'
        )
        html_parts.append(
            f'<div style="color:#333;font-family:SimSun;font-size:14px;'
            f'margin-bottom:8px;">缺陷总数：'
            f'<span style="color:#d32f2f;font-weight:bold;">{len(self.detections)}</span> 个</div>'
        )
        html_parts.append(
            '<hr style="border:none;border-top:1px solid #e0e0e0;margin:4px 0 8px 0;"/>'
        )

        # 统计区
        class_count = {}
        for det in self.detections:
            cn = det["class_cn"]
            class_count[cn] = class_count.get(cn, 0) + 1

        html_parts.append(
            '<div style="font-family:SimSun;font-size:14px;font-weight:bold;'
            'color:#333;margin-bottom:4px;">缺陷统计</div>'
        )
        html_parts.append(
            '<table style="font-family:SimSun;font-size:14px;'
            'color:#333;border-collapse:collapse;margin-bottom:8px;">'
        )
        for cn, count in class_count.items():
            html_parts.append(
                f'<tr><td style="padding:2px 8px 2px 4px;color:#333;">• {cn}</td>'
                f'<td style="padding:2px 4px;color:#1976d2;font-weight:bold;">{count} 个</td></tr>'
            )
        html_parts.append('</table>')

        # 详情区
        html_parts.append(
            '<hr style="border:none;border-top:1px solid #e0e0e0;margin:4px 0 8px 0;"/>'
        )
        html_parts.append(
            '<div style="font-family:SimSun;font-size:14px;font-weight:bold;'
            'color:#333;margin-bottom:4px;">检测结果详情</div>'
        )
        for i, det in enumerate(self.detections, 1):
            html_parts.append(
                f'<div style="font-family:SimSun;font-size:14px;color:#333;'
                f'padding:2px 0 2px 4px;">'
                f'<span style="color:#1976d2;font-weight:bold;">{i:>2}.</span> '
                f'{det["class_cn"]} '
                f'<span style="color:#777;">({det["class"]})</span> '
                f'<span style="color:#555;">— 置信度:</span> '
                f'<span style="color:#1976d2;font-weight:bold;">{det["confidence"]:.2%}</span>'
                f'</div>'
            )

        self.info_text.setHtml("".join(html_parts))

    def on_table_row_clicked(self, row, col):
        if row < 0 or row >= len(self.detections):
            return
        det = self.detections[row]
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # 批量检测：每条 detection 带有 image_path，需切换到对应图片
        img_path = det.get("image_path")
        if img_path and img_path != getattr(self, "_current_image_path", None):
            orig_bgr = cv2.imread(img_path)
            if orig_bgr is not None:
                orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                self.original_view.set_image_preserve_zoom(orig_rgb)
                self.original_image = orig_rgb
                h, w = orig_rgb.shape[:2]
                self.left_image_info.setText(
                    f"{os.path.basename(img_path)}  ({w}x{h})"
                )
                res_path = det.get("result_image_path")
                if res_path and os.path.exists(res_path):
                    res_bgr = cv2.imread(res_path)
                    if res_bgr is not None:
                        self.result_view.set_image_bgr_preserve_zoom(res_bgr)
                        self.result_image = res_bgr
                        rh, rw = res_bgr.shape[:2]
                        self.right_image_info.setText(
                            f"检测结果  ({rw}x{rh})"
                        )
                self._current_image_path = img_path

        # 若处于"默认适应视图"状态（zoom < 1.2，无滚动条），自动升到 2.5x
        # 让 centerOn 生效；用户自己手动放大过的状态不受影响
        _AUTO_ZOOM_TARGET = 2.5

        def ensure_zoom_for_center(view):
            if view._zoom_factor < 1.2:
                factor = _AUTO_ZOOM_TARGET
                view._zoom_factor = factor
                view.scale(factor, factor)
                view.zoom_changed.emit(factor)
                view._sync_to_linked()

        ensure_zoom_for_center(self.original_view)
        ensure_zoom_for_center(self.result_view)

        # 无论单图/批量，都把该缺陷放到视图正中央
        self.original_view.center_on(cx, cy)
        self.result_view.center_on(cx, cy)

    def update_table(self):
        self.table.setRowCount(len(self.detections))
        for i, det in enumerate(self.detections):
            # 序号 - 居中
            idx_item = QTableWidgetItem(str(i + 1))
            idx_item.setForeground(QColor("black"))
            idx_item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.table.setItem(i, 0, idx_item)

            # 缺陷类型 - 左对齐
            cn_item = QTableWidgetItem(det["class_cn"])
            cn_item.setForeground(QColor("black"))
            cn_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            self.table.setItem(i, 1, cn_item)

            # 英文名称 - 左对齐
            en_item = QTableWidgetItem(det["class"])
            en_item.setForeground(QColor("black"))
            en_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            self.table.setItem(i, 2, en_item)

            # 置信度 - 右对齐
            conf_item = QTableWidgetItem(f"{det['confidence']:.2%}")
            conf_item.setForeground(QColor("black"))
            conf_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            self.table.setItem(i, 3, conf_item)

    def save_result(self):
        if self.result_image is None:
            QMessageBox.warning(self, "警告", "没有检测结果可保存")
            return

        project_root = get_project_root()
        save_dir = os.path.join(project_root, "save_data")
        os.makedirs(save_dir, exist_ok=True)

        if self.current_image_path:
            base_name = os.path.splitext(
                os.path.basename(self.current_image_path)
            )[0]
        else:
            base_name = "result"

        save_path = os.path.join(save_dir, f"{base_name}_detected.jpg")
        cv2.imwrite(save_path, self.result_image)

        info_path = os.path.join(save_dir, f"{base_name}_info.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            img_name = (
                os.path.basename(self.current_image_path)
                if self.current_image_path else "未知"
            )
            f.write(f"检测图像: {img_name}\n")
            f.write(f"检测到缺陷总数: {len(self.detections)}\n\n")
            for i, det in enumerate(self.detections, 1):
                f.write(
                    f"{i}. {det['class_cn']}({det['class']}) "
                    f"- 置信度: {det['confidence']:.2f}\n"
                )

        self.status_bar.showMessage(f"结果已保存到: {save_path}")
        QMessageBox.information(
            self, "保存成功",
            f"检测结果图像已保存到:\n{save_path}\n\n"
            f"检测信息已保存到:\n{info_path}"
        )

    def evaluate_model(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        self.btn_evaluate.setEnabled(False)
        self.status_bar.showMessage("正在评估模型...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.bottom_tabs.setCurrentIndex(1)
        self.eval_info_text.clear()
        self.eval_info_text.append("正在运行模型评估，请稍候...")

        self.eval_thread = EvaluateThread(self.model)
        self.eval_thread.finished.connect(self.on_evaluate_finished)
        self.eval_thread.progress.connect(self.on_eval_progress)
        self.eval_thread.start()

    def on_eval_progress(self, text, current, total):
        self.status_bar.showMessage(text)
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)

    def on_evaluate_finished(self, result):
        self.btn_evaluate.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("模型评估完成")

        y_true = result["y_true"]
        y_pred = result["y_pred"]
        total = result["total_instances"]
        save_dir = result["save_dir"]

        if total == 0:
            self.eval_info_text.setHtml(
                '<div style="font-family:SimSun;font-size:14px;'
                'color:#d32f2f;padding:4px;">'
                '没有找到有效的评估数据！<br>请确保 PCB_DATASET 目录下有标注数据。</div>'
            )
            return

        class_labels = list(range(NUM_CLASSES + 1))
        cm = confusion_matrix_np(y_true, y_pred, labels=class_labels)
        CLASS_CN_NAMES = [CLASS_CN.get(name, name) for name in CLASS_NAMES]

        # --- HTML 构建：标题 + 基本信息 ---
        html = []
        html.append(
            '<div style="font-family:SimSun;font-size:17px;font-weight:bold;'
            'color:#1976d2;margin-bottom:4px;">模型评估报告</div>'
        )
        html.append(
            '<hr style="border:none;border-top:1px solid #d0d0d0;margin:4px 0 8px 0;"/>'
        )
        html.append(
            f'<div style="font-family:SimSun;font-size:14px;color:#333;'
            f'margin-bottom:2px;">总评估实例数：'
            f'<span style="color:#1976d2;font-weight:bold;">{total}</span></div>'
        )
        html.append(
            f'<div style="font-family:SimSun;font-size:14px;color:#555;'
            f'margin-bottom:8px;">保存位置：'
            f'<span style="color:#1976d2;">{save_dir}</span></div>'
        )

        # --- 各类别召回率 ---
        html.append(
            '<hr style="border:none;border-top:1px solid #e0e0e0;margin:4px 0 8px 0;"/>'
        )
        html.append(
            '<div style="font-family:SimSun;font-size:15px;font-weight:bold;'
            'color:#333;margin-bottom:4px;">各类别指标</div>'
        )
        html.append(
            '<table style="font-family:SimSun;font-size:14px;color:#333;'
            'border-collapse:collapse;width:100%;">'
        )
        html.append(
            '<tr style="background-color:#f5f5f5;">'
            '<th style="padding:6px 10px;text-align:left;border:1px solid #e0e0e0;">类别</th>'
            '<th style="padding:6px 10px;text-align:center;border:1px solid #e0e0e0;">TP</th>'
            '<th style="padding:6px 10px;text-align:center;border:1px solid #e0e0e0;">GT</th>'
            '<th style="padding:6px 10px;text-align:right;border:1px solid #e0e0e0;">召回率</th>'
            '</tr>'
        )
        for i, cls_name in enumerate(CLASS_NAMES):
            tp = int(np.sum((np.array(y_true) == i) & (np.array(y_pred) == i)))
            total_gt = int(np.sum(np.array(y_true) == i))
            recall = tp / total_gt if total_gt > 0 else 0.0
            cn = CLASS_CN.get(cls_name, cls_name)
            recall_color = "#2e7d32" if recall >= 0.8 else (
                "#f57c00" if recall >= 0.5 else "#d32f2f"
            )
            html.append(
                f'<tr><td style="padding:5px 10px;border:1px solid #e8e8e8;">'
                f'{cn} <span style="color:#888;">({cls_name})</span></td>'
                f'<td style="padding:5px 10px;text-align:center;border:1px solid #e8e8e8;">{tp}</td>'
                f'<td style="padding:5px 10px;text-align:center;border:1px solid #e8e8e8;">{total_gt}</td>'
                f'<td style="padding:5px 10px;text-align:right;border:1px solid #e8e8e8;'
                f'font-weight:bold;color:{recall_color};">{recall:.2%}</td></tr>'
            )
        html.append('</table>')

        # --- 混淆矩阵（文字版） ---
        html.append(
            '<hr style="border:none;border-top:1px solid #e0e0e0;margin:8px 0 8px 0;"/>'
        )
        html.append(
            '<div style="font-family:SimSun;font-size:15px;font-weight:bold;'
            'color:#333;margin-bottom:4px;">混淆矩阵（行=真实，列=预测）</div>'
        )
        html.append(
            '<div style="font-family:SimSun;font-size:13px;color:#666;'
            'margin-bottom:4px;">数字为每个真实类被预测为对应类的数量。右侧图表区可查看可视化结果。</div>'
        )
        header_labels = CLASS_CN_NAMES + ["背景"]
        html.append(
            '<table style="font-family:SimSun;font-size:13px;color:#333;'
            'border-collapse:collapse;margin-top:4px;">'
        )
        # header row
        header_html = '<tr><td style="padding:4px 8px;background-color:#f5f5f5;'
        header_html += 'border:1px solid #e0e0e0;font-weight:bold;">类别</td>'
        for h in header_labels:
            header_html += (
                f'<td style="padding:4px 8px;background-color:#f5f5f5;'
                f'border:1px solid #e0e0e0;font-weight:bold;text-align:center;">{h}</td>'
            )
        header_html += '</tr>'
        html.append(header_html)

        for i, row in enumerate(cm):
            label = header_labels[i] if i < len(header_labels) else str(i)
            row_html = (
                f'<tr><td style="padding:4px 8px;background-color:#fafafa;'
                f'border:1px solid #e0e0e0;font-weight:bold;">{label}</td>'
            )
            row_sum = int(np.sum(row))
            for j, v in enumerate(row):
                val = int(v)
                # diagonal = correct prediction, highlight
                if i == j:
                    bg = "#e8f5e9"
                    text_color = "#2e7d32"
                    weight = "bold"
                elif val > 0:
                    bg = "#ffebee"
                    text_color = "#d32f2f"
                    weight = "bold"
                else:
                    bg = "#ffffff"
                    text_color = "#999"
                    weight = "normal"
                row_html += (
                    f'<td style="padding:4px 8px;text-align:center;'
                    f'border:1px solid #e8e8e8;background-color:{bg};'
                    f'color:{text_color};font-weight:{weight};">{val}</td>'
                )
            row_html += '</tr>'
            html.append(row_html)
        html.append('</table>')

        self.eval_info_text.setHtml("".join(html))

        # Generate and display confusion matrix chart
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plot_confusion_matrix_chart(cm, CLASS_CN_NAMES, cm_path)
        cm_norm_path = os.path.join(save_dir, "confusion_matrix_normalized.png")
        plot_normalized_cm_chart(cm, CLASS_CN_NAMES, cm_norm_path)

        # 读取并显示到 ZoomableImageView（支持滚轮缩放 + 拖拽）
        img = cv2.imread(cm_norm_path)
        if img is not None:
            self.eval_cm_view.set_image_bgr(img)

        QMessageBox.information(
            self, "评估完成",
            f"模型评估完成！\n\n"
            f"总评估实例: {total}\n"
            f"混淆矩阵和各类指标请查看「模型评估」标签页\n"
            f"图表已保存到:\n{save_dir}"
        )

    def clear(self):
        self.original_image = None
        self.result_image = None
        self.current_image_path = None
        self.detections = []
        self.original_view.set_placeholder("请打开一张PCB板图像")
        self.original_view.clear_image()
        self.result_view.set_placeholder("检测结果将显示在这里")
        self.result_view.clear_image()
        self.left_image_info.setText("")
        self.right_image_info.setText("")
        self.left_zoom_pct.setText("100%")
        self.right_zoom_pct.setText("100%")
        self.info_text.clear()
        self.table.setRowCount(0)
        self.btn_detect.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.status_bar.showMessage("已清空 - 请打开新图像或文件夹")

    def closeEvent(self, event):
        event.accept()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("SimSun", 10))
    window = PCBDetectionGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
