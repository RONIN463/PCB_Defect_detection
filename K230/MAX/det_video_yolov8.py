# -*- coding: utf-8 -*-

import gc
import time
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.PipeLine import PipeLine
from libs.Utils import *
import nncase_runtime as nn
import ulab.numpy as np


try:
    ALIGN_UP
except NameError:
    def ALIGN_UP(x, align):
        return (x + align - 1) // align * align


display_mode = "lcd"
rgb888p_size = [1280, 720]
root_path = "/sdcard/mp_deployment_source/"

deploy_conf = read_json(root_path + "/deploy_config.json")
kmodel_path = root_path + deploy_conf["kmodel_path"]
labels = deploy_conf["categories"]
confidence_threshold = deploy_conf["confidence_threshold"]
nms_threshold = deploy_conf["nms_threshold"]
model_input_size = deploy_conf["img_size"]
max_candidates = 80
max_detections = 20
gc_interval = 10


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0
    return inter / union


class YOLOv8DetectionApp(AIBase):
    def __init__(self, mode, kmodel_path, labels, model_input_size, confidence_threshold,
                 nms_threshold, rgb888p_size, display_size, debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.mode = mode
        self.labels = labels
        self.num_classes = len(labels)
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.color_four = get_colors(len(self.labels))
        self.cur_result = {"boxes": [], "scores": [], "idx": []}
        self.fps = 0
        self._frame_count = 0
        self._fps_tick = time.ticks_ms()
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right, _ = center_pad_param(ai2d_input_size, self.model_input_size)
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build(
                [1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                [1, 3, self.model_input_size[1], self.model_input_size[0]]
            )

    def _get_value(self, pred, channels_first, c, i):
        if channels_first:
            return pred[c][i]
        return pred[i][c]

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            self.cur_result = {"boxes": [], "scores": [], "idx": []}
            pred = results[0]
            shape = pred.shape
            if self.debug_mode > 0:
                print("yolov8 output shape:", shape)

            if len(shape) == 3:
                pred = pred[0]
                shape = pred.shape

            if len(shape) != 2:
                print("Unsupported YOLOv8 output shape:", shape)
                return self.cur_result

            expected_channels = 4 + self.num_classes
            if shape[0] == expected_channels:
                channels_first = True
                num_boxes = shape[1]
            elif shape[1] == expected_channels:
                channels_first = False
                num_boxes = shape[0]
            else:
                print("Unexpected YOLOv8 output shape:", shape, "classes:", self.num_classes)
                return self.cur_result

            in_w = self.rgb888p_size[0]
            in_h = self.rgb888p_size[1]
            net_w = self.model_input_size[0]
            net_h = self.model_input_size[1]
            scale = min(net_w / in_w, net_h / in_h)
            inv_scale = 1 / scale
            pad_x = (net_w - in_w * scale) / 2
            pad_y = (net_h - in_h * scale) / 2

            candidates = []
            if channels_first:
                box_x = pred[0]
                box_y = pred[1]
                box_w = pred[2]
                box_h = pred[3]
                cls_scores = []
                for c in range(self.num_classes):
                    cls_scores.append(pred[4 + c])

                for i in range(num_boxes):
                    best_score = 0
                    best_cls = 0
                    for c in range(self.num_classes):
                        score = cls_scores[c][i]
                        if score > best_score:
                            best_score = score
                            best_cls = c
                    if best_score < self.confidence_threshold:
                        continue

                    cx = box_x[i]
                    cy = box_y[i]
                    w = box_w[i]
                    h = box_h[i]

                    x1 = (cx - w / 2 - pad_x) * inv_scale
                    y1 = (cy - h / 2 - pad_y) * inv_scale
                    x2 = (cx + w / 2 - pad_x) * inv_scale
                    y2 = (cy + h / 2 - pad_y) * inv_scale

                    x1 = int(max(0, min(in_w - 1, x1)))
                    y1 = int(max(0, min(in_h - 1, y1)))
                    x2 = int(max(0, min(in_w - 1, x2)))
                    y2 = int(max(0, min(in_h - 1, y2)))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    candidates.append([best_score, best_cls, [x1, y1, x2, y2]])
            else:
                for i in range(num_boxes):
                    item = pred[i]
                    best_score = 0
                    best_cls = 0
                    for c in range(self.num_classes):
                        score = item[4 + c]
                        if score > best_score:
                            best_score = score
                            best_cls = c
                    if best_score < self.confidence_threshold:
                        continue

                    cx = item[0]
                    cy = item[1]
                    w = item[2]
                    h = item[3]

                    x1 = (cx - w / 2 - pad_x) * inv_scale
                    y1 = (cy - h / 2 - pad_y) * inv_scale
                    x2 = (cx + w / 2 - pad_x) * inv_scale
                    y2 = (cy + h / 2 - pad_y) * inv_scale

                    x1 = int(max(0, min(in_w - 1, x1)))
                    y1 = int(max(0, min(in_h - 1, y1)))
                    x2 = int(max(0, min(in_w - 1, x2)))
                    y2 = int(max(0, min(in_h - 1, y2)))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    candidates.append([best_score, best_cls, [x1, y1, x2, y2]])

            candidates.sort(key=lambda item: item[0], reverse=True)
            candidates = candidates[:max_candidates]

            picked = []
            for cand in candidates:
                keep = True
                for prev in picked:
                    if cand[1] == prev[1] and box_iou(cand[2], prev[2]) > self.nms_threshold:
                        keep = False
                        break
                if keep:
                    picked.append(cand)
                if len(picked) >= max_detections:
                    break

            for score, cls_id, box in picked:
                self.cur_result["scores"].append(score)
                self.cur_result["idx"].append(cls_id)
                self.cur_result["boxes"].append(box)
            return self.cur_result

    def draw_result(self, draw_img, res):
        with ScopedTiming("draw osd", self.debug_mode > 0):
            if self.mode == "video":
                draw_img.clear()
            draw_img.draw_string_advanced(5, 5, 24, "FPS " + str(round(self.fps, 1)), color=(0, 255, 0))
            for i in range(len(res["boxes"])):
                x = int(res["boxes"][i][0] * self.display_size[0] // self.rgb888p_size[0])
                y = int(res["boxes"][i][1] * self.display_size[1] // self.rgb888p_size[1])
                w = int((res["boxes"][i][2] - res["boxes"][i][0]) * self.display_size[0] // self.rgb888p_size[0])
                h = int((res["boxes"][i][3] - res["boxes"][i][1]) * self.display_size[1] // self.rgb888p_size[1])
                cls_id = res["idx"][i]
                draw_img.draw_rectangle(x, y, w, h, color=self.color_four[cls_id])
                draw_img.draw_string_advanced(
                    x, y - 50, 24,
                    self.labels[cls_id] + " " + str(round(res["scores"][i], 2)),
                    color=self.color_four[cls_id]
                )

    def update_fps(self):
        self._frame_count += 1
        now = time.ticks_ms()
        diff = time.ticks_diff(now, self._fps_tick)
        if diff >= 1000:
            self.fps = self._frame_count * 1000 / diff
            self._frame_count = 0
            self._fps_tick = now


inference_mode = "video"
debug_mode = 0

pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
pl.create()
display_size = pl.get_display_size()

det_app = YOLOv8DetectionApp(
    inference_mode, kmodel_path, labels, model_input_size,
    confidence_threshold, nms_threshold, rgb888p_size, display_size,
    debug_mode=debug_mode
)
det_app.config_preprocess()

frame_id = 0
while True:
    with ScopedTiming("total", debug_mode > 0):
        img = pl.get_frame()
        res = det_app.run(img)
        det_app.update_fps()
        det_app.draw_result(pl.osd_img, res)
        pl.show_image()
        frame_id += 1
        if frame_id % gc_interval == 0:
            gc.collect()

det_app.deinit()
pl.destroy()
