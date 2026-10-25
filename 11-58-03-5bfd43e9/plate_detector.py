# -*- coding: utf-8 -*-
"""YOLOv5 专用车牌检测器 (ONNX Runtime, yakhyo/yolov5-license-plate-detection)"""
import cv2
import numpy as np
import onnxruntime as ort


def nms_xyxy(boxes, scores, iou_thr=0.5):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(1e-6, areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]
    return keep


class PlateDetector:
    def __init__(self, model_path="plate_det/anpr_yolov5s.onnx",
                 conf_thr=0.4, nms_thr=0.45, img_size=640):
        self.sess = ort.InferenceSession(model_path, providers=["DmlExecutionProvider"])
        self.conf_thr, self.nms_thr, self.img_size = conf_thr, nms_thr, img_size

    def __call__(self, img_bgr):
        h, w = img_bgr.shape[:2]
        blob = cv2.resize(img_bgr, (self.img_size, self.img_size))
        blob = blob.astype(np.float32) / 255.0
        x = blob.transpose(2, 0, 1)[None]
        out = self.sess.run(None, {"images": x})[0][0]  # (25200, 6)
        conf = out[:, 4]
        m = conf >= self.conf_thr
        if not m.any():
            return []
        dets = out[m]
        sx, sy = w / self.img_size, h / self.img_size
        boxes = np.stack([dets[:, 0] * sx - dets[:, 2] * sx / 2,
                          dets[:, 1] * sy - dets[:, 3] * sy / 2,
                          dets[:, 0] * sx + dets[:, 2] * sx / 2,
                          dets[:, 1] * sy + dets[:, 3] * sy / 2], axis=1)
        keep = nms_xyxy(boxes, dets[:, 4], self.nms_thr)
        return [(boxes[i], float(dets[i, 4])) for i in keep]
