# -*- coding: utf-8 -*-
"""
交通流量标注流水线 v2 (CPU, 无品牌)
- PP-YOLOE+ 车辆检测 -> IoU 跟踪器生成轨迹
- RapidOCR 仅检测车牌位置(不识别号码) -> 车牌颜色分类(能源类型)
- 行车道 ROI + 运动过滤(排除路边停放车辆)
- 输出: 轨迹 JSON + 贴标预览视频 + 统计
用法: python track_annotate.py dashcam.mp4 --start 0 --end 60 --out out_v2
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from classify_plate_color import classify_plate_color

VEH_CLASSES = {2: "car", 5: "bus", 7: "truck"}  # 仅四轮机动车; 两轮车(摩托/电动)与自行车不统计
ENERGY_TEXT = {"green_nev": "新能源", "blue_fuel": "燃油", "yellow": "黄牌",
               "white": "白牌", "black": "黑牌", "unknown": ""}
ENERGY_COLOR = {"green_nev": (60, 180, 60), "blue_fuel": (230, 120, 40),
                "yellow": (30, 200, 230), "unknown": (150, 150, 150)}

# 行车道 ROI (1600x900 坐标, 依据首段视频标定, 可用 --roi 覆盖)
DEFAULT_ROI = [(0, 900), (0, 560), (250, 470), (1150, 460), (1600, 560), (1600, 900)]
# 对向车道排除区 (沿中央护栏走向, 可用 --exclude-zone 覆盖)
DEFAULT_EXCLUDE = [(0, 0), (0, 900), (750, 900), (313, 650), (188, 544), (625, 506)]
TRACK_MAX_MISSED = 25  # 轨迹存活: 25 个处理帧
REVIVE_WINDOW = 48     # 失活后多少处理帧内允许外观复活 (遮挡恢复, ≈2秒)
REID_CAND = 0.45       # 进入候选的最小相似度
REID_THRESH = 0.60     # 融合指纹匹配接受阈值
GATE_A = 0.55          # 通道A运动匹配的外观门禁 (同车>=0.83, 异车0.36-0.45, 白色同款车曾致ID劫持)
REID_MIN_WIDTH = 40    # 小于此宽度不提取外观指纹
MIN_ACTIVE_WIDTH = 50  # 激活阈值: 框宽(原图像素)达到此值才计入统计/标注/投票
MIN_TRACK_SECONDS = 0.7  # 最短存活时长
MIN_DISPLACEMENT = 10  # 最短位移(像素): 只排除静止目标(停放/等灯), 同向慢速车不再误伤
MIN_PLATE_W, MIN_PLATE_H = 18, 8   # 车牌最小尺寸
BRAND_MIN_WIDTH = 55   # 车型识别最小车宽
BRAND_EVERY = 16       # 每 N 个处理帧做一轮批量车型识别
BRAND_AGREE_BONUS = 0.12   # 指纹融合: 车型一致加分
BRAND_CONFLICT_PENALTY = 0.15  # 指纹融合: 车型冲突罚分
FONT = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 26)
FONT_SMALL = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 18)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def box_center_in(poly, box):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0


class Track:
    _next_id = 1
    fps = 24  # 由 main 设置

    def __init__(self, box):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.first_box = box
        self.vel = [0.0, 0.0, 0.0, 0.0]  # 框速度(每处理帧)
        self.history = []          # [(frame_idx, box)]
        self.votes = []            # 能源类型投票
        self.missed = 0
        self.active = True
        self.energy = "unknown"
        self.embedding = None      # 外观指纹 (512维, L2归一化)
        self.brand_votes = {}      # 车型类别投票 {class_idx: count}
        self.activated = (box[2] - box[0]) >= MIN_ACTIVE_WIDTH  # 近处才激活
        self.counted_flag = False  # 是否已触发实时计数(音效+面板)

    @property
    def energy_stable(self):
        """能源判定是否已稳定 (票数>=3 且领先项占比>=2/3)"""
        if len(self.votes) < 3:
            return False
        top = max(set(self.votes), key=self.votes.count)
        return self.votes.count(top) / len(self.votes) >= 0.66

    @property
    def brand_top(self):
        return max(self.brand_votes, key=self.brand_votes.get) if self.brand_votes else None

    @property
    def brand_confident(self):
        if not self.brand_votes:
            return False
        tot = sum(self.brand_votes.values())
        top = self.brand_votes[max(self.brand_votes, key=self.brand_votes.get)]
        return tot >= 3 and top / tot >= 0.6

    def vote_brand(self, cls_idx):
        self.brand_votes[cls_idx] = self.brand_votes.get(cls_idx, 0) + 1

    @property
    def lifetime_s(self):
        return (self.history[-1][0] - self.history[0][0]) / Track.fps

    def predicted_box(self):
        k = 1 + self.missed
        return [self.box[i] + self.vel[i] * k for i in range(4)]

    def update(self, box, feat=None):
        for i in range(4):
            self.vel[i] = 0.6 * self.vel[i] + 0.4 * (box[i] - self.box[i])
        self.box = box
        if not self.activated and (box[2] - box[0]) >= MIN_ACTIVE_WIDTH:
            self.activated = True  # 驶近激活: 从此计入统计/标注/投票
        if feat is not None:
            if self.embedding is None:
                self.embedding = feat
            else:
                e = 0.8 * self.embedding + 0.2 * feat
                self.embedding = e / (np.linalg.norm(e) + 1e-9)

    def vote(self, energy):
        if energy != "unknown":
            self.votes.append(energy)
            self.energy = max(set(self.votes), key=self.votes.count)

    def absorb(self, other):
        """把另一条( fragment )轨迹并入自己"""
        self.history.extend(other.history)
        self.history.sort(key=lambda h: h[0])
        self.votes.extend(other.votes)
        for k, v in other.brand_votes.items():
            self.brand_votes[k] = self.brand_votes.get(k, 0) + v
        self.activated = self.activated or other.activated
        if self.votes:
            self.energy = max(set(self.votes), key=self.votes.count)
        if other.embedding is not None:
            self.update(self.box, other.embedding)

    def finalize(self):
        if self.votes:
            self.energy = max(set(self.votes), key=self.votes.count)

    @property
    def displacement(self):
        cx1 = (self.first_box[0] + self.first_box[2]) / 2
        cy1 = (self.first_box[1] + self.first_box[3]) / 2
        cx2 = (self.box[0] + self.box[2]) / 2
        cy2 = (self.box[1] + self.box[3]) / 2
        return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5


def draw_label(frame_bgr, labeled, pending):
    """labeled: [(box, track_id, energy, brand)] 已判定能源 -> 车顶彩色倒三角箭头
       pending: [(box, track_id)] 已识别未判定 -> 灰色方框"""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    # 未判定: 灰色方框
    for box, tid in pending:
        x1, y1, x2, y2 = [int(v) for v in box]
        d.rectangle([x1, y1, x2, y2], outline=(150, 150, 150), width=2)
        tb = d.textbbox((0, 0), f"#{tid}", font=FONT_SMALL)
        d.rectangle([x1, y1 - 22, x1 + (tb[2] - tb[0]) + 8, y1], fill=(110, 110, 110))
        d.text((x1 + 4, y1 - 19), f"#{tid}", font=FONT_SMALL, fill=(255, 255, 255))
    # 已判定: 车顶倒三角箭头 + 标签(不再画框)
    for box, tid, energy, brand in labeled:
        x1, y1, x2, y2 = [int(v) for v in box]
        bw = x2 - x1
        color = ENERGY_COLOR.get(energy, (150, 150, 150))
        cx = (x1 + x2) / 2
        tw = min(max(bw * 0.28, 16), 44)
        th = tw * 0.85
        d.polygon([(cx, y1 + 1), (cx - tw / 2, y1 - th), (cx + tw / 2, y1 - th)],
                  fill=color, outline=(255, 255, 255))
        small = bw < 110
        parts = [f"#{tid}"]
        if not small:
            if brand:
                parts.append(brand)
            parts.append(ENERGY_TEXT[energy])
        text = " ".join(parts)
        font = FONT_SMALL if small else FONT
        tb = d.textbbox((0, 0), text, font=font)
        lh = 24 if small else 34
        twd = tb[2] - tb[0]
        d.rectangle([cx - twd / 2 - 4, y1 - th - lh, cx + twd / 2 + 4, y1 - th],
                    fill=color)
        d.text((cx - twd / 2, y1 - th - lh + 3), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------- 车型/品牌识别 (ONNX ResNet34, erquren/vehicle_recognition) ----------------
_BRAND_SESS = None
_BRAND_NAMES = None
_BRAND_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_BRAND_STD = np.array([0.229, 0.224, 0.225], np.float32)


def init_brand(model_dir="brand_model"):
    global _BRAND_SESS, _BRAND_NAMES
    import onnxruntime as ort
    _BRAND_SESS = ort.InferenceSession(os.path.join(model_dir, "vehicle_brand.onnx"),
                                       providers=["CPUExecutionProvider"])
    with open(os.path.join(model_dir, "class_car.json"), encoding="utf-8") as f:
        _BRAND_NAMES = json.load(f)


def classify_brand(frame, box):
    if _BRAND_SESS is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
    x = ((crop - _BRAND_MEAN) / _BRAND_STD).transpose(2, 0, 1)[None].astype(np.float32)
    logits = _BRAND_SESS.run(None, {"input": x})[0][0]
    return int(logits.argmax())


def classify_brand_batch(frame, boxes):
    """批量车型识别: 一次 ONNX 调用处理多辆车"""
    if _BRAND_SESS is None:
        return [None] * len(boxes)
    crops = []
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            crops.append(None)
            continue
        x = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
        crops.append(((x - _BRAND_MEAN) / _BRAND_STD).transpose(2, 0, 1))
    valid = [c for c in crops if c is not None]
    if not valid:
        return [None] * len(boxes)
    out = _BRAND_SESS.run(None, {"input": np.stack(valid).astype(np.float32)})[0]
    it = iter(int(v) for v in out.argmax(1))
    return [None if c is None else next(it) for c in crops]


# ---------------- 车辆专用重识别 (OSNet-AIN, VeRi-776 训练, OpenVINO 模型库) ----------------
_REID_SESS = None


def init_reid(path="reid/osnet_vehicle_reid.onnx"):
    global _REID_SESS
    import onnxruntime as ort
    _REID_SESS = ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def reid_features(frame, boxes):
    """批量提取外观指纹 -> {det_idx: 512维L2归一化向量} (太小的目标跳过)"""
    crops, idxs = [], []
    for i, b in enumerate(boxes):
        if b[2] - b[0] < REID_MIN_WIDTH:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        x = cv2.resize(crop, (208, 208)).astype(np.float32) / 255.0
        crops.append(((x - _BRAND_MEAN) / _BRAND_STD).transpose(2, 0, 1))
        idxs.append(i)
    if not crops:
        return {}
    out = _REID_SESS.run(None, {"input": np.stack(crops).astype(np.float32)})[0]
    return {i: out[k] / (np.linalg.norm(out[k]) + 1e-9) for k, i in enumerate(idxs)}


# ---------------- 左上角实时计数面板 ----------------
def draw_hud(frame_bgr, counts):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0 = 12, 12
    row_h = 34
    rows = [("新能源", counts["green_nev"], (60, 180, 60)),
            ("燃油车", counts["blue_fuel"], (230, 120, 40)),
            ("未判定", counts["unknown"], (140, 140, 140))]
    w = 190
    h = row_h * (len(rows) + 1) + 10
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=(0, 0, 0, 140))
    total = counts.get("total", sum(counts.values()))
    d.text((x0 + 12, y0 + 6), f"通过车辆 {total}", font=FONT, fill=(255, 255, 255))
    for i, (label, n, color) in enumerate(rows):
        yy = y0 + row_h * (i + 1) + 4
        d.rectangle([x0 + 12, yy + 6, x0 + 26, yy + 20], fill=color + (255,))
        d.text((x0 + 34, yy), f"{label}  {n}", font=FONT, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)



def brand_short(idx):
    if idx is None or _BRAND_NAMES is None:
        return ""
    import re
    return re.sub(r"_?20\d\d款?$", "", _BRAND_NAMES[str(idx)])[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=60.0)
    ap.add_argument("--step", type=int, default=2, help="每 N 帧处理 1 帧")
    ap.add_argument("--plate-every", type=int, default=6, help="每 N 个处理帧做一次车牌检测")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default="out_v2")
    ap.add_argument("--roi", default=None, help="ROI 多边形 json 文件路径")
    ap.add_argument("--exclude-zone", default=None, help="排除区多边形 json (对向车道等)")
    ap.add_argument("--state", default=None,
                    help="轨迹状态 pickle 路径: 存在则续跑, 退出时保存 (用于分段处理长视频)")
    ap.add_argument("--show-brand", action="store_true",
                    help="在画面标注中显示车型/品牌 (默认关闭: 该模型在记录仪画面上准确率不足, 仅作内部指纹)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    roi = np.array(json.load(open(args.roi)) if args.roi else DEFAULT_ROI, np.int32)
    excl = np.array(json.load(open(args.exclude_zone)) if args.exclude_zone
                    else DEFAULT_EXCLUDE, np.int32)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    Track.fps = fps

    def qualifies(t):
        """是否满足计入条件 (激活+存活+位移+ROI+非排除区)"""
        if not t.activated or t.lifetime_s < MIN_TRACK_SECONDS:
            return False
        if t.displacement < MIN_DISPLACEMENT or getattr(t, "in_exclude", False):
            return False
        x1, y1, x2, y2 = t.box
        return cv2.pointPolygonTest(roi, ((x1 + x2) / 2, y2), False) >= 0

    def uncount_merged(b):
        """b 被并入其他轨迹: 撤销其计数 (总数必减; 已归档分类则对应分类也减)"""
        if getattr(b, "counted_flag", None) is True:
            counts["total"] -= 1
            key = getattr(b, "counted_key", None)
            if key is not None:
                counts[key] -= 1
            b.counted_flag = "spent"

    from paddlex import create_model
    det_model = create_model("PP-YOLOE_plus-S")
    init_reid()   # 车辆专用外观指纹 (OSNet-AIN, VeRi-776)
    init_brand()
    from plate_detector import PlateDetector
    plate_det = PlateDetector()

    video_name = (f"annotated_{args.start:g}s_{args.end:g}s.mp4" if args.state
                  else "annotated.mp4")
    vw = cv2.VideoWriter(os.path.join(args.out, video_name),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps / args.step, (1280, 720))

    tracks = []
    pending = []
    counts = {"total": 0, "green_nev": 0, "blue_fuel": 0, "unknown": 0}
    if args.state and os.path.exists(args.state):
        import pickle
        with open(args.state, "rb") as f:
            _st = pickle.load(f)
        tracks, pending = _st["tracks"], _st["pending"]
        Track._next_id = _st["next_id"]
        counts.update(_st.get("counts", {}))
        print(f"续跑: 载入 {len(tracks)} 条轨迹, 下一路径 ID={Track._next_id}")
    end_frame = int((args.end - args.start) * fps)
    frame_idx = 0
    analyzed = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= end_frame:
            break
        frame_idx += 1
        if frame_idx % args.step != 0:
            continue
        analyzed += 1
        gframe = int(args.start * fps) + frame_idx  # 全局帧号

        res = list(det_model.predict(frame))[0]
        dets = []
        boxes = res.get("boxes", [])
        items = boxes.get("label", []) if isinstance(boxes, dict) else boxes
        for b in (items or []):
            if b["cls_id"] in VEH_CLASSES and b["score"] >= args.conf:
                dets.append([float(v) for v in b["coordinate"]] + [float(b["score"])])

        # --- 外观指纹提取 (批量, 太小目标无意义跳过) ---
        feats = [None] * len(dets)
        for i, f in reid_features(frame, [d[:4] for d in dets]).items():
            feats[i] = f

        # --- 双通道匹配: 先运动(IoU), 再外观指纹(ReID) ---
        dets.sort(key=lambda d: -d[4])
        order = sorted(range(len(dets)), key=lambda i: -dets[i][4])
        matched = set()
        used_dets = set()

        # 通道 A: 运动匹配 (仅近期可见轨迹; 有指纹时加外观门禁防拥堵场景 ID 跳变)
        for di in order:
            det = dets[di]
            best, best_iou = None, 0.15
            dcx = (det[0] + det[2]) / 2
            dcy = (det[1] + det[3]) / 2
            for t in tracks:
                if t.id in matched or not t.active or t.missed > 6:
                    continue
                # 外观门禁: 相邻帧间同车指纹必然相近, 明显不同则拒绝运动匹配
                if (feats[di] is not None and t.embedding is not None
                        and float(np.dot(feats[di], t.embedding)) < GATE_A):
                    continue
                pred = t.predicted_box()
                v = iou(pred, det[:4])
                if v < 0.15:
                    # 退化准则: 检测框中心落在预测框 1.5 倍范围内也接受
                    cx = (pred[0] + pred[2]) / 2
                    cy = (pred[1] + pred[3]) / 2
                    hw = (pred[2] - pred[0]) * 0.75
                    hh = (pred[3] - pred[1]) * 0.75
                    if abs(dcx - cx) <= hw and abs(dcy - cy) <= hh:
                        v = 0.16
                if v > best_iou:
                    best, best_iou = t, v
            if best:
                best.update(det[:4], feats[di])
                best.missed = 0
                best.history.append((gframe, det[:4]))
                matched.add(best.id)
                used_dets.add(di)

        # 通道 B: 融合指纹匹配 (外观 ReID + 车型类别 -> 找回丢失的轨迹)
        det_brands = {}  # di -> 车型类别 (按需惰性识别)

        def det_brand(di):
            if di not in det_brands:
                det_brands[di] = (classify_brand(frame, dets[di][:4])
                                  if dets[di][2] - dets[di][0] >= BRAND_MIN_WIDTH
                                  else None)
            return det_brands[di]

        cands = []
        for di in order:
            if di in used_dets or feats[di] is None:
                continue
            for t in tracks:
                if t.id in matched or t.embedding is None:
                    continue
                # 失活不久(刚被遮挡)的轨迹也参与匹配, 命中即复活, 保留原ID和投票
                if not t.active and analyzed - getattr(t, "died_at", -10 ** 9) > REVIVE_WINDOW:
                    continue
                sim = float(np.dot(feats[di], t.embedding))
                if sim < REID_CAND:
                    continue
                fused = sim
                if t.brand_confident:
                    db = det_brand(di)
                    if db is not None:
                        if db == t.brand_top:
                            fused += BRAND_AGREE_BONUS   # 三路指纹互证: 加分
                        else:
                            fused -= BRAND_CONFLICT_PENALTY  # 车型冲突: 罚分
                if fused >= REID_THRESH:
                    cands.append((fused, sim, di, t))
        cands.sort(key=lambda c: -c[0])
        for fused, sim, di, t in cands:
            if di in used_dets or t.id in matched:
                continue
            det = dets[di]
            was_dead = not t.active
            t.update(det[:4], feats[di])
            t.active = True
            t.missed = 0
            t.history.append((gframe, det[:4]))
            if was_dead:
                t.revived = getattr(t, "revived", 0) + 1
            if det_brands.get(di) is not None:
                t.vote_brand(det_brands[di])
            matched.add(t.id)
            used_dets.add(di)

        # 已激活轨迹的定期批量自识别: 持续累积车型投票
        if analyzed % BRAND_EVERY == 0:
            targets = [t for t in tracks
                       if t.active and t.activated and not t.in_exclude
                       and t.box[2] - t.box[0] >= BRAND_MIN_WIDTH]
            for t, cls in zip(targets, classify_brand_batch(frame, [t.box for t in targets])):
                if cls is not None:
                    t.vote_brand(cls)

        # 未匹配的检测 -> 待确认队列
        for di in order:
            if di in used_dets:
                continue
            det = dets[di]
            hit = False
            for p in pending:
                if iou(p["box"], det[:4]) > 0.3:
                    p["hits"] += 1
                    p["age"] = 0
                    p["box"] = det[:4]
                    hit = True
                    if p["hits"] >= 2:
                        dup = next((t for t in tracks
                                    if t.active and iou(t.box, p["box"]) > 0.5), None)
                        if dup is not None:
                            # 同位置已有轨迹: 并入而不是新建
                            dup.update(det[:4])
                            dup.missed = 0
                            dup.history.append((gframe, det[:4]))
                            if feats[di] is not None:
                                dup.update(dup.box, feats[di])
                        else:
                            t = Track(det[:4])
                            t.history.append((gframe, det[:4]))
                            t.embedding = feats[di]
                            tracks.append(t)
                        pending.remove(p)
                    break
            if not hit:
                pending.append({"box": det[:4], "hits": 1, "age": 0})
        for p in pending:
            p["age"] += 1
        pending[:] = [p for p in pending if p["age"] <= 1]
        for t in tracks:
            if t.active and t.id not in matched:
                t.missed += 1
                if t.missed > TRACK_MAX_MISSED:
                    t.active = False
                    t.died_at = analyzed

        # 对向车道(排除区内)的车辆: 不分析不标注
        for t in tracks:
            t.in_exclude = box_center_in(excl, t.box)

        # 同帧去重: 同中心且(高重叠或一方包含另一方) -> 同辆车的碎片, 合并
        # (框形差异大的重复检测 IoU 可能不高, 用中心距+包含关系兜底; 外观明显不同的除外)
        act = [t for t in tracks if t.active]
        for i in range(len(act)):
            if not act[i].active:
                continue
            for j in range(i + 1, len(act)):
                if not act[j].active:
                    continue
                a, b = act[i], act[j]
                ca = [(a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2]
                cb = [(b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2]
                cdist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                if cdist > 40:
                    continue
                inter_w = max(0, min(a.box[2], b.box[2]) - max(a.box[0], b.box[0]))
                inter_h = max(0, min(a.box[3], b.box[3]) - max(a.box[1], b.box[1]))
                inter = inter_w * inter_h
                area_a = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1])
                area_b = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1])
                contain = inter / max(1e-6, min(area_a, area_b))
                if iou(a.box, b.box) <= 0.6 and contain <= 0.6:
                    continue
                if (a.embedding is not None and b.embedding is not None
                        and float(np.dot(a.embedding, b.embedding)) < REID_CAND):
                    continue
                if a.id > b.id:
                    a, b = b, a
                a.absorb(b)
                b.active = False
                uncount_merged(b)

        # --- 车牌能源类型投票 (拥堵自适应频率: 活跃车越多判帧越密, 多轮投票累积) ---
        n_active = len([t for t in tracks
                        if t.active and t.activated and not t.in_exclude])
        plate_every = (1 if n_active >= 12 else 2 if n_active >= 8
                       else 4 if n_active >= 4 else args.plate_every)
        if analyzed % plate_every == 0:
            for (px1, py1, px2, py2), pconf in plate_det(frame):
                if px2 - px1 < MIN_PLATE_W or py2 - py1 < MIN_PLATE_H:
                    continue
                cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
                px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
                pad = 3
                crop = frame[max(0, py1-pad):py2+pad, max(0, px1-pad):px2+pad]
                if crop.size == 0:
                    continue
                energy, _ = classify_plate_color(crop)
                for t in tracks:
                    if not t.active or t.in_exclude or not t.activated:
                        continue
                    x1, y1, x2, y2 = t.box
                    if x1 - 10 <= cx <= x2 + 10 and y1 - 10 <= cy <= y2 + 10:
                        t.vote(energy)

        # --- 实时计数: 挂标签即计总数(只增不减); 分类归档/迁移见下方 ---
        for t in tracks:
            if t.counted_flag or not qualifies(t):
                continue
            if t.active and t.energy == "unknown":
                continue
            t.counted_flag = True
            counts["total"] += 1

        # --- 分类归档: 已计总数的轨迹, 票数稳定(或已失活)时归入分类格;
        #     已归档后翻牌且重新稳定 -> 分类迁移, 总数不变 ---
        for t in tracks:
            if t.counted_flag is not True:
                continue
            cur = t.energy if t.energy in ("green_nev", "blue_fuel") else "unknown"
            if getattr(t, "counted_key", None) is None:
                if t.energy_stable or not t.active:
                    t.counted_key = cur
                    counts[cur] += 1
            elif t.energy_stable and t.counted_key != cur:
                counts[t.counted_key] -= 1
                counts[cur] += 1
                t.counted_key = cur

        # --- 绘制输出帧 ---
        # 已判定能源 -> 车顶倒三角; 已识别未判定 -> 灰色方框 (均需激活/非排除区/ROI内)
        in_roi_box = lambda b: cv2.pointPolygonTest(
            roi, ((b[0] + b[2]) / 2, b[3]), False) >= 0
        labeled = [t for t in tracks if t.active and not t.in_exclude
                   and t.activated and t.energy != "unknown" and in_roi_box(t.box)]
        undecided = [t for t in tracks if t.active and not t.in_exclude
                     and t.activated and t.energy == "unknown" and in_roi_box(t.box)]
        disp = cv2.resize(frame, (1280, 720))
        scale = 1280 / frame.shape[1]
        disp_items = [([v * scale for v in t.box], t.id, t.energy,
                       brand_short(t.brand_top) if (args.show_brand and t.brand_confident) else "")
                      for t in labeled]
        disp_pend = [([v * scale for v in t.box], t.id) for t in undecided]
        vw.write(draw_hud(draw_label(disp, disp_items, disp_pend), counts))
        if analyzed % 50 == 0:
            print(f"[{time.time()-t0:.0f}s] 处理 {analyzed} 帧, 活跃轨迹 {len(labeled)}, "
                  f"累计轨迹 {len(tracks)}", flush=True)

    vw.release()
    cap.release()

    # 收尾: 满足条件但始终未挂标签的轨迹补计总数; 所有已计数轨迹强制归档分类
    for t in tracks:
        if not t.counted_flag and qualifies(t):
            t.counted_flag = True
            counts["total"] += 1
    for t in tracks:
        if t.counted_flag is True and getattr(t, "counted_key", None) is None:
            key = t.energy if t.energy in ("green_nev", "blue_fuel") else "unknown"
            t.counted_key = key
            counts[key] += 1

    # --- 收尾: 断轨拼接 + 过滤 + 导出 ---
    # 时间相近且首尾位置衔接的轨迹 -> 拼接回同一辆车
    merged = True
    while merged:
        merged = False
        tracks.sort(key=lambda t: t.history[0][0])
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                gap = b.history[0][0] - a.history[-1][0]
                if 0 < gap <= 40 and iou(a.history[-1][1], b.history[0][1]) > 0.35:
                    a.absorb(b)
                    uncount_merged(b)
                    tracks.remove(b)
                    merged = True
                    break
            if merged:
                break

    if args.state:
        import pickle
        with open(args.state, "wb") as f:
            pickle.dump({"tracks": tracks, "pending": pending,
                         "next_id": Track._next_id, "counts": counts}, f)

    for t in tracks:
        t.finalize()

    def in_roi(t):
        x1, y1, x2, y2 = t.box
        return cv2.pointPolygonTest(roi, ((x1 + x2) / 2, y2), False) >= 0

    counted = [t for t in tracks
               if t.activated and t.lifetime_s >= MIN_TRACK_SECONDS
               and t.displacement > MIN_DISPLACEMENT and in_roi(t)
               and not getattr(t, "in_exclude", False)]

    # 最终保险: 已计入轨迹间的重叠去重 (框形差异大的重复检测在逐帧阶段可能漏网)
    merged = True
    while merged:
        merged = False
        for i in range(len(counted)):
            for j in range(i + 1, len(counted)):
                a, b = counted[i], counted[j]
                fa = {h[0]: h[1] for h in a.history}
                fb = {h[0]: h[1] for h in b.history}
                common = set(fa) & set(fb)
                if len(common) < 10:
                    continue
                ds = [((fa[f][0] + fa[f][2] - fb[f][0] - fb[f][2]) / 2) ** 2
                      + ((fa[f][1] + fa[f][3] - fb[f][1] - fb[f][3]) / 2) ** 2
                      for f in common]
                if (sum(ds) / len(ds)) ** 0.5 > 40:
                    continue
                if (a.embedding is not None and b.embedding is not None
                        and float(np.dot(a.embedding, b.embedding)) < REID_CAND):
                    continue
                a.absorb(b)
                uncount_merged(b)
                counted.remove(b)
                merged = True
                break
            if merged:
                break

    excluded_oncoming = [t for t in tracks
                         if t not in counted and getattr(t, "in_exclude", False)]
    too_far = [t for t in tracks
               if t not in counted and not getattr(t, "in_exclude", False)
               and not t.activated]
    parked = [t for t in tracks
              if t not in counted and t not in excluded_oncoming and t not in too_far]

    traj = [{
        "track_id": t.id, "cls": "vehicle",
        "energy_type": getattr(t, "counted_key", None) or t.energy,
        "brand_model": brand_short(t.brand_top) if t.brand_votes else "",
        "n_energy_votes": len(t.votes),
        "appear_s": [round(t.history[0][0] / fps, 1),
                     round(t.history[-1][0] / fps, 1)],
        "trajectory": [[fr] + [round(v, 1) for v in box] for fr, box in t.history],
    } for t in counted]
    with open(os.path.join(args.out, "trajectories.json"), "w", encoding="utf-8") as f:
        json.dump({"video_fps": fps, "frames_range": [args.start, args.end],
                   "tracks": traj}, f, ensure_ascii=False)

    def cnt(e):
        return sum(1 for t in counted
                   if (getattr(t, "counted_key", None) or t.energy) == e)
    stats = {
        "segment_s": [args.start, args.end],
        "vehicles_moving_in_lane": counts["total"],
        "excluded_oncoming_zone": len(excluded_oncoming),
        "ignored_too_far": len(too_far),
        "excluded_parked_or_short": len(parked),
        "nev_green": cnt("green_nev"), "fuel_blue": cnt("blue_fuel"),
        "yellow": cnt("yellow"), "unknown_color": cnt("unknown"),
        "hud_counts": counts,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("\n===== 统计 =====")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"输出: {args.out}\\annotated.mp4 | trajectories.json | stats.json")


if __name__ == "__main__":
    main()
