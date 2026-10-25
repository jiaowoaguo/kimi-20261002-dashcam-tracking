# -*- coding: utf-8 -*-
"""
行车记录仪车牌分析原型流水线
- 抽帧 -> 车牌检测+OCR (PaddleOCR PP-OCRv5)
- 车牌颜色分类: 蓝(燃油) / 绿(新能源) / 黄 / 白 / 黑
- 跨帧去重, 输出 CSV / JSON 统计 + 标注样例图
用法: python analyze.py <视频路径> [--step 10] [--max-width 1280] [--out output]
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import cv2
import numpy as np

PLATE_RE = re.compile(r"^[一-龥][A-Z][A-Z0-9]{5,6}$")

# 颜色参考: 蓝牌 hue~110, 绿牌 hue~70, 黄牌 hue~25 (OpenCV HSV, H:0-179)
COLOR_RANGES = [
    ("green_nev", 45, 90),
    ("blue_fuel", 95, 125),
    ("yellow", 15, 40),
]


def classify_plate_color(img_bgr):
    """裁剪图 -> 颜色标签。返回 (标签, 描述)"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 只看饱和度高且不太暗的像素(避免白色字符干扰)
    mask = (s > 60) & (v > 80)
    if mask.sum() < 50:
        return ("unknown", "像素不足")
    mean_h = float(np.median(h[mask]))
    for name, lo, hi in COLOR_RANGES:
        if lo <= mean_h <= hi:
            desc = {"green_nev": "新能源(绿牌)", "blue_fuel": "传统燃油(蓝牌)",
                    "yellow": "黄牌"}[name]
            return (name, desc)
    # 低饱和度亮色 -> 白牌; 暗色 -> 黑牌
    if s[mask].mean() < 60 and v[mask].mean() > 150:
        return ("white", "白牌")
    if v[mask].mean() < 90:
        return ("black", "黑牌")
    return ("unknown", f"未识别 H={mean_h:.0f}")


def is_plate_candidate(text):
    t = re.sub(r"[\s\-·.]", "", text.upper())
    if len(t) < 6 or len(t) > 8:
        return False
    if PLATE_RE.match(t):
        return True
    # 宽松匹配: 含省份简称开头或末尾连续5+位字母数字
    if re.match(r"^[一-龥]", t) and re.search(r"[A-Z0-9]{5,6}$", t):
        return True
    return False


def expand_box(box, w, h, ratio=0.15):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * ratio))
    y1 = max(0, int(y1 - bh * ratio))
    x2 = min(w, int(x2 + bw * ratio))
    y2 = min(h, int(y2 + bh * ratio))
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--step", type=int, default=10, help="每隔多少帧分析一帧")
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--out", default="output")
    ap.add_argument("--limit", type=int, default=0, help="最多分析多少帧(0=不限)")
    ap.add_argument("--start", type=float, default=0.0, help="从第几秒开始分析(用于分段跑长视频)")
    ap.add_argument("--tag", default="", help="分段运行标签(输出文件加后缀)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    print(f"视频: {total} 帧, {fps:.1f} fps, 总时长 {total/fps:.1f}s, 起点 {args.start:.0f}s")

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, lang="ch")

    plates = {}  # 车牌文本 -> 记录
    frame_idx = 0
    analyzed = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.step != 0:
            continue
        analyzed += 1
        if args.limit and analyzed > args.limit:
            break

        # 缩放加速
        h, w = frame.shape[:2]
        if w > args.max_width:
            scale = args.max_width / w
            frame = cv2.resize(frame, (args.max_width, int(h * scale)))
        h, w = frame.shape[:2]

        ts = args.start + frame_idx / fps
        results = ocr.predict(frame)
        found = 0
        for res in results:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("rec_polys", [])
            for text, score, poly in zip(texts, scores, polys):
                if score < 0.5 or not is_plate_candidate(text):
                    continue
                found += 1
                x1, y1, x2, y2 = expand_box(np.array(poly).tolist(), w, h)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                color, color_desc = classify_plate_color(crop)
                clean = re.sub(r"[\s\-·.]", "", text.upper())
                rec = plates.get(clean)
                if rec is None or score > rec["best_score"]:
                    img_path = os.path.join(args.out, "samples",
                                            f"{clean or 'unknown'}_{int(ts*1000)}.jpg")
                    cv2.imwrite(img_path, crop)
                    plates[clean] = {
                        "plate": clean, "color": color, "color_desc": color_desc,
                        "best_score": round(float(score), 3),
                        "first_seen_s": round(ts, 1),
                        "sample_img": img_path,
                    }
                # 在原帧上画框(首张样例)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{clean} {color}", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if found and analyzed <= 5:
            cv2.imwrite(os.path.join(args.out, "samples", f"frame_{int(ts*10)}.jpg"), frame)
        if analyzed % 20 == 0:
            el = time.time() - t0
            print(f"已分析 {analyzed} 帧 ({frame_idx}/{total}), 发现车牌 {len(plates)} 张, 用时 {el:.0f}s",
                  flush=True)

    cap.release()

    # 输出
    rows = sorted(plates.values(), key=lambda r: r["first_seen_s"])
    suffix = f"_{args.tag}" if args.tag else ""
    with open(os.path.join(args.out, f"plates{suffix}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wtr = csv.DictWriter(f, fieldnames=["plate", "color", "color_desc",
                                            "best_score", "first_seen_s", "sample_img"])
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)

    stats = {
        "total_plates": len(rows),
        "nev_green": sum(1 for r in rows if r["color"] == "green_nev"),
        "fuel_blue": sum(1 for r in rows if r["color"] == "blue_fuel"),
        "yellow": sum(1 for r in rows if r["color"] == "yellow"),
        "white": sum(1 for r in rows if r["color"] == "white"),
        "black": sum(1 for r in rows if r["color"] == "black"),
        "unknown": sum(1 for r in rows if r["color"] == "unknown"),
        "elapsed_s": round(time.time() - t0, 1),
        "frames_analyzed": analyzed,
    }
    with open(os.path.join(args.out, f"stats{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "plates": rows}, f, ensure_ascii=False, indent=2)

    print("\n===== 统计结果 =====")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n明细: {os.path.join(args.out, 'plates.csv')}")


if __name__ == "__main__":
    main()
