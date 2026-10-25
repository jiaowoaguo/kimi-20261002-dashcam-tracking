# -*- coding: utf-8 -*-
"""车牌颜色分类 (共享模块)

判色依据 (解决运动模糊小图的误判):
- 黄牌: 黄底色相(12-40)占比高
- 新能源绿牌: 左侧有纯绿区(38-80); 即使模糊, 绿->青渐变会产生大量
  过渡色(80-95)像素, 这是绿牌独有的特征 (蓝牌色相集中在 95-125, 过渡带≈0)
- 蓝牌: 色相集中在 95-125 且无绿区/过渡带
"""
import cv2
import numpy as np

YELLOW = (12, 40)
GREEN = (38, 80)
TRANSITION = (80, 95)   # 绿牌渐变青色过渡带
BLUE = (95, 125)


def _share(hsv):
    h, s, v = cv2.split(hsv)
    mask = (s > 50) & (v > 70)
    n = int(mask.sum())
    if n < 30:
        return None
    hs = h[mask]
    return {
        "yellow": float(((hs >= YELLOW[0]) & (hs <= YELLOW[1])).sum()) / n,
        "green": float(((hs >= GREEN[0]) & (hs <= GREEN[1])).sum()) / n,
        "trans": float(((hs >= TRANSITION[0]) & (hs <= TRANSITION[1])).sum()) / n,
        "blue": float(((hs >= BLUE[0]) & (hs <= BLUE[1])).sum()) / n,
        "n": n,
    }


def classify_plate_color(img_bgr):
    """车牌裁剪图 -> (标签, 描述)"""
    sh = _share(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
    if sh is None:
        return ("unknown", "像素不足")
    # 牌面中心区域再判一次, 排除边缘车身颜色污染
    hgt, wid = img_bgr.shape[:2]
    inset = img_bgr[int(hgt * 0.18):int(hgt * 0.82),
                    int(wid * 0.06):int(wid * 0.94)]
    sh2 = _share(cv2.cvtColor(inset, cv2.COLOR_BGR2HSV)) if inset.size > 30 else None
    g = max(sh["green"], sh2["green"] if sh2 else 0)
    t = max(sh["trans"], sh2["trans"] if sh2 else 0)
    b = max(sh["blue"], sh2["blue"] if sh2 else 0)
    y = max(sh["yellow"], sh2["yellow"] if sh2 else 0)
    if y >= 0.25 and y > g and y > b:
        return ("yellow", "黄牌")
    if g >= 0.08:
        return ("green_nev", "新能源(绿牌)")
    if t >= 0.15 and b >= 0.30:
        return ("green_nev", "新能源(绿牌-模糊)")
    if b >= 0.50:
        return ("blue_fuel", "传统燃油(蓝牌)")
    return ("unknown", "未识别")
