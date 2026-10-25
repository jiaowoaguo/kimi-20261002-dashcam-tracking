# Kimi 任务备份：行车视频车辆检测与跟踪（2026-10-02）

## 来源

- 原始位置：`C:\Users\<你的用户名>\Documents\kimi\tasks\2026-10-02`
- 由 Kimi（月之暗面）生成的任务工作区，本仓库为归档备份。

## 任务内容

对行车记录仪视频 `dashcam.mp4` 做车辆检测、车牌识别、品牌识别与轨迹跟踪，输出带标注的视频与统计结果。

## 本仓库包含（脚本与最终结果）

- `analyze.py`、`track_annotate.py`、`plate_detector.py`、`classify_plate_color.py` — 主流程脚本
- `brand_model/` — 车辆品牌识别模型代码（`model.py`、`predict.py`、`export_onnx.py`）
- `out_final/stats.json` — 最终统计结果

## 未包含（约 7 GB，见 `EXCLUDED_FILES.csv`）

- `.venv/` 虚拟环境（约 4 GB，可由 pip 重建）
- 视频输出 `out_*/`、源视频 `dashcam.mp4`（约 3 GB）
- 模型权重 `.onnx` / `.onnx.data`、调试截图 `.jpg`、中间轨迹 `trajectories.json`

被排除的大文件清单（路径 + 大小）保存在根目录 `EXCLUDED_FILES.csv`，需要时可从原始路径取回。
