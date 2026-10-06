#!/usr/bin/env bash
# 导出 yolo26n onnx 模型到 models/ 目录。
#
# 用法:
#   bash scripts/export-model.sh          # fp32 原始模型 -> yolo26n.onnx
#   bash scripts/export-model.sh int8     # INT8 量化 + NMS-free 端到端 -> yolo26n_nms_int8.onnx
#
# int8 导出 (本质是): yolo export model=yolo26n.pt format=onnx nms=False quantize=int8
#   - nms=False 触发 end2end 分支 (TopK 烘焙进图, 输出 [1,300,6], 无需运行期 NMS)
#   - quantize=int8 使用 onnxruntime 静态量化 (校准数据默认 coco8.yaml, 自动下载)
#
# 若本机无法安装 ultralytics，可在任意 x86 电脑上导出后拷贝 models/*.onnx 到本目录。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! python3 -c "import ultralytics" 2>/dev/null; then
  echo "未安装 ultralytics，尝试安装..."
  pip3 install --user --break-system-packages ultralytics onnx onnxruntime
fi

cd "$ROOT_DIR/models"
if [ "${1:-}" = "int8" ]; then
  python3 -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').export(format='onnx', imgsz=640, nms=False, quantize='int8', data='coco8.yaml')
"
  echo "已导出: $ROOT_DIR/models/yolo26n_nms_int8.onnx"
else
  python3 -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').export(format='onnx', imgsz=640)
"
  echo "已导出: $ROOT_DIR/models/yolo26n.onnx"
fi
