#!/usr/bin/env bash
# 导出 yolo26n.onnx (NMS-free 端到端模型) 到 models/ 目录。
# 需要: pip install ultralytics onnx
# 若本机无法安装 ultralytics，可在任意 x86 电脑上导出后拷贝 models/yolo26n.onnx 到本目录。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! python3 -c "import ultralytics" 2>/dev/null; then
  echo "未安装 ultralytics，尝试安装..."
  pip3 install --user --break-system-packages ultralytics onnx
fi

cd "$ROOT_DIR/models"
yolo export model=yolo26n.pt format=onnx imgsz=640
echo "已导出: $ROOT_DIR/models/yolo26n.onnx"
