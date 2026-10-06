#!/usr/bin/env bash
# 导出 yolo26n onnx 模型到 models/ 目录。
#
# 用法:
#   bash scripts/export-model.sh          # fp32 原始模型 -> yolo26n.onnx
#   bash scripts/export-model.sh e2e      # fp32 NMS-free 端到端 -> yolo26n_e2e_fp32.onnx
#   bash scripts/export-model.sh int8     # INT8(QOperator) + NMS-free 端到端 -> yolo26n_nms_int8.onnx
#
# int8 流程: 先用 e2e 导出 fp32 端到端模型, 再运行 scripts/quantize-int8.py
# 做 QOperator 静态量化 (激活张量全程 int8, 内存约为 QDQ 格式的 1/4,
# 适合 60MB 级内存芯片; ultralytics 自带的 quantize=int8 为 QDQ 格式,
# 反量化后激活仍为 fp32, 低内存设备上更耗内存, 故不使用)。
# 校准数据默认 ultralytics coco8; 建议用 --calib-dir 指定与部署场景一致的数据。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! python3 -c "import ultralytics" 2>/dev/null; then
  echo "未安装 ultralytics，尝试安装..."
  pip3 install --user --break-system-packages ultralytics onnx onnxruntime
fi

cd "$ROOT_DIR/models"
IMGSZ="${IMGSZ:-640}"
case "${1:-}" in
  e2e)
    python3 -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').export(format='onnx', imgsz=$IMGSZ, nms=False)
"
    cp -f yolo26n.onnx yolo26n_e2e_fp32.onnx
    echo "已导出: $ROOT_DIR/models/yolo26n_e2e_fp32.onnx"
    ;;
  int8)
    bash "$ROOT_DIR/scripts/export-model.sh" e2e
    python3 "$ROOT_DIR/scripts/quantize-int8.py" \
      "$ROOT_DIR/models/yolo26n_e2e_fp32.onnx" "$ROOT_DIR/models/yolo26n_nms_int8.onnx" \
      --size "$IMGSZ" ${CALIB_DIR:+--calib-dir "$CALIB_DIR"}
    ;;
  *)
    python3 -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').export(format='onnx', imgsz=640)
"
    echo "已导出: $ROOT_DIR/models/yolo26n.onnx"
    ;;
esac
