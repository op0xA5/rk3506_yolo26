#!/usr/bin/env bash
# yolo26n -> ncnn (.ncnn.param/.bin) fp32 + int8 (post-training 量化) 转换脚本。
#
# 用法:
#   bash scripts/convert-ncnn.sh [IMGSZ] [校准图片目录]
#   例如: bash scripts/convert-ncnn.sh 480
#
# 输出 (models/ 目录, 以 480 为例):
#   yolo26n480.ncnn.param / .bin           fp32 模型 (pnnx 转换)
#   yolo26n480-int8.ncnn.param / .bin      int8 量化模型 (激活/权重均 int8)
#   yolo26n480.table                       量化 scale 表 (校准产物)
#
# 流程:
#   1. ultralytics 导出 torchscript (原始检测头, 输出 [1,84,N])
#   2. pnnx 转换为 ncnn 格式 (自动烘焙 Shape/Gather/动态 Slice 等动态图结构,
#      onnx2ncnn 对这些支持不完整, 会得到 NaN 结果, 故使用官方维护的 pnnx)
#   3. ncnn2table 在宿主机跑 fp32 推理收集激活分布生成 scale 表,
#      归一化参数 (mean=0, norm=1/255, pixel=RGB) 必须与 Go 预处理一致
#   4. ncnn2int8 生成 int8 参数/权重
#
# 注意:
#   - 依赖: pip3 install --user pnnx; ultralytics; ncnn 宿主机工具
#     (ninja -C build/ncnn-host ncnn2table ncnn2int8)
#   - ncnn2table 校准用拉伸缩放 (非 letterbox), 与运行时预处理略有差异,
#     建议用更多与部署场景一致的图片做校准 (数量越多越好)。
#   - 替换自行训练的模型: 只需将 models/yolo26n.pt 换成自己的权重再跑本脚本。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMGSZ="${1:-480}"
CALIB_DIR="${2:-}"
NAME="yolo26n${IMGSZ}"
TOOLS_HOST=build/ncnn-host/tools/quantize
for t in "$TOOLS_HOST/ncnn2table" "$TOOLS_HOST/ncnn2int8"; do
  [ -x "$t" ] || { echo "缺少工具 $t"; echo "先运行: ninja -C build/ncnn-host ncnn2table ncnn2int8"; exit 1; }
done
command -v pnnx >/dev/null || { echo "缺少 pnnx: pip3 install --user --break-system-packages pnnx"; exit 1; }

# ---------- 1. ultralytics 导出 torchscript ----------
python3 - "$IMGSZ" <<'EOF'
import sys
from ultralytics import YOLO
sz = int(sys.argv[1])
YOLO('models/yolo26n.pt').export(format='torchscript', imgsz=sz)
EOF
TS="models/yolo26n.torchscript"
[ -f "$TS" ] || TS="yolo26n.torchscript"   # 兼容在 models/ 目录下运行

# ---------- 2. pnnx 转换 (输出文件名与输入同名) ----------
pnnx "$TS" "inputshape=[1,3,$IMGSZ,$IMGSZ]"
PNNX_BASE="$(basename "$TS" .torchscript)"
PNNX_DIR="$(dirname "$TS")"
mv -f "$PNNX_DIR/$PNNX_BASE.ncnn.param" "models/$NAME.ncnn.param"
mv -f "$PNNX_DIR/$PNNX_BASE.ncnn.bin" "models/$NAME.ncnn.bin"
rm -f "$PNNX_DIR/$PNNX_BASE.pnnx.bin" "$PNNX_DIR/$PNNX_BASE.pnnx.onnx" "$PNNX_DIR/$PNNX_BASE.pnnx.param" \
      "$PNNX_DIR/${PNNX_BASE}_pnnx.py" "$PNNX_DIR/${PNNX_BASE}_ncnn.py" "$TS"
echo "fp32 模型: models/$NAME.ncnn.param/.bin (输入 ${IMGSZ}x${IMGSZ})"

# ---------- 3. 准备校准图片列表 ----------
if [ -z "$CALIB_DIR" ]; then
  CAND="/home/zhujl/lednum/yolo26/datasets/coco8/images/train"
  [ -d "$CAND" ] && CALIB_DIR="$CAND" || CALIB_DIR="images"
fi
LIST=$(mktemp)
find "$CALIB_DIR" \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | sort > "$LIST"
N=$(wc -l < "$LIST")
echo "校准图片 $N 张, 来自 $CALIB_DIR"

# ---------- 4. ncnn2table: 收集激活分布, 生成 scale 表 ----------
"$TOOLS_HOST/ncnn2table" \
  "models/$NAME.ncnn.param" "models/$NAME.ncnn.bin" \
  "$LIST" "models/$NAME.table" \
  "mean=[0,0,0]" "norm=[0.003921569,0.003921569,0.003921569]" \
  "shape=[$IMGSZ,$IMGSZ,3]" "pixel=RGB"

# ---------- 5. ncnn2int8: 生成 int8 参数/权重 ----------
"$TOOLS_HOST/ncnn2int8" \
  "models/$NAME.ncnn.param" "models/$NAME.ncnn.bin" \
  "models/$NAME-int8.ncnn.param" "models/$NAME-int8.ncnn.bin" \
  "models/$NAME.table"

rm -f "$LIST"
echo "int8 模型: models/$NAME-int8.ncnn.param/.bin"
echo "板端运行:  ./ncnn-demo -model models/$NAME-int8.ncnn.param -size $IMGSZ -threads 3"
