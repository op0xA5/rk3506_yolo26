#!/usr/bin/env bash
# yolo26n -> ncnn (.ncnn.param/.bin) fp32 + int8 量化 转换脚本。
#
# 用法:
#   bash scripts/convert-ncnn.sh [IMGSZ] [校准图片目录]
#   例如: bash scripts/convert-ncnn.sh 480
#   全 int8 (检测头也量化, 更快但精度略降): FULL_INT8=1 bash scripts/convert-ncnn.sh 480
#
# 输出 (models/ 目录, 以 480 为例):
#   yolo26n480.ncnn.param / .bin           fp32 模型 (pnnx 转换)
#   yolo26n480-int8.ncnn.param / .bin      int8 量化模型 (默认检测头 fp32 混合精度)
#   yolo26n480.table                       量化 scale 表 (校准产物)
#
# 流程:
#   1. ultralytics 导出 torchscript (原始检测头, 输出 [1,84,N])
#   2. pnnx 转换为 ncnn 格式 (自动烘焙 Shape/Gather/动态 Slice 等动态图结构,
#      onnx2ncnn 对这些支持不完整, 会得到 NaN 结果, 故使用官方维护的 pnnx)
#   3. ncnn2table 生成 per-channel 权重 scale (激活 scale 不用它的:
#      其直方图分位数裁剪对校准集之外输入大面积 clipping, 实测部分层输出
#      退化为全零; 激活 scale 改由 ncnncalib 以 absmax 统计)
#   4. ncnncalib 用与运行时完全一致的预处理 (letterbox+/255+NCHW) 统计激活
#      absmax, 生成 scale = 127/absmax (语义与 ncnn 表一致)
#   5. 默认过滤检测头 (cv2/cv3 towers) 保持 fp32: 检测头计算量占比 <5%,
#      但直接决定分数/坐标, int8 量化损失主要来源; FULL_INT8=1 可关闭
#   6. ncnn2int8 生成 int8 参数/权重
#
# 依赖 (一次性): bash scripts/build-ncnn-host-tools.sh; pip3 install pnnx ultralytics
# 替换自行训练的模型: 将 models/yolo26n.pt 换成自己的权重再跑本脚本。
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMGSZ="${1:-480}"
CALIB_DIR="${2:-}"
NAME="yolo26n${IMGSZ}"
T=build/ncnn-host
Q=$T/tools/quantize

for t in "$Q/ncnn2table" "$Q/ncnn2int8" "$T/ncnncalib"; do
  [ -x "$t" ] || { echo "缺少工具 $t"; echo "先运行: bash scripts/build-ncnn-host-tools.sh"; exit 1; }
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

# ---------- 2. pnnx 转换 (输出文件名与输入同名, 输出到输入所在目录) ----------
pnnx "$TS" "inputshape=[1,3,$IMGSZ,$IMGSZ]"
PNNX_BASE="$(basename "$TS" .torchscript)"
PNNX_DIR="$(dirname "$TS")"
mv -f "$PNNX_DIR/$PNNX_BASE.ncnn.param" "models/$NAME.ncnn.param"
mv -f "$PNNX_DIR/$PNNX_BASE.ncnn.bin" "models/$NAME.ncnn.bin"
rm -f "$PNNX_DIR/$PNNX_BASE.pnnx.bin" "$PNNX_DIR/$PNNX_BASE.pnnx.onnx" "$PNNX_DIR/$PNNX_BASE.pnnx.param" \
      "$PNNX_DIR/${PNNX_BASE}_pnnx.py" "$PNNX_DIR/${PNNX_BASE}_ncnn.py" "$TS"
echo "fp32 模型: models/$NAME.ncnn.param/.bin (输入 ${IMGSZ}x${IMGSZ})"

# ---------- 3. 校准集准备 (默认 coco128, letterbox + /255 + NCHW raw) ----------
CALIB_RAW="build/calib${IMGSZ}_raw"
if [ ! -d "$CALIB_RAW" ] || [ -n "$CALIB_DIR" ]; then
  if [ -z "$CALIB_DIR" ]; then
    COCO=build/coco/coco128/images/train2017
    if [ ! -d "$COCO" ]; then
      echo "下载 coco128 校准集..."
      mkdir -p build/coco && cd build/coco
      curl -sfL -o coco128.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
      unzip -q -o coco128.zip
      cd "$ROOT_DIR"
    fi
    CALIB_SRC="$COCO"
  else
    CALIB_SRC="$CALIB_DIR"
  fi
  echo "生成 letterbox 校准输入 ($IMGSZ)..."
  python3 - "$CALIB_SRC" "$CALIB_RAW" "$IMGSZ" <<'EOF'
import numpy as np, os, sys
from PIL import Image
src, dst, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
n = 0
for f in sorted(os.listdir(src)):
    if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img = Image.open(os.path.join(src, f)).convert('RGB')
    sw, sh = img.size
    scale = min(size / sw, size / sh)
    nw, nh = int(sw * scale + 0.5), int(sh * scale + 0.5)
    r = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    canvas.paste(r, ((size - nw) // 2, (size - nh) // 2))
    a = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    name = os.path.splitext(f)[0] + '.raw'
    open(os.path.join(dst, name), 'wb').write(a.tobytes())
    n += 1
print(f'{n} 张 -> {dst}')
EOF
fi
ls "$CALIB_RAW"/*.raw > /tmp/ncnn_calib_list.txt
echo "校准图片 $(wc -l < /tmp/ncnn_calib_list.txt) 张 (letterbox+$IMGSZ, RGB, /255, NCHW)"

# ---------- 4. 权重 scale (ncnn2table) + 激活 scale (ncnncalib absmax) ----------
"$Q/ncnn2table" \
  "models/$NAME.ncnn.param" "models/$NAME.ncnn.bin" \
  /tmp/ncnn_calib_list.txt "models/$NAME.table" \
  "mean=[0,0,0]" "norm=[0.003921569,0.003921569,0.003921569]" \
  "shape=[$IMGSZ,$IMGSZ,3]" "pixel=RGB" >/dev/null
"$T/ncnncalib" \
  "models/$NAME.ncnn.param" "models/$NAME.ncnn.bin" \
  /tmp/ncnn_calib_list.txt "models/$NAME.table" "models/$NAME.table.act" 2>/dev/null
mv -f "models/$NAME.table.act" "models/$NAME.table"

# ---------- 5. 检测头 fp32 混合精度 (默认) ----------
FINAL_TABLE="models/$NAME.table"
if [ "${FULL_INT8:-0}" != "1" ]; then
  python3 - "models/$NAME.table" <<'EOF'
import sys
# yolo26 检测头 tower 层名 (pnnx 命名, 320/480 一致): 删除其 scale 条目即保持 fp32
head = [f"conv_{i}" for i in range(79, 97)] + [f"convdw_{i}" for i in range(212, 218)]
src = sys.argv[1]
lines = open(src).readlines()
out = [l for l in lines if l.split()[0].replace("_param_0", "") not in head]
dropped = len(lines) - len(out)
assert dropped >= 40, f"检测头条目异常: 只删除 {dropped} 条"
open(src, 'w').writelines(out)
print(f"检测头保持 fp32 (删除 {dropped} 条 head scale)")
EOF
fi

# ---------- 6. ncnn2int8 ----------
"$Q/ncnn2int8" \
  "models/$NAME.ncnn.param" "models/$NAME.ncnn.bin" \
  "models/$NAME-int8.ncnn.param" "models/$NAME-int8.ncnn.bin" \
  "$FINAL_TABLE"

echo "int8 模型: models/$NAME-int8.ncnn.param/.bin"
echo "板端运行:  ./ncnn-demo -model models/$NAME-int8.ncnn.param -size $IMGSZ -threads 3"
