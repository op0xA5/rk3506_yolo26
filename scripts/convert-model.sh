#!/usr/bin/env bash
# 将 YOLO ONNX 模型转换为 ORT 格式 (图固定) 并生成算子裁剪配置。
#
# 用于配合 onnxruntime minimal build + REDUCED_OPS_BUILD:
#   1. (可选) ultralytics 导出 onnx
#   2. convert_onnx_models_to_ort.py  --optimization_style Fixed
#      - 转换期完成全部图优化/常量折叠, 生成 .ort 模型
#      - 同时生成 required_operators.config (运行时实际用到的算子清单)
#   3. 之后用该 config 重新交叉编译 onnxruntime (见 scripts/build-onnxruntime-arm.sh)
#
# 替换为自己训练的模型: 把模型放到 models/ (或传路径参数), 重新运行本脚本,
# 再重新运行交叉编译脚本即可。
#
# 用法:
#   bash scripts/convert-model.sh [模型.onnx ...]   默认处理 models/*.onnx
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT_PY="$ROOT_DIR/onnxruntime-src/tools/python"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  shopt -s nullglob
  MODELS=("$ROOT_DIR"/models/*.onnx)
  [ ${#MODELS[@]} -gt 0 ] || { echo "models/ 下没有 .onnx 模型，请先导出或指定路径"; exit 1; }
fi

# 1) 转换为 .ort (Fixed: 转换期固定图, 运行时不依赖优化器; 同时输出算子配置)
for m in "${MODELS[@]}"; do
  echo "==> 转换 $(basename "$m") 为 .ort (Fixed 图优化)"
  (cd "$ORT_PY" && python3 convert_onnx_models_to_ort.py \
      --optimization_style Fixed \
      --output_dir "$(dirname "$m")" \
      "$m")
done

# 2) 汇总算子配置到 models/ 顶层，供交叉编译脚本使用
#    convert 脚本为每个模型生成 *.required_operators.config，
#    多个模型时合并去重。
OUT_CFG="$ROOT_DIR/models/required_operators.config"
: > "$OUT_CFG"
for c in "${MODELS[@]}"; do
  cfg="${c%.onnx}.required_operators.config"
  [ -f "$cfg" ] || { echo "缺少算子配置: $cfg"; exit 1; }
  grep -v '^#' "$cfg" >> "$OUT_CFG"
done
sort -u "$OUT_CFG" -o "$OUT_CFG"

echo ""
echo "==> 生成的文件:"
for m in "${MODELS[@]}"; do
  d="$(dirname "$m")"
  ls -lh "${m%.onnx}"*.ort "$OUT_CFG" 2>/dev/null | awk '{print "    ", $NF, "("$5")"}' | sort -u
done
echo ""
echo "算子清单 ($OUT_CFG):"
cat "$OUT_CFG"
echo ""
echo "下一步: 重新交叉编译 (裁剪算子):"
echo "  OPS_CONFIG=$OUT_CFG bash scripts/build-onnxruntime-arm.sh"
