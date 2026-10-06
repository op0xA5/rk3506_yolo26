#!/usr/bin/env python3
"""INT8 量化脚本: fp32 onnx -> QOperator 格式 int8 onnx。

与 ultralytics 默认的 QDQ 格式不同, QOperator 格式将 Conv/MatMul/Add 等替换为
QLinearConv/QLinearMatMul/QLinearAdd, 激活张量全程以 int8 传递 (QDQ 格式中
每个权重/激活都会被反量化为 fp32, 内存与 fp32 相当, 且多出大量 Q/DQ 中间缓冲,
在低内存 ARM32 设备上反而更耗内存)。

用法:
  python3 scripts/quantize-int8.py [fp32.onnx] [int8.onnx] [--calib-dir DIR]

默认: 输入 models/yolo26n_e2e_fp32.onnx, 输出 models/yolo26n_nms_int8.onnx
校准数据默认取 ultralytics coco8 训练集 (如不存在则退化为 images/*.jpg,
并提示精度可能下降)。
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image


def letterbox(img: Image.Image, size: int) -> np.ndarray:
    """等比缩放+居中填充到 size*size, RGB, /255, CHW (与推理预处理一致)。"""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    x = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC RGB
    return x.transpose(2, 0, 1)[None]  # NCHW


class ImageCalibDataReader:
    def __init__(self, img_glob: str, size: int = 640):
        self.files = sorted(glob.glob(img_glob))
        if not self.files:
            raise SystemExit(f"校准图片不存在: {img_glob}")
        self.size = size
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        img = Image.open(self.files[self.idx]).convert("RGB")
        self.idx += 1
        return {"images": letterbox(img, self.size)}

    def rewind(self):
        self.idx = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="models/yolo26n_e2e_fp32.onnx")
    ap.add_argument("output", nargs="?", default="models/yolo26n_nms_int8.onnx")
    ap.add_argument("--calib-dir", default=None, help="校准图片目录 (jpg)")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    calib_dir = args.calib_dir
    if calib_dir is None:
        try:
            from ultralytics.cfg import SETTINGS

            cand = os.path.join(SETTINGS["datasets_dir"], "coco8", "images", "train")
            if os.path.isdir(cand):
                calib_dir = cand
        except Exception:
            pass
    if calib_dir is None:
        calib_dir = "images"
        print("警告: 使用 images/ 作为校准数据, 量化精度可能下降; "
              "建议用 --calib-dir 指定与训练/部署场景一致的图片目录", file=sys.stderr)

    dr = ImageCalibDataReader(os.path.join(calib_dir, "*.jpg"), args.size)
    print(f"校准图片 {len(dr.files)} 张, 来自 {calib_dir}")

    quantize_static(
        model_input=args.input,
        model_output=args.output,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QOperator,      # 激活 int8 全程传递 (低内存)
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul", "Add", "Mul"],
    )
    print(f"已生成: {args.output}")


if __name__ == "__main__":
    main()
