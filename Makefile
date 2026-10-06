# 顶层 Makefile - 一键构建: 模型转换 -> 推理引擎静态库 -> Go 演示程序
#
# onnxruntime 路径: make            (go-yolo26n)
# ncnn 路径:        make ncnn       (go-ncnn-demo, 交叉编译 + Go 二进制)
# ncnn 模型转换:    make ncnn-convert IMGSZ=480
.PHONY: all onnxruntime convert go ncnn ncnn-convert clean distclean

all: onnxruntime go

# .onnx -> .ort + required_operators.config (需 models/*.onnx 与宿主机 onnxruntime python 包)
convert:
	bash scripts/convert-model.sh

onnxruntime:
	bash scripts/build-onnxruntime-arm.sh

go: onnxruntime
	$(MAKE) -C go-yolo26n

# ---- ncnn 路径 ----
ncnn:
	bash scripts/build-ncnn-arm.sh
	$(MAKE) -C go-ncnn-demo

ncnn-convert:
	bash scripts/convert-ncnn.sh $(IMGSZ)

clean:
	rm -rf build dist
	$(MAKE) -C go-yolo26n clean
	$(MAKE) -C go-ncnn-demo clean

distclean: clean
	rm -rf onnxruntime-src ncnn-src
