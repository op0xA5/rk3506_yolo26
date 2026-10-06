# 顶层 Makefile - 一键构建: 模型转换 -> onnxruntime 静态库 -> Go 演示程序
.PHONY: all onnxruntime convert go clean distclean

all: onnxruntime go

# .onnx -> .ort + required_operators.config (需 models/*.onnx 与宿主机 onnxruntime python 包)
convert:
	bash scripts/convert-model.sh

onnxruntime:
	bash scripts/build-onnxruntime-arm.sh

go: onnxruntime
	$(MAKE) -C go-yolo26n

clean:
	rm -rf build dist
	$(MAKE) -C go-yolo26n clean

distclean: clean
	rm -rf onnxruntime-src
