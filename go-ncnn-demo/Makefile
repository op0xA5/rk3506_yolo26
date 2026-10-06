# ncnn-demo Makefile - 交叉编译 Go + cgo(ncnn 静态库) 为 ARM32 armv7l 二进制
#
# 前置条件:
#   1. 已完成 ncnn 交叉编译:  bash scripts/build-ncnn-arm.sh
#      产物位于 ../dist/ncnn-arm/{include,lib}
#   2. 宿主机安装 Go (>=1.21)
#
# 常用目标:
#   make            编译 ARM 二进制 ncnn-demo
#   make run        通过 qemu-arm 本地试运行 (需 qemu-user)
#   make compare    fp32 与 int8 模型 qemu 对比 (绝对耗时不代表真实芯片)
#   make clean

SDK       ?= /home/zhujl/rk3506/arm-buildroot-linux-gnueabihf_sdk-buildroot
DIST      ?= ../dist/ncnn-arm
TARGET     = ncnn-demo

CC_ARM     = $(SDK)/bin/arm-buildroot-linux-gnueabihf-gcc
CXX_ARM    = $(SDK)/bin/arm-buildroot-linux-gnueabihf-g++
NCNN_LIB  := $(DIST)/lib/libncnn.a

export GOOS        = linux
export GOARCH      = arm
export GOARM       = 7
export CGO_ENABLED = 1
export CC          = $(CC_ARM)
export CXX         = $(CXX_ARM)
export CGO_LDFLAGS = -Wl,--start-group $(abspath $(NCNN_LIB)) -Wl,--end-group -lstdc++ -lpthread -lm
export CGO_LDFLAGS_ALLOW = -Wl,.*

.PHONY: all run compare compare-ort clean check

all: $(TARGET)

check:
	@test -f $(NCNN_LIB) || { echo "未找到 libncnn.a，请先运行: bash scripts/build-ncnn-arm.sh"; exit 1; }

$(TARGET): *.go *.cpp *.h $(NCNN_LIB)
	go build -trimpath -ldflags "-s -w" -o $(TARGET) .

run: $(TARGET)
	qemu-arm -L $(SDK)/arm-buildroot-linux-gnueabihf/sysroot ./$(TARGET) $(ARGS)

# 对比测试: fp32 与 int8 (post-training 量化) 两个 ncnn 模型
# 注意: qemu 为指令模拟, 绝对耗时不代表真实芯片, 真实对比请在目标板上运行
compare: $(TARGET)
	@echo "================= FP32 ncnn: yolo26n480 ================="
	qemu-arm -L $(SDK)/arm-buildroot-linux-gnueabihf/sysroot ./$(TARGET) \
	  -model ../models/yolo26n480.ncnn.param -image ../images/bus.jpg $(ARGS)
	@echo ""
	@echo "================= INT8 ncnn: yolo26n480-int8 ==============="
	qemu-arm -L $(SDK)/arm-buildroot-linux-gnueabihf/sysroot ./$(TARGET) \
	  -model ../models/yolo26n480-int8.ncnn.param -image ../images/bus.jpg $(ARGS)

# 320 分辨率 int8 验证 (原始输出 + Go NMS; 与 onnxruntime 版 compare320 对比)
compare320: $(TARGET)
	@echo "================= INT8 ncnn 320: yolo26n320-int8 ================"
	qemu-arm -L $(SDK)/arm-buildroot-linux-gnueabihf/sysroot ./$(TARGET) \
	  -model ../models/yolo26n320-int8.ncnn.param -image ../images/bus.jpg -size 320 $(ARGS)

# 与 onnxruntime 版 (go-yolo26n) 对比需在板端分别运行两个二进制,
# onnxruntime int8 端到端基线 (480, threads=3):
#   图片解码 110ms + 预处理 344ms + 推理 1318ms + 后处理 0.08ms = 1772ms
compare-ort: $(TARGET)
	@echo "onnxruntime 版基线请运行: make -C ../go-yolo26n run ARGS=\"-model ../models/yolo26n480_nms_int8.ort\""
	@echo "本程序 (ncnn) 请运行:     make -C go-ncnn-demo run"

clean:
	rm -f $(TARGET)
