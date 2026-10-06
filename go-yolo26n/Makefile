# yolo26n-demo Makefile - 交叉编译 Go + cgo(onnxruntime 静态库) 为 ARM32 armv7l 二进制
#
# 前置条件:
#   1. 已完成 onnxruntime 交叉编译:  make -C .. onnxruntime
#      产物位于 ../dist/armv7l/{include,lib}
#   2. 宿主机安装 Go (>=1.21)
#
# 常用目标:
#   make            编译 ARM 二进制 yolo26n-demo
#   make run        通过 qemu-arm 本地试运行 (需 qemu-user)
#   make clean

SDK       ?= /home/zhujl/rk3506/arm-buildroot-linux-gnueabihf_sdk-buildroot
DIST      ?= ../dist/armv7l
TARGET     = yolo26n-demo

CC_ARM     = $(SDK)/bin/arm-buildroot-linux-gnueabihf-gcc
ORT_LIBS  := $(wildcard $(DIST)/lib/*.a)

export GOOS        = linux
export GOARCH      = arm
export GOARM       = 7
export CGO_ENABLED = 1
export CC          = $(CC_ARM)
export CGO_CFLAGS  = -I$(abspath $(DIST)/include)
# --start-group 解决静态库循环依赖; onnxruntime 需要 dl/pthread/m。
# CGO_LDFLAGS_ALLOW 放行 -Wl,--start-group 等非白名单标志。
export CGO_LDFLAGS = -Wl,--start-group $(abspath $(ORT_LIBS)) -Wl,--end-group -lstdc++ -ldl -lpthread -lm
export CGO_LDFLAGS_ALLOW = -Wl,.*

.PHONY: all run clean check

all: $(TARGET)

$(TARGET): *.go $(ORT_LIBS)
	go build -trimpath -ldflags "-s -w" -o $(TARGET) .

check:
	@test -n "$(ORT_LIBS)" || { echo "未找到 onnxruntime 静态库，请先运行: make -C .. onnxruntime"; exit 1; }

run: $(TARGET)
	qemu-arm -L $(SDK)/arm-buildroot-linux-gnueabihf/sysroot ./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET)
