#!/usr/bin/env bash
# 交叉编译 NCNN 为 ARM32 armv7l 静态库。
#
# 产物: dist/ncnn-arm/{include,lib}/libncnn.a
#
# 说明:
#   - 复用 onnxruntime 的 buildroot 工具链文件, 但 ARCH_FLAGS 覆盖为 -mfpu=neon:
#     ncnn master 在 neon-vfpv4 下编译 gemm_wq_int8.h 有 fp16 intrinsics 守卫漏洞
#     (__ARM_FP&2 为真但 gcc 不暴露 vcvt_f16_f32), 而 -mfpu=neon + NCNN_RUNTIME_CPU
#     是 ncnn CI 的标准 armv7 配置, VFPv4 内核运行期自动探测启用。
#   - ncnn 源码自动克隆 (浅克隆)。首次运行会注入自定义 Shape 层
#     (scripts/ncnn-shape-layer/, 转换工具 ncnn2table/ncnn2int8 加载模型需要;
#     pnnx 转换的主路径已把 Shape 折叠, 该层仅作保险)。
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d ncnn-src/.git ]; then
  git clone --depth 1 https://github.com/Tencent/ncnn.git ncnn-src
fi

# 注入自定义 Shape 层 (幂等)
if [ ! -f ncnn-src/src/layer/shape.cpp ]; then
  cp scripts/ncnn-shape-layer/shape.h scripts/ncnn-shape-layer/shape.cpp ncnn-src/src/layer/
  sed -i 's/^ncnn_add_layer(ArgMax OFF)$/ncnn_add_layer(ArgMax OFF)\nncnn_add_layer(Shape)/' ncnn-src/src/CMakeLists.txt
  echo "已注入自定义 Shape 层"
fi

SDK=/home/zhujl/rk3506/arm-buildroot-linux-gnueabihf_sdk-buildroot
source $SDK/environment-setup

# ARCH_FLAGS 仅首次配置生效 (CMAKE_*_FLAGS_INIT), 变更时需删除 build/ncnn-arm
STAMP=build/ncnn-arm/.arch_flags
FLAGS="-march=armv7-a -mfpu=neon -mfloat-abi=hard"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$FLAGS" ]; then
  echo "ARCH_FLAGS 变更, 删除旧构建目录"
  rm -rf build/ncnn-arm
fi
mkdir -p build/ncnn-arm && echo -n "$FLAGS" > "$STAMP"

/usr/bin/cmake -S ncnn-src -B build/ncnn-arm -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$PWD/cmake/armv7l-buildroot.toolchain.cmake \
  -DARCH_FLAGS="$FLAGS" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=ON \
  -DNCNN_INT8=ON -DNCNN_BF16=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/dist/ncnn-arm
~/.local/bin/ninja -C build/ncnn-arm install
echo "完成: dist/ncnn-arm/lib/libncnn.a"
