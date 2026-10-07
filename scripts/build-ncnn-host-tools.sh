#!/usr/bin/env bash
# 构建宿主机 ncnn 工具链 (模型转换/量化用):
#   - protobuf 静态库 (宿主机)      -> build/protobuf-host-install
#   - ncnn 宿主机工具 (x86)        -> build/ncnn-host/tools/quantize/{ncnn2table,ncnn2int8}
#   - 自定义校准器 ncnncalib       -> build/ncnn-host/ncnncalib
#
# 说明:
#   - ncnn2table 在本项目中仅用于生成 per-channel 权重 scale (激活 scale 由
#     ncnncalib 生成, 见 scripts/convert-ncnn.sh; ncnn2table 的直方图裁剪
#     对校准集外输入会造成大面积 clipping, 实测输出退化为全零)。
#   - protobuf 源码优先复用 onnxruntime 构建下载的 _deps, 否则从 github 下载。
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------- 1. 宿主机 protobuf ----------
if [ ! -f build/protobuf-host-install/lib/libprotobuf.a ]; then
  SRC=build/onnxruntime-armv7l/_deps/protobuf-src
  if [ ! -d "$SRC" ]; then
    mkdir -p build/protobuf-src && cd build/protobuf-src
    curl -sfL -o protobuf.zip https://github.com/protocolbuffers/protobuf/archive/refs/tags/v21.12.zip
    unzip -q -o protobuf.zip && cd ../..
    SRC=build/protobuf-src/protobuf-21.12
  fi
  /usr/bin/cmake -S "$SRC" -B build/protobuf-host \
    -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_BUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=$PWD/build/protobuf-host-install
  make -C build/protobuf-host -j"$(nproc)"
  make -C build/protobuf-host install
fi

# ---------- 2. ncnn 宿主机工具 ----------
/usr/bin/cmake -S ncnn-src -B build/ncnn-host -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DProtobuf_INCLUDE_DIR=$PWD/build/protobuf-host-install/include \
  -DProtobuf_LIBRARY=$PWD/build/protobuf-host-install/lib/libprotobuf.a \
  -DProtobuf_PROTOC_EXECUTABLE=$PWD/tools/protoc-21.12/bin/protoc \
  -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_OPENMP=OFF -DNCNN_SIMPLEOMP=ON -DNCNN_INT8=ON
~/.local/bin/ninja -C build/ncnn-host ncnn2table ncnn2int8

# ---------- 3. 自定义校准器 ----------
g++ -O2 -I ncnn-src/src -I build/ncnn-host/src tools/ncnncalib.cpp \
  build/ncnn-host/src/libncnn.a -fopenmp -lpthread -lm \
  -o build/ncnn-host/ncnncalib

echo "完成: build/ncnn-host/{ncnn2table 所在目录, ncnncalib}"
