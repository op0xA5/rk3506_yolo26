#!/usr/bin/env bash
# 交叉编译 onnxruntime (ARM32 armv7l + NEON) 静态库。
#
# 说明:
#   * SDK 自带的 cmake (3.28) 无法下载文件，故固定使用 /usr/bin/cmake。
#   * onnx 的 .proto 代码生成需要宿主机 protoc，版本必须与源码锁定的
#     protobuf 一致 (见 onnxruntime-src/cmake/deps.txt, 当前 v21.12)，
#     通过 -DONNX_CUSTOM_PROTOC_EXECUTABLE 指定 tools/protoc-21.12。
#   * 默认 MINIMAL_BUILD=ON: 仅支持 .ort 格式模型 (需先用
#     scripts/convert-model.sh 将 .onnx 转为 .ort, 图优化固定)。
#     如需直接加载 .onnx, 用 MINIMAL_BUILD=OFF (编译全部算子, 体积更大)。
#   * 设置 OPS_CONFIG (默认 models/required_operators.config, 由
#     scripts/convert-model.sh 生成) 后，只编译模型用到的算子。
#   * 若修改 ONNXRUNTIME_VERSION，请同步更新 deps.txt 中的 protobuf 版本
#     并重新下载对应版本的 protoc。
#
# 常用环境变量覆盖:
#   MINIMAL_BUILD=ON|OFF           默认 ON
#   OPS_CONFIG=xxx.config          算子裁剪配置 (传空字符串关闭裁剪)
#   BUILD_TYPE=Release|MinSizeRel  默认 Release
#   JOBS=N                         默认 nproc
#   SKIP_CONFIGURE=1               仅重新编译 (增量)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/onnxruntime-src"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/onnxruntime-armv7l}"
DIST_DIR="$ROOT_DIR/dist/armv7l"
CMAKE="${CMAKE:-/usr/bin/cmake}"
NINJA="$(command -v ninja || echo "$HOME/.local/bin/ninja")"
PROTOC="$ROOT_DIR/tools/protoc-21.12/bin/protoc"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
MINIMAL_BUILD="${MINIMAL_BUILD:-ON}"
OPS_CONFIG="${OPS_CONFIG-$ROOT_DIR/models/required_operators.config}"

[ -x "$PROTOC" ] || { echo "缺少宿主机 protoc: $PROTOC"; echo "请先运行: make protoc"; exit 1; }

REDUCED_OPS_CMAKE_ARGS=()
if [ -n "$OPS_CONFIG" ]; then
  [ -f "$OPS_CONFIG" ] || { echo "OPS_CONFIG 不存在: $OPS_CONFIG"; exit 1; }
  echo "==> 算子裁剪配置: $OPS_CONFIG"
else
  echo "==> 未提供算子裁剪配置，编译全部标准算子"
fi

if [ "${SKIP_CONFIGURE:-0}" != "1" ]; then
  rm -rf "$BUILD_DIR"

  # 算子裁剪: 在 cmake 配置前生成 op_reduction.generated (cmake 自动拾取)
  if [ -n "$OPS_CONFIG" ]; then
    python3 "$SRC_DIR/tools/ci_build/reduce_op_kernels.py" \
      "$OPS_CONFIG" \
      --cmake_build_dir "$BUILD_DIR"
    REDUCED_OPS_CMAKE_ARGS=(-Donnxruntime_REDUCED_OPS_BUILD=ON)
  fi

  "$CMAKE" -S "$SRC_DIR/cmake" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$NINJA" \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT_DIR/cmake/armv7l-buildroot.toolchain.cmake" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DONNX_CUSTOM_PROTOC_EXECUTABLE="$PROTOC" \
    -Donnxruntime_MINIMAL_BUILD="$MINIMAL_BUILD" \
    -Donnxruntime_DISABLE_EXCEPTIONS=OFF \
    -Donnxruntime_CROSS_COMPILING=ON \
    -Donnxruntime_BUILD_SHARED_LIB=OFF \
    -Donnxruntime_BUILD_UNIT_TESTS=OFF \
    -Donnxruntime_USE_FULL_PROTOBUF=OFF \
    -Donnxruntime_ENABLE_CPUINFO=ON \
    -Donnxruntime_ENABLE_PYTHON=OFF \
    -Donnxruntime_USE_DNNL=OFF \
    -Donnxruntime_USE_XNNPACK=OFF \
    "${REDUCED_OPS_CMAKE_ARGS[@]}"
fi

# 静态构建不产出合并的 libonnxruntime.a，而是组件库
# (onnxruntime_session/framework/graph/mlas/providers/... + onnx/protobuf/absl 等)，
# 全部构建后统一收集，由最终链接方以 --start-group 方式链接。
"$CMAKE" --build "$BUILD_DIR" --parallel "$JOBS"

# re2 为 EXCLUDE_FROM_ALL (静态库场景无最终链接步骤)，需显式构建
if "$CMAKE" --build "$BUILD_DIR" --target help 2>/dev/null | grep -q '^re2'; then
  "$CMAKE" --build "$BUILD_DIR" --parallel "$JOBS" --target re2
fi

# ---- 汇总产物到 dist/armv7l (头文件 + 全部静态库 + pkg-config 文件) ----
mkdir -p "$DIST_DIR/include" "$DIST_DIR/lib"

# onnxruntime C API 公开头文件
cp "$SRC_DIR"/include/onnxruntime/core/session/*.h "$DIST_DIR/include/"

# 收集构建产物中的全部静态库（除测试相关）
find "$BUILD_DIR" -name '*.a' ! -name '*test*' -exec cp {} "$DIST_DIR/lib/" \;

# 生成 pkg-config 文件，供手动编译参考（Go 侧由 Makefile 直接注入 CGO_LDFLAGS）
LIB_LIST=""
for a in "$DIST_DIR"/lib/*.a; do
  name="$(basename "$a")"            # libfoo.a -> foo
  LIB_LIST="$LIB_LIST -l${name#lib}"
done

cat > "$DIST_DIR/libonnxruntime-armv7l.pc" <<EOF
prefix=$DIST_DIR
includedir=\${prefix}/include
libdir=\${prefix}/lib

Name: onnxruntime
Description: onnxruntime static library for armv7l (NEON)
Version: 1.29.1
Cflags: -I\${includedir}
Libs: -Wl,--start-group $LIB_LIST -Wl,--end-group -ldl -lpthread -lm
EOF

echo "==== 构建完成 ===="
echo "静态库目录: $DIST_DIR/lib ($(ls "$DIST_DIR/lib" | wc -l) 个库文件)"
ls -lh "$DIST_DIR/lib" | head -20
file "$DIST_DIR/lib/libonnxruntime_session.a" | head -1
