# CMake toolchain file: Buildroot SDK -> ARM32 (armv7l) hard-float + NEON
# 用于交叉编译 onnxruntime 静态库，目标平台 RK3506 (Cortex-A7, armv7l)。
#
# 用法:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/armv7l-buildroot.toolchain.cmake ...
#
# 注意: Buildroot SDK 自带的 toolchainfile.cmake 将 CMAKE_SYSTEM_NAME 设为
# "Buildroot"，onnxruntime 的部分判定依赖 CMAKE_SYSTEM_PROCESSOR 匹配 "arm*"，
# 因此这里单独维护一份面向 onnxruntime 的工具链文件。

set(SDK_PATH "/home/zhujl/rk3506/arm-buildroot-linux-gnueabihf_sdk-buildroot"
    CACHE PATH "Buildroot SDK 安装目录")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv7)

set(CMAKE_SYSROOT "${SDK_PATH}/arm-buildroot-linux-gnueabihf/sysroot")

set(CMAKE_C_COMPILER   "${SDK_PATH}/bin/arm-buildroot-linux-gnueabihf-gcc")
set(CMAKE_CXX_COMPILER "${SDK_PATH}/bin/arm-buildroot-linux-gnueabihf-g++")
set(CMAKE_AR           "${SDK_PATH}/bin/arm-buildroot-linux-gnueabihf-gcc-ar")
set(CMAKE_RANLIB       "${SDK_PATH}/bin/arm-buildroot-linux-gnueabihf-gcc-ranlib")
set(CMAKE_STRIP        "${SDK_PATH}/bin/arm-buildroot-linux-gnueabihf-strip")
set(CMAKE_OBJCOPY      "${SDK_PATH}/bin/arm-buildroot-linux-gnueabihf-objcopy")
set(CMAKE_LINKER       "${SDK_PATH}/bin/arm-buildroot-linux-gnueabihf-ld")
set(CMAKE_NM           "${SDK_PATH}/bin/arm-buildroot-linux-gnueabihf-gcc-nm")

# Cortex-A7: armv7-a + NEON/VFPv4 + hard-float。
# 可通过 -DARCH_FLAGS="..." 覆盖，例如针对 armv7ve:
#   -DARCH_FLAGS="-march=armv7ve -mfpu=neon-vfpv4 -mfloat-abi=hard"
set(ARCH_FLAGS "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard"
    CACHE STRING "目标 CPU 架构/浮点/NEON 编译选项")

set(CMAKE_C_FLAGS_INIT   "${ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${ARCH_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${ARCH_FLAGS}")

set(CMAKE_C_FLAGS_RELEASE_INIT   "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -DNDEBUG")

set(CMAKE_EXE_LINKER_FLAGS_INIT    "")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "")

# 只在 sysroot 中查找目标库/头文件，程序(如 protoc)在宿主机查找。
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
