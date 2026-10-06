# onnxruntime (ARM32 NEON minimal build) + YOLO26n Go 演示项目

目标平台: RK3506 (ARM32 armv7l, Cortex-A7, NEON, hard-float)。

运行时为 **onnxruntime minimal build + 算子裁剪** 的静态库，
模型为 **ORT 格式**（图优化在转换期固定），仅包含模型实际用到的算子。

## 目录结构

```
onnxruntime-src/          onnxruntime v1.29.1 源码 (git clone --recursive, 不入库)
cmake/armv7l-buildroot.toolchain.cmake   交叉编译工具链文件
scripts/build-onnxruntime-arm.sh         onnxruntime 交叉编译脚本 (minimal+算子裁剪)
scripts/convert-model.sh   .onnx -> .ort 转换 + 算子配置生成 (替换模型用)
scripts/export-model.sh    ultralytics 导出 yolo26n.onnx
tools/protoc-21.12/        宿主机 protoc (与源码锁定的 protobuf v21.12 对应)
dist/armv7l/               交叉编译产物 (头文件 + 全部静态库 + .pc)
go-yolo26n/                Go cgo 演示项目 (检测 + 分阶段耗时统计)
models/                    yolo26n.onnx / yolo26n.ort / required_operators.config
build/                     cmake 构建目录
```

## 构建步骤 (首次)

```bash
# 1. 获取 onnxruntime 源码
git clone --depth 1 --branch v1.29.1 --recursive --shallow-submodules \
    https://github.com/microsoft/onnxruntime.git onnxruntime-src

# 2. 宿主机依赖: protoc (版本须匹配 cmake/deps.txt 中的 protobuf), ninja,
#    以及用于模型转换的 python 包
mkdir -p tools && curl -fSL -o tools/protoc.zip \
    https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-linux-x86_64.zip
unzip -q tools/protoc.zip -d tools/protoc-21.12
pip3 install --user onnxruntime==1.29.0 onnx   # 版本与 onnxruntime-src 一致
pip3 install --user ninja

# 3. 导出并转换模型 (生成 yolo26n.ort + required_operators.config)
bash scripts/export-model.sh      # 得到 models/yolo26n.onnx
bash scripts/convert-model.sh     # 得到 models/yolo26n.ort + 算子配置

# 3.1 (可选) INT8 量化 + NMS-free 端到端模型 (用于对比测试)
bash scripts/export-model.sh int8 # 得到 models/yolo26n_nms_int8.onnx
bash scripts/convert-model.sh models/yolo26n.onnx models/yolo26n_nms_int8.onnx
                                  # 合并生成两模型共用的算子配置 (含 QLinear* 量化算子)

# 4. 交叉编译 onnxruntime 静态库 (minimal + 只保留模型用到的算子)
make onnxruntime                  # 或: bash scripts/build-onnxruntime-arm.sh

# 5. 编译 Go 演示程序
make -C go-yolo26n

# 6. 拷贝到目标板运行
scp go-yolo26n/yolo26n-demo models/yolo26n.ort images/bus.jpg <板子>:~/
ssh <板子> ./yolo26n-demo -model yolo26n.ort -image bus.jpg -iters 20
```

## 替换为自己训练的模型

```bash
cp my_model.onnx models/          # 或由 ultralytics 导出
bash scripts/convert-model.sh models/my_model.onnx   # 重新生成 .ort + 算子配置
bash scripts/build-onnxruntime-arm.sh                # 按新算子清单重新裁剪编译
```

注意: 算子裁剪后运行**只支持配置中列出的算子**；
若需保留全部算子 (直接加载任意 .onnx)，用 `MINIMAL_BUILD=OFF OPS_CONFIG="" bash scripts/build-onnxruntime-arm.sh`。

## INT8 量化 NMS-free 模型 (对比测试)

`yolo26n_nms_int8.ort` 由官方 yolo26n.pt 经 INT8 静态量化 (QDQ/QLinear)
并烘焙 NMS-free 端到端解码 (TopK) 得到:

- 输出 `[1, 300, 6]` = (x1, y1, x2, y2, score, class)，无需 NMS
- 模型体积 9.6M -> 3.3M，输入仍为 float32，Go 程序无需修改
- 导出: `bash scripts/export-model.sh int8`
  (本质是 `yolo export format=onnx nms=False quantize=int8`，校准数据默认 coco8)

对比运行 (宿主机 qemu 模拟，绝对耗时不代表真实芯片):
`make -C go-yolo26n compare ARGS="-iters 10"`
目标板上分别用 `-model yolo26n.ort` 与 `-model yolo26n_nms_int8.ort` 运行即可对比。

## 程序说明

`yolo26n-demo` 通过 cgo 静态链接 onnxruntime，输出每个阶段耗时:

- 模型加载、图片解码、预处理 (letterbox 等比缩放/填充、RGB、/255 归一化、NCHW)
- 推理 (默认重复 10 次输出 avg/min/max，`-iters` 调整)
- 后处理 (自动适配 NMS-free 端到端 `[1,N,6]` / 原始 `[1,84,8400]` 输出)

```
./yolo26n-demo -model yolo26n.ort -image bus.jpg -conf 0.25 -threads 4 -iters 20 -out result.jpg
```

用 `-threads` 对比不同线程数下的 NEON 推理性能。
宿主机可用 `make -C go-yolo26n run ARGS="-model ../models/yolo26n.ort -image ../images/bus.jpg"` 通过 qemu-arm 试运行。

## 手工调整

- CPU 架构/NEON/浮点选项: `cmake/armv7l-buildroot.toolchain.cmake` 的 `ARCH_FLAGS`
  (默认 `-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard`)。
- onnxruntime 构建选项: `scripts/build-onnxruntime-arm.sh` 中 cmake 参数
  (`BUILD_TYPE=MinSizeRel` 可切换体积优先)。
- 链接库: `go-yolo26n/Makefile` 的 `CGO_LDFLAGS`。
