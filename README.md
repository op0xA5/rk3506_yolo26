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

## INT8 量化 NMS-free 模型 (对比测试 / 低内存推荐)

`yolo26n_nms_int8.ort` 由官方 yolo26n.pt 烘焙 NMS-free 端到端解码 (TopK) 后，
经 **QOperator 格式** INT8 静态量化得到:

- 输出 `[1, 300, 6]` = (x1, y1, x2, y2, score, class)，无需 NMS
- 模型体积 9.6M -> ~3M，输入仍为 float32，Go 程序无需修改
- **激活张量全程 int8 传递** (QLinearConv/QLinearMul/QLinearAdd...)，
  峰值内存约为 fp32 的 1/4，适合 60MB 级内存芯片
- 导出: `bash scripts/export-model.sh int8`
  (e2e 导出 -> `scripts/quantize-int8.py` 量化; 校准数据默认 ultralytics coco8,
  建议设 `CALIB_DIR=自己的图片目录` 或用 `python3 scripts/quantize-int8.py --calib-dir DIR`)

对比运行 (宿主机 qemu 模拟，绝对耗时不代表真实芯片):
`make -C go-yolo26n compare ARGS="-iters 10"` (fp32-640 vs int8-640)
`make -C go-yolo26n compare480 ARGS="-iters 10"` (int8-480 端到端 vs int8-480 原始输出+Go NMS)
目标板上分别用 `-model yolo26n.ort` 与 `-model yolo26n_nms_int8.ort` 运行即可对比。

### 端到端 vs Go NMS 后处理对比 (int8-480, qemu)

| 模型 | 输出 | 推理 | 后处理 |
|---|---|---|---|
| yolo26n480_nms_int8.ort | [1,300,6] TopK 已烘焙 | ~2500ms | ~0.3ms |
| yolo26n480_int8.ort | [1,84,4725] 原始输出 | ~2530ms | ~8.4ms (解码+NMS) |

端到端模型把 TopK/NMS 烘焙进图 (CPU 上几乎零开销)，Go 侧只做阈值过滤;
原始输出模型推理相同, 但 Go 需解码 4725 个候选框并做 NMS, 后处理差 ~28x。
生成原始输出 480 int8: `IMGSZ=480 bash scripts/export-model.sh` 后
`python3 scripts/quantize-int8.py models/yolo26n480_raw_fp32.onnx models/yolo26n480_int8.onnx --size 480`
再 `bash scripts/convert-model.sh models/*.onnx` (注意: 转换时需带上所有在用的模型, 配置为合并生成)。

## 内存说明 (目标芯片可用内存 ~60MB)

1. **32 位 onnxruntime 无内存池**: `allocator_utils.cc` 中 `sizeof(void*)==4`
   时整体禁用 CPU arena，张量分配全部走 malloc/free，ORT 侧无可调内存池参数。
2. Go 程序已做零拷贝/复用优化: 输入 OrtValue 只建一次、输出零拷贝视图、
   预处理缓冲复用、Go 堆软上限 `-memlimit` (默认 16MB)。
   优化前 (每轮推理拷贝输出+无 GC 限制) 在板端 threads=2 实测约 160MB 峰值。
3. 实测峰值 (程序内 VmRSS, qemu guest 视角, threads=2):

   | 模型 | 加载后 | 推理峰值 | 推理耗时(qemu) |
   |---|---|---|---|
   | yolo26n.ort (fp32 640) | ~40MB | ~58MB | ~8100ms |
   | yolo26n_nms_int8.ort (int8 640) | ~35MB | ~59MB | ~4200ms |
   | yolo26n480_nms_int8.ort (int8 480) | ~35MB | ~54MB | ~2500ms |

   注: 宿主机 qemu 进程的 RSS 会比上表高 20~40MB (TCG 指令翻译缓存,
   可用 `qemu-arm -tb-size 8` 缩小), 不代表真实芯片。
4. 60MB 芯片建议: **int8 模型** (速度快 2 倍) + `-threads 1~2` +
   内存紧张时用 480 输入 (`export-model.sh e2e` 前设 `IMGSZ=480` 重新导出量化,
   或直接用 models/yolo26n480_nms_int8.ort) + 不使用 `-out` 保存标注图。

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
