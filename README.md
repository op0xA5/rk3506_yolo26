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

### 端到端 vs Go NMS 后处理对比 (int8, qemu, threads=2)

| 分辨率 | 模型 | 输出 | 推理 | 后处理 |
|---|---|---|---|---|
| 640 | yolo26n_nms_int8.ort (e2e) | [1,300,6] | ~4200ms | ~0.3ms |
| 480 | yolo26n480_nms_int8.ort (e2e) | [1,300,6] | ~2460ms | ~0.3ms |
| 480 | yolo26n480_int8.ort (raw) | [1,84,4725] | ~2430ms | ~8.4ms |
| 320 | yolo26n320_nms_int8.ort (e2e) | [1,300,6] | ~1170ms | ~3.0ms |
| 320 | yolo26n320_int8.ort (raw) | [1,84,2100] | ~1170ms | ~4.4ms |

推理耗时与分辨率近似平方关系 (640->320 快 ~3.6x); 原始输出模型的
Go NMS 后处理随候选框数线性增长 (2100->4725: 4.4ms->8.4ms)。
`make -C go-yolo26n compare480 / compare320` 可分别复现。
内存峰值 (推理后 VmRSS): 320 ~53MB, 480 ~55MB, 640 ~59MB。

生成方法 (以 320 为例, 480/640 同理改 IMGSZ):
```bash
IMGSZ=320 bash scripts/export-model.sh          # 原始输出 fp32 -> yolo26n320_raw_fp32.onnx (需改名保存)
IMGSZ=320 bash scripts/export-model.sh e2e      # 端到端 fp32 -> yolo26n320_e2e_fp32.onnx
python3 scripts/quantize-int8.py models/yolo26n320_e2e_fp32.onnx models/yolo26n320_nms_int8.onnx --size 320
python3 scripts/quantize-int8.py models/yolo26n320_raw_fp32.onnx models/yolo26n320_int8.onnx --size 320
bash scripts/convert-model.sh models/*.onnx     # 注意: 传入所有在用模型, 配置为合并生成
```

## NCNN 替换 onnxruntime (性能对比)

`go-ncnn-demo/` 是功能对齐的 ncnn 版演示程序 (cgo 静态链接 libncnn.a),
预处理/后处理代码与 go-yolo26n 完全一致，仅推理引擎不同，可直接对比。

构建与转换:
```bash
pip3 install --user --break-system-packages pnnx   # 模型转换 (官方维护路径)
make ncnn                 # 交叉编译 libncnn.a + go-ncnn-demo (首次自动克隆 ncnn)
make ncnn-convert IMGSZ=480   # yolo26n.pt -> yolo26n480.ncnn.param/.bin + int8 量化
```
转换链: ultralytics 导出 torchscript -> **pnnx** 转 ncnn -> ncnn2table 校准
-> ncnn2int8。不用 onnx2ncnn: 它对 YOLO26 检测头的 Shape/Gather/动态 Slice
烘焙子图支持不完整 (fp16 守卫编译失败 + 转换后输出 NaN)，pnnx 能正确折叠。

板端运行:
```
./ncnn-demo -model yolo26n480-int8.ncnn.param -size 480 -threads 3 -image bus.jpg
```

### qemu 对比 (480, int8, threads=2; 绝对值不代表真实芯片)

| 引擎 | 模型 | 推理 | 后处理 | 加载后 VmRSS |
|---|---|---|---|---|
| onnxruntime | yolo26n480_nms_int8.ort (e2e) | ~2458ms | 0.3ms (解码) | ~40MB |
| ncnn | yolo26n480-int8 (raw+Go NMS) | ~2538ms | 7.3ms (解码+NMS) | ~24MB |
| ncnn | yolo26n480 (fp32) | ~7790ms | 7.6ms | ~39MB |

- ncnn int8 较其 fp32 提速 ~3.1x; 检测精度损失可忽略 (int8 漏检 1 个
  conf=0.51 的弱目标, 其余框/分数与 fp32 基本一致)。
- ncnn 运行时明显更轻 (模型加载后 24MB vs 40MB), 对 60MB 内存芯片更友好;
  端到端总耗时请在目标板上用同一图片实测对比 (`-threads` 建议 1~3)。

### ncnn 构建要点 (scripts/build-ncnn-arm.sh)

- `-mfpu=neon` (非 neon-vfpv4) + `NCNN_RUNTIME_CPU`: ncnn CI 标准 armv7
  配置; master 在 neon-vfpv4 下编译 gemm int8 内核有 fp16 intrinsics
  守卫漏洞。VFPv4 (Cortex-A7 支持) 由运行期探测自动启用。
- `NCNN_OPENMP=OFF NCNN_SIMPLEOMP=ON`: 单文件 OpenMP 实现, 静态链接无
  libgomp 依赖; 线程数经 `net.opt.num_threads` 设置, **须在加载模型前**
  (Gemm 层在 load 时冻结该值)。
- 自定义 Shape 层随脚本注入 (scripts/ncnn-shape-layer/), 供
  ncnn2table/ncnn2int8 等宿主机工具加载模型使用。
- 量化校准归一化参数 (mean=0, norm=1/255, pixel=RGB) 必须与 Go 预处理
  一致; 校准图片用 `bash scripts/convert-ncnn.sh 480 <图片目录>` 指定。



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
- ncnn 构建选项: `scripts/build-ncnn-arm.sh` 中 cmake 参数
  (`NCNN_INT8`、线程实现、`ARCH_FLAGS` 等)。
- ncnn 模型转换/量化: `scripts/convert-ncnn.sh` (分辨率/校准目录参数)。
- 链接库: `go-yolo26n/Makefile` 与 `go-ncnn-demo/Makefile` 的 `CGO_LDFLAGS`。
