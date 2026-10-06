// ncnn C 包装层头文件: 以 extern "C" API 封装 ncnn C++ 接口,
// 供 cgo 调用 (cgo 无法直接使用 C++ 类与模板)。
//
// 内存策略 (目标芯片可用内存 ~60MB):
//   - 输入缓冲由 C 侧按 ncnn 对齐要求 (fastMalloc, 64 字节对齐) 分配并复用,
//     Go 侧通过返回的指针直接写入 NCHW float32 数据, 推理零拷贝。
//   - 输出 Mat 保存在 handle 内, Go 侧直接消费其数据指针, 下次 Run 前有效,
//     避免每轮推理的输出拷贝。
#ifndef NCNN_WRAPPER_H
#define NCNN_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ncnn_handle ncnn_handle_t;

ncnn_handle_t* ncnn_create(void);

// 加载 .param + .bin 模型文件, 成功返回 0。
int ncnn_load(ncnn_handle_t* h, const char* param_path, const char* bin_path);

// 获取/分配复用的对齐输入缓冲 (NCHW float32, 3*c*h*w 字节)。
// 首次调用按尺寸分配, 后续相同尺寸直接复用; 返回 NULL 表示分配失败。
float* ncnn_input_buf(ncnn_handle_t* h, int c, int h_, int w);

// 执行推理 (输入已写入 ncnn_input_buf 返回的缓冲)。
// threads 为推理线程数 (每次调用可不同, 便于对比测试)。成功返回 0。
int ncnn_run(ncnn_handle_t* h, int threads);

// 获取上次推理输出: 返回数据指针 (handle 持有, 下次 Run 前有效),
// ndims/shape 描述输出维度 (至多 4 维)。
const float* ncnn_output(ncnn_handle_t* h, int* ndims, int shape[4]);

void ncnn_destroy(ncnn_handle_t* h);

// 加载前设置默认线程数 (Gemm 层在 load 时冻结该值)。
void ncnn_set_default_threads(ncnn_handle_t* h, int threads);

#ifdef __cplusplus
}
#endif

#endif // NCNN_WRAPPER_H
