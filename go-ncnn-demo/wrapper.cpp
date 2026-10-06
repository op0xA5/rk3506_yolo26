// ncnn C 包装层实现。
#include "wrapper.h"
#include "net.h"
#include "mat.h"
#include <vector>

struct ncnn_handle
{
    ncnn::Net net;
    ncnn::Mat in;  // 复用的对齐输入 Mat (外部数据指针由 Go 直接写入)
    ncnn::Mat out; // 上次推理输出 (handle 持有, 避免拷贝)
};

ncnn_handle_t* ncnn_create(void)
{
    return new ncnn_handle_t;
}

// 设置默认线程数。注意: ncnn Gemm 层在 load 时冻结 num_threads,
// 因此须在加载模型前设置, 运行期修改对 Gemm 不生效。
void ncnn_set_default_threads(ncnn_handle_t* h, int threads)
{
    h->net.opt.num_threads = threads;
}

int ncnn_load(ncnn_handle_t* h, const char* param_path, const char* bin_path)
{
    if (h->net.load_param(param_path) != 0)
        return -1;
    if (h->net.load_model(bin_path) != 0)
        return -2;
    return 0;
}

float* ncnn_input_buf(ncnn_handle_t* h, int c, int h_, int w)
{
    if (h->in.empty() || h->in.dims != 3 || h->in.c != c || h->in.h != h_ || h->in.w != w)
    {
        h->in.create(w, h_, c);
        if (h->in.empty())
            return NULL;
    }
    return (float*)h->in.data;
}

// 输入/输出 blob 名候选: onnx2ncnn 用 images/output0, pnnx 用 in0/out0。
static const char* INPUT_CANDIDATES[] = {"images", "in0", "input", "x"};
static const char* OUTPUT_CANDIDATES[] = {"output0", "out0", "output", "y"};

int ncnn_run(ncnn_handle_t* h, int threads)
{
    h->net.opt.num_threads = threads;
    ncnn::Extractor ex = h->net.create_extractor();
    // 用 net.input_names()/output_names() 取真实 blob 名 (onnx2ncnn 为
    // images/output0, pnnx 为 in0/out0), 避免按候选名探测产生的失败日志。
    const std::vector<const char*>& in_names = h->net.input_names();
    const std::vector<const char*>& out_names = h->net.output_names();
    if (in_names.empty() || out_names.empty())
        return -2;
    if (ex.input(in_names[0], h->in) != 0)
        return -2;
    if (ex.extract(out_names[0], h->out) != 0)
        return -3;
    return 0;
}

const float* ncnn_output(ncnn_handle_t* h, int* ndims, int shape[4])
{
    const ncnn::Mat& m = h->out;
    *ndims = m.dims;
    shape[0] = 1;
    shape[1] = 1;
    shape[2] = 1;
    shape[3] = 1;
    if (m.dims == 1)      { shape[0] = m.w; }
    else if (m.dims == 2) { shape[0] = m.h; shape[1] = m.w; }
    else if (m.dims == 3) { shape[0] = m.c; shape[1] = m.h; shape[2] = m.w; }
    else if (m.dims == 4) { shape[0] = m.c; shape[1] = m.d; shape[2] = m.h; shape[3] = m.w; }
    return (const float*)m.data;
}

void ncnn_destroy(ncnn_handle_t* h)
{
    delete h;
}
