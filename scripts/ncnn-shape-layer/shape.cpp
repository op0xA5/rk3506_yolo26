// Copyright 2026 zhujl
#include "shape.h"

namespace ncnn {

Shape::Shape()
{
    one_blob_only = true;
    support_inplace = false;
}

// ONNX Shape 语义 (无 batch 维, 对应原 ONNX 张量 [1, ...]):
//   1 维 [w]             -> [1, 1, w]
//   2 维 [h, w]          -> [1, h, w]
//   3 维 [c, h, w]       -> [c, h, w]
//   4 维 [n, c, h, w]    -> [n*c, h, w]
// 输出 1 维 float32 Mat [3]。
int Shape::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int d0 = 1, d1 = 1, d2 = 1;
    if (bottom_blob.dims == 1)
    {
        d2 = bottom_blob.w;
    }
    else if (bottom_blob.dims == 2)
    {
        d1 = bottom_blob.h;
        d2 = bottom_blob.w;
    }
    else if (bottom_blob.dims == 3)
    {
        d0 = bottom_blob.c;
        d1 = bottom_blob.h;
        d2 = bottom_blob.w;
    }
    else // dims == 4
    {
        d0 = bottom_blob.c * bottom_blob.d;
        d1 = bottom_blob.h;
        d2 = bottom_blob.w;
    }

    top_blob.create(3);
    if (top_blob.empty())
        return -100;

    float* p = top_blob;
    p[0] = d0;
    p[1] = d1;
    p[2] = d2;
    return 0;
}

} // namespace ncnn
