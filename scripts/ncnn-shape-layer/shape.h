// Copyright 2026 zhujl
// Shape 层: 输出输入张量的形状 (1 维 int32-as-float Mat)。
// 用于 onnx2ncnn 转换 ultralytics YOLO 检测头时烘焙的 Shape 子图
// (anchor 解码需要 shape[1] 计算通道切分点)。ncnn 官方未内置该层。
#ifndef LAYER_SHAPE_H
#define LAYER_SHAPE_H

#include "layer.h"

namespace ncnn {

class Shape : public Layer
{
public:
    Shape();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob) const;
};

} // namespace ncnn

#endif // LAYER_SHAPE_H
