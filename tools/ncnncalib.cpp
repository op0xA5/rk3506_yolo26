// 自定义 int8 校准器: 用与运行时完全一致的预处理输入 (letterbox+/255+NCHW raw),
// 统计每个量化层的输入 blob absmax, 生成 activation scale = 127/absmax 的 table
// (权重条目从 ncnn2table 生成的表复制, 其 per-channel 权重 scale 质量良好)。
//
// 背景: ncnn2table 用直方图分位数裁剪 (threshold), 对校准集之外的输入
// (如颜色饱和的 bus.jpg) 造成大面积 clipping, 实测部分层输出退化为全零。
//
// 用法: ncnncalib fp32.param fp32.bin list.txt ncnn2table.table out.table
#include "net.h"
#include "mat.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <map>

int main(int argc, char** argv)
{
    // 1. 解析 param, 记录 Convolution/ConvDepthWise/Gemm 层及其输入 blob
    struct L { std::string type, in_blob; };
    std::vector<L> layers;
    std::vector<std::string> order; // 层名顺序 (与 table 对齐用)
    {
        FILE* pf = fopen(argv[1], "r");
        char line[4096];
        fgets(line, sizeof line, pf); fgets(line, sizeof line, pf);
        while (fgets(line, sizeof line, pf))
        {
            char type[64], name[256]; int ni, no;
            if (sscanf(line, "%63s %255s %d %d", type, name, &ni, &no) != 4) continue;
            if (!strstr(type, "Convolution") && strcmp(type, "Gemm") != 0) continue;
            char* p = line;
            for (int k = 0; k < 4; k++) { while (*p == ' ' || *p == '\t') p++; while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++; }
            char blob[256]; int i = 0;
            while (*p == ' ' || *p == '\t') p++;
            while (p[i] && p[i] != ' ' && p[i] != '\t' && p[i] != '\n' && i < 255) { blob[i] = p[i]; i++; }
            blob[i] = 0;
            layers.push_back({type, blob});
            order.push_back(name);
        }
        fclose(pf);
    }
    fprintf(stderr, "%zu 个量化层\n", layers.size());

    // 2. 逐张前向统计 absmax
    std::vector<float> absmax(layers.size(), 0.f);
    std::vector<std::string> files;
    {
        FILE* lf = fopen(argv[3], "r");
        char line[1024];
        while (fgets(line, sizeof line, lf))
        {
            line[strcspn(line, "\r\n")] = 0;
            if (line[0]) files.push_back(line);
        }
        fclose(lf);
    }
    ncnn::Net net;
    net.load_param(argv[1]);
    net.load_model(argv[2]);

    int nf = 0;
    for (auto& file : files)
    {
        FILE* f = fopen(file.c_str(), "rb");
        if (!f) continue;
        std::vector<float> buf(3 * 480 * 480);
        // 输入尺寸从 Input 层读: 固定按 param 首层, 这里用 480/320 自适应
        // 简化: 依据文件大小判断
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        buf.resize(sz / 4);
        if (fread(buf.data(), 4, buf.size(), f) != buf.size()) { fclose(f); continue; }
        fclose(f);
        int plane = (int)buf.size() / 3;
        int side = 1;
        while (side * side < plane) side++;
        ncnn::Mat in(side, side, 3, (void*)buf.data());

        ncnn::Extractor ex = net.create_extractor();
        ex.input(net.input_names()[0], in);
        for (size_t i = 0; i < layers.size(); i++)
        {
            ncnn::Mat m;
            if (ex.extract(layers[i].in_blob.c_str(), m) != 0) continue;
            const float* p = m;
            // int8 模型才需要 scale, fp32 网络此处都是 fp32 blob
            if (m.elemsize != 4u) continue;
            float am = 0;
            for (int j = 0; j < m.total(); j++)
            {
                float v = fabsf(p[j]);
                if (v > am) am = v;
            }
            if (am > absmax[i]) absmax[i] = am;
        }
        nf++;
        if (nf % 32 == 0) fprintf(stderr, "  %d/%zu\n", nf, files.size());
    }

    // 3. 输出 table: 复制权重条目, 激活条目用 127/absmax
    std::map<std::string, float> act;
    for (size_t i = 0; i < layers.size(); i++)
    {
        float am = absmax[i];
        act[order[i]] = am > 0 ? 127.f / am : 1.f;
        fprintf(stderr, "%-24s absmax=%10.4f scale=%.6f\n", order[i].c_str(), am, act[order[i]]);
    }

    FILE* tf = fopen(argv[4], "r");   // ncnn2table 的表 (取权重条目)
    FILE* of = fopen(argv[5], "w");
    char line[65536];
    while (fgets(line, sizeof line, tf))
    {
        char key[512];
        if (sscanf(line, "%511s", key) != 1) continue;
        char* sp = strchr(key, 0);
        (void)sp;
        if (strstr(key, "_param_0"))
        {
            fputs(line, of); // 权重条目原样复制
        }
        else
        {
            auto it = act.find(key);
            if (it != act.end())
                fprintf(of, "%s %f \n", key, it->second);
            else
                fputs(line, of);
        }
    }
    fclose(tf);
    fclose(of);
    fprintf(stderr, "完成: %s\n", argv[5]);
    return 0;
}
