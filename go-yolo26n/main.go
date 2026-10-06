// yolo26n-demo - 基于 onnxruntime (ARM32 NEON 静态库) 的 YOLO26n NMS-free 目标检测演示。
//
// 全流程: 图片解码 -> letterbox 预处理 -> onnxruntime 推理 -> 后处理解码 -> 结果绘制，
// 每个阶段统计耗时，用于评估芯片推理性能。
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"runtime/debug"
	"strconv"
	"strings"
	"time"
)

// vmRSS 读取 /proc/self/statm 的第 2 个字段 (常驻页数)*页size, 用于内存分析。
func vmRSS() int {
	b, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		fmt.Fprintln(os.Stderr, "[内存] 读取 /proc/self/statm 失败:", err)
		return -1
	}
	f := strings.Fields(string(b))
	if len(f) < 2 {
		return -1
	}
	pages, _ := strconv.ParseInt(f[1], 10, 64)
	return int(pages * 4096 / 1024)
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func main() {
	var (
		model     = flag.String("model", "../models/yolo26n.ort", "onnx/ort 模型路径 (.ort 需 minimal build 运行时)")
		imagePath = flag.String("image", "../images/test.jpg", "输入图片路径")
		outPath   = flag.String("out", "", "标注结果输出图片路径 (默认: 不保存)")
		conf      = flag.Float64("conf", 0.25, "置信度阈值")
		iters     = flag.Int("iters", 10, "推理重复次数 (取统计值)")
		threads   = flag.Int("threads", 2, "onnxruntime intra-op 线程数")
		nmsFlag   = flag.Bool("nms", true, "对原始输出格式 [1,4+C,N] 做 NMS (端到端 [1,N,6] 输出不受影响)")
		numCls    = flag.Int("classes", 80, "类别数 (COCO=80)")
		memLimit  = flag.Int("memlimit", 16, "Go 堆内存软上限 (MB, 供 GC 参考; 0 不限制)")
	)
	flag.Parse()

	// Go 堆内存控制: 内存受限芯片上压低 GC 阈值, 避免 Go 堆无节制增长
	if *memLimit > 0 {
		debug.SetMemoryLimit(int64(*memLimit) << 20)
		debug.SetGCPercent(50)
	}

	// ---------- 1. 模型加载 ----------
	t0 := time.Now()
	ort, err := NewOrt(*model, *threads)
	if err != nil {
		fmt.Fprintln(os.Stderr, "加载模型失败:", err)
		os.Exit(1)
	}
	defer ort.Close()
	loadDur := time.Since(t0)
	if len(ort.InputShape) != 4 {
		fmt.Fprintln(os.Stderr, "模型输入应为 4 维 NCHW, 实际:", ort.InputShape)
		os.Exit(1)
	}
	inH, inW := int(ort.InputShape[2]), int(ort.InputShape[3])
	fmt.Printf("模型: %s\n", *model)
	fmt.Printf("输入: %s  [%d x %d], intra-op threads = %d\n",
		shapeStr(ort.InputShape), inW, inH, *threads)
	fmt.Printf("[内存] 模型加载后 VmRSS = %d KB\n\n", vmRSS())

	// ---------- 2. 图片解码 + 预处理 (写入复用的输入缓冲, 零分配) ----------
	t0 = time.Now()
	img, err := decodeImage(*imagePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	decodeDur := time.Since(t0)

	t0 = time.Now()
	input := ort.Input() // 复用缓冲: 预处理后直接推理, 无拷贝
	meta := letterboxPreprocess(img, inW, inH, input, nil)
	preprocessDur := time.Since(t0)

	// ---------- 3. 推理 (含 warmup; 输入/输出均零拷贝, 重复轮次零分配) ----------
	out, err := ort.Run() // warmup: 线程池初始化/缓存预热
	if err != nil {
		fmt.Fprintln(os.Stderr, "warmup 推理失败:", err)
		os.Exit(1)
	}
	var total, min, max time.Duration
	for i := 0; i < *iters; i++ {
		out.Close()
		t0 = time.Now()
		if out, err = ort.Run(); err != nil {
			fmt.Fprintln(os.Stderr, "推理失败:", err)
			os.Exit(1)
		}
		d := time.Since(t0)
		total += d
		if i == 0 || d < min {
			min = d
		}
		if d > max {
			max = d
		}
	}
	inferAvg := total / time.Duration(*iters)
	fmt.Printf("[内存] 推理结束后 VmRSS = %d KB\n", vmRSS())

	// ---------- 4. 后处理 (直接消费零拷贝输出, 立即释放) ----------
	t0 = time.Now()
	dets := postprocess(out.Data, out.Shape, meta, *conf, *numCls, *nmsFlag)
	outShape := out.Shape
	out.Close()
	postprocessDur := time.Since(t0)

	// ---------- 5. 结果输出 ----------
	fmt.Printf("输出 shape: %s, 检出目标 %d 个 (conf>=%.2f)\n", shapeStr(outShape), len(dets), *conf)
	for _, d := range dets {
		fmt.Printf("  %-15s %.2f  [%.0f, %.0f, %.0f, %.0f]\n",
			cocoLabel(d.ClassID), d.Score, d.X1, d.Y1, d.X2, d.Y2)
	}

	if *outPath != "" {
		if err := saveAnnotated(img, dets, *outPath); err != nil {
			fmt.Fprintln(os.Stderr, "保存结果图失败:", err)
		} else {
			fmt.Printf("\n标注图已保存: %s\n", *outPath)
		}
	}

	// ---------- 6. 耗时统计 ----------
	fmt.Printf(`
===== 耗时统计 (%d 次推理) =====
  模型加载      : %8.2f ms   (仅首次)
  图片解码      : %8.2f ms
  预处理        : %8.2f ms   (letterbox+%d*%d, RGB, /255, NCHW)
  推理 avg      : %8.2f ms   (min %.2f / max %.2f)
  后处理        : %8.2f ms   (解码%s)
  单帧端到端    : %8.2f ms   (解码+预处理+推理avg+后处理)
`,
		*iters,
		ms(loadDur), ms(decodeDur), ms(preprocessDur), inW, inH,
		ms(inferAvg), ms(min), ms(max), ms(postprocessDur),
		map[bool]string{true: "+NMS", false: ""}[*nmsFlag],
		ms(decodeDur)+ms(preprocessDur)+ms(inferAvg)+ms(postprocessDur))
}

func shapeStr(s []int64) string {
	out := "["
	for i, d := range s {
		if i > 0 {
			out += ","
		}
		out += fmt.Sprint(d)
	}
	return out + "]"
}

// saveAnnotated 在原图上绘制检测框并保存。
func saveAnnotated(src image.Image, dets []Detection, path string) error {
	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			dst.Set(x, y, src.At(b.Min.X+x, b.Min.Y+y))
		}
	}
	for _, d := range dets {
		c := color.RGBA{255, uint8(60 * (d.ClassID % 4)), uint8(80 * (d.ClassID % 3)), 255}
		drawRect(dst, int(d.X1), int(d.Y1), int(d.X2), int(d.Y2), 2, c)
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	switch ext := path[len(path)-3:]; ext {
	case "png":
		return png.Encode(f, dst)
	default:
		return jpeg.Encode(f, dst, &jpeg.Options{Quality: 90})
	}
}

func drawRect(img *image.RGBA, x1, y1, x2, y2, t int, c color.RGBA) {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	clamp := func(v, lo, hi int) int {
		if v < lo {
			return lo
		}
		if v > hi {
			return hi
		}
		return v
	}
	x1, x2 = clamp(x1, 0, w-1), clamp(x2, 0, w-1)
	y1, y2 = clamp(y1, 0, h-1), clamp(y2, 0, h-1)
	for k := 0; k < t; k++ {
		for x := x1; x <= x2; x++ {
			img.SetRGBA(x, y1+k, c)
			img.SetRGBA(x, y2-k, c)
		}
		for y := y1; y <= y2; y++ {
			img.SetRGBA(x1+k, y, c)
			img.SetRGBA(x2-k, y, c)
		}
	}
}
