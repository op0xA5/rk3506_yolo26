package main

import (
	"fmt"
	"image"
	"os"
)

// letterboxMeta 记录 letterbox 缩放信息，用于将模型输出坐标映射回原图。
type letterboxMeta struct {
	scale float64 // 原图 -> 模型输入的缩放比
	padX  float64 // 宽边填充（左右）
	padY  float64 // 高边填充（上下）
	inW   int     // 模型输入宽
	inH   int     // 模型输入高
	srcW  int     // 原图宽
	srcH  int     // 原图高
}

// decodeImage 读取 jpeg/png 图片。
func decodeImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("解码图片失败 %s: %w", path, err)
	}
	return img, nil
}

// letterboxPreprocess 将图片等比缩放并居中填充到 inW x inH，
// 转换为 NCHW float32，像素归一化到 [0,1]，RGB 通道顺序。
// 与 Ultralytics 默认预处理保持一致 (scaleup=false, fill=114)。
func letterboxPreprocess(img image.Image, inW, inH int) ([]float32, letterboxMeta) {
	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()

	scale := float64(inW) / float64(srcW)
	if s := float64(inH) / float64(srcH); s < scale {
		scale = s
	}
	newW := int(float64(srcW)*scale + 0.5)
	newH := int(float64(srcH)*scale + 0.5)
	padX := float64(inW-newW) / 2
	padY := float64(inH-newH) / 2

	// 先等比缩放到 (newW, newH)
	resized := resizeBilinear(img, newW, newH)

	// 填充 + 归一化 + HWC->CHW + RGB
	input := make([]float32, 3*inH*inW)
	for y := 0; y < inH; y++ {
		sy := y - int(padY)
		for x := 0; x < inW; x++ {
			sx := x - int(padX)
			r, g, bch := uint8(114), uint8(114), uint8(114)
			if sx >= 0 && sx < newW && sy >= 0 && sy < newH {
				c := resized[3*(sy*newW+sx):]
				r, g, bch = c[0], c[1], c[2]
			}
			// RGB -> 归一化, 按 CHW 布局
			input[0*inH*inW+y*inW+x] = float32(r) / 255
			input[1*inH*inW+y*inW+x] = float32(g) / 255
			input[2*inH*inW+y*inW+x] = float32(bch) / 255
		}
	}
	return input, letterboxMeta{scale: scale, padX: padX, padY: padY,
		inW: inW, inH: inH, srcW: srcW, srcH: srcH}
}

// resizeBilinear 双线性缩放，返回 HWC RGB 字节数组（避免引入 image/draw 重采样依赖）。
func resizeBilinear(img image.Image, dstW, dstH int) []uint8 {
	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	out := make([]uint8, dstW*dstH*3)

	// 中心对齐的采样坐标
	for dy := 0; dy < dstH; dy++ {
		sy := (float64(dy)+0.5)*float64(srcH)/float64(dstH) - 0.5
		y0 := int(sy)
		fy := sy - float64(y0)
		if y0 < 0 {
			y0, fy = 0, 0
		}
		y1 := y0 + 1
		if y1 >= srcH {
			y1 = srcH - 1
		}
		for dx := 0; dx < dstW; dx++ {
			sx := (float64(dx)+0.5)*float64(srcW)/float64(dstW) - 0.5
			x0 := int(sx)
			fx := sx - float64(x0)
			if x0 < 0 {
				x0, fx = 0, 0
			}
			x1 := x0 + 1
			if x1 >= srcW {
				x1 = srcW - 1
			}

			r00, g00, b00, _ := img.At(b.Min.X+x0, b.Min.Y+y0).RGBA()
			r10, g10, b10, _ := img.At(b.Min.X+x1, b.Min.Y+y0).RGBA()
			r01, g01, b01, _ := img.At(b.Min.X+x0, b.Min.Y+y1).RGBA()
			r11, g11, b11, _ := img.At(b.Min.X+x1, b.Min.Y+y1).RGBA()

			w00 := (1 - fx) * (1 - fy)
			w10 := fx * (1 - fy)
			w01 := (1 - fx) * fy
			w11 := fx * fy

			i := 3 * (dy*dstW + dx)
			out[i+0] = uint8((w00*float64(r00>>8) + w10*float64(r10>>8) + w01*float64(r01>>8) + w11*float64(r11>>8)) + 0.5)
			out[i+1] = uint8((w00*float64(g00>>8) + w10*float64(g10>>8) + w01*float64(g01>>8) + w11*float64(g11>>8)) + 0.5)
			out[i+2] = uint8((w00*float64(b00>>8) + w10*float64(b10>>8) + w01*float64(b01>>8) + w11*float64(b11>>8)) + 0.5)
		}
	}
	return out
}

// mapToSource 将 letterbox 坐标系下的坐标映射回原图坐标 (并钳制到图像范围内)。
func (m letterboxMeta) mapToSource(x, y float64) (float64, float64) {
	x = (x - m.padX) / m.scale
	y = (y - m.padY) / m.scale
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x > float64(m.srcW) {
		x = float64(m.srcW)
	}
	if y > float64(m.srcH) {
		y = float64(m.srcH)
	}
	return x, y
}
