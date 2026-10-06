// ncnn.go - ncnn (ARM32 NEON 静态库) 的 cgo 封装。
//
// 编译时通过 Makefile 注入:
//
//	CGO_CXXFLAGS = -I$(dist/ncnn-arm/include/ncnn)
//	CGO_LDFLAGS  = $(dist/ncnn-arm/lib/libncnn.a) -lstdc++ -lpthread -lm
//
// 内存说明 (目标芯片可用内存 ~60MB):
//   - 输入缓冲由 C 侧 ncnn 对齐分配器创建并复用, Go 直接写入, 推理零拷贝。
//   - 输出 Mat 由 handle 持有, Go 侧直接消费数据指针, 无每轮拷贝。
package main

/*
#cgo CXXFLAGS: -I${SRCDIR}/../dist/ncnn-arm/include/ncnn
#cgo LDFLAGS: -L${SRCDIR}/../dist/ncnn-arm/lib -lncnn -lstdc++ -lpthread -lm
#include <stdlib.h>
#include "wrapper.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"
)

// Ncnn 封装 ncnn 网络 (输入/输出零拷贝, 缓冲复用)。
type Ncnn struct {
	h      *C.ncnn_handle_t
	inBuf  []float32 // ncnn_input_buf 缓冲的 Go 视图 (复用, 禁止重新分配)
	inHW   [2]int    // 当前输入尺寸 (c 固定 3)
	OutDim [3]int    // 最近一次的输出形状 [c,h,w] (dims<=3)
}

// NewNcnn 加载模型 (.param 路径, .bin 同名派生), threads 仅作默认展示。
func NewNcnn(paramPath string, threads int) (*Ncnn, error) {
	binPath := strings.TrimSuffix(paramPath, ".param") + ".bin"
	h := C.ncnn_create()
	if h == nil {
		return nil, errors.New("ncnn_create 失败")
	}
	C.ncnn_set_default_threads(h, C.int(threads))
	n := &Ncnn{h: h}
	cparam := C.CString(paramPath)
	cbin := C.CString(binPath)
	defer C.free(unsafe.Pointer(cparam))
	defer C.free(unsafe.Pointer(cbin))
	if rc := C.ncnn_load(h, cparam, cbin); rc != 0 {
		C.ncnn_destroy(h)
		return nil, fmt.Errorf("ncnn_load(%s/%s) 失败: rc=%d", paramPath, binPath, int(rc))
	}
	return n, nil
}

// Input 返回复用的 NCHW 输入缓冲 (3*c*h*w, 写入后直接调用 Run)。
func (n *Ncnn) Input(inH, inW int) []float32 {
	if n.inBuf == nil || n.inHW != [2]int{inH, inW} {
		p := C.ncnn_input_buf(n.h, 3, C.int(inH), C.int(inW))
		if p == nil {
			return nil
		}
		n.inBuf = unsafe.Slice((*float32)(p), 3*inH*inW)
		n.inHW = [2]int{inH, inW}
	}
	return n.inBuf
}

// Run 执行推理, 返回零拷贝输出视图 (下次 Run 前有效, 禁止越界写)。
func (n *Ncnn) Run(threads int) ([]float32, [3]int, error) {
	if rc := C.ncnn_run(n.h, C.int(threads)); rc != 0 {
		return nil, [3]int{}, fmt.Errorf("ncnn_run 失败: rc=%d", int(rc))
	}
	var ndims C.int
	var shape [4]C.int
	p := C.ncnn_output(n.h, &ndims, (*C.int)(unsafe.Pointer(&shape[0])))
	if p == nil {
		return nil, [3]int{}, errors.New("ncnn_output 返回空")
	}
	// 统一为 [c,h,w] 语义, dims<3 时高位补 1
	d := [3]int{1, 1, 1}
	switch ndims {
	case 1:
		d[2] = int(shape[0])
	case 2:
		d[1], d[2] = int(shape[0]), int(shape[1])
	case 3:
		d[0], d[1], d[2] = int(shape[0]), int(shape[1]), int(shape[2])
	default: // 4 维 NCHW: 展平为 [N*C, H, W]
		d[0], d[1], d[2] = int(shape[0])*int(shape[1]), int(shape[2]), int(shape[3])
	}
	numel := d[0] * d[1] * d[2]
	n.OutDim = d
	return unsafe.Slice((*float32)(p), numel), d, nil
}

// Close 释放模型资源。
func (n *Ncnn) Close() {
	if n.h != nil {
		C.ncnn_destroy(n.h)
		n.h = nil
		n.inBuf = nil
	}
}
