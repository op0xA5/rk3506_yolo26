// ort.go - onnxruntime C API 的 cgo 封装 (静态库链接)。
//
// 内存说明 (目标芯片可用内存 ~60MB):
//   - 输入 OrtValue 只创建一次并复用 (CreateTensorWithDataAsOrtValue 零拷贝包装,
//     ORT 仅在 Run 期间读取输入, 复用安全), 避免每轮推理重复创建。
//   - 输出零拷贝: Run 直接暴露 OrtValue 底层内存的切片视图, 后处理就地消费,
//     避免 84*8400*4 ~= 2.8MB 的每轮拷贝与 GC 压力, Close 时才释放。
//   - 注意: onnxruntime 在 32 位构建上禁用 CPU 内存池 (allocator_utils.cc:
//     sizeof(void*)==4 时 arena 关闭), ARM32 上所有张量分配走 malloc/free,
//     不存在可收缩/可配置的 arena; 降低峰值内存的主要手段是零拷贝/缓冲复用
//     以及使用 INT8 模型 (激活张量约为 fp32 的 1/4)。
//
// 注意: cgo 无法直接索引 OrtApi 结构体的函数指针成员，
// 因此所有 API 调用都在下方的 C 包装函数中完成。
//
// 编译时通过 Makefile 注入:
//
//	CGO_CFLAGS  = -I$(dist/armv7l/include)
//	CGO_LDFLAGS = -Wl,--start-group $(dist/armv7l/lib/*.a) -Wl,--end-group -lstdc++ -ldl -lpthread -lm
package main

/*
#cgo CFLAGS: -I${SRCDIR}/../dist/armv7l/include
#include <stdlib.h>
#include <string.h>
#include "onnxruntime_c_api.h"

static OrtApi* ort_get_api(void) {
	const OrtApiBase* base = OrtGetApiBase();
	if (base == NULL) return NULL;
	return (OrtApi*)(base->GetApi(ORT_API_VERSION));
}

// ---- 以下为 OrtApi 调用包装 (Go 侧不直接访问 api 结构体成员) ----

static OrtStatus* w_create_env(OrtApi* a, OrtEnv** out) {
	return a->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "yolo26n-demo", out);
}
static OrtStatus* w_create_session_options(OrtApi* a, OrtSessionOptions** o) {
	return a->CreateSessionOptions(o);
}
static OrtStatus* w_set_intra_threads(OrtApi* a, OrtSessionOptions* o, int n) {
	return a->SetIntraOpNumThreads(o, n);
}
static OrtStatus* w_set_inter_threads(OrtApi* a, OrtSessionOptions* o, int n) {
	return a->SetInterOpNumThreads(o, n);
}
static OrtStatus* w_set_graph_opt(OrtApi* a, OrtSessionOptions* o) {
	return a->SetSessionGraphOptimizationLevel(o, ORT_ENABLE_ALL);
}
static OrtStatus* w_create_session(OrtApi* a, OrtEnv* env, const char* path,
                                   OrtSessionOptions* o, OrtSession** s) {
	return a->CreateSession(env, path, o, s);
}
static void w_release_session_options(OrtApi* a, OrtSessionOptions* o) { a->ReleaseSessionOptions(o); }
static void w_release_session(OrtApi* a, OrtSession* s)               { a->ReleaseSession(s); }
static void w_release_env(OrtApi* a, OrtEnv* e)                       { a->ReleaseEnv(e); }
static void w_release_meminfo(OrtApi* a, OrtMemoryInfo* m)            { a->ReleaseMemoryInfo(m); }
static void w_release_value(OrtApi* a, OrtValue* v)                   { a->ReleaseValue(v); }
static void w_release_typeinfo(OrtApi* a, OrtTypeInfo* t)             { a->ReleaseTypeInfo(t); }
static void w_release_tensor_shape(OrtApi* a, OrtTensorTypeAndShapeInfo* t) {
	a->ReleaseTensorTypeAndShapeInfo(t);
}
static const char* w_status_message(OrtApi* a, OrtStatus* s) { return a->GetErrorMessage(s); }
static void w_release_status(OrtApi* a, OrtStatus* s)        { a->ReleaseStatus(s); }

static OrtStatus* w_create_cpu_meminfo(OrtApi* a, OrtMemoryInfo** m) {
	return a->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, m);
}
static OrtStatus* w_get_default_allocator(OrtApi* a, OrtAllocator** al) {
	return a->GetAllocatorWithDefaultOptions(al);
}
static OrtStatus* w_session_input_name(OrtApi* a, OrtSession* s, size_t idx,
                                       OrtAllocator* al, char** name) {
	return a->SessionGetInputName(s, idx, al, name);
}
static OrtStatus* w_session_output_name(OrtApi* a, OrtSession* s, size_t idx,
                                        OrtAllocator* al, char** name) {
	return a->SessionGetOutputName(s, idx, al, name);
}
static OrtStatus* w_session_output_count(OrtApi* a, OrtSession* s, size_t* n) {
	return a->SessionGetOutputCount(s, n);
}
static OrtStatus* w_session_input_typeinfo(OrtApi* a, OrtSession* s, size_t idx, OrtTypeInfo** t) {
	return a->SessionGetInputTypeInfo(s, idx, t);
}
static OrtStatus* w_cast_tensorinfo(OrtApi* a, OrtTypeInfo* t, const OrtTensorTypeAndShapeInfo** o) {
	return a->CastTypeInfoToTensorInfo(t, o);
}
static OrtStatus* w_dims_count(OrtApi* a, const OrtTensorTypeAndShapeInfo* t, size_t* n) {
	return a->GetDimensionsCount(t, n);
}
static OrtStatus* w_dims(OrtApi* a, const OrtTensorTypeAndShapeInfo* t, int64_t* dims, size_t n) {
	return a->GetDimensions(t, dims, n);
}
static OrtStatus* w_create_tensor(OrtApi* a, OrtMemoryInfo* m, void* data, size_t bytes,
                                  const int64_t* shape, size_t ndims, OrtValue** v) {
	return a->CreateTensorWithDataAsOrtValue(m, data, bytes, shape, ndims,
	                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, v);
}

// 注意: onnxruntime 在 32 位构建上整体禁用 CPU 内存池
// (onnxruntime/core/framework/allocator_utils.cc: sizeof(void*)==4 -> arena 关闭,
// 防止整数溢出死循环), ARM32 上所有张量分配直接走 malloc/free,
// 因此不存在可收缩的 arena; 内存优化只能靠零拷贝/缓冲复用 (见文件头注释)。

static OrtStatus* w_run(OrtApi* a, OrtSession* s, OrtRunOptions* ro,
                        const char* in_name, OrtValue* in,
                        const char* out_name, OrtValue** out) {
	const OrtValue* in_ptr = in;
	return a->Run(s, ro, &in_name, &in_ptr, 1, &out_name, 1, out);
}
static OrtStatus* w_tensor_type_and_shape(OrtApi* a, OrtValue* v, OrtTensorTypeAndShapeInfo** t) {
	return a->GetTensorTypeAndShape(v, t);
}
static OrtStatus* w_tensor_mutable_data(OrtApi* a, OrtValue* v, void** p) {
	return a->GetTensorMutableData(v, p);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Ort 封装 onnxruntime 运行时会话 (输入/输出零拷贝, 缓冲复用)。
type Ort struct {
	api        *C.OrtApi
	env        *C.OrtEnv
	session    *C.OrtSession
	memInfo    *C.OrtMemoryInfo
	inputName  *C.char // 由 ORT allocator 分配
	outputName *C.char

	inputValue *C.OrtValue // 复用的输入 tensor (包装 inputBuf 内存)
	inputBuf   []float32   // 复用的 NCHW 输入缓冲 (外部须写入此切片)

	InputShape []int64 // NCHW, 例如 [1,3,640,640]
}

// statusErr 将 OrtStatus 转为 Go error 并释放。
func (o *Ort) statusErr(st *C.OrtStatus) error {
	if st == nil {
		return nil
	}
	msg := C.GoString(C.w_status_message(o.api, st))
	C.w_release_status(o.api, st)
	return errors.New(msg)
}

// NewOrt 加载模型并创建会话。threads 为 intra-op 线程数。
func NewOrt(modelPath string, threads int) (*Ort, error) {
	api := C.ort_get_api()
	if api == nil {
		return nil, errors.New("OrtGetApiBase()->GetApi 失败，onnxruntime 版本不匹配？")
	}
	o := &Ort{api: api}

	if err := o.statusErr(C.w_create_env(api, &o.env)); err != nil {
		return nil, fmt.Errorf("CreateEnv: %w", err)
	}

	if err := o.statusErr(C.w_create_cpu_meminfo(api, &o.memInfo)); err != nil {
		return nil, fmt.Errorf("CreateCpuMemoryInfo: %w", err)
	}

	var opts *C.OrtSessionOptions
	if err := o.statusErr(C.w_create_session_options(api, &opts)); err != nil {
		return nil, fmt.Errorf("CreateSessionOptions: %w", err)
	}
	defer C.w_release_session_options(api, opts)

	if err := o.statusErr(C.w_set_intra_threads(api, opts, C.int(threads))); err != nil {
		return nil, fmt.Errorf("SetIntraOpNumThreads: %w", err)
	}
	if err := o.statusErr(C.w_set_inter_threads(api, opts, 1)); err != nil {
		return nil, fmt.Errorf("SetInterOpNumThreads: %w", err)
	}
	if err := o.statusErr(C.w_set_graph_opt(api, opts)); err != nil {
		return nil, fmt.Errorf("SetSessionGraphOptimizationLevel: %w", err)
	}

	cpath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cpath))
	if err := o.statusErr(C.w_create_session(api, o.env, cpath, opts, &o.session)); err != nil {
		return nil, fmt.Errorf("CreateSession(%s): %w", modelPath, err)
	}

	var alloc *C.OrtAllocator
	if err := o.statusErr(C.w_get_default_allocator(api, &alloc)); err != nil {
		return nil, err
	}
	if err := o.statusErr(C.w_session_input_name(api, o.session, 0, alloc, &o.inputName)); err != nil {
		return nil, fmt.Errorf("SessionGetInputName: %w", err)
	}

	var typeInfo *C.OrtTypeInfo
	if err := o.statusErr(C.w_session_input_typeinfo(api, o.session, 0, &typeInfo)); err != nil {
		return nil, fmt.Errorf("SessionGetInputTypeInfo: %w", err)
	}
	defer C.w_release_typeinfo(api, typeInfo)

	var tinfo *C.OrtTensorTypeAndShapeInfo
	if err := o.statusErr(C.w_cast_tensorinfo(api, typeInfo, &tinfo)); err != nil {
		return nil, err
	}
	var ndims C.size_t
	if err := o.statusErr(C.w_dims_count(api, tinfo, &ndims)); err != nil {
		return nil, err
	}
	dims := make([]C.int64_t, ndims)
	if err := o.statusErr(C.w_dims(api, tinfo, &dims[0], ndims)); err != nil {
		return nil, err
	}
	o.InputShape = make([]int64, ndims)
	for i := range dims {
		o.InputShape[i] = int64(dims[i])
	}

	if err := o.statusErr(C.w_session_output_name(api, o.session, 0, alloc, &o.outputName)); err != nil {
		return nil, fmt.Errorf("SessionGetOutputName: %w", err)
	}
	var count C.size_t
	if err := o.statusErr(C.w_session_output_count(api, o.session, &count)); err != nil {
		return nil, err
	}
	if count != 1 {
		return nil, fmt.Errorf("期望单输出模型，实际输出数 %d", int(count))
	}

	// 预分配输入缓冲并一次性创建输入 OrtValue (后续 Run 复用, 零分配)
	numel := 1
	for _, d := range o.InputShape {
		numel *= int(d)
	}
	o.inputBuf = make([]float32, numel)
	if err := o.statusErr(C.w_create_tensor(api, o.memInfo,
		unsafe.Pointer(&o.inputBuf[0]), C.size_t(numel*4),
		(*C.int64_t)(unsafe.Pointer(&o.InputShape[0])), C.size_t(len(o.InputShape)),
		&o.inputValue)); err != nil {
		return nil, fmt.Errorf("CreateTensor(input): %w", err)
	}
	return o, nil
}

// Input 返回可复用的 NCHW 输入缓冲；预处理后直接调用 Run，无需任何拷贝。
func (o *Ort) Input() []float32 { return o.inputBuf }

// Output 推理输出的零拷贝视图，Close 后 Data 失效。
type Output struct {
	Data  []float32 // 直接映射 OrtValue 底层内存, 禁止越界写
	Shape []int64
	value *C.OrtValue
	ort   *Ort
}

// Close 释放输出 tensor (调用后 Data 不可再访问)。
func (out *Output) Close() {
	if out.value != nil {
		C.w_release_value(out.ort.api, out.value)
		out.value = nil
		out.Data = nil
	}
}

// Close 释放会话资源。
func (o *Ort) Close() {
	if o.inputValue != nil {
		C.w_release_value(o.api, o.inputValue)
	}
	if o.session != nil {
		C.w_release_session(o.api, o.session)
	}
	if o.memInfo != nil {
		C.w_release_meminfo(o.api, o.memInfo)
	}
	if o.env != nil {
		C.w_release_env(o.api, o.env)
	}
}

// Run 执行推理。输入须已写入 Input() 返回的缓冲。
// 返回零拷贝输出视图 (调用方负责 Close, 建议立即消费)。
func (o *Ort) Run() (*Output, error) {
	var outputTensor *C.OrtValue
	err := o.statusErr(C.w_run(o.api, o.session, nil, o.inputName, o.inputValue, o.outputName, &outputTensor))
	if err != nil {
		return nil, fmt.Errorf("SessionRun: %w", err)
	}

	var outInfo *C.OrtTensorTypeAndShapeInfo
	if err := o.statusErr(C.w_tensor_type_and_shape(o.api, outputTensor, &outInfo)); err != nil {
		C.w_release_value(o.api, outputTensor)
		return nil, err
	}
	var ndims C.size_t
	if err := o.statusErr(C.w_dims_count(o.api, outInfo, &ndims)); err != nil {
		C.w_release_tensor_shape(o.api, outInfo)
		C.w_release_value(o.api, outputTensor)
		return nil, err
	}
	dims := make([]C.int64_t, ndims)
	if err := o.statusErr(C.w_dims(o.api, outInfo, &dims[0], ndims)); err != nil {
		C.w_release_tensor_shape(o.api, outInfo)
		C.w_release_value(o.api, outputTensor)
		return nil, err
	}
	C.w_release_tensor_shape(o.api, outInfo)

	shape := make([]int64, ndims)
	numel := 1
	for i := range dims {
		shape[i] = int64(dims[i])
		numel *= int(shape[i])
	}

	var data unsafe.Pointer
	if err := o.statusErr(C.w_tensor_mutable_data(o.api, outputTensor, &data)); err != nil {
		C.w_release_value(o.api, outputTensor)
		return nil, err
	}
	return &Output{
		Data:  unsafe.Slice((*float32)(data), numel),
		Shape: shape,
		value: outputTensor,
		ort:   o,
	}, nil
}
