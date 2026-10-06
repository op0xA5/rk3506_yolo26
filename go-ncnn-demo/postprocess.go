package main

import "sort"

// Detection 一个检测框（原图坐标系，xyxy）。
type Detection struct {
	X1, Y1, X2, Y2 float64
	Score          float32
	ClassID        int
}

// postprocess 解析模型输出。
// 自动识别两种输出格式:
//
//	A) NMS-free 端到端输出: [1, N, 6]，每行 (x1,y1,x2,y2,score,class)
//	B) Ultralytics 经典输出: [1, 4+numClasses, N] (cx,cy,w,h + 各类得分)，
//	   此时需要再做阈值过滤与 NMS。
func postprocess(data []float32, shape []int64, meta letterboxMeta, confThres float64, numClasses int, applyNMS bool) []Detection {
	if len(shape) == 3 && shape[0] == 1 && shape[2] == 6 {
		return decodeEnd2End(data, int(shape[1]), meta, confThres)
	}
	if len(shape) == 3 && shape[0] == 1 && shape[1] == int64(4+numClasses) {
		return decodeRaw(data, int(shape[1]), int(shape[2]), meta, confThres, applyNMS)
	}
	// 兜底: [1, N, 4+numClasses] (end2end, 每行最后两位为 score/class)
	if len(shape) == 3 && shape[0] == 1 && shape[2] == int64(4+numClasses) {
		return decodeEnd2EndNC(data, int(shape[1]), int(shape[2]), meta, confThres)
	}
	return nil
}

// decodeEnd2End 解析 [1, N, 6] 端到端输出 (xyxy + score + class)。
func decodeEnd2End(data []float32, n int, meta letterboxMeta, confThres float64) []Detection {
	dets := make([]Detection, 0, n)
	for i := 0; i < n; i++ {
		row := data[i*6:]
		score := row[4]
		if float64(score) < confThres {
			continue
		}
		x1, y1 := meta.mapToSource(float64(row[0]), float64(row[1]))
		x2, y2 := meta.mapToSource(float64(row[2]), float64(row[3]))
		dets = append(dets, Detection{X1: x1, Y1: y1, X2: x2, Y2: y2,
			Score: score, ClassID: int(row[5])})
	}
	return dets
}

// decodeEnd2EndNC 解析 [1, N, 4+C] 端到端输出 (xyxy + C个得分, 取最大类)。
func decodeEnd2EndNC(data []float32, n, nc int, meta letterboxMeta, confThres float64) []Detection {
	dets := make([]Detection, 0, n)
	for i := 0; i < n; i++ {
		row := data[i*nc:]
		bestScore, bestCls := float32(0), 0
		for c := 4; c < nc; c++ {
			if row[c] > bestScore {
				bestScore, bestCls = row[c], c-4
			}
		}
		if float64(bestScore) < confThres {
			continue
		}
		x1, y1 := meta.mapToSource(float64(row[0]), float64(row[1]))
		x2, y2 := meta.mapToSource(float64(row[2]), float64(row[3]))
		dets = append(dets, Detection{X1: x1, Y1: y1, X2: x2, Y2: y2,
			Score: bestScore, ClassID: bestCls})
	}
	return dets
}

// decodeRaw 解析 [1, 4+C, N] 输出 (cx,cy,w,h + 各类得分)。
func decodeRaw(data []float32, nc, n int, meta letterboxMeta, confThres float64, applyNMS bool) []Detection {
	cand := make([]Detection, 0, 256)
	for i := 0; i < n; i++ {
		bestScore, bestCls := float32(confThres), -1
		for c := 4; c < nc; c++ {
			if s := data[c*n+i]; s > bestScore {
				bestScore, bestCls = s, c-4
			}
		}
		if bestCls < 0 {
			continue
		}
		cx, cy := float64(data[i]), float64(data[n+i])
		w, h := float64(data[2*n+i]), float64(data[3*n+i])
		x1, y1 := meta.mapToSource(cx-w/2, cy-h/2)
		x2, y2 := meta.mapToSource(cx+w/2, cy+h/2)
		cand = append(cand, Detection{X1: x1, Y1: y1, X2: x2, Y2: y2,
			Score: bestScore, ClassID: bestCls})
	}
	if applyNMS {
		return nms(cand, 0.45)
	}
	return cand
}

// nms 按类别做 greedy IoU NMS。
func nms(dets []Detection, iouThres float64) []Detection {
	sort.Slice(dets, func(i, j int) bool { return dets[i].Score > dets[j].Score })
	keep := make([]Detection, 0, len(dets))
	suppressed := make([]bool, len(dets))
	for i := range dets {
		if suppressed[i] {
			continue
		}
		keep = append(keep, dets[i])
		for j := i + 1; j < len(dets); j++ {
			if suppressed[j] || dets[j].ClassID != dets[i].ClassID {
				continue
			}
			if iou(dets[i], dets[j]) > iouThres {
				suppressed[j] = true
			}
		}
	}
	return keep
}

func iou(a, b Detection) float64 {
	ix1, iy1 := max(a.X1, b.X1), max(a.Y1, b.Y1)
	ix2, iy2 := min(a.X2, b.X2), min(a.Y2, b.Y2)
	iw, ih := ix2-ix1, iy2-iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	areaA := (a.X2 - a.X1) * (a.Y2 - a.Y1)
	areaB := (b.X2 - b.X1) * (b.Y2 - b.Y1)
	return inter / (areaA + areaB - inter)
}
