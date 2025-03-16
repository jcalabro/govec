package govec

import (
	"math"
	"unsafe"
)

type BF16 uint16

func DotProductSlow(a, b []BF16) float32 {
	if len(a) != len(b) {
		panic("no")
	}

	var sum float32
	for i := 0; i < len(a); i++ {
		sum += (BF16toFloat32(a[i]) * BF16toFloat32(b[i]))
	}

	return sum
}

func ToBF16(f float32) BF16 {
	v := *(*uint32)(unsafe.Pointer(&f))
	return BF16(v >> 16)
}

func BF16toFloat32(f BF16) float32 {
	v := uint32(f) << 16
	return *(*float32)(unsafe.Pointer(&v))

}

func L2DistanceSlow(a, b []BF16) float32 {
	var final float32
	for i := 0; i < len(a); i++ {
		v := BF16toFloat32(a[i]) - BF16toFloat32(b[i])
		final += v * v
	}

	return float32(math.Sqrt(float64(final)))
}
