package govec

import (
	"github.com/x448/float16"
)

type Float16 = float16.Float16

func DotProductSlow(a, b []Float16) Float16 {
	if len(a) != len(b) {
		panic("no")
	}

	var sum float32
	for i := 0; i < len(a); i++ {
		sum += (a[i].Float32() * b[i].Float32())
	}

	return float16.Fromfloat32(sum)
}
