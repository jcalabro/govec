package govec

import (
	"fmt"
	"math/rand"
	"testing"
)

func randomVec16(n int) []Float16 {
	out := make([]Float16, n)

	for i := 0; i < n; i++ {
		out[i] = Float16(rand.Uint32())
	}

	return out
}

func TestDotProduct(t *testing.T) {
	a := randomVec16(512)
	b := randomVec16(512)

	dpa := DotProductSlow(a, b)
	dpb := DotProductFast(a, b)

	fmt.Println(dpa, dpb)
}
