package govec

import (
	"fmt"
	"math/rand"
	"testing"
)

func randomVec16(n int) []BF16 {
	out := make([]BF16, n)

	for i := 0; i < n; i++ {
		out[i] = ToBF16(rand.Float32())
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

func BenchmarkDotProduct(b *testing.B) {
	arrlen := 1000
	vecs := make([][]BF16, arrlen)
	for i := 0; i < arrlen; i++ {
		vecs[i] = randomVec16(512)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		DotProductFast(vecs[rand.Intn(arrlen)], vecs[rand.Intn(arrlen)])
	}
}

func BenchmarkDotProductSlow(b *testing.B) {
	arrlen := 1000
	vecs := make([][]BF16, arrlen)
	for i := 0; i < arrlen; i++ {
		vecs[i] = randomVec16(512)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		DotProductSlow(vecs[rand.Intn(arrlen)], vecs[rand.Intn(arrlen)])
	}
}
