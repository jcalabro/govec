package govec

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/x448/float16"
)

func randomVecBF16(n int) []BF16 {
	out := make([]BF16, n)

	for i := 0; i < n; i++ {
		out[i] = ToBF16(rand.Float32())
	}

	return out
}

func randomVecFP16(n int) []float16.Float16 {
	out := make([]float16.Float16, n)

	for i := 0; i < n; i++ {
		out[i] = float16.Fromfloat32(rand.Float32())
	}

	return out
}

func bf16VecTofp16(a []BF16) []float16.Float16 {
	out := make([]float16.Float16, len(a))
	for i := 0; i < len(a); i++ {
		out[i] = float16.Fromfloat32(a[i].Float32())
	}

	return out
}

func TestDotProduct(t *testing.T) {
	a := randomVecBF16(512)
	b := randomVecBF16(512)

	ca := bf16VecTofp16(a)
	cb := bf16VecTofp16(b)

	dpa := DotProductSlow(a, b)
	//dpb := DotProductFast(a, b)
	dpb := 0
	dpc := DotProductFastFP16(ca, cb)
	dpd := DotProductSlowFP16(ca, cb)

	fmt.Println(dpa, dpb, dpc, dpd)
}

/*
func BenchmarkDotProductBF16(b *testing.B) {
	arrlen := 1000
	vecs := make([][]BF16, arrlen)
	for i := 0; i < arrlen; i++ {
		vecs[i] = randomVecBF16(512)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		DotProductFast(vecs[rand.Intn(arrlen)], vecs[rand.Intn(arrlen)])
	}
}
*/

func BenchmarkDotProductSlow(b *testing.B) {
	arrlen := 1000
	vecs := make([][]BF16, arrlen)
	for i := 0; i < arrlen; i++ {
		vecs[i] = randomVecBF16(512)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		DotProductSlow(vecs[rand.Intn(arrlen)], vecs[rand.Intn(arrlen)])
	}
}

func BenchmarkDotProductFastFP16(b *testing.B) {
	arrlen := 1000
	vecs := make([][]float16.Float16, arrlen)
	for i := 0; i < arrlen; i++ {
		vecs[i] = randomVecFP16(512)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		DotProductFastFP16(vecs[rand.Intn(arrlen)], vecs[rand.Intn(arrlen)])
	}
}
