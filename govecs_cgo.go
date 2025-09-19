//go:build amd64

package govec

/*
#cgo CFLAGS: -mavx512f -mavx512bw -mavx512vl -mavx512bf16

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
#include <string.h>

float dot_product_avx_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
    size_t aligned_n = (n / 32) * 32;

    __m512 sum_vec = _mm512_setzero_ps();

    for (size_t i = 0; i < aligned_n; i += 32) {
        __m512bh aval = *((__m512bh*)&a[i]);
        __m512bh bval = *((__m512bh*)&b[i]);

        sum_vec = _mm512_dpbf16_ps(sum_vec, aval, bval);
    }

    float result = _mm512_reduce_add_ps(sum_vec);

    for (size_t i = aligned_n; i < n; i++) {
        // Simple scalar multiplication for remaining elements
        result += (float)a[i] * (float)b[i];
    }

    return result;
}

float dot_product_avx_fp16_conv(const uint16_t* a, const uint16_t* b, size_t n) {
    size_t aligned_n = (n / 16) * 16;

    __m512 sum_vec = _mm512_setzero_ps();

    float sum = 0;

    for (size_t i = 0; i < aligned_n; i += 16) {
        __m256i aval = *((__m256i*)&a[i]);
        __m256i bval = *((__m256i*)&b[i]);

	__m512 upconva = _mm512_cvtph_ps(aval);
	__m512 upconvb = _mm512_cvtph_ps(bval);

	__m512 mulval = _mm512_mul_ps(upconva, upconvb);

	sum += _mm512_reduce_add_ps(mulval);
    }

    for (size_t i = aligned_n; i < n; i++) {
        sum += (float)a[i] * (float)b[i];
    }

    return sum;
}
*/
import "C"
import (
	"unsafe"

	"github.com/x448/float16"
)

func DotProductFast(a, b []BF16) float32 {
	aPtr := (*C.uint16_t)(unsafe.Pointer(&a[0]))
	bPtr := (*C.uint16_t)(unsafe.Pointer(&b[0]))

	result := C.dot_product_avx_bf16(aPtr, bPtr, C.size_t(len(a)))
	return float32(result)
}

func DotProductFastFP16(a, b []float16.Float16) float32 {
	aPtr := (*C.uint16_t)(unsafe.Pointer(&a[0]))
	bPtr := (*C.uint16_t)(unsafe.Pointer(&b[0]))

	result := C.dot_product_avx_fp16_conv(aPtr, bPtr, C.size_t(len(a)))
	return float32(result)
}

func DotProductFastFP16_rawBuffer(a, b []byte) float32 {
	aPtr := (*C.uint16_t)(unsafe.Pointer(&a[0]))
	bPtr := (*C.uint16_t)(unsafe.Pointer(&b[0]))

	result := C.dot_product_avx_fp16_conv(aPtr, bPtr, C.size_t(len(a)/2))
	return float32(result)
}
