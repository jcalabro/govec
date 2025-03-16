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
*/
import "C"
import "unsafe"

func DotProductFast(a, b []BF16) float32 {
	aPtr := (*C.uint16_t)(unsafe.Pointer(&a[0]))
	bPtr := (*C.uint16_t)(unsafe.Pointer(&b[0]))

	result := C.dot_product_avx_bf16(aPtr, bPtr, C.size_t(len(a)))
	return float32(result)
}
