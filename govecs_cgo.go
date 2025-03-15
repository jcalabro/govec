//go:build amd64

package govec

/*
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <string.h>

// Alternative implementation using DPFP16 instructions if available
float dot_product_avx(const float16_t* a, const float16_t* b, size_t n) {
    size_t aligned_n = (n / 32) * 32;

    __m512 sum_vec = _mm512_setzero_ps();

    for (size_t i = 0; i < aligned_n; i += 32) {
        __m512h a_vec = _mm512_loadu_ph(&a[i]);
        __m512h b_vec = _mm512_loadu_ph(&b[i]);

        sum_vec = _mm512_dpfp16_ps(a_vec, b_vec, sum_vec);
    }

    float result = _mm512_reduce_add_ps(sum_vec);

    // Handle remaining elements
    for (size_t i = aligned_n; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}
*/
import "C"
import "unsafe"

func DotProductFast(a, b []uint16) float32 {
	aPtr := (*C.float16_t)(unsafe.Pointer(&a[0]))
	bPtr := (*C.float16_t)(unsafe.Pointer(&b[0]))

	result := C.dot_product_avx(aPtr, bPtr, C.int(len(a)))
	return float32(result)

}
