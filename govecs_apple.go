//go:build arm64

package govec

/*
#cgo CFLAGS: -O3 -march=native

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>
#include <arm_fp16.h>

float fallback_accum(const float16x8_t v) {
	float result = 0;
	result += (float)(*(((float16_t*)(&v)) + 0));
	result += (float)(*(((float16_t*)(&v)) + 1));
	result += (float)(*(((float16_t*)(&v)) + 2));
	result += (float)(*(((float16_t*)(&v)) + 3));
	result += (float)(*(((float16_t*)(&v)) + 4));
	result += (float)(*(((float16_t*)(&v)) + 5));
	result += (float)(*(((float16_t*)(&v)) + 6));
	result += (float)(*(((float16_t*)(&v)) + 7));

	return result;
}

// Function to compute the dot product of two float16 arrays using ARM NEON intrinsics
__attribute__((target("fullfp16")))
float dot_product_arm_fp16(const float16_t* a, const float16_t* b, size_t n) {
    // Ensure n is a multiple of 8 (NEON can process 8 float16 values at once)
    size_t aligned_n = (n / 8) * 8;

    // Initialize accumulators for partial sums
    float16x8_t mulvec = vdupq_n_f16(0.0f);
    float16x8_t accum = vdupq_n_f16(0.0f);

    // Process in chunks of 8 float16 values (2 sets of 4 values)
    for (size_t i = 0; i < aligned_n; i += 8) {
        // Load 8 float16 values from arrays a and b
        float16x8_t a_vec = vld1q_f16(&a[i]);
        float16x8_t b_vec = vld1q_f16(&b[i]);

	mulvec = vmulq_f16(a_vec, b_vec);

	accum = vaddq_f16(accum, mulvec);
    }

    // Combine partial results
    //float16_t sum = vaddvq_f16(accum);
    float sum = fallback_accum(accum);

    // Handle remaining elements
    for (size_t i = aligned_n; i < n; i++) {
        sum += (float)a[i] * (float)b[i];
    }

    return sum;
}

// Function to compute the dot product of two float16 arrays using ARM NEON intrinsics
float dot_product_arm_fp16_voidcast(const void* a, const void* b, size_t n) {
	return dot_product_arm_fp16(a, b, n);
}
*/
import "C"
import (
	"unsafe"

	"github.com/x448/float16"
)

func DotProductFastFP16(a, b []float16.Float16) float32 {
	aPtr := unsafe.Pointer(&a[0])
	bPtr := unsafe.Pointer(&b[0])

	result := C.dot_product_arm_fp16_voidcast(aPtr, bPtr, C.size_t(len(a)))
	return float32(result)

}
