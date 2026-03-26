/* eakv_ggml.h — ggml type integration for eakv Q4 KV cache.
 *
 * Defines the block format and kernel functions needed to register
 * GGML_TYPE_Q4_1_EAKV in llama.cpp's type system.
 *
 * Usage:  --cache-type-k q4_1_eakv --cache-type-v q4_1_eakv
 *
 * Block format (40 bytes, 64 elements):
 *   qs[32]  — split-packed nibbles: lo[k] = value[k], hi[k] = value[k+32]
 *   d       — scale (float32)
 *   m       — min/bias (float32)
 *
 * Dequantize: value[k] = (nibble[k] * d) + m
 */
#ifndef EAKV_GGML_H
#define EAKV_GGML_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QK_EAKV 64

typedef struct {
    uint8_t qs[32];  /* split packed: lo nibble = val[k], hi = val[k+32] */
    float   d;       /* scale */
    float   m;       /* min (bias) */
} block_q4_1_eakv;

_Static_assert(sizeof(block_q4_1_eakv) == 40, "block_q4_1_eakv must be 40 bytes");

/* ggml type_traits callbacks — match ggml_from_float_t / ggml_to_float_t signatures */
void eakv_quantize_row(const float * src, void * dst, int64_t k);
void eakv_dequantize_row(const void * src, float * dst, int64_t k);

/* ggml type_traits_cpu callback — matches ggml_vec_dot_t signature.
 * Computes dot product of Q4_1_EAKV (x) with F32 (y).
 * n:   number of elements
 * s:   output scalar(s)
 * bs:  stride between output scalars (0 for single)
 * vx:  Q4_1_EAKV data
 * bx:  stride between x rows (0 for single)
 * vy:  F32 data
 * by:  stride between y rows (0 for single)
 * nrc: number of rows to compute (1 or 2)
 */
void eakv_vec_dot_q4_f32(int n, float * s, size_t bs,
                          const void * vx, size_t bx,
                          const void * vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif

#endif /* EAKV_GGML_H */
