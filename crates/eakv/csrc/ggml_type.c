/* ggml type adapter for eakv Q4_1 KV cache compression.
 *
 * Implements from_float, to_float, and vec_dot for the Q4_1_EAKV block
 * format, enabling transparent integration with llama.cpp's KV cache
 * via --cache-type-k q4_1_eakv --cache-type-v q4_1_eakv.
 *
 * All hot paths use AVX-512. Scalar fallbacks included for correctness.
 */
#include "eakv_ggml.h"
#include <math.h>
#include <string.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

/* ========================================================================
 * from_float — F32 → Q4_1_EAKV (called by ggml_set_rows on KV write)
 * ======================================================================== */

void eakv_quantize_row(const float *src, void *dst, int64_t k) {
    block_q4_1_eakv *blocks = (block_q4_1_eakv *)dst;
    const int nb = (int)(k / QK_EAKV);

    for (int i = 0; i < nb; i++) {
        const float *x = src + (int64_t)i * QK_EAKV;

        /* Find min/max */
        float min_val = x[0], max_val = x[0];

#ifdef __AVX512F__
        __m512 vmin = _mm512_loadu_ps(x);
        __m512 vmax = vmin;
        for (int j = 16; j < 64; j += 16) {
            __m512 v = _mm512_loadu_ps(x + j);
            vmin = _mm512_min_ps(vmin, v);
            vmax = _mm512_max_ps(vmax, v);
        }
        min_val = _mm512_reduce_min_ps(vmin);
        max_val = _mm512_reduce_max_ps(vmax);
#else
        for (int j = 1; j < 64; j++) {
            if (x[j] < min_val) min_val = x[j];
            if (x[j] > max_val) max_val = x[j];
        }
#endif

        const float range = max_val - min_val;
        const float d = range / 15.0f;
        const float id = range > 0.0f ? 15.0f / range : 0.0f;

        blocks[i].d = d;
        blocks[i].m = min_val;

        /* Split pack: lo nibble = value[k], hi nibble = value[k+32] */
#ifdef __AVX512F__
        const __m512 vid = _mm512_set1_ps(id);
        const __m512 vm  = _mm512_set1_ps(min_val);
        const __m512 v0  = _mm512_setzero_ps();
        const __m512 v15 = _mm512_set1_ps(15.0f);
        const __m512 vhalf = _mm512_set1_ps(0.5f);

        /* Process k=0..15: lo from x[k], hi from x[k+32] */
        for (int c = 0; c < 2; c++) {
            __m512 lo_f = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(x + c * 16), vm), vid);
            __m512 hi_f = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(x + c * 16 + 32), vm), vid);
            lo_f = _mm512_min_ps(v15, _mm512_max_ps(v0, _mm512_add_ps(lo_f, vhalf)));
            hi_f = _mm512_min_ps(v15, _mm512_max_ps(v0, _mm512_add_ps(hi_f, vhalf)));
            __m512i lo_i = _mm512_cvttps_epi32(lo_f);
            __m512i hi_i = _mm512_cvttps_epi32(hi_f);
            hi_i = _mm512_slli_epi32(hi_i, 4);
            __m512i packed = _mm512_or_si512(lo_i, hi_i);

            /* Compress 16 × i32 → 16 × u8 */
            __m128i p8 = _mm512_cvtepi32_epi8(packed);
            _mm_storeu_si128((__m128i *)(blocks[i].qs + c * 16), p8);
        }
#else
        for (int j = 0; j < 32; j++) {
            int lo = (int)(0.5f + (x[j] - min_val) * id);
            int hi = (int)(0.5f + (x[j + 32] - min_val) * id);
            if (lo < 0) lo = 0; if (lo > 15) lo = 15;
            if (hi < 0) hi = 0; if (hi > 15) hi = 15;
            blocks[i].qs[j] = (uint8_t)(lo | (hi << 4));
        }
#endif
    }
}

/* ========================================================================
 * to_float — Q4_1_EAKV → F32 (called by flash_attn_ext for V read)
 * ======================================================================== */

void eakv_dequantize_row(const void *src, float *dst, int64_t k) {
    const block_q4_1_eakv *blocks = (const block_q4_1_eakv *)src;
    const int nb = (int)(k / QK_EAKV);

    for (int i = 0; i < nb; i++) {
        const float d = blocks[i].d;
        const float m = blocks[i].m;
        float *out = dst + (int64_t)i * QK_EAKV;

#ifdef __AVX512F__
        const __m512 vd = _mm512_set1_ps(d);
        const __m512 vm = _mm512_set1_ps(m);

        for (int c = 0; c < 2; c++) {
            /* Load 16 packed bytes */
            __m128i raw = _mm_loadu_si128((const __m128i *)(blocks[i].qs + c * 16));

            /* Lo nibbles → values[c*16 .. c*16+15] */
            __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
            __m512i lo_i32 = _mm512_cvtepu8_epi32(lo);
            __m512 lo_f32 = _mm512_cvtepi32_ps(lo_i32);
            _mm512_storeu_ps(out + c * 16, _mm512_fmadd_ps(lo_f32, vd, vm));

            /* Hi nibbles → values[c*16+32 .. c*16+47] */
            __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
            __m512i hi_i32 = _mm512_cvtepu8_epi32(hi);
            __m512 hi_f32 = _mm512_cvtepi32_ps(hi_i32);
            _mm512_storeu_ps(out + c * 16 + 32, _mm512_fmadd_ps(hi_f32, vd, vm));
        }
#else
        for (int j = 0; j < 32; j++) {
            out[j]      = (blocks[i].qs[j] & 0x0F) * d + m;
            out[j + 32] = (blocks[i].qs[j] >> 4)    * d + m;
        }
#endif
    }
}

/* ========================================================================
 * vec_dot — Q4_1_EAKV · F32 (called by mul_mat, flash_attn_ext K path)
 *
 * Computes dot product of quantized KV row (x) with f32 query row (y).
 * Supports nrc=1 (single) and nrc=2 (batched, future).
 * ======================================================================== */

void eakv_vec_dot_q4_f32(int n, float * restrict s, size_t bs,
                          const void * restrict vx, size_t bx,
                          const void * restrict vy, size_t by, int nrc) {
    const int nb = n / QK_EAKV;

    /* nrc=1: single dot product (common path) */
    if (nrc == 1) {
        const block_q4_1_eakv *blocks = (const block_q4_1_eakv *)vx;
        const float *y = (const float *)vy;

#ifdef __AVX512F__
        __m512 acc = _mm512_setzero_ps();

        for (int i = 0; i < nb; i++) {
            const __m512 vd = _mm512_set1_ps(blocks[i].d);
            const __m512 vm = _mm512_set1_ps(blocks[i].m);

            for (int c = 0; c < 2; c++) {
                __m128i raw = _mm_loadu_si128((const __m128i *)(blocks[i].qs + c * 16));

                /* Lo nibbles × query[c*16 .. c*16+15] */
                __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
                __m512 lo_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo));
                __m512 val_lo = _mm512_fmadd_ps(lo_f, vd, vm);
                acc = _mm512_fmadd_ps(val_lo, _mm512_loadu_ps(y + c * 16), acc);

                /* Hi nibbles × query[c*16+32 .. c*16+47] */
                __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
                __m512 hi_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi));
                __m512 val_hi = _mm512_fmadd_ps(hi_f, vd, vm);
                acc = _mm512_fmadd_ps(val_hi, _mm512_loadu_ps(y + c * 16 + 32), acc);
            }

            y += QK_EAKV;
        }

        *s = _mm512_reduce_add_ps(acc);
#else
        float sum = 0.0f;
        const float *y_ptr = y;
        for (int i = 0; i < nb; i++) {
            const float d = blocks[i].d;
            const float m = blocks[i].m;
            for (int j = 0; j < 32; j++) {
                float v_lo = (blocks[i].qs[j] & 0x0F) * d + m;
                float v_hi = (blocks[i].qs[j] >> 4)    * d + m;
                sum += v_lo * y_ptr[j] + v_hi * y_ptr[j + 32];
            }
            y_ptr += QK_EAKV;
        }
        *s = sum;
#endif
        return;
    }

    /* nrc=2: batched — compute two dot products at once */
    for (int r = 0; r < nrc; r++) {
        const block_q4_1_eakv *blocks = (const block_q4_1_eakv *)((const char *)vx + r * bx);
        const float *y = (const float *)((const char *)vy + r * by);
        float *out = (float *)((char *)s + r * bs);

#ifdef __AVX512F__
        __m512 acc = _mm512_setzero_ps();

        for (int i = 0; i < nb; i++) {
            const __m512 vd = _mm512_set1_ps(blocks[i].d);
            const __m512 vm = _mm512_set1_ps(blocks[i].m);

            for (int c = 0; c < 2; c++) {
                __m128i raw = _mm_loadu_si128((const __m128i *)(blocks[i].qs + c * 16));

                __m128i lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
                __m512 lo_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(lo));
                __m512 val_lo = _mm512_fmadd_ps(lo_f, vd, vm);
                acc = _mm512_fmadd_ps(val_lo, _mm512_loadu_ps(y + c * 16), acc);

                __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
                __m512 hi_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(hi));
                __m512 val_hi = _mm512_fmadd_ps(hi_f, vd, vm);
                acc = _mm512_fmadd_ps(val_hi, _mm512_loadu_ps(y + c * 16 + 32), acc);
            }

            y += QK_EAKV;
        }

        *out = _mm512_reduce_add_ps(acc);
#else
        float sum = 0.0f;
        for (int i = 0; i < nb; i++) {
            const float d = blocks[i].d;
            const float m = blocks[i].m;
            for (int j = 0; j < 32; j++) {
                float v_lo = (blocks[i].qs[j] & 0x0F) * d + m;
                float v_hi = (blocks[i].qs[j] >> 4)    * d + m;
                sum += v_lo * y[j] + v_hi * y[j + 32];
            }
            y += QK_EAKV;
        }
        *out = sum;
#endif
    }
}
