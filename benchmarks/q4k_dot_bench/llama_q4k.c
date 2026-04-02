// Extracted from llama.cpp ggml/src/ggml-cpu/arch/arm/quants.c
// Pure NEON DOTPROD path for Q4_K × Q8_K (nrc=1, no SVE, no I8MM)
// License: MIT (llama.cpp)

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#define QK_K 256
#define K_SCALE_SIZE 12
#define GGML_RESTRICT __restrict__
#define UNUSED(x) (void)(x)

// f16 → f32 via ARM NEON
static inline float fp16_to_fp32(uint16_t h) {
    __fp16 tmp;
    memcpy(&tmp, &h, sizeof(tmp));
    return (float)tmp;
}

// block_q4_K: 144 bytes
typedef struct {
    uint16_t d;          // offset 0-1: f16 scale
    uint16_t dmin;       // offset 2-3: f16 min scale
    uint8_t scales[12];  // offset 4-15: packed 6-bit scales+mins
    uint8_t qs[128];     // offset 16-143: 4-bit nibbles
} block_q4_K;

// block_q8_K: 292 bytes
typedef struct {
    float d;             // offset 0-3: scale
    int8_t qs[256];      // offset 4-259: quantized values
    int16_t bsums[16];   // offset 260-291: block sums
} block_q8_K;

// ggml compat macros
#define ggml_vdotq_s32(acc, a, b) vdotq_s32(acc, a, b)

typedef struct { int8x16_t val[2]; } ggml_int8x16x2_t;
typedef struct { uint8x16_t val[2]; } ggml_uint8x16x2_t;

static inline ggml_uint8x16x2_t ggml_vld1q_u8_x2(const uint8_t *p) {
    ggml_uint8x16x2_t r;
    r.val[0] = vld1q_u8(p);
    r.val[1] = vld1q_u8(p + 16);
    return r;
}

static inline ggml_int8x16x2_t ggml_vld1q_s8_x2(const int8_t *p) {
    ggml_int8x16x2_t r;
    r.val[0] = vld1q_s8(p);
    r.val[1] = vld1q_s8(p + 16);
    return r;
}

void ggml_vec_dot_q4_K_q8_K(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int32x4_t mzero = vdupq_n_s32(0);

    ggml_int8x16x2_t q4bytes;
    ggml_int8x16x2_t q8bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * fp16_to_fp32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, K_SCALE_SIZE);

        uint32x2_t mins8 = { 0 };
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins))
        );
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            const ggml_uint8x16x2_t q4bits = ggml_vld1q_u8_x2(q4); q4 += 32;

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q4bits.val[1], m4b));

            const int32x4_t p1 = ggml_vdotq_s32(
                ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]),
                q4bytes.val[1], q8bytes.val[1]
            );
            sumi1 += vaddvq_s32(p1) * scales[2*j+0];

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p2 = ggml_vdotq_s32(
                ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]),
                q4bytes.val[1], q8bytes.val[1]
            );
            sumi2 += vaddvq_s32(p2) * scales[2*j+1];
        }

        sumf += d * (sumi1 + sumi2);
    }

    *s = sumf;
}
