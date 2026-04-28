// test_q3k_kernel.c — verify q3k_dot_q8k against a pure-C reference port of
// llama.cpp's dequantize_row_q3_K (ggml-quants.c:1243).
//
// Build (x86 host):
//   PATH=$(realpath ../eacompute/target/release):$PATH \
//     ea kernels/q3k_dot.ea --lib                       # → ./q3k_dot.so
//   gcc -O2 -o test_q3k_kernel tests/test_q3k_kernel.c -ldl -lm
//   ./test_q3k_kernel ./q3k_dot.so
//
// Build (Pi 5 cross):
//   PATH=$(realpath ../eacompute/target/release):$PATH \
//     ea kernels/q3k_dot_arm.ea --target-triple=aarch64-unknown-linux-gnu \
//        --target=cortex-a76 --dotprod --lib            # → ./q3k_dot_arm.so
//   aarch64-linux-gnu-gcc -O2 -o test_q3k_kernel_arm tests/test_q3k_kernel.c -ldl -lm
//   scp q3k_dot_arm.so test_q3k_kernel_arm pi:~/
//   ssh pi 'LD_LIBRARY_PATH=~ ~/test_q3k_kernel_arm ~/q3k_dot_arm.so'
//
// Exit 0 iff every test passes.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

#define Q3K_BLOCK_BYTES 110
#define Q3K_BLOCK_SLACK 16     // f16_d on ARM reads 8 bytes from offset 108; pad final block.
#define BLOCK_ELEMS     256
#define SUBBLOCKS       16     // 16 sub-blocks of 16 elements per block

// Current ABI — matches export func in kernels/q3k_dot{,_arm}.ea.
typedef float (*q3k_dot_fn)(
    const int8_t*  q3,
    const int8_t*  q8,
    const int16_t* bsums,
    int32_t        n_blocks,
    const float*   q8_d,
    const float*   pow2);

// ───── f16 → f32 (IEEE-754 binary16 → binary32) ─────
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    uint32_t r;
    if (exp == 0) {
        if (frac == 0) { r = sign << 31; float f; memcpy(&f, &r, 4); return f; }
        int e = 0; uint32_t ff = frac;
        while (!(ff & 0x400)) { ff <<= 1; e--; }
        ff &= 0x3FF;
        r = (sign << 31) | ((uint32_t)(127 - 15 + 1 + e) << 23) | (ff << 13);
    } else if (exp == 31) {
        r = (sign << 31) | (0xFFu << 23) | (frac << 13);
    } else {
        r = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float f; memcpy(&f, &r, 4); return f;
}

// f32 → f16 (round-to-nearest-even, no NaN/Inf paths needed for tests).
static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = x & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        frac |= 0x800000;
        uint32_t shift = 14 - exp;
        uint32_t f16f = frac >> shift;
        if ((frac >> (shift - 1)) & 1) f16f++;
        return (uint16_t)(sign | f16f);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    uint32_t f16f = frac >> 13;
    if (frac & 0x1000) f16f++;
    return (uint16_t)(sign | (exp << 10) | f16f);
}

// ───── Pure-C dequantize_row_q3_K (verbatim port of ggml-quants.c:1243) ─────
static void dequant_q3k(const uint8_t* x, float* y, int nb) {
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {
        const uint8_t* block = x + i * Q3K_BLOCK_BYTES;
        const uint8_t* hm = block;             // hmask[0..31]
        const uint8_t* q  = block + 32;        // qs[0..63]
        uint16_t d_raw = (uint16_t)block[108] | ((uint16_t)block[109] << 8);
        float d_all = f16_to_f32(d_raw);

        // Scale unpack — exact transform from ggml-quants.c:1262
        memcpy(aux, block + 96, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = ( aux[0]       & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = ( aux[1]       & kmask2) | (((tmp >> 2) & kmask1) << 4);

        uint8_t m = 1;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl1 = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int q3 = (int)((q[l +  0] >> shift) & 3) - ((hm[l +  0] & m) ? 0 : 4);
                    *y++ = dl1 * (float)q3;
                }
                float dl2 = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int q3 = (int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                    *y++ = dl2 * (float)q3;
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

// Reference dot: dequant Q3_K → f32, dot against (q8_d * q8[i]).
static float ref_q3k_dot(
    const uint8_t* q3, const int8_t* q8,
    int n_blocks, const float* q8_d)
{
    float* dq = (float*)malloc((size_t)n_blocks * BLOCK_ELEMS * sizeof(float));
    dequant_q3k(q3, dq, n_blocks);
    double acc = 0.0;
    for (int blk = 0; blk < n_blocks; blk++) {
        double q8d = (double)q8_d[blk];
        for (int i = 0; i < BLOCK_ELEMS; i++) {
            acc += (double)dq[blk * BLOCK_ELEMS + i] * q8d * (double)q8[blk * BLOCK_ELEMS + i];
        }
    }
    free(dq);
    return (float)acc;
}

// ───── Test infrastructure ─────

// Pow2 table for the x86 kernel's f16 unpack (matches kernels/q4k_dot.ea pattern).
// pow2[exp] = 2^(exp - 15) for normalized f16 exponents; pow2[1] used for subnormals.
static void make_pow2(float* pow2) {
    for (int e = 0; e < 32; e++) {
        pow2[e] = ldexpf(1.0f, e - 15);
    }
}

// Compute Q8_K bsums[16] = sum-of-16 over qs, store as i16.
static void compute_bsums(const int8_t* q8, int16_t* bsums, int n_blocks) {
    for (int blk = 0; blk < n_blocks; blk++) {
        for (int g = 0; g < SUBBLOCKS; g++) {
            int s = 0;
            for (int l = 0; l < 16; l++) s += q8[blk * BLOCK_ELEMS + g * 16 + l];
            bsums[blk * SUBBLOCKS + g] = (int16_t)s;
        }
    }
}

static int report(const char* name, float ref, float kern, float abs_tol, float rel_tol) {
    float err = fabsf(ref - kern);
    float rel = fabsf(ref) > 1e-6f ? err / fabsf(ref) : 0.0f;
    int pass = (err <= abs_tol) || (rel <= rel_tol);
    printf("  %-28s  ref=%14.6f  kern=%14.6f  abs=%9.4g  rel=%9.4g  %s\n",
           name, ref, kern, err, rel, pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char** argv) {
    const char* lib_path = (argc > 1) ? argv[1] : "./q3k_dot.so";
    void* lib = dlopen(lib_path, RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen(%s): %s\n", lib_path, dlerror());
        return 2;
    }
    q3k_dot_fn kernel = (q3k_dot_fn)dlsym(lib, "q3k_dot_q8k");
    if (!kernel) {
        fprintf(stderr, "dlsym(q3k_dot_q8k): %s\n", dlerror());
        return 2;
    }
    printf("Loaded q3k_dot_q8k from %s\n\n", lib_path);

    float pow2[32]; make_pow2(pow2);
    int total_pass = 0, total_run = 0;

    // ===== Test 1: Trivial — d=1, all zero scales (centered at -32 → after -4 shift gives nonzero) =====
    // Actually with scales packed = 32 (i.e. signed 0), the dequant is 0·q3 = 0 → result 0.
    // Use d=1, scale-packed bytes = 0x40 (decodes to 64 → after-32 = 32 signed scale).
    // q3 values: hmask=0xFF means high bit always set → q3∈[0..3]; qs=0x00 → q3=0; +4 from hmask: q3=4 (after -4 shift = 0).
    // So we need a more meaningful trivial: hmask=0, qs=0 → q3=0|0 - 4 = -4 for every element.
    printf("=== Test 1: All q3 = -4 (hmask=0, qs=0), single block ===\n");
    {
        uint8_t q3blk[Q3K_BLOCK_BYTES + Q3K_BLOCK_SLACK];
        memset(q3blk, 0, sizeof(q3blk));
        // d = 1.0
        uint16_t d_h = f32_to_f16(1.0f);
        q3blk[108] = (uint8_t)(d_h & 0xFF);
        q3blk[109] = (uint8_t)(d_h >> 8);
        // scales: pack so every decoded scale byte = 33 (subtract 32 → +1).
        // The unpack assembles scales[i] from low-nibble + high-2-bits. Setting:
        //   bytes 0..7 = 0x21 (low nibble = 1, high nibble = 2) — these contribute the low 4 bits
        //   bytes 8..11 = 0x80 (high 2 bits per scale = 10b = 2)
        // Then scales[0..3] = (0x21 & 0x0F) | ((0x80 >> 0 & 0x03) << 4) = 0x01 | 0x00 = 1  ❌
        // Easier: manually solve. Goal: every scales[i] = 33. 33 in 6 bits = 0x21 = 0b100001.
        //   low 4 bits = 0001, high 2 bits = 10.
        // Layout for i in 0..3: aux[0] holds low4 of scales[0..3] in bytes 0..3.
        //   So bytes 96..99 each have low nibble = 0x1, but since aux[0] = (a0 & kmask2) | (low2<<4),
        //   the byte stored is just whatever — only its low nibble matters for scales[0..3] low4
        //   and its high nibble matters for scales[8..11] low4.
        // Net result, every "low 4 bits" position needs to be 1 and every "high 2 bits" position needs 2:
        //   bytes 96..103: each = 0x11 (low nibble 1 → scales[0..7] low4=1, high nibble 1 → scales[8..15] low4=1)
        //   bytes 104..107: each = 0xAA (= 0b10101010, four packed 2-bit values all = 10b = 2)
        for (int b = 0; b < 8; b++) q3blk[96 + b] = 0x11;
        for (int b = 0; b < 4; b++) q3blk[104 + b] = 0xAA;

        int8_t  q8[BLOCK_ELEMS];
        for (int i = 0; i < BLOCK_ELEMS; i++) q8[i] = 1;
        int16_t bsums[SUBBLOCKS]; compute_bsums(q8, bsums, 1);
        float   q8_d[1] = { 1.0f };

        // Expected: every q3 = -4, every q8 = 1, scale = 1, d = 1
        //   sum = 256 * (-4) * 1 = -1024
        //   result = 1.0 * 1.0 * (-4) * 1 ... per element, d_all * scale_signed * q3 * (q8_d * q8)
        //          = 1.0 * 1 * (-4) * 1 = -4 per element, ×256 = -1024
        float ref  = ref_q3k_dot(q3blk, q8, 1, q8_d);
        float kern = kernel((const int8_t*)q3blk, q8, bsums, 1, q8_d, pow2);
        total_pass += report("trivial (-4, all=1)", ref, kern, 1e-3f, 1e-5f); total_run++;
    }

    // ===== Test 2: Single block, random data =====
    printf("\n=== Test 2: Random single block (256 elements) ===\n");
    {
        srand(7);
        uint8_t q3blk[Q3K_BLOCK_BYTES + Q3K_BLOCK_SLACK];
        for (int i = 0; i < Q3K_BLOCK_BYTES; i++) q3blk[i] = (uint8_t)(rand() & 0xFF);
        for (int i = Q3K_BLOCK_BYTES; i < (int)sizeof(q3blk); i++) q3blk[i] = 0;
        // Random d in a sensible range: f16(0.01..0.5)
        float d_f = 0.01f + ((float)rand() / RAND_MAX) * 0.49f;
        uint16_t d_h = f32_to_f16(d_f);
        q3blk[108] = (uint8_t)(d_h & 0xFF);
        q3blk[109] = (uint8_t)(d_h >> 8);

        int8_t q8[BLOCK_ELEMS];
        for (int i = 0; i < BLOCK_ELEMS; i++) q8[i] = (int8_t)((rand() % 255) - 127);
        int16_t bsums[SUBBLOCKS]; compute_bsums(q8, bsums, 1);
        float   q8_d[1] = { 0.05f };

        float ref  = ref_q3k_dot(q3blk, q8, 1, q8_d);
        float kern = kernel((const int8_t*)q3blk, q8, bsums, 1, q8_d, pow2);
        total_pass += report("random 1 block", ref, kern, 1e-2f, 1e-4f); total_run++;
    }

    // ===== Test 3: Multiple blocks, random data =====
    printf("\n=== Test 3: Random multi-block (8 blocks = 2048 elements) ===\n");
    {
        srand(31);
        const int NB = 8;
        size_t bytes = (size_t)NB * Q3K_BLOCK_BYTES + Q3K_BLOCK_SLACK;
        uint8_t* q3 = (uint8_t*)calloc(bytes, 1);
        for (int blk = 0; blk < NB; blk++) {
            uint8_t* b = q3 + blk * Q3K_BLOCK_BYTES;
            for (int i = 0; i < Q3K_BLOCK_BYTES - 2; i++) b[i] = (uint8_t)(rand() & 0xFF);
            float d_f = 0.005f + ((float)rand() / RAND_MAX) * 0.2f;
            if ((rand() & 1) == 0) d_f = -d_f;     // mix sign of d
            uint16_t d_h = f32_to_f16(d_f);
            b[108] = (uint8_t)(d_h & 0xFF);
            b[109] = (uint8_t)(d_h >> 8);
        }
        int8_t* q8   = (int8_t*)malloc(NB * BLOCK_ELEMS);
        for (int i = 0; i < NB * BLOCK_ELEMS; i++) q8[i] = (int8_t)((rand() % 255) - 127);
        int16_t* bsums = (int16_t*)malloc(NB * SUBBLOCKS * sizeof(int16_t));
        compute_bsums(q8, bsums, NB);
        float* q8_d = (float*)malloc(NB * sizeof(float));
        for (int b = 0; b < NB; b++) q8_d[b] = 0.01f + ((float)rand() / RAND_MAX) * 0.04f;

        float ref  = ref_q3k_dot(q3, q8, NB, q8_d);
        float kern = kernel((const int8_t*)q3, q8, bsums, NB, q8_d, pow2);
        total_pass += report("random 8 blocks", ref, kern, 5e-2f, 1e-4f); total_run++;

        free(q3); free(q8); free(bsums); free(q8_d);
    }

    // ===== Test 4: Magnitude / overflow check =====
    // Stress the i32 accumulator: max |scale|=32, max |q3|=4, max |q8|=127, 256 elements per block.
    // Per block: |Σscale·q3·q8| ≤ 32·4·127·256 = 4,161,536 — well within i32. Verify kernel handles it.
    printf("\n=== Test 4: Max-magnitude stress (1 block) ===\n");
    {
        uint8_t q3blk[Q3K_BLOCK_BYTES + Q3K_BLOCK_SLACK];
        memset(q3blk, 0xFF, Q3K_BLOCK_BYTES);    // hmask=0xFF (all high bits set), qs=0xFF
        memset(q3blk + Q3K_BLOCK_BYTES, 0, Q3K_BLOCK_SLACK);
        // Scales: pack to maximum magnitude. scales[i] should encode 0 or 63 (→ -32 or +31 signed).
        // Set every scale byte to 63: bytes 96..103 = 0xFF (low nibble F, high nibble F),
        // bytes 104..107 = 0xFF (all 2-bit groups = 11). Then scales[i] = 0xF | (0x3 << 4) = 0x3F = 63.
        for (int b = 96; b < 108; b++) q3blk[b] = 0xFF;
        // d: a moderate value
        uint16_t d_h = f32_to_f16(0.125f);
        q3blk[108] = (uint8_t)(d_h & 0xFF);
        q3blk[109] = (uint8_t)(d_h >> 8);

        int8_t q8[BLOCK_ELEMS];
        for (int i = 0; i < BLOCK_ELEMS; i++) q8[i] = (i & 1) ? 127 : -127;
        int16_t bsums[SUBBLOCKS]; compute_bsums(q8, bsums, 1);
        float   q8_d[1] = { 0.5f };

        float ref  = ref_q3k_dot(q3blk, q8, 1, q8_d);
        float kern = kernel((const int8_t*)q3blk, q8, bsums, 1, q8_d, pow2);
        total_pass += report("max-magnitude", ref, kern, 1e-1f, 1e-4f); total_run++;
    }

    // ===== Test 5: Sub-block index sanity — only sub-block 5 nonzero =====
    // If the sub-block ordering (which scale applies to which 16 elements) is wrong,
    // a sparse single-sub-block input will compute a different value than the reference.
    printf("\n=== Test 5: Sparse sub-block (only sub-block 5 active) ===\n");
    {
        uint8_t q3blk[Q3K_BLOCK_BYTES + Q3K_BLOCK_SLACK];
        memset(q3blk, 0, sizeof(q3blk));
        // Per ggml-quants.c:1270 loop ordering, sub-block 5 is hit at:
        //   n=0 (first 128 elem), j=2 (third inner iteration), m=4, scales[5]
        //   The inner loop's second 16-element pass writes y[80..95] (l=0..15 with offset +16 in q[],
        //   but element index = n + j*32 + 16 + l = 0 + 64 + 16 + l = 80..95).
        // So active region: elements 80..95, which read q[16..31] >> 4 & 3 and hm[16..31] & 4.
        // We'll set q[24] (one byte covering elements 84,85,86,87 at shifts 0/2/4/6) = 0x10
        //   → bits at shift 4 = 0x01 = 1, others = 0.
        // And hm[24] = 0xFF (so bit-2 = 1, hmask "active" → -0 not -4).
        // Result: element y[84+1=85] from "(q[24] >> 4) & 3 - 0" = 1 - 0 = 1.
        //
        // Easier: set EVERY byte in the second-half (l=16..31) qs slot to a specific pattern.
        // qs offset for n=0 is q3blk[32..63]; the j=2 inner pass uses q[16..31] = q3blk[48..63] >> 4.
        // Set q3blk[48..63] = 0x10 → (>> 4) & 3 = 1 for element l=0..15 (16 elements).
        // Set hm[16..31] = 0xFF → bit 2 (m=4) is set → +0 correction → q3 = 1 - 0 = 1.
        for (int i = 48; i < 64; i++) q3blk[i] = 0x10;
        for (int i = 16; i < 32; i++) q3blk[i] = 0xFF;
        // scales[5] = 33 (so signed = +1). Other scales = 32 (signed = 0) so they contribute nothing.
        // 33 = 0b100001 → low4=0001, high2=10. 32 = 0b100000 → low4=0000, high2=10.
        // Layout: scales[5] → aux[1] byte 1 → byte 97 low nibble.
        //         scales[0..3] → aux[0] byte 0..3 → byte 96..99 low nibbles.
        //         scales[4..7] → aux[1] bytes 0..3 → byte 100..103 low nibbles.
        //   → byte 100 (scales[4]) low4 = 0, byte 101 (scales[5]) low4 = 1, others low4 = 0.
        // High-2 bits all need to be 10 (= 2): byte 104 controls scales[0..3] high2,
        //   byte 105 controls scales[4..7] high2, byte 106 controls scales[8..11] high2,
        //   byte 107 controls scales[12..15] high2. Each = 0xAA.
        for (int b = 96; b <= 103; b++) q3blk[b] = 0x00;
        q3blk[101] = 0x01;                                    // scales[5] low nibble = 1
        for (int b = 104; b <= 107; b++) q3blk[b] = 0xAA;     // all scales' high2 = 10

        uint16_t d_h = f32_to_f16(2.0f);
        q3blk[108] = (uint8_t)(d_h & 0xFF);
        q3blk[109] = (uint8_t)(d_h >> 8);

        int8_t q8[BLOCK_ELEMS];
        for (int i = 0; i < BLOCK_ELEMS; i++) q8[i] = 0;
        // Activate q8 in the same range as our nonzero q3 (elements 80..95 of sub-block 5).
        for (int i = 80; i < 96; i++) q8[i] = 3;
        int16_t bsums[SUBBLOCKS]; compute_bsums(q8, bsums, 1);
        float   q8_d[1] = { 1.0f };

        // Expected hand computation:
        //   d=2, scale[5]=+1, all 16 active q3s = 1, all 16 active q8s = 3, q8_d=1
        //   contribution = d * scale * Σ(q3·q8) * q8_d = 2 * 1 * (16*1*3) * 1 = 96.
        // Other sub-blocks: scale=0 → contribute nothing.
        float ref  = ref_q3k_dot(q3blk, q8, 1, q8_d);
        float kern = kernel((const int8_t*)q3blk, q8, bsums, 1, q8_d, pow2);
        total_pass += report("sparse sub-block 5", ref, kern, 1e-3f, 1e-5f); total_run++;
        printf("    (expected ≈ 96.0)\n");
    }

    // ===== Test 6: Sign isolation — two blocks identical except d sign, expect 0 =====
    printf("\n=== Test 6: Sign cancellation (block 0 d=+0.5, block 1 d=-0.5) ===\n");
    {
        srand(99);
        size_t bytes = 2 * Q3K_BLOCK_BYTES + Q3K_BLOCK_SLACK;
        uint8_t* q3 = (uint8_t*)calloc(bytes, 1);
        // Same byte pattern for both blocks (except d).
        for (int i = 0; i < Q3K_BLOCK_BYTES - 2; i++) {
            uint8_t v = (uint8_t)(rand() & 0xFF);
            q3[i] = v;
            q3[Q3K_BLOCK_BYTES + i] = v;
        }
        uint16_t d_pos = f32_to_f16(+0.5f);
        uint16_t d_neg = f32_to_f16(-0.5f);
        q3[108] = (uint8_t)(d_pos & 0xFF); q3[109] = (uint8_t)(d_pos >> 8);
        q3[Q3K_BLOCK_BYTES + 108] = (uint8_t)(d_neg & 0xFF);
        q3[Q3K_BLOCK_BYTES + 109] = (uint8_t)(d_neg >> 8);

        int8_t q8[2 * BLOCK_ELEMS];
        for (int i = 0; i < BLOCK_ELEMS; i++) {
            int8_t v = (int8_t)((rand() % 255) - 127);
            q8[i] = v;
            q8[BLOCK_ELEMS + i] = v;
        }
        int16_t bsums[2 * SUBBLOCKS]; compute_bsums(q8, bsums, 2);
        float q8_d[2] = { 1.0f, 1.0f };

        // Per-block individual probes
        float ref_b0  = ref_q3k_dot(q3, q8, 1, q8_d);
        float kern_b0 = kernel((const int8_t*)q3, q8, bsums, 1, q8_d, pow2);
        float ref_b1  = ref_q3k_dot(q3 + Q3K_BLOCK_BYTES, q8 + BLOCK_ELEMS, 1, q8_d + 1);
        float kern_b1 = kernel((const int8_t*)(q3 + Q3K_BLOCK_BYTES),
                               q8 + BLOCK_ELEMS, bsums + SUBBLOCKS, 1, q8_d + 1, pow2);
        printf("    Block 0 (d=+0.5):  ref=%14.6f  kern=%14.6f  diff=%9.4g\n",
               ref_b0, kern_b0, fabsf(ref_b0 - kern_b0));
        printf("    Block 1 (d=-0.5):  ref=%14.6f  kern=%14.6f  diff=%9.4g\n",
               ref_b1, kern_b1, fabsf(ref_b1 - kern_b1));
        printf("    Block 0 + Block 1 (expected 0):  ref=%9.4g  kern=%9.4g\n",
               ref_b0 + ref_b1, kern_b0 + kern_b1);

        float ref  = ref_q3k_dot(q3, q8, 2, q8_d);
        float kern = kernel((const int8_t*)q3, q8, bsums, 2, q8_d, pow2);
        total_pass += report("sign cancel (2 blocks)", ref, kern, 1e-3f, 1e-4f); total_run++;

        free(q3);
    }

    dlclose(lib);
    printf("\n%d/%d tests passed.\n", total_pass, total_run);
    return (total_pass == total_run) ? 0 : 1;
}
