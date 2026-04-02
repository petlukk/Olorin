// q4k_dot_bench — isolate Q4K dot product: inline-scales kernel correctness test
//
// Generates identical Q4K + Q8K data with proper f16 headers, calls kernel
// with inline d/dmin reading (pow2 table), verifies against scalar reference.
//
// Build:
//   gcc -O2 -o bench bench.c -ldl -lm
//
// Run:
//   ./bench <path-to-libq4k_dot.so>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>

#define QK_K 256
#define Q4K_BLOCK_SIZE 144

// New kernel signature: inline d/dmin from block headers via pow2 table
typedef float (*q4k_dot_fn)(
    const uint8_t *q4,
    const int8_t *q8,
    const int32_t *bsums,
    int32_t n_blocks,
    const float *q8_d,
    const float *pow2
);

// Simple pseudo-random for reproducibility
static uint32_t rng_state = 42;
static uint32_t xorshift32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// f32 → f16 (round to nearest)
static uint16_t f32_to_f16(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000;
    int32_t exp = ((b >> 23) & 0xFF) - 127;
    uint32_t frac = b & 0x7FFFFF;
    if (exp > 15) return sign | 0x7C00;     // overflow → inf
    if (exp < -14) return sign;              // underflow → 0
    return sign | ((exp + 15) << 10) | (frac >> 13);
}

// f16 → f32 (reference, bitcast method)
static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((h >> 15) & 1);
    uint32_t exp = ((h >> 10) & 0x1F);
    uint32_t frac = (h & 0x3FF);
    if (exp == 0) return 0.0f;
    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
    float result;
    memcpy(&result, &bits, 4);
    return result;
}

// Build pow2 table: pow2[i] = 2^(i-15) for i=1..30
static void build_pow2(float pow2[32]) {
    memset(pow2, 0, 32 * sizeof(float));
    for (int i = 1; i <= 30; i++) {
        uint32_t bits = (uint32_t)(i + 112) << 23;
        memcpy(&pow2[i], &bits, 4);
    }
}

// Unpack 6-bit scale
static int get_scale(const uint8_t *p, int sp, int j) {
    if (j < 2) return p[sp + j*2] & 63;
    return (p[sp + j*2 + 4] & 15) | ((p[sp + j*2 - 4] >> 6) << 4);
}
static int get_scale_hi(const uint8_t *p, int sp, int j) {
    if (j < 2) return p[sp + j*2 + 1] & 63;
    return (p[sp + j*2 + 5] & 15) | ((p[sp + j*2 - 3] >> 6) << 4);
}

// Scalar reference Q4K × Q8K dot product (reads d/dmin from block header)
static float ref_q4k_dot(
    const uint8_t *q4, const int8_t *q8, const int32_t *bsums,
    int n_blocks, const float *q8_d
) {
    float result = 0.0f;
    for (int blk = 0; blk < n_blocks; blk++) {
        int bp = blk * 144;
        int sp = bp + 4;
        int nib = bp + 16;
        int q8_off = blk * 256;
        int bs = blk * 16;

        // Read d/dmin from block header (f16 at bytes 0-3)
        uint16_t d_raw = q4[bp] | ((uint16_t)q4[bp+1] << 8);
        uint16_t dm_raw = q4[bp+2] | ((uint16_t)q4[bp+3] << 8);
        float d = f16_to_f32(d_raw) * q8_d[blk];
        float dm = f16_to_f32(dm_raw) * q8_d[blk];

        int sumi = 0;
        for (int j = 0; j < 4; j++) {
            int dot_lo = 0, dot_hi = 0;
            for (int k = 0; k < 32; k++) {
                dot_lo += (q4[nib + j*32 + k] & 0xF) * q8[q8_off + j*64 + k];
                dot_hi += (q4[nib + j*32 + k] >> 4) * q8[q8_off + j*64 + 32 + k];
            }
            sumi += dot_lo * get_scale(q4, sp, j) + dot_hi * get_scale_hi(q4, sp, j);
        }

        // Mins correction
        int summs = 0;
        for (int k = 0; k < 4; k++)
            summs += (q4[sp + 4 + k] & 63) * (bsums[bs + k*2] + bsums[bs + k*2 + 1]);
        for (int k = 0; k < 4; k++)
            summs += ((q4[sp + 8 + k] >> 4) | ((q4[sp + 4 + k] >> 6) << 4))
                     * (bsums[bs + 8 + k*2] + bsums[bs + 8 + k*2 + 1]);

        result += d * (float)sumi - dm * (float)summs;
    }
    return result;
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <libq4k_dot.so>\n", argv[0]);
        return 1;
    }

    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    q4k_dot_fn kernel_dot = (q4k_dot_fn)dlsym(lib, "q4k_dot_q8k");
    if (!kernel_dot) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }
    printf("loaded kernel from %s\n\n", argv[1]);

    float pow2[32];
    build_pow2(pow2);

    const int n_blocks = 12;
    const int ITERS = 100000;

    uint8_t *q4_raw = calloc(n_blocks, Q4K_BLOCK_SIZE);
    int8_t *q8_qs = calloc(n_blocks, 256);
    float *q8_d = calloc(n_blocks, sizeof(float));
    int32_t *bsums = calloc(n_blocks * 16, sizeof(int32_t));

    // Generate Q4K blocks WITH proper f16 headers
    for (int b = 0; b < n_blocks; b++) {
        uint8_t *block = q4_raw + b * Q4K_BLOCK_SIZE;
        float d_f32 = 0.001f + (float)(xorshift32() % 100) * 0.0001f;
        float dm_f32 = 0.0001f + (float)(xorshift32() % 50) * 0.00001f;
        uint16_t d_f16 = f32_to_f16(d_f32);
        uint16_t dm_f16 = f32_to_f16(dm_f32);
        // Store f16 in block header (little-endian)
        block[0] = d_f16 & 0xFF;
        block[1] = d_f16 >> 8;
        block[2] = dm_f16 & 0xFF;
        block[3] = dm_f16 >> 8;
        // scales (bytes 4-15)
        for (int i = 4; i < 16; i++) block[i] = xorshift32() & 0x3F;
        // nibbles (bytes 16-143)
        for (int i = 16; i < 144; i++) block[i] = xorshift32() & 0xFF;
    }

    // Generate Q8K data
    for (int b = 0; b < n_blocks; b++) {
        q8_d[b] = 0.01f + (float)(xorshift32() % 100) * 0.001f;
        int16_t bs16[16] = {0};
        for (int i = 0; i < 256; i++) {
            int8_t v = (int8_t)(xorshift32() & 0xFF);
            q8_qs[b * 256 + i] = v;
            bs16[i / 16] += v;
        }
        for (int j = 0; j < 16; j++)
            bsums[b * 16 + j] = bs16[j];
    }

    // --- Correctness check ---
    float ref = ref_q4k_dot(q4_raw, q8_qs, bsums, n_blocks, q8_d);
    float kernel = kernel_dot(q4_raw, q8_qs, bsums, n_blocks, q8_d, pow2);

    printf("reference result: %.6f\n", ref);
    printf("kernel result:    %.6f\n", kernel);
    float err = fabsf(ref - kernel);
    float rel = (fabsf(ref) > 1e-10f) ? err / fabsf(ref) : err;
    printf("abs error:        %.2e\n", err);
    printf("rel error:        %.2e\n", rel);
    printf("PASS:             %s\n\n", rel < 1e-4f ? "YES" : "NO");

    // --- Benchmark ---
    volatile float sink = 0;

    // warmup
    for (int i = 0; i < 1000; i++)
        sink = kernel_dot(q4_raw, q8_qs, bsums, n_blocks, q8_d, pow2);

    uint64_t t0 = now_ns();
    for (int i = 0; i < ITERS; i++)
        sink = kernel_dot(q4_raw, q8_qs, bsums, n_blocks, q8_d, pow2);
    uint64_t t1 = now_ns();
    double kernel_ns = (double)(t1 - t0) / ITERS;

    printf("=== Benchmark (n_blocks=%d, %d iters) ===\n", n_blocks, ITERS);
    printf("kernel: %.1f ns/call\n", kernel_ns);

    free(q4_raw); free(q8_qs); free(q8_d); free(bsums);
    dlclose(lib);
    return 0;
}
