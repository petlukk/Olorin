// quant_q8k_bench — Q8_K quantization kernel correctness + timing
//
// Loads libq4k_quant.so, verifies quant_f32_q8k against scalar reference,
// then benchmarks ns/call.
//
// Build:
//   gcc -O3 -march=armv8.2-a+dotprod -o bench bench.c -ldl -lm -DNDEBUG
//
// Run:
//   ./bench ~/.olorin/lib/*/libq4k_quant.so

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>

typedef void (*quant_fn)(const float *src, int8_t *dst_qs, float *dst_d, int32_t *dst_bsums, int n);

// Simple pseudo-random for reproducibility
static uint32_t rng_state = 42;
static uint32_t xorshift32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float rng_float(void) {
    // [-1, 1)
    return ((float)(xorshift32() & 0xFFFFFF) / (float)0x800000) - 1.0f;
}

// Q8_K scalar reference: per-256-block quantization
static void ref_quant_q8k(const float *src, int8_t *qs, float *d, int32_t *bsums, int n) {
    int nb = n / 256;
    for (int b = 0; b < nb; b++) {
        const float *block = src + b * 256;
        float amax = 0.0f;
        for (int i = 0; i < 256; i++) {
            float v = fabsf(block[i]);
            if (v > amax) amax = v;
        }
        d[b] = amax / 127.0f;
        float id = (amax > 0) ? 127.0f / amax : 0.0f;
        int8_t *qb = qs + b * 256;
        for (int i = 0; i < 256; i++) {
            int v = (int)roundf(block[i] * id);
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            qb[i] = (int8_t)v;
        }
        // Block sums: sum of 16 consecutive q values
        int32_t *bs = bsums + b * 16;
        for (int j = 0; j < 16; j++) {
            int32_t s = 0;
            for (int k = 0; k < 16; k++) s += qb[j * 16 + k];
            bs[j] = s;
        }
    }
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <libq4k_quant.so>\n", argv[0]);
        return 1;
    }

    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    quant_fn kernel_quant = (quant_fn)dlsym(lib, "quant_f32_q8k");
    if (!kernel_quant) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }
    printf("loaded kernel from %s\n\n", argv[1]);

    const int N = 3072;
    const int NB = N / 256;
    const int ITERS = 50000;

    float   *src       = malloc(N * sizeof(float));
    int8_t  *ref_qs    = malloc(N * sizeof(int8_t));
    float   *ref_d     = malloc(NB * sizeof(float));
    int32_t *ref_bsums = malloc(NB * 16 * sizeof(int32_t));
    int8_t  *out_qs    = malloc(N * sizeof(int8_t));
    float   *out_d     = malloc(NB * sizeof(float));
    int32_t *out_bsums = malloc(NB * 16 * sizeof(int32_t));

    // Fill with deterministic random data
    for (int i = 0; i < N; i++) src[i] = rng_float();

    // --- Correctness check ---
    ref_quant_q8k(src, ref_qs, ref_d, ref_bsums, N);
    kernel_quant(src, out_qs, out_d, out_bsums, N);

    int qs_mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (ref_qs[i] != out_qs[i]) qs_mismatches++;
    }
    int d_mismatches = 0;
    float max_d_err = 0.0f;
    for (int b = 0; b < NB; b++) {
        float err = fabsf(ref_d[b] - out_d[b]);
        if (err > max_d_err) max_d_err = err;
        if (err > 1e-10f) d_mismatches++;
    }
    int bsum_mismatches = 0;
    for (int i = 0; i < NB * 16; i++) {
        if (ref_bsums[i] != out_bsums[i]) bsum_mismatches++;
    }

    printf("qs mismatches:    %d / %d\n", qs_mismatches, N);
    printf("scale mismatches: %d / %d  (max err: %.2e)\n", d_mismatches, NB, max_d_err);
    printf("bsum mismatches:  %d / %d\n", bsum_mismatches, NB * 16);
    int pass = (qs_mismatches == 0) && (bsum_mismatches == 0) && (max_d_err < 1e-10f);
    printf("PASS:             %s\n\n", pass ? "YES" : "NO");

    // --- Benchmark ---
    volatile int8_t sink = 0;

    // warmup
    for (int i = 0; i < 500; i++) {
        kernel_quant(src, out_qs, out_d, out_bsums, N);
        sink += out_qs[0];
    }

    uint64_t t0 = now_ns();
    for (int i = 0; i < ITERS; i++) {
        kernel_quant(src, out_qs, out_d, out_bsums, N);
        sink += out_qs[0];
    }
    uint64_t t1 = now_ns();
    double kernel_ns = (double)(t1 - t0) / ITERS;

    printf("=== Benchmark (n=%d, %d iters) ===\n", N, ITERS);
    printf("kernel: %.1f ns/call\n", kernel_ns);

    free(src); free(ref_qs); free(ref_d); free(ref_bsums);
    free(out_qs); free(out_d); free(out_bsums);
    dlclose(lib);
    return 0;
}
