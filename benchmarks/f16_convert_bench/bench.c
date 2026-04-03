// f16_convert_bench — f32↔f16 conversion kernel correctness test
//
// Loads libf16_convert.so, runs f32_to_f16 and f16_to_f32 against scalar
// references, round-trip test, checks rel error < 1e-3, then benchmarks.
//
// Build:
//   gcc -O3 -march=armv8.2-a+dotprod -o bench bench.c -ldl -lm -DNDEBUG
//
// Run:
//   ./bench ~/.olorin/lib/*/libf16_convert.so

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>

typedef void (*f32_to_f16_fn)(const float *src, uint16_t *dst, int n);
typedef void (*f16_to_f32_fn)(const uint16_t *src, float *dst, int n);

static uint32_t rng_state = 42;
static uint32_t xorshift32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float randf(void) {
    return ((float)(xorshift32() & 0xFFFF) / 65535.0f) * 2.0f - 1.0f;
}

// Scalar f32 → f16
static uint16_t ref_f32_to_f16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000;
    int32_t  exp  = ((b >> 23) & 0xFF) - 127;
    uint32_t frac = b & 0x7FFFFF;
    if (exp > 15)  return sign | 0x7C00;
    if (exp < -14) return sign;
    return sign | ((exp + 15) << 10) | (frac >> 13);
}

// Scalar f16 → f32
static float ref_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) return 0.0f;
    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
    float r; memcpy(&r, &bits, 4); return r;
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <libf16_convert.so>\n", argv[0]);
        return 1;
    }

    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    f32_to_f16_fn kernel_enc = (f32_to_f16_fn)dlsym(lib, "f32_to_f16");
    f16_to_f32_fn kernel_dec = (f16_to_f32_fn)dlsym(lib, "f16_to_f32");
    if (!kernel_enc) { fprintf(stderr, "dlsym f32_to_f16: %s\n", dlerror()); return 1; }
    if (!kernel_dec) { fprintf(stderr, "dlsym f16_to_f32: %s\n", dlerror()); return 1; }
    printf("loaded kernel from %s\n\n", argv[1]);

    const int N = 3072;
    const int ITERS = 100000;

    float    *src_f32      = malloc(N * sizeof(float));
    uint16_t *ref_f16      = malloc(N * sizeof(uint16_t));
    uint16_t *kern_f16     = malloc(N * sizeof(uint16_t));
    float    *ref_rt_f32   = malloc(N * sizeof(float));
    float    *kern_rt_f32  = malloc(N * sizeof(float));

    // Generate test data
    for (int i = 0; i < N; i++) src_f32[i] = randf();

    // Scalar reference: f32 → f16
    for (int i = 0; i < N; i++) ref_f16[i] = ref_f32_to_f16(src_f32[i]);

    // --- Correctness: f32_to_f16 ---
    kernel_enc(src_f32, kern_f16, N);

    int enc_mismatches = 0;
    for (int i = 0; i < N; i++)
        if (kern_f16[i] != ref_f16[i]) enc_mismatches++;
    printf("f32_to_f16 (n=%d):\n", N);
    printf("  bit-exact mismatches: %d / %d\n", enc_mismatches, N);
    printf("  PASS:                 %s\n\n", enc_mismatches == 0 ? "YES" : "NO");

    // --- Correctness: f16_to_f32 ---
    // scalar ref: decode ref_f16 back
    for (int i = 0; i < N; i++) ref_rt_f32[i] = ref_f16_to_f32(ref_f16[i]);
    kernel_dec(ref_f16, kern_rt_f32, N);

    float max_dec_err = 0, max_dec_ref = 0;
    for (int i = 0; i < N; i++) {
        float e = fabsf(ref_rt_f32[i] - kern_rt_f32[i]);
        float r = fabsf(ref_rt_f32[i]);
        if (e > max_dec_err) max_dec_err = e;
        if (r > max_dec_ref) max_dec_ref = r;
    }
    float dec_rel = (max_dec_ref > 1e-10f) ? max_dec_err / max_dec_ref : max_dec_err;
    printf("f16_to_f32 (n=%d):\n", N);
    printf("  max abs error: %.2e\n", max_dec_err);
    printf("  max rel error: %.2e\n", dec_rel);
    printf("  PASS:          %s\n\n", dec_rel < 1e-3f ? "YES" : "NO");

    // --- Round-trip: f32 → f16 → f32 ---
    kernel_dec(kern_f16, kern_rt_f32, N);

    float max_rt_err = 0, max_rt_ref = 0;
    for (int i = 0; i < N; i++) {
        // expected: scalar round-trip
        float expected = ref_f16_to_f32(ref_f16[i]);
        float e = fabsf(expected - kern_rt_f32[i]);
        float r = fabsf(expected);
        if (e > max_rt_err) max_rt_err = e;
        if (r > max_rt_ref) max_rt_ref = r;
    }
    float rt_rel = (max_rt_ref > 1e-10f) ? max_rt_err / max_rt_ref : max_rt_err;
    printf("round-trip f32->f16->f32 (n=%d):\n", N);
    printf("  max abs error: %.2e\n", max_rt_err);
    printf("  max rel error: %.2e\n", rt_rel);
    printf("  PASS:          %s\n\n", rt_rel < 1e-3f ? "YES" : "NO");

    // --- Benchmark ---
    volatile float sink = 0;

    // warmup
    for (int i = 0; i < 100; i++) {
        kernel_enc(src_f32, kern_f16, N);
        kernel_dec(kern_f16, kern_rt_f32, N);
        sink += kern_rt_f32[0];
    }

    uint64_t t0 = now_ns();
    for (int i = 0; i < ITERS; i++)
        kernel_enc(src_f32, kern_f16, N);
    uint64_t t1 = now_ns();
    double enc_ns = (double)(t1 - t0) / ITERS;

    t0 = now_ns();
    for (int i = 0; i < ITERS; i++)
        kernel_dec(kern_f16, kern_rt_f32, N);
    t1 = now_ns();
    double dec_ns = (double)(t1 - t0) / ITERS;

    printf("=== Benchmark (n=%d, %d iters) ===\n", N, ITERS);
    printf("f32_to_f16: %.1f ns/call\n", enc_ns);
    printf("f16_to_f32: %.1f ns/call\n", dec_ns);
    (void)sink;

    free(src_f32); free(ref_f16); free(kern_f16);
    free(ref_rt_f32); free(kern_rt_f32);
    dlclose(lib);
    return 0;
}
