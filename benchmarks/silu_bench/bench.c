// silu_bench — silu_mul_f32 kernel correctness test
//
// Loads libsilu_mul.so, runs silu_mul_f32 against scalar reference,
// checks rel error < 1e-4, then benchmarks.
//
// Build:
//   gcc -O3 -march=armv8.2-a+dotprod -o bench bench.c -ldl -lm -DNDEBUG
//
// Run:
//   ./bench ~/.olorin/lib/*/libsilu_mul.so

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>

typedef void (*silu_mul_fn)(const float *gate, const float *up, float *out, int n);

static uint32_t rng_state = 42;
static uint32_t xorshift32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float randf(void) {
    return ((float)(xorshift32() & 0xFFFF) / 65535.0f) * 4.0f - 2.0f;
}

static void ref_silu_mul(const float *gate, const float *up, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float g   = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        out[i]    = g * sig * up[i];
    }
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <libsilu_mul.so>\n", argv[0]);
        return 1;
    }

    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    silu_mul_fn kernel_silu = (silu_mul_fn)dlsym(lib, "silu_mul_f32");
    if (!kernel_silu) { fprintf(stderr, "dlsym silu_mul_f32: %s\n", dlerror()); return 1; }
    printf("loaded kernel from %s\n\n", argv[1]);

    const int N = 8192;
    const int ITERS = 50000;

    float *gate    = malloc(N * sizeof(float));
    float *up      = malloc(N * sizeof(float));
    float *ref_out = malloc(N * sizeof(float));
    float *kern_out = malloc(N * sizeof(float));

    // Generate test data
    for (int i = 0; i < N; i++) { gate[i] = randf(); up[i] = randf(); }

    // Run scalar reference
    ref_silu_mul(gate, up, ref_out, N);

    // Run kernel
    kernel_silu(gate, up, kern_out, N);

    // Compare
    float max_err = 0, max_ref = 0;
    for (int i = 0; i < N; i++) {
        float e = fabsf(ref_out[i] - kern_out[i]);
        float r = fabsf(ref_out[i]);
        if (e > max_err) max_err = e;
        if (r > max_ref) max_ref = r;
    }
    float rel = (max_ref > 1e-10f) ? max_err / max_ref : max_err;

    printf("silu_mul_f32 (n=%d):\n", N);
    printf("  max abs error: %.2e\n", max_err);
    printf("  max rel error: %.2e\n", rel);
    printf("  PASS:          %s\n\n", rel < 1e-4f ? "YES" : "NO");

    // --- Benchmark ---
    volatile float sink = 0;

    // warmup
    for (int i = 0; i < 500; i++) {
        kernel_silu(gate, up, kern_out, N);
        sink += kern_out[0];
    }

    uint64_t t0 = now_ns();
    for (int i = 0; i < ITERS; i++)
        kernel_silu(gate, up, kern_out, N);
    uint64_t t1 = now_ns();
    double silu_ns = (double)(t1 - t0) / ITERS;

    printf("=== Benchmark (n=%d, %d iters) ===\n", N, ITERS);
    printf("silu_mul_f32: %.1f ns/call\n", silu_ns);
    (void)sink;

    free(gate); free(up); free(ref_out); free(kern_out);
    dlclose(lib);
    return 0;
}
