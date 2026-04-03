// rmsnorm_bench — RMSNorm kernel correctness + timing
//
// Loads libbitnet_rmsnorm.so, verifies rmsnorm_f32 against scalar reference,
// then benchmarks ns/call.
//
// Build:
//   gcc -O3 -march=armv8.2-a+dotprod -o bench bench.c -ldl -lm -DNDEBUG
//
// Run:
//   ./bench ~/.olorin/lib/*/libbitnet_rmsnorm.so

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>

typedef void (*rmsnorm_fn)(const float *x, const float *weight, float *out, int n, float eps);

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

static void ref_rmsnorm(const float *x, const float *w, float *out, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * rms * w[i];
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <libbitnet_rmsnorm.so>\n", argv[0]);
        return 1;
    }

    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    rmsnorm_fn kernel_rmsnorm = (rmsnorm_fn)dlsym(lib, "rmsnorm_f32");
    if (!kernel_rmsnorm) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }
    printf("loaded kernel from %s\n\n", argv[1]);

    const int N = 3072;
    const float EPS = 1e-5f;
    const int ITERS = 100000;

    float *x      = malloc(N * sizeof(float));
    float *weight = malloc(N * sizeof(float));
    float *ref    = malloc(N * sizeof(float));
    float *out    = malloc(N * sizeof(float));

    // Fill with deterministic random data
    for (int i = 0; i < N; i++) x[i]      = rng_float();
    for (int i = 0; i < N; i++) weight[i]  = rng_float();

    // --- Correctness check ---
    ref_rmsnorm(x, weight, ref, N, EPS);
    kernel_rmsnorm(x, weight, out, N, EPS);

    float max_abs = 0.0f, max_rel = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabsf(ref[i] - out[i]);
        float rel = (fabsf(ref[i]) > 1e-10f) ? err / fabsf(ref[i]) : err;
        if (err > max_abs) max_abs = err;
        if (rel > max_rel) max_rel = rel;
    }
    printf("max abs error: %.2e\n", max_abs);
    printf("max rel error: %.2e\n", max_rel);
    printf("PASS:          %s\n\n", max_rel < 1e-5f ? "YES" : "NO");

    // --- Benchmark ---
    volatile float sink = 0;

    // warmup
    for (int i = 0; i < 1000; i++) {
        kernel_rmsnorm(x, weight, out, N, EPS);
        sink += out[0];
    }

    uint64_t t0 = now_ns();
    for (int i = 0; i < ITERS; i++) {
        kernel_rmsnorm(x, weight, out, N, EPS);
        sink += out[0];
    }
    uint64_t t1 = now_ns();
    double kernel_ns = (double)(t1 - t0) / ITERS;

    printf("=== Benchmark (n=%d, %d iters) ===\n", N, ITERS);
    printf("kernel: %.1f ns/call\n", kernel_ns);

    free(x); free(weight); free(ref); free(out);
    dlclose(lib);
    return 0;
}
