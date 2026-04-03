// softmax_bench — softmax_f32 kernel correctness test
//
// Loads libsoftmax.so, runs softmax_f32 against scalar reference,
// checks sum ≈ 1.0 and max element error < 1e-3, then benchmarks.
//
// Build:
//   gcc -O3 -march=armv8.2-a+dotprod -o bench bench.c -ldl -lm -DNDEBUG
//
// Run:
//   ./bench ~/.olorin/lib/*/libsoftmax.so

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>

typedef void (*softmax_fn)(float *data, int n, float scale);

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

static void ref_softmax(float *data, int n, float scale) {
    float max_val = -1e30f;
    for (int i = 0; i < n; i++) { data[i] *= scale; if (data[i] > max_val) max_val = data[i]; }
    float sum = 0;
    for (int i = 0; i < n; i++) { data[i] = expf(data[i] - max_val); sum += data[i]; }
    for (int i = 0; i < n; i++) data[i] /= sum;
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <libsoftmax.so>\n", argv[0]);
        return 1;
    }

    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    softmax_fn kernel_sm = (softmax_fn)dlsym(lib, "softmax_f32");
    if (!kernel_sm) { fprintf(stderr, "dlsym softmax_f32: %s\n", dlerror()); return 1; }
    printf("loaded kernel from %s\n\n", argv[1]);

    const int N = 512;
    const float SCALE = 0.08838835f;  // 1/sqrt(128)
    const int ITERS = 100000;

    float *base_data  = malloc(N * sizeof(float));
    float *ref_data   = malloc(N * sizeof(float));
    float *kern_data  = malloc(N * sizeof(float));

    // Generate base test data (raw attention logits)
    for (int i = 0; i < N; i++) base_data[i] = randf();

    // Run scalar reference
    memcpy(ref_data, base_data, N * sizeof(float));
    ref_softmax(ref_data, N, SCALE);

    // Run kernel
    memcpy(kern_data, base_data, N * sizeof(float));
    kernel_sm(kern_data, N, SCALE);

    // Check sum ≈ 1.0
    float ref_sum = 0, kern_sum = 0;
    for (int i = 0; i < N; i++) { ref_sum += ref_data[i]; kern_sum += kern_data[i]; }

    // Check max element error
    float max_err = 0, max_ref = 0;
    for (int i = 0; i < N; i++) {
        float e = fabsf(ref_data[i] - kern_data[i]);
        float r = fabsf(ref_data[i]);
        if (e > max_err) max_err = e;
        if (r > max_ref) max_ref = r;
    }
    float rel = (max_ref > 1e-10f) ? max_err / max_ref : max_err;

    printf("softmax_f32 (n=%d, scale=%.6f):\n", N, SCALE);
    printf("  ref  sum:      %.8f\n", ref_sum);
    printf("  kern sum:      %.8f\n", kern_sum);
    printf("  sum error:     %.2e\n", fabsf(kern_sum - 1.0f));
    printf("  max abs error: %.2e\n", max_err);
    printf("  max rel error: %.2e\n", rel);
    int pass = (fabsf(kern_sum - 1.0f) < 1e-5f) && (rel < 1e-3f);
    printf("  PASS:          %s\n\n", pass ? "YES" : "NO");

    // --- Benchmark ---
    volatile float sink = 0;

    // warmup
    for (int i = 0; i < 1000; i++) {
        memcpy(kern_data, base_data, N * sizeof(float));
        kernel_sm(kern_data, N, SCALE);
        sink += kern_data[0];
    }

    uint64_t t0 = now_ns();
    for (int i = 0; i < ITERS; i++) {
        memcpy(kern_data, base_data, N * sizeof(float));
        kernel_sm(kern_data, N, SCALE);
    }
    uint64_t t1 = now_ns();
    double sm_ns = (double)(t1 - t0) / ITERS;

    printf("=== Benchmark (n=%d, %d iters) ===\n", N, ITERS);
    printf("softmax_f32: %.1f ns/call  (includes memcpy setup)\n", sm_ns);
    (void)sink;

    free(base_data); free(ref_data); free(kern_data);
    dlclose(lib);
    return 0;
}
