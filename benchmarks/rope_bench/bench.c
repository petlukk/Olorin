// rope_bench — RoPE (Rotary Position Encoding) kernel correctness + timing
//
// Loads librope.so, verifies apply_rope_f32 against scalar reference,
// then benchmarks ns/call.
//
// Build:
//   gcc -O3 -march=armv8.2-a+dotprod -o bench bench.c -ldl -lm -DNDEBUG
//
// Run:
//   ./bench ~/.olorin/lib/*/librope.so

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>

typedef void (*rope_fn)(const float *data, const float *freqs, float *out, int head_dim, int n_heads);

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

static void ref_rope(const float *data, const float *freqs, float *out, int hd, int nh) {
    for (int h = 0; h < nh; h++) {
        for (int i = 0; i < hd / 2; i++) {
            float x0 = data[h * hd + i];
            float x1 = data[h * hd + hd / 2 + i];
            float cos_f = cosf(freqs[i]);
            float sin_f = sinf(freqs[i]);
            out[h * hd + i]          = x0 * cos_f - x1 * sin_f;
            out[h * hd + hd / 2 + i] = x1 * cos_f + x0 * sin_f;
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
        fprintf(stderr, "usage: %s <librope.so>\n", argv[0]);
        return 1;
    }

    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    rope_fn kernel_rope = (rope_fn)dlsym(lib, "apply_rope_f32");
    if (!kernel_rope) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }
    printf("loaded kernel from %s\n\n", argv[1]);

    const int HEAD_DIM = 128;
    const int N_HEADS  = 32;
    const int TOTAL    = HEAD_DIM * N_HEADS;
    const int ITERS    = 100000;

    float *data  = malloc(TOTAL * sizeof(float));
    float *freqs = malloc((HEAD_DIM / 2) * sizeof(float));
    float *ref   = malloc(TOTAL * sizeof(float));
    float *out   = malloc(TOTAL * sizeof(float));

    // Fill with deterministic random data
    for (int i = 0; i < TOTAL; i++) data[i]  = rng_float();
    // Frequencies: theta_i = 1 / (10000^(2i/head_dim)), as in llama.cpp
    for (int i = 0; i < HEAD_DIM / 2; i++) {
        float theta = 1.0f / powf(10000.0f, (float)(2 * i) / (float)HEAD_DIM);
        freqs[i] = theta;
    }

    // --- Correctness check ---
    ref_rope(data, freqs, ref, HEAD_DIM, N_HEADS);
    kernel_rope(data, freqs, out, HEAD_DIM, N_HEADS);

    float max_abs = 0.0f, max_rel = 0.0f;
    for (int i = 0; i < TOTAL; i++) {
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
        kernel_rope(data, freqs, out, HEAD_DIM, N_HEADS);
        sink += out[0];
    }

    uint64_t t0 = now_ns();
    for (int i = 0; i < ITERS; i++) {
        kernel_rope(data, freqs, out, HEAD_DIM, N_HEADS);
        sink += out[0];
    }
    uint64_t t1 = now_ns();
    double kernel_ns = (double)(t1 - t0) / ITERS;

    printf("=== Benchmark (head_dim=%d, n_heads=%d, %d iters) ===\n", HEAD_DIM, N_HEADS, ITERS);
    printf("kernel: %.1f ns/call\n", kernel_ns);

    free(data); free(freqs); free(ref); free(out);
    dlclose(lib);
    return 0;
}
