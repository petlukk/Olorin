// q4k_dot_bench — isolate Q4K dot product: Olorin (Eä) vs llama.cpp (GCC)
//
// Generates identical Q4K + Q8K data, calls both kernels, compares
// throughput and correctness.
//
// Build on Pi 5:
//   gcc -O3 -march=armv8.2-a+dotprod -o bench bench.c llama_q4k.c \
//       -ldl -lm -DNDEBUG
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

// --- Q4K block layout (144 bytes) ---
// offset 0-1:  d     (f16, we store as f32 separately)
// offset 2-3:  dmin  (f16, we store as f32 separately)
// offset 4-15: scales (12 bytes, packed 6-bit)
// offset 16-143: qs  (128 bytes, 4-bit nibbles)
#define QK_K 256
#define K_SCALE_SIZE 12
#define Q4K_BLOCK_SIZE 144

// --- Q8K block layout (292 bytes) ---
// offset 0-3:   d (float)
// offset 4-259: qs (256 x int8)
// offset 260-291: bsums (16 x int16)
#define Q8K_BLOCK_SIZE 292

// Olorin kernel signature (from q4k_dot_arm.ea)
typedef float (*olorin_q4k_dot_fn)(
    const int8_t *q4,
    const int8_t *q8,
    const int32_t *bsums,
    int32_t n_blocks,
    const float *d_arr,
    const float *dmin_arr
);

// llama.cpp kernel (linked from llama_q4k.c)
extern void ggml_vec_dot_q4_K_q8_K(
    int n, float *s, size_t bs,
    const void *vx, size_t bx,
    const void *vy, size_t by,
    int nrc
);

// Simple pseudo-random for reproducibility
static uint32_t rng_state = 42;
static uint32_t xorshift32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static int8_t rand_i8(void) { return (int8_t)(xorshift32() & 0xFF); }
static uint8_t rand_u8(void) { return (uint8_t)(xorshift32() & 0xFF); }
static float rand_f32(void) { return (float)(xorshift32() % 1000) / 1000.0f; }

// Generate n_blocks of Q4K data
static void gen_q4k(int n_blocks, uint8_t *raw, float *d_arr, float *dmin_arr) {
    for (int b = 0; b < n_blocks; b++) {
        uint8_t *block = raw + b * Q4K_BLOCK_SIZE;
        // d, dmin as f16 placeholder (bytes 0-3)
        float d = rand_f32() * 0.01f;
        float dmin = rand_f32() * 0.001f;
        d_arr[b] = d;
        dmin_arr[b] = dmin;
        // Store f16 placeholders (not used by Olorin kernel directly)
        memset(block, 0, 4);
        // scales (bytes 4-15)
        for (int i = 4; i < 16; i++) block[i] = rand_u8() & 0x3F;
        // nibbles (bytes 16-143)
        for (int i = 16; i < 144; i++) block[i] = rand_u8();
    }
}

// Generate n_blocks of Q8K data
static void gen_q8k(int n_blocks, uint8_t *raw, int32_t *bsums_out) {
    for (int b = 0; b < n_blocks; b++) {
        uint8_t *block = raw + b * Q8K_BLOCK_SIZE;
        // d (float, bytes 0-3)
        float d = rand_f32() * 0.1f;
        memcpy(block, &d, 4);
        // qs (bytes 4-259)
        int16_t bsums[16] = {0};
        for (int i = 0; i < 256; i++) {
            int8_t v = rand_i8();
            block[4 + i] = (uint8_t)v;
            bsums[i / 16] += v;
        }
        // bsums (bytes 260-291)
        memcpy(block + 260, bsums, 32);
        // Also store bsums as i32 for Olorin kernel
        for (int j = 0; j < 16; j++) {
            bsums_out[b * 16 + j] = bsums[j];
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
        fprintf(stderr, "usage: %s <libq4k_dot.so>\n", argv[0]);
        return 1;
    }

    // Load Olorin kernel
    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }
    olorin_q4k_dot_fn olorin_dot = (olorin_q4k_dot_fn)dlsym(lib, "q4k_dot_q8k");
    if (!olorin_dot) {
        fprintf(stderr, "dlsym: %s\n", dlerror());
        return 1;
    }
    printf("loaded Olorin kernel from %s\n", argv[1]);

    // Setup: typical Llama 3B hidden_dim=3072 → n_blocks = 3072/256 = 12
    const int n_blocks = 12;
    const int n_elem = n_blocks * QK_K;
    const int ITERS = 100000;

    // Allocate
    uint8_t *q4_raw = calloc(n_blocks, Q4K_BLOCK_SIZE);
    uint8_t *q8_raw = calloc(n_blocks, Q8K_BLOCK_SIZE);
    float *d_arr = calloc(n_blocks, sizeof(float));
    float *dmin_arr = calloc(n_blocks, sizeof(float));
    int32_t *bsums = calloc(n_blocks * 16, sizeof(int32_t));

    // Generate data
    gen_q4k(n_blocks, q4_raw, d_arr, dmin_arr);
    gen_q8k(n_blocks, q8_raw, bsums);

    // --- Warmup ---
    float olorin_result = 0, llama_result = 0;
    for (int i = 0; i < 1000; i++) {
        olorin_result = olorin_dot(
            (int8_t*)q4_raw, (int8_t*)(q8_raw + 4), bsums,
            n_blocks, d_arr, dmin_arr
        );
    }
    for (int i = 0; i < 1000; i++) {
        ggml_vec_dot_q4_K_q8_K(
            n_elem, &llama_result, 0,
            q4_raw, 0, q8_raw, 0, 1
        );
    }
    printf("olorin result: %.6f\n", olorin_result);
    printf("llama  result: %.6f\n", llama_result);

    // --- Benchmark Olorin ---
    uint64_t t0 = now_ns();
    volatile float sink = 0;
    for (int i = 0; i < ITERS; i++) {
        sink = olorin_dot(
            (int8_t*)q4_raw, (int8_t*)(q8_raw + 4), bsums,
            n_blocks, d_arr, dmin_arr
        );
    }
    uint64_t t1 = now_ns();
    double olorin_ns = (double)(t1 - t0) / ITERS;

    // --- Benchmark llama.cpp ---
    uint64_t t2 = now_ns();
    for (int i = 0; i < ITERS; i++) {
        ggml_vec_dot_q4_K_q8_K(
            n_elem, (float*)&sink, 0,
            q4_raw, 0, q8_raw, 0, 1
        );
    }
    uint64_t t3 = now_ns();
    double llama_ns = (double)(t3 - t2) / ITERS;

    printf("\n=== Results (n_blocks=%d, %d iters) ===\n", n_blocks, ITERS);
    printf("olorin: %.1f ns/call\n", olorin_ns);
    printf("llama:  %.1f ns/call\n", llama_ns);
    printf("ratio:  %.2fx\n", olorin_ns / llama_ns);

    // --- perf counters hint ---
    printf("\nFor cache analysis:\n");
    printf("  perf stat -e cycles,instructions,cache-misses,cache-references ./bench %s\n", argv[1]);

    free(q4_raw); free(q8_raw); free(d_arr); free(dmin_arr); free(bsums);
    dlclose(lib);
    return 0;
}
