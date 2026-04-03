// attn_f16_bench — f16 KV-cache attention dot + vsum kernel correctness test
//
// Loads libattn_f16.so, runs attn_dot_f16 and attn_vsum_f16 against scalar
// references, checks rel error < 1e-3 (f16 precision), then benchmarks.
//
// Build:
//   gcc -O3 -march=armv8.2-a+dotprod -o bench bench.c -ldl -lm -DNDEBUG
//
// Run:
//   ./bench ~/.olorin/lib/*/libattn_f16.so

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>

typedef void (*attn_dot_fn)(const float *query, const uint16_t *k_cache, float *scores, int seq_len, int head_dim);
typedef void (*attn_vsum_fn)(const float *weights, const uint16_t *v_cache, float *out, int seq_len, int head_dim);

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

// f16 → f32
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) return 0.0f;
    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
    float r; memcpy(&r, &bits, 4); return r;
}

// f32 → f16
static uint16_t f32_to_f16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000;
    int32_t  exp  = ((b >> 23) & 0xFF) - 127;
    uint32_t frac = b & 0x7FFFFF;
    if (exp > 15)  return sign | 0x7C00;
    if (exp < -14) return sign;
    return sign | ((exp + 15) << 10) | (frac >> 13);
}

static void ref_attn_dot(const float *q, const uint16_t *k, float *scores, int seq_len, int hd) {
    for (int t = 0; t < seq_len; t++) {
        float dot = 0;
        for (int d = 0; d < hd; d++)
            dot += q[d] * f16_to_f32(k[t * hd + d]);
        scores[t] = dot;
    }
}

static void ref_attn_vsum(const float *w, const uint16_t *v, float *out, int seq_len, int hd) {
    for (int d = 0; d < hd; d++) out[d] = 0;
    for (int t = 0; t < seq_len; t++)
        for (int d = 0; d < hd; d++)
            out[d] += w[t] * f16_to_f32(v[t * hd + d]);
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <libattn_f16.so>\n", argv[0]);
        return 1;
    }

    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    attn_dot_fn  kernel_dot  = (attn_dot_fn) dlsym(lib, "attn_dot_f16");
    attn_vsum_fn kernel_vsum = (attn_vsum_fn)dlsym(lib, "attn_vsum_f16");
    if (!kernel_dot)  { fprintf(stderr, "dlsym attn_dot_f16: %s\n",  dlerror()); return 1; }
    if (!kernel_vsum) { fprintf(stderr, "dlsym attn_vsum_f16: %s\n", dlerror()); return 1; }
    printf("loaded kernel from %s\n\n", argv[1]);

    const int SEQ = 512;
    const int HD  = 128;
    const int ITERS = 10000;

    float    *query   = malloc(HD  * sizeof(float));
    uint16_t *k_cache = malloc(SEQ * HD * sizeof(uint16_t));
    uint16_t *v_cache = malloc(SEQ * HD * sizeof(uint16_t));
    float    *weights = malloc(SEQ * sizeof(float));
    float    *ref_scores  = malloc(SEQ * sizeof(float));
    float    *kern_scores = malloc(SEQ * sizeof(float));
    float    *ref_out  = malloc(HD * sizeof(float));
    float    *kern_out = malloc(HD * sizeof(float));

    // Generate test data
    for (int i = 0; i < HD; i++)  query[i] = randf();
    for (int i = 0; i < SEQ * HD; i++) {
        k_cache[i] = f32_to_f16(randf());
        v_cache[i] = f32_to_f16(randf());
    }
    // weights: random positive, normalised (simulate post-softmax)
    float wsum = 0;
    for (int i = 0; i < SEQ; i++) { weights[i] = (float)(xorshift32() & 0xFF) + 1.0f; wsum += weights[i]; }
    for (int i = 0; i < SEQ; i++) weights[i] /= wsum;

    // --- Correctness: attn_dot ---
    ref_attn_dot(query, k_cache, ref_scores, SEQ, HD);
    kernel_dot(query, k_cache, kern_scores, SEQ, HD);

    float max_dot_err = 0, max_dot_ref = 0;
    for (int i = 0; i < SEQ; i++) {
        float e = fabsf(ref_scores[i] - kern_scores[i]);
        float r = fabsf(ref_scores[i]);
        if (e > max_dot_err) max_dot_err = e;
        if (r > max_dot_ref) max_dot_ref = r;
    }
    float dot_rel = (max_dot_ref > 1e-10f) ? max_dot_err / max_dot_ref : max_dot_err;
    printf("attn_dot_f16 (seq=%d, hd=%d):\n", SEQ, HD);
    printf("  max abs error: %.2e\n", max_dot_err);
    printf("  max rel error: %.2e\n", dot_rel);
    printf("  PASS:          %s\n\n", dot_rel < 1e-3f ? "YES" : "NO");

    // --- Correctness: attn_vsum ---
    ref_attn_vsum(weights, v_cache, ref_out, SEQ, HD);
    kernel_vsum(weights, v_cache, kern_out, SEQ, HD);

    float max_vs_err = 0, max_vs_ref = 0;
    for (int i = 0; i < HD; i++) {
        float e = fabsf(ref_out[i] - kern_out[i]);
        float r = fabsf(ref_out[i]);
        if (e > max_vs_err) max_vs_err = e;
        if (r > max_vs_ref) max_vs_ref = r;
    }
    float vs_rel = (max_vs_ref > 1e-10f) ? max_vs_err / max_vs_ref : max_vs_err;
    printf("attn_vsum_f16 (seq=%d, hd=%d):\n", SEQ, HD);
    printf("  max abs error: %.2e\n", max_vs_err);
    printf("  max rel error: %.2e\n", vs_rel);
    printf("  PASS:          %s\n\n", vs_rel < 1e-3f ? "YES" : "NO");

    // --- Benchmark ---
    volatile float sink = 0;

    // warmup
    for (int i = 0; i < 100; i++) {
        kernel_dot(query, k_cache, kern_scores, SEQ, HD);
        kernel_vsum(weights, v_cache, kern_out, SEQ, HD);
        sink += kern_scores[0] + kern_out[0];
    }

    uint64_t t0 = now_ns();
    for (int i = 0; i < ITERS; i++)
        kernel_dot(query, k_cache, kern_scores, SEQ, HD);
    uint64_t t1 = now_ns();
    double dot_ns = (double)(t1 - t0) / ITERS;

    t0 = now_ns();
    for (int i = 0; i < ITERS; i++)
        kernel_vsum(weights, v_cache, kern_out, SEQ, HD);
    t1 = now_ns();
    double vsum_ns = (double)(t1 - t0) / ITERS;

    printf("=== Benchmark (seq=%d, hd=%d, %d iters) ===\n", SEQ, HD, ITERS);
    printf("attn_dot_f16:  %.1f ns/call\n", dot_ns);
    printf("attn_vsum_f16: %.1f ns/call\n", vsum_ns);
    (void)sink;

    free(query); free(k_cache); free(v_cache); free(weights);
    free(ref_scores); free(kern_scores); free(ref_out); free(kern_out);
    dlclose(lib);
    return 0;
}
