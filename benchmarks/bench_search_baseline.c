// Benchmark: vault search baseline (256-dim byte-histogram cosine similarity)
//
// Measures:
//   1. batch_cosine latency at various vector counts (64, 256, 1024, 4096)
//   2. top_k latency
//   3. Precision: planted known-best vector must rank #1
//
// Build:
//   gcc -O2 -o bench_search_baseline bench_search_baseline.c -ldl -lm
// Run:
//   ./bench_search_baseline <path-to-libsearch.so>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <stdint.h>

#define DIM 256
#define WARMUP 100
#define ITERS 1000

typedef void (*batch_cosine_fn)(
    const float *query, float query_norm,
    const float *vecs, int dim, int n_vecs,
    float *out_scores);

typedef void (*top_k_fn)(
    const float *scores, int n, int k,
    int *out_indices, float *out_scores);

typedef void (*normalize_fn)(float *vecs, int dim, int n_vecs);

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static float l2_norm(const float *v, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

// Simulate byte-histogram embedding: count byte frequencies of text
static void embed_text(const char *text, float *out) {
    memset(out, 0, DIM * sizeof(float));
    for (const char *p = text; *p; p++) {
        out[(unsigned char)*p] += 1.0f;
    }
}

// Simple xorshift64 for reproducible random
static uint64_t rng_state = 0xDEADBEEFCAFEBABEULL;
static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static float rand_float(void) {
    return (float)(xorshift64() & 0xFFFFFF) / (float)0xFFFFFF;
}

// Generate realistic histogram vectors from random "text"
static void gen_random_histogram(float *out) {
    memset(out, 0, DIM * sizeof(float));
    // Simulate 200-4000 byte text block (typical vault block)
    int len = 200 + (int)(xorshift64() % 3800);
    for (int i = 0; i < len; i++) {
        // ASCII-biased: mostly printable chars (32-126)
        int b = 32 + (int)(xorshift64() % 95);
        out[b] += 1.0f;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <libsearch.so>\n", argv[0]);
        return 1;
    }

    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }

    batch_cosine_fn batch_cosine = dlsym(lib, "batch_cosine");
    top_k_fn top_k = dlsym(lib, "top_k");
    normalize_fn normalize_vectors = dlsym(lib, "normalize_vectors");

    if (!batch_cosine || !top_k || !normalize_vectors) {
        fprintf(stderr, "Missing symbols: cosine=%p top_k=%p norm=%p\n",
                (void*)batch_cosine, (void*)top_k, (void*)normalize_vectors);
        dlclose(lib);
        return 1;
    }

    printf("=== Vault Search Baseline Benchmark (256-dim histogram) ===\n");
    printf("CPU: AMD EPYC 9354P, kernel: %s\n\n", argv[1]);

    int sizes[] = {64, 256, 1024, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int si = 0; si < n_sizes; si++) {
        int n_vecs = sizes[si];
        printf("--- n_vecs = %d ---\n", n_vecs);

        // Allocate vectors + query
        float *vecs = aligned_alloc(64, (size_t)n_vecs * DIM * sizeof(float));
        float *scores = aligned_alloc(64, (size_t)n_vecs * sizeof(float));
        float query[DIM];
        int top_indices[10];
        float top_scores[10];

        // Generate random histogram vectors
        rng_state = 0xDEADBEEFCAFEBABEULL; // reproducible
        for (int i = 0; i < n_vecs; i++) {
            gen_random_histogram(&vecs[i * DIM]);
        }
        normalize_vectors(vecs, DIM, n_vecs);

        // Plant a known-best match at a random position
        int plant_idx = n_vecs / 3;
        embed_text("SIMD kernel optimization for ARM NEON and AVX-512 vector processing", &vecs[plant_idx * DIM]);
        normalize_vectors(&vecs[plant_idx * DIM], DIM, 1);

        // Query: similar to planted vector
        embed_text("SIMD vector optimization for AVX-512 and ARM", query);
        normalize_vectors(query, DIM, 1);
        float qnorm = l2_norm(query, DIM);

        // --- Warmup ---
        for (int i = 0; i < WARMUP; i++) {
            batch_cosine(query, qnorm, vecs, DIM, n_vecs, scores);
        }

        // --- batch_cosine latency ---
        double t0 = now_ns();
        for (int i = 0; i < ITERS; i++) {
            batch_cosine(query, qnorm, vecs, DIM, n_vecs, scores);
        }
        double cosine_ns = (now_ns() - t0) / ITERS;

        // --- top_k latency ---
        for (int i = 0; i < WARMUP; i++) {
            top_k(scores, n_vecs, 5, top_indices, top_scores);
        }
        t0 = now_ns();
        for (int i = 0; i < ITERS; i++) {
            top_k(scores, n_vecs, 5, top_indices, top_scores);
        }
        double topk_ns = (now_ns() - t0) / ITERS;

        // --- Precision check ---
        top_k(scores, n_vecs, 5, top_indices, top_scores);
        int planted_rank = -1;
        for (int i = 0; i < 5; i++) {
            if (top_indices[i] == plant_idx) {
                planted_rank = i;
                break;
            }
        }

        double total_ns = cosine_ns + topk_ns;
        double data_mb = (double)n_vecs * DIM * sizeof(float) / (1024.0 * 1024.0);
        double bw_gbs = (data_mb / 1024.0) / (cosine_ns / 1e9);

        printf("  batch_cosine: %8.1f ns  (%.2f us)\n", cosine_ns, cosine_ns / 1000.0);
        printf("  top_k(5):     %8.1f ns  (%.2f us)\n", topk_ns, topk_ns / 1000.0);
        printf("  total:        %8.1f ns  (%.2f us)\n", total_ns, total_ns / 1000.0);
        printf("  bandwidth:    %.2f GB/s\n", bw_gbs);
        printf("  planted vec:  rank=%d  score=%.4f  %s\n",
               planted_rank, planted_rank >= 0 ? top_scores[planted_rank] : 0.0f,
               planted_rank == 0 ? "PASS" : "FAIL");

        // Show top-5 scores
        printf("  top-5 scores:");
        for (int i = 0; i < 5 && i < n_vecs; i++) {
            printf(" [%d]=%.4f", top_indices[i], top_scores[i]);
        }
        printf("\n\n");

        free(vecs);
        free(scores);
    }

    // --- Recall precision test ---
    printf("=== Recall Precision Test ===\n");
    int n = 1024;
    float *vecs = aligned_alloc(64, (size_t)n * DIM * sizeof(float));
    float *scores = aligned_alloc(64, (size_t)n * sizeof(float));

    const char *corpus[] = {
        "The Rust programming language guarantees memory safety",
        "Python is great for machine learning and data science",
        "SIMD instructions accelerate vector math on modern CPUs",
        "ChaCha20 is a fast stream cipher for encryption",
        "KV-cache compression reduces memory bandwidth for LLMs",
        "Johnson-Lindenstrauss lemma preserves distances in projection",
        "Walsh-Hadamard transform is an orthogonal transformation",
        "ARM NEON provides 128-bit SIMD on mobile and edge devices",
        "Quantization reduces model size with minimal quality loss",
        "Cosine similarity measures angle between embedding vectors",
    };
    int n_corpus = sizeof(corpus) / sizeof(corpus[0]);

    // Fill store: repeat corpus + random filler
    rng_state = 42;
    for (int i = 0; i < n; i++) {
        if (i < n_corpus) {
            embed_text(corpus[i], &vecs[i * DIM]);
        } else {
            gen_random_histogram(&vecs[i * DIM]);
        }
    }
    normalize_vectors(vecs, DIM, n);

    // Queries and expected best-match index
    struct { const char *query; int expected; const char *label; } tests[] = {
        {"Rust memory safety guarantees", 0, "Rust"},
        {"SIMD vector acceleration", 2, "SIMD"},
        {"stream cipher encryption", 3, "ChaCha20"},
        {"KV cache memory bandwidth", 4, "KV-cache"},
        {"random projection distance", 5, "JL-lemma"},
        {"ARM NEON edge computing", 7, "ARM"},
    };
    int n_tests = sizeof(tests) / sizeof(tests[0]);
    int pass = 0;

    float query[DIM];
    int top_indices[5];
    float top_scores_buf[5];

    for (int t = 0; t < n_tests; t++) {
        embed_text(tests[t].query, query);
        normalize_vectors(query, DIM, 1);
        float qn = l2_norm(query, DIM);
        batch_cosine(query, qn, vecs, DIM, n, scores);
        top_k(scores, n, 5, top_indices, top_scores_buf);

        int found_rank = -1;
        for (int i = 0; i < 5; i++) {
            if (top_indices[i] == tests[t].expected) {
                found_rank = i;
                break;
            }
        }

        int ok = found_rank >= 0;
        if (ok) pass++;
        printf("  %-25s -> expected[%d] rank=%d score=%.4f %s\n",
               tests[t].label, tests[t].expected,
               found_rank,
               found_rank >= 0 ? top_scores_buf[found_rank] : 0.0f,
               ok ? "PASS" : "FAIL");
    }

    printf("\nPrecision: %d/%d queries found expected doc in top-5\n", pass, n_tests);

    free(vecs);
    free(scores);
    dlclose(lib);
    return 0;
}
