// Benchmark-after: vault search with JL-projected 64-dim vectors.
//
// Same test as bench_search_baseline.c but projects 256-dim histograms
// to 64-dim via JL before searching. Compares latency and precision.
//
// Build:
//   gcc -O2 -o bench_search_jl bench_search_jl.c -ldl -lm
// Run:
//   ./bench_search_jl <libjl_project.so> <libsearch.so>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <stdint.h>

#define IN_DIM 256
#define OUT_DIM 64
#define WARMUP 100
#define ITERS 1000

typedef void (*jl_project_fn)(const float*, const float*, int, int, float*, float*);
typedef void (*jl_project_batch_fn)(const float*, const float*, int, int, float*, float*, int);
typedef void (*batch_cosine_fn)(const float*, float, const float*, int, int, float*);
typedef void (*top_k_fn)(const float*, int, int, int*, float*);
typedef void (*normalize_fn)(float*, int, int);

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static float l2_norm(const float *v, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += v[i]*v[i]; return sqrtf(s);
}

static void embed_text(const char *text, float *out) {
    memset(out, 0, IN_DIM * sizeof(float));
    for (const char *p = text; *p; p++) out[(unsigned char)*p] += 1.0f;
}

static uint64_t rng_state = 0xDEADBEEFCAFEBABEULL;
static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_state = x; return x;
}

static void gen_random_histogram(float *out) {
    memset(out, 0, IN_DIM * sizeof(float));
    int len = 200 + (int)(xorshift64() % 3800);
    for (int i = 0; i < len; i++) {
        int b = 32 + (int)(xorshift64() % 95);
        out[b] += 1.0f;
    }
}

static void gen_signs(float *signs, int n) {
    uint64_t seed = 0x1234567890ABCDEFULL;
    for (int i = 0; i < n; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        signs[i] = (seed % 2) ? 1.0f : -1.0f;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <libjl_project.so> <libsearch.so>\n", argv[0]);
        return 1;
    }

    void *jl_lib = dlopen(argv[1], RTLD_NOW);
    void *search_lib = dlopen(argv[2], RTLD_NOW);
    if (!jl_lib || !search_lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    jl_project_fn jl_project = dlsym(jl_lib, "jl_project");
    jl_project_batch_fn jl_project_batch = dlsym(jl_lib, "jl_project_batch");
    batch_cosine_fn batch_cosine = dlsym(search_lib, "batch_cosine");
    top_k_fn top_k = dlsym(search_lib, "top_k");
    normalize_fn normalize_vectors = dlsym(search_lib, "normalize_vectors");

    float *signs = aligned_alloc(64, IN_DIM * sizeof(float));
    float *scratch = aligned_alloc(64, IN_DIM * sizeof(float));
    gen_signs(signs, IN_DIM);

    printf("=== Vault Search JL-Projected Benchmark (256->64 dim) ===\n");
    printf("CPU: AMD EPYC 9354P, search: %s\n\n", argv[2]);

    // --- Latency benchmark ---
    int sizes[] = {64, 256, 1024, 4096};
    for (int si = 0; si < 4; si++) {
        int n_vecs = sizes[si];
        printf("--- n_vecs = %d ---\n", n_vecs);

        float *vecs_256 = aligned_alloc(64, n_vecs * IN_DIM * sizeof(float));
        float *vecs_64 = aligned_alloc(64, n_vecs * OUT_DIM * sizeof(float));
        float *scores = aligned_alloc(64, n_vecs * sizeof(float));
        float query_256[IN_DIM], query_64[OUT_DIM];
        int top_indices[10]; float top_scores[10];

        rng_state = 0xDEADBEEFCAFEBABEULL;
        for (int i = 0; i < n_vecs; i++)
            gen_random_histogram(&vecs_256[i * IN_DIM]);
        normalize_vectors(vecs_256, IN_DIM, n_vecs);

        // Plant known-best
        int plant_idx = n_vecs / 3;
        embed_text("SIMD kernel optimization for ARM NEON and AVX-512 vector processing",
                   &vecs_256[plant_idx * IN_DIM]);
        normalize_vectors(&vecs_256[plant_idx * IN_DIM], IN_DIM, 1);

        // Project all to 64-dim
        double t0 = now_ns();
        jl_project_batch(vecs_256, signs, IN_DIM, OUT_DIM, vecs_64, scratch, n_vecs);
        double project_ns = now_ns() - t0;
        normalize_vectors(vecs_64, OUT_DIM, n_vecs);

        // Query
        embed_text("SIMD vector optimization for AVX-512 and ARM", query_256);
        normalize_vectors(query_256, IN_DIM, 1);
        jl_project(query_256, signs, IN_DIM, OUT_DIM, query_64, scratch);
        normalize_vectors(query_64, OUT_DIM, 1);
        float qn = l2_norm(query_64, OUT_DIM);

        // Warmup
        for (int i = 0; i < WARMUP; i++)
            batch_cosine(query_64, qn, vecs_64, OUT_DIM, n_vecs, scores);

        // Bench
        t0 = now_ns();
        for (int i = 0; i < ITERS; i++)
            batch_cosine(query_64, qn, vecs_64, OUT_DIM, n_vecs, scores);
        double cosine_ns = (now_ns() - t0) / ITERS;

        for (int i = 0; i < WARMUP; i++)
            top_k(scores, n_vecs, 5, top_indices, top_scores);
        t0 = now_ns();
        for (int i = 0; i < ITERS; i++)
            top_k(scores, n_vecs, 5, top_indices, top_scores);
        double topk_ns = (now_ns() - t0) / ITERS;

        top_k(scores, n_vecs, 5, top_indices, top_scores);
        int planted_rank = -1;
        for (int i = 0; i < 5; i++)
            if (top_indices[i] == plant_idx) { planted_rank = i; break; }

        double total_ns = cosine_ns + topk_ns;
        double data_mb = (double)n_vecs * OUT_DIM * sizeof(float) / (1024.0*1024.0);
        double bw_gbs = (data_mb / 1024.0) / (cosine_ns / 1e9);

        printf("  batch_cosine: %8.1f ns  (%.2f us)\n", cosine_ns, cosine_ns/1000.0);
        printf("  top_k(5):     %8.1f ns  (%.2f us)\n", topk_ns, topk_ns/1000.0);
        printf("  total:        %8.1f ns  (%.2f us)\n", total_ns, total_ns/1000.0);
        printf("  bandwidth:    %.2f GB/s\n", bw_gbs);
        printf("  project:      %.1f us (one-time)\n", project_ns/1000.0);
        printf("  planted vec:  rank=%d  score=%.4f  %s\n",
               planted_rank, planted_rank >= 0 ? top_scores[planted_rank] : 0.0f,
               planted_rank >= 0 ? "PASS" : "FAIL");
        printf("  top-5 scores:");
        for (int i = 0; i < 5 && i < n_vecs; i++)
            printf(" [%d]=%.4f", top_indices[i], top_scores[i]);
        printf("\n\n");

        free(vecs_256); free(vecs_64); free(scores);
    }

    // --- Recall precision test (same as baseline) ---
    printf("=== Recall Precision Test (JL-projected) ===\n");
    int n = 1024;
    float *vecs_256 = aligned_alloc(64, n * IN_DIM * sizeof(float));
    float *vecs_64 = aligned_alloc(64, n * OUT_DIM * sizeof(float));
    float *scores = aligned_alloc(64, n * sizeof(float));

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
    int n_corpus = 10;

    rng_state = 42;
    for (int i = 0; i < n; i++) {
        if (i < n_corpus)
            embed_text(corpus[i], &vecs_256[i * IN_DIM]);
        else
            gen_random_histogram(&vecs_256[i * IN_DIM]);
    }
    normalize_vectors(vecs_256, IN_DIM, n);
    jl_project_batch(vecs_256, signs, IN_DIM, OUT_DIM, vecs_64, scratch, n);
    normalize_vectors(vecs_64, OUT_DIM, n);

    struct { const char *query; int expected; const char *label; } tests[] = {
        {"Rust memory safety guarantees", 0, "Rust"},
        {"SIMD vector acceleration", 2, "SIMD"},
        {"stream cipher encryption", 3, "ChaCha20"},
        {"KV cache memory bandwidth", 4, "KV-cache"},
        {"random projection distance", 5, "JL-lemma"},
        {"ARM NEON edge computing", 7, "ARM"},
    };
    int n_tests = 6, pass = 0;

    float query_256[IN_DIM], query_64[OUT_DIM];
    int top_idx[5]; float top_sc[5];

    for (int t = 0; t < n_tests; t++) {
        embed_text(tests[t].query, query_256);
        normalize_vectors(query_256, IN_DIM, 1);
        jl_project(query_256, signs, IN_DIM, OUT_DIM, query_64, scratch);
        normalize_vectors(query_64, OUT_DIM, 1);
        float qn = l2_norm(query_64, OUT_DIM);
        batch_cosine(query_64, qn, vecs_64, OUT_DIM, n, scores);
        top_k(scores, n, 5, top_idx, top_sc);

        int found = -1;
        for (int i = 0; i < 5; i++)
            if (top_idx[i] == tests[t].expected) { found = i; break; }

        int ok = found >= 0;
        if (ok) pass++;
        printf("  %-25s -> expected[%d] rank=%d score=%.4f %s\n",
               tests[t].label, tests[t].expected, found,
               found >= 0 ? top_sc[found] : 0.0f, ok ? "PASS" : "FAIL");
    }
    printf("\nPrecision: %d/%d queries found expected doc in top-5\n", pass, n_tests);

    free(vecs_256); free(vecs_64); free(scores); free(signs); free(scratch);
    dlclose(jl_lib); dlclose(search_lib);
    return 0;
}
