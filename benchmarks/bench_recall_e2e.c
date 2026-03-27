// End-to-end recall benchmark: simulates VectorStore insert + recall.
//
// Measures the full pipeline that changed:
//   256-dim mode:  embed → normalize → store → cosine search (256-dim)
//   64-dim mode:   embed → normalize → JL project → normalize → store → cosine search (64-dim)
//
// Build:
//   gcc -O2 -o bench_recall_e2e bench_recall_e2e.c -ldl -lm
// Run (256-dim baseline):
//   ./bench_recall_e2e <libsearch.so> none
// Run (64-dim JL):
//   ./bench_recall_e2e <libsearch.so> <libjl_project.so>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <stdint.h>

#define RAW_DIM 256
#define JL_DIM 64

typedef void (*batch_cosine_fn)(const float*, float, const float*, int, int, float*);
typedef void (*top_k_fn)(const float*, int, int, int*, float*);
typedef void (*normalize_fn)(float*, int, int);
typedef void (*jl_project_fn)(const float*, const float*, int, int, float*, float*);

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static float l2_norm(const float *v, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += v[i]*v[i]; return sqrtf(s);
}

static void embed_text(const char *text, float *out) {
    memset(out, 0, RAW_DIM * sizeof(float));
    for (const char *p = text; *p; p++) out[(unsigned char)*p] += 1.0f;
}

static void gen_signs(float *signs) {
    uint64_t rng = 0x4F6C6F72696E4A4CULL;
    for (int i = 0; i < RAW_DIM; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        signs[i] = (rng % 2 == 0) ? 1.0f : -1.0f;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <libsearch.so> <libjl_project.so|none>\n", argv[0]);
        return 1;
    }

    void *search_lib = dlopen(argv[1], RTLD_NOW);
    if (!search_lib) { fprintf(stderr, "dlopen search: %s\n", dlerror()); return 1; }

    batch_cosine_fn batch_cosine = dlsym(search_lib, "batch_cosine");
    top_k_fn top_k = dlsym(search_lib, "top_k");
    normalize_fn normalize = dlsym(search_lib, "normalize_vectors");

    int use_jl = strcmp(argv[2], "none") != 0;
    jl_project_fn jl_project = NULL;
    void *jl_lib = NULL;
    if (use_jl) {
        jl_lib = dlopen(argv[2], RTLD_NOW);
        if (!jl_lib) { fprintf(stderr, "dlopen jl: %s\n", dlerror()); return 1; }
        jl_project = dlsym(jl_lib, "jl_project");
    }

    int store_dim = use_jl ? JL_DIM : RAW_DIM;
    int capacity = 1024;

    float *signs = aligned_alloc(64, RAW_DIM * sizeof(float));
    float *scratch = aligned_alloc(64, RAW_DIM * sizeof(float));
    gen_signs(signs);

    float *store_vecs = aligned_alloc(64, capacity * store_dim * sizeof(float));
    float *scores = aligned_alloc(64, capacity * sizeof(float));
    float raw_buf[RAW_DIM];
    float proj_buf[JL_DIM];
    int top_idx[5]; float top_sc[5];

    const char *corpus[] = {
        "SIMD vector optimization for ARM NEON and AVX-512 processing units and pipelines",
        "The Rust programming language guarantees memory safety through ownership and borrowing",
        "Python machine learning with TensorFlow PyTorch and scikit-learn deep learning frameworks",
        "ChaCha20 is a fast stream cipher used in TLS WireGuard and secure communications",
        "KV-cache compression reduces memory bandwidth requirements for large language model inference",
        "Johnson-Lindenstrauss lemma preserves pairwise distances under random projection in math",
        "Walsh-Hadamard transform is a fast orthogonal transformation for signal processing tasks",
        "ARM NEON provides 128-bit SIMD on mobile and edge devices like Raspberry Pi five board",
        "Quantization reduces neural network model size with minimal quality degradation overhead",
        "Cosine similarity measures the angle between high-dimensional embedding vectors in space",
    };
    int n_corpus = 10;

    const char *queries[] = {
        "SIMD vector acceleration for modern CPUs",
        "Rust memory safety ownership model",
        "stream cipher encryption TLS protocol",
        "KV cache memory bandwidth LLM inference",
        "random projection distance preservation math",
    };
    int n_queries = 5;

    printf("=== Recall E2E Benchmark (%s, %d-dim store) ===\n\n",
           use_jl ? "JL-projected" : "raw histogram", store_dim);

    // --- Benchmark INSERT ---
    int insert_iters = 100;
    double t0 = now_ns();
    for (int iter = 0; iter < insert_iters; iter++) {
        for (int i = 0; i < capacity; i++) {
            embed_text(corpus[i % n_corpus], raw_buf);
            normalize(raw_buf, RAW_DIM, 1);
            if (use_jl) {
                jl_project(raw_buf, signs, RAW_DIM, JL_DIM, proj_buf, scratch);
                normalize(proj_buf, JL_DIM, 1);
                memcpy(&store_vecs[i * store_dim], proj_buf, store_dim * sizeof(float));
            } else {
                memcpy(&store_vecs[i * store_dim], raw_buf, store_dim * sizeof(float));
            }
        }
    }
    double insert_total_ns = now_ns() - t0;
    double per_insert_us = insert_total_ns / (insert_iters * capacity) / 1000.0;
    double insert_1024_ms = per_insert_us * capacity / 1000.0;

    printf("Insert (fill 1024 entries, %d iterations):\n", insert_iters);
    printf("  Per insert:   %.2f us\n", per_insert_us);
    printf("  1024 inserts: %.2f ms\n\n", insert_1024_ms);

    // --- Benchmark RECALL ---
    int recall_iters = 10000;

    // Warmup
    for (int i = 0; i < 100; i++) {
        for (int q = 0; q < n_queries; q++) {
            embed_text(queries[q], raw_buf);
            normalize(raw_buf, RAW_DIM, 1);
            float *qvec = raw_buf;
            float qbuf[JL_DIM];
            int qdim = RAW_DIM;
            if (use_jl) {
                jl_project(raw_buf, signs, RAW_DIM, JL_DIM, qbuf, scratch);
                normalize(qbuf, JL_DIM, 1);
                qvec = qbuf;
                qdim = JL_DIM;
            }
            float qn = l2_norm(qvec, qdim);
            batch_cosine(qvec, qn, store_vecs, qdim, capacity, scores);
            top_k(scores, capacity, 5, top_idx, top_sc);
        }
    }

    t0 = now_ns();
    for (int iter = 0; iter < recall_iters; iter++) {
        for (int q = 0; q < n_queries; q++) {
            embed_text(queries[q], raw_buf);
            normalize(raw_buf, RAW_DIM, 1);
            float *qvec = raw_buf;
            float qbuf[JL_DIM];
            int qdim = RAW_DIM;
            if (use_jl) {
                jl_project(raw_buf, signs, RAW_DIM, JL_DIM, qbuf, scratch);
                normalize(qbuf, JL_DIM, 1);
                qvec = qbuf;
                qdim = JL_DIM;
            }
            float qn = l2_norm(qvec, qdim);
            batch_cosine(qvec, qn, store_vecs, qdim, capacity, scores);
            top_k(scores, capacity, 5, top_idx, top_sc);
        }
    }
    double recall_total_ns = now_ns() - t0;
    int total_recalls = recall_iters * n_queries;
    double per_recall_us = recall_total_ns / total_recalls / 1000.0;

    printf("Recall (top-5 from 1024 entries, %d iterations × %d queries):\n",
           recall_iters, n_queries);
    printf("  Per recall:   %.2f us\n", per_recall_us);
    printf("  Total:        %.1f ms\n\n", recall_total_ns / 1e6);

    // --- Memory ---
    double mem_kb = (double)capacity * store_dim * sizeof(float) / 1024.0;
    printf("Memory: %.0f KB for %d vectors (%d B/vec)\n\n",
           mem_kb, capacity, store_dim * (int)sizeof(float));

    // --- Summary ---
    printf("Per-message overhead (1 insert + 1 recall):\n");
    printf("  %.2f us total\n", per_insert_us + per_recall_us);

    free(signs); free(scratch); free(store_vecs); free(scores);
    if (jl_lib) dlclose(jl_lib);
    dlclose(search_lib);
    return 0;
}
