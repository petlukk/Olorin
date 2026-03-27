// Test: JL projection kernel correctness + benchmark
//
// Verifies:
//   1. Norm preservation (E[||proj||^2] ≈ ||orig||^2)
//   2. Distance preservation (JL lemma)
//   3. Cosine similarity preservation
//   4. Batch projection consistency
//   5. Benchmark: projection + search latency
//
// Build:
//   gcc -O2 -o test_jl_project test_jl_project.c -ldl -lm
// Run:
//   ./test_jl_project <libjl_project.so> <libsearch.so>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>
#include <stdint.h>

#define IN_DIM 256
#define OUT_DIM 64
#define EPS 0.15f

typedef void (*jl_project_fn)(const float*, const float*, int, int, float*, float*);
typedef void (*jl_project_batch_fn)(const float*, const float*, int, int, float*, float*, int);
typedef void (*batch_cosine_fn)(const float*, float, const float*, int, int, float*);
typedef void (*top_k_fn)(const float*, int, int, int*, float*);
typedef void (*normalize_fn)(float*, int, int);

static uint64_t rng = 0xDEADBEEF42ULL;
static uint64_t xorshift64(void) {
    uint64_t x = rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng = x; return x;
}
static float rand_float(void) {
    return (float)(xorshift64() & 0xFFFFFF) / (float)0xFFFFFF;
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static float l2_norm(const float *v, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += v[i]*v[i]; return sqrtf(s);
}

static float dot(const float *a, const float *b, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += a[i]*b[i]; return s;
}

static void embed_text(const char *text, float *out) {
    memset(out, 0, IN_DIM * sizeof(float));
    for (const char *p = text; *p; p++) out[(unsigned char)*p] += 1.0f;
}

static void gen_signs(float *signs, int n) {
    for (int i = 0; i < n; i++) signs[i] = (xorshift64() % 2) ? 1.0f : -1.0f;
}

static int pass_count = 0, fail_count = 0;
static void check(const char *name, int ok) {
    if (ok) { pass_count++; printf("  PASS: %s\n", name); }
    else    { fail_count++; printf("  FAIL: %s\n", name); }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <libjl_project.so> <libsearch.so>\n", argv[0]);
        return 1;
    }

    void *jl_lib = dlopen(argv[1], RTLD_NOW);
    void *search_lib = dlopen(argv[2], RTLD_NOW);
    if (!jl_lib || !search_lib) {
        fprintf(stderr, "dlopen: %s\n", dlerror()); return 1;
    }

    jl_project_fn jl_project = dlsym(jl_lib, "jl_project");
    jl_project_batch_fn jl_project_batch = dlsym(jl_lib, "jl_project_batch");
    batch_cosine_fn batch_cosine = dlsym(search_lib, "batch_cosine");
    top_k_fn top_k = dlsym(search_lib, "top_k");
    normalize_fn normalize_vectors = dlsym(search_lib, "normalize_vectors");

    if (!jl_project || !jl_project_batch || !batch_cosine || !top_k || !normalize_vectors) {
        fprintf(stderr, "Missing symbols\n"); return 1;
    }

    float *signs = aligned_alloc(64, IN_DIM * sizeof(float));
    float *scratch = aligned_alloc(64, IN_DIM * sizeof(float));
    gen_signs(signs, IN_DIM);

    printf("=== JL Projection Tests (256 -> 64 dim) ===\n\n");

    // --- Test 1: Norm preservation ---
    printf("[1] Norm preservation (average over 100 random vectors)\n");
    float ratio_sum = 0;
    float ratio_min = 1e9, ratio_max = 0;
    for (int t = 0; t < 100; t++) {
        float vec[IN_DIM], proj[OUT_DIM];
        for (int i = 0; i < IN_DIM; i++) vec[i] = rand_float() * 4.0f - 2.0f;
        float norm_orig = l2_norm(vec, IN_DIM);
        jl_project(vec, signs, IN_DIM, OUT_DIM, proj, scratch);
        float norm_proj = l2_norm(proj, OUT_DIM);
        float ratio = norm_proj / norm_orig;
        ratio_sum += ratio;
        if (ratio < ratio_min) ratio_min = ratio;
        if (ratio > ratio_max) ratio_max = ratio;
    }
    float ratio_avg = ratio_sum / 100.0f;
    char name1[128];
    snprintf(name1, sizeof(name1),
        "avg ratio=%.4f min=%.4f max=%.4f (expect ~1.0)", ratio_avg, ratio_min, ratio_max);
    check(name1, fabsf(ratio_avg - 1.0f) < 0.1f);

    // --- Test 2: Distance preservation (JL core guarantee) ---
    printf("\n[2] Pairwise distance preservation (20 pairs)\n");
    int dist_pass = 0;
    for (int t = 0; t < 20; t++) {
        float a[IN_DIM], b[IN_DIM], pa[OUT_DIM], pb[OUT_DIM];
        for (int i = 0; i < IN_DIM; i++) {
            a[i] = rand_float() * 4.0f - 2.0f;
            b[i] = rand_float() * 4.0f - 2.0f;
        }
        float dist_orig = 0;
        for (int i = 0; i < IN_DIM; i++) { float d = a[i]-b[i]; dist_orig += d*d; }

        jl_project(a, signs, IN_DIM, OUT_DIM, pa, scratch);
        jl_project(b, signs, IN_DIM, OUT_DIM, pb, scratch);
        float dist_proj = 0;
        for (int i = 0; i < OUT_DIM; i++) { float d = pa[i]-pb[i]; dist_proj += d*d; }

        float rel_err = fabsf(dist_orig - dist_proj) / dist_orig;
        if (rel_err < 0.3f) dist_pass++;
    }
    char name2[64];
    snprintf(name2, sizeof(name2), "%d/20 pairs within 30%% (need >=15)", dist_pass);
    check(name2, dist_pass >= 15);

    // --- Test 3: Cosine similarity preservation ---
    printf("\n[3] Cosine similarity preservation (text embeddings)\n");
    const char *texts[] = {
        "SIMD vector optimization for ARM NEON processors",
        "SIMD kernel acceleration for AVX-512 instructions",
        "Python machine learning with TensorFlow and PyTorch",
        "Rust memory safety and ownership model design",
    };
    float vecs[4][IN_DIM], projs[4][OUT_DIM];
    for (int i = 0; i < 4; i++) {
        embed_text(texts[i], vecs[i]);
        // Normalize original
        float n = l2_norm(vecs[i], IN_DIM);
        for (int j = 0; j < IN_DIM; j++) vecs[i][j] /= n;
        jl_project(vecs[i], signs, IN_DIM, OUT_DIM, projs[i], scratch);
    }
    // SIMD pair (0,1) should have higher similarity than SIMD vs Python (0,2)
    float cos_01_orig = dot(vecs[0], vecs[1], IN_DIM);
    float cos_02_orig = dot(vecs[0], vecs[2], IN_DIM);
    float n0p = l2_norm(projs[0], OUT_DIM);
    float n1p = l2_norm(projs[1], OUT_DIM);
    float n2p = l2_norm(projs[2], OUT_DIM);
    float cos_01_proj = dot(projs[0], projs[1], OUT_DIM) / (n0p * n1p);
    float cos_02_proj = dot(projs[0], projs[2], OUT_DIM) / (n0p * n2p);

    printf("  orig: cos(SIMD,SIMD)=%.4f  cos(SIMD,Python)=%.4f\n", cos_01_orig, cos_02_orig);
    printf("  proj: cos(SIMD,SIMD)=%.4f  cos(SIMD,Python)=%.4f\n", cos_01_proj, cos_02_proj);
    check("SIMD pair more similar than SIMD-Python (orig)", cos_01_orig > cos_02_orig);
    check("SIMD pair more similar than SIMD-Python (proj)", cos_01_proj > cos_02_proj);

    // --- Test 4: Batch projection consistency ---
    printf("\n[4] Batch projection matches single projection\n");
    int n_batch = 8;
    float *batch_in = aligned_alloc(64, n_batch * IN_DIM * sizeof(float));
    float *batch_out = aligned_alloc(64, n_batch * OUT_DIM * sizeof(float));
    float single_out[OUT_DIM];

    rng = 12345;
    for (int i = 0; i < n_batch * IN_DIM; i++) batch_in[i] = rand_float() * 2.0f - 1.0f;

    jl_project_batch(batch_in, signs, IN_DIM, OUT_DIM, batch_out, scratch, n_batch);

    rng = 12345; // reset to get same data
    for (int i = 0; i < n_batch * IN_DIM; i++) batch_in[i] = rand_float() * 2.0f - 1.0f;

    int batch_ok = 1;
    for (int v = 0; v < n_batch; v++) {
        jl_project(&batch_in[v * IN_DIM], signs, IN_DIM, OUT_DIM, single_out, scratch);
        for (int j = 0; j < OUT_DIM; j++) {
            if (fabsf(batch_out[v * OUT_DIM + j] - single_out[j]) > 1e-5f) {
                batch_ok = 0;
                break;
            }
        }
    }
    check("batch matches single projection", batch_ok);

    // --- Benchmark: JL project + search ---
    printf("\n=== Benchmark: 256-dim vs JL-projected 64-dim search ===\n");
    int sizes[] = {64, 256, 1024, 4096};
    int n_sizes = 4;
    int warmup = 100, iters = 1000;

    for (int si = 0; si < n_sizes; si++) {
        int n_vecs = sizes[si];

        // Generate 256-dim histogram vectors
        float *vecs_256 = aligned_alloc(64, n_vecs * IN_DIM * sizeof(float));
        float *vecs_64 = aligned_alloc(64, n_vecs * OUT_DIM * sizeof(float));
        float *scores_256 = aligned_alloc(64, n_vecs * sizeof(float));
        float *scores_64 = aligned_alloc(64, n_vecs * sizeof(float));
        float query_256[IN_DIM], query_64[OUT_DIM];
        int top_idx[5]; float top_sc[5];

        rng = 0xBEEF;
        for (int i = 0; i < n_vecs * IN_DIM; i++)
            vecs_256[i] = rand_float() * 10.0f;
        // Normalize 256-dim
        normalize_vectors(vecs_256, IN_DIM, n_vecs);

        // Project all to 64-dim
        double t0 = now_ns();
        jl_project_batch(vecs_256, signs, IN_DIM, OUT_DIM, vecs_64, scratch, n_vecs);
        double project_ns = now_ns() - t0;
        // Normalize 64-dim
        normalize_vectors(vecs_64, OUT_DIM, n_vecs);

        // Query
        embed_text("SIMD optimization for ARM NEON vector processing", query_256);
        normalize_vectors(query_256, IN_DIM, 1);
        float qn_256 = l2_norm(query_256, IN_DIM);
        jl_project(query_256, signs, IN_DIM, OUT_DIM, query_64, scratch);
        normalize_vectors(query_64, OUT_DIM, 1);
        float qn_64 = l2_norm(query_64, OUT_DIM);

        // Warmup
        for (int i = 0; i < warmup; i++) {
            batch_cosine(query_256, qn_256, vecs_256, IN_DIM, n_vecs, scores_256);
            batch_cosine(query_64, qn_64, vecs_64, OUT_DIM, n_vecs, scores_64);
        }

        // Bench 256-dim
        t0 = now_ns();
        for (int i = 0; i < iters; i++)
            batch_cosine(query_256, qn_256, vecs_256, IN_DIM, n_vecs, scores_256);
        double ns_256 = (now_ns() - t0) / iters;

        // Bench 64-dim
        t0 = now_ns();
        for (int i = 0; i < iters; i++)
            batch_cosine(query_64, qn_64, vecs_64, OUT_DIM, n_vecs, scores_64);
        double ns_64 = (now_ns() - t0) / iters;

        // Top-k comparison
        top_k(scores_256, n_vecs, 5, top_idx, top_sc);
        int top_256[5]; for (int i=0;i<5;i++) top_256[i] = top_idx[i];

        top_k(scores_64, n_vecs, 5, top_idx, top_sc);
        int overlap = 0;
        for (int i=0;i<5;i++)
            for (int j=0;j<5;j++)
                if (top_idx[i] == top_256[j]) overlap++;

        printf("n=%4d | 256d: %7.1f ns | 64d: %7.1f ns | speedup: %.1fx | project: %.1f us | top5 overlap: %d/5\n",
               n_vecs, ns_256, ns_64, ns_256/ns_64, project_ns/1000.0, overlap);

        free(vecs_256); free(vecs_64); free(scores_256); free(scores_64);
    }

    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);

    free(signs); free(scratch); free(batch_in); free(batch_out);
    dlclose(jl_lib); dlclose(search_lib);
    return fail_count > 0 ? 1 : 0;
}
