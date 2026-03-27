// Test: eakv pre-rotated quantization end-to-end.
//
// Verifies that attention scores and output are correct after
// pre-rotation by comparing against naive f32 reference.
//
// Build (link against Olorin's libeakv):
//   gcc -O2 -I../../crates/eakv/csrc -o test_eakv_rotated test_eakv_rotated.c \
//     -L../../target/debug -leakv -lm -lpthread
//
// Or build standalone using the static lib from cargo:
//   (cargo build -p eakv first, then link target/debug/libeakv.a)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "eakv.h"

static int pass = 0, fail = 0;

static void check(const char *name, int ok, const char *detail) {
    if (ok) { pass++; printf("  PASS: %s\n", name); }
    else    { fail++; printf("  FAIL: %s — %s\n", name, detail); }
}

static float dot_f32(const float *a, const float *b, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
}

int main(void) {
    int nl = 1, nh = 4, hd = 128, sl = 32;
    int total = nl * 2 * nh * sl * hd;

    printf("=== eakv Pre-Rotated Quantization E2E Test ===\n");
    printf("  %d layers, %d heads, %d dim, %d seq\n\n", nl, nh, hd, sl);

    // Generate synthetic KV data with outliers (realistic scenario)
    float *data = malloc((size_t)total * sizeof(float));
    uint64_t rng = 42;
    for (int i = 0; i < total; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        float u1 = (float)(rng & 0xFFFFFF) / (float)0x1000000;
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        float u2 = (float)(rng & 0xFFFFFF) / (float)0x1000000;
        if (u1 < 1e-10f) u1 = 1e-10f;
        data[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        // 5% outliers
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        if ((rng & 0xFF) < 13) data[i] *= 10.0f;
    }

    // Create cache and load data (this applies pre-rotation + Q4 quantization)
    eakv_cache_t *cache = eakv_cache_create(nl, nh, hd, sl);
    if (!cache) { fprintf(stderr, "Failed to create cache\n"); return 1; }

    int rc = eakv_cache_load_raw(cache, data, sl);
    check("load_raw succeeds", rc == 0, "eakv_cache_load_raw failed");

    // Generate query vectors
    float *queries = malloc((size_t)nh * hd * sizeof(float));
    for (int i = 0; i < nh * hd; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        queries[i] = (float)(rng & 0xFFFF) / 32768.0f - 1.0f;
    }

    // Compute attention scores via eakv (pre-rotated Q4)
    float *scores_q4 = malloc((size_t)nh * sl * sizeof(float));
    eakv_attention_scores(cache, queries, 0, nh, nh, scores_q4);

    // Compute reference f32 attention scores
    // K data is at data[0..nh*sl*hd] (layer 0, kv=0)
    float *scores_ref = malloc((size_t)nh * sl * sizeof(float));
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < nh; h++) {
        const float *q = queries + h * hd;
        for (int t = 0; t < sl; t++) {
            const float *k = data + h * sl * hd + t * hd;
            scores_ref[h * sl + t] = dot_f32(q, k, hd) * scale;
        }
    }

    // Compare: Q4 scores should be close to f32 reference
    float max_score_err = 0;
    double score_mse = 0;
    for (int i = 0; i < nh * sl; i++) {
        float e = fabsf(scores_q4[i] - scores_ref[i]);
        if (e > max_score_err) max_score_err = e;
        score_mse += (double)e * e;
    }
    score_mse /= (nh * sl);
    float score_rmse = sqrtf((float)score_mse);

    char detail[128];
    snprintf(detail, sizeof(detail), "RMSE=%.4f max=%.4f", score_rmse, max_score_err);
    check("attention scores close to f32 reference", score_rmse < 0.5f, detail);
    printf("    scores RMSE=%.4f  max_err=%.4f\n", score_rmse, max_score_err);

    // Compute attention output via eakv
    float *weights = malloc((size_t)nh * sl * sizeof(float));
    float wsum = 1.0f / sl;
    for (int i = 0; i < nh * sl; i++) weights[i] = wsum;

    float *output_q4 = calloc(nh * hd, sizeof(float));
    eakv_attention_output(cache, weights, 0, nh, nh, output_q4);

    // Reference f32 output: weighted sum of V vectors
    // V data is at data[nh*sl*hd .. 2*nh*sl*hd] (layer 0, kv=1)
    float *v_data = data + nh * sl * hd;
    float *output_ref = calloc(nh * hd, sizeof(float));
    for (int h = 0; h < nh; h++) {
        for (int t = 0; t < sl; t++) {
            const float *v = v_data + h * sl * hd + t * hd;
            float w = weights[h * sl + t];
            for (int d = 0; d < hd; d++)
                output_ref[h * hd + d] += w * v[d];
        }
    }

    float max_out_err = 0;
    double out_mse = 0;
    for (int i = 0; i < nh * hd; i++) {
        float e = fabsf(output_q4[i] - output_ref[i]);
        if (e > max_out_err) max_out_err = e;
        out_mse += (double)e * e;
    }
    out_mse /= (nh * hd);
    float out_rmse = sqrtf((float)out_mse);

    snprintf(detail, sizeof(detail), "RMSE=%.4f max=%.4f", out_rmse, max_out_err);
    check("attention output close to f32 reference", out_rmse < 0.5f, detail);
    printf("    output RMSE=%.4f  max_err=%.4f\n", out_rmse, max_out_err);

    // Ranking preservation: top-scoring positions should match
    for (int h = 0; h < nh; h++) {
        int top_ref = 0, top_q4 = 0;
        for (int t = 1; t < sl; t++) {
            if (scores_ref[h * sl + t] > scores_ref[h * sl + top_ref]) top_ref = t;
            if (scores_q4[h * sl + t] > scores_q4[h * sl + top_q4]) top_q4 = t;
        }
        char name[64];
        snprintf(name, sizeof(name), "head %d top-1 position matches", h);
        snprintf(detail, sizeof(detail), "ref=%d q4=%d", top_ref, top_q4);
        check(name, top_ref == top_q4, detail);
    }

    printf("\n=== Results: %d passed, %d failed ===\n", pass, fail);

    free(data); free(queries); free(scores_q4); free(scores_ref);
    free(weights); free(output_q4); free(output_ref);
    eakv_cache_free(cache);
    return fail > 0 ? 1 : 0;
}
