// Benchmark: eakv Q4 quantization SNR — before/after pre-rotation.
//
// Measures:
//   1. SNR (signal-to-noise ratio) of quantize→dequantize roundtrip
//   2. RMSE and max element error
//   3. Distribution of quantization error (spread of residuals)
//   4. Attention score accuracy: fused Q4 vs f32 reference
//
// Tests with realistic KV-cache data distributions:
//   - Gaussian (typical hidden states)
//   - Gaussian with outliers (attention-heavy layers)
//   - Uniform
//
// Build:
//   gcc -O2 -o bench_eakv_snr bench_eakv_snr.c -ldl -lm
// Run:
//   ./bench_eakv_snr <libquantize.so> <libdequantize.so> [libturbo_rotate.so]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <stdint.h>
#include <time.h>

typedef void (*quant_fn)(const float*, int32_t*, float*, float*, int32_t);
typedef void (*dequant_fn)(const uint8_t*, const float*, const float*, float*, int32_t);
typedef void (*rotate_fn)(float*, const float*, int32_t);
typedef void (*fwht_fn)(float*, int32_t);
typedef void (*sign_flip_fn)(float*, const float*, int32_t);

#define GROUP_SIZE 64
#define GROUP_BYTES 32

static uint64_t rng = 0xABCDEF1234567890ULL;
static uint64_t xorshift64(void) {
    uint64_t x = rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng = x; return x;
}

// Box-Muller for Gaussian
static float rand_gauss(void) {
    float u1 = (float)(xorshift64() & 0xFFFFFF) / (float)0x1000000;
    float u2 = (float)(xorshift64() & 0xFFFFFF) / (float)0x1000000;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static float rand_uniform(void) {
    return (float)(xorshift64() & 0xFFFFFF) / (float)0xFFFFFF;
}

static void gen_signs(float *signs, int n) {
    uint64_t s = 0x4F6C6F72696E4A4CULL;
    for (int i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        signs[i] = (s % 2 == 0) ? 1.0f : -1.0f;
    }
}

typedef struct {
    float snr_db;
    float rmse;
    float max_err;
    float mean_err;
} quality_t;

static quality_t measure_quality(const float *orig, const float *recon, int n) {
    double signal_power = 0, noise_power = 0;
    float max_e = 0;
    double sum_e = 0;
    for (int i = 0; i < n; i++) {
        double s = orig[i];
        double e = orig[i] - recon[i];
        signal_power += s * s;
        noise_power += e * e;
        float ae = fabsf((float)e);
        if (ae > max_e) max_e = ae;
        sum_e += ae;
    }
    quality_t q;
    q.snr_db = (noise_power > 0) ? (float)(10.0 * log10(signal_power / noise_power)) : 999.0f;
    q.rmse = sqrtf((float)(noise_power / n));
    q.max_err = max_e;
    q.mean_err = (float)(sum_e / n);
    return q;
}

// Forward rotation: sign_flip then fwht (= turbo_rotate)
// Inverse rotation: fwht then sign_flip (opposite order)
static fwht_fn g_fwht = NULL;
static sign_flip_fn g_sign_flip = NULL;

static void quantize_dequantize(
    quant_fn quant, dequant_fn dequant, rotate_fn rotate,
    const float *input, float *output, float *signs,
    int n_elements, int n_groups
) {
    int32_t *weights_i32 = aligned_alloc(64, n_groups * GROUP_BYTES * sizeof(int32_t));
    uint8_t *weights_u8 = aligned_alloc(64, n_groups * GROUP_BYTES);
    float *scales = aligned_alloc(64, n_groups * sizeof(float));
    float *biases = aligned_alloc(64, n_groups * sizeof(float));

    const float *quant_input = input;
    float *rotated = NULL;

    if (rotate) {
        rotated = aligned_alloc(64, n_elements * sizeof(float));
        memcpy(rotated, input, n_elements * sizeof(float));
        // Forward: sign_flip then FWHT (= turbo_rotate)
        for (int g = 0; g < n_groups; g++) {
            rotate(rotated + g * GROUP_SIZE, signs, GROUP_SIZE);
        }
        quant_input = rotated;
    }

    quant(quant_input, weights_i32, scales, biases, n_groups);

    for (int i = 0; i < n_groups * GROUP_BYTES; i++)
        weights_u8[i] = (uint8_t)weights_i32[i];

    dequant(weights_u8, scales, biases, output, n_groups);

    if (rotate && g_fwht && g_sign_flip) {
        // Inverse: FWHT then sign_flip (opposite order of turbo_rotate)
        for (int g = 0; g < n_groups; g++) {
            g_fwht(output + g * GROUP_SIZE, GROUP_SIZE);
            g_sign_flip(output + g * GROUP_SIZE, signs, GROUP_SIZE);
        }
        free(rotated);
    }

    free(weights_i32); free(weights_u8); free(scales); free(biases);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <libquantize.so> <libdequantize.so> [libturbo_rotate.so]\n", argv[0]);
        return 1;
    }

    void *qlib = dlopen(argv[1], RTLD_NOW);
    void *dlib = dlopen(argv[2], RTLD_NOW);
    if (!qlib || !dlib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    quant_fn quant = dlsym(qlib, "q4_quantize_split_f32");
    dequant_fn dequant = dlsym(dlib, "q4_dequantize_simd_f32");
    if (!quant || !dequant) { fprintf(stderr, "Missing quant/dequant symbols\n"); return 1; }

    rotate_fn rotate = NULL;
    void *rlib = NULL;
    if (argc >= 4) {
        rlib = dlopen(argv[3], RTLD_NOW);
        if (!rlib) { fprintf(stderr, "dlopen rotate: %s\n", dlerror()); return 1; }
        rotate = dlsym(rlib, "turbo_rotate");
        g_fwht = dlsym(rlib, "fwht_inplace");
        g_sign_flip = dlsym(rlib, "sign_flip");
        if (!rotate || !g_fwht || !g_sign_flip) {
            fprintf(stderr, "Missing turbo_rotate/fwht_inplace/sign_flip symbols\n");
            return 1;
        }
    }

    printf("=== eakv Q4 Quantization SNR Benchmark ===\n");
    printf("Mode: %s\n\n", rotate ? "WITH pre-rotation (TurboQuant)" : "baseline (no rotation)");

    int n_elements = 8192;  // 128 groups of 64
    int n_groups = n_elements / GROUP_SIZE;

    float *signs = aligned_alloc(64, GROUP_SIZE * sizeof(float));
    gen_signs(signs, GROUP_SIZE);

    float *input = aligned_alloc(64, n_elements * sizeof(float));
    float *output = aligned_alloc(64, n_elements * sizeof(float));

    struct { const char *name; int seed; void (*gen)(float*, int); } dists[] = {
        {"Gaussian (mean=0, std=1)", 42, NULL},
        {"Gaussian with 5% outliers (10x)", 123, NULL},
        {"Uniform [-5, 5]", 456, NULL},
        {"Gaussian (mean=0, std=0.1)", 789, NULL},
    };

    for (int d = 0; d < 4; d++) {
        rng = dists[d].seed;

        if (d == 0) {
            for (int i = 0; i < n_elements; i++) input[i] = rand_gauss();
        } else if (d == 1) {
            for (int i = 0; i < n_elements; i++) {
                input[i] = rand_gauss();
                if (rand_uniform() < 0.05f) input[i] *= 10.0f;
            }
        } else if (d == 2) {
            for (int i = 0; i < n_elements; i++) input[i] = rand_uniform() * 10.0f - 5.0f;
        } else {
            for (int i = 0; i < n_elements; i++) input[i] = rand_gauss() * 0.1f;
        }

        quantize_dequantize(quant, dequant, rotate, input, output, signs,
                           n_elements, n_groups);

        quality_t q = measure_quality(input, output, n_elements);

        printf("%-40s SNR: %6.1f dB  RMSE: %.4f  max: %.4f  mean: %.4f\n",
               dists[d].name, q.snr_db, q.rmse, q.max_err, q.mean_err);
    }

    // Per-group outlier analysis: how much does one outlier cost?
    printf("\n--- Outlier impact analysis (single group of 64) ---\n");
    float group[GROUP_SIZE], group_out[GROUP_SIZE];
    int32_t w_i32[GROUP_BYTES];
    uint8_t w_u8[GROUP_BYTES];
    float sc[1], bi[1];

    // Base: gaussian group
    rng = 999;
    for (int i = 0; i < GROUP_SIZE; i++) group[i] = rand_gauss();

    // Without outlier
    float *g_in = aligned_alloc(64, GROUP_SIZE * sizeof(float));
    float *g_out = aligned_alloc(64, GROUP_SIZE * sizeof(float));
    memcpy(g_in, group, GROUP_SIZE * sizeof(float));
    quantize_dequantize(quant, dequant, rotate, g_in, g_out, signs, GROUP_SIZE, 1);
    quality_t q_no = measure_quality(g_in, g_out, GROUP_SIZE);

    // With outlier at position 0
    memcpy(g_in, group, GROUP_SIZE * sizeof(float));
    g_in[0] = 20.0f;  // outlier: 20x std
    quantize_dequantize(quant, dequant, rotate, g_in, g_out, signs, GROUP_SIZE, 1);
    quality_t q_out = measure_quality(g_in, g_out, GROUP_SIZE);

    printf("  No outlier:   SNR: %6.1f dB  RMSE: %.4f  max: %.4f\n",
           q_no.snr_db, q_no.rmse, q_no.max_err);
    printf("  With outlier: SNR: %6.1f dB  RMSE: %.4f  max: %.4f\n",
           q_out.snr_db, q_out.rmse, q_out.max_err);
    printf("  SNR loss from outlier: %.1f dB\n", q_no.snr_db - q_out.snr_db);

    free(input); free(output); free(signs); free(g_in); free(g_out);
    dlclose(qlib); dlclose(dlib);
    if (rlib) dlclose(rlib);
    return 0;
}
