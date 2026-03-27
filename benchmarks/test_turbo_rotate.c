// Test: turbo_rotate kernel correctness
//
// Verifies:
//   1. FWHT preserves L2 norm (orthogonal transform)
//   2. Sign-flip preserves L2 norm
//   3. turbo_rotate (sign_flip + FWHT) preserves L2 norm
//   4. FWHT is its own inverse (up to scaling)
//   5. Inner product preservation (JL guarantee)
//
// Build:
//   gcc -O2 -o test_turbo_rotate test_turbo_rotate.c -ldl -lm
// Run:
//   ./test_turbo_rotate <path-to-libturbo_rotate.so>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <stdint.h>

#define DIM 256
#define EPS 1e-4f

typedef void (*sign_flip_fn)(float *vec, const float *signs, int dim);
typedef void (*fwht_fn)(float *vec, int dim);
typedef void (*turbo_rotate_fn)(float *vec, const float *signs, int dim);

static uint64_t rng = 0xCAFEBABE12345678ULL;
static uint64_t xorshift64(void) {
    uint64_t x = rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng = x;
    return x;
}

static float rand_float(void) {
    return (float)(xorshift64() & 0xFFFFFF) / (float)0xFFFFFF;
}

static float l2_norm(const float *v, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

static float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void gen_signs(float *signs, int dim) {
    for (int i = 0; i < dim; i++) {
        signs[i] = (xorshift64() % 2) ? 1.0f : -1.0f;
    }
}

static int pass_count = 0;
static int fail_count = 0;

static void check(const char *name, int ok) {
    if (ok) { pass_count++; printf("  PASS: %s\n", name); }
    else    { fail_count++; printf("  FAIL: %s\n", name); }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <libturbo_rotate.so>\n", argv[0]);
        return 1;
    }

    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    sign_flip_fn sign_flip = dlsym(lib, "sign_flip");
    fwht_fn fwht_inplace = dlsym(lib, "fwht_inplace");
    turbo_rotate_fn turbo_rotate = dlsym(lib, "turbo_rotate");

    if (!sign_flip || !fwht_inplace || !turbo_rotate) {
        fprintf(stderr, "Missing symbols\n");
        dlclose(lib);
        return 1;
    }

    printf("=== turbo_rotate kernel tests ===\n\n");

    float *vec = aligned_alloc(64, DIM * sizeof(float));
    float *orig = aligned_alloc(64, DIM * sizeof(float));
    float *signs = aligned_alloc(64, DIM * sizeof(float));
    float *vec2 = aligned_alloc(64, DIM * sizeof(float));
    float *orig2 = aligned_alloc(64, DIM * sizeof(float));

    // --- Test 1: FWHT preserves L2 norm ---
    printf("[1] FWHT norm preservation\n");
    for (int t = 0; t < 5; t++) {
        for (int i = 0; i < DIM; i++) vec[i] = rand_float() * 2.0f - 1.0f;
        float before = l2_norm(vec, DIM);
        fwht_inplace(vec, DIM);
        float after = l2_norm(vec, DIM);
        float err = fabsf(before - after) / before;
        char name[64];
        snprintf(name, sizeof(name), "fwht norm trial %d (err=%.6f)", t, err);
        check(name, err < EPS);
    }

    // --- Test 2: sign_flip preserves L2 norm ---
    printf("\n[2] sign_flip norm preservation\n");
    gen_signs(signs, DIM);
    for (int i = 0; i < DIM; i++) vec[i] = rand_float() * 10.0f;
    float before_sf = l2_norm(vec, DIM);
    sign_flip(vec, signs, DIM);
    float after_sf = l2_norm(vec, DIM);
    float err_sf = fabsf(before_sf - after_sf) / before_sf;
    check("sign_flip norm", err_sf < EPS);

    // --- Test 3: turbo_rotate preserves L2 norm ---
    printf("\n[3] turbo_rotate norm preservation\n");
    for (int t = 0; t < 5; t++) {
        gen_signs(signs, DIM);
        for (int i = 0; i < DIM; i++) vec[i] = rand_float() * 5.0f - 2.5f;
        float before = l2_norm(vec, DIM);
        turbo_rotate(vec, signs, DIM);
        float after = l2_norm(vec, DIM);
        float err = fabsf(before - after) / before;
        char name[64];
        snprintf(name, sizeof(name), "turbo_rotate norm trial %d (err=%.6f)", t, err);
        check(name, err < EPS);
    }

    // --- Test 4: FWHT is self-inverse (apply twice = identity * dim) ---
    printf("\n[4] FWHT self-inverse\n");
    for (int i = 0; i < DIM; i++) { vec[i] = rand_float() * 3.0f; orig[i] = vec[i]; }
    // FWHT scales by 1/sqrt(dim), so two applications scale by 1/dim.
    // To invert: apply FWHT, then multiply by dim, then apply FWHT again... no.
    // Actually: FWHT with 1/sqrt(dim) scaling is exactly unitary.
    // H * H = I when H is normalized Hadamard.
    fwht_inplace(vec, DIM);
    fwht_inplace(vec, DIM);
    float max_err = 0.0f;
    for (int i = 0; i < DIM; i++) {
        float e = fabsf(vec[i] - orig[i]);
        if (e > max_err) max_err = e;
    }
    char name4[64];
    snprintf(name4, sizeof(name4), "fwht double-apply = identity (max_err=%.6f)", max_err);
    check(name4, max_err < 0.01f);

    // --- Test 5: Inner product preservation (JL core property) ---
    printf("\n[5] Inner product preservation after turbo_rotate\n");
    gen_signs(signs, DIM);
    for (int t = 0; t < 10; t++) {
        for (int i = 0; i < DIM; i++) {
            vec[i] = rand_float() * 4.0f - 2.0f;
            orig[i] = vec[i];
            vec2[i] = rand_float() * 4.0f - 2.0f;
            orig2[i] = vec2[i];
        }
        float dot_before = dot(orig, orig2, DIM);
        turbo_rotate(vec, signs, DIM);
        turbo_rotate(vec2, signs, DIM);
        float dot_after = dot(vec, vec2, DIM);
        float rel_err = (fabsf(dot_before) > 1e-6f)
            ? fabsf(dot_before - dot_after) / fabsf(dot_before)
            : fabsf(dot_before - dot_after);
        char name[80];
        snprintf(name, sizeof(name),
            "inner product trial %d (before=%.3f after=%.3f err=%.6f)", t,
            dot_before, dot_after, rel_err);
        check(name, rel_err < EPS);
    }

    // --- Test 6: Cosine similarity preservation ---
    printf("\n[6] Cosine similarity preservation\n");
    gen_signs(signs, DIM);
    for (int t = 0; t < 5; t++) {
        for (int i = 0; i < DIM; i++) {
            vec[i] = rand_float() * 4.0f - 2.0f;
            orig[i] = vec[i];
            vec2[i] = rand_float() * 4.0f - 2.0f;
            orig2[i] = vec2[i];
        }
        float cos_before = dot(orig, orig2, DIM) /
            (l2_norm(orig, DIM) * l2_norm(orig2, DIM));
        turbo_rotate(vec, signs, DIM);
        turbo_rotate(vec2, signs, DIM);
        float cos_after = dot(vec, vec2, DIM) /
            (l2_norm(vec, DIM) * l2_norm(vec2, DIM));
        float err = fabsf(cos_before - cos_after);
        char name[80];
        snprintf(name, sizeof(name),
            "cosine sim trial %d (before=%.4f after=%.4f err=%.6f)", t,
            cos_before, cos_after, err);
        check(name, err < EPS);
    }

    printf("\n=== Results: %d passed, %d failed ===\n",
           pass_count, fail_count);

    free(vec); free(orig); free(signs); free(vec2); free(orig2);
    dlclose(lib);
    return fail_count > 0 ? 1 : 0;
}
