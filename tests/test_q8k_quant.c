// test_q8k_quant.c — verify quant_f32_q8k kernel against reference
//
// Build: gcc -O2 -o test_q8k_quant tests/test_q8k_quant.c -ldl -lm
// Run:   ./test_q8k_quant

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

// quant_f32_q8k(input: *f32, qs: *mut i8, d: *mut f32, bsums: *mut i32, n: i32)
typedef void (*quant_q8k_fn)(
    const float* input, int8_t* qs, float* d, int32_t* bsums, int32_t n);

// Reference Q8_K quantization:
// Per 256-element block:
//   d = max(|x|) / 127
//   qs[i] = round(x[i] / d)
//   bsums[g] = sum of qs[g*16..g*16+15] for g=0..15
void ref_quant_q8k(const float* input, int8_t* qs, float* d_out, int32_t* bsums, int n) {
    int n_blocks = n / 256;
    for (int blk = 0; blk < n_blocks; blk++) {
        const float* x = input + blk * 256;
        int8_t* q = qs + blk * 256;

        // Find max abs
        float amax = 0.0f;
        for (int i = 0; i < 256; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }

        float d = amax / 127.0f;
        d_out[blk] = d;

        float id = (d > 1e-10f) ? 127.0f / amax : 0.0f;
        for (int i = 0; i < 256; i++) {
            float v = x[i] * id;
            int vi = (int)roundf(v);
            if (vi > 127) vi = 127;
            if (vi < -127) vi = -127; // note: Q8_K range is -127..127 (not -128)
            q[i] = (int8_t)vi;
        }

        // Compute bsums
        for (int g = 0; g < 16; g++) {
            int32_t sum = 0;
            for (int i = 0; i < 16; i++) {
                sum += q[blk * 0 + g * 16 + i]; // qs is already offset
            }
            bsums[blk * 16 + g] = sum;
        }
    }
}

int main() {
    const char* home = getenv("HOME");
    char lib_path[512];
    snprintf(lib_path, sizeof(lib_path),
        "%s/.olorin/lib/v0.6.0-b712be379a8c8607/libq4k_quant.so", home);

    void* lib = dlopen(lib_path, RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    quant_q8k_fn kernel = (quant_q8k_fn)dlsym(lib, "quant_f32_q8k");
    if (!kernel) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }
    printf("Loaded quant kernel from %s\n\n", lib_path);

    // Test 1: Simple ramp
    printf("=== Test 1: Ramp 0..255 ===\n");
    {
        float input[256];
        for (int i = 0; i < 256; i++) input[i] = (float)i - 128.0f;

        int8_t kern_qs[272] = {0}; // +16 padding (Rust does this)
        float kern_d[1] = {0};
        int32_t kern_bsums[16] = {0};

        int8_t ref_qs[256] = {0};
        float ref_d[1] = {0};
        int32_t ref_bsums[16] = {0};

        kernel(input, kern_qs, kern_d, kern_bsums, 256);
        ref_quant_q8k(input, ref_qs, ref_d, ref_bsums, 256);

        printf("  d:  kern=%f  ref=%f\n", kern_d[0], ref_d[0]);

        int qs_match = 1, qs_max_diff = 0;
        for (int i = 0; i < 256; i++) {
            int diff = abs(kern_qs[i] - ref_qs[i]);
            if (diff > qs_max_diff) qs_max_diff = diff;
            if (diff > 1) qs_match = 0; // allow rounding diff of 1
        }
        printf("  qs:  max_diff=%d  %s\n", qs_max_diff, qs_max_diff <= 1 ? "OK" : "MISMATCH");

        int bs_match = 1, bs_max_diff = 0;
        for (int i = 0; i < 16; i++) {
            int diff = abs(kern_bsums[i] - ref_bsums[i]);
            if (diff > bs_max_diff) bs_max_diff = diff;
            if (diff > 16) bs_match = 0; // rounding can accumulate
        }
        printf("  bsums: max_diff=%d  %s\n", bs_max_diff, bs_match ? "OK" : "MISMATCH");
        printf("  kern bsums: ");
        for (int i = 0; i < 16; i++) printf("%d ", kern_bsums[i]);
        printf("\n  ref  bsums: ");
        for (int i = 0; i < 16; i++) printf("%d ", ref_bsums[i]);
        printf("\n\n");
    }

    // Test 2: Random data, 4 blocks (1024 elements)
    printf("=== Test 2: Random data (4 blocks) ===\n");
    {
        srand(42);
        int n = 1024;
        int n_blocks = n / 256;
        float input[1024];
        for (int i = 0; i < n; i++) input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        int8_t kern_qs[1040] = {0};
        float kern_d[4] = {0};
        int32_t kern_bsums[64] = {0};

        int8_t ref_qs[1024] = {0};
        float ref_d[4] = {0};
        int32_t ref_bsums[64] = {0};

        kernel(input, kern_qs, kern_d, kern_bsums, n);
        ref_quant_q8k(input, ref_qs, ref_d, ref_bsums, n);

        int total_qs_diff = 0, max_qs_diff = 0;
        for (int i = 0; i < n; i++) {
            int diff = abs(kern_qs[i] - ref_qs[i]);
            total_qs_diff += diff;
            if (diff > max_qs_diff) max_qs_diff = diff;
        }
        printf("  d values: ");
        for (int b = 0; b < n_blocks; b++)
            printf("kern=%.6f ref=%.6f  ", kern_d[b], ref_d[b]);
        printf("\n");
        printf("  qs: max_diff=%d  total_diff=%d  %s\n", max_qs_diff, total_qs_diff,
            max_qs_diff <= 1 ? "OK" : "MISMATCH");

        int max_bs_diff = 0;
        for (int i = 0; i < n_blocks * 16; i++) {
            int diff = abs(kern_bsums[i] - ref_bsums[i]);
            if (diff > max_bs_diff) max_bs_diff = diff;
        }
        printf("  bsums: max_diff=%d  %s\n", max_bs_diff, max_bs_diff <= 16 ? "OK" : "MISMATCH");
    }

    // Test 3: End-to-end: quant then dot, compare with f32 dot
    printf("\n=== Test 3: End-to-end quant→dot vs f32 dot ===\n");
    {
        // Load q4k_dot kernel too
        char dot_path[512];
        snprintf(dot_path, sizeof(dot_path),
            "%s/.olorin/lib/v0.6.0-b712be379a8c8607/libq4k_dot.so", home);
        void* dot_lib = dlopen(dot_path, RTLD_NOW);
        if (!dot_lib) { fprintf(stderr, "dlopen dot: %s\n", dlerror()); return 1; }
        typedef float (*q4k_dot_fn)(const uint8_t*, const int8_t*, const int32_t*,
            const uint8_t*, const uint8_t*, int32_t, float, float);
        q4k_dot_fn dot_fn = (q4k_dot_fn)dlsym(dot_lib, "q4k_dot_q8k");

        srand(99);
        int n = 256; // 1 block

        // Create Q4K weight block
        uint8_t w_block[144];
        // f16 d = 0.5, dmin = 0.1
        uint16_t d_f16 = 0x3800; // f16 for 0.5
        uint16_t dm_f16 = 0x2E66; // f16 for ~0.1
        memcpy(w_block, &d_f16, 2);
        memcpy(w_block + 2, &dm_f16, 2);
        // Packed scales: all scales=4, all mins=2
        memset(w_block + 4, 0x04, 4);  // scales[0..3] = 4
        memset(w_block + 8, 0x02, 4);  // mins[0..3] = 2
        memset(w_block + 12, 0x00, 4); // high bits = 0
        // Random nibbles
        for (int i = 0; i < 128; i++) w_block[16 + i] = rand() & 0xFF;

        // Create f32 activations
        float x[256];
        for (int i = 0; i < 256; i++) x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        // Quantize to Q8K
        int8_t q8[272] = {0};
        float q8_d[1] = {0};
        int32_t bsums[16] = {0};
        kernel(x, q8, q8_d, bsums, 256);

        // Unpack scales
        uint8_t scales[8], mins[8];
        // (reuse unpack from test_q4k_kernel.c)
        for (int i = 0; i < 4; i++) {
            scales[i] = w_block[4+i] & 0x3F;
            mins[i] = w_block[8+i] & 0x3F;
        }
        for (int i = 0; i < 4; i++) {
            scales[4+i] = (w_block[12+i] & 0x0F) | ((w_block[4+i] >> 6) << 4);
            mins[4+i] = (w_block[12+i] >> 4) | ((w_block[8+i] >> 6) << 4);
        }

        // Dequant weight to f32
        float w_f32[256];
        float wd = 0.5f;   // approximate
        float wdm = 0.1f;
        uint8_t* qs_ptr = w_block + 16;
        for (int i = 0; i < 256; i++) {
            int byte_idx = i / 2;
            int nibble = (i % 2 == 0) ? (qs_ptr[byte_idx] & 0x0F) : (qs_ptr[byte_idx] >> 4);
            int sb = i / 32;
            w_f32[i] = wd * scales[sb] * nibble - wdm * mins[sb];
        }

        // f32 reference dot
        float f32_dot = 0.0f;
        for (int i = 0; i < 256; i++) f32_dot += w_f32[i] * x[i];

        // Kernel dot
        float d_val = 0.5f * q8_d[0]; // pre-multiply
        float dm_val = 0.1f * q8_d[0];
        float kern_dot = dot_fn(w_block + 16, q8, bsums, scales, mins, 1, d_val, dm_val);

        printf("  f32 dot:  %f\n", f32_dot);
        printf("  kern dot: %f\n", kern_dot);
        printf("  err: %f  rel: %e\n", fabsf(f32_dot - kern_dot),
            fabsf(f32_dot) > 1e-6 ? fabsf(f32_dot - kern_dot) / fabsf(f32_dot) : 0.0);

        dlclose(dot_lib);
    }

    dlclose(lib);
    printf("\nDone.\n");
    return 0;
}
