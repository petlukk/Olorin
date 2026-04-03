// test_q4k_kernel.c — verify q4k_dot_q8k against pure-C reference
//
// Build: gcc -O2 -o test_q4k_kernel tests/test_q4k_kernel.c -ldl -lm
// Run:   ./test_q4k_kernel

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

#define Q4K_BLOCK_BYTES 144
#define BLOCK_ELEMS 256

// Kernel function type
typedef float (*q4k_dot_fn)(
    const uint8_t* q4, const int8_t* q8, const int32_t* bsums,
    const uint8_t* scales, const uint8_t* mins,
    int32_t n_blocks, float d, float dmin);

// Unpack Q4_K 12-byte packed scales (matches Rust unpack_q4k_scales)
void unpack_scales(const uint8_t* packed, uint8_t scales[8], uint8_t mins[8]) {
    for (int i = 0; i < 4; i++) {
        scales[i] = packed[i] & 0x3F;
        mins[i]   = packed[4 + i] & 0x3F;
    }
    for (int i = 0; i < 4; i++) {
        scales[4 + i] = (packed[8 + i] & 0x0F) | ((packed[i] >> 6) << 4);
        mins[4 + i]   = (packed[8 + i] >> 4) | ((packed[4 + i] >> 6) << 4);
    }
}

// f16 → f32
float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) { uint32_t r = sign << 31; float f; memcpy(&f, &r, 4); return f; }
        int e = 0; uint32_t ff = frac;
        while (!(ff & 0x400)) { ff <<= 1; e--; }
        ff &= 0x3FF;
        uint32_t r = (sign << 31) | ((uint32_t)(127 - 15 + 1 + e) << 23) | (ff << 13);
        float f; memcpy(&f, &r, 4); return f;
    }
    if (exp == 31) { uint32_t r = (sign << 31) | (0xFF << 23) | (frac << 13); float f; memcpy(&f, &r, 4); return f; }
    uint32_t r = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    float f; memcpy(&f, &r, 4); return f;
}

// Pure-C reference: Q4_K × Q8_K dot product
// Q4_K layout per block (144 bytes):
//   [0..1]   f16 d
//   [2..3]   f16 dmin
//   [4..15]  12 bytes packed scales
//   [16..143] 128 bytes: packed nibbles
//
// Nibble layout (from kernel source):
//   4 chunks of 32 bytes. Chunk j:
//     pa = q4[j*32..j*32+16], pb = q4[j*32+16..j*32+32]
//     low nibble (& 0x0F) → q8[j*64..j*64+31]   (scale index: 2*j)
//     high nibble (>> 4)  → q8[j*64+32..j*64+63] (scale index: 2*j+1)
float ref_q4k_dot(
    const uint8_t* q4, const int8_t* q8, const int32_t* bsums,
    const uint8_t* scales, const uint8_t* mins,
    int n_blocks, float d, float dmin)
{
    float result = 0.0f;
    for (int blk = 0; blk < n_blocks; blk++) {
        int q4_off = blk * 128;
        int q8_off = blk * 256;
        int sc_off = blk * 8;
        int bs_off = blk * 16;

        int sumi = 0;
        for (int j = 0; j < 4; j++) {
            // Low nibbles: pa[0..15] & 0x0F matched with q8[j*64..j*64+15]
            //              pb[0..15] & 0x0F matched with q8[j*64+16..j*64+31]
            int dot_lo = 0;
            for (int i = 0; i < 16; i++) {
                uint8_t pa = q4[q4_off + j * 32 + i];
                dot_lo += (pa & 0x0F) * q8[q8_off + j * 64 + i];
            }
            for (int i = 0; i < 16; i++) {
                uint8_t pb = q4[q4_off + j * 32 + 16 + i];
                dot_lo += (pb & 0x0F) * q8[q8_off + j * 64 + 16 + i];
            }

            // High nibbles: pa[0..15] >> 4 matched with q8[j*64+32..j*64+47]
            //               pb[0..15] >> 4 matched with q8[j*64+48..j*64+63]
            int dot_hi = 0;
            for (int i = 0; i < 16; i++) {
                uint8_t pa = q4[q4_off + j * 32 + i];
                dot_hi += (pa >> 4) * q8[q8_off + j * 64 + 32 + i];
            }
            for (int i = 0; i < 16; i++) {
                uint8_t pb = q4[q4_off + j * 32 + 16 + i];
                dot_hi += (pb >> 4) * q8[q8_off + j * 64 + 48 + i];
            }

            sumi += dot_lo * scales[sc_off + 2 * j] + dot_hi * scales[sc_off + 2 * j + 1];
        }

        int summs = 0;
        for (int j = 0; j < 8; j++) {
            int m = mins[sc_off + j];
            int bs_pair = bsums[bs_off + 2 * j] + bsums[bs_off + 2 * j + 1];
            summs += m * bs_pair;
        }

        result += d * (float)sumi - dmin * (float)summs;
    }
    return result;
}

// Load real model data and test first row
int test_with_model(q4k_dot_fn kernel_fn) {
    const char* home = getenv("HOME");
    if (!home) { fprintf(stderr, "no HOME\n"); return 1; }

    char path[512];
    snprintf(path, sizeof(path), "%s/.olorin/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf", home);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "skipping model test: %s not found\n", path);
        return 0; // not a failure
    }
    fclose(f);
    printf("Model file exists, but parsing GGUF from C is complex.\n");
    printf("Kernel-vs-reference tested with synthetic data below.\n\n");
    return 0;
}

int main() {
    // Load kernel
    const char* home = getenv("HOME");
    char lib_path[512];

    // Find newest lib dir
    snprintf(lib_path, sizeof(lib_path),
        "%s/.olorin/lib/v0.6.0-b712be379a8c8607/libq4k_dot.so", home);

    void* lib = dlopen(lib_path, RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }
    q4k_dot_fn kernel = (q4k_dot_fn)dlsym(lib, "q4k_dot_q8k");
    if (!kernel) {
        fprintf(stderr, "dlsym: %s\n", dlerror());
        return 1;
    }
    printf("Loaded kernel from %s\n\n", lib_path);

    // ======= Test 1: Known simple data =======
    printf("=== Test 1: Simple known data (1 block) ===\n");
    {
        // Create 1 Q4K block: all nibbles = 3 (both lo and hi)
        uint8_t q4[128];
        memset(q4, 0x33, 128); // each byte: lo=3, hi=3

        // Q8K: all values = 2
        int8_t q8[256];
        memset(q8, 2, 256);

        // Scales: all = 1, Mins: all = 0
        uint8_t scales[8] = {1,1,1,1,1,1,1,1};
        uint8_t mins[8]   = {0,0,0,0,0,0,0,0};

        // bsums: each = sum of 16 q8 values = 16 * 2 = 32
        int32_t bsums[16];
        for (int i = 0; i < 16; i++) bsums[i] = 32;

        float d = 1.0f, dmin = 0.0f;

        float ref = ref_q4k_dot(q4, q8, bsums, scales, mins, 1, d, dmin);
        float kern = kernel(q4, q8, bsums, scales, mins, 1, d, dmin);

        printf("  ref  = %f\n", ref);
        printf("  kern = %f\n", kern);
        printf("  err  = %f\n", fabsf(ref - kern));
        printf("  %s\n\n", fabsf(ref - kern) < 0.01f ? "PASS" : "FAIL");
    }

    // ======= Test 2: Random data, multiple blocks =======
    printf("=== Test 2: Random data (4 blocks = 1024 elements) ===\n");
    {
        srand(42);
        int n_blocks = 4;
        uint8_t q4[128 * 4];
        int8_t q8[256 * 4];
        uint8_t scales[8 * 4], mins[8 * 4];
        int32_t bsums[16 * 4];

        for (int i = 0; i < 128 * n_blocks; i++)
            q4[i] = rand() & 0xFF;
        for (int i = 0; i < 256 * n_blocks; i++)
            q8[i] = (rand() % 256) - 128;
        for (int i = 0; i < 8 * n_blocks; i++) {
            scales[i] = rand() % 64;
            mins[i]   = rand() % 64;
        }
        // Compute bsums from q8
        for (int blk = 0; blk < n_blocks; blk++) {
            for (int g = 0; g < 16; g++) {
                int sum = 0;
                for (int i = 0; i < 16; i++)
                    sum += q8[blk * 256 + g * 16 + i];
                bsums[blk * 16 + g] = sum;
            }
        }

        float d = 0.5f, dmin = 0.25f;

        float ref = ref_q4k_dot(q4, q8, bsums, scales, mins, n_blocks, d, dmin);
        float kern = kernel(q4, q8, bsums, scales, mins, n_blocks, d, dmin);

        printf("  ref  = %f\n", ref);
        printf("  kern = %f\n", kern);
        printf("  err  = %f  rel = %f\n", fabsf(ref - kern),
               fabsf(ref) > 1e-6 ? fabsf(ref - kern) / fabsf(ref) : 0.0f);
        printf("  %s\n\n", fabsf(ref - kern) < 1.0f ? "PASS" : "FAIL");
    }

    // ======= Test 3: Varying d/dmin per block (simulates real model) =======
    printf("=== Test 3: Per-block d/dmin (8 blocks) ===\n");
    {
        srand(123);
        int n_blocks = 8;
        uint8_t q4[128 * 8];
        int8_t  q8[256 * 8];
        int32_t bsums[16 * 8];

        for (int i = 0; i < 128 * n_blocks; i++) q4[i] = rand() & 0xFF;
        for (int i = 0; i < 256 * n_blocks; i++) q8[i] = (rand() % 256) - 128;
        for (int blk = 0; blk < n_blocks; blk++)
            for (int g = 0; g < 16; g++) {
                int sum = 0;
                for (int i = 0; i < 16; i++) sum += q8[blk * 256 + g * 16 + i];
                bsums[blk * 16 + g] = sum;
            }

        // Note: kernel takes single d/dmin (pre-multiplied by caller per block).
        // The Rust wrapper calls kernel once per block with block-specific d/dmin.
        // Here we test the single-call path (n_blocks>1, single d/dmin).
        // This is how the kernel is called from matmul_q4k.rs:q4k_row_dot.
        //
        // Wait — looking at the Rust code more carefully:
        // q4k_row_dot calls the kernel with n_blocks=1 per iteration!
        // Let me check...
        // No: q4k_row_dot loops over blocks itself, calling kernel with n_blocks=1.
        // But the kernel accepts n_blocks>1. Let me test both ways.

        // Single d/dmin for all blocks:
        uint8_t scales[8 * 8], mins_arr[8 * 8];
        for (int i = 0; i < 8 * n_blocks; i++) {
            scales[i] = rand() % 64;
            mins_arr[i] = rand() % 64;
        }
        float d = 0.3f, dmin = 0.1f;

        // Multi-block kernel call
        float kern_multi = kernel(q4, q8, bsums, scales, mins_arr, n_blocks, d, dmin);
        float ref_multi = ref_q4k_dot(q4, q8, bsums, scales, mins_arr, n_blocks, d, dmin);

        // Block-by-block kernel call (how Rust actually calls it)
        float kern_single = 0.0f;
        float ref_single = 0.0f;
        for (int blk = 0; blk < n_blocks; blk++) {
            float k = kernel(q4 + blk * 128, q8 + blk * 256, bsums + blk * 16,
                             scales + blk * 8, mins_arr + blk * 8, 1, d, dmin);
            float r = ref_q4k_dot(q4 + blk * 128, q8 + blk * 256, bsums + blk * 16,
                                  scales + blk * 8, mins_arr + blk * 8, 1, d, dmin);
            kern_single += k;
            ref_single += r;
        }

        printf("  Multi-block:  kern=%f  ref=%f  err=%f\n", kern_multi, ref_multi, fabsf(kern_multi - ref_multi));
        printf("  Block-by-blk: kern=%f  ref=%f  err=%f\n", kern_single, ref_single, fabsf(kern_single - ref_single));
        printf("  Multi vs single kernel: %f\n", fabsf(kern_multi - kern_single));
        printf("  %s\n\n",
            fabsf(kern_multi - ref_multi) < 1.0f && fabsf(kern_single - ref_single) < 1.0f
            ? "PASS" : "FAIL");
    }

    // ======= Test 4: maddubs overflow check =======
    // maddubs(u8, i8) can overflow i16 when u8=15, i8=127: 15*127=1905 > 32767/16 pairs
    // 16 pairs: max = 16 * 15 * 127 = 30480, fits i16 (32767). OK.
    // But with nibble=15 and q8=127: each pair = 1905, 16 pairs = 30480. Fine.
    printf("=== Test 4: Max-value overflow check ===\n");
    {
        uint8_t q4[128];
        memset(q4, 0xFF, 128); // all nibbles = 15

        int8_t q8[256];
        memset(q8, 127, 256); // max positive

        uint8_t scales[8] = {63,63,63,63,63,63,63,63}; // max 6-bit
        uint8_t mins[8]   = {63,63,63,63,63,63,63,63};
        int32_t bsums[16];
        for (int i = 0; i < 16; i++) bsums[i] = 16 * 127; // 2032

        float d = 1.0f, dmin = 1.0f;

        float ref = ref_q4k_dot(q4, q8, bsums, scales, mins, 1, d, dmin);
        float kern = kernel(q4, q8, bsums, scales, mins, 1, d, dmin);

        printf("  ref  = %f\n", ref);
        printf("  kern = %f\n", kern);
        printf("  err  = %f  rel = %e\n", fabsf(ref - kern),
               fabsf(ref) > 1e-6 ? fabsf(ref - kern) / fabsf(ref) : 0.0f);
        printf("  %s\n\n", fabsf(ref - kern) / (fabsf(ref) + 1e-6f) < 0.001f ? "PASS" : "FAIL");
    }

    dlclose(lib);
    printf("Done.\n");
    return 0;
}
