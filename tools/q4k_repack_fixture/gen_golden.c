// gen_golden.c — generate the ggml-reference repack of a Q4K input slice.
//
// This is a one-shot, off-build tool. It produces the golden fixture
// (golden.bin) that tests/gemma4_batch_verify.rs::batch1_repack_q4k_bytes_match_ggml_golden
// compares olorin's q4k_repack_8x8 kernel against.
//
// The make_block_q4_Kx8 function below is a verbatim transcription of
// llama.cpp build 8685, ggml/src/ggml-cpu/repack.cpp lines 2836-2911 (the
// blck_size_interleave == 8 path), reduced to a self-contained C file. No
// ggml headers are pulled in — the structs are redeclared here so the
// generator builds with plain `cc -std=c11`.
//
// If ggml ever changes its repack layout, copy the new version verbatim and
// regenerate the fixture (see ../regenerate.sh) and bump the research note
// docs/superpowers/research/2026-04-08-ggml-q4k-8x8-format.md.
//
// Usage:
//   cc -O2 -std=c11 -o gen_golden gen_golden.c
//   ./gen_golden <nrows> <ncols> <input.bin> <golden.bin>
//
// Constraints: nrows % 8 == 0, ncols % 256 == 0.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define QK_K          256
#define K_SCALE_SIZE  12

typedef uint16_t ggml_half;

typedef struct {
    ggml_half d;
    ggml_half dmin;
    uint8_t   scales[K_SCALE_SIZE];
    uint8_t   qs[QK_K / 2];
} block_q4_K;

_Static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

typedef struct {
    ggml_half d[8];
    ggml_half dmin[8];
    uint8_t   scales[96];
    uint8_t   qs[1024];
} block_q4_Kx8;

_Static_assert(sizeof(block_q4_Kx8) == 1152, "block_q4_Kx8 must be 1152 bytes");

// Verbatim from ggml repack.cpp::make_block_q4_Kx8 (blck_size_interleave=8 only).
static block_q4_Kx8 make_block_q4_Kx8(block_q4_K * in) {
    block_q4_Kx8 out;
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }
    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].dmin;
    }

    const int end = QK_K * 4 / 8; // 128

    for (int i = 0; i < end; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * 8;
        int dst_offset = i * 8;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], 8);
        memcpy(&out.qs[dst_offset], &elems, 8);
    }

    uint8_t s[8], m[8];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i]     & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }
        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i]     & 192) >> 2) | (in[j].scales[i + 8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
        }
        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    return out;
}

// Verbatim outer loop from ggml repack.cpp::repack_q4_K_to_q4_K_8_bl
// (blck_size_interleave=8), minus the ggml_tensor wrapping.
static void repack(const block_q4_K * src, block_q4_Kx8 * dst, int nrow, int nblocks) {
    block_q4_K dst_tmp[8];
    for (int b = 0; b < nrow; b += 8) {
        for (int x = 0; x < nblocks; x++) {
            for (int i = 0; i < 8; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx8(dst_tmp);
        }
        src += 8 * nblocks;
    }
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <nrows> <ncols> <input.bin> <golden.bin>\n", argv[0]);
        return 2;
    }
    int nrows = atoi(argv[1]);
    int ncols = atoi(argv[2]);
    if (nrows <= 0 || (nrows % 8) != 0) { fprintf(stderr, "nrows must be > 0 and %% 8\n"); return 2; }
    if (ncols <= 0 || (ncols % 256) != 0) { fprintf(stderr, "ncols must be > 0 and %% 256\n"); return 2; }

    int nblocks = ncols / 256;
    size_t row_bytes = (size_t)nblocks * sizeof(block_q4_K);
    size_t total = (size_t)nrows * row_bytes;

    FILE * fi = fopen(argv[3], "rb");
    if (!fi) { perror(argv[3]); return 1; }
    void * input = malloc(total);
    void * output = malloc(total);
    if (!input || !output) { fprintf(stderr, "OOM\n"); return 1; }
    if (fread(input, 1, total, fi) != total) {
        fprintf(stderr, "%s: short read (expected %zu)\n", argv[3], total);
        return 1;
    }
    fclose(fi);

    repack((const block_q4_K *)input, (block_q4_Kx8 *)output, nrows, nblocks);

    FILE * fo = fopen(argv[4], "wb");
    if (!fo) { perror(argv[4]); return 1; }
    if (fwrite(output, 1, total, fo) != total) { perror("write"); return 1; }
    fclose(fo);

    fprintf(stderr, "wrote %zu bytes to %s\n", total, argv[4]);
    free(input); free(output);
    return 0;
}
