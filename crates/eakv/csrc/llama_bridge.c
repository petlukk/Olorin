/* llama.cpp bridge — parse state buffer into eakv cache.
 *
 * This file requires llama.cpp headers for ggml_fp16_to_fp32.
 * It is NOT compiled into the core libeakv — build separately
 * when linking against llama.cpp.
 */
#include "eakv_llama.h"
#include "eakv.h"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fp16_to_f32(const uint16_t *src, float *dst, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = ggml_fp16_to_fp32(src[i]);
}

int eakv_from_llama_state(const uint8_t *state_buf, size_t state_size,
                          int n_layers, int n_kv_heads, int head_dim,
                          int max_seq_len, eakv_cache_t **out) {
    if (!state_buf || !out || state_size < 8)
        return EAKV_ERR_INVALID;
    if (n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0)
        return EAKV_ERR_INVALID;

    const uint8_t *ptr = state_buf;
    const uint8_t *end = state_buf + state_size;

    /* Header: n_stream, cell_count */
    if (ptr + 8 > end) return EAKV_ERR_FORMAT;
    uint32_t n_stream = *(const uint32_t *)ptr; ptr += 4;
    uint32_t cell_count = *(const uint32_t *)ptr; ptr += 4;
    (void)n_stream;

    if (cell_count == 0 || (int)cell_count > max_seq_len)
        return EAKV_ERR_FORMAT;

    /* Skip cell metadata */
    for (uint32_t i = 0; i < cell_count; i++) {
        if (ptr + 8 > end) return EAKV_ERR_FORMAT;
        ptr += sizeof(int32_t);  /* pos */
        uint32_t n_seq_id = *(const uint32_t *)ptr; ptr += 4;
        ptr += n_seq_id * sizeof(int32_t);
        if (ptr > end) return EAKV_ERR_FORMAT;
    }

    /* v_trans, n_layer */
    if (ptr + 8 > end) return EAKV_ERR_FORMAT;
    uint32_t v_trans = *(const uint32_t *)ptr; ptr += 4;
    uint32_t file_n_layer = *(const uint32_t *)ptr; ptr += 4;

    if ((int)file_n_layer != n_layers)
        return EAKV_ERR_FORMAT;

    int n_embd_k_gqa = n_kv_heads * head_dim;
    int seq_len = (int)cell_count;

    /* Create cache */
    eakv_cache_t *cache = eakv_cache_create(n_layers, n_kv_heads, head_dim, max_seq_len);
    if (!cache) return EAKV_ERR_ALLOC;

    /* Temp buffer for one layer's transposed data: [head][pos][dim] */
    float *transposed = malloc((size_t)n_embd_k_gqa * seq_len * sizeof(float));
    float *f32_row = malloc((size_t)n_embd_k_gqa * sizeof(float));
    if (!transposed || !f32_row) {
        free(transposed); free(f32_row);
        eakv_cache_free(cache);
        return EAKV_ERR_ALLOC;
    }

    /* Extract K layers: header is [type: i32][size_row: u64] */
    for (int l = 0; l < n_layers; l++) {
        if (ptr + 12 > end) goto format_err;
        ptr += 4;  /* type (i32) */
        uint64_t size_row = *(const uint64_t *)ptr; ptr += 8;

        if (ptr + size_row * cell_count > end) goto format_err;

        const uint16_t *fp16_data = (const uint16_t *)ptr;

        /* Transpose [pos][head][dim] -> [head][pos][dim] */
        for (int t = 0; t < seq_len; t++) {
            fp16_to_f32(fp16_data + t * n_embd_k_gqa, f32_row, n_embd_k_gqa);
            for (int h = 0; h < n_kv_heads; h++)
                memcpy(transposed + (h * seq_len + t) * head_dim,
                       f32_row + h * head_dim,
                       head_dim * sizeof(float));
        }

        eakv_cache_append(cache, transposed, l, 0, seq_len);
        ptr += size_row * cell_count;
    }

    /* Extract V layers */
    if (v_trans) {
        /*
         * v_trans=1: header is [type: i32][size_el: u32][n_embd_v_gqa: u32]
         * Data is column-major: for each embedding dim j, cells are contiguous.
         * Layout in buffer: [j=0: cell0..cellN][j=1: cell0..cellN]...
         * We need [head][pos][dim] output.
         */
        for (int l = 0; l < n_layers; l++) {
            if (ptr + 12 > end) goto format_err;
            ptr += 4;  /* type (i32) */
            uint32_t size_el = *(const uint32_t *)ptr; ptr += 4;
            uint32_t n_embd_v = *(const uint32_t *)ptr; ptr += 4;

            size_t v_data_size = (size_t)n_embd_v * cell_count * size_el;
            if (ptr + v_data_size > end) goto format_err;

            /* Column-major F16 data: [dim][pos] */
            const uint16_t *fp16_data = (const uint16_t *)ptr;

            /* Reconstruct [head][pos][dim] */
            for (int h = 0; h < n_kv_heads; h++) {
                for (int t = 0; t < seq_len; t++) {
                    for (int d = 0; d < head_dim; d++) {
                        int embd_idx = h * head_dim + d;
                        uint16_t val = fp16_data[embd_idx * cell_count + t];
                        transposed[(h * seq_len + t) * head_dim + d] =
                            ggml_fp16_to_fp32(val);
                    }
                }
            }

            eakv_cache_append(cache, transposed, l, 1, seq_len);
            ptr += v_data_size;
        }
    } else {
        /* v_trans=0: same format as K — [type: i32][size_row: u64] */
        for (int l = 0; l < n_layers; l++) {
            if (ptr + 12 > end) goto format_err;
            ptr += 4;
            uint64_t size_row = *(const uint64_t *)ptr; ptr += 8;

            if (ptr + size_row * cell_count > end) goto format_err;

            const uint16_t *fp16_data = (const uint16_t *)ptr;

            for (int t = 0; t < seq_len; t++) {
                fp16_to_f32(fp16_data + t * n_embd_k_gqa, f32_row, n_embd_k_gqa);
                for (int h = 0; h < n_kv_heads; h++)
                    memcpy(transposed + (h * seq_len + t) * head_dim,
                           f32_row + h * head_dim,
                           head_dim * sizeof(float));
            }

            eakv_cache_append(cache, transposed, l, 1, seq_len);
            ptr += size_row * cell_count;
        }
    }

    eakv_cache_advance(cache, seq_len);

    free(transposed);
    free(f32_row);
    *out = cache;
    return EAKV_OK;

format_err:
    free(transposed);
    free(f32_row);
    eakv_cache_free(cache);
    return EAKV_ERR_FORMAT;
}

int eakv_from_llama_state_append(eakv_cache_t *cache,
                                  const uint8_t *state_buf, size_t state_size,
                                  int n_layers, int n_kv_heads, int head_dim,
                                  int start_pos) {
    if (!cache || !state_buf || state_size < 8)
        return EAKV_ERR_INVALID;
    if (start_pos < 0 || start_pos != eakv_cache_seq_len(cache))
        return EAKV_ERR_INVALID;

    const uint8_t *ptr = state_buf;
    const uint8_t *end = state_buf + state_size;

    if (ptr + 8 > end) return EAKV_ERR_FORMAT;
    ptr += 4; /* n_stream */
    uint32_t cell_count = *(const uint32_t *)ptr; ptr += 4;

    if ((int)cell_count <= start_pos)
        return EAKV_OK; /* nothing new */

    int new_tokens = (int)cell_count - start_pos;
    int n_embd_k_gqa = n_kv_heads * head_dim;

    /* Skip cell metadata */
    for (uint32_t i = 0; i < cell_count; i++) {
        if (ptr + 8 > end) return EAKV_ERR_FORMAT;
        ptr += sizeof(int32_t);
        uint32_t n_seq_id = *(const uint32_t *)ptr; ptr += 4;
        ptr += n_seq_id * sizeof(int32_t);
        if (ptr > end) return EAKV_ERR_FORMAT;
    }

    if (ptr + 8 > end) return EAKV_ERR_FORMAT;
    uint32_t v_trans = *(const uint32_t *)ptr; ptr += 4;
    uint32_t file_n_layer = *(const uint32_t *)ptr; ptr += 4;
    if ((int)file_n_layer != n_layers) return EAKV_ERR_FORMAT;

    float *transposed = malloc((size_t)n_embd_k_gqa * new_tokens * sizeof(float));
    float *f32_row = malloc((size_t)n_embd_k_gqa * sizeof(float));
    if (!transposed || !f32_row) {
        free(transposed); free(f32_row);
        return EAKV_ERR_ALLOC;
    }

    /* K layers */
    for (int l = 0; l < n_layers; l++) {
        if (ptr + 12 > end) goto format_err;
        ptr += 4;
        uint64_t size_row = *(const uint64_t *)ptr; ptr += 8;
        if (ptr + size_row * cell_count > end) goto format_err;

        const uint16_t *fp16_data = (const uint16_t *)ptr;
        for (int t = 0; t < new_tokens; t++) {
            int src_pos = start_pos + t;
            for (int j = 0; j < n_embd_k_gqa; j++)
                f32_row[j] = ggml_fp16_to_fp32(fp16_data[src_pos * n_embd_k_gqa + j]);
            for (int h = 0; h < n_kv_heads; h++)
                memcpy(transposed + (h * new_tokens + t) * head_dim,
                       f32_row + h * head_dim,
                       head_dim * sizeof(float));
        }
        eakv_cache_append(cache, transposed, l, 0, new_tokens);
        ptr += size_row * cell_count;
    }

    /* V layers */
    if (v_trans) {
        for (int l = 0; l < n_layers; l++) {
            if (ptr + 12 > end) goto format_err;
            ptr += 4;
            uint32_t size_el = *(const uint32_t *)ptr; ptr += 4;
            uint32_t n_embd_v = *(const uint32_t *)ptr; ptr += 4;
            size_t v_data_size = (size_t)n_embd_v * cell_count * size_el;
            if (ptr + v_data_size > end) goto format_err;

            const uint16_t *fp16_data = (const uint16_t *)ptr;
            for (int h = 0; h < n_kv_heads; h++) {
                for (int t = 0; t < new_tokens; t++) {
                    int src_pos = start_pos + t;
                    for (int d = 0; d < head_dim; d++) {
                        int embd_idx = h * head_dim + d;
                        uint16_t val = fp16_data[embd_idx * cell_count + src_pos];
                        transposed[(h * new_tokens + t) * head_dim + d] =
                            ggml_fp16_to_fp32(val);
                    }
                }
            }
            eakv_cache_append(cache, transposed, l, 1, new_tokens);
            ptr += v_data_size;
        }
    } else {
        for (int l = 0; l < n_layers; l++) {
            if (ptr + 12 > end) goto format_err;
            ptr += 4;
            uint64_t size_row = *(const uint64_t *)ptr; ptr += 8;
            if (ptr + size_row * cell_count > end) goto format_err;

            const uint16_t *fp16_data = (const uint16_t *)ptr;
            for (int t = 0; t < new_tokens; t++) {
                int src_pos = start_pos + t;
                for (int j = 0; j < n_embd_k_gqa; j++)
                    f32_row[j] = ggml_fp16_to_fp32(fp16_data[src_pos * n_embd_k_gqa + j]);
                for (int h = 0; h < n_kv_heads; h++)
                    memcpy(transposed + (h * new_tokens + t) * head_dim,
                           f32_row + h * head_dim,
                           head_dim * sizeof(float));
            }
            eakv_cache_append(cache, transposed, l, 1, new_tokens);
            ptr += size_row * cell_count;
        }
    }

    eakv_cache_advance(cache, new_tokens);
    free(transposed);
    free(f32_row);
    return EAKV_OK;

format_err:
    free(transposed);
    free(f32_row);
    return EAKV_ERR_FORMAT;
}
