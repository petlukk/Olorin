#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char MAGIC[4] = {'E', 'A', 'K', 'V'};
#define HEADER_SIZE 512

static size_t align64(size_t x) { return (x + 63) & ~(size_t)63; }

#pragma pack(push, 1)
typedef struct {
    char     magic[4];
    uint16_t version;
    uint16_t quant_scheme;
    uint32_t group_size;
    uint16_t orig_dtype;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t seq_len;
    uint32_t max_seq_len;
    int16_t  compression;
    char     model_hash[32];
    char     tokenizer_hash[32];
    uint64_t checksum;
} eakv_header_t;
#pragma pack(pop)

static void write_zeros(FILE *f, size_t n) {
    uint8_t zeros[64] = {0};
    while (n > 0) {
        size_t chunk = n < 64 ? n : 64;
        fwrite(zeros, 1, chunk, f);
        n -= chunk;
    }
}

int eakv_cache_save(const eakv_cache_t *cache, const char *path) {
    if (!cache || !path) return EAKV_ERR_INVALID;

    FILE *f = fopen(path, "wb");
    if (!f) return EAKV_ERR_IO;

    int n_groups = (cache->n_kv_heads * cache->head_dim * cache->seq_len) / 64;
    size_t weights_size = (size_t)n_groups * 32;
    size_t scales_size  = (size_t)n_groups * sizeof(float);
    size_t biases_size  = (size_t)n_groups * sizeof(float);
    size_t block_raw    = weights_size + scales_size + biases_size;
    size_t block_aligned = align64(block_raw);

    uint8_t header[HEADER_SIZE];
    memset(header, 0, HEADER_SIZE);

    eakv_header_t *h = (eakv_header_t *)header;
    memcpy(h->magic, MAGIC, 4);
    h->version      = 1;
    h->quant_scheme = 0;
    h->group_size   = 64;
    h->orig_dtype   = 0;
    h->n_layers     = (uint32_t)cache->n_layers;
    h->n_heads      = (uint32_t)cache->n_kv_heads;
    h->head_dim     = (uint32_t)cache->head_dim;
    h->seq_len      = (uint32_t)cache->seq_len;
    h->max_seq_len  = (uint32_t)cache->seq_len;
    h->compression  = 0;

    fwrite(header, 1, HEADER_SIZE, f);

    size_t idx_table_size = (size_t)cache->n_layers * 2 * 8;
    size_t data_start = align64(HEADER_SIZE + idx_table_size);

    size_t cur = data_start;
    for (int l = 0; l < cache->n_layers; l++) {
        uint64_t k_off = cur;
        cur += block_aligned;
        uint64_t v_off = cur;
        cur += block_aligned;
        fwrite(&k_off, 8, 1, f);
        fwrite(&v_off, 8, 1, f);
    }

    size_t pos = HEADER_SIZE + idx_table_size;
    if (pos < data_start)
        write_zeros(f, data_start - pos);

    for (int l = 0; l < cache->n_layers; l++) {
        for (int kv = 0; kv < 2; kv++) {
            eakv_kv_data_t *d = &cache->kv[l * 2 + kv];
            fwrite(d->weights, 1, weights_size, f);
            fwrite(d->scales, sizeof(float), (size_t)n_groups, f);
            fwrite(d->biases, sizeof(float), (size_t)n_groups, f);

            size_t pad = block_aligned - block_raw;
            if (pad > 0)
                write_zeros(f, pad);
        }
    }

    fclose(f);
    return EAKV_OK;
}

int eakv_cache_load(const char *path, eakv_cache_t **out) {
    if (!path || !out) return EAKV_ERR_INVALID;

    FILE *f = fopen(path, "rb");
    if (!f) return EAKV_ERR_IO;

    uint8_t header[HEADER_SIZE];
    if (fread(header, 1, HEADER_SIZE, f) != HEADER_SIZE) {
        fclose(f); return EAKV_ERR_FORMAT;
    }

    eakv_header_t *h = (eakv_header_t *)header;
    if (memcmp(h->magic, MAGIC, 4) != 0) {
        fclose(f); return EAKV_ERR_FORMAT;
    }
    if (h->version != 1) {
        fclose(f); return EAKV_ERR_FORMAT;
    }

    eakv_cache_t *cache = eakv_cache_create(
        (int)h->n_layers, (int)h->n_heads, (int)h->head_dim, (int)h->seq_len);
    if (!cache) { fclose(f); return EAKV_ERR_ALLOC; }

    int n_groups = ((int)h->n_heads * (int)h->head_dim * (int)h->seq_len) / 64;
    size_t weights_size = (size_t)n_groups * 32;

    size_t idx_table_size = (size_t)h->n_layers * 2 * 8;
    uint64_t *offsets = malloc(idx_table_size);
    if (!offsets) { eakv_cache_free(cache); fclose(f); return EAKV_ERR_ALLOC; }
    if (fread(offsets, 1, idx_table_size, f) != idx_table_size) {
        free(offsets); eakv_cache_free(cache); fclose(f); return EAKV_ERR_FORMAT;
    }

    for (int l = 0; l < (int)h->n_layers; l++) {
        for (int kv = 0; kv < 2; kv++) {
            uint64_t off = offsets[l * 2 + kv];
            fseek(f, (long)off, SEEK_SET);

            eakv_kv_data_t *d = &cache->kv[l * 2 + kv];
            if (fread(d->weights, 1, weights_size, f) != weights_size) goto fail;
            if (fread(d->scales, sizeof(float), (size_t)n_groups, f) != (size_t)n_groups) goto fail;
            if (fread(d->biases, sizeof(float), (size_t)n_groups, f) != (size_t)n_groups) goto fail;
        }
    }

    cache->seq_len = (int)h->seq_len;
    free(offsets);
    fclose(f);
    *out = cache;
    return EAKV_OK;

fail:
    free(offsets);
    eakv_cache_free(cache);
    fclose(f);
    return EAKV_ERR_FORMAT;
}
