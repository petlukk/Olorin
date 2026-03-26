#include "internal.h"
#include <stdlib.h>
#include <string.h>

static size_t block_size(int max_groups) {
    return (size_t)max_groups * 32
         + (size_t)max_groups * sizeof(float)
         + (size_t)max_groups * sizeof(float);
}

eakv_cache_t *eakv_cache_create(int n_layers, int n_kv_heads,
                                 int head_dim, int max_seq_len) {
    if (n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0)
        return NULL;
    if ((n_kv_heads * head_dim) % 64 != 0)
        return NULL;

    eakv_cache_t *c = calloc(1, sizeof(*c));
    if (!c) return NULL;

    c->n_layers    = n_layers;
    c->n_kv_heads  = n_kv_heads;
    c->head_dim    = head_dim;
    c->max_seq_len = max_seq_len;
    c->seq_len     = 0;
    c->groups_per_token = (n_kv_heads * head_dim) / 64;
    c->max_groups       = c->groups_per_token * max_seq_len;

    c->kv = calloc(n_layers * 2, sizeof(eakv_kv_data_t));
    if (!c->kv) { free(c); return NULL; }

    size_t blk = block_size(c->max_groups);
    size_t total = blk * n_layers * 2;
    c->data_buf = calloc(1, total);
    if (!c->data_buf) { free(c->kv); free(c); return NULL; }

    char *ptr = (char *)c->data_buf;
    for (int l = 0; l < n_layers; l++) {
        for (int kv = 0; kv < 2; kv++) {
            eakv_kv_data_t *d = &c->kv[l * 2 + kv];
            d->weights = (uint8_t *)ptr;
            ptr += (size_t)c->max_groups * 32;
            d->scales = (float *)ptr;
            ptr += (size_t)c->max_groups * sizeof(float);
            d->biases = (float *)ptr;
            ptr += (size_t)c->max_groups * sizeof(float);
        }
    }

    return c;
}

void eakv_cache_free(eakv_cache_t *cache) {
    if (!cache) return;
    free(cache->data_buf);
    free(cache->kv);
    free(cache);
}

int eakv_cache_load_raw(eakv_cache_t *cache, const float *data, int seq_len) {
    if (!cache || !data || seq_len <= 0)
        return EAKV_ERR_INVALID;
    if (seq_len > cache->max_seq_len)
        return EAKV_ERR_INVALID;

    int hd = cache->head_dim;
    int nh = cache->n_kv_heads;
    int gpd = hd / 64;  /* groups per token per head */
    int gph = cache->max_seq_len * gpd;  /* groups per head in buffer (fixed stride) */
    int n_groups_per_head = seq_len * gpd;
    int elems_per_lkv = nh * seq_len * hd;

    int32_t *tmp = malloc(n_groups_per_head * 32 * sizeof(int32_t));
    if (!tmp) return EAKV_ERR_ALLOC;

    for (int l = 0; l < cache->n_layers; l++) {
        for (int kv = 0; kv < 2; kv++) {
            const float *lkv_src = data + (l * 2 + kv) * elems_per_lkv;
            eakv_kv_data_t *d = &cache->kv[l * 2 + kv];

            for (int h = 0; h < nh; h++) {
                const float *src = lkv_src + h * seq_len * hd;
                int group_base = h * gph;

                q4_quantize_split_f32(src, tmp,
                                       d->scales + group_base,
                                       d->biases + group_base,
                                       n_groups_per_head);

                uint8_t *dst = d->weights + group_base * 32;
                for (int i = 0; i < n_groups_per_head * 32; i++)
                    dst[i] = (uint8_t)tmp[i];
            }
        }
    }

    cache->seq_len = seq_len;
    free(tmp);
    return EAKV_OK;
}

int eakv_cache_append(eakv_cache_t *cache, const float *data,
                      int layer, int kv_idx, int n_tokens) {
    if (!cache || !data || n_tokens <= 0)
        return EAKV_ERR_INVALID;
    if (layer < 0 || layer >= cache->n_layers)
        return EAKV_ERR_INVALID;
    if (kv_idx < 0 || kv_idx > 1)
        return EAKV_ERR_INVALID;
    if (cache->seq_len + n_tokens > cache->max_seq_len)
        return EAKV_ERR_INVALID;

    /*
     * Group layout in the flat buffer (matching bulk load_raw):
     *   [head0: pos0_g0..pos0_gN, pos1_g0..pos1_gN, ...]
     *   [head1: pos0_g0..pos0_gN, pos1_g0..pos1_gN, ...]
     *
     * groups_per_dim = head_dim / 64 (groups per token per head)
     * groups_per_head = max_seq_len * groups_per_dim
     *
     * Input data layout: [head][token][dim]
     * We quantize per-head and write at the right offset.
     */
    int hd = cache->head_dim;
    int nh = cache->n_kv_heads;
    int gpd = hd / 64;  /* groups per token per head */
    int gph = cache->max_seq_len * gpd;  /* groups per head in buffer */
    int n_groups_per_head = n_tokens * gpd;

    int32_t *tmp = malloc(n_groups_per_head * 32 * sizeof(int32_t));
    if (!tmp) return EAKV_ERR_ALLOC;

    eakv_kv_data_t *d = &cache->kv[layer * 2 + kv_idx];

    for (int h = 0; h < nh; h++) {
        const float *src = data + h * n_tokens * hd;
        int group_base = h * gph + cache->seq_len * gpd;

        q4_quantize_split_f32(src, tmp,
                               d->scales + group_base,
                               d->biases + group_base,
                               n_groups_per_head);

        uint8_t *dst = d->weights + group_base * 32;
        for (int i = 0; i < n_groups_per_head * 32; i++)
            dst[i] = (uint8_t)tmp[i];
    }

    free(tmp);
    return EAKV_OK;
}

int eakv_cache_advance(eakv_cache_t *cache, int n_tokens) {
    if (!cache || n_tokens <= 0)
        return EAKV_ERR_INVALID;
    if (cache->seq_len + n_tokens > cache->max_seq_len)
        return EAKV_ERR_INVALID;
    cache->seq_len += n_tokens;
    return EAKV_OK;
}

void eakv_cache_clear(eakv_cache_t *cache) {
    if (cache) cache->seq_len = 0;
}

int eakv_checkpoint(eakv_cache_t *cache) {
    if (!cache) return 0;
    return cache->seq_len;
}

int eakv_restore(eakv_cache_t *cache, int seq_len) {
    if (!cache) return EAKV_ERR_INVALID;
    if (seq_len < 0 || seq_len > cache->seq_len) return EAKV_ERR_INVALID;
    cache->seq_len = seq_len;
    return EAKV_OK;
}

int eakv_cache_seq_len(const eakv_cache_t *cache) {
    return cache ? cache->seq_len : 0;
}

int eakv_cache_n_layers(const eakv_cache_t *cache) {
    return cache ? cache->n_layers : 0;
}

int eakv_cache_n_heads(const eakv_cache_t *cache) {
    return cache ? cache->n_kv_heads : 0;
}

int eakv_cache_head_dim(const eakv_cache_t *cache) {
    return cache ? cache->head_dim : 0;
}

int eakv_cache_max_seq_len(const eakv_cache_t *cache) {
    return cache ? cache->max_seq_len : 0;
}

float eakv_cache_compression_ratio(const eakv_cache_t *cache) {
    if (!cache || cache->seq_len == 0) return 0.0f;
    return 40.0f / 256.0f;
}
