#include "internal.h"

void eakv_attention_scores(const eakv_cache_t *cache, const float *queries,
                           int layer, int n_q_heads, int n_kv_heads,
                           float *scores_out) {
    const eakv_kv_data_t *k = &cache->kv[layer * 2 + 0];
    int32_t groups_per_head = cache->max_seq_len * (cache->head_dim / 64);

    if (cache->head_dim == 64) {
        if (n_q_heads == n_kv_heads) {
            q4_fused_k_score_multi_64_f32(
                queries, k->weights, k->scales, k->biases,
                scores_out, cache->seq_len, n_q_heads, groups_per_head);
        } else {
            q4_k_score_gqa_64_f32(
                queries, k->weights, k->scales, k->biases,
                scores_out, cache->seq_len, n_q_heads, n_kv_heads,
                groups_per_head);
        }
    } else {
        if (n_q_heads == n_kv_heads) {
            q4_fused_k_score_multi_f32(
                queries, k->weights, k->scales, k->biases,
                scores_out, cache->seq_len, n_q_heads, groups_per_head);
        } else {
            q4_k_score_gqa_f32(
                queries, k->weights, k->scales, k->biases,
                scores_out, cache->seq_len, n_q_heads, n_kv_heads,
                groups_per_head);
        }
    }
}

void eakv_attention_output(const eakv_cache_t *cache, const float *weights,
                           int layer, int n_q_heads, int n_kv_heads,
                           float *output_out) {
    const eakv_kv_data_t *v = &cache->kv[layer * 2 + 1];
    int32_t groups_per_head = cache->max_seq_len * (cache->head_dim / 64);

    if (cache->head_dim == 64) {
        if (n_q_heads == n_kv_heads) {
            q4_fused_v_sum_multi_64_f32(
                weights, v->weights, v->scales, v->biases,
                output_out, cache->seq_len, n_q_heads, groups_per_head);
        } else {
            q4_v_sum_gqa_64_f32(
                weights, v->weights, v->scales, v->biases,
                output_out, cache->seq_len, n_q_heads, n_kv_heads,
                groups_per_head);
        }
    } else {
        if (n_q_heads == n_kv_heads) {
            q4_fused_v_sum_multi_f32(
                weights, v->weights, v->scales, v->biases,
                output_out, cache->seq_len, n_q_heads, groups_per_head);
        } else {
            q4_v_sum_gqa_f32(
                weights, v->weights, v->scales, v->biases,
                output_out, cache->seq_len, n_q_heads, n_kv_heads,
                groups_per_head);
        }
    }
}
