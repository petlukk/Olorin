/* Internal header — not part of public API. */
#ifndef EAKV_INTERNAL_H
#define EAKV_INTERNAL_H

#include "eakv.h"
#include <stdint.h>

typedef struct {
    uint8_t *weights;   /* packed Q4 nibbles: n_groups * 32 bytes */
    float   *scales;    /* per-group scale: n_groups floats */
    float   *biases;    /* per-group bias: n_groups floats */
} eakv_kv_data_t;

struct eakv_cache {
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;
    int seq_len;

    int groups_per_token;  /* n_kv_heads * head_dim / 64 */
    int max_groups;        /* groups_per_token * max_seq_len */

    void *data_buf;        /* single flat allocation for all layer data */
    eakv_kv_data_t *kv;   /* indexed views: kv[layer * 2 + kv_idx] */
};

/* Ea kernel declarations (linked from .o files) */
extern void q4_quantize_split_f32(
    const float *src, int32_t *weights_out,
    float *scales_out, float *biases_out, int32_t n_groups);

extern void q4_dequantize_avx512_f32(
    const uint8_t *weights, const float *scales,
    const float *biases, float *out, int32_t n_groups);

extern void q4_fused_k_score_multi_f32(
    const float *q_vecs, const uint8_t *k_packed,
    const float *k_scales, const float *k_biases,
    float *all_scores, int32_t seq_len,
    int32_t n_heads, int32_t groups_per_head);

extern void q4_fused_k_score_multi_64_f32(
    const float *q_vecs, const uint8_t *k_packed,
    const float *k_scales, const float *k_biases,
    float *all_scores, int32_t seq_len,
    int32_t n_heads, int32_t groups_per_head);

extern void q4_k_score_gqa_f32(
    const float *q_vecs, const uint8_t *k_packed,
    const float *k_scales, const float *k_biases,
    float *all_scores, int32_t seq_len,
    int32_t n_q_heads, int32_t n_kv_heads,
    int32_t groups_per_head);

extern void q4_k_score_gqa_64_f32(
    const float *q_vecs, const uint8_t *k_packed,
    const float *k_scales, const float *k_biases,
    float *all_scores, int32_t seq_len,
    int32_t n_q_heads, int32_t n_kv_heads,
    int32_t groups_per_head);

extern void q4_fused_v_sum_multi_f32(
    const float *all_weights, const uint8_t *v_packed,
    const float *v_scales, const float *v_biases,
    float *all_out, int32_t seq_len,
    int32_t n_heads, int32_t groups_per_head);

extern void q4_fused_v_sum_multi_64_f32(
    const float *all_weights, const uint8_t *v_packed,
    const float *v_scales, const float *v_biases,
    float *all_out, int32_t seq_len,
    int32_t n_heads, int32_t groups_per_head);

extern void q4_v_sum_gqa_f32(
    const float *all_weights, const uint8_t *v_packed,
    const float *v_scales, const float *v_biases,
    float *all_out, int32_t seq_len,
    int32_t n_q_heads, int32_t n_kv_heads,
    int32_t groups_per_head);

extern void q4_v_sum_gqa_64_f32(
    const float *all_weights, const uint8_t *v_packed,
    const float *v_scales, const float *v_biases,
    float *all_out, int32_t seq_len,
    int32_t n_q_heads, int32_t n_kv_heads,
    int32_t groups_per_head);

extern int32_t q4_validate(
    const float *scales, const float *biases,
    const int32_t *scales_bits, const int32_t *biases_bits,
    int32_t n_groups);

#endif /* EAKV_INTERNAL_H */
