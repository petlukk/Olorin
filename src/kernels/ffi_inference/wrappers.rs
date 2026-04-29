//! Thin pub wrappers over the kernel function pointers in the loaded
//! KernelTableInference. Each wrapper looks up the static table via `k()`
//! and forwards through the function pointer.

use super::k;

pub unsafe fn quant_f32_q8k(
    src: *const f32, dst_qs: *mut i8, dst_d: *mut f32, dst_bsums: *mut i16, n: i32,
) {
    (k().quant_f32_q8k)(src, dst_qs, dst_d, dst_bsums, n)
}

pub unsafe fn q4k_dot_q8k(
    q4: *const u8, q8: *const i8, bsums: *const i16,
    n_blocks: i32, q8_d: *const f32, pow2: *const f32,
) -> f32 {
    (k().q4k_dot_q8k)(q4, q8, bsums, n_blocks, q8_d, pow2)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_dot_q8k_4row(
    rw0: *const u8, rw1: *const u8, rw2: *const u8, rw3: *const u8,
    q8: *const i8, bsums: *const i16,
    scores: *mut f32, n_blocks: i32, q8_d: *const f32, pow2: *const f32,
) {
    (k().q4k_dot_q8k_4row)(rw0, rw1, rw2, rw3, q8, bsums, scores, n_blocks, q8_d, pow2)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_dot_q8k_4row_dual(
    gw0: *const u8, gw1: *const u8, gw2: *const u8, gw3: *const u8,
    uw0: *const u8, uw1: *const u8, uw2: *const u8, uw3: *const u8,
    q8: *const i8, bsums: *const i16,
    gate_scores: *mut f32, up_scores: *mut f32, n_blocks: i32,
    q8_d: *const f32, pow2: *const f32,
) {
    (k().q4k_dot_q8k_4row_dual)(
        gw0, gw1, gw2, gw3, uw0, uw1, uw2, uw3,
        q8, bsums, gate_scores, up_scores, n_blocks, q8_d, pow2)
}

pub unsafe fn q3k_dot_q8k(
    q3: *const u8, q8: *const i8, bsums: *const i16,
    n_blocks: i32, q8_d: *const f32, pow2: *const f32,
) -> f32 {
    (k().q3k_dot_q8k)(q3, q8, bsums, n_blocks, q8_d, pow2)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q3k_dot_q8k_4row(
    rw0: *const u8, rw1: *const u8, rw2: *const u8, rw3: *const u8,
    q8: *const i8, bsums: *const i16,
    scores: *mut f32, n_blocks: i32, q8_d: *const f32, pow2: *const f32,
) {
    (k().q3k_dot_q8k_4row)(rw0, rw1, rw2, rw3, q8, bsums, scores, n_blocks, q8_d, pow2)
}

pub unsafe fn q5k_dot_q8k(
    q5: *const u8, q8: *const i8, bsums: *const i16,
    n_blocks: i32, q8_d: *const f32, pow2: *const f32,
) -> f32 {
    (k().q5k_dot_q8k)(q5, q8, bsums, n_blocks, q8_d, pow2)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q5k_dot_q8k_4row(
    rw0: *const u8, rw1: *const u8, rw2: *const u8, rw3: *const u8,
    q8: *const i8, bsums: *const i16,
    scores: *mut f32, n_blocks: i32, q8_d: *const f32, pow2: *const f32,
) {
    (k().q5k_dot_q8k_4row)(rw0, rw1, rw2, rw3, q8, bsums, scores, n_blocks, q8_d, pow2)
}

pub unsafe fn q6k_dot_q8k(
    weight: *const u8, q8: *const i8, bsums: *const i16,
    n_blocks: i32, d_arr: *const f32,
) -> f32 {
    (k().q6k_dot_q8k)(weight, q8, bsums, n_blocks, d_arr)
}

pub unsafe fn q6k_dot_q8k_4row(
    w0: *const u8, w1: *const u8, w2: *const u8, w3: *const u8,
    q8: *const i8, bsums: *const i16, scores: *mut f32, n_blocks: i32,
    d0: *const f32, d1: *const f32, d2: *const f32, d3: *const f32,
) {
    (k().q6k_dot_q8k_4row)(
        w0, w1, w2, w3, q8, bsums, scores, n_blocks, d0, d1, d2, d3)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q6k_dot_q8k_4row_repacked(
    packed: *const u8, q8: *const i8, bsums: *const i16,
    scores: *mut f32, n_blocks: i32, d_arr: *const f32,
) {
    (k().q6k_dot_q8k_4row_repacked)(packed, q8, bsums, scores, n_blocks, d_arr)
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn q6k_gemm(
    weight: *const u8, q8_a: *const u8, scratch: *mut u8,
    out: *mut f32, output_stride: i32, n_inner: i32, nr: i32, nc: i32,
) {
    (k().q6k_gemm)(weight, q8_a, scratch, out, output_stride, n_inner, nr, nc)
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn q5k_gemm(
    weight: *const u8, q8_a: *const u8, scratch: *mut u8,
    out: *mut f32, output_stride: i32, n_inner: i32, nr: i32, nc: i32,
) {
    (k().q5k_gemm)(weight, q8_a, scratch, out, output_stride, n_inner, nr, nc)
}

pub unsafe fn f16_to_f32(src: *const u16, dst: *mut f32, n: i32) {
    (k().f16_to_f32)(src, dst, n)
}

pub unsafe fn softmax_f32(data: *mut f32, n: i32, scale: f32) {
    (k().softmax_f32)(data, n, scale)
}

pub fn gemma4_rmsnorm(x: *const f32, weight: *const f32, out: *mut f32, n: i32, eps: f32) {
    unsafe { (k().gemma4_rmsnorm)(x, weight, out, n, eps) }
}

pub fn gelu_mul(gate: *const f32, up: *const f32, out: *mut f32, n: i32) {
    unsafe { (k().gelu_mul)(gate, up, out, n) }
}

pub fn gemma4_rope(data: *mut f32, cos_table: *const f32, sin_table: *const f32, head_dim: i32, n_heads: i32) {
    unsafe { (k().gemma4_rope)(data, cos_table, sin_table, head_dim, n_heads) }
}

pub unsafe fn bf16_dot_f32(
    weight: *const u16, input: *const f32, scratch: *mut i32, n_cols: i32,
) -> f32 {
    (k().bf16_dot_f32)(weight, input, scratch, n_cols)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn bf16_dot_multi_input(
    w_row: *const u16, inputs: *const f32, out_scores: *mut f32,
    scratch: *mut i32, n_tokens: i32, n_cols: i32,
    input_stride: i32, output_stride: i32,
) {
    (k().bf16_dot_multi_input)(
        w_row, inputs, out_scores, scratch,
        n_tokens, n_cols, input_stride, output_stride,
    )
}

pub unsafe fn bf16_dot_f32_4row(
    w0: *const u16, w1: *const u16, w2: *const u16, w3: *const u16,
    input: *const f32, scores: *mut f32, scratch: *mut i32, n_cols: i32,
) {
    (k().bf16_dot_f32_4row)(w0, w1, w2, w3, input, scores, scratch, n_cols)
}

pub fn vec_add_f32(a: *const f32, b: *const f32, out: *mut f32, n: i32) {
    unsafe { (k().vec_add_f32)(a, b, out, n) }
}

pub fn vec_scale_f32(a: *const f32, out: *mut f32, s: f32, n: i32) {
    unsafe { (k().vec_scale_f32)(a, out, s, n) }
}

pub fn vec_fma_f32(a: *const f32, b: *const f32, out: *mut f32, s: f32, n: i32) {
    unsafe { (k().vec_fma_f32)(a, b, out, s, n) }
}

pub fn f32_dot(a: *const f32, b: *const f32, n: i32) -> f32 {
    unsafe { (k().f32_dot)(a, b, n) }
}

pub fn f32_dot_acc(out: *mut f32, a: *const f32, s: f32, n: i32) {
    unsafe { (k().f32_dot_acc)(out, a, s, n) }
}

pub fn bare_rmsnorm_f32(x: *mut f32, n: i32, eps: f32) {
    unsafe { (k().bare_rmsnorm_f32)(x, n, eps) }
}

pub fn softcap_f32(data: *mut f32, n: i32, cap: f32) {
    unsafe { (k().softcap_f32)(data, n, cap) }
}

pub unsafe fn q4k_repack_8x8(
    src: *const u8, dst: *mut u8, n_rows: i32, n_cols: i32,
) {
    (k().q4k_repack_8x8)(src, dst, n_rows, n_cols)
}

pub unsafe fn q5k_repack_8x8(
    src: *const u8, dst: *mut u8, n_rows: i32, n_cols: i32,
) {
    (k().q5k_repack_8x8)(src, dst, n_rows, n_cols)
}

pub unsafe fn q3k_repack_8x8(
    src: *const u8, dst: *mut u8, n_rows: i32, n_cols: i32,
) {
    (k().q3k_repack_8x8)(src, dst, n_rows, n_cols)
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn q3k_8x8_q8k_gemm(
    packed: *const u8,
    q8_a: *const u8,
    scratch: *mut u8,
    out: *mut f32,
    bs: i32,
    n: i32,
    nr: i32,
    nc: i32,
) {
    (k().q3k_8x8_q8k_gemm)(packed, q8_a, scratch, out, bs, n, nr, nc)
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn q5k_8x8_q8k_matvec(
    packed: *const u8,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i16,
    pow2: *const f32,
    scratch: *mut u8,
    out: *mut f32,
    n_rows: i32,
    n_cols: i32,
) {
    (k().q5k_8x8_q8k_matvec)(packed, q8_qs, q8_d, q8_bsums, pow2, scratch, out, n_rows, n_cols)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_8x8_q8k_matvec(
    packed: *const u8,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i16,
    pow2: *const f32,
    scratch: *mut u8,
    out: *mut f32,
    n_rows: i32,
    n_cols: i32,
) {
    (k().q4k_8x8_q8k_matvec)(
        packed, q8_qs, q8_d, q8_bsums, pow2, scratch, out, n_rows, n_cols,
    )
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_8x8_q8k_matvec_dual(
    packed_a: *const u8,
    packed_b: *const u8,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i16,
    pow2: *const f32,
    scratch: *mut u8,
    out_a: *mut f32,
    out_b: *mut f32,
    n_rows: i32,
    n_cols: i32,
) {
    (k().q4k_8x8_q8k_matvec_dual)(
        packed_a, packed_b, q8_qs, q8_d, q8_bsums, pow2, scratch,
        out_a, out_b, n_rows, n_cols,
    )
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q8k_repack_4(
    row0_qs:    *const i8,
    row1_qs:    *const i8,
    row2_qs:    *const i8,
    row3_qs:    *const i8,
    row_d:      *const f32,
    row0_bsums: *const i16,
    row1_bsums: *const i16,
    row2_bsums: *const i16,
    row3_bsums: *const i16,
    dst:        *mut u8,
    nb:         i32,
) {
    (k().q8k_repack_4)(
        row0_qs, row1_qs, row2_qs, row3_qs,
        row_d,
        row0_bsums, row1_bsums, row2_bsums, row3_bsums,
        dst, nb,
    )
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_8x8_q8k_gemm(
    packed: *const u8,
    q8_a: *const u8,
    scratch: *mut u8,
    out: *mut f32,
    bs: i32,
    n: i32,
    nr: i32,
    nc: i32,
) {
    (k().q4k_8x8_q8k_gemm)(packed, q8_a, scratch, out, bs, n, nr, nc)
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn q5k_8x8_q8k_gemm(
    packed: *const u8,
    q8_a: *const u8,
    scratch: *mut u8,
    out: *mut f32,
    bs: i32,
    n: i32,
    nr: i32,
    nc: i32,
) {
    (k().q5k_8x8_q8k_gemm)(packed, q8_a, scratch, out, bs, n, nr, nc)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_fused_batched(
    q: *const f32,
    k_cache: *const u16,
    v_cache: *const u16,
    dst: *mut f32,
    scores_buf: *mut f32,
    kv_scratch: *mut f32,
    head_dim: i32,
    q_stride: i32,
    out_stride: i32,
    stride_kv: i32,
    kv_head_offset: i32,
    n_kv: i32,
    n_batch: i32,
    cache_start: i32,
    attn_scale: f32,
) {
    (k().attn_fused_batched)(
        q, k_cache, v_cache, dst, scores_buf, kv_scratch,
        head_dim, q_stride, out_stride, stride_kv, kv_head_offset,
        n_kv, n_batch, cache_start, attn_scale,
    )
}
