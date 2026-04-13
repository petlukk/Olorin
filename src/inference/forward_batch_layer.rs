//! Per-layer batched forward — constant barrier count matching forward_graph.rs.
//!
//! All threads participate in every matmul via work-stealing batch variants.
//! Small ops (norm, quant, rope, residual, PLE) are thread-0 only.
//! Batch-quantize all N tokens once, then single batch matvec = constant barriers.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::{compute_rope_tables, Gemma4State};
use crate::inference::matmul;
use crate::inference::matmul_graph;
use crate::kernels::ffi_inference;
use crate::inference::threadpool::SpinBarrier;

/// Dispatch batch matvec: repacked 8x8 or fallback, work-stealing.
#[inline]
#[allow(clippy::too_many_arguments)]
fn matvec_batch_step(
    repacked: Option<&[u8]>,
    dtype: u32,
    weight: *const u8,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i16,
    output: *mut f32,
    d_scratch: *mut f32,
    n_rows: usize,
    n_cols: usize,
    n_tokens: usize,
    output_stride: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    nth: usize,
) {
    match repacked {
        Some(p) => matmul_graph::q4k_matvec_8x8_batch_ws(
            p.as_ptr(), q8_qs, q8_d, q8_bsums, output,
            n_rows, n_cols, n_tokens, output_stride,
            current_chunk, ith, nth,
        ),
        None => matmul_graph::matvec_batch_ws(
            dtype, weight, q8_qs, q8_d, q8_bsums, output, d_scratch,
            n_rows, n_cols, n_tokens, output_stride,
            current_chunk, ith, nth,
        ),
    }
}

/// Batch-quantize N tokens from `src` (stride `dim`) into batch Q8K buffers.
#[inline]
fn batch_quant(
    src: &[f32], dim: usize, n: usize,
    qs: &mut [i8], d: &mut [f32], bsums: &mut [i16],
) {
    let nb = dim / 256;
    let qs_stride = dim + 12;
    for t in 0..n {
        matmul::quant_input(
            &src[t * dim..(t + 1) * dim],
            &mut qs[t * qs_stride..(t + 1) * qs_stride],
            &mut d[t * nb..(t + 1) * nb],
            &mut bsums[t * nb * 16..(t + 1) * nb * 16],
        );
    }
}

/// Per-layer batched forward. Constant barrier count matching forward_graph.rs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward_batch(
    state: &mut Gemma4State,
    model: &Gemma4Model,
    il: usize,
    n: usize,
    seq_len: usize,
    barrier: &SpinBarrier,
    current_chunk: &AtomicI32,
    ith: usize,
    nth: usize,
) {
    let hd = model.hidden_dim;
    let n_heads = model.n_heads;
    let n_kv_heads = model.n_kv_heads;
    let gqa_ratio = n_heads / n_kv_heads;
    let lw = &model.layers[il];
    let head_dim = model.head_dim_k[il];
    let head_dim_v = model.head_dim_v[il];
    let has_kv = model.kv_shared_source[il].is_none();
    let qkv_dim = n_heads * head_dim;

    let n_rot = if model.is_swa[il] { model.rope_dim_swa } else { model.rope_dim_global };
    let rope_theta = if model.is_swa[il] { model.rope_theta_swa } else { model.rope_theta_global };
    let freq_factors = if !model.is_swa[il] { model.rope_freqs.as_deref() } else { None };

    // ── 1. Attn norm + batch quant (thread 0) ────────────────────
    if ith == 0 {
        for t in 0..n {
            ffi_inference::gemma4_rmsnorm(
                state.batch_x[t * hd..].as_ptr(), lw.attn_norm,
                state.batch_x_norm[t * hd..].as_mut_ptr(), hd as i32, model.rms_eps,
            );
        }
        batch_quant(
            &state.batch_x_norm, hd, n,
            &mut state.batch_q8_qs, &mut state.batch_q8_d, &mut state.batch_q8_bsums,
        );
    }
    barrier.wait(); // B1

    // ── 2. Q projection (work-stealing, all threads) ─────────────
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait(); // B2
    matvec_batch_step(
        lw.wq_repacked.as_deref(), lw.wq_dtype, lw.wq,
        state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
        state.batch_q8_bsums.as_ptr(),
        state.batch_q.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        qkv_dim, hd, n, qkv_dim,
        current_chunk, ith, nth,
    );
    barrier.wait(); // B3

    // ── 3. Q norm + RoPE per token (thread 0) ────────────────────
    if ith == 0 {
        for t in 0..n {
            compute_rope_tables(
                &mut state.cos_table, &mut state.sin_table,
                seq_len + t, n_rot, rope_theta, freq_factors,
            );
            if !lw.q_norm.is_null() {
                for h in 0..n_heads {
                    let off = t * qkv_dim + h * head_dim;
                    ffi_inference::gemma4_rmsnorm(
                        unsafe { state.batch_q.as_ptr().add(off) }, lw.q_norm,
                        state.x_norm.as_mut_ptr(), head_dim as i32, model.rms_eps,
                    );
                    state.batch_q[off..off + head_dim].copy_from_slice(&state.x_norm[..head_dim]);
                }
            }
            ffi_inference::gemma4_rope(
                unsafe { state.batch_q.as_mut_ptr().add(t * qkv_dim) },
                state.cos_table.as_ptr(), state.sin_table.as_ptr(),
                head_dim as i32, n_heads as i32,
            );
        }
    }
    barrier.wait(); // B4

    // ── 4. K/V projections + norms + RoPE + cache ────────────────
    // Input already batch-quantized in step 1 (same batch_x_norm -> batch_q8).
    if has_kv {
        let kv_dim = n_kv_heads * head_dim;
        let kv_dim_v = n_kv_heads * head_dim_v;

        // K matmul (work-stealing batch)
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait(); // B5
        matvec_batch_step(
            lw.wk_repacked.as_deref(), lw.wk_dtype, lw.wk,
            state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
            state.batch_q8_bsums.as_ptr(),
            state.batch_k.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            kv_dim, hd, n, kv_dim,
            current_chunk, ith, nth,
        );
        barrier.wait(); // B6

        // V matmul (work-stealing batch)
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait(); // B7
        matvec_batch_step(
            lw.wv_repacked.as_deref(), lw.wv_dtype, lw.wv,
            state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
            state.batch_q8_bsums.as_ptr(),
            state.batch_v.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            kv_dim_v, hd, n, kv_dim_v,
            current_chunk, ith, nth,
        );
        barrier.wait(); // B8

        // K/V norms + RoPE + cache store (thread 0)
        if ith == 0 {
            for t in 0..n {
                compute_rope_tables(
                    &mut state.cos_table, &mut state.sin_table,
                    seq_len + t, n_rot, rope_theta, freq_factors,
                );
                if !lw.k_norm.is_null() {
                    for h in 0..n_kv_heads {
                        let off = t * kv_dim + h * head_dim;
                        ffi_inference::gemma4_rmsnorm(
                            unsafe { state.batch_k.as_ptr().add(off) }, lw.k_norm,
                            state.x_norm.as_mut_ptr(), head_dim as i32, model.rms_eps,
                        );
                        state.batch_k[off..off + head_dim]
                            .copy_from_slice(&state.x_norm[..head_dim]);
                    }
                }
                for h in 0..n_kv_heads {
                    let off = t * kv_dim_v + h * head_dim_v;
                    super::forward::bare_rmsnorm(
                        &mut state.batch_v[off..off + head_dim_v], model.rms_eps,
                    );
                }
                ffi_inference::gemma4_rope(
                    unsafe { state.batch_k.as_mut_ptr().add(t * kv_dim) },
                    state.cos_table.as_ptr(), state.sin_table.as_ptr(),
                    head_dim as i32, n_kv_heads as i32,
                );
            }
            state.cache.store_batch(
                il, &state.batch_k[..kv_dim * n], &state.batch_v[..kv_dim_v * n], n,
            );
        }
        barrier.wait(); // B9
    }

    // ── 5. Attention (heads split across threads, fused kernel) ──
    {
        let n_kv = if model.is_swa[il] {
            (seq_len + n).min(model.sliding_window)
        } else {
            seq_len + n
        };
        let k_ptr = state.cache.k_ptr(il);
        let v_ptr = state.cache.v_ptr(il);
        let stride_kv = n_kv_heads * head_dim;
        let attn_scores_stride = state.attn_scores_stride;
        let kv_scratch_stride = state.kv_scratch_stride;

        let per = (n_heads + nth - 1) / nth;
        let h_start = ith * per;
        let h_end = ((ith + 1) * per).min(n_heads);

        for h in h_start..h_end {
            let kv_h = h / gqa_ratio;
            unsafe {
                ffi_inference::attn_fused_batched(
                    state.batch_q.as_ptr().add(h * head_dim),
                    k_ptr, v_ptr,
                    state.batch_attn_out.as_mut_ptr().add(h * head_dim),
                    state.attn_scores.as_mut_ptr().add(ith * attn_scores_stride),
                    state.kv_f32_scratch.as_mut_ptr().add(ith * kv_scratch_stride),
                    head_dim as i32,
                    qkv_dim as i32,   // q_stride
                    qkv_dim as i32,   // out_stride
                    stride_kv as i32,
                    (kv_h * head_dim) as i32,
                    n_kv as i32,
                    n as i32,
                    seq_len as i32,
                    1.0f32,
                );
            }
        }
    }
    barrier.wait(); // B10

    // ── 6. Wo projection (batch quant + batch matvec) ────────────
    if ith == 0 {
        let ao_dim = n_heads * head_dim;
        batch_quant(
            &state.batch_attn_out, ao_dim, n,
            &mut state.batch_q8_qs, &mut state.batch_q8_d, &mut state.batch_q8_bsums,
        );
    }
    barrier.wait(); // B11

    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait(); // B12
    matvec_batch_step(
        lw.wo_repacked.as_deref(), lw.wo_dtype, lw.wo,
        state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
        state.batch_q8_bsums.as_ptr(),
        state.batch_wo_out.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, n_heads * head_dim, n, hd,
        current_chunk, ith, nth,
    );
    barrier.wait(); // B13

    // ── 7. Post-attn norm + residual + FFN norm + batch quant (thread 0) ──
    if ith == 0 {
        for t in 0..n {
            let off = t * hd;
            if !lw.post_attn_norm.is_null() {
                ffi_inference::gemma4_rmsnorm(
                    state.batch_wo_out[off..].as_ptr(), lw.post_attn_norm,
                    state.x_norm.as_mut_ptr(), hd as i32, model.rms_eps,
                );
                ffi_inference::vec_add_f32(
                    state.x_norm.as_ptr(), state.batch_x[off..].as_ptr(),
                    state.batch_attn_res[off..].as_mut_ptr(), hd as i32,
                );
            } else {
                ffi_inference::vec_add_f32(
                    state.batch_wo_out[off..].as_ptr(), state.batch_x[off..].as_ptr(),
                    state.batch_attn_res[off..].as_mut_ptr(), hd as i32,
                );
            }
            ffi_inference::gemma4_rmsnorm(
                state.batch_attn_res[off..].as_ptr(), lw.ffn_norm,
                state.batch_x_norm[off..].as_mut_ptr(), hd as i32, model.rms_eps,
            );
        }
        batch_quant(
            &state.batch_x_norm, hd, n,
            &mut state.batch_q8_qs, &mut state.batch_q8_d, &mut state.batch_q8_bsums,
        );
    }
    barrier.wait(); // B14

    // ── 8. FFN gate+up (batch matvec) ────────────────────────────
    let ffn_dim = model.ffn_dim[il];
    // Gate + Up: always 4 barriers (store+barrier, gate matvec, barrier,
    // store+barrier, up matvec, barrier). No batch dual variant available.
    // All threads take the same path for a given layer.
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait(); // B15
    matvec_batch_step(
        lw.w_gate_repacked.as_deref(), lw.w_gate_dtype, lw.w_gate,
        state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
        state.batch_q8_bsums.as_ptr(),
        state.batch_gate.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        ffn_dim, hd, n, ffn_dim,
        current_chunk, ith, nth,
    );
    barrier.wait(); // B16
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait(); // B17
    matvec_batch_step(
        lw.w_up_repacked.as_deref(), lw.w_up_dtype, lw.w_up,
        state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
        state.batch_q8_bsums.as_ptr(),
        state.batch_up.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        ffn_dim, hd, n, ffn_dim,
        current_chunk, ith, nth,
    );
    barrier.wait(); // B18

    // ── 9. GELU + batch quant + down matmul ──────────────────────
    if ith == 0 {
        for t in 0..n {
            ffi_inference::gelu_mul(
                state.batch_gate[t * ffn_dim..].as_ptr(),
                state.batch_up[t * ffn_dim..].as_ptr(),
                state.batch_gate[t * ffn_dim..].as_mut_ptr(), ffn_dim as i32,
            );
        }
        batch_quant(
            &state.batch_gate, ffn_dim, n,
            &mut state.batch_ffn_q8_qs, &mut state.batch_ffn_q8_d,
            &mut state.batch_ffn_q8_bsums,
        );
    }
    barrier.wait(); // B17/B19

    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait(); // B18/B20
    matvec_batch_step(
        lw.w_down_repacked.as_deref(), lw.w_down_dtype, lw.w_down,
        state.batch_ffn_q8_qs.as_ptr(), state.batch_ffn_q8_d.as_ptr(),
        state.batch_ffn_q8_bsums.as_ptr(),
        state.batch_down.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, ffn_dim, n, hd,
        current_chunk, ith, nth,
    );
    barrier.wait(); // B19/B21

    // ── 10. Post-FFN norm + residual + PLE + scale (thread 0) ───
    if ith == 0 {
        for t in 0..n {
            let off = t * hd;
            if !lw.post_ffn_norm.is_null() {
                ffi_inference::gemma4_rmsnorm(
                    state.batch_down[off..].as_ptr(), lw.post_ffn_norm,
                    state.x_norm.as_mut_ptr(), hd as i32, model.rms_eps,
                );
                ffi_inference::vec_add_f32(
                    state.x_norm.as_ptr(), state.batch_attn_res[off..].as_ptr(),
                    state.batch_x[off..].as_mut_ptr(), hd as i32,
                );
            } else {
                ffi_inference::vec_add_f32(
                    state.batch_down[off..].as_ptr(), state.batch_attn_res[off..].as_ptr(),
                    state.batch_x[off..].as_mut_ptr(), hd as i32,
                );
            }

            // PLE
            if model.ple_dim > 0 && !lw.inp_gate.is_null() && !lw.proj.is_null() {
                let ple_dim = model.ple_dim;
                let ple_total = ple_dim * model.n_layers;
                let ple_off = il * ple_dim;

                matmul::quant_input(
                    &state.batch_x[off..off + hd],
                    &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums,
                );
                matmul::matvec(
                    lw.inp_gate_dtype, lw.inp_gate,
                    &state.q8_qs, &state.q8_d, &state.q8_bsums,
                    &mut state.ple_gate, &mut state.q6k_d_scratch, ple_dim, hd,
                );
                ffi_inference::gelu_mul(
                    state.ple_gate.as_ptr(),
                    state.batch_ple_signal[t * ple_total + ple_off..].as_ptr(),
                    state.ple_gate.as_mut_ptr(), ple_dim as i32,
                );
                matmul::quant_input(
                    &state.ple_gate[..ple_dim],
                    &mut state.ple_q8_qs, &mut state.ple_q8_d, &mut state.ple_q8_bsums,
                );
                matmul::matvec(
                    lw.proj_dtype, lw.proj,
                    &state.ple_q8_qs, &state.ple_q8_d, &state.ple_q8_bsums,
                    &mut state.ple_out, &mut state.q6k_d_scratch, hd, ple_dim,
                );
                if !lw.post_norm.is_null() {
                    ffi_inference::gemma4_rmsnorm(
                        state.ple_out.as_ptr(), lw.post_norm,
                        state.ple_out.as_mut_ptr(), hd as i32, model.rms_eps,
                    );
                }
                ffi_inference::vec_add_f32(
                    state.batch_x[off..].as_ptr(), state.ple_out.as_ptr(),
                    state.batch_x[off..].as_mut_ptr(), hd as i32,
                );
            }

            let out_scale = lw.layer_output_scale;
            if out_scale != 1.0 {
                ffi_inference::vec_scale_f32(
                    state.batch_x[off..].as_ptr(), state.batch_x[off..].as_mut_ptr(),
                    out_scale, hd as i32,
                );
            }
        }
    }
    barrier.wait(); // B20/B22
}
