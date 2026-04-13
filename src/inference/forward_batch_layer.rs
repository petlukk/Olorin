//! Per-layer batched forward — matches forward_graph.rs threading exactly.
//!
//! All threads participate in every matmul via work-stealing (matvec_step).
//! Small ops (norm, quant, rope, residual, PLE) are thread-0 only.
//! Each matmul processes one token at a time with all threads work-stealing rows.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::{compute_rope_tables, Gemma4State};
use crate::inference::matmul;
use crate::inference::matmul_graph;
use crate::kernels::ffi_inference;
use crate::inference::threadpool::SpinBarrier;

/// Dispatch a single matvec_ws call — repacked 8x8 or standard fallback.
/// Identical to forward_graph.rs:matvec_step.
#[inline]
#[allow(clippy::too_many_arguments)]
fn matvec_step(
    dtype: u32,
    weight: *const u8,
    repacked: Option<&[u8]>,
    q8: *const i8,
    q8_d: *const f32,
    bsums: *const i16,
    output: *mut f32,
    d_scratch: *mut f32,
    n_rows: usize,
    n_cols: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    nth: usize,
) {
    match repacked {
        Some(p) => matmul_graph::q4k_matvec_8x8_ws(
            p.as_ptr(), q8, q8_d, bsums, output,
            n_rows, n_cols, current_chunk, ith, nth,
        ),
        None => matmul_graph::matvec_ws(
            dtype, weight, q8, q8_d, bsums, output, d_scratch,
            n_rows, n_cols, current_chunk, ith, nth,
        ),
    }
}

/// Per-layer batched forward. Mirrors layer_forward_graph exactly,
/// processing N tokens per op with work-stealing matmuls.
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

    // ── 1. Attn norm + quant per token (thread 0) ────────────────
    // Same as forward_graph.rs step 1, looped N times.
    if ith == 0 {
        for t in 0..n {
            ffi_inference::gemma4_rmsnorm(
                state.batch_x[t * hd..].as_ptr(), lw.attn_norm,
                state.batch_x_norm[t * hd..].as_mut_ptr(), hd as i32, model.rms_eps,
            );
        }
    }
    barrier.wait();

    // ── 2. Q projection per token (work-stealing, all threads) ───
    // Each token: quant → barrier → WS matvec → barrier
    for t in 0..n {
        if ith == 0 {
            matmul::quant_input(
                &state.batch_x_norm[t * hd..(t + 1) * hd],
                &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums,
            );
        }
        barrier.wait();
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matvec_step(
            lw.wq_dtype, lw.wq, lw.wq_repacked.as_deref(),
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            unsafe { state.batch_q.as_mut_ptr().add(t * qkv_dim) },
            state.q6k_d_scratch.as_mut_ptr(),
            qkv_dim, hd, current_chunk, ith, nth,
        );
        barrier.wait();
    }

    // ── 3. Q norm + RoPE per token (thread 0) ────────────────────
    if ith == 0 {
        for t in 0..n {
            compute_rope_tables(&mut state.cos_table, &mut state.sin_table,
                seq_len + t, n_rot, rope_theta, freq_factors);
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
    barrier.wait();

    // ── 4. K/V projections + norms + RoPE + cache (WS matmul, thread-0 small ops) ──
    if has_kv {
        let kv_dim = n_kv_heads * head_dim;
        let kv_dim_v = n_kv_heads * head_dim_v;

        // K matmul per token (work-stealing)
        for t in 0..n {
            if ith == 0 {
                matmul::quant_input(
                    &state.batch_x_norm[t * hd..(t + 1) * hd],
                    &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums,
                );
            }
            barrier.wait();
            current_chunk.store(nth as i32, Ordering::Relaxed);
            barrier.wait();
            matvec_step(
                lw.wk_dtype, lw.wk, lw.wk_repacked.as_deref(),
                state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
                unsafe { state.batch_k.as_mut_ptr().add(t * kv_dim) },
                state.q6k_d_scratch.as_mut_ptr(),
                kv_dim, hd, current_chunk, ith, nth,
            );
            barrier.wait();
        }

        // V matmul per token (work-stealing)
        for t in 0..n {
            if ith == 0 {
                matmul::quant_input(
                    &state.batch_x_norm[t * hd..(t + 1) * hd],
                    &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums,
                );
            }
            barrier.wait();
            current_chunk.store(nth as i32, Ordering::Relaxed);
            barrier.wait();
            matvec_step(
                lw.wv_dtype, lw.wv, lw.wv_repacked.as_deref(),
                state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
                unsafe { state.batch_v.as_mut_ptr().add(t * kv_dim_v) },
                state.q6k_d_scratch.as_mut_ptr(),
                kv_dim_v, hd, current_chunk, ith, nth,
            );
            barrier.wait();
        }

        // K/V norms + RoPE + cache store (thread 0)
        if ith == 0 {
            for t in 0..n {
                compute_rope_tables(&mut state.cos_table, &mut state.sin_table,
                    seq_len + t, n_rot, rope_theta, freq_factors);
                if !lw.k_norm.is_null() {
                    for h in 0..n_kv_heads {
                        let off = t * kv_dim + h * head_dim;
                        ffi_inference::gemma4_rmsnorm(
                            unsafe { state.batch_k.as_ptr().add(off) }, lw.k_norm,
                            state.x_norm.as_mut_ptr(), head_dim as i32, model.rms_eps,
                        );
                        state.batch_k[off..off + head_dim].copy_from_slice(&state.x_norm[..head_dim]);
                    }
                }
                for h in 0..n_kv_heads {
                    let off = t * kv_dim_v + h * head_dim_v;
                    super::forward::bare_rmsnorm(&mut state.batch_v[off..off + head_dim_v], model.rms_eps);
                }
                ffi_inference::gemma4_rope(
                    unsafe { state.batch_k.as_mut_ptr().add(t * kv_dim) },
                    state.cos_table.as_ptr(), state.sin_table.as_ptr(),
                    head_dim as i32, n_kv_heads as i32,
                );
            }
            state.cache.store_batch(il, &state.batch_k[..kv_dim * n], &state.batch_v[..kv_dim_v * n], n);
        }
        barrier.wait();
    } else {
        // Shared KV layer — Q norm + RoPE only (already done in step 3)
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
        let kv_scratch_stride = state.kv_scratch_stride;
        let attn_scores_stride = state.attn_scores_stride;

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
                    qkv_dim as i32,       // q_stride
                    qkv_dim as i32,       // out_stride
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
    barrier.wait();

    // ── 6. Wo projection per token (WS matmul) ──────────────────
    for t in 0..n {
        let attn_out_dim = n_heads * head_dim;
        if ith == 0 {
            matmul::quant_input(
                &state.batch_attn_out[t * attn_out_dim..(t + 1) * attn_out_dim],
                &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums,
            );
        }
        barrier.wait();
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matvec_step(
            lw.wo_dtype, lw.wo, lw.wo_repacked.as_deref(),
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            unsafe { state.batch_wo_out.as_mut_ptr().add(t * hd) },
            state.q6k_d_scratch.as_mut_ptr(),
            hd, attn_out_dim, current_chunk, ith, nth,
        );
        barrier.wait();
    }

    // ── 7. Post-attn norm + residual + FFN norm (thread 0) ──────
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
    }
    barrier.wait();

    // ── 8. FFN gate+up per token (WS matmul) ────────────────────
    // Match forward_graph.rs: dual dispatch when both are Q4K, separate otherwise.
    let ffn_dim = model.ffn_dim[il];
    if lw.w_gate_dtype == matmul::GGML_TYPE_Q4_K && lw.w_up_dtype == matmul::GGML_TYPE_Q4_K {
        debug_assert!(
            lw.w_gate_repacked.is_some() == lw.w_up_repacked.is_some(),
            "ffn_gate/ffn_up repack invariant violated in layer {il}"
        );
        for t in 0..n {
            if ith == 0 {
                matmul::quant_input(
                    &state.batch_x_norm[t * hd..(t + 1) * hd],
                    &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums,
                );
            }
            barrier.wait();
            current_chunk.store(nth as i32, Ordering::Relaxed);
            barrier.wait();
            match (lw.w_gate_repacked.as_deref(), lw.w_up_repacked.as_deref()) {
                (Some(g), Some(u)) => matmul_graph::q4k_matvec_dual_8x8_ws(
                    g.as_ptr(), u.as_ptr(),
                    state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
                    unsafe { state.batch_gate.as_mut_ptr().add(t * ffn_dim) },
                    unsafe { state.batch_up.as_mut_ptr().add(t * ffn_dim) },
                    ffn_dim, hd, current_chunk, ith, nth,
                ),
                _ => matmul_graph::q4k_matvec_dual_ws(
                    lw.w_gate, lw.w_up,
                    state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
                    unsafe { state.batch_gate.as_mut_ptr().add(t * ffn_dim) },
                    unsafe { state.batch_up.as_mut_ptr().add(t * ffn_dim) },
                    ffn_dim, hd, current_chunk, ith, nth,
                ),
            }
            barrier.wait();
        }
    } else {
        for t in 0..n {
            if ith == 0 {
                matmul::quant_input(
                    &state.batch_x_norm[t * hd..(t + 1) * hd],
                    &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums,
                );
            }
            barrier.wait();
            current_chunk.store(nth as i32, Ordering::Relaxed);
            barrier.wait();
            matvec_step(
                lw.w_gate_dtype, lw.w_gate, lw.w_gate_repacked.as_deref(),
                state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
                unsafe { state.batch_gate.as_mut_ptr().add(t * ffn_dim) },
                state.q6k_d_scratch.as_mut_ptr(),
                ffn_dim, hd, current_chunk, ith, nth,
            );
            barrier.wait();
            if ith == 0 {
                matmul::quant_input(
                    &state.batch_x_norm[t * hd..(t + 1) * hd],
                    &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums,
                );
            }
            barrier.wait();
            current_chunk.store(nth as i32, Ordering::Relaxed);
            barrier.wait();
            matvec_step(
                lw.w_up_dtype, lw.w_up, lw.w_up_repacked.as_deref(),
                state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
                unsafe { state.batch_up.as_mut_ptr().add(t * ffn_dim) },
                state.q6k_d_scratch.as_mut_ptr(),
                ffn_dim, hd, current_chunk, ith, nth,
            );
            barrier.wait();
        }
    }

    // ── 9. GELU + quant + down matmul per token (WS) ────────────
    for t in 0..n {
        if ith == 0 {
            ffi_inference::gelu_mul(
                state.batch_gate[t * ffn_dim..].as_ptr(),
                state.batch_up[t * ffn_dim..].as_ptr(),
                state.batch_gate[t * ffn_dim..].as_mut_ptr(), ffn_dim as i32,
            );
            matmul::quant_input(
                &state.batch_gate[t * ffn_dim..(t + 1) * ffn_dim],
                &mut state.ffn_q8_qs, &mut state.ffn_q8_d, &mut state.ffn_q8_bsums,
            );
        }
        barrier.wait();
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matvec_step(
            lw.w_down_dtype, lw.w_down, lw.w_down_repacked.as_deref(),
            state.ffn_q8_qs.as_ptr(), state.ffn_q8_d.as_ptr(), state.ffn_q8_bsums.as_ptr(),
            unsafe { state.batch_down.as_mut_ptr().add(t * hd) },
            state.q6k_d_scratch.as_mut_ptr(),
            hd, ffn_dim, current_chunk, ith, nth,
        );
        barrier.wait();
    }

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
    barrier.wait();
}
