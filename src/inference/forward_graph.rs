//! Graph-threaded forward pass — all threads execute together.
//!
//! Replaces per-op pool.run() dispatch with a single graph execution where
//! all threads loop through the forward pass with spin-barriers between ops.
//! Matches llama.cpp's ggml_graph_compute_thread pattern.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::{compute_rope_tables, Gemma4State};
use crate::inference::matmul;
use crate::inference::matmul_graph;
use crate::inference::dequant;
use crate::kernels::ffi_inference;
use crate::inference::threadpool::SpinBarrier;

/// Shared context for graph-threaded forward pass.
/// All pointers remain valid for the duration of forward_one_graph.
struct FwdCtx<'a> {
    state: &'a mut Gemma4State,
    model: &'a Gemma4Model,
    token_id: u32,
    pos: usize,
    barrier: &'a SpinBarrier,
    current_chunk: &'a AtomicI32,
}

// FwdCtx contains raw pointers via model/state. We guarantee disjoint access
// by ith/nth splitting within each op. Each thread accesses different output
// ranges, and shared reads (weights, input) are immutable.
unsafe impl<'a> Send for FwdCtx<'a> {}
unsafe impl<'a> Sync for FwdCtx<'a> {}

/// Run one decode step using graph-loop threading.
/// Called by ONE thread (dispatcher), which then wakes all workers.
/// All n_threads (including dispatcher as ith=0) execute this together.
pub(crate) fn forward_one_inner(
    state: &mut Gemma4State,
    model: &Gemma4Model,
    token_id: u32,
    barrier: &SpinBarrier,
    current_chunk: &AtomicI32,
    ith: usize,
    nth: usize,
) {
    let hd = model.hidden_dim;
    let pos = state.cache.seq_len();

    // ── Pre-loop: embed + scale + PLE (thread 0 only) ────────────
    if ith == 0 {
        dequant::q6k_embed_lookup(model.embed_weight, token_id as usize, &mut state.x, hd);
        let embed_scale = (hd as f32).sqrt();
        ffi_inference::vec_scale_f32(
            state.x.as_ptr(), state.x.as_mut_ptr(), embed_scale, hd as i32,
        );
        state.prepare_ple(model, token_id);
    }
    barrier.wait();

    // ── Per-layer transformer blocks ─────────────────────────────
    for il in 0..model.n_layers {
        layer_forward_graph(state, model, il, pos, barrier, current_chunk, ith, nth);
    }

    // ── Post-loop: final norm (thread 0) ─────────────────────────
    if ith == 0 {
        ffi_inference::gemma4_rmsnorm(
            state.x.as_ptr(),
            model.norm_weight,
            state.x_norm.as_mut_ptr(),
            hd as i32,
            model.rms_eps,
        );
        matmul::quant_input(
            &state.x_norm,
            &mut state.q8_qs,
            &mut state.q8_d,
            &mut state.q8_bsums,
        );
    }
    barrier.wait();

    // ── Output matmul (work-stealing, all threads) ───────────────
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matmul_graph::matvec_ws(
        model.embed_dtype, model.embed_weight,
        state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
        state.logits.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        model.vocab_size, hd,
        current_chunk, ith, nth,
    );
    barrier.wait();

    // ── Softcap (thread 0) ───────────────────────────────────────
    if ith == 0 {
        if model.logit_softcap > 0.0 {
            ffi_inference::softcap_f32(
                state.logits.as_mut_ptr(), model.vocab_size as i32, model.logit_softcap,
            );
        }
        state.cache.advance();
    }
    barrier.wait();
}

/// Per-layer forward matching layer_forward exactly, but with ith/nth + barriers.
fn layer_forward_graph(
    state: &mut Gemma4State,
    model: &Gemma4Model,
    il: usize,
    pos: usize,
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

    // ── 1. RoPE tables + attn_norm + quant (thread 0) ────────────
    if ith == 0 {
        let n_rot = if model.is_swa[il] { model.rope_dim_swa } else { model.rope_dim_global };
        let rope_theta = if model.is_swa[il] { model.rope_theta_swa } else { model.rope_theta_global };
        let freq_factors = if !model.is_swa[il] { model.rope_freqs.as_deref() } else { None };
        compute_rope_tables(&mut state.cos_table, &mut state.sin_table, pos, n_rot, rope_theta, freq_factors);

        ffi_inference::gemma4_rmsnorm(
            state.x.as_ptr(), lw.attn_norm, state.x_norm.as_mut_ptr(), hd as i32, model.rms_eps,
        );
        matmul::quant_input(&state.x_norm, &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums);
    }
    barrier.wait();

    // ── 2. Q projection (work-stealing) ──────────────────────────
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matmul_graph::matvec_ws(
        lw.wq_dtype, lw.wq,
        state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
        state.q.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        n_heads * head_dim, hd,
        current_chunk, ith, nth,
    );
    barrier.wait();

    // ── 3. Q norm + RoPE (thread 0) ──────────────────────────────
    if ith == 0 {
        // Q norm per-head
        if !lw.q_norm.is_null() {
            for h in 0..n_heads {
                let off = h * head_dim;
                ffi_inference::gemma4_rmsnorm(
                    unsafe { state.q.as_ptr().add(off) },
                    lw.q_norm,
                    unsafe { state.x_norm.as_mut_ptr() }, // scratch
                    head_dim as i32, model.rms_eps,
                );
                state.q[off..off + head_dim].copy_from_slice(&state.x_norm[..head_dim]);
            }
        }
        ffi_inference::gemma4_rope(
            state.q.as_mut_ptr(), state.cos_table.as_ptr(), state.sin_table.as_ptr(),
            head_dim as i32, n_heads as i32,
        );
    }
    barrier.wait();

    // ── 4. K/V + norms + RoPE + cache (thread 0 for small ops, WS for matmul) ──
    if has_kv {
        let kv_dim = n_kv_heads * head_dim;
        let kv_dim_v = n_kv_heads * head_dim_v;

        // K matmul (work-stealing)
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matmul_graph::matvec_ws(
            lw.wk_dtype, lw.wk,
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.k.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            kv_dim, hd,
            current_chunk, ith, nth,
        );
        barrier.wait();

        // V matmul (work-stealing)
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matmul_graph::matvec_ws(
            lw.wv_dtype, lw.wv,
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.v.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            kv_dim_v, hd,
            current_chunk, ith, nth,
        );
        barrier.wait();

        // K norm + V bare norm + K rope + cache store (thread 0)
        if ith == 0 {
            if !lw.k_norm.is_null() {
                for h in 0..n_kv_heads {
                    let off = h * head_dim;
                    ffi_inference::gemma4_rmsnorm(
                        unsafe { state.k.as_ptr().add(off) }, lw.k_norm,
                        unsafe { state.x_norm.as_mut_ptr() }, head_dim as i32, model.rms_eps,
                    );
                    state.k[off..off + head_dim].copy_from_slice(&state.x_norm[..head_dim]);
                }
            }
            // V bare norm
            for h in 0..n_kv_heads {
                let off = h * head_dim_v;
                super::forward::bare_rmsnorm(&mut state.v[off..off + head_dim_v], model.rms_eps);
            }
            ffi_inference::gemma4_rope(
                state.k.as_mut_ptr(), state.cos_table.as_ptr(), state.sin_table.as_ptr(),
                head_dim as i32, n_kv_heads as i32,
            );
            state.cache.store(il, &state.k[..kv_dim], &state.v[..kv_dim_v]);
        }
        barrier.wait();
    }

    // ── 5. Attention (split by heads across threads) ─────────────
    {
        let attn_scale = 1.0f32;
        let attn_len = state.cache.attn_len(il);
        let k_ptr = state.cache.k_ptr(il);
        let v_ptr = state.cache.v_ptr(il);
        let kv_dim = n_kv_heads * head_dim;
        let stride_kv = kv_dim;
        let kv_scratch_stride = state.kv_scratch_stride;
        let attn_scores_stride = state.attn_scores_stride;

        // Split heads across threads
        let per = (n_heads + nth - 1) / nth;
        let h_start = ith * per;
        let h_end = ((ith + 1) * per).min(n_heads);

        for h in h_start..h_end {
            let kv_h = h / gqa_ratio;
            let q_off = h * head_dim;

            let q_slice_ptr = unsafe { state.q.as_ptr().add(q_off) };
            let kv_scratch_base = unsafe { state.kv_f32_scratch.as_mut_ptr().add(ith * kv_scratch_stride) };
            let attn_scores_base = unsafe { state.attn_scores.as_mut_ptr().add(ith * attn_scores_stride) };

            for p in 0..attn_len {
                let k_offset = p * stride_kv + kv_h * head_dim;
                let k_src = unsafe { k_ptr.add(k_offset) };
                unsafe { ffi_inference::f16_to_f32(k_src, kv_scratch_base, head_dim as i32); }
                let dot = ffi_inference::f32_dot(q_slice_ptr, kv_scratch_base as *const f32, head_dim as i32);
                unsafe { *attn_scores_base.add(p) = dot; }
            }

            unsafe { ffi_inference::softmax_f32(attn_scores_base, attn_len as i32, attn_scale); }

            let out_base = unsafe { state.attn_out.as_mut_ptr().add(q_off) };
            unsafe { std::ptr::write_bytes(out_base, 0, head_dim); }
            for p in 0..attn_len {
                let v_offset = p * stride_kv + kv_h * head_dim;
                let v_src = unsafe { v_ptr.add(v_offset) };
                unsafe { ffi_inference::f16_to_f32(v_src, kv_scratch_base, head_dim as i32); }
                let s = unsafe { *attn_scores_base.add(p) };
                ffi_inference::f32_dot_acc(out_base, kv_scratch_base as *const f32, s, head_dim as i32);
            }
        }
    }
    barrier.wait();

    // ── 6. Wo matmul (work-stealing) ─────────────────────────────
    if ith == 0 {
        let attn_out_dim = n_heads * head_dim;
        matmul::quant_input(
            &state.attn_out[..attn_out_dim],
            &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums,
        );
    }
    barrier.wait();

    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matmul_graph::matvec_ws(
        lw.wo_dtype, lw.wo,
        state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
        state.wo_out.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, n_heads * head_dim,
        current_chunk, ith, nth,
    );
    barrier.wait();

    // ── 7. Post-attn norm + residual (thread 0) ─────────────────
    if ith == 0 {
        if !lw.post_attn_norm.is_null() {
            ffi_inference::gemma4_rmsnorm(
                state.wo_out.as_ptr(), lw.post_attn_norm, state.x_norm.as_mut_ptr(),
                hd as i32, model.rms_eps,
            );
            ffi_inference::vec_add_f32(
                state.x_norm.as_ptr(), state.x.as_ptr(), state.attn_res.as_mut_ptr(), hd as i32,
            );
        } else {
            ffi_inference::vec_add_f32(
                state.wo_out.as_ptr(), state.x.as_ptr(), state.attn_res.as_mut_ptr(), hd as i32,
            );
        }

        // FFN norm
        ffi_inference::gemma4_rmsnorm(
            state.attn_res.as_ptr(), lw.ffn_norm, state.x_norm.as_mut_ptr(),
            hd as i32, model.rms_eps,
        );
        matmul::quant_input(&state.x_norm, &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums);
    }
    barrier.wait();

    // ── 8. FFN gate+up (work-stealing) ───────────────────────────
    let ffn_dim = model.ffn_dim[il];
    if lw.w_gate_dtype == matmul::GGML_TYPE_Q4_K && lw.w_up_dtype == matmul::GGML_TYPE_Q4_K {
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matmul_graph::q4k_matvec_dual_ws(
            lw.w_gate, lw.w_up,
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.gate.as_mut_ptr(), state.up.as_mut_ptr(),
            ffn_dim, hd,
            current_chunk, ith, nth,
        );
        barrier.wait();
    } else {
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matmul_graph::matvec_ws(
            lw.w_gate_dtype, lw.w_gate,
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.gate.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            ffn_dim, hd, current_chunk, ith, nth,
        );
        barrier.wait();
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matmul_graph::matvec_ws(
            lw.w_up_dtype, lw.w_up,
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.up.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            ffn_dim, hd, current_chunk, ith, nth,
        );
        barrier.wait();
    }

    // ── 9. GELU + quant + down matmul ────────────────────────────
    if ith == 0 {
        ffi_inference::gelu_mul(
            state.gate.as_ptr(), state.up.as_ptr(), state.gate.as_mut_ptr(), ffn_dim as i32,
        );
        matmul::quant_input(
            &state.gate[..ffn_dim],
            &mut state.ffn_q8_qs, &mut state.ffn_q8_d, &mut state.ffn_q8_bsums,
        );
    }
    barrier.wait();

    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matmul_graph::matvec_ws(
        lw.w_down_dtype, lw.w_down,
        state.ffn_q8_qs.as_ptr(), state.ffn_q8_d.as_ptr(), state.ffn_q8_bsums.as_ptr(),
        state.down.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, ffn_dim,
        current_chunk, ith, nth,
    );
    barrier.wait();

    // ── 10. Post-FFN norm + residual + PLE + scale (thread 0) ───
    if ith == 0 {
        if !lw.post_ffn_norm.is_null() {
            ffi_inference::gemma4_rmsnorm(
                state.down.as_ptr(), lw.post_ffn_norm, state.x_norm.as_mut_ptr(),
                hd as i32, model.rms_eps,
            );
            ffi_inference::vec_add_f32(
                state.x_norm.as_ptr(), state.attn_res.as_ptr(), state.x.as_mut_ptr(), hd as i32,
            );
        } else {
            ffi_inference::vec_add_f32(
                state.down.as_ptr(), state.attn_res.as_ptr(), state.x.as_mut_ptr(), hd as i32,
            );
        }

        // PLE
        if model.ple_dim > 0 && !lw.inp_gate.is_null() && !lw.proj.is_null() {
            let ple_dim = model.ple_dim;
            let ple_off = il * ple_dim;

            matmul::quant_input(&state.x[..hd], &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums);
            matmul::matvec(
                lw.inp_gate_dtype, lw.inp_gate,
                &state.q8_qs, &state.q8_d, &state.q8_bsums,
                &mut state.ple_gate, &mut state.q6k_d_scratch, ple_dim, hd,
            );

            ffi_inference::gelu_mul(
                state.ple_gate.as_ptr(), state.ple_signal[ple_off..].as_ptr(),
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
                    state.ple_out.as_ptr(), lw.post_norm, state.ple_out.as_mut_ptr(),
                    hd as i32, model.rms_eps,
                );
            }
            ffi_inference::vec_add_f32(
                state.x.as_ptr(), state.ple_out.as_ptr(), state.x.as_mut_ptr(), hd as i32,
            );
        }

        // Layer output scale
        let out_scale = lw.layer_output_scale;
        if out_scale != 1.0 {
            ffi_inference::vec_scale_f32(
                state.x.as_ptr(), state.x.as_mut_ptr(), out_scale, hd as i32,
            );
        }
    }
    barrier.wait();
}
