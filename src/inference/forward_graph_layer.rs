//! Per-layer decode forward with graph threading — extracted from forward_graph.rs.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::{compute_rope_tables, Gemma4State};
use crate::inference::matmul;
use crate::inference::matmul_graph;
use crate::inference::threadpool::SpinBarrier;
use crate::kernels::ffi_inference;

/// Parallel Q8K quantization across threads, split by 256-element blocks.
#[inline]
pub(super) fn parallel_quant_decode(
    src: *const f32, qs: *mut i8, d: *mut f32, bsums: *mut i16,
    dim: usize, ith: usize, nth: usize,
) {
    let nb = dim / 256;
    let per = (nb + nth - 1) / nth;
    let start = ith * per;
    let end = (start + per).min(nb);
    if start < nb {
        let n = (end - start) * 256;
        unsafe {
            ffi_inference::quant_f32_q8k(
                src.add(start * 256), qs.add(start * 256),
                d.add(start), bsums.add(start * 16), n as i32,
            );
        }
    }
}

/// Dispatch a single matvec_ws call through either the repacked 8x8 path
/// or the standard 4-row matvec_ws fallback, depending on whether the
/// weight has been repacked at model load time.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn matvec_step(
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

/// Per-layer forward with optional per-op timing (thread 0 only).
#[allow(clippy::too_many_arguments)]
pub(super) fn layer_forward_graph_timed(
    state: &mut Gemma4State,
    model: &Gemma4Model,
    il: usize,
    pos: usize,
    barrier: &SpinBarrier,
    current_chunk: &AtomicI32,
    ith: usize,
    nth: usize,
    timing: bool,
    t_norm_quant: &mut u64, t_q: &mut u64, t_q_norm_rope: &mut u64,
    t_kv: &mut u64, t_kv_norm_cache: &mut u64,
    t_attn: &mut u64, t_wo_quant: &mut u64, t_wo: &mut u64,
    t_post_attn: &mut u64, t_gate_up: &mut u64, t_gelu_quant: &mut u64,
    t_down: &mut u64, t_post_ffn_ple: &mut u64,
) {
    use std::time::Instant;
    macro_rules! t { () => { if timing { Some(Instant::now()) } else { None } }; }
    macro_rules! acc { ($s:expr, $f:expr) => { if let Some(s) = $s { *$f += s.elapsed().as_micros() as u64; } }; }
    let hd = model.hidden_dim;
    let n_heads = model.n_heads;
    let n_kv_heads = model.n_kv_heads;
    let gqa_ratio = n_heads / n_kv_heads;
    let lw = &model.layers[il];
    let head_dim = model.head_dim_k[il];
    let head_dim_v = model.head_dim_v[il];
    let has_kv = model.kv_shared_source[il].is_none();

    // ── 1. RoPE tables + attn_norm (thread 0) + parallel quant ───
    let t0 = t!();
    if ith == 0 {
        let n_rot = if model.is_swa[il] { model.rope_dim_swa } else { model.rope_dim_global };
        let rope_theta = if model.is_swa[il] { model.rope_theta_swa } else { model.rope_theta_global };
        let freq_factors = if !model.is_swa[il] { model.rope_freqs.as_deref() } else { None };
        compute_rope_tables(&mut state.cos_table, &mut state.sin_table, pos, n_rot, rope_theta, freq_factors);

        ffi_inference::gemma4_rmsnorm(
            state.x.as_ptr(), lw.attn_norm, state.x_norm.as_mut_ptr(), hd as i32, model.rms_eps,
        );
    }
    barrier.wait(); // x_norm ready
    parallel_quant_decode(
        state.x_norm.as_ptr(), state.q8_qs.as_mut_ptr(),
        state.q8_d.as_mut_ptr(), state.q8_bsums.as_mut_ptr(),
        hd, ith, nth,
    );
    barrier.wait();

    acc!(t0, t_norm_quant);
    // ── 2. Q projection (work-stealing) ──────────────────────────
    let t0 = t!();
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matvec_step(
        lw.wq_dtype, lw.wq, lw.wq_repacked.as_deref(),
        state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
        state.q.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        n_heads * head_dim, hd,
        current_chunk, ith, nth,
    );
    barrier.wait();

    acc!(t0, t_q);
    // ── 3. Q norm + RoPE (thread 0) ──────────────────────────────
    let t0 = t!();
    if ith == 0 {
        // Q norm per-head
        if !lw.q_norm.is_null() {
            for h in 0..n_heads {
                let off = h * head_dim;
                ffi_inference::gemma4_rmsnorm(
                    unsafe { state.q.as_ptr().add(off) },
                    lw.q_norm,
                    state.x_norm.as_mut_ptr(), // scratch
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

    acc!(t0, t_q_norm_rope);
    // ── 4. K/V + norms + RoPE + cache (thread 0 for small ops, WS for matmul) ──
    let t0 = t!();
    if has_kv {
        let kv_dim = n_kv_heads * head_dim;
        let kv_dim_v = n_kv_heads * head_dim_v;

        // K matmul (work-stealing)
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matvec_step(
            lw.wk_dtype, lw.wk, lw.wk_repacked.as_deref(),
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.k.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            kv_dim, hd,
            current_chunk, ith, nth,
        );
        barrier.wait();

        // V matmul (work-stealing)
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matvec_step(
            lw.wv_dtype, lw.wv, lw.wv_repacked.as_deref(),
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.v.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            kv_dim_v, hd,
            current_chunk, ith, nth,
        );
        barrier.wait();

        acc!(t0, t_kv);
        // K norm + V bare norm + K rope + cache store (thread 0).
        if ith == 0 {
            if !lw.k_norm.is_null() {
                for h in 0..n_kv_heads {
                    let off = h * head_dim;
                    ffi_inference::gemma4_rmsnorm(
                        unsafe { state.k.as_ptr().add(off) }, lw.k_norm,
                        state.x_norm.as_mut_ptr(), head_dim as i32, model.rms_eps,
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

    acc!(t0, t_kv_norm_cache);
    // ── 5. Attention (split by heads across threads) ─────────────
    let t0 = t!();
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

    acc!(t0, t_attn);
    // ── 6. Wo: parallel quant + matmul (work-stealing) ───────────
    let t0 = t!();
    {
        let attn_out_dim = n_heads * head_dim;
        parallel_quant_decode(
            state.attn_out.as_ptr(), state.q8_qs.as_mut_ptr(),
            state.q8_d.as_mut_ptr(), state.q8_bsums.as_mut_ptr(),
            attn_out_dim, ith, nth,
        );
    }
    barrier.wait();

    acc!(t0, t_wo_quant);
    let t0 = t!();
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matvec_step(
        lw.wo_dtype, lw.wo, lw.wo_repacked.as_deref(),
        state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
        state.wo_out.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, n_heads * head_dim,
        current_chunk, ith, nth,
    );
    barrier.wait();

    acc!(t0, t_wo);
    // ── 7. Post-attn norm + residual + FFN norm (T0) + parallel quant
    let t0 = t!();
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
        ffi_inference::gemma4_rmsnorm(
            state.attn_res.as_ptr(), lw.ffn_norm, state.x_norm.as_mut_ptr(),
            hd as i32, model.rms_eps,
        );
    }
    barrier.wait(); // x_norm ready
    parallel_quant_decode(
        state.x_norm.as_ptr(), state.q8_qs.as_mut_ptr(),
        state.q8_d.as_mut_ptr(), state.q8_bsums.as_mut_ptr(),
        hd, ith, nth,
    );
    barrier.wait();

    acc!(t0, t_post_attn);
    // ── 8. FFN gate+up (work-stealing) ───────────────────────────
    let t0 = t!();
    let ffn_dim = model.ffn_dim[il];
    if lw.w_gate_dtype == matmul::GGML_TYPE_Q4_K && lw.w_up_dtype == matmul::GGML_TYPE_Q4_K {
        debug_assert!(
            lw.w_gate_repacked.is_some() == lw.w_up_repacked.is_some(),
            "ffn_gate/ffn_up repack invariant violated in layer {il}"
        );
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        match (lw.w_gate_repacked.as_deref(), lw.w_up_repacked.as_deref()) {
            (Some(g), Some(u)) => matmul_graph::q4k_matvec_dual_8x8_ws(
                g.as_ptr(), u.as_ptr(),
                state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
                state.gate.as_mut_ptr(), state.up.as_mut_ptr(),
                ffn_dim, hd,
                current_chunk, ith, nth,
            ),
            _ => matmul_graph::q4k_matvec_dual_ws(
                lw.w_gate, lw.w_up,
                state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
                state.gate.as_mut_ptr(), state.up.as_mut_ptr(),
                ffn_dim, hd,
                current_chunk, ith, nth,
            ),
        }
        barrier.wait();
    } else {
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matvec_step(
            lw.w_gate_dtype, lw.w_gate, lw.w_gate_repacked.as_deref(),
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.gate.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            ffn_dim, hd, current_chunk, ith, nth,
        );
        barrier.wait();
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matvec_step(
            lw.w_up_dtype, lw.w_up, lw.w_up_repacked.as_deref(),
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.up.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            ffn_dim, hd, current_chunk, ith, nth,
        );
        barrier.wait();
    }

    acc!(t0, t_gate_up);
    // ── 9. Fused parallel GELU + quant ──────────────────────────
    let t0 = t!();
    {
        let nb = ffn_dim / 256;
        let per = (nb + nth - 1) / nth;
        let blk_start = ith * per;
        let blk_end = (blk_start + per).min(nb);
        if blk_start < nb {
            let elem_start = blk_start * 256;
            let n_elem = (blk_end - blk_start) * 256;
            unsafe {
                ffi_inference::gelu_mul(
                    state.gate.as_ptr().add(elem_start),
                    state.up.as_ptr().add(elem_start),
                    state.gate.as_mut_ptr().add(elem_start),
                    n_elem as i32,
                );
                ffi_inference::quant_f32_q8k(
                    state.gate.as_ptr().add(elem_start),
                    state.ffn_q8_qs.as_mut_ptr().add(elem_start),
                    state.ffn_q8_d.as_mut_ptr().add(blk_start),
                    state.ffn_q8_bsums.as_mut_ptr().add(blk_start * 16),
                    n_elem as i32,
                );
            }
        }
    }
    barrier.wait();

    acc!(t0, t_gelu_quant);
    let t0 = t!();
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    if let Some(ref q6k_buf) = lw.w_down_q6k_repacked {
        matmul_graph::q6k_repacked_batch_ws(
            q6k_buf.as_ptr(), lw.w_down,
            state.ffn_q8_qs.as_ptr(), state.ffn_q8_d.as_ptr(), state.ffn_q8_bsums.as_ptr(),
            state.down.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            hd, ffn_dim, 1, hd,
            current_chunk, ith, nth,
        );
    } else {
        matvec_step(
            lw.w_down_dtype, lw.w_down, lw.w_down_repacked.as_deref(),
            state.ffn_q8_qs.as_ptr(), state.ffn_q8_d.as_ptr(), state.ffn_q8_bsums.as_ptr(),
            state.down.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            hd, ffn_dim,
            current_chunk, ith, nth,
        );
    }
    barrier.wait();

    acc!(t0, t_down);
    // ── 10. Post-FFN norm + residual + PLE + scale (thread 0) ───
    let t0 = t!();
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
            matmul::matvec_maybe_repacked(
                lw.inp_gate_dtype, lw.inp_gate, lw.inp_gate_repacked.as_deref(),
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
            matmul::matvec_maybe_repacked(
                lw.proj_dtype, lw.proj, lw.proj_repacked.as_deref(),
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
    acc!(t0, t_post_ffn_ple);
}
