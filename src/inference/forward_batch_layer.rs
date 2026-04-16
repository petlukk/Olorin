//! Per-layer batched forward — parallel quant + true GEMM.
//! Matches llama.cpp repack.cpp:4296-4384.

use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Instant;
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::{compute_rope_tables_into, Gemma4State};
use crate::inference::matmul;
use crate::inference::matmul_graph;
use crate::inference::threadpool::SpinBarrier;
use crate::kernels::ffi_inference;

pub(crate) use super::forward_batch_helpers::{
    BatchLayerTiming, parallel_batch_quant, repack_q8_for_gemm, matvec_batch_step,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward_batch(
    state: &mut Gemma4State, model: &Gemma4Model,
    il: usize, n: usize, seq_len: usize,
    barrier: &SpinBarrier, current_chunk: &AtomicI32, ith: usize, nth: usize,
    timing: Option<&mut BatchLayerTiming>,
) {
    let timing_on = timing.is_some();
    // Reborrow helper — we pass `timing` through as Option to avoid borrow issues.
    // Macro captures start instant and accumulates into a field.
    macro_rules! t_start { () => { if timing_on { Some(Instant::now()) } else { None } }; }
    macro_rules! t_accum {
        ($start:expr, $field:ident, $tm:expr) => {
            if let (Some(s), Some(ref mut t)) = ($start, $tm) { t.$field += s.elapsed().as_micros() as u64; }
        };
    }
    // We need a raw pointer to timing to work around borrow checker with barriers.
    // SAFETY: only thread 0 accesses timing, and it's valid for the entire call.
    let tp: *mut BatchLayerTiming = match timing {
        Some(t) => t as *mut _,
        None => std::ptr::null_mut(),
    };
    macro_rules! tm { () => { if tp.is_null() { None } else { Some(unsafe { &mut *tp }) } }; }
    let hd = model.hidden_dim;
    let n_heads = model.n_heads;
    let n_kv_heads = model.n_kv_heads;
    let gqa_ratio = n_heads / n_kv_heads;
    let lw = &model.layers[il];
    let head_dim = model.head_dim_k[il];
    let head_dim_v = model.head_dim_v[il];
    let has_kv = model.kv_shared_source[il].is_none();
    let qkv_dim = n_heads * head_dim;
    let n_pad = (n + 3) & !3; // round up to 4 for GEMM

    let n_rot = if model.is_swa[il] { model.rope_dim_swa } else { model.rope_dim_global };
    let rope_theta = if model.is_swa[il] { model.rope_theta_swa } else { model.rope_theta_global };
    let freq_factors = if !model.is_swa[il] { model.rope_freqs.as_deref() } else { None };

    // ── 1a. Attn norm (all threads, token-strided) ────────────────
    let t0 = t_start!();
    {
        let mut t = ith;
        while t < n {
            ffi_inference::gemma4_rmsnorm(
                state.batch_x[t * hd..].as_ptr(), lw.attn_norm,
                state.batch_x_norm[t * hd..].as_mut_ptr(), hd as i32, model.rms_eps,
            );
            t += nth;
        }
    }
    barrier.wait(); // B1
    if ith == 0 { t_accum!(t0, attn_norm, tm!()); }
    // ── 1b. Parallel quant (all threads) ────────────────────────
    let t0 = t_start!();
    parallel_batch_quant(
        &state.batch_x_norm, hd, n, n_pad,
        &mut state.batch_q8_qs, &mut state.batch_q8_d, &mut state.batch_q8_bsums,
        ith, nth,
    );
    barrier.wait(); // B2
    if ith == 0 { t_accum!(t0, quant_input, tm!()); }
    // ── 1c. Q8K repack + chunk store ────────────────────────────
    let t0 = t_start!();
    if ith == 0 {
        repack_q8_for_gemm(
            &state.batch_q8_qs, &state.batch_q8_d, &state.batch_q8_bsums,
            &mut state.batch_q8_a, hd, n_pad,
        );
    }
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait(); // B3
    if ith == 0 { t_accum!(t0, repack_q8, tm!()); }
    // ── 2. Q projection GEMM ────────────────────────────────────
    let t0 = t_start!();
    matvec_batch_step(
        lw.wq_repacked.as_deref(), lw.wq_dtype, lw.wq,
        state.batch_q8_a.as_ptr(),
        state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
        state.batch_q8_bsums.as_ptr(),
        state.batch_q.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        qkv_dim, hd, n, n_pad, qkv_dim,
        current_chunk, ith, nth,
    );
    barrier.wait(); // B4
    if ith == 0 { t_accum!(t0, gemm_q, tm!()); }
    // ── 3. Q norm + RoPE (all threads, token-strided) ─────────────
    let rope_half = n_rot / 2;
    let t0 = t_start!();
    {
        let mut t = ith;
        while t < n {
            let cos_off = ith * rope_half;
            let sin_off = ith * rope_half;
            compute_rope_tables_into(
                &mut state.batch_cos_tables[cos_off..cos_off + rope_half],
                &mut state.batch_sin_tables[sin_off..sin_off + rope_half],
                seq_len + t, n_rot, rope_theta, freq_factors,
            );
            if !lw.q_norm.is_null() {
                let scratch_off = ith * head_dim;
                for h in 0..n_heads {
                    let off = t * qkv_dim + h * head_dim;
                    ffi_inference::gemma4_rmsnorm(
                        unsafe { state.batch_q.as_ptr().add(off) }, lw.q_norm,
                        state.batch_head_scratch[scratch_off..scratch_off + head_dim].as_mut_ptr(),
                        head_dim as i32, model.rms_eps,
                    );
                    state.batch_q[off..off + head_dim]
                        .copy_from_slice(&state.batch_head_scratch[scratch_off..scratch_off + head_dim]);
                }
            }
            ffi_inference::gemma4_rope(
                unsafe { state.batch_q.as_mut_ptr().add(t * qkv_dim) },
                state.batch_cos_tables[cos_off..].as_ptr(),
                state.batch_sin_tables[sin_off..].as_ptr(),
                head_dim as i32, n_heads as i32,
            );
            t += nth;
        }
    }
    barrier.wait(); // B5
    if ith == 0 { t_accum!(t0, q_norm_rope, tm!()); }
    // ── 4. K/V projections + norms + RoPE + cache ───────────────
    if has_kv {
        let kv_dim = n_kv_heads * head_dim;
        let kv_dim_v = n_kv_heads * head_dim_v;

        // K — reuse Q8K + repack from step 1
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait(); // B6
        let t0 = t_start!();
        matvec_batch_step(
            lw.wk_repacked.as_deref(), lw.wk_dtype, lw.wk,
            state.batch_q8_a.as_ptr(),
            state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
            state.batch_q8_bsums.as_ptr(),
            state.batch_k.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            kv_dim, hd, n, n_pad, kv_dim,
            current_chunk, ith, nth,
        );
        barrier.wait(); // B7
        if ith == 0 { t_accum!(t0, gemm_k, tm!()); }

        // V — reuse Q8K + repack from step 1
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait(); // B8
        let t0 = t_start!();
        matvec_batch_step(
            lw.wv_repacked.as_deref(), lw.wv_dtype, lw.wv,
            state.batch_q8_a.as_ptr(),
            state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
            state.batch_q8_bsums.as_ptr(),
            state.batch_v.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            kv_dim_v, hd, n, n_pad, kv_dim_v,
            current_chunk, ith, nth,
        );
        barrier.wait(); // B9
        if ith == 0 { t_accum!(t0, gemm_v, tm!()); }

        // K/V norms + RoPE (all threads, token-strided)
        let t0 = t_start!();
        {
            let mut t = ith;
            while t < n {
                let cos_off = ith * rope_half;
                let sin_off = ith * rope_half;
                compute_rope_tables_into(
                    &mut state.batch_cos_tables[cos_off..cos_off + rope_half],
                    &mut state.batch_sin_tables[sin_off..sin_off + rope_half],
                    seq_len + t, n_rot, rope_theta, freq_factors,
                );
                if !lw.k_norm.is_null() {
                    let scratch_off = ith * head_dim;
                    for h in 0..n_kv_heads {
                        let off = t * kv_dim + h * head_dim;
                        ffi_inference::gemma4_rmsnorm(
                            unsafe { state.batch_k.as_ptr().add(off) }, lw.k_norm,
                            state.batch_head_scratch[scratch_off..scratch_off + head_dim].as_mut_ptr(),
                            head_dim as i32, model.rms_eps,
                        );
                        state.batch_k[off..off + head_dim]
                            .copy_from_slice(&state.batch_head_scratch[scratch_off..scratch_off + head_dim]);
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
                    state.batch_cos_tables[cos_off..].as_ptr(),
                    state.batch_sin_tables[sin_off..].as_ptr(),
                    head_dim as i32, n_kv_heads as i32,
                );
                t += nth;
            }
        }
        barrier.wait(); // sync before cache store
        if ith == 0 {
            state.cache.store_batch(
                il, &state.batch_k[..kv_dim * n], &state.batch_v[..kv_dim_v * n], n,
            );
        }
        barrier.wait(); // B10
        if ith == 0 { t_accum!(t0, kv_norm_rope_cache, tm!()); }
    }

    // ── 5. Attention (heads split across threads, fused kernel) ──
    let t0 = t_start!();
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
                    qkv_dim as i32,
                    qkv_dim as i32,
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
    barrier.wait(); // B11
    if ith == 0 { t_accum!(t0, attention, tm!()); }
    // ── 6. Wo: parallel quant + GEMM ────────────────────────────
    let t0 = t_start!();
    {
        let ao_dim = n_heads * head_dim;
        parallel_batch_quant(
            &state.batch_attn_out, ao_dim, n, n_pad,
            &mut state.batch_q8_qs, &mut state.batch_q8_d, &mut state.batch_q8_bsums,
            ith, nth,
        );
    }
    barrier.wait(); // B12
    if ith == 0 { t_accum!(t0, quant_wo, tm!()); }
    let t0 = t_start!();
    if ith == 0 {
        let ao_dim = n_heads * head_dim;
        repack_q8_for_gemm(
            &state.batch_q8_qs, &state.batch_q8_d, &state.batch_q8_bsums,
            &mut state.batch_q8_a, ao_dim, n_pad,
        );
    }
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait(); // B13
    if ith == 0 { t_accum!(t0, repack_wo, tm!()); }
    let t0 = t_start!();
    matvec_batch_step(
        lw.wo_repacked.as_deref(), lw.wo_dtype, lw.wo,
        state.batch_q8_a.as_ptr(),
        state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
        state.batch_q8_bsums.as_ptr(),
        state.batch_wo_out.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, n_heads * head_dim, n, n_pad, hd,
        current_chunk, ith, nth,
    );
    barrier.wait(); // B14
    if ith == 0 { t_accum!(t0, gemm_wo, tm!()); }
    // ── 7. Post-attn residual + FFN norm (all threads, token-strided) ──
    let t0 = t_start!();
    {
        let mut t = ith;
        while t < n {
            let off = t * hd;
            if !lw.post_attn_norm.is_null() {
                ffi_inference::gemma4_rmsnorm(
                    state.batch_wo_out[off..].as_ptr(), lw.post_attn_norm,
                    state.batch_x_norm[off..].as_mut_ptr(), hd as i32, model.rms_eps,
                );
                ffi_inference::vec_add_f32(
                    state.batch_x_norm[off..].as_ptr(), state.batch_x[off..].as_ptr(),
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
            t += nth;
        }
    }
    barrier.wait(); // B15
    if ith == 0 { t_accum!(t0, post_attn_ffn_norm, tm!()); }
    // ── 7b. Parallel quant for FFN ──────────────────────────────
    let t0 = t_start!();
    parallel_batch_quant(
        &state.batch_x_norm, hd, n, n_pad,
        &mut state.batch_q8_qs, &mut state.batch_q8_d, &mut state.batch_q8_bsums,
        ith, nth,
    );
    barrier.wait(); // B16
    if ith == 0 { t_accum!(t0, quant_ffn, tm!()); }
    let ffn_dim = model.ffn_dim[il];
    let t0 = t_start!();
    if ith == 0 {
        repack_q8_for_gemm(
            &state.batch_q8_qs, &state.batch_q8_d, &state.batch_q8_bsums,
            &mut state.batch_q8_a, hd, n_pad,
        );
    }
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait(); // B17
    if ith == 0 { t_accum!(t0, repack_ffn, tm!()); }
    // ── 8. FFN gate + up ────────────────────────────────────────
    let t0 = t_start!();
    matvec_batch_step(
        lw.w_gate_repacked.as_deref(), lw.w_gate_dtype, lw.w_gate,
        state.batch_q8_a.as_ptr(),
        state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
        state.batch_q8_bsums.as_ptr(),
        state.batch_gate.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        ffn_dim, hd, n, n_pad, ffn_dim,
        current_chunk, ith, nth,
    );
    barrier.wait(); // B18
    if ith == 0 { t_accum!(t0, gemm_gate, tm!()); }
    // FFN up — reuse Q8K + repack
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait(); // B19
    let t0 = t_start!();
    matvec_batch_step(
        lw.w_up_repacked.as_deref(), lw.w_up_dtype, lw.w_up,
        state.batch_q8_a.as_ptr(),
        state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
        state.batch_q8_bsums.as_ptr(),
        state.batch_up.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        ffn_dim, hd, n, n_pad, ffn_dim,
        current_chunk, ith, nth,
    );
    barrier.wait(); // B20
    if ith == 0 { t_accum!(t0, gemm_up, tm!()); }
    // ── 9. GELU + FFN down ──────────────────────────────────────
    let t0 = t_start!();
    {
        let mut t = ith;
        while t < n {
            ffi_inference::gelu_mul(
                state.batch_gate[t * ffn_dim..].as_ptr(),
                state.batch_up[t * ffn_dim..].as_ptr(),
                state.batch_gate[t * ffn_dim..].as_mut_ptr(), ffn_dim as i32,
            );
            t += nth;
        }
    }
    barrier.wait(); // B21
    if ith == 0 { t_accum!(t0, gelu_mul, tm!()); }
    let t0 = t_start!();
    parallel_batch_quant(
        &state.batch_gate, ffn_dim, n, n_pad,
        &mut state.batch_ffn_q8_qs, &mut state.batch_ffn_q8_d,
        &mut state.batch_ffn_q8_bsums,
        ith, nth,
    );
    barrier.wait(); // B22
    if ith == 0 { t_accum!(t0, quant_down, tm!()); }
    let t0 = t_start!();
    if ith == 0 {
        repack_q8_for_gemm(
            &state.batch_ffn_q8_qs, &state.batch_ffn_q8_d, &state.batch_ffn_q8_bsums,
            &mut state.batch_ffn_q8_a, ffn_dim, n_pad,
        );
    }
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait(); // B23
    if ith == 0 { t_accum!(t0, repack_down, tm!()); }
    let t0 = t_start!();
    if lw.w_down_dtype == matmul::GGML_TYPE_Q6_K && cfg!(target_arch = "aarch64") {
        // Q6K GEMM: processes all tokens × weight rows using block_q8_Kx4 input
        current_chunk.store(nth as i32, Ordering::Relaxed);
        matmul_graph::q6k_gemm_batch_ws(
            lw.w_down, state.batch_ffn_q8_a.as_ptr(),
            state.batch_down.as_mut_ptr(),
            ffn_dim, hd, n_pad, hd,
            current_chunk, ith, nth,
        );
    } else {
        matvec_batch_step(
            lw.w_down_repacked.as_deref(), lw.w_down_dtype, lw.w_down,
            state.batch_ffn_q8_a.as_ptr(),
            state.batch_ffn_q8_qs.as_ptr(), state.batch_ffn_q8_d.as_ptr(),
            state.batch_ffn_q8_bsums.as_ptr(),
            state.batch_down.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            hd, ffn_dim, n, n_pad, hd,
            current_chunk, ith, nth,
        );
    }
    barrier.wait(); // B24
    if ith == 0 { t_accum!(t0, gemm_down, tm!()); }
    // ── 10a. Post-FFN residual (all threads, token-strided) ─────
    let t0 = t_start!();
    {
        let mut t = ith;
        while t < n {
            let off = t * hd;
            if !lw.post_ffn_norm.is_null() {
                ffi_inference::gemma4_rmsnorm(
                    state.batch_down[off..].as_ptr(), lw.post_ffn_norm,
                    state.batch_x_norm[off..].as_mut_ptr(), hd as i32, model.rms_eps,
                );
                ffi_inference::vec_add_f32(
                    state.batch_x_norm[off..].as_ptr(), state.batch_attn_res[off..].as_ptr(),
                    state.batch_x[off..].as_mut_ptr(), hd as i32,
                );
            } else {
                ffi_inference::vec_add_f32(
                    state.batch_down[off..].as_ptr(), state.batch_attn_res[off..].as_ptr(),
                    state.batch_x[off..].as_mut_ptr(), hd as i32,
                );
            }
            t += nth;
        }
    }
    barrier.wait();
    if ith == 0 { t_accum!(t0, post_ffn_residual, tm!()); }

    // ── 10b. Batched PLE (all threads) ──────────────────────────
    let t0 = t_start!();
    super::forward_batch_ple::ple_batch(
        state, model, il, n, barrier, current_chunk, ith, nth,
    );
    if ith == 0 { t_accum!(t0, ple_total, tm!()); }

    // ── 10c. Output scale (all threads, token-strided) ──────────
    let t0 = t_start!();
    {
        let out_scale = lw.layer_output_scale;
        if out_scale != 1.0 {
            let mut t = ith;
            while t < n {
                let off = t * hd;
                ffi_inference::vec_scale_f32(
                    state.batch_x[off..].as_ptr(), state.batch_x[off..].as_mut_ptr(),
                    out_scale, hd as i32,
                );
                t += nth;
            }
        }
    }
    barrier.wait(); // B25
    if ith == 0 { t_accum!(t0, output_scale, tm!()); }
}
