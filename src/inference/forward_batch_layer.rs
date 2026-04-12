//! Per-layer batched forward — called from forward_batch.rs.
//!
//! Replicates every operation in forward_graph::layer_forward_graph,
//! but uses gemm for Q4K matmuls and the fused attention kernel for
//! multi-token causal attention.

use std::sync::atomic::AtomicI32;
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::{compute_rope_tables, Gemma4State};
use crate::inference::matmul;
use crate::inference::matmul_batch;
use crate::kernels::ffi_inference;
use crate::inference::threadpool::SpinBarrier;

/// Zero-pad Q8K buffers for columns n..n_pad (make gemm's N a multiple of 4).
fn zero_pad_q8(qs: &mut [i8], d: &mut [f32], bsums: &mut [i16],
               n: usize, n_pad: usize, dim: usize) {
    let nb = dim / 256;
    let qs_stride = dim + 12;
    for t in n..n_pad {
        qs[t * qs_stride..(t + 1) * qs_stride].fill(0);
        d[t * nb..(t + 1) * nb].fill(0.0);
        bsums[t * nb * 16..(t + 1) * nb * 16].fill(0);
    }
}

/// Per-layer batched forward. Mirrors layer_forward_graph exactly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward_batch(
    state: &mut Gemma4State,
    model: &Gemma4Model,
    il: usize,
    n: usize,
    seq_len: usize,
    barrier: &SpinBarrier,
    _current_chunk: &AtomicI32,
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
    let n_pad = (n + 3) & !3;

    // ── 1. Attn norm + Q8K quant per token (thread 0) ────────────
    if ith == 0 {
        let nb = hd / 256;
        let qs_stride = hd + 12;
        for t in 0..n {
            ffi_inference::gemma4_rmsnorm(
                state.batch_x[t * hd..].as_ptr(), lw.attn_norm,
                state.batch_x_norm[t * hd..].as_mut_ptr(), hd as i32, model.rms_eps,
            );
            matmul::quant_input(
                &state.batch_x_norm[t * hd..(t + 1) * hd],
                &mut state.batch_q8_qs[t * qs_stride..(t + 1) * qs_stride],
                &mut state.batch_q8_d[t * nb..(t + 1) * nb],
                &mut state.batch_q8_bsums[t * nb * 16..(t + 1) * nb * 16],
            );
        }
        zero_pad_q8(&mut state.batch_q8_qs, &mut state.batch_q8_d,
                     &mut state.batch_q8_bsums, n, n_pad, hd);
    }
    barrier.wait();

    // ── 2. Q gemm (thread 0) ────────────────────────────────────
    if ith == 0 {
        unsafe {
            matmul_batch::gemm_q4k_8x8(
                lw.wq_repacked.as_deref().unwrap().as_ptr(),
                state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
                state.batch_q8_bsums.as_ptr(), state.batch_q8_a.as_mut_ptr(),
                state.gemm_scratch.as_mut_ptr(), state.batch_q.as_mut_ptr(),
                hd, qkv_dim, n_pad,
            );
        }
    }
    barrier.wait();

    // ── 3-4. K/V gemm + per-token norms + RoPE + cache store ────
    let n_rot = if model.is_swa[il] { model.rope_dim_swa } else { model.rope_dim_global };
    let rope_theta = if model.is_swa[il] { model.rope_theta_swa } else { model.rope_theta_global };
    let freq_factors = if !model.is_swa[il] { model.rope_freqs.as_deref() } else { None };

    if has_kv {
        let kv_dim = n_kv_heads * head_dim;
        let kv_dim_v = n_kv_heads * head_dim_v;

        if ith == 0 {
            unsafe {
                matmul_batch::gemm_q4k_8x8(
                    lw.wk_repacked.as_deref().unwrap().as_ptr(),
                    state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
                    state.batch_q8_bsums.as_ptr(), state.batch_q8_a.as_mut_ptr(),
                    state.gemm_scratch.as_mut_ptr(), state.batch_k.as_mut_ptr(),
                    hd, kv_dim, n_pad,
                );
                matmul_batch::gemm_q4k_8x8(
                    lw.wv_repacked.as_deref().unwrap().as_ptr(),
                    state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
                    state.batch_q8_bsums.as_ptr(), state.batch_q8_a.as_mut_ptr(),
                    state.gemm_scratch.as_mut_ptr(), state.batch_v.as_mut_ptr(),
                    hd, kv_dim_v, n_pad,
                );
            }
        }
        barrier.wait();

        if ith == 0 {
            for t in 0..n {
                compute_rope_tables(&mut state.cos_table, &mut state.sin_table,
                    seq_len + t, n_rot, rope_theta, freq_factors);

                // Q norm per-head
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

                // K norm per-head
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
                // V bare norm per-head
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
        // Shared KV layer — Q norm + RoPE only
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
                    head_dim as i32, qkv_dim as i32, qkv_dim as i32,
                    stride_kv as i32, (kv_h * head_dim) as i32,
                    n_kv as i32, n as i32, seq_len as i32, 1.0f32,
                );
            }
        }
    }
    barrier.wait();

    // ── 6. Quant attn_out + Wo gemm (thread 0) ─────────────────
    if ith == 0 {
        let ao_dim = n_heads * head_dim;
        let nb_ao = ao_dim / 256;
        let qs_ao = ao_dim + 12;
        for t in 0..n {
            matmul::quant_input(
                &state.batch_attn_out[t * ao_dim..(t + 1) * ao_dim],
                &mut state.batch_q8_qs[t * qs_ao..(t + 1) * qs_ao],
                &mut state.batch_q8_d[t * nb_ao..(t + 1) * nb_ao],
                &mut state.batch_q8_bsums[t * nb_ao * 16..(t + 1) * nb_ao * 16],
            );
        }
        zero_pad_q8(&mut state.batch_q8_qs, &mut state.batch_q8_d,
                     &mut state.batch_q8_bsums, n, n_pad, ao_dim);
        unsafe {
            matmul_batch::gemm_q4k_8x8(
                lw.wo_repacked.as_deref().unwrap().as_ptr(),
                state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
                state.batch_q8_bsums.as_ptr(), state.batch_q8_a.as_mut_ptr(),
                state.gemm_scratch.as_mut_ptr(), state.batch_wo_out.as_mut_ptr(),
                ao_dim, hd, n_pad,
            );
        }
    }
    barrier.wait();

    // ── 7. Post-attn norm + residual + FFN norm + quant (thread 0) ──
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
        let nb = hd / 256;
        let qs_stride = hd + 12;
        for t in 0..n {
            matmul::quant_input(
                &state.batch_x_norm[t * hd..(t + 1) * hd],
                &mut state.batch_q8_qs[t * qs_stride..(t + 1) * qs_stride],
                &mut state.batch_q8_d[t * nb..(t + 1) * nb],
                &mut state.batch_q8_bsums[t * nb * 16..(t + 1) * nb * 16],
            );
        }
        zero_pad_q8(&mut state.batch_q8_qs, &mut state.batch_q8_d,
                     &mut state.batch_q8_bsums, n, n_pad, hd);
    }
    barrier.wait();

    // ── 8. FFN: gate gemm + up gemm (thread 0) ─────────────────
    let ffn_dim = model.ffn_dim[il];
    if ith == 0 {
        unsafe {
            matmul_batch::gemm_q4k_8x8(
                lw.w_gate_repacked.as_deref().unwrap().as_ptr(),
                state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
                state.batch_q8_bsums.as_ptr(), state.batch_q8_a.as_mut_ptr(),
                state.gemm_scratch.as_mut_ptr(), state.batch_gate.as_mut_ptr(),
                hd, ffn_dim, n_pad,
            );
            matmul_batch::gemm_q4k_8x8(
                lw.w_up_repacked.as_deref().unwrap().as_ptr(),
                state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
                state.batch_q8_bsums.as_ptr(), state.batch_q8_a.as_mut_ptr(),
                state.gemm_scratch.as_mut_ptr(), state.batch_up.as_mut_ptr(),
                hd, ffn_dim, n_pad,
            );
        }
    }
    barrier.wait();

    // ── 9. GELU_mul + quant + down gemm (thread 0) ─────────────
    if ith == 0 {
        for t in 0..n {
            ffi_inference::gelu_mul(
                state.batch_gate[t * ffn_dim..].as_ptr(),
                state.batch_up[t * ffn_dim..].as_ptr(),
                state.batch_gate[t * ffn_dim..].as_mut_ptr(), ffn_dim as i32,
            );
        }
        let nb_ffn = ffn_dim / 256;
        let qs_ffn = ffn_dim + 12;
        for t in 0..n {
            matmul::quant_input(
                &state.batch_gate[t * ffn_dim..(t + 1) * ffn_dim],
                &mut state.batch_ffn_q8_qs[t * qs_ffn..(t + 1) * qs_ffn],
                &mut state.batch_ffn_q8_d[t * nb_ffn..(t + 1) * nb_ffn],
                &mut state.batch_ffn_q8_bsums[t * nb_ffn * 16..(t + 1) * nb_ffn * 16],
            );
        }
        zero_pad_q8(&mut state.batch_ffn_q8_qs, &mut state.batch_ffn_q8_d,
                     &mut state.batch_ffn_q8_bsums, n, n_pad, ffn_dim);
        unsafe {
            matmul_batch::gemm_q4k_8x8(
                lw.w_down_repacked.as_deref().unwrap().as_ptr(),
                state.batch_ffn_q8_qs.as_ptr(), state.batch_ffn_q8_d.as_ptr(),
                state.batch_ffn_q8_bsums.as_ptr(), state.batch_ffn_q8_a.as_mut_ptr(),
                state.gemm_scratch.as_mut_ptr(), state.batch_down.as_mut_ptr(),
                ffn_dim, hd, n_pad,
            );
        }
    }
    barrier.wait();

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
