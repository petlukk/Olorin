//! Batched PLE (Per-Layer Embedding) — parallel quant + GEMM.
//!
//! Replaces the per-token scalar matvec loop in forward_batch_layer section 10b.
//! inp_gate GEMM: [hd=1536] → [ple_dim=256], proj GEMM: [ple_dim=256] → [hd=1536].

use std::sync::atomic::{AtomicI32, Ordering};
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::Gemma4State;
use crate::inference::threadpool::SpinBarrier;
use crate::kernels::ffi_inference;

use super::forward_batch_layer::{parallel_batch_quant, repack_q8_for_gemm, matvec_batch_step};

#[allow(clippy::too_many_arguments)]
pub(crate) fn ple_batch(
    state: &mut Gemma4State, model: &Gemma4Model,
    il: usize, n: usize,
    barrier: &SpinBarrier, current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    let ple_dim = model.ple_dim;
    let lw = &model.layers[il];
    if ple_dim == 0 || lw.inp_gate.is_null() || lw.proj.is_null() {
        return;
    }

    let hd = model.hidden_dim;
    let n_pad = (n + 3) & !3;
    let ple_total = ple_dim * model.n_layers;
    let ple_off = il * ple_dim;

    // ── Step 1: Parallel quant batch_x (hd-dim) into batch_q8 buffers ──
    // Reuse the main Q8K buffers — same hd dimension, safe after FFN barrier.
    parallel_batch_quant(
        &state.batch_x, hd, n, n_pad,
        &mut state.batch_q8_qs, &mut state.batch_q8_d, &mut state.batch_q8_bsums,
        ith, nth,
    );
    barrier.wait();

    // ── Step 2: All threads repack into batch_q8_a ──
    repack_q8_for_gemm(
        &state.batch_q8_qs, &state.batch_q8_d, &state.batch_q8_bsums,
        &mut state.batch_q8_a, hd, n_pad,
        ith, nth,
    );
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();

    // ── Step 3: GEMM inp_gate: [ple_dim, hd] × Q8K → batch_ple_gate_out ──
    matvec_batch_step(
        lw.inp_gate_repacked.as_deref(), lw.inp_gate_dtype, lw.inp_gate,
        state.batch_q8_a.as_ptr(), state.batch_q8_qs.as_ptr(),
        state.batch_q8_d.as_ptr(), state.batch_q8_bsums.as_ptr(),
        state.batch_ple_gate_out.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        ple_dim, hd, n, n_pad, ple_dim,
        current_chunk, ith, nth,
    );
    barrier.wait();

    // ── Step 4: Parallel gelu_mul with PLE signal (token-strided) ──
    {
        let mut t = ith;
        while t < n {
            let gate_off = t * ple_dim;
            let sig_off = t * ple_total + ple_off;
            ffi_inference::gelu_mul(
                state.batch_ple_gate_out[gate_off..].as_ptr(),
                state.batch_ple_signal[sig_off..].as_ptr(),
                state.batch_ple_gate_out[gate_off..].as_mut_ptr(),
                ple_dim as i32,
            );
            t += nth;
        }
    }
    barrier.wait();

    // ── Step 5: Parallel quant ple_dim-dim gate output into PLE Q8K buffers ──
    parallel_batch_quant(
        &state.batch_ple_gate_out, ple_dim, n, n_pad,
        &mut state.batch_ple_q8_qs, &mut state.batch_ple_q8_d,
        &mut state.batch_ple_q8_bsums,
        ith, nth,
    );
    barrier.wait();

    // ── Step 6: All threads repack PLE Q8K into batch_ple_q8_a ──
    repack_q8_for_gemm(
        &state.batch_ple_q8_qs, &state.batch_ple_q8_d, &state.batch_ple_q8_bsums,
        &mut state.batch_ple_q8_a, ple_dim, n_pad,
        ith, nth,
    );
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();

    // ── Step 7: GEMM proj: [hd, ple_dim] × Q8K → batch_ple_proj_out ──
    matvec_batch_step(
        lw.proj_repacked.as_deref(), lw.proj_dtype, lw.proj,
        state.batch_ple_q8_a.as_ptr(), state.batch_ple_q8_qs.as_ptr(),
        state.batch_ple_q8_d.as_ptr(), state.batch_ple_q8_bsums.as_ptr(),
        state.batch_ple_proj_out.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, ple_dim, n, n_pad, hd,
        current_chunk, ith, nth,
    );
    barrier.wait();

    // ── Step 8: Parallel post_norm + vec_add back into batch_x ──
    {
        let mut t = ith;
        while t < n {
            let x_off = t * hd;
            let proj_off = t * hd;
            if !lw.post_norm.is_null() {
                ffi_inference::gemma4_rmsnorm(
                    state.batch_ple_proj_out[proj_off..].as_ptr(),
                    lw.post_norm,
                    state.batch_ple_proj_out[proj_off..].as_mut_ptr(),
                    hd as i32,
                    model.rms_eps,
                );
            }
            ffi_inference::vec_add_f32(
                state.batch_x[x_off..].as_ptr(),
                state.batch_ple_proj_out[proj_off..].as_ptr(),
                state.batch_x[x_off..].as_mut_ptr(),
                hd as i32,
            );
            t += nth;
        }
    }
    barrier.wait();
}
