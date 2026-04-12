//! Batched forward pass — processes N tokens through all layers using gemm.
//!
//! Replaces both forward_one (Path A) and forward_one_graph (Path B) for
//! prompt evaluation. Uses gemm_q4k_8x8 for all Q4K weight projections and
//! the fused attention kernel for multi-token causal attention.
//!
//! Supporting ops (rmsnorm, rope, quant) loop N times over existing
//! single-token kernels.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::Gemma4State;
use crate::inference::matmul;
use crate::inference::matmul_graph;
use crate::inference::dequant;
use crate::kernels::ffi_inference;
use crate::inference::threadpool::SpinBarrier;

/// Run a batched forward pass for `tokens.len()` tokens.
/// All n_threads execute this together via SpinBarrier.
pub(crate) fn forward_batch_inner(
    state: &mut Gemma4State,
    model: &Gemma4Model,
    tokens: &[u32],
    barrier: &SpinBarrier,
    current_chunk: &AtomicI32,
    ith: usize,
    nth: usize,
) {
    let n = tokens.len();
    let hd = model.hidden_dim;

    // ── Pre-loop: embed all N tokens, prepare PLE per token (thread 0) ──
    if ith == 0 {
        let embed_scale = (hd as f32).sqrt();
        for t in 0..n {
            dequant::q6k_embed_lookup(
                model.embed_weight, tokens[t] as usize, &mut state.x, hd,
            );
            ffi_inference::vec_scale_f32(
                state.x.as_ptr(), state.x.as_mut_ptr(), embed_scale, hd as i32,
            );
            state.batch_x[t * hd..(t + 1) * hd].copy_from_slice(&state.x[..hd]);

            // PLE: prepare_ple writes into ple_signal, copy to batch_ple_signal
            state.prepare_ple(model, tokens[t]);
            let ple_total = model.ple_dim * model.n_layers;
            if ple_total > 0 {
                state.batch_ple_signal[t * ple_total..(t + 1) * ple_total]
                    .copy_from_slice(&state.ple_signal[..ple_total]);
            }
        }
    }
    barrier.wait();

    // ── Per-layer transformer blocks ─────────────────────────────
    let seq_len = state.cache.seq_len();
    for il in 0..model.n_layers {
        super::forward_batch_layer::layer_forward_batch(
            state, model, il, n, seq_len, barrier, current_chunk, ith, nth,
        );
    }

    // ── Post-loop: final norm on last token only (thread 0) ──────
    if ith == 0 {
        let last = n - 1;
        ffi_inference::gemma4_rmsnorm(
            state.batch_x[last * hd..].as_ptr(),
            model.norm_weight,
            state.x_norm.as_mut_ptr(),
            hd as i32,
            model.rms_eps,
        );
        matmul::quant_input(
            &state.x_norm[..hd],
            &mut state.q8_qs,
            &mut state.q8_d,
            &mut state.q8_bsums,
        );
    }
    barrier.wait();

    // ── Output matmul (Q6K work-stealing, last token only) ───────
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

    // ── Softcap + advance cache by N (thread 0) ─────────────────
    if ith == 0 {
        if model.logit_softcap > 0.0 {
            ffi_inference::softcap_f32(
                state.logits.as_mut_ptr(), model.vocab_size as i32, model.logit_softcap,
            );
        }
        state.cache.advance_n(n);
    }
    barrier.wait();
}
