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
use crate::inference::forward::{timing_enabled, Gemma4State};
use crate::inference::forward_batch_layer::{BatchLayerTiming, parallel_batch_quant};
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
    let timing = timing_enabled();
    let mut layer_timing = if timing && ith == 0 {
        Some(BatchLayerTiming::new())
    } else {
        None
    };
    for il in 0..model.n_layers {
        super::forward_batch_layer::layer_forward_batch(
            state, model, il, n, seq_len, barrier, current_chunk, ith, nth,
            layer_timing.as_mut(),
        );
    }
    if let Some(ref t) = layer_timing {
        t.print_summary(model.n_layers, n);
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

    // ── Output matmul (work-stealing, last token only) ────────────
    // Prefill always uses full vocab (no hot-vocab optimization)
    let logit_rows = model.vocab_size;
    if ith == 0 { state.logit_rows = logit_rows; }
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    if let Some(ref q6k_buf) = model.embed_q6k_repacked {
        matmul_graph::q6k_repacked_batch_ws(
            q6k_buf.as_ptr(), model.embed_weight,
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.logits.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            model.vocab_size, hd, 1, model.vocab_size,
            current_chunk, ith, nth,
        );
    } else {
        matmul_graph::matvec_ws(
            model.embed_dtype, model.embed_weight,
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.logits.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            model.vocab_size, hd,
            current_chunk, ith, nth,
        );
    }
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

/// Like `forward_batch_inner`, but computes final RMSNorm + output projection
/// for EVERY row, filling `state.batch_logits[0..n*vocab_size]`.
///
/// Body is a near-duplicate of `forward_batch_inner` through the layer loop,
/// then diverges in the post-loop to per-row-normalize, per-row-quantize, and
/// run the output matmul with `n_tokens = n` over all rows.
///
/// Used exclusively by speculative verify — `forward_batch` remains the hot
/// prefill path and is unchanged.
pub(crate) fn forward_batch_all_logits_inner(
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

            state.prepare_ple(model, tokens[t]);
            let ple_total = model.ple_dim * model.n_layers;
            if ple_total > 0 {
                state.batch_ple_signal[t * ple_total..(t + 1) * ple_total]
                    .copy_from_slice(&state.ple_signal[..ple_total]);
            }
        }
    }
    barrier.wait();

    // ── Per-layer transformer blocks (identical to prefill) ─────
    let seq_len = state.cache.seq_len();
    let timing = timing_enabled();
    let mut layer_timing = if timing && ith == 0 {
        Some(BatchLayerTiming::new())
    } else {
        None
    };
    for il in 0..model.n_layers {
        super::forward_batch_layer::layer_forward_batch(
            state, model, il, n, seq_len, barrier, current_chunk, ith, nth,
            layer_timing.as_mut(),
        );
    }
    if let Some(ref t) = layer_timing {
        t.print_summary(model.n_layers, n);
    }

    // ── Post-loop: final RMSNorm for EVERY row (token-strided) ───
    {
        let mut t = ith;
        while t < n {
            ffi_inference::gemma4_rmsnorm(
                state.batch_x[t * hd..].as_ptr(),
                model.norm_weight,
                state.batch_x_norm[t * hd..].as_mut_ptr(),
                hd as i32,
                model.rms_eps,
            );
            t += nth;
        }
    }
    barrier.wait();

    // ── Per-row Q8K quantize (all rows) ──────────────────────────
    // n_pad rounds up to 4 for GEMM; output matmul uses q6k_repacked_batch_ws
    // which does NOT require the x4 repack (it consumes raw Q8K), but the
    // repack path does. We feed q6k_repacked_batch_ws directly from batch_q8
    // which matches forward_batch's single-row call convention extended to N.
    let n_pad = (n + 3) & !3;
    parallel_batch_quant(
        &state.batch_x_norm, hd, n, n_pad,
        &mut state.batch_q8_qs, &mut state.batch_q8_d, &mut state.batch_q8_bsums,
        ith, nth,
    );
    barrier.wait();

    // ── Output matmul: per-row over all N tokens ─────────────────
    let logit_rows = model.vocab_size;
    if ith == 0 { state.logit_rows = logit_rows; }
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();

    // Output matmul kernels consume Q8K with stride = hd + 12 (matches
    // parallel_batch_quant layout). q6k_repacked_batch_ws accepts n_tokens
    // directly and writes row-major into `output` with `output_stride`.
    if let Some(ref q6k_buf) = model.embed_q6k_repacked {
        matmul_graph::q6k_repacked_batch_ws(
            q6k_buf.as_ptr(), model.embed_weight,
            state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(), state.batch_q8_bsums.as_ptr(),
            state.batch_logits.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            model.vocab_size, hd, n, model.vocab_size,
            current_chunk, ith, nth,
        );
    } else {
        // matvec_ws is single-row: loop over rows, resetting chunk counter
        // between rows. Thread 0 does the reset; all threads cooperate per row.
        let nb = hd / 256;
        let qs_stride = hd + 12;
        for t in 0..n {
            barrier.wait();
            if ith == 0 { current_chunk.store(nth as i32, Ordering::Relaxed); }
            barrier.wait();
            unsafe {
                matmul_graph::matvec_ws(
                    model.embed_dtype, model.embed_weight,
                    state.batch_q8_qs.as_ptr().add(t * qs_stride),
                    state.batch_q8_d.as_ptr().add(t * nb),
                    state.batch_q8_bsums.as_ptr().add(t * nb * 16),
                    state.batch_logits.as_mut_ptr().add(t * model.vocab_size),
                    state.q6k_d_scratch.as_mut_ptr(),
                    model.vocab_size, hd,
                    current_chunk, ith, nth,
                );
            }
        }
    }
    barrier.wait();

    // ── Softcap over all N rows + advance cache (thread 0) ───────
    if ith == 0 {
        if model.logit_softcap > 0.0 {
            ffi_inference::softcap_f32(
                state.batch_logits.as_mut_ptr(),
                (n * model.vocab_size) as i32,
                model.logit_softcap,
            );
        }
        state.cache.advance_n(n);
    }
    barrier.wait();
}
