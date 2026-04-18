//! Batched forward pass — processes N tokens through all layers using gemm.
//!
//! Replaces both forward_one (Path A) and forward_one_graph (Path B) for
//! prompt evaluation. Uses work-stealing Q4K 8x8 GEMM for all Q4K weight
//! projections and the fused attention kernel for multi-token causal attention.
//!
//! Supporting ops (rmsnorm, rope, quant) loop N times over existing
//! single-token kernels.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::{timing_enabled, Gemma4State};
use crate::inference::forward_batch_layer::BatchLayerTiming;
use crate::inference::matmul;
use crate::inference::matmul_graph;
use crate::inference::dequant;
use crate::kernels::ffi_inference;
use crate::inference::threadpool::SpinBarrier;

/// Run a batched forward pass for `tokens.len()` tokens.
/// All n_threads execute this together via SpinBarrier.
///
/// When `compute_logits` is false, the final_norm + output_matmul + softcap
/// stage is skipped — used for intermediate ubatch chunks whose logits will
/// be overwritten by a later chunk.
pub(crate) fn forward_batch_inner(
    state: &mut Gemma4State,
    model: &Gemma4Model,
    tokens: &[u32],
    compute_logits: bool,
    barrier: &SpinBarrier,
    current_chunk: &AtomicI32,
    ith: usize,
    nth: usize,
) {
    let n = tokens.len();
    let hd = model.hidden_dim;
    let timing = timing_enabled();
    let print = timing && ith == 0;

    // ── Pre-loop: embed + PLE, fully parallel ──────────────────────
    // Phase 1 (token-parallel): embed + scale each token's x vector.
    // Phase 2 (row-parallel, inside prepare_ple_batch): BF16 matvec
    //   against ple_model_proj — each weight row reused across all
    //   n_tokens inputs in one kernel call, so the 27.5 MB weight
    //   matrix streams through DRAM once per prefill instead of
    //   n_tokens times.
    // Phase 3 (token-parallel, inside prepare_ple_batch): scale +
    //   RMSNorm + FMA combine.
    let t_pre = if print { Some(std::time::Instant::now()) } else { None };
    let embed_scale = (hd as f32).sqrt();
    let ple_total = model.ple_dim * model.n_layers;
    let per = (n + nth - 1) / nth;
    let t_start = ith * per;
    let t_end = (t_start + per).min(n);

    for t in t_start..t_end {
        let slot = &mut state.batch_x[t * hd..(t + 1) * hd];
        dequant::q6k_embed_lookup(model.embed_weight, tokens[t] as usize, slot, hd);
        ffi_inference::vec_scale_f32(
            slot.as_ptr(), slot.as_mut_ptr(), embed_scale, hd as i32,
        );
    }
    barrier.wait();

    if ple_total > 0 {
        // Borrow state's buffers via disjoint split_at so Rust is satisfied
        // with the multiple &mut required by prepare_ple_batch.
        let batch_x_ptr = state.batch_x.as_ptr();
        let x_view = unsafe { std::slice::from_raw_parts(batch_x_ptr, n * hd) };
        super::forward::prepare_ple_batch(
            model, tokens,
            x_view,
            &mut state.batch_ple_signal[..n * ple_total],
            &mut state.batch_ple_proj_scratch[..n * ple_total],
            barrier, ith, nth,
        );
    }
    let t_pre_elapsed = t_pre.map(|s| s.elapsed().as_micros() as u64).unwrap_or(0);
    barrier.wait();

    // ── Per-layer transformer blocks ─────────────────────────────
    let seq_len = state.cache.seq_len();
    let t_layers_start = if print { Some(std::time::Instant::now()) } else { None };
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
    let t_layers_us = t_layers_start.map(|s| s.elapsed().as_micros() as u64).unwrap_or(0);

    // ── Post-loop: final norm + output logits + softcap ──
    // Intermediate ubatch chunks skip this entire block; only the final
    // chunk needs logits for the last token. Cache advance is handled
    // separately below so KV positions stay correct in either case.
    let t_final_norm_us;
    let t_logits_us;
    let t_softcap_us;
    if compute_logits {
        let t_final_norm_start = if print { Some(std::time::Instant::now()) } else { None };
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
        t_final_norm_us = t_final_norm_start.map(|s| s.elapsed().as_micros() as u64).unwrap_or(0);
        barrier.wait();

        // Output matmul (work-stealing, last token only).
        // Full 262K vocab by default. See forward_graph.rs for the rationale
        // — Gemma 4's vocab isn't low-ID frequency-ordered and the 32K cutoff
        // drops roughly half of the tokens llama.cpp argmaxes.
        let logit_rows = if std::env::var("OLORIN_HOT_VOCAB").is_ok() {
            model.vocab_size.min(32768)
        } else {
            model.vocab_size
        };
        if ith == 0 { state.logit_rows = logit_rows; }
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        let t_logits_start = if print { Some(std::time::Instant::now()) } else { None };
        if let Some(ref q6k_buf) = model.embed_q6k_repacked {
            matmul_graph::q6k_repacked_batch_ws(
                q6k_buf.as_ptr(), model.embed_weight,
                state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
                state.logits.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
                logit_rows, hd, 1, logit_rows,
                current_chunk, ith, nth,
            );
        } else {
            matmul_graph::matvec_ws(
                model.embed_dtype, model.embed_weight,
                state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
                state.logits.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
                logit_rows, hd,
                current_chunk, ith, nth,
            );
        }
        barrier.wait();
        t_logits_us = t_logits_start.map(|s| s.elapsed().as_micros() as u64).unwrap_or(0);

        // Softcap (thread 0).
        let t_softcap_start = if print { Some(std::time::Instant::now()) } else { None };
        if ith == 0 && model.logit_softcap > 0.0 {
            ffi_inference::softcap_f32(
                state.logits.as_mut_ptr(), logit_rows as i32, model.logit_softcap,
            );
        }
        t_softcap_us = t_softcap_start.map(|s| s.elapsed().as_micros() as u64).unwrap_or(0);
    } else {
        t_final_norm_us = 0;
        t_logits_us = 0;
        t_softcap_us = 0;
    }

    // Advance cache — always, so the next chunk picks up at the right position.
    if ith == 0 { state.cache.advance_n(n); }
    barrier.wait();

    if print {
        let ms = |us: u64| us as f64 / 1000.0;
        eprintln!("[prefill-stages] n_tokens={n} (pre_total is thread-0 slice time, compute_logits={compute_logits})");
        eprintln!("  pre_total       {:7.1}ms  (embed + ple, per-thread {} tokens)", ms(t_pre_elapsed), per);
        eprintln!("  layer_loop      {:7.1}ms", ms(t_layers_us));
        eprintln!("  final_norm      {:7.1}ms", ms(t_final_norm_us));
        eprintln!("  output_logits   {:7.1}ms", ms(t_logits_us));
        eprintln!("  softcap+advance {:7.1}ms", ms(t_softcap_us));
    }
}
