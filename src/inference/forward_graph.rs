//! Graph-threaded forward pass — all threads execute together.
//!
//! Replaces per-op pool.run() dispatch with a single graph execution where
//! all threads loop through the forward pass with spin-barriers between ops.
//! Matches llama.cpp's ggml_graph_compute_thread pattern.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::Gemma4State;
use crate::inference::matmul;
use crate::inference::matmul_graph;
use crate::inference::dequant;
use crate::kernels::ffi_inference;
use crate::inference::threadpool::SpinBarrier;

use super::forward_graph_layer::layer_forward_graph_timed;

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
        dequant::embed_lookup(model.embed_weight, model.embed_dtype, token_id as usize, &mut state.x, hd);
        let embed_scale = (hd as f32).sqrt();
        ffi_inference::vec_scale_f32(
            state.x.as_ptr(), state.x.as_mut_ptr(), embed_scale, hd as i32,
        );
        state.prepare_ple(model, token_id);
    }
    barrier.wait();

    // ── Per-layer transformer blocks ─────────────────────────────
    let timing = crate::inference::forward::timing_enabled() && ith == 0;
    let mut t_norm_quant: u64 = 0;
    let mut t_q: u64 = 0;
    let mut t_q_norm_rope: u64 = 0;
    let mut t_kv: u64 = 0;
    let mut t_kv_norm_cache: u64 = 0;
    let mut t_attn: u64 = 0;
    let mut t_wo_quant: u64 = 0;
    let mut t_wo: u64 = 0;
    let mut t_post_attn: u64 = 0;
    let mut t_gate_up: u64 = 0;
    let mut t_gelu_quant: u64 = 0;
    let mut t_down: u64 = 0;
    let mut t_post_ffn_ple: u64 = 0;
    for il in 0..model.n_layers {
        layer_forward_graph_timed(
            state, model, il, pos, barrier, current_chunk, ith, nth,
            timing,
            &mut t_norm_quant, &mut t_q, &mut t_q_norm_rope,
            &mut t_kv, &mut t_kv_norm_cache,
            &mut t_attn, &mut t_wo_quant, &mut t_wo,
            &mut t_post_attn, &mut t_gate_up, &mut t_gelu_quant,
            &mut t_down, &mut t_post_ffn_ple,
        );
    }
    if timing {
        let ms = |us: u64| us as f64 / 1000.0;
        let total = t_norm_quant + t_q + t_q_norm_rope + t_kv + t_kv_norm_cache
            + t_attn + t_wo_quant + t_wo + t_post_attn + t_gate_up + t_gelu_quant
            + t_down + t_post_ffn_ple;
        let pct = |us: u64| if total > 0 { us as f64 / total as f64 * 100.0 } else { 0.0 };
        eprintln!("[decode-timing] {n} layers, total {:.1}ms", ms(total), n = model.n_layers);
        eprintln!("  norm+quant      {:7.1}ms  ({:4.1}%)", ms(t_norm_quant), pct(t_norm_quant));
        eprintln!("  gemv_q          {:7.1}ms  ({:4.1}%)", ms(t_q), pct(t_q));
        eprintln!("  q_norm+rope     {:7.1}ms  ({:4.1}%)", ms(t_q_norm_rope), pct(t_q_norm_rope));
        eprintln!("  gemv_kv         {:7.1}ms  ({:4.1}%)", ms(t_kv), pct(t_kv));
        eprintln!("  kv_norm+cache   {:7.1}ms  ({:4.1}%)", ms(t_kv_norm_cache), pct(t_kv_norm_cache));
        eprintln!("  attention       {:7.1}ms  ({:4.1}%)", ms(t_attn), pct(t_attn));
        eprintln!("  wo_quant        {:7.1}ms  ({:4.1}%)", ms(t_wo_quant), pct(t_wo_quant));
        eprintln!("  gemv_wo         {:7.1}ms  ({:4.1}%)", ms(t_wo), pct(t_wo));
        eprintln!("  post_attn       {:7.1}ms  ({:4.1}%)", ms(t_post_attn), pct(t_post_attn));
        eprintln!("  gemv_gate+up    {:7.1}ms  ({:4.1}%)", ms(t_gate_up), pct(t_gate_up));
        eprintln!("  gelu+quant      {:7.1}ms  ({:4.1}%)", ms(t_gelu_quant), pct(t_gelu_quant));
        eprintln!("  gemv_down       {:7.1}ms  ({:4.1}%)", ms(t_down), pct(t_down));
        eprintln!("  post_ffn+ple    {:7.1}ms  ({:4.1}%)", ms(t_post_ffn_ple), pct(t_post_ffn_ple));
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
    // Full 262K vocab by default. Gemma 4's vocab is NOT low-ID frequency-
    // ordered the way the old comment assumed — common punctuation pieces
    // like "?" (236881) and many word-piece suffixes live above 32K, and
    // so do roughly half the tokens llama.cpp argmaxes on typical prompts.
    // A 32K cutoff drops them silently, producing the classic "coherent
    // first sentence, then repetition loop" failure on structured prompts
    // (e.g. narrating a rune/tool JSON result).
    // Opt-in to the experiment with OLORIN_HOT_VOCAB=1 for perf benchmarking.
    let logit_rows = if std::env::var("OLORIN_HOT_VOCAB").is_ok() {
        model.vocab_size.min(32768)
    } else {
        model.vocab_size
    };
    let t_out_start = if timing { Some(std::time::Instant::now()) } else { None };
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    if let (Some(ref q6k_buf), Some(ref d_arr)) =
        (&model.embed_q6k_repacked, &model.embed_q6k_d_arr)
    {
        matmul_graph::q6k_repacked_batch_ws_pre_d(
            q6k_buf.as_ptr(), d_arr.as_ptr(),
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
    if let Some(s) = t_out_start {
        eprintln!("  output_logits   {:7.1}ms", s.elapsed().as_micros() as f64 / 1000.0);
    }

    // ── Softcap (thread 0) ───────────────────────────────────────
    if ith == 0 {
        if model.logit_softcap > 0.0 {
            ffi_inference::softcap_f32(
                state.logits.as_mut_ptr(), logit_rows as i32, model.logit_softcap,
            );
        }
        state.logit_rows = logit_rows;
        state.cache.advance();
    }
    barrier.wait();
}

