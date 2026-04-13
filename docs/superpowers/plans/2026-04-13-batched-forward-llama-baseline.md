# Batched Forward — Match llama.cpp Threading Baseline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **HARD RULES (apply to ALL agents):**
> - No file exceeds 500 lines. Split before you hit the limit.
> - Every feature proven by end-to-end test. If it's not tested, it doesn't exist.
> - No fake functions. No silent fallbacks.
> - Olorin is Ea's showcase — every SIMD op must be an Ea kernel. **Do NOT simplify kernel code to scalar Rust.**
> - Match llama.cpp **exactly first.** Same code, same order, same math. Then measure. Then optimize.
> - **x86 kernels target AVX2 ONLY.** No AVX-512.
> - **ARM kernels target Cortex-A76 (NEON + dotprod, NO i8mm).**
> - eacompute compiler: `/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release/ea`
> - Build: `PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release`
> - Branch: `gemma4-batched-prompt-eval`

**Goal:** Rewrite `forward_batch_layer.rs` so all threads participate in every matmul via work-stealing — matching `forward_graph.rs` (the proven llama.cpp-matching decode path) exactly, but looping N tokens per op. Kill the `if ith == 0` single-thread bottleneck. Then wire `generate.rs` to use the batched path for prefill. Measure baseline.

**Architecture:** The batched forward processes each op for ALL N tokens before the next barrier, with all threads work-stealing each per-token matmul. Small ops (norm, quant, rope, residual, PLE) remain thread-0 because they're fast scalar work on one vector. Matmuls (Q, K, V, Wo, gate, up, down) are the bottleneck and use `matvec_step` (same work-stealing dispatch as `forward_graph.rs`). The gemm kernel and its `matmul_batch.rs` wrapper are NOT used — we revert to per-token matvec work-stealing to match llama.cpp first.

**Tech Stack:** Rust, existing `matmul_graph.rs` work-stealing functions, existing `SpinBarrier` + `GraphPool`, Ea SIMD kernels.

**Baseline numbers to record:** prefill tok/s and decode tok/s on WSL x86 and Pi 5.

---

## Background: Why This Rewrite

The current `forward_batch_layer.rs` gates all matmuls behind `if ith == 0` — only thread 0 does work, while N-1 threads spin at barriers burning 100% CPU doing nothing. This causes:
- 100% CPU on all cores (spin-wait) but only 1 core doing useful work
- Slower than single-token `forward_graph.rs` which work-steals every matmul
- Test hangs / timeouts on memory-constrained systems due to ~120 MB batch buffer allocation

The proven `forward_graph.rs` path already matches llama.cpp's threading model: work-stealing matvecs for every projection, thread-0-only for small ops, barriers between ops. We replicate that pattern for N tokens.

---

## Per-Task Verification Gates

**Gate 1 — Build clean.**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
grep -c "^warning" /tmp/olorin-build.log
```

**Gate 2 — Line limit.**

```bash
find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'
```

Expected: only chacha20_search_v2.ea / chacha20_search_v2_arm.ea.

**Gate 3 — Existing tests.**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression 2>&1 | tail -6
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -15
```

---

## File Map

**Rewritten files:**
- `src/inference/forward_batch.rs` — outer loop (embed, layer loop, output matmul)
- `src/inference/forward_batch_layer.rs` — per-layer: all threads work-steal matmuls, thread-0 does small ops

**Modified files:**
- `src/inference/forward.rs` — shrink batch buffer allocations (from 512 to actual batch size, or make lazy)
- `src/inference/generate.rs` — wire prefill to `forward_batch`, decode stays `forward_one_graph`
- `tests/forward_batch_verify.rs` — update to use actual model + verify

**Not touched:**
- `src/inference/forward_graph.rs` — decode path stays as-is (known working)
- `src/inference/matmul_graph.rs` — work-stealing functions reused as-is
- `src/inference/matmul_batch.rs` — gemm helper kept but unused until optimization phase
- Ea kernels — no kernel changes

---

## Task 1: Rewrite `forward_batch_layer.rs` — work-stealing matmuls

**Goal:** Replace the `if ith == 0` gated gemm calls with per-token work-stealing matvecs that ALL threads participate in. Mirror `forward_graph.rs:layer_forward_graph` line-for-line, but with an outer loop over N tokens for small ops and per-token barrier+WS for matmuls.

**Files:**
- Rewrite: `src/inference/forward_batch_layer.rs`
- Read (reference, no edit): `src/inference/forward_graph.rs` — the template to replicate

The rewritten function has the SAME barrier pattern as `forward_graph.rs` for each matmul: `current_chunk.store(nth) → barrier → matvec_step (all threads) → barrier`. Small ops (norm, quant, rope, PLE) stay `if ith == 0` — same as `forward_graph.rs`.

- [ ] **Step 1: Rewrite `forward_batch_layer.rs`**

Replace the entire file content with:

```rust
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
```

- [ ] **Step 2: Verify line count**

```bash
wc -l src/inference/forward_batch_layer.rs
```

Expected: ~320 lines (under 500 limit).

- [ ] **Step 3: Build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -c "^error"
```

Expected: 0 errors (may have warnings — the old batch Q8K buffers `batch_q8_qs` etc are now unused in this file, which is fine).

- [ ] **Step 4: Commit**

```bash
git add src/inference/forward_batch_layer.rs
git commit -m "feat: forward_batch_layer uses work-stealing matvecs — all threads participate

Match forward_graph.rs threading exactly: per-token quant(thread-0) -> barrier ->
WS matvec(all threads) -> barrier. Kill single-thread gemm bottleneck."
```

---

## Task 2: Rewrite `forward_batch.rs` — match `forward_one_inner` structure

**Goal:** Simplify the outer loop to match `forward_graph.rs:forward_one_inner` pattern. Uses single-token Q8K buffers (`state.q8_qs` etc) for per-token quant — same as decode path. No batch Q8K buffers needed for matmuls (they're reused per token).

**Files:**
- Rewrite: `src/inference/forward_batch.rs`
- Read (reference): `src/inference/forward_graph.rs:72-140`

- [ ] **Step 1: Rewrite `forward_batch.rs`**

Replace the entire file content with:

```rust
//! Batched forward pass — processes N tokens through all layers.
//!
//! Matches forward_graph.rs threading: work-stealing matmuls, thread-0 small ops.
//! Batch buffers hold per-token activations [dim, N] column-major.
//! Per-token quant uses single-token Q8K buffers (reused per token).

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
```

- [ ] **Step 2: Build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -c "^error"
```

Expected: 0.

- [ ] **Step 3: Commit**

```bash
git add src/inference/forward_batch.rs
git commit -m "feat: forward_batch outer loop — match forward_one_inner structure"
```

---

## Task 3: Shrink batch buffer allocations

**Goal:** The current batch buffers are sized for `max_batch = 512` tokens, allocating ~120 MB. Most of this is unused and wastes memory. Since the batched path now uses single-token Q8K buffers per matmul (reused per token), the batch Q8K buffers (`batch_q8_qs`, `batch_q8_d`, `batch_q8_bsums`) and gemm scratch (`batch_q8_a`, `batch_ffn_q8_a`, `gemm_scratch`, `batch_ffn_q8_qs`, `batch_ffn_q8_d`, `batch_ffn_q8_bsums`) are unused. Remove them and leave the activation buffers at `max_batch = 512` (those are still needed for the batched attention kernel and per-token activation storage).

**Files:**
- Modify: `src/inference/forward.rs` — remove unused batch Q8K/gemm fields and allocations

- [ ] **Step 1: Remove unused batch fields from Gemma4State**

In `src/inference/forward.rs`, remove these fields from the struct:

```
    pub(crate) batch_q8_qs: Vec<i8>,
    pub(crate) batch_q8_d: Vec<f32>,
    pub(crate) batch_q8_bsums: Vec<i16>,
    pub(crate) batch_ffn_q8_qs: Vec<i8>,
    pub(crate) batch_ffn_q8_d: Vec<f32>,
    pub(crate) batch_ffn_q8_bsums: Vec<i16>,
    pub(crate) batch_q8_a: Vec<u8>,
    pub(crate) batch_ffn_q8_a: Vec<u8>,
    pub(crate) gemm_scratch: Vec<u8>,
```

And their corresponding allocations in `new()`:

```
            batch_q8_qs: vec![0; (max_q8_dim + 12) * max_batch],
            batch_q8_d: vec![0.0; nb_max * max_batch],
            batch_q8_bsums: vec![0; nb_max * 16 * max_batch],
            batch_ffn_q8_qs: vec![0; (max_ffn + 12) * max_batch],
            batch_ffn_q8_d: vec![0.0; n_blocks_ffn * max_batch],
            batch_ffn_q8_bsums: vec![0; n_blocks_ffn * 16 * max_batch],
            batch_q8_a: vec![0; q8_a_groups * block_q8_kx4_size],
            batch_ffn_q8_a: vec![0; q8_a_groups * block_q8_kx4_size],
            gemm_scratch: vec![0; 128],
```

Also remove the now-unused locals `max_q8_dim`, `nb_max`, `block_q8_kx4_size`, `q8_a_groups` if nothing else uses them.

**Keep** all the `batch_x`, `batch_q`, `batch_k`, `batch_v`, `batch_attn_out`, `batch_wo_out`, `batch_attn_res`, `batch_gate`, `batch_up`, `batch_down`, `batch_x_norm`, `batch_ple_signal`, `max_batch` fields — these store per-token activations needed across ops.

- [ ] **Step 2: Fix any compile errors from removed fields**

If `forward_batch_layer.rs` or any other file references the removed fields, update them. With the Task 1 rewrite, the layer function uses `state.q8_qs` / `state.ffn_q8_qs` (single-token buffers) instead, so no references should remain.

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep "^error" | head -10
```

- [ ] **Step 3: Calculate memory savings**

```
Removed:
  batch_q8_qs:      (12288 + 12) * 512 * 1 =  6,297,600 bytes
  batch_q8_d:       48 * 512 * 4            =     98,304
  batch_q8_bsums:   48 * 16 * 512 * 2       =    786,432
  batch_ffn_q8_qs:  (12288 + 12) * 512 * 1  =  6,297,600
  batch_ffn_q8_d:   48 * 512 * 4            =     98,304
  batch_ffn_q8_bsums: 48 * 16 * 512 * 2     =    786,432
  batch_q8_a:       128 * 56064             =  7,176,192
  batch_ffn_q8_a:   128 * 56064             =  7,176,192
  gemm_scratch:     128                     =        128
                                    TOTAL ≈ 28.7 MB saved
```

- [ ] **Step 4: Commit**

```bash
git add src/inference/forward.rs
git commit -m "refactor: remove unused batch Q8K/gemm buffers — save ~29 MB per state"
```

---

## Task 4: N=1 bit-exact test

**Goal:** Verify `forward_batch(&[BOS])` produces identical logits to `forward_one_graph(BOS)`. This is the critical correctness gate. The existing `tests/forward_batch_verify.rs` should work with the rewritten path.

**Files:**
- Read: `tests/forward_batch_verify.rs` (already exists from previous plan)

- [ ] **Step 1: Run the existing N=1 test**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test forward_batch_verify -- --nocapture 2>&1 | tail -20
```

Expected: PASS with "forward_batch(N=1) bit-exact match".

If it fails: debug by adding per-layer L2-norm prints in both paths. The most likely divergence sources are:
- **Q projection**: the batched path re-quantizes from `batch_x_norm` per-token, vs. the graph path quantizing from `x_norm` — make sure both use the same norm output.
- **Attention**: the fused batched kernel with N=1 must match the per-position loop in forward_graph.rs.
- **FFN dual dispatch**: verify the Q4K dual path matches.

**Do NOT proceed past this task until the test passes.**

- [ ] **Step 2: Run existing regression tests**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression 2>&1 | tail -6
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -15
```

Both must still pass — the existing `forward_one_graph` path is untouched.

- [ ] **Step 3: Commit (if test file needed changes)**

```bash
git add tests/forward_batch_verify.rs
git commit -m "test: forward_batch(N=1) bit-exact gate PASS"
```

---

## Task 5: Wire `generate.rs` — batched prefill

**Goal:** Switch `generate.rs` to use `forward_batch` for the prompt prefill (all tokens in one call). Keep decode using `forward_one_graph` — the known-working path.

**Files:**
- Modify: `src/inference/generate.rs`

- [ ] **Step 1: Read current generate.rs**

```bash
grep -n "forward_one_graph\|forward_batch" src/inference/generate.rs
```

- [ ] **Step 2: Replace prefill loop with single `forward_batch` call**

In `src/inference/generate.rs`, find the prefill section (currently lines ~104-112):

```rust
        // 4. Prefill: forward each prompt token (discard logits except last)
        let n_prompt = tokens.len();
        for &tok in &tokens[..n_prompt - 1] {
            self.state.forward_one_graph(&self.model, tok, &self.graph_pool);
        }
        let mut logits_snapshot = {
            let logits = self.state.forward_one_graph(&self.model, tokens[n_prompt - 1], &self.graph_pool);
            logits.to_vec()
        };
```

Replace with:

```rust
        // 4. Prefill: batched forward (all prompt tokens at once)
        let mut logits_snapshot = {
            let logits = self.state.forward_batch(&self.model, &tokens, &self.graph_pool);
            logits.to_vec()
        };
```

Leave the decode loop unchanged — it still uses `forward_one_graph`.

- [ ] **Step 3: Build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep "^error" | head -5
```

- [ ] **Step 4: Run smoke test**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test gemma4_smoke -- --nocapture 2>&1 | tail -20
```

Expected: generates coherent text (same quality as before).

- [ ] **Step 5: Commit**

```bash
git add src/inference/generate.rs
git commit -m "feat: generate.rs uses forward_batch for prefill

Prompt tokens evaluated in one batched call with work-stealing matvecs.
Decode loop stays on forward_one_graph (known working)."
```

---

## Task 6: Baseline benchmark

**Goal:** Record prefill and decode throughput numbers. This is the llama.cpp-matching baseline we'll optimize against.

**Files:**
- Read: `tests/bench_decode_speed.rs`

- [ ] **Step 1: Run decode benchmark (includes prefill now)**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test bench_decode_speed -- --nocapture 2>&1 | tail -30
```

Record:
- Prefill tok/s (batched forward_batch)
- Decode tok/s (forward_one_graph, should be unchanged)

- [ ] **Step 2: Run on Pi 5 (if available)**

```bash
ssh peter@10.46.0.27 "cd /tmp/olorin && GEMMA4_DIAG=1 ./olorin_test 2>&1 | tail -20"
```

Record Pi 5 numbers for comparison with llama.cpp baseline (6.4 tok/s decode).

- [ ] **Step 3: Document results**

Add a comment to this commit with the numbers:

```bash
git commit --allow-empty -m "bench: batched forward baseline numbers

WSL x86 ($(nproc) threads):
  prefill: [XX.X] tok/s  (was ~8.5 t/s token-by-token)
  decode:  [XX.X] tok/s  (unchanged)

Pi 5 (4 threads):
  prefill: [XX.X] tok/s
  decode:  [XX.X] tok/s  (was ~2.4 t/s)"
```

---

## Summary of changes

| What | Before | After |
|------|--------|-------|
| Matmul threading in batch path | Thread 0 only (gemm) | All threads work-steal (matvec_step) |
| Barrier pattern | Same count, 7 threads idle | Same as forward_graph.rs per token |
| Batch Q8K buffers | ~29 MB allocated, used for gemm | Removed, reuse single-token buffers |
| Prefill in generate.rs | Token-by-token forward_one_graph | Single forward_batch call |
| Gemm kernel/matmul_batch.rs | Used | Kept but unused (future optimization) |
| Correctness | Untested | Bit-exact N=1 gate + regression suite |
