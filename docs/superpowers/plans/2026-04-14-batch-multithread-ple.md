# Batch Multi-Threading + PLE GEMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Multi-thread all single-threaded operations in the batch forward layer and replace scalar PLE matvecs with batched GEMM, eliminating thread-0 bottlenecks that waste 3 of 4 Pi 5 cores.

**Architecture:** Three changes: (1) parallelize norms/residuals across tokens using the existing `ith/nth` striding pattern, (2) replace per-token scalar matvec in PLE with batched quant + repack + GEMM using the same infrastructure as Q/K/V/gate/up/down, (3) split `forward_batch_layer.rs` (632 lines, already over 500) by extracting PLE into its own module. All operations use work-stealing GEMM or token-strided parallelism — zero thread-0-only serial loops remain.

**Tech Stack:** Rust, Ea SIMD kernels, existing `matmul_graph::q4k_gemm_8x8_batch_ws` / `matvec_batch_ws`

---

## Current Bottlenecks (from GEMMA4_TIMING=1, 16 tokens × 35 layers)

| Operation | Time | % | Threading |
|-----------|------|---|-----------|
| gemm_down | 230ms | 33.4% | multi (work-stealing) |
| gemm_gate | 123ms | 17.8% | multi |
| gemm_up | 122ms | 17.7% | multi |
| gemm_wo | 73ms | 10.6% | multi |
| gemm_q | 60ms | 8.7% | multi |
| **post_ffn+ple** | **48ms** | **7.0%** | **thread 0 only** |
| gelu_mul | 11ms | 1.6% | multi |
| q_norm_rope | 4.3ms | 0.6% | thread 0 only |
| gemm_k | 4.2ms | 0.6% | multi |
| gemm_v | 3.2ms | 0.5% | multi |
| attention | 2.8ms | 0.4% | multi (head-split) |
| kv_norm_rope | 2.1ms | 0.3% | thread 0 only |
| quant_down | 2.4ms | 0.4% | multi |
| post_attn+norm | 1.7ms | 0.2% | thread 0 only |
| attn_norm | 0.6ms | 0.1% | thread 0 only |
| other (quant/repack) | ~2ms | ~0.3% | mixed |

**Thread-0-only total: ~57ms (8.2%)** — threads 1-3 spin-wait at barriers during this time.

### PLE breakdown (inside post_ffn+ple, per token per layer)
The PLE block does this per token, sequentially, 35×16 = 560 times:
1. `quant_input(batch_x[t], q8_qs, q8_d, q8_bsums)` — quantize 1536-dim
2. `matvec(inp_gate, q8_*, ple_gate, 256, 1536)` — scalar 1536→256
3. `gelu_mul(ple_gate, ple_signal[t], ple_gate, 256)` — activation
4. `quant_input(ple_gate, ple_q8_*, 256)` — quantize 256-dim
5. `matvec(proj, ple_q8_*, ple_out, 1536, 256)` — scalar 256→1536
6. `rmsnorm(ple_out, post_norm, 1536)` — normalize
7. `vec_add(batch_x[t], ple_out, batch_x[t], 1536)` — residual add

That's **1120 scalar matvecs** total. Batching into GEMM should give ~4× speedup.

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/inference/forward_batch_layer.rs` | Modify | Remove PLE from section 10, parallelize norms/residuals |
| `src/inference/forward_batch_ple.rs` | Create | Batched PLE: parallel quant + repack + GEMM for inp_gate and proj |
| `src/inference/forward.rs` | Modify | Add PLE batch buffers to `Gemma4State` |
| `src/inference/mod.rs` | Modify | Add `mod forward_batch_ple;` |
| `tests/batch_ple.rs` | Create | End-to-end test: PLE output matches single-token path |

## Scratch buffers needed for PLE GEMM

PLE operates on different dimensions than the main projections:
- **inp_gate**: weight is `[hd=1536, ple_dim=256]`, input is `batch_x` (hd-wide), output is `ple_gate` (ple_dim-wide)
  - Reuses existing `batch_q8_a` (already sized for hd input) for the Q8K repack
  - Needs output buffer: `batch_ple_gate_out[ple_dim * n_pad]`
- **proj**: weight is `[ple_dim=256, hd=1536]`, input is `ple_gate` (ple_dim-wide)
  - Needs its own Q8K buffers: `batch_ple_q8_qs`, `batch_ple_q8_d`, `batch_ple_q8_bsums`, `batch_ple_q8_a` (sized for ple_dim input)
  - Needs output buffer: `batch_ple_out[hd * n_pad]`

Since `ple_dim=256` = 1 Q8K superblock, the repack is minimal.

---

### Task 1: Parallelize norms and residuals across tokens

**Files:**
- Modify: `src/inference/forward_batch_layer.rs`

Currently, sections 1a (attn_norm), 3 (q_norm_rope), 4 (kv_norm_rope), 7 (post_attn+ffn_norm), and the residual parts of section 10 (post_ffn) are all gated by `if ith == 0` and loop over tokens serially. Change them to stride across tokens using `ith/nth`.

The key constraint is scratch buffers: `x_norm` is a single-token scratch buffer shared across all operations. For norms that write to `batch_x_norm` (which is token-indexed), each thread can write directly to the token's slice — no scratch conflict. For norms that write to `x_norm` (q_norm, k_norm, post_ffn_norm), we need per-thread scratch or in-place output.

- [ ] **Step 1: Parallelize attn_norm (section 1a)**

The attn_norm writes `batch_x_norm[t*hd..]` which is already per-token. Each thread handles a subset of tokens. Replace:

```rust
// ── 1a. Attn norm (all threads, token-strided) ─────────────
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
```

- [ ] **Step 2: Parallelize post_attn residual + FFN norm (section 7)**

Same pattern — each token's output goes to `batch_attn_res[t*hd..]` and `batch_x_norm[t*hd..]`. No shared scratch needed:

```rust
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
```

Note: the original uses `x_norm` as intermediate scratch for post_attn_norm then copies to `batch_attn_res`. We can use `batch_x_norm[off..]` as intermediate scratch instead (it gets overwritten immediately after by ffn_norm), avoiding the shared `x_norm` buffer entirely.

- [ ] **Step 3: Parallelize Q norm (section 3) — needs per-thread scratch**

Q norm uses `x_norm` as scratch for per-head rmsnorm, then copies back to `batch_q`. With 4 threads, we need 4 scratch slots. Add `batch_head_scratch: Vec<f32>` to `Gemma4State`, sized `max_head_dim * nth`. Each thread uses `batch_head_scratch[ith * max_head_dim..]`.

In `forward.rs`, add to `Gemma4State`:
```rust
pub(crate) batch_head_scratch: Vec<f32>, // [max_head_dim * n_threads]
```

In allocation:
```rust
batch_head_scratch: vec![0.0; max_head_dim * n_threads],
```

Where `max_head_dim = 512` (global layers) and `n_threads` comes from the thread pool.

Then in section 3, stride across tokens:
```rust
// ── 3. Q norm + RoPE (all threads, token-strided) ──────────
let t0 = t_start!();
{
    let head_scratch = &mut state.batch_head_scratch[ith * head_dim..(ith + 1) * head_dim];
    let mut t = ith;
    while t < n {
        compute_rope_tables(
            &mut state.cos_table, &mut state.sin_table,
            seq_len + t, n_rot, rope_theta, freq_factors,
        );
        if !lw.q_norm.is_null() {
            for h in 0..n_heads {
                let off = t * qkv_dim + h * head_dim;
                ffi_inference::gemma4_rmsnorm(
                    unsafe { state.batch_q.as_ptr().add(off) }, lw.q_norm,
                    head_scratch.as_mut_ptr(), head_dim as i32, model.rms_eps,
                );
                state.batch_q[off..off + head_dim].copy_from_slice(&head_scratch[..head_dim]);
            }
        }
        ffi_inference::gemma4_rope(
            unsafe { state.batch_q.as_mut_ptr().add(t * qkv_dim) },
            state.cos_table.as_ptr(), state.sin_table.as_ptr(),
            head_dim as i32, n_heads as i32,
        );
        t += nth;
    }
}
barrier.wait(); // B5
if ith == 0 { t_accum!(t0, q_norm_rope, tm!()); }
```

**Problem:** `cos_table` and `sin_table` are shared mutable state. Each thread computes different RoPE tables for different tokens. We need per-thread RoPE tables. Add to `Gemma4State`:
```rust
pub(crate) batch_cos_tables: Vec<f32>, // [max_rope_dim * n_threads]
pub(crate) batch_sin_tables: Vec<f32>, // [max_rope_dim * n_threads]
```

Then each thread uses its own slice:
```rust
let cos_off = ith * rope_dim_max;
let sin_off = ith * rope_dim_max;
// ... compute_rope_tables into batch_cos_tables[cos_off..], batch_sin_tables[sin_off..]
```

But `compute_rope_tables` takes `&mut state.cos_table` — it writes to `Gemma4State` fields. We need a free-standing version that writes to arbitrary slices. Extract:
```rust
pub(crate) fn compute_rope_tables_into(
    cos: &mut [f32], sin: &mut [f32],
    pos: usize, n_rot: usize, theta: f64, freq_factors: Option<&[f32]>,
)
```

- [ ] **Step 4: Parallelize KV norm + RoPE (section 4) — same pattern**

Same approach as Q norm. K norm needs per-thread head scratch (already added). RoPE needs per-thread cos/sin tables (already added). V bare_rmsnorm is in-place, so it's trivially parallelizable:

```rust
// K/V norms + RoPE + cache store
let t0 = t_start!();
{
    let mut t = ith;
    while t < n {
        let head_scratch = &mut state.batch_head_scratch[ith * head_dim..(ith + 1) * head_dim];
        compute_rope_tables_into(
            &mut state.batch_cos_tables[ith * n_rot..],
            &mut state.batch_sin_tables[ith * n_rot..],
            seq_len + t, n_rot, rope_theta, freq_factors,
        );
        if !lw.k_norm.is_null() {
            for h in 0..n_kv_heads {
                let off = t * kv_dim + h * head_dim;
                ffi_inference::gemma4_rmsnorm(
                    unsafe { state.batch_k.as_ptr().add(off) }, lw.k_norm,
                    head_scratch.as_mut_ptr(), head_dim as i32, model.rms_eps,
                );
                state.batch_k[off..off + head_dim].copy_from_slice(&head_scratch[..head_dim]);
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
            state.batch_cos_tables[ith * n_rot..].as_ptr(),
            state.batch_sin_tables[ith * n_rot..].as_ptr(),
            head_dim as i32, n_kv_heads as i32,
        );
        t += nth;
    }
}
barrier.wait(); // sync before cache store
// Cache store must be sequential (single writer to cache)
if ith == 0 {
    state.cache.store_batch(
        il, &state.batch_k[..kv_dim * n], &state.batch_v[..kv_dim_v * n], n,
    );
}
barrier.wait();
if ith == 0 { t_accum!(t0, kv_norm_rope_cache, tm!()); }
```

- [ ] **Step 5: Parallelize post-FFN residual (section 10, non-PLE part)**

The residual add + post_ffn_norm + output scale are all per-token independent. Split section 10 into two parts:
1. Residual + post_ffn_norm + scale — parallelized across tokens
2. PLE — extracted to Task 2

```rust
// ── 10a. Post-FFN residual + scale (all threads, token-strided) ─
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
        let out_scale = lw.layer_output_scale;
        if out_scale != 1.0 {
            ffi_inference::vec_scale_f32(
                state.batch_x[off..].as_ptr(), state.batch_x[off..].as_mut_ptr(),
                out_scale, hd as i32,
            );
        }
        t += nth;
    }
}
barrier.wait(); // sync before PLE reads batch_x
```

Note: output scale moves BEFORE PLE. PLE adds to `batch_x` after scale, which matches the original order (scale was inside the same per-token loop, before PLE's vec_add).

Wait — looking at the original code more carefully: scale happens AFTER PLE's vec_add. The order is:
1. residual add → `batch_x[t]`
2. PLE → adds to `batch_x[t]`
3. scale `batch_x[t]`

So scale must happen after PLE. We'll need to handle this. Options:
- Scale in the PLE function after adding
- Or: parallelize residual, then barrier, then PLE (batched), then barrier, then parallel scale

The cleanest: parallelize residual+norm, barrier, PLE GEMM (all threads), barrier, parallel scale.

- [ ] **Step 6: Add per-thread scratch buffers to Gemma4State**

In `src/inference/forward.rs`, add:

```rust
pub(crate) batch_head_scratch: Vec<f32>,   // [max_head_dim * n_threads]
pub(crate) batch_cos_tables: Vec<f32>,     // [max_rope_dim * n_threads]
pub(crate) batch_sin_tables: Vec<f32>,     // [max_rope_dim * n_threads]
```

Allocate in `Gemma4State::new()`:
```rust
let max_head_dim = *model.head_dim_k.iter().max().unwrap();
let max_rope_dim = model.rope_dim_swa.max(model.rope_dim_global);
// ...
batch_head_scratch: vec![0.0; max_head_dim * n_threads],
batch_cos_tables: vec![0.0; max_rope_dim * n_threads],
batch_sin_tables: vec![0.0; max_rope_dim * n_threads],
```

- [ ] **Step 7: Extract `compute_rope_tables_into` free function**

In `src/inference/forward.rs`, add a standalone version that writes to arbitrary slices:

```rust
pub(crate) fn compute_rope_tables_into(
    cos: &mut [f32], sin: &mut [f32],
    pos: usize, n_rot: usize, theta: f64, freq_factors: Option<&[f32]>,
) {
    for i in 0..n_rot / 2 {
        let freq = 1.0 / theta.powf(2.0 * i as f64 / n_rot as f64);
        let freq = match freq_factors {
            Some(ff) => freq / ff[i] as f64,
            None => freq,
        };
        let val = pos as f64 * freq;
        cos[i] = val.cos() as f32;
        sin[i] = val.sin() as f32;
    }
}
```

- [ ] **Step 8: Build and run profiling on Pi**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo build --release --target aarch64-unknown-linux-gnu
scp -i ~/.ssh/id_ed25519_pi target/aarch64-unknown-linux-gnu/release/olorin peter@10.46.0.27:~/olorin
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 \
  'echo -e "Hello\n/quit" | GEMMA4_TIMING=1 ~/olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf 2>&1 | tail -40'
```

Expected: attn_norm, q_norm_rope, kv_norm_rope, post_attn+norm should all drop by ~3-4×. post_ffn+ple will still be high (PLE not batched yet).

- [ ] **Step 9: Commit**

```bash
git add src/inference/forward_batch_layer.rs src/inference/forward.rs
git commit -m "perf: parallelize norms + residuals across tokens in batch forward"
```

- [ ] **Step 10: N=1 bit-exact test**

**Goal:** Verify `forward_batch(&[BOS])` still produces identical logits to `forward_one_graph(BOS)`.

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test forward_batch_verify -- --nocapture 2>&1 | tail -15
```

Expected: PASS with "forward_batch(N=1) bit-exact match".

**Do NOT proceed until this passes.** If it fails, debug per-layer.

- [ ] **Step 11: Regression tests**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test gemma4_parallel_regression 2>&1 | tail -6
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -15
```

Both must pass.

x86 smoke test:
```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test gemma4_smoke -- --nocapture 2>&1 | tail -10
```

- [ ] **Step 12: Cross-compile and Pi test**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo build --release --target aarch64-unknown-linux-gnu 2>&1 | tail -3
scp -i ~/.ssh/id_ed25519_pi target/aarch64-unknown-linux-gnu/release/olorin peter@10.46.0.27:~/olorin
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 \
  'echo -e "Hello\n/quit" | timeout 90 ~/olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf 2>&1'
```

Must complete within 90s and produce coherent output. If it hangs, the threading change has a barrier bug.

---

### Task 2: Batch PLE with GEMM

**Files:**
- Create: `src/inference/forward_batch_ple.rs`
- Modify: `src/inference/forward_batch_layer.rs` (call into new module)
- Modify: `src/inference/forward.rs` (add PLE batch buffers)
- Modify: `src/inference/mod.rs` (add `mod forward_batch_ple;`)

The PLE block does two matvecs per token: `inp_gate` (1536→256) and `proj` (256→1536). Currently 560 scalar matvecs across the prefill. Replace with:
1. Batch quant all N tokens' `batch_x` (for inp_gate input) — **already done in section 10a residual**
2. Repack Q8K for GEMM
3. GEMM: `inp_gate` weight × all tokens → `batch_ple_gate[ple_dim * n_pad]`
4. Parallel gelu_mul with PLE signal
5. Batch quant the 256-dim `batch_ple_gate`
6. Repack Q8K for GEMM (different dimension: ple_dim)
7. GEMM: `proj` weight × all tokens → `batch_ple_out[hd * n_pad]`
8. Parallel post_norm + vec_add into `batch_x`

- [ ] **Step 1: Add PLE batch buffers to Gemma4State**

In `src/inference/forward.rs`:

```rust
// PLE batch buffers
pub(crate) batch_ple_gate_out: Vec<f32>,   // [ple_dim * max_batch]
pub(crate) batch_ple_proj_out: Vec<f32>,   // [hd * max_batch]
pub(crate) batch_ple_q8_qs: Vec<i8>,      // [(ple_dim + 12) * max_batch]
pub(crate) batch_ple_q8_d: Vec<f32>,      // [(ple_dim/256) * max_batch]
pub(crate) batch_ple_q8_bsums: Vec<i16>,  // [(ple_dim/256)*16 * max_batch]
pub(crate) batch_ple_q8_a: Vec<u8>,       // repack tiles for ple_dim input
```

Allocations:
```rust
let ple_dim = model.ple_dim.max(1);
let ple_nb = (ple_dim / 256).max(1);
batch_ple_gate_out: vec![0.0; ple_dim * max_batch],
batch_ple_proj_out: vec![0.0; hd * max_batch],
batch_ple_q8_qs: vec![0; (ple_dim + 12) * max_batch],
batch_ple_q8_d: vec![0.0; ple_nb * max_batch],
batch_ple_q8_bsums: vec![0; ple_nb * 16 * max_batch],
batch_ple_q8_a: vec![0u8; (max_batch / 4) * ple_nb * 1168],
```

- [ ] **Step 2: Create `forward_batch_ple.rs`**

```rust
//! Batched PLE (Per-Layer Embedding) — parallel quant + GEMM.
//!
//! Replaces the per-token scalar matvec loop in forward_batch_layer section 10.
//! inp_gate GEMM: [hd=1536] → [ple_dim=256], proj GEMM: [ple_dim=256] → [hd=1536].

use std::sync::atomic::{AtomicI32, Ordering};
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::Gemma4State;
use crate::inference::matmul;
use crate::inference::matmul_graph;
use crate::inference::threadpool::SpinBarrier;
use crate::kernels::ffi_inference;

use super::forward_batch_layer::{parallel_batch_quant, repack_q8_for_gemm, matvec_batch_step};
```

**Problem:** `parallel_batch_quant`, `repack_q8_for_gemm`, `matvec_batch_step` are currently private in `forward_batch_layer.rs`. Change them to `pub(crate)`.

Then the main function:

```rust
/// Batched PLE for all N tokens. All threads participate.
/// Requires batch_x to already contain the post-FFN residual (pre-PLE, pre-scale).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ple_batch(
    state: &mut Gemma4State, model: &Gemma4Model,
    il: usize, n: usize,
    barrier: &SpinBarrier, current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    let hd = model.hidden_dim;
    let ple_dim = model.ple_dim;
    let n_pad = (n + 3) & !3;
    let lw = &model.layers[il];

    if ple_dim == 0 || lw.inp_gate.is_null() || lw.proj.is_null() {
        return;
    }

    let ple_total = ple_dim * model.n_layers;
    let ple_off = il * ple_dim;

    // ── 1. Quant batch_x for inp_gate (all threads, token-strided) ──
    parallel_batch_quant(
        &state.batch_x, hd, n, n_pad,
        &mut state.batch_q8_qs, &mut state.batch_q8_d, &mut state.batch_q8_bsums,
        ith, nth,
    );
    barrier.wait();

    // ── 2. Repack Q8K + inp_gate GEMM ──────────────────────────────
    if ith == 0 {
        repack_q8_for_gemm(
            &state.batch_q8_qs, &state.batch_q8_d, &state.batch_q8_bsums,
            &mut state.batch_q8_a, hd, n_pad,
        );
    }
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();

    // inp_gate: [hd, ple_dim] — no repacked variant exists, use matvec fallback
    matvec_batch_step(
        None, lw.inp_gate_dtype, lw.inp_gate,
        state.batch_q8_a.as_ptr(),
        state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
        state.batch_q8_bsums.as_ptr(),
        state.batch_ple_gate_out.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        ple_dim, hd, n, n_pad, ple_dim,
        current_chunk, ith, nth,
    );
    barrier.wait();

    // ── 3. GELU-mul with PLE signal (all threads, token-strided) ───
    {
        let mut t = ith;
        while t < n {
            ffi_inference::gelu_mul(
                state.batch_ple_gate_out[t * ple_dim..].as_ptr(),
                state.batch_ple_signal[t * ple_total + ple_off..].as_ptr(),
                state.batch_ple_gate_out[t * ple_dim..].as_mut_ptr(),
                ple_dim as i32,
            );
            t += nth;
        }
    }
    barrier.wait();

    // ── 4. Quant ple_gate for proj (all threads, token-strided) ────
    parallel_batch_quant(
        &state.batch_ple_gate_out, ple_dim, n, n_pad,
        &mut state.batch_ple_q8_qs, &mut state.batch_ple_q8_d,
        &mut state.batch_ple_q8_bsums,
        ith, nth,
    );
    barrier.wait();

    // ── 5. Repack Q8K + proj GEMM ──────────────────────────────────
    if ith == 0 {
        repack_q8_for_gemm(
            &state.batch_ple_q8_qs, &state.batch_ple_q8_d, &state.batch_ple_q8_bsums,
            &mut state.batch_ple_q8_a, ple_dim, n_pad,
        );
    }
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();

    // proj: [ple_dim, hd] — no repacked variant, use matvec fallback
    matvec_batch_step(
        None, lw.proj_dtype, lw.proj,
        state.batch_ple_q8_a.as_ptr(),
        state.batch_ple_q8_qs.as_ptr(), state.batch_ple_q8_d.as_ptr(),
        state.batch_ple_q8_bsums.as_ptr(),
        state.batch_ple_proj_out.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, ple_dim, n, n_pad, hd,
        current_chunk, ith, nth,
    );
    barrier.wait();

    // ── 6. Post-norm + residual add (all threads, token-strided) ───
    {
        let mut t = ith;
        while t < n {
            let off = t * hd;
            if !lw.post_norm.is_null() {
                ffi_inference::gemma4_rmsnorm(
                    state.batch_ple_proj_out[off..].as_ptr(), lw.post_norm,
                    state.batch_ple_proj_out[off..].as_mut_ptr(), hd as i32, model.rms_eps,
                );
            }
            ffi_inference::vec_add_f32(
                state.batch_x[off..].as_ptr(), state.batch_ple_proj_out[off..].as_ptr(),
                state.batch_x[off..].as_mut_ptr(), hd as i32,
            );
            t += nth;
        }
    }
    barrier.wait();
}
```

- [ ] **Step 3: Wire PLE into forward_batch_layer.rs section 10**

Replace the entire PLE block in section 10 with a call:

```rust
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
if ith == 0 { t_accum!(t0, post_ffn_ple, tm!()); } // rename timing field later

// ── 10b. Batched PLE (all threads) ──────────────────────────
super::forward_batch_ple::ple_batch(
    state, model, il, n, barrier, current_chunk, ith, nth,
);

// ── 10c. Output scale (all threads, token-strided) ──────────
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
```

- [ ] **Step 4: Add `mod forward_batch_ple;` to mod.rs**

- [ ] **Step 5: Make helpers pub(crate) in forward_batch_layer.rs**

Change `parallel_batch_quant`, `repack_q8_for_gemm`, `matvec_batch_step` from private to `pub(crate)`.

- [ ] **Step 6: Build and run profiling on Pi**

Same deploy commands as Task 1 Step 8.

Expected: `post_ffn+ple` should drop from 48ms to roughly:
- Residual+norm: ~0.4ms (was ~1ms for non-PLE part, now 4-threaded)
- PLE inp_gate GEMM: ~2ms (256 output rows × 16 tokens, work-stealing)
- PLE proj GEMM: ~5ms (1536 output rows × 16 tokens, work-stealing)
- Misc (gelu, quant, repack): ~1ms
- Total ~8-10ms vs 48ms = **~5× speedup** on this section

- [ ] **Step 7: Commit**

```bash
git add src/inference/forward_batch_ple.rs src/inference/forward_batch_layer.rs \
        src/inference/forward.rs src/inference/mod.rs
git commit -m "perf: batched PLE with GEMM — replace 1120 scalar matvecs with 70 batched GEMMs"
```

- [ ] **Step 8: N=1 bit-exact test**

**Goal:** Verify `forward_batch(&[BOS])` still produces identical logits to `forward_one_graph(BOS)`.

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test forward_batch_verify -- --nocapture 2>&1 | tail -15
```

Expected: PASS with "forward_batch(N=1) bit-exact match".

**Do NOT proceed until this passes.** If it fails, debug per-layer.

- [ ] **Step 9: Regression tests**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test gemma4_parallel_regression 2>&1 | tail -6
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -15
```

Both must pass.

x86 smoke test:
```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test gemma4_smoke -- --nocapture 2>&1 | tail -10
```

- [ ] **Step 10: Cross-compile and Pi test**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo build --release --target aarch64-unknown-linux-gnu 2>&1 | tail -3
scp -i ~/.ssh/id_ed25519_pi target/aarch64-unknown-linux-gnu/release/olorin peter@10.46.0.27:~/olorin
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 \
  'echo -e "Hello\n/quit" | timeout 90 ~/olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf 2>&1'
```

Must complete within 90s and produce coherent output. If it hangs, the PLE batching has a barrier bug.

---

### Task 3: Add PLE timing to profiling

**Files:**
- Modify: `src/inference/forward_batch_layer.rs` (split post_ffn_ple timing)

- [ ] **Step 1: Split the timing field**

Add new fields to `BatchLayerTiming`:
```rust
pub post_ffn_residual: u64,   // was post_ffn_ple (residual-only part)
pub ple_total: u64,           // new: entire PLE GEMM block
pub output_scale: u64,        // new: output scaling
```

Remove `post_ffn_ple` and update `print_summary` accordingly.

- [ ] **Step 2: Wire timing into `ple_batch`**

Pass `Option<&mut BatchLayerTiming>` into `ple_batch` and accumulate `ple_total`.

- [ ] **Step 3: Build, deploy, run profiling**

- [ ] **Step 4: Commit**

```bash
git add src/inference/forward_batch_layer.rs src/inference/forward_batch_ple.rs
git commit -m "bench: split PLE timing in batch profiling"
```

- [ ] **Step 5: N=1 bit-exact test**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test forward_batch_verify -- --nocapture 2>&1 | tail -15
```

Expected: PASS. Timing changes should be zero-impact on output.

- [ ] **Step 6: Cross-compile and Pi profiling run**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo build --release --target aarch64-unknown-linux-gnu 2>&1 | tail -3
scp -i ~/.ssh/id_ed25519_pi target/aarch64-unknown-linux-gnu/release/olorin peter@10.46.0.27:~/olorin
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 \
  'echo -e "Hello\n/quit" | GEMMA4_TIMING=1 ~/olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf 2>&1 | tail -40'
```

Verify the new split timing fields (post_ffn_residual, ple_total, output_scale) print correctly and sum matches the previous post_ffn+ple total.

---

## Execution Order

Tasks 1 → 2 → 3 (sequential — each builds on the previous).

## Expected Impact

| Section | Before | After (est.) | Speedup |
|---------|--------|--------------|---------|
| attn_norm | 0.6ms | 0.2ms | 3× |
| q_norm_rope | 4.3ms | 1.5ms | 3× |
| kv_norm_rope | 2.1ms | 0.7ms | 3× |
| post_attn+norm | 1.7ms | 0.5ms | 3× |
| post_ffn+ple | 48ms | ~10ms | 5× |
| **Total saved** | | **~44ms** | |
| **New prefill** | 690ms | **~646ms** | 1.07× |

The 44ms savings is modest relative to the 690ms total because the GEMMs (already multi-threaded) dominate. But the PLE GEMM batching eliminates 1120 function call + quant overhead per prefill, and the norm parallelization removes artificial serialization points that would become bottlenecks at larger batch sizes.

## Risks

1. **`q6k_d_scratch` contention** — `matvec_batch_step` takes `d_scratch` for Q6K fallback path. If PLE weights are Q6K (check `inp_gate_dtype`/`proj_dtype`), all threads write to the same scratch. Mitigation: PLE weights are likely Q4K (check at runtime), and the Q4K GEMM path doesn't use `d_scratch`. If Q6K, we need per-thread d_scratch or PLE-specific scratch.

2. **Barrier count** — PLE adds ~6 barriers per layer (30 total for 5 sync points × 1 function call, but internal barriers in `ple_batch`). With 4 threads and spin barriers this is negligible, but worth monitoring.

3. **`batch_q8_a` reuse** — PLE step 2 reuses `batch_q8_a` for inp_gate input (same hd dimension). This is safe because the FFN GEMM is done by then. But PLE step 5 needs its own `batch_ple_q8_a` (different ple_dim dimension). Already accounted for in buffer design.
