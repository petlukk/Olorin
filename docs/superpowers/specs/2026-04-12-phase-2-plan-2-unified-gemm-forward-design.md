# Phase 2 — Plan 2: Unified Gemm Forward Path

## Goal

Replace `forward_one_graph` (Path B) with a unified `forward` function that always uses the Q4K 8x8 gemm kernel for all Q4K matmuls, handling any N (1 for decode, N>1 for prompt eval). Land a fused batched attention Ea kernel. Close the 4.4x prefill gap vs llama.cpp.

## Decision Log

Decisions made during brainstorming (2026-04-12):

1. **Gemm everywhere, no gemv.** The gemm kernel with N=1 degenerates to matvec. No reason to keep separate matvec dispatch for decode. One kernel, one path.
2. **Separate `forward_batch` first (Approach A), collapse later.** Build the new unified `forward` as a replacement for Path B (`forward_one_graph`). Path A (`forward_one` + ThreadPool) stays until the new path is proven, then gets deleted.
3. **Fused batched attention Ea kernel (Approach B).** One kernel per head does QK^T → causal mask → softmax → V multiply without writing scores to DRAM. x86 AVX2 + ARM NEON+dotprod. Causal mask is computed inline (no separate mask buffer).
4. **Supporting ops loop in Rust (Approach B).** rmsnorm, rope, gelu_mul, q8k quant — call existing single-column Ea kernels N times in a Rust loop. Not compute-bound, not worth writing batched kernels for now.

## What Ships

- **1 new Ea kernel** (x86 + ARM): `attn_fused_batched` — per-head fused causal attention
- **1 new Rust function**: `Gemma4State::forward()` replacing `forward_one_graph()`, gemm for all Q4K matmuls regardless of N
- **`generate.rs` update**: `forward(&prompt_tokens)` for prefill, `forward(&[token])` for decode
- **Path A deletion**: `forward_one`, ThreadPool-based matmul wrappers, associated dead code

## Architecture

### Forward Function

```
forward(tokens: &[u32]) -> &[f32]
  N = tokens.len()

  embed + scale (loop N, existing q6k_embed_lookup kernel)

  for each layer (0..n_layers):
    // ── Attention block ──
    rmsnorm(x -> x_norm)                    [loop N, existing kernel]
    q8k_quant(x_norm -> q8_qs/d/bsums)     [loop N, existing kernel]
    Q = gemm(wq_repacked, q8k_input, N)     [q4k_8x8_q8k_gemm]
    K = gemm(wk_repacked, q8k_input, N)     [q4k_8x8_q8k_gemm]
    V = gemm(wv_repacked, q8k_input, N)     [q4k_8x8_q8k_gemm]
    per-head Q norm + rope                   [loop N, existing kernels]
    per-head K norm + rope                   [loop N, existing kernels]
    KV cache store (N positions)
    attn_fused_batched per head              [new Ea kernel]
    q8k_quant(attn_out -> q8k)              [loop N]
    Wo = gemm(wo_repacked, q8k, N)          [q4k_8x8_q8k_gemm]
    residual + post-attn norm                [loop N]

    // ── FFN block ──
    q8k_quant(attn_res -> q8k)              [loop N]
    gate = gemm(wgate_repacked, q8k, N)     [q4k_8x8_q8k_gemm]
    up   = gemm(wup_repacked, q8k, N)       [q4k_8x8_q8k_gemm]
    gelu_mul(gate, up -> gate)               [loop N, existing kernel]
    q8k_quant(gate -> q8k)                  [loop N]
    down = gemm(wdown_repacked, q8k, N)     [q4k_8x8_q8k_gemm]
    residual + post-ffn norm                 [loop N]

    // ── PLE ──
    PLE gating                               [loop N]

  final norm                                 [loop N]
  output matmul (Q6K embed weights)          [matvec x N — not Q4K, not repacked]
  return &logits[last token]
```

### Gemm Calls

All Q4K matmuls use the existing `q4k_8x8_q8k_gemm` kernel from Plan 1. Input is Q8K-quantized column-major `[n_cols, N]`, weights are repacked Q4K 8x8 layout (already done in Phase A/B). Output is f32 column-major `[n_rows, N]`.

7 gemm calls per layer: wq, wk, wv, wo, wgate, wup, wdown.

The Q6K output/embedding matmul stays as matvec-in-a-loop (Q6K weights are not repacked to 8x8, and this is a single matmul at the end, not per-layer).

### Batched Buffers

Column-major `[dim, N]` for all activation tensors. Allocated once in `Gemma4State::new()` with `max_batch = 512` (matches Gemma 4 sliding window cap). Decode uses the same buffers with N=1.

New fields on `Gemma4State`:

```rust
// Batched activation buffers — column-major, sized for max_batch.
// Column k of an [dim, N] tensor is at offset k * dim.
batch_x: Vec<f32>,          // [hd, max_batch]
batch_x_norm: Vec<f32>,     // [hd, max_batch]
batch_q: Vec<f32>,          // [n_heads * head_dim_k, max_batch]
batch_k: Vec<f32>,          // [n_kv_heads * head_dim_k, max_batch]
batch_v: Vec<f32>,          // [n_kv_heads * head_dim_v, max_batch]
batch_attn_out: Vec<f32>,   // [n_heads * head_dim_k, max_batch]
batch_wo_out: Vec<f32>,     // [hd, max_batch]
batch_attn_res: Vec<f32>,   // [hd, max_batch]
batch_gate: Vec<f32>,       // [ffn_dim, max_batch]
batch_up: Vec<f32>,         // [ffn_dim, max_batch]
batch_down: Vec<f32>,       // [hd, max_batch]
batch_q8_qs: Vec<i8>,       // Q8K quantized input (largest dim * max_batch)
batch_q8_d: Vec<f32>,       // Q8K scales
batch_q8_bsums: Vec<i16>,   // Q8K block sums
max_batch: usize,
```

### Fused Attention Kernel

One Ea kernel per architecture (x86 AVX2, ARM NEON+dotprod). Called once per head, dispatched across threads.

**Signature:**

```
export func attn_fused_batched(
    q: *f32,             // [head_dim, N] query vectors for this head
    k_cache: *f16,       // [head_dim, n_kv] full K cache including new entries
    v_cache: *f16,       // [head_dim, n_kv] full V cache including new entries
    out dst: *mut f32,   // [head_dim, N] output
    head_dim: i32,
    n_kv: i32,           // total K/V positions (seq_len_before + N)
    n_batch: i32,        // N
    cache_start: i32,    // seq_len_before (for causal mask computation)
    attn_scale: f32      // 1/sqrt(head_dim)
)
```

**Causal mask logic (inside kernel):** For query token `i` (0-indexed within the batch), valid K positions are `0 .. cache_start + i + 1`. Positions `>= cache_start + i + 1` get `-inf` before softmax. No separate mask buffer.

**Inner loop per query token `i`:**

1. Compute `scores[j] = dot(q[i], k_cache[j]) * attn_scale` for `j = 0..n_kv`
2. Set `scores[j] = -inf` for `j >= cache_start + i + 1`
3. Row-wise softmax over `scores[0..n_kv]`
4. `dst[i] = sum_j(scores[j] * v_cache[j])` (weighted sum)

Scores buffer: at most 512 floats per query token (sliding window cap) = 2 KB. Fits in L1.

**Thread dispatch:** Same pattern as current `attention_decode` — split `n_heads` across threads, each thread processes its assigned heads by calling the kernel once per head.

### KV Cache

Before attention, store all N K/V vectors into the cache at positions `seq_len .. seq_len + N`. The cache already supports positional writes via `cache.store()`; this just needs to be called N times (or a batch store helper).

After `forward()` returns, advance `seq_len += N`.

### Thread Strategy

Follows `forward_one_graph` pattern:

- Gemm calls: work-stealing across row tiles (inherited from existing gemm dispatch)
- Attention: heads split across threads
- Supporting ops (norm, rope, quant, gelu, residual, PLE): thread 0 with barriers between parallel sections

### generate.rs Changes

```rust
// Before (current):
for &tok in &tokens[..n_prompt - 1] {
    self.state.forward_one_graph(&self.model, tok, &self.graph_pool);
}
let logits = self.state.forward_one_graph(&self.model, tokens[n_prompt - 1], &self.graph_pool);

// After:
let logits = self.state.forward(&self.model, &tokens, &self.graph_pool);

// Decode loop stays the same shape:
let logits = self.state.forward(&self.model, &[token_id], &self.graph_pool);
```

## Correctness

- **Decode (N=1):** Must remain bit-exact with current `forward_one_graph` output. `gemma4_parallel_regression` snapshot must pass without regeneration.
- **Batched (N>1):** Verify layer-by-layer L2 norms against llama-eval-callback dumps, same approach as `gemma4_verify`. Drift < 1e-4 at each layer boundary.
- **Attention:** Fused kernel tested standalone against N independent calls to existing `attention_decode`.

## Performance Targets

- **Prompt eval:** >= 15 t/s on x86 workstation (current: ~8.5 t/s token-by-token). llama.cpp reference: 37.3 t/s.
- **Decode:** No regression from current ~8.7 t/s on x86.
- **Pi 5:** Expect larger prefill gains (memory-bandwidth-bound, gemm amortizes weight loads).

## What Gets Deleted

- `Gemma4State::forward_one()` (Path A)
- ThreadPool-based matmul wrappers in `matmul.rs` that are only called from Path A
- `forward_attn.rs` Path A attention helpers (if separate from Path B)
- Any dead code surfaced by removing Path A callers

## Out of Scope

- Deleting matvec `.ea` kernel files from `kernels/` (follow-up cleanup)
- Q5K/Q6K gemm kernels (embed/unembed stay matvec)
- Batched Ea kernels for supporting ops (rmsnorm, rope, gelu_mul, q8k quant)
- Flash attention / online softmax optimization
- Pi 5 deployment and cross-arch testing
