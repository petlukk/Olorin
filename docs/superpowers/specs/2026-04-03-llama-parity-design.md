# Spec: llama.cpp 1:1 Pipeline Parity

**Date:** 2026-04-03
**Branch:** from_the_beginning
**Goal:** Make Olorin's decode and prefill pipeline identical to llama.cpp so we can benchmark apple-to-apple and find where to optimize.

**llama.cpp reference:** `/mnt/c/Users/Peter.lukka/Desktop/DEV/llama.cpp/` (master, commit 08f2145). Use this as source for extracting reference C code for benchmarks and verifying pipeline behavior. Key paths:
- Attention: `ggml/src/ggml-cpu/ops.cpp` (`ggml_compute_forward_flash_attn_ext`)
- Q4K dot: `ggml/src/ggml-cpu/arch/arm/quants.c` (`ggml_vec_dot_q4_K_q8_K`)
- RMSNorm: `ggml/src/ggml-cpu/ops.cpp` (`ggml_compute_forward_rms_norm`)
- RoPE: `ggml/src/ggml-cpu/ops.cpp` (`ggml_compute_forward_rope`)
- Q8K quant: `ggml/src/ggml-cpu/quants.c` (`quantize_row_q8_K`)

## Principle

Every step in the pipeline must match llama.cpp exactly. No extra passes, no security features (TurboQuant, JL-rotation), no Olorin-specific attention kernels. Dead code gets deleted — not commented, not feature-flagged.

Hard rules apply: no file > 500 lines, no fake functions, no silent fallbacks, delete don't comment.

**Olorin is Eä's showcase.** Every SIMD operation must be an Eä kernel compiled through eacompute. No Rust scalar fallbacks where SIMD is possible. This includes f32→f16 conversion, attention dot products, V summation, and SiLU. Subagents must not simplify kernel code to scalar Rust.

## Pipeline (per layer, decode)

Identical to llama.cpp `llama_decode_internal` → `llm_build_llama`:

```
embed_lookup
for layer in 0..n_layers:
    1. rmsnorm(x, attn_norm)           — kernel: rmsnorm_f32
    2. quant(x_norm) → q8k             — kernel: quant_f32_q8k
    3. Q = matmul(wq, q8k)             — kernel: q4k_dot_q8k
    4. K = matmul(wk, q8k)             — kernel: q4k_dot_q8k
    5. V = matmul(wv, q8k)             — kernel: q4k_dot_q8k
    6. add_bias(Q, q_bias)             — Rust scalar
    7. add_bias(K, k_bias)             — Rust scalar (BEFORE cache now)
    8. add_bias(V, v_bias)             — Rust scalar
    9. rope(Q), rope(K)                — kernel: rope
   10. kv_store(K → cache_f16)         — NEW: f32→f16 store
   11. kv_store(V → cache_f16)         — NEW: f32→f16 store
   12. scores = Q · K_cache^T / √hd    — NEW kernel: attn_dot_f16
   13. softmax(scores)                 — Eä kernel: softmax_f32
   14. attn_out = scores · V_cache     — NEW kernel: attn_vsum_f16
   15. quant(attn_out) → q8k           — kernel: quant_f32_q8k
   16. x += matmul(wo, q8k)            — kernel: q4k_dot_q8k + residual
   17. rmsnorm(x, ffn_norm)            — kernel: rmsnorm_f32
   18. quant(x_norm) → q8k             — kernel: quant_f32_q8k
   19. gate = matmul(w_gate, q8k)      — kernel: q4k_dot_q8k
   20. up = matmul(w_up, q8k)          — kernel: q4k_dot_q8k
   21. hidden = silu(gate) * up         — Eä kernel: silu_mul
   22. quant(hidden) → q8k             — kernel: quant_f32_q8k
   23. x += matmul(w_down, q8k)        — kernel: q4k_dot_q8k + residual
final rmsnorm + output_proj
```

## 1. New: F16KvCache

Replaces `EakvCache` entirely. Same file: `src/inference/cache.rs`.

```rust
pub struct F16KvCache {
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
    /// Layout: [layer][kv_idx][head][seq * head_dim] as f16 (u16)
    data: Vec<u16>,
}
```

**API:**
- `new(n_layers, n_kv_heads, head_dim, max_seq_len) → F16KvCache`
- `store_k(layer, data_f32, n_tokens)` — convert f32→f16 and store
- `store_v(layer, data_f32, n_tokens)` — same
- `k_head_ptr(layer, head) → *const u16` — pointer to K[head][0..seq_len*hd]
- `v_head_ptr(layer, head) → *const u16` — pointer to V[head][0..seq_len*hd]
- `advance(n)`, `clear()`, `seq_len()`, `checkpoint()`, `restore()`

**Memory layout per layer:**
- K: `n_kv_heads * max_seq_len * head_dim` u16 values
- V: same
- Total per layer: `2 * n_kv_heads * max_seq_len * head_dim * 2` bytes
- Llama 3.2 3B (28 layers, 8 KV heads, 128 dim, 2048 seq): 2 * 8 * 2048 * 128 * 2 * 28 = 117MB

**f32→f16 conversion:** Eä kernel on both ARM (`fcvtn`) and x86 (`vcvtps2ph`). No Rust scalar — Olorin showcases Eä.

**No:** TurboRotate, quantize_simd, jl_signs, fwht, sign_flip. None of it.

## 2. New: f16 Attention Kernels

Two new Eä kernels, one for each arch (ARM NEON + x86 AVX2/SSE):

### attn_dot_f16

```
fn attn_dot_f16(
    query: *const f32,      // [head_dim] f32
    k_cache: *const u16,    // [seq_len * head_dim] f16
    scores_out: *mut f32,   // [seq_len] f32
    seq_len: i32,
    head_dim: i32,
)
```

Computes: `scores[t] = (1/√hd) * Σ_d query[d] * f16_to_f32(k_cache[t*hd + d])`

ARM: `fcvtl` to widen f16→f32, `fmla` for dot product.
x86: `vcvtph2ps` (F16C) to widen, `vfmadd231ps` for FMA.

### attn_vsum_f16

```
fn attn_vsum_f16(
    weights: *const f32,    // [seq_len] f32 (softmax'd)
    v_cache: *const u16,    // [seq_len * head_dim] f16
    out: *mut f32,          // [head_dim] f32
    seq_len: i32,
    head_dim: i32,
)
```

Computes: `out[d] = Σ_t weights[t] * f16_to_f32(v_cache[t*hd + d])`

Same SIMD strategy as attn_dot_f16.

### C benchmarks

`benchmarks/attn_f16_bench/bench.c` — tests both kernels:
- Generate random f32 query, random f16 K/V cache
- Scalar reference implementation
- Correctness check (rel error < 1e-3, f16 loses some precision)
- Benchmark ns/call at realistic sizes (seq_len=512, head_dim=128)

`benchmarks/attn_f16_bench/llama_ref.c` — extracted llama.cpp attention for comparison timing.

## 3. Code to DELETE

### Eä kernel files (delete entirely)

- `turbo_rotate.ea` + `.ea.json` (+ olorin/ variant)
- `quantize_simd.ea` + `_arm.ea` + `.ea.json` variants (+ eakv/ variants)
- `fused_k_score.ea` + `_arm` + `_64` + `_gqa` + all .ea.json (8 .ea files, 16+ .ea.json)
- `fused_v_sum.ea` + `_arm` + `_64` + all .ea.json (4 .ea files, 12+ .ea.json)
- `flash_decode_attn.ea` + `_arm` + `.ea.json`
- `fused_causal_attn_gqa_arm.ea` + `.ea.json`
- `fused_k_score_causal_gqa.ea.json` + `_arm.ea.json`

### Rust code (delete)

**src/kernels/ffi.rs:**
- Type aliases: `SignFlipFn`, `FwhtFn`, `TurboRotateFn`
- KernelTable fields: `sign_flip`, `fwht_inplace`, `turbo_rotate`
- Library loading: `jl_project_lib`, `turbo_rotate_lib`
- Symbol loading for these
- Public wrappers: `sign_flip()`, `fwht_inplace()`, `turbo_rotate()`

**src/kernels/ffi_inference.rs:**
- Type aliases: `QuantizeSIMDFn`, `DequantizeSIMDFn`, `KScoreMhaFn`, `KScoreGqaFn`, `VSumMhaFn`, `VSumGqaFn`, `FusedCausalAttnFn`, `FlashDecodeAttnFn`
- KernelTableInference fields: `quantize_simd`, `dequantize_simd`, all `fused_k_score*`, `fused_v_sum*`, `fused_causal_attn_gqa`, `flash_decode_attn`
- Library loading: quantize_lib, k_score*, v_sum*, causal_attn_lib, flash_decode_lib
- Public wrappers: `quantize_simd()`, `dequantize_simd()`, all `fused_k_score*()`, `fused_v_sum*()`, `fused_attention()`, `has_fused_causal_attn()`, `has_flash_decode_attn()`, `fused_causal_attn_gqa()`, `flash_decode_attn()`

**src/inference/cache.rs:**
- Delete entire file contents, replace with F16KvCache
- Delete `pub mod attention` (attention_scores, attention_output)
- Delete `gen_jl_signs()`, `rotate_groups()`, `load_raw()`, `append()` with quantize_simd
- Delete all TurboQuant accessors: `weights()`, `scales()`, `biases()`, `k_ptrs()`, `v_ptrs()`, `groups_per_head()`, `jl_signs()`

**src/inference/forward_llama.rs:**
- Delete flash_decode_attn path (lines 188-216)
- Delete 3-pass TurboQuant fallback attention (lines 217-248)
- Replace with f16 attention: per-head attn_dot_f16 → softmax → attn_vsum_f16
- Move K-bias to BEFORE cache store (line 173-175 → before append)

**src/inference/prefill_llama.rs:**
- Delete fused_causal_attn_gqa path (lines 171-206)
- Delete 3-pass TurboQuant fallback attention (lines 207-238)
- Replace with f16 attention loop
- Delete KV transpose (head-major for TurboQuant append, lines 149-163)
- K-bias before cache store

**src/inference/forward.rs (BitNet):**
- Update attention calls from cache::attention::* to new f16 attention
- Or: keep BitNet path separate if it doesn't use Q4K pipeline

**tests/cache.rs:**
- Rewrite for F16KvCache

**tests/test_flash_decode.rs:**
- Remove `has_flash_decode_attn()` reference

## 4. C Benchmarks

One benchmark per kernel in the pipeline. Each follows the q4k_dot_bench pattern:

| Benchmark dir | Kernel tested | llama_ref.c |
|---|---|---|
| `benchmarks/rmsnorm_bench/` | `rmsnorm_f32` | llama.cpp `ggml_compute_forward_rms_norm` |
| `benchmarks/quant_q8k_bench/` | `quant_f32_q8k` | llama.cpp `quantize_row_q8_K` |
| `benchmarks/q4k_dot_bench/` | `q4k_dot_q8k` | ✓ already exists |
| `benchmarks/rope_bench/` | `rope` | llama.cpp `ggml_rope` |
| `benchmarks/attn_f16_bench/` | `attn_dot_f16` + `attn_vsum_f16` | llama.cpp attention loop |
| `benchmarks/silu_bench/` | `silu_mul` | scalar reference |
| `benchmarks/softmax_bench/` | `softmax_f32` | scalar reference |
| `benchmarks/f16_convert_bench/` | `f32_to_f16` + `f16_to_f32` | scalar reference |

Each benchmark:
- `bench.c` — test harness with deterministic RNG, correctness check, timing
- `llama_ref.c` — extracted llama.cpp reference implementation
- `build.sh` — ARM build (armv8.2-a+dotprod), links via dlopen

## 5. What stays unchanged

- Q4K dot product kernel and all variants (q4k_dot_q8k, 4row, dual) — proven at parity
- RMSNorm, Q8K quantization, RoPE kernels — already identical to llama.cpp
- Fused gate+up SiLU dispatch — optimization, doesn't affect correctness comparison
- Q6K matmul path (for mixed quant models)
- Embedding lookup (Q4K/Q6K/f16)
- ThreadPool, GEMM tiling, work-stealing — orchestration, not kernel
- BitNet I2S forward pass (separate model type, not part of this work)

## 6. Verification

After implementation:
1. All C benchmarks pass (PASS: YES, rel error < 1e-3 for f16, < 1e-4 for everything else)
2. `cargo build` succeeds with no warnings about dead code
3. End-to-end test: same prompt → same output token sequence (within sampling randomness)
4. Benchmark on Pi 5: decode ms/tok and prefill tok/s compared to llama.cpp
5. No kernel files remain that aren't used in the active pipeline
