# Gemma 4 E2B — Text-only Inference for Olorin

## Goal

Replace all existing inference code with a clean Gemma 4 E2B (2.3B effective) forward pass.
Text-only, Q4K quantized, Pi 5 (ARM NEON) + x86 (AVX2). Every compute operation as an Ea SIMD kernel.
Layer-by-layer verified against llama.cpp. No scalar fallbacks. No creativity until output matches.

Future: E2B becomes draft model for E4B (4.5B) speculative decode.

## Gemma 4 E2B Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 26 |
| Hidden dim | 2048 |
| Head dim | 256 |
| Q heads | 8 |
| KV heads | 4 (GQA 2:1) |
| FFN dim | 8192 |
| Vocab | 262144 |
| Sliding window | 512 tokens |
| Global:local ratio | 4:1 (every 5th layer is global) |
| Context | 128k |
| PLE dim | Read from GGUF (lower than hidden) |
| Shared KV layers | Read from GGUF (last N layers reuse KV) |
| RoPE theta | Read from GGUF |
| Activation | GeGLU (GELU-gated FFN) |

All architecture parameters read from GGUF metadata. Nothing hardcoded.

## What Gets Deleted

- `src/inference/forward.rs` (BitNet forward pass)
- `src/inference/forward_llama.rs` (Llama forward pass)
- `src/inference/prefill.rs` (BitNet prefill)
- `src/inference/prefill_llama.rs` (Llama prefill)
- `src/inference/speculative.rs` (rebuild later with E4B)
- `src/inference/gemm_i2s.rs` (BitNet GEMM)
- `src/inference/gemm_q4k.rs` (old Q4K GEMM)
- `src/inference/gemm_q6k.rs` (old Q6K GEMM)
- `src/inference/matmul_q4k.rs` (old wrappers)
- `src/inference/matmul_q6k.rs` (old wrappers)
- `src/inference/math.rs` (unused helpers)
- `src/inference/ptr.rs` (unused pointer helpers)
- All `bitnet_*.ea` kernels (14 files)
- `rope.ea`, `rope_arm.ea` (replaced by gemma4_rope.ea)
- `silu_mul.ea` (replaced by gemma4_gelu.ea)
- `attn_f16.ea`, `attn_f16_arm.ea` (replaced by clean attention)
- `q4k_fused_gemm.ea`, `q4k_gemm_tile.ea` and ARM variants

## What Gets Kept

- `src/inference/gguf.rs` — stripped to Gemma 4 tensors only
- `src/inference/tokenizer.rs` — BPE, works with Gemma 4
- `src/inference/threadpool.rs` — unchanged
- `src/inference/generate.rs` — simplified for single model type
- `src/kernels/ffi.rs` — KernelTable infrastructure (core/safety/search untouched)
- `src/kernels/ffi_inference.rs` — rewired for Gemma 4 kernels only
- `build.rs` — auto-discover/compile (unchanged, just discovers new .ea files)
- All non-inference code (vault, safety, router, tools, UI)

## What Gets Kept (Ea kernels, verified correct)

- `q4k_dot.ea` + `q4k_dot_arm.ea` — Q4K dot product (proven 1.05x vs llama.cpp)
- `q4k_quant.ea` + `q4k_quant_arm.ea` — Q8K input quantization
- `q6k_dot.ea` + `q6k_dot_arm.ea` — Q6K dot product
- `f16_convert.ea` + `f16_convert_arm.ea` — f32<->f16
- `softmax.ea` — re-verify with Gemma 4 scales

## New Ea Kernels

| Kernel | Purpose | Platforms |
|--------|---------|-----------|
| `gemma4_rope.ea` | Dual RoPE: standard (sliding) + proportional (global) | x86 + ARM |
| `gemma4_gelu.ea` | Fused GELU activation for GeGLU FFN | x86 + ARM |
| `gemma4_rmsnorm.ea` | RMSNorm with Gemma weight+1 convention | x86 + ARM |

## Forward Pass — Step by Step

Matches llama.cpp exactly. No deviations.

```
1. token -> embedding lookup (vocab x hidden, Q6K dequant)
2. + PLE signal (per-layer embedding table + learned projection)

Per layer (0..25):
  3. RMSNorm(x)  — weight+1 convention
  4. Wq, Wk, Wv matmul (Q4K dot product)
  5. RoPE:
     - Sliding layer: standard RoPE (base theta)
     - Global layer: proportional RoPE (limited to high-freq dims)
  6. KV cache:
     - Normal layer: store K,V as f16
     - Shared KV layer: reuse K,V pointer from source layer
  7. Attention:
     - Sliding layer: mask to 512-token window (ring buffer)
     - Global layer: full context
  8. Wo matmul (Q4K)
  9. Residual add + PLE modulation
  10. RMSNorm(x)
  11. FFN: gate = Wgate*x, up = Wup*x, out = Wdown*(GELU(gate) * up)
  12. Residual add

13. Final RMSNorm
14. Output matmul -> logits (vocab x hidden)
15. Sample (temperature, top-k, top-p, min-p)
```

## KV Cache Design

```
KvCache:
  k: Vec<Vec<u16>>   -- [layer][head * seq_pos * head_dim] as f16 bits
  v: Vec<Vec<u16>>

  Sliding window layers:
    - Ring buffer: position = seq_pos % 512
    - Overwrites oldest entry
    - Attention mask handles wrap-around

  Global layers:
    - Grows up to max context length
    - Standard sequential layout

  Shared layers:
    - No own storage
    - Points to source layer's KV data
    - Source is last non-shared layer of same attention type

  Store: f32 -> f16 via f16_convert kernel
  Load: f16 -> f32 via f16_convert kernel
```

## GGUF Tensor Names (Gemma 4)

```
token_embd.weight              — Embedding (vocab x hidden, Q6K)
output.weight                  — Output projection (vocab x hidden)
output_norm.weight             — Final RMSNorm

blk.{N}.attn_norm.weight       — Attention pre-norm
blk.{N}.attn_q.weight          — Query projection
blk.{N}.attn_k.weight          — Key projection
blk.{N}.attn_v.weight          — Value projection
blk.{N}.attn_output.weight     — Output projection (Wo)

blk.{N}.ffn_norm.weight        — FFN pre-norm
blk.{N}.ffn_gate.weight        — Gate projection (GeGLU)
blk.{N}.ffn_up.weight          — Up projection
blk.{N}.ffn_down.weight        — Down projection

PLE tensors:                   — Names TBD from GGUF inspection
```

## Verification Plan

Each step built and verified against llama.cpp before the next.
Test harness: C program with llama.cpp eval callback dumping per-tensor L2 norms.
Already built on Pi from previous sessions (test_layers.c).

| Step | Component | Metric | Tolerance |
|------|-----------|--------|-----------|
| 1 | GGUF load + tensor names | All tensors found, correct dims | exact |
| 2 | Embedding (Q6K dequant) | L2 norm | +/- 1e-5 |
| 3 | RMSNorm | L2 norm | +/- 1e-6 |
| 4 | QKV matmul (Q4K) | L2 norm | +/- 1e-4 |
| 5 | RoPE (both variants) | L2 norm | +/- 1e-6 |
| 6 | KV cache store/load roundtrip | L2 norm | +/- 1e-3 (f16) |
| 7 | Attention scores (QK) | L2 norm | +/- 1e-4 |
| 8 | Softmax | L2 norm | +/- 1e-5 |
| 9 | Attention output (V sum) | L2 norm | +/- 1e-4 |
| 10 | FFN (GeGLU) | L2 norm | +/- 1e-4 |
| 11 | PLE modulation | L2 norm | +/- 1e-5 |
| 12 | Full layer 0 output | L2 norm | +/- 1e-3 |
| 13 | Full forward (all layers) | Same top-5 tokens | exact |

## File Structure After Cleanup

```
src/inference/
  engine.rs        — Gemma4Model: GGUF load, tensor lookup, metadata
  forward.rs       — Gemma 4 forward pass (decode, single token)
  cache.rs         — KvCache: sliding window + shared layers
  matmul.rs        — Q4K/Q6K matmul wrappers (slim)
  gguf.rs          — GGUF format parser (Gemma 4 tensors only)
  tokenizer.rs     — BPE tokenizer (unchanged)
  generate.rs      — Public API (one model type)
  threadpool.rs    — Thread pool (unchanged)
  mod.rs           — Module declarations

src/kernels/
  ffi.rs           — KernelTable (core/safety/search — unchanged)
  ffi_inference.rs — Inference FFI (stripped to Gemma 4 kernels)

kernels/
  # Kept (verified correct)
  q4k_dot.ea / q4k_dot_arm.ea
  q4k_quant.ea / q4k_quant_arm.ea
  q6k_dot.ea / q6k_dot_arm.ea
  f16_convert.ea / f16_convert_arm.ea
  softmax.ea

  # New
  gemma4_rope.ea               — Dual RoPE (x86 + ARM via #[cfg])
  gemma4_gelu.ea               — Fused GELU*up (x86 + ARM via #[cfg])
  gemma4_rmsnorm.ea            — RMSNorm with weight+1 (x86 + ARM via #[cfg])

  # Non-inference (kept, unchanged)
  chacha20.ea, chacha20_search*.ea, fused_safety.ea, sanitizer.ea,
  validate.ea, search.ea, search_avx512.ea, zeroize.ea,
  byte_classifier.ea, ansi_parser.ea, terminal_diff.ea,
  command_router.ea, intent_router.ea, leak_scanner.ea,
  pretokenize.ea, expr_eval.ea, jl_project.ea
```

## Implementation Order

Bottom-up, kernel-first. Each step verified before the next.

1. Clean branch — delete old inference code, keep infra
2. GGUF parser — strip to Gemma 4, load E2B model
3. Embedding — Q6K dequant, verify L2 vs llama.cpp
4. RMSNorm kernel — write gemma4_rmsnorm.ea, verify
5. Q4K matmul — reuse existing kernels, verify QKV projections
6. RoPE kernel — write gemma4_rope.ea (dual mode), verify
7. KV cache — sliding window ring buffer + shared layers
8. Attention — verify scores + output
9. FFN — write gemma4_gelu.ea, verify GeGLU output
10. PLE — verify modulation
11. Full forward pass — wire everything, verify per-layer
12. Generate loop — token sampling, end-to-end test
13. Cross-platform — verify on Pi 5 + x86
