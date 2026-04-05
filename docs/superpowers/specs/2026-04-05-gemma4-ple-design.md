# Gemma 4 PLE (Per-Layer Embeddings) Design

## Context

Gemma 4 E2B uses PLE — a gated bottleneck adapter applied after each transformer layer. Without PLE, the forward pass produces garbage output. All PLE weights are already loaded from GGUF; the computation is skipped (Task 13).

## Reference

llama.cpp `src/models/gemma4-iswa.cpp` — `project_per_layer_inputs()` and per-layer PLE block.

## Dimensions

- `ple_dim` = 256 (from GGUF metadata)
- `hidden_dim` = 1536
- `n_layers` = 35
- `ple_signal` total = 256 × 35 = 8960 elements

## Phase A: `prepare_ple()` — once per token, before layer loop

Called from `forward_one()` after embedding + scale, before the layer loop.

Input: `scaled_embedding[hidden_dim]`, `token_id`

1. **Q6K dequant lookup** `ple_token_embd[token_id]` → `raw_ple[ple_dim * n_layers]` (8960), scale × √ple_dim
2. **BF16 matvec** `ple_model_proj @ scaled_embedding` → `proj_ple[ple_dim * n_layers]` (8960), scale × 1/√hidden_dim
3. **RMSNorm** each `[ple_dim]` slice of `proj_ple` with `ple_proj_norm` weights (Gemma4 norm, no +1)
4. **Add** `ple_signal[i] = proj_ple[i] + raw_ple[i]`, scale × 1/√2

Result stored in `self.ple_signal[0..8960]`.

### BF16 matvec kernel

New kernel `bf16_matvec.ea`. `ple_model_proj` is stored as BF16 in GGUF, shape `[ple_dim * n_layers, hidden_dim]` = `[8960, 1536]`.

Signature: `bf16_matvec(weight: *const u16, input: *const f32, output: *mut f32, n_rows: i32, n_cols: i32)`

Per row: dot product of BF16 weight row with f32 input. BF16→f32 conversion is `(u16 as u32) << 16` reinterpreted as f32.

## Phase B: per-layer PLE — in `layer_forward()`, after FFN+residual, before out_scale

Input: `x[hidden_dim]` (current hidden state), `ple_signal[il * ple_dim .. (il+1) * ple_dim]`

1. **quant_input** `x` → Q8K
2. **q4k_matvec** `inp_gate @ x` → `ple_gate[ple_dim]` (down-projection, 256 output)
3. **gelu_mul** `gelu(ple_gate) * ple_signal_slice` → `ple_gate` (fused GELU + element-wise gate)
4. **quant_input** `ple_gate[ple_dim]` → Q8K (small: 1 block of 256)
5. **q4k_matvec** `proj @ ple_gate` → `ple_out[hidden_dim]` (up-projection, 1536 output)
6. **gemma4_rmsnorm** `ple_out` with `post_norm` weight → `ple_out`
7. **vector add** `x[i] += ple_out[i]`

All per-layer tensors (`inp_gate`, `proj`) are Q4K. Reuses existing `q4k_matvec` and `quant_input`.

## New buffers in Gemma4State

| Buffer | Size | Purpose |
|--------|------|---------|
| `ple_signal` | `ple_dim * n_layers` (8960) | Phase A output, reused every token |
| `ple_gate` | `ple_dim` (256) | Phase B gate scratch |
| `ple_out` | `hidden_dim` (1536) | Phase B up-projection scratch |
| `ple_q8_qs` | `ple_dim + 12` (268) | Q8K for ple_dim input |
| `ple_q8_d` | 1 | Q8K scale (1 block) |
| `ple_q8_bsums` | 16 | Q8K bsums (1 block) |

## New kernel

`bf16_matvec.ea` — BF16 × f32 dot product, 4-row batched.

## Files modified

| File | Change |
|------|--------|
| `kernels/bf16_matvec.ea` | New kernel |
| `src/kernels/ffi_inference.rs` | FFI wrapper for bf16_matvec |
| `src/inference/matmul.rs` | `bf16_matvec()` wrapper |
| `src/inference/forward.rs` | Add PLE buffers to Gemma4State, call `prepare_ple()` |
| `src/inference/forward_attn.rs` | Implement Phase B in `layer_forward()` |
| `src/inference/dequant.rs` | Q6K dequant for PLE token embeddings (ple_dim * n_layers per row) |
| `tests/gemma4_verify.rs` | Step 4: PLE verification test |

## Verification

Step 4 test: compute PLE Phase A for BOS token, compare `ple_signal` L2 and first4 against llama.cpp. Then run single layer with PLE and compare post-PLE L2.
