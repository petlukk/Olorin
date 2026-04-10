# llama.cpp Gemma4 Forward Pass — Exact Orchestration

Source: `src/models/gemma4-iswa.cpp` + `src/llama-graph.cpp`, HEAD 2026-04-10.

## Pre-loop

```
inpL = embed(token)
inpL = inpL * sqrt(n_embd)                           // embed scaling
inp_per_layer = PLE_prepare(inpL, token)              // per-layer embedding (Phase A)
```

### PLE Phase A (project_per_layer_inputs)
```
per_layer_tok = get_rows(per_layer_tok_embd, token)   // Q6K dequant, shape [ple_dim * n_layer]
per_layer_tok = per_layer_tok * sqrt(ple_dim)          // scale

per_layer_proj = matmul(per_layer_model_proj, inpL)   // BF16 matvec, [ple_dim*n_layer, n_embd]
per_layer_proj = per_layer_proj * (1/sqrt(n_embd))    // scale

per_layer_proj = rms_norm(per_layer_proj, ple_proj_norm)  // per [ple_dim] slice

inp_per_layer = (per_layer_tok + per_layer_proj) * (1/sqrt(2))
```

## Per-layer loop (il = 0..34)

### 1. Pre-attention RMSNorm
```
cur = rms_norm(inpL) * attn_norm_weight               // fused norm+weight
```

### 2. Q projection (ALWAYS, even for shared-KV layers)
```
Qcur = matmul(wq, cur)                                // Q4K or Q6K matvec
Qcur = reshape_3d(Qcur, head_dim, n_heads, 1)
Qcur = rms_norm(Qcur) * q_norm_weight                 // per-head Q norm
Qcur = rope(Qcur, pos, freq_base, freq_factors)       // NEOX mode
```

### 3. K/V projection + norm + RoPE (only if has_kv)
```
if has_kv(il):
    Kcur = matmul(wk, cur)                             // Q5K matvec
    Vcur = matmul(wv, cur)                             // Q6K matvec

    Kcur = reshape_3d(Kcur, head_dim, n_kv_heads, 1)
    Vcur = reshape_3d(Vcur, head_dim, n_kv_heads, 1)

    Kcur = rms_norm(Kcur) * k_norm_weight              // per-head K norm
    Vcur = rms_norm(Vcur)                              // BARE norm (no weight!)

    Kcur = rope(Kcur, pos, freq_base, freq_factors)    // NEOX mode

    // Store to KV cache (converts to f16)
    cache_k[il][pos] = Kcur
    cache_v[il][pos] = Vcur
```

### 4. Attention (build_attn_mha → flash_attn or standard)
```
Q = Qcur                                              // [head_dim, n_heads, 1]
K = cache_k[il][0..seq_len]                            // from cache (f16)
V = cache_v[il][0..seq_len]                            // from cache (f16)

// For shared-KV layers: K, V come from earlier layer's cache
// (no new K/V computed, Q still uses cur from this layer's norm)

kqv_out = attention(Q, K, V, scale=1.0, mask)
// Internally: Q·K^T (with f16→f32 for K), softmax, V·scores
```

### 5. Wo + post-attention norm + residual
```
cur = matmul(wo, kqv_out)                              // Q4K/Q6K matvec

cur = rms_norm(cur) * post_attn_norm_weight            // post-attention RMSNorm
attn_out = cur + inpL                                  // residual with ORIGINAL input
```

### 6. FFN (GeGLU)
```
cur = rms_norm(attn_out) * ffn_norm_weight             // pre-FFN norm on attn_out

// GeGLU: gate and up computed in PARALLEL from same input (LLM_FFN_PAR)
gate = matmul(w_gate, cur)                             // Q4K matvec
up   = matmul(w_up, cur)                               // Q4K matvec
cur  = geglu(gate, up)                                 // = gelu(gate) * up (fused)

cur = matmul(w_down, cur)                              // Q6K/Q4K matvec (down projection)
```

### 7. Post-FFN norm + residual
```
cur = rms_norm(cur) * post_ffn_norm_weight
cur = cur + attn_out                                   // residual with attn_out (NOT inpL!)
```

### 8. PLE (Per-Layer Embedding)
```
if has_ple:
    pe_in = cur                                        // save for residual

    gate = matmul(per_layer_inp_gate, cur)             // down-project [n_embd → ple_dim]
    gate = gelu(gate)                                  // NOTE: plain gelu, NOT geglu
    gate = gate * inp_per_layer[il]                    // element-wise with pre-computed PLE signal

    cur = matmul(per_layer_proj, gate)                 // up-project [ple_dim → n_embd]
    cur = rms_norm(cur) * per_layer_post_norm_weight

    cur = pe_in + cur                                  // residual
```

### 9. Layer output scale
```
cur = cur * layer_output_scale[il]                     // scalar scale per layer
inpL = cur                                             // feed to next layer
```

## Post-loop

```
cur = rms_norm(inpL) * output_norm_weight              // final norm
logits = matmul(embed_weight, cur)                     // output projection (tied weights)
logits = softcap(logits, 30.0)                         // final_logit_softcapping
```

## Key Orchestration Details

### Residual connections
1. **Post-attention:** `attn_out = post_attn_norm(wo_out) + inpL` — adds to ORIGINAL layer input
2. **Post-FFN:** `cur = post_ffn_norm(ffn_out) + attn_out` — adds to post-attention output (NOT inpL)
3. **Post-PLE:** `cur = pe_in + ple_out` — adds to post-FFN output

### Shared KV layers
- Some layers (kv_shared_source[il].is_some()) do NOT compute K/V
- They still compute Q from their own attn_norm(inpL)
- They reuse K/V from an earlier layer's cache
- Attention still runs with the shared K/V

### V normalization
- **BARE RMSNorm** — no weight multiplication, just normalize
- llama.cpp: `ggml_rms_norm(Vcur, eps)` (no weight tensor passed)

### PLE gating
- llama uses `ggml_gelu(gate)` — plain GELU, not GeGLU
- Then element-wise multiply with PLE signal: `ggml_mul(gelu_gate, inp_this_layer)`
- Olorin uses `gelu_mul(gate, signal)` — fused, should be equivalent

### FFN gating
- llama uses `ggml_geglu_split(gate, up)` with `LLM_FFN_PAR` — parallel gate+up from same input
- This is `gelu(gate) * up` fused
- Olorin calls `gelu_mul(gate, up)` — same operation

### KV cache format
- K and V stored as f16 in cache
- Converted back to f32 at attention time
- Both SWA and global layers use same cache format
