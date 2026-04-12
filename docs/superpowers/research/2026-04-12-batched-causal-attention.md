# Batched Causal Attention — Kernel Contract for Fused Ea Kernel

**Date:** 2026-04-12
**Phase:** Phase 2 Plan 2, Task 1
**Purpose:** Document exactly how llama.cpp computes batched causal attention so the Ea kernel matches.

---

## 1. llama.cpp Attention Flow (non-flash path)

Source: `llama.cpp/src/llama-graph.cpp`, `build_attn_mha` function (~line 1849).

### 1.1 Tensor Layout After Permute

Before the Q*K^T multiply, all three tensors are permuted to
`[head_dim, n_kv, n_heads, n_streams]`:

```cpp
q = ggml_permute(ctx0, q, 0, 2, 1, 3);  // [hd, n_tokens, n_heads, n_stream]
k = ggml_permute(ctx0, k, 0, 2, 1, 3);  // [hd, n_kv, n_heads, n_stream]
v = ggml_permute(ctx0, v, 0, 2, 1, 3);  // [hd, n_kv, n_heads, n_stream]
```

### 1.2 Q * K^T (Score Matrix)

```cpp
ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
```

Shape of `kq`: `[n_kv, n_tokens, n_heads, n_stream]`

This is the full score matrix: `kq[i][j]` = dot(q[i, h], k[j, h]) for query
token `i` and key position `j`. Computed in F32 precision explicitly.

Gemma 4 uses no `attn_soft_cap` and no ALiBi, so no additional scaling or
clamping is applied before softmax.

### 1.3 Causal Mask Shape and Application

Mask tensor shape: `[n_kv, n_tokens/n_stream, 1, n_stream]` (F32).

The mask is initialised to `-INFINITY` for every element, then selectively set
to `0.0f` for positions the query token is allowed to attend to:

```cpp
std::fill(data, data + ggml_nelements(mask), -INFINITY);
// then for each (query_token_i, kv_position_j):
//   if j belongs to same sequence AND j.pos <= i.pos AND SWA OK:
//       data[i * n_kv + j] = 0.0f   (attend)
//   else:
//       data[i * n_kv + j] = -INFINITY  (masked)
```

Source: `llama-kv-cache.cpp`, `set_input_kq_mask_impl` (~line 1433).

The key rule for causal masking (line 1539-1543):
```cpp
if (causal) {
    if (p0 > p1) {   // cache cell position > query token position
        goto skip;    // => -INFINITY
    }
}
```

The mask is **additive**: it is added to the raw scores before softmax. So
masked positions become `-INFINITY + score ≈ -INFINITY` and vanish in softmax.

Applied via `ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f)`.

### 1.4 Softmax

Source: `ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_soft_max_f32` (~line 5232).

Per-row (per query token, per head):
```
wp[j] = kq[i][j] * scale
wp[j] += mask[i][j]          // additive: 0.0 or -INFINITY
max    = max(wp)
dp[j]  = exp(wp[j] - max)
dp    /= sum(dp)
```

`kq_scale` for Gemma 4: **1.0f** (no `1/sqrt(head_dim)` — Gemma uses unit
scale, verified by the forward pass using `attn_scale = 1.0f32`).

No ALiBi (Gemma 4 uses RoPE), so `max_bias = 0.0f` and `slope = 1.0f`.

### 1.5 V Accumulation (kqv)

```cpp
ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);  // kq is now softmax scores
```

Shape: `[head_dim, n_tokens, n_heads, n_stream]`

This is the weighted sum: for each query token `i` and each head `h`:
```
out[i][h] = sum_j( softmax_scores[i][j] * v[j][h] )
```

Result is then permuted back and reshaped to `[n_heads * head_dim, n_tokens]`.

---

## 2. Olorin's Current Single-Token Attention

Source: `src/inference/forward_graph.rs`, lines 265-310.

Olorin currently processes one query token at a time. For each head `h`:

```rust
let kv_h = h / gqa_ratio;        // GQA: 8 Q heads → 1 KV head
let attn_scale = 1.0f32;

// K/V cache layout: stride = n_kv_heads * head_dim
// Position p: k_ptr[p * stride_kv + kv_h * head_dim .. +head_dim]

for p in 0..attn_len {
    let k_offset = p * stride_kv + kv_h * head_dim;
    f16_to_f32(k_ptr + k_offset, scratch, head_dim);
    scores[p] = f32_dot(q_slice, scratch, head_dim);
}

softmax_f32(scores, attn_len, attn_scale);   // scale = 1.0, in-place

// zero output, then accumulate
for p in 0..attn_len {
    let v_offset = p * stride_kv + kv_h * head_dim;
    f16_to_f32(v_ptr + v_offset, scratch, head_dim);
    f32_dot_acc(out + h*head_dim, scratch, scores[p], head_dim);
}
```

Key facts confirmed:
- K and V are stored as **f16** in the cache, converted to f32 per block before use.
- The scratch buffer is per-thread (`kv_scratch_stride` elements), re-used for K then V.
- `attn_scale = 1.0f32` — matches llama.cpp for Gemma 4.
- GQA ratio is 8: all 8 Q heads read from KV head 0.
- `attn_len = cache.attn_len(il)` = total positions currently in cache for this layer (i.e., `seq_len_before + 1` after the current token is stored).
- No causal mask needed for single-token inference: all stored positions are in the past.

---

## 3. Batched Attention Kernel Contract

For a prefill of **N query tokens** into a layer whose cache already holds
`seq_start` positions (positions `0 .. seq_start`), the post-store cache
holds `n_kv = seq_start + N` positions.

### 3.1 Causal Masking Rule

Query token `i` (0-indexed, position `seq_start + i`) may attend to cache
positions `j` where `j <= seq_start + i`:

```
attend(i, j) = (j <= seq_start + i)
```

For the SWA layers, additionally require:
```
attend_swa(i, j) = attend(i, j) AND (pos(j) > pos(i) - n_swa)
                 = attend(i, j) AND (j > seq_start + i - 512)   // n_swa=512
```

For the global layers: no window restriction, full causal only.

In the kernel, mask is built inline (no separate mask buffer). Positions that
fail the attend condition get `-INFINITY` added before softmax. Equivalently,
the inner loop over `j` runs only up to `seq_start + i + 1` and (for SWA)
starts from `max(0, seq_start + i + 1 - n_swa)`.

### 3.2 Score Computation

For query token `i`, KV head `kv_h = h / gqa_ratio`:

```
scores[i][j] = dot(Q[i*hd .. (i+1)*hd],
                   K_cache[j*stride_kv + kv_h*hd .. +hd]) * attn_scale
```

Where:
- `hd = head_dim` (256 for SWA layers, 512 for global layers)
- `stride_kv = n_kv_heads * hd = 1 * hd = hd` (GQA 8:1, one KV head)
- `attn_scale = 1.0f` (Gemma 4, always)
- K values in cache are **f16**, must be converted to f32 for the dot product.

### 3.3 Softmax

Row-wise over the `j` dimension, applied after the scores for token `i` are
fully computed:

```
softmax(scores[i][0..n_kv_i])   where n_kv_i = seq_start + i + 1
```

For SWA: only the window positions participate; out-of-window positions were
never computed (loop range exclusion), so their score slots are unused/garbage
and must not be included in the softmax denominator. The kernel loops exactly
over the valid window for normalization.

Scale = 1.0f, standard numerically-stable (subtract row max before exp).

### 3.4 Output Accumulation

For query token `i`, head `h`:

```
out[i*hd .. (i+1)*hd] = sum_{j in attended} ( softmax_scores[i][j]
                           * V_cache[j*stride_kv + kv_h*hd .. +hd] )
```

V values are f16 in cache, converted to f32 per position before accumulation.
Output is f32, shape `[N, n_heads * head_dim]` (heads packed, N rows).

### 3.5 Scores Buffer Sizing

Per query token, per head: at most:
- **SWA layers:** 512 f32 scores (sliding window width = 512)
- **Global layers:** `seq_start + N` f32 scores (full sequence)

For a fused kernel call over all N tokens, the scores buffer may be reused
row-by-row (process one query token fully before moving to the next), keeping
the working set at one row: `max(512, seq_start + N)` floats.

Alternatively, allocate `N * max_attend` for a fully materialised score
matrix, but for large prefills this is wasteful. The row-by-row approach
matches the existing single-token pattern and keeps cache pressure low.

### 3.6 F16→F32 Conversion

All K and V reads from the cache require f16→f32 conversion. In the existing
code this is done via the `f16_to_f32` Ea kernel (`ffi_inference::f16_to_f32`).
The fused batched kernel must do the same — no raw f16 dot products.

### 3.7 Olorin-Specific Constants

| Parameter       | Value         | Notes                              |
|-----------------|---------------|------------------------------------|
| `n_heads`       | 8             | Q heads                            |
| `n_kv_heads`    | 1             | KV heads (GQA 8:1)                 |
| `gqa_ratio`     | 8             | `n_heads / n_kv_heads`             |
| `head_dim` (SWA)| 256           | SWA layers                         |
| `head_dim` (GL) | 512           | Global layers                      |
| `attn_scale`    | 1.0f          | Gemma 4, always                    |
| `n_swa`         | 512           | Sliding window size                |
| `stride_kv`     | `hd`          | = `n_kv_heads * head_dim`          |

---

## 4. Key Differences: Single-Token vs. Batched

| Aspect              | Single-token (current)            | Batched (fused kernel)            |
|---------------------|-----------------------------------|-----------------------------------|
| Causal mask needed  | No — all cache positions are past | Yes — token `i` masks `j > seq_start + i` |
| Score buffer size   | `attn_len` floats                 | `min(n_swa, seq_start+i+1)` per row |
| Outer loop          | Called once per token in generate | Called once per layer for all N tokens |
| Q input             | Single vector `[head_dim]`        | Matrix `[N, head_dim]`            |
| Output              | Single vector `[n_heads * hd]`    | Matrix `[N, n_heads * hd]`        |
| K/V conv scratch    | `head_dim` f32 per thread         | Same — one row at a time          |

---

## 5. Implementation Notes for Ea Kernel (Tasks 2–3)

1. **Loop structure:** outer over query tokens `i`, then over heads `h`, then
   over cache positions `j in [j_start, j_end)`. This keeps the K/V scratch
   hot for consecutive queries on the same head.

2. **SWA window:** `j_start = max(0, seq_start + i + 1 - n_swa)`,
   `j_end = seq_start + i + 1`. For global layers: `j_start = 0`.

3. **No separate mask buffer.** The attend condition is computed inline.
   Positions outside the window are simply not accumulated.

4. **Score reuse.** For SWA, scores are up to 512 f32; for global up to
   `seq_start + N`. Pre-allocate a scratch slice of `seq_start + N` floats per
   thread and reuse across all `i` and `h` iterations.

5. **Output buffer.** Shape `[N * n_heads * head_dim]` f32, zeroed before use
   (same as the existing `write_bytes(out_base, 0, head_dim)` pattern).

6. **Thread split.** Split heads across threads (same as current single-token
   code): each thread owns `heads_per_thread = n_heads / nth` heads and
   processes all N query tokens for those heads.

7. **Correctness check.** With `N=1` and `seq_start=S`, the fused kernel must
   produce bit-identical output to the current `forward_one_graph` single-token
   path (Task 10 gate).
