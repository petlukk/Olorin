# llama.cpp ARM NEON — Component Reference

Exact documentation of how llama.cpp implements each compute component on ARM NEON (Pi 5 / Cortex-A76).
This is our ground truth. Olorin Eä kernels must produce identical results.

Source: llama.cpp HEAD as of 2026-04-10, `ggml/src/ggml-cpu/`.

## 1. Q8K Quantization

**File:** `ggml-quants.c:2692` — `quantize_row_q8_K_ref` (scalar, ARM has no NEON variant)

**Layout:** `block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; }`

**Algorithm per 256-element block:**
1. Find `max` (signed) and `amax` (absolute) across 256 elements
2. `iscale = -127.0f / max`
3. `qs[j] = MIN(127, nearest_int(iscale * x[j]))` for j in 0..255
4. `bsums[g] = sum(qs[g*16 .. g*16+15])` for g in 0..15
5. `d = 1.0f / iscale`

**nearest_int** (magic-number round-to-even):
```c
float val = fval + 12582912.0f;
int i; memcpy(&i, &val, sizeof(int));
return (i & 0x007fffff) - 0x00400000;
```

### Olorin implementation

**File:** `kernels/q4k_quant_arm.ea` — SIMD via f32x4 NEON (llama is scalar here)

**Differences from llama:**
- **Sign convention:** Olorin uses `d = amax / 127.0` (always positive), llama uses `d = 1/(-127/max)` (sign of max). qs signs are flipped accordingly. Product `d * qs` is identical.
- **d precision:** 1 ULP difference from `amax/127` vs `1/(-127/max)` (two divisions vs one). Max reconstituted error: 1.9e-6.
- **Rounding:** Both use magic-number round-to-even (12582912 trick). Bit-exact qs magnitudes.

### Verification result (step8_q8k_quant_real_embedding)

Tested on real BOS embedding (1536 floats, 6 blocks):
- **qs:** 0 / 1536 magnitude mismatches ✓
- **bsums:** 0 / 96 magnitude mismatches ✓
- **d:** 1 ULP difference in 1/6 blocks (f32 precision limit) ✓
- **Reconstituted max error:** 1.9e-6 ✓

**Status: VERIFIED** — bit-exact magnitudes, negligible d precision difference.

---

## 2. Q4K Dot Product (NEON)

**File:** `arch/arm/quants.c:2709` — `ggml_vec_dot_q4_K_q8_K` `#elif defined __ARM_NEON`

**Layout:** `block_q4_K { ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qs[128]; }`
- 256 elements stored as 128 bytes of packed 4-bit nibbles
- 12 bytes encode 8 scales + 8 mins (6-bit each, bit-shuffled)

**Algorithm per block:**
1. `d = y.d * f16_to_f32(x.d)`, `dmin = y.d * f16_to_f32(x.dmin)`
2. **Mins correction:**
   - `q8sums = vpaddq_s16(bsums[0..7], bsums[8..15])` — pairwise add to 8 values
   - Extract 8 mins from `x.scales[12]` via utmp bit-shuffle (kmask1=0x3f3f3f3f, kmask2=0x0f0f0f0f, kmask3=0x03030303)
   - `mins8 = {utmp[1] & kmask1, ((utmp[2]>>4) & kmask2) | (((utmp[1]>>6) & kmask3) << 4)}`
   - `vmovl_u8(mins8)` → i16, `vmull_s16` × 2 halves, `vaddvq_s32` → scalar `sumi_mins`
3. **Scales extraction:**
   - `utmp[1] = (utmp[2] & kmask2) | (((utmp[0]>>6) & kmask3) << 4)`
   - `utmp[0] &= kmask1`
   - `scales = (uint8_t*)utmp` — 8 scale bytes
4. **Dot product (4 groups × 64 elements):**
   - Load 32 bytes: `ggml_vld1q_u8_x2(q4)` → 2×16 bytes
   - Lo nibbles: `vand(q4, 0xf)` → `ggml_vdotq_s32` with q8 → `vaddvq_s32 * scales[2*j+0]` → `sumi1`
   - Hi nibbles: `vshrq_n_u8(q4, 4)` → `ggml_vdotq_s32` with q8 → `vaddvq_s32 * scales[2*j+1]` → `sumi2`
5. `sumf += d * (sumi1 + sumi2)` per block, `sumf -= dmin * sumi_mins` per block

**Note:** `ggml_vdotq_s32` maps to SDOT on dotprod-capable cores (Cortex-A76 has this).

### Olorin implementation

**File:** `kernels/q4k_dot_arm.ea` — NEON SDOT via `vdot_i32`, unrolled 4 groups

**Differences from llama:**
- **Accumulation:** Olorin uses single `sumi` variable (interleaved lo/hi groups). llama uses separate `sumi1`/`sumi2`. Mathematically identical — integer addition is commutative.
- **Mins correction:** Olorin uses scalar extraction (8 multiply-adds). llama uses `vmovl_u8` + `vmull_s16`. Same result.
- **Hi nibble:** Olorin does `(pa >> shift4) & mask_lo` (redundant mask). llama does `vshrq_n_u8(q4, 4)`. Identical result.
- **Sign convention:** Olorin Q8K has positive d/qs. Dot product and mins correction cancel out — produces identical final value.

### Verification result (step9_q4k_dot_vs_llama_ref)

Tested on real model weights (layer 0 gate, Q4K) with real BOS embedding input, 8 rows:
- **Max absolute error:** 7.5e-6 ✓
- **Max relative error:** 1.0e-5 ✓
- Errors are f32 accumulation order differences (SIMD reduce vs scalar loop)

**Status: VERIFIED** — matches llama scalar reference within f32 precision.

---

## 3. Q5K Dot Product (NEON)

**File:** `arch/arm/quants.c:2804`

**Layout:** `block_q5_K { ggml_half d; ggml_half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; }`
- 256 elements: 4-bit base in qs[128] + 1 high bit in qh[32] = 5-bit total

**Algorithm per block:**
1. Same scales/mins extraction as Q4K (utmp bit-shuffle)
2. **Mins:** `vld1_u8(utmp+8)` → `vmovl_u8` → `vmull_s16` with paired q8sums → scalar `sumi_mins`
3. **High bits:** Load 32 bytes `qh`, process 2 bits per iteration:
   - `q5h[0] = (qhbits & 1) << 4`, `q5h[2] = (qhbits & 2) << 3`
   - `qhbits >>= 2` after each group
4. **Reconstruct 5-bit:** `q5bytes = (qs & 0xf) | q5h` (lo), `(qs >> 4) | q5h` (hi)
5. `ggml_vdotq_s32` per 2×16 elements × scale, accumulate
6. `sumf += d * sumi - dmin * sumi_mins`

### Olorin implementation

**File:** `kernels/q5k_dot_arm.ea` — NEON SDOT, unrolled 4 groups, explicit bit shifts

**Differences from llama:**
- **High-bit extraction:** llama uses `(qh & 1)<<4` / `(qh & 2)<<3` + `qhbits>>=2` per iteration. Olorin uses explicit `(qh >> N) & 1) << 4` per group. Same result.
- Same scales/mins/accumulation structure as Q4K.

### Verification result (step10_q5k_dot_vs_llama_ref)

Tested on real Wk weights (Q5K, layer 0), 8 rows:
- **Max absolute error:** 7.6e-6 ✓
- **Max relative error:** 1.0e-5 ✓

**Status: VERIFIED** — matches llama scalar reference within f32 precision.

---

## 4. Q6K Dot Product (NEON)

**File:** `arch/arm/quants.c:3412`

**Layout:** `block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; ggml_half d; }`
- 256 elements: 4-bit lo in ql[128] + 2-bit hi in qh[64] = 6-bit total
- 16 × int8_t scales (NOT 6-bit packed like Q4K)

**Algorithm per block:**
1. `d_all = f16_to_f32(x.d)`
2. **Mins correction:**
   - `scales` → `vmovl_s8` → `vmull_s16` with `q8.bsums` → `vaddvq_s32` → `isum_mins`
3. **2 iterations × 128 elements each:**
   - Load 64 bytes `ql` + 32 bytes `qh`
   - First half: `(ql & 0xf) | ((qh & 3) << 4)` — 4 chunks × 16 elements
   - Second half: `(ql >> 4) | ((qh >> N & 3) << 4)` — shift qh by 4,6 per iteration
   - `vdotq_s32` per 16-byte chunk × `scale[i]`, accumulate `isum`
4. `sum += d_all * y.d * (isum - 32 * isum_mins)`

**Note:** The `- 32 * isum_mins` bias correction is Q6K-specific (values stored unsigned 0..63, bias=32).

### Olorin implementation

**File:** `kernels/q6k_dot_arm.ea` — NEON SDOT, 2 iterations × 4 groups, inline bias correction

**Differences from llama:**
- **Bias correction:** Olorin subtracts `32 * bsums[i]` per sub-group inline. llama pre-computes `isum_mins = Σ(scales[i] * bsums[i])` and subtracts `32 * isum_mins` at the end. Mathematically identical (distributive property).
- **d pre-computation:** Olorin takes `d_arr[blk] = d_q6k * q8_d` pre-computed by Rust wrapper. llama computes inline. Same value.

### Verification result (step11_q6k_dot_vs_llama_ref)

Tested on real Wq weights (Q6K, layer 0), 8 rows:
- **Max absolute error:** 1.9e-6 ✓
- **Max relative error:** 1.1e-7 ✓

**Status: VERIFIED** — matches llama scalar reference within f32 precision.

---

## 5. RMSNorm

**File:** `ops.cpp:3713` — `ggml_compute_forward_rms_norm_f32`

**Algorithm:**
1. `sum = Σ(x[i]²)` accumulated in **double precision** (`ggml_float` = double)
2. `mean = sum / n`
3. `scale = 1.0f / sqrtf(mean + eps)`
4. `y[i] = x[i] * scale`

**IMPORTANT:** Weight multiplication is a SEPARATE `mul` op in the compute graph, NOT fused into rmsnorm.
The separate mul does: `y[i] = y[i] * (1.0 + weight[i])` for Gemma4 (weight+1 convention).

### Olorin implementation

**File:** `kernels/gemma4_rmsnorm.ea` — f32x4 SIMD, dual accumulators, fused weight multiply

**Differences from llama:**
- **Precision:** Olorin uses f32 SIMD for sum-of-squares. llama uses double. At n=1536, no measurable difference.
- **Weight fusion:** Olorin does `x * scale * weight` in one pass. llama does `rms_norm(x)` then separate `mul(weight)`. Mathematically identical.
- **Weight convention:** GGUF stores Gemma4 norm weights as-is (already includes +1). Both llama and Olorin use them directly.

### Verification result (step12_rmsnorm_vs_llama_ref)

Tested on BOS embedding (1536 floats), layer 0 attn_norm:
- **Max absolute error:** 0.0 (bit-exact!) ✓
- **L2:** 452.893280 vs 452.893280 ✓

**Status: VERIFIED** — bit-exact match with llama double-precision reference.

---

## 6. RoPE

**File:** `ops.cpp:5729` — `ggml_compute_forward_rope_flt`

**Frequency computation (`ggml_rope_cache_init`):**
1. `theta_scale = powf(freq_base, -2.0f / n_dims)`
2. Per dimension pair i0 = 0,2,4,...:
   - `theta = theta_base` (= position), multiplied by `theta_scale` each step
   - If `freq_factors`: `theta_effective = theta / freq_factors[i0/2]`
   - `cos[i0] = cosf(theta_effective)`, `sin[i0] = sinf(theta_effective)`

**Rotation (Gemma4 uses NEOX mode):**
- `rotate_pairs(n_dims, n_dims/2, cache, src, dst)`:
  - `dst[i] = src[i] * cos - src[i + n_dims/2] * sin`
  - `dst[i + n_dims/2] = src[i] * sin + src[i + n_dims/2] * cos`
- Dimensions beyond `n_dims` are passed through unchanged

**Note:** Gemma4 has two RoPE configs: SWA layers (theta=10000, dim=256) and global layers (theta=1000000, dim=512, with freq_factors for proportional scaling).

### Olorin implementation

**File:** `kernels/gemma4_rope.ea` — **SIMD** (f32x4, 4 dimension pairs per iteration)

**Differences from llama:**
- **Table computation:** Olorin uses `powf(theta, 2*d/n_rot)` per dimension. llama accumulates `theta *= theta_scale`. Tiny f32 drift at high dimensions (max 4.8e-7).
- **Rotation:** Identical NEOX-mode scalar: `re*cos - im*sin`, `re*sin + im*cos`.

### Verification result (step13_rope_vs_llama_ref)

Tested at pos=5, SWA (dim=256, theta=10000) and global (dim=512, theta=1000000):
- **Cos/sin table max error:** 4.8e-7 (SWA), 4.2e-7 (global) ✓
- **RoPE output max error:** 1.9e-5 ✓

**Status: VERIFIED** — llama scalar, Olorin SIMD. Matches within f32 precision.

---

## 7. GELU

**File:** `vec.h:986` — `ggml_gelu_f32`

```c
gelu(x) = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)))
```

Where:
- `GELU_COEF_A = 0.044715f`
- `SQRT_2_OVER_PI = 0.79788456080286535587989211986876f`

**No SIMD specialization for ARM.** Scalar per element. Applied as part of GeGLU: `output = gelu(gate) * up`.

### Olorin implementation

**File:** `kernels/gemma4_gelu.ea` — **SIMD** (f32x4 with `exp` intrinsic + manual tanh)

**Differences from llama:**
- **SIMD vs scalar:** Olorin uses f32x4 NEON SIMD. llama uses scalar `tanhf()` on ARM.
- **tanh implementation:** Olorin computes `1 - 2/(exp(2x)+1)`. llama calls libc `tanhf`. Same math, tiny precision difference.
- **Fused:** Olorin does `gelu(gate) * up` in one kernel. llama does separate gelu then mul ops.

### Verification result (step14_gelu_vs_llama_ref)

Tested on 256 synthetic gate/up values:
- **Max absolute error:** 2.9e-7 ✓
- **Max relative error:** 6.1e-5 ✓

**Status: VERIFIED** — Olorin SIMD matches llama scalar within f32 precision.

---

## 8. Softmax (in attention)

**File:** `vec.cpp:593` — NEON path

**Algorithm:**
1. Find `max` across all scores
2. NEON loop (4 elements at a time):
   - `val = exp(x[i] - max)` via `ggml_v_expf(vsubq_f32(...))`
   - Store to output, accumulate `sum` in **double** (`ggml_float`)
3. Scalar tail for remaining elements
4. `y[i] /= sum` (via `ggml_vec_scale_f32(n, y, 1.0/sum)`)

**In flash-attention context:** scale is applied BEFORE softmax (as `s = s * scale`), not after.

## 9. Softcap (in attention)

**File:** `ops.cpp:8298` — inside flash attention

**Algorithm:**
1. Pre-compute: `scale /= logit_softcap` (folds softcap into scale)
2. Per KQ score: `s = s * scale` then `s = logit_softcap * tanhf(s)`
3. Effectively: `s = softcap * tanh(kq_dot / softcap)`
4. Applied AFTER scale, BEFORE softmax

**Gemma4:** `logit_softcap = 30.0`, `f_attention_scale = 1.0`.
So: `s = 30.0 * tanh(kq_dot * (1.0/30.0))` = `30 * tanh(kq/30)`.

---

### Olorin Softmax implementation

**File:** `kernels/softmax_arm.ea` — **SIMD** (f32x4 `exp` intrinsic)

**Differences from llama:**
- **Sum precision:** Olorin uses f32 sum. llama casts to double at the end. At small n (attention lengths), no measurable difference.
- **Fused scale:** Olorin applies attention scale in pass 1. llama does separate `ggml_vec_scale_f32` then softmax. Same result.

### Olorin Softcap implementation

**File:** `kernels/softcap_arm.ea` — **SIMD** (f32x4 `exp`-based tanh)

**Differences from llama:**
- llama: scalar `ggml_scale(1/cap) → ggml_tanh → ggml_scale(cap)` (3 graph ops, scalar tanh on ARM)
- Olorin: fused `cap * tanh(x / cap)` in one SIMD pass

**IMPORTANT:** Gemma4 uses `final_logit_softcapping` on output logits only. No KQ-score softcap in attention (unlike Gemma2).

### Verification result (step15_softmax_softcap_vs_llama_ref)

- **Softmax n=16:** bit-exact (0.0 error) ✓
- **Softmax n=128:** bit-exact (0.0 error) ✓
- **Softcap (cap=30):** max abs 3.8e-6, max rel 2.0e-7 ✓
- **Softcap range:** all values in (-30, +30) ✓

**Status: BOTH VERIFIED** — matches llama reference within f32 precision.

---

## Attention Flow (complete)

For decode (single token, nrc=1):

1. Q·K dot: `score = dot(Q_head, K_cached_pos)` (f32 dot with f16→f32 conversion of cached K)
2. Scale: `score *= attention_scale` (1.0 for Gemma4)
3. Softcap: `score = 30 * tanh(score / 30)`
4. Softmax across all cached positions (with double-precision sum)
5. V weighted sum: `out = Σ(softmax_score * V_cached_pos)` (f16→f32 conversion of cached V)

**KV cache stores f16.** Conversion to f32 happens at attention time.

---

## Cross-platform Verification (2026-04-10)

All 9 components verified on both x86 (WSL/SSE2) and ARM (Pi 5/NEON).
ARM softmax tail handling was fixed to match x86 (no overlap-load).

### Results: x86 vs ARM produce identical error magnitudes

| # | Component | llama ARM | Olorin | x86 max err | ARM max err |
|---|-----------|-----------|--------|-------------|-------------|
| 1 | Q8K quant | Scalar | **SIMD** | 1 ULP d | 1 ULP d |
| 2 | Q4K dot | SIMD (sdot) | **SIMD** (vdot) | 7.5e-6 | 7.5e-6 |
| 3 | Q5K dot | SIMD (sdot) | **SIMD** (vdot) | 7.6e-6 | 7.6e-6 |
| 4 | Q6K dot | SIMD (sdot) | **SIMD** (vdot) | 1.9e-6 | 1.9e-6 |
| 5 | RMSNorm | Scalar (f64) | **SIMD** (fma) | 0.0 | 0.0 |
| 6 | RoPE | Scalar | **SIMD** (f32x4) | 1.9e-5 | 1.9e-5 |
| 7 | GELU | Scalar (tanhf) | **SIMD** (exp) | 2.9e-7 | 2.9e-7 |
| 8 | Softmax | SIMD (exp) | **SIMD** (exp) | 0.0 | 0.0 |
| 9 | Softcap | Scalar (tanhf) | **SIMD** (exp) | 3.8e-6 | 3.8e-6 |

**Olorin: 9/9 SIMD.** llama.cpp ARM: 4/9 SIMD (Q4K/Q5K/Q6K dot + softmax).

All errors are within f32 precision. No logic differences between x86 and ARM kernels.

### Test suite

17 tests in `tests/gemma4_verify.rs` (steps 0-15):
- Steps 0-6: original forward-pass verification (embedding, rmsnorm, QKV, layer, PLE, logits, 2-token)
- Step 7: Q8K quant synthetic half-boundary test
- Step 8: Q8K quant real BOS embedding (1536 floats, 6 blocks)
- Step 9: Q4K dot vs llama scalar reference (gate weights, 8 rows)
- Step 10: Q5K dot vs llama scalar reference (Wk weights, 8 rows)
- Step 11: Q6K dot vs llama scalar reference (Wq weights, 8 rows)
- Step 12: RMSNorm vs llama double-precision reference
- Step 13: RoPE cos/sin tables + rotation (SWA + global layers)
- Step 14: GELU fused kernel vs llama scalar tanhf
- Step 15: Softmax + Softcap vs llama reference

### Fix applied

`kernels/softmax_arm.ea`: Removed overlap-load tail handling, replaced with scalar tail matching x86 exactly. Previous version re-computed last 4 elements via SIMD when n%4 != 0.
