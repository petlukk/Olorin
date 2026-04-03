# llama.cpp 1:1 Pipeline Parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **HARD RULES (apply to ALL agents):**
> - No file exceeds 500 lines. Split before you hit the limit.
> - No fake functions. No silent fallbacks.
> - Delete, don't comment. Dead code gets removed.
> - Olorin is Eä's showcase — every SIMD op must be an Eä kernel. Do NOT simplify kernel code to scalar Rust.
> - llama.cpp reference: `/mnt/c/Users/Peter.lukka/Desktop/DEV/llama.cpp/` (master, 08f2145)
> - eacompute compiler: `/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release/ea`
> - Build: `PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build`

**Goal:** Replace TurboQuant KV cache with f16 KV cache and make decode/prefill pipeline 1:1 with llama.cpp for benchmarking parity.

**Architecture:** Delete all TurboQuant/JL-rotation code and kernels. Write F16KvCache with simple f16 storage. Write new Eä kernels for f16 attention (attn_dot_f16, attn_vsum_f16) and utility ops (silu_mul, softmax_f32, f32↔f16). C benchmark per kernel. Rewire forward_llama.rs and prefill_llama.rs to use f16 attention.

**Tech Stack:** Rust, Eä (eacompute), C benchmarks, ARM NEON + x86 AVX2/SSE SIMD

---

### Task 1: Delete TurboQuant kernel files

**Files:**
- Delete: all `kernels/turbo_rotate*`, `kernels/quantize_simd*`, `kernels/fused_k_score*`, `kernels/fused_v_sum*`, `kernels/flash_decode_attn*`, `kernels/fused_causal_attn*`, `kernels/fused_attention*`, `kernels/dequantize_*` (both `.ea` and `.ea.json`, including `_arm`, `_64`, `_gqa` variants and `eakv/`, `olorin/` subdirs)

- [ ] **Step 1: Delete all TurboQuant-related kernel files**

```bash
# turbo_rotate (core ffi.rs)
rm -f kernels/turbo_rotate.ea kernels/turbo_rotate.ea.json
rm -f kernels/olorin/turbo_rotate.ea.json

# quantize_simd (TurboQuant Q4 packing)
rm -f kernels/quantize_simd.ea kernels/quantize_simd_arm.ea
rm -f kernels/quantize_simd.ea.json kernels/quantize_simd_arm.ea.json
rm -f kernels/eakv/quantize_simd.ea.json kernels/eakv/quantize_simd_arm.ea.json

# dequantize (TurboQuant Q4 unpacking)
rm -f kernels/dequantize_simd.ea kernels/dequantize_avx2.ea kernels/dequantize_avx512.ea
rm -f kernels/dequantize_simd.ea.json kernels/dequantize_avx2.ea.json kernels/dequantize_avx512.ea.json
rm -f kernels/dequantize_simd_arm.ea kernels/dequantize_simd_arm.ea.json

# fused_k_score (all variants)
rm -f kernels/fused_k_score.ea kernels/fused_k_score_arm.ea
rm -f kernels/fused_k_score_64.ea kernels/fused_k_score_64_arm.ea
rm -f kernels/fused_k_score_gqa.ea kernels/fused_k_score_gqa_arm.ea
rm -f kernels/fused_k_score_gqa_64.ea kernels/fused_k_score_gqa_64_arm.ea
rm -f kernels/fused_k_score.ea.json kernels/fused_k_score_arm.ea.json
rm -f kernels/fused_k_score_64.ea.json kernels/fused_k_score_64_arm.ea.json
rm -f kernels/fused_k_score_gqa.ea.json kernels/fused_k_score_gqa_arm.ea.json
rm -f kernels/fused_k_score_gqa_64.ea.json kernels/fused_k_score_gqa_64_arm.ea.json
rm -f kernels/fused_k_score_causal_gqa.ea.json kernels/fused_k_score_causal_gqa_arm.ea.json
rm -f kernels/eakv/fused_k_score*.ea.json

# fused_v_sum (all variants)
rm -f kernels/fused_v_sum.ea kernels/fused_v_sum_arm.ea
rm -f kernels/fused_v_sum_64.ea kernels/fused_v_sum_64_arm.ea
rm -f kernels/fused_v_sum.ea.json kernels/fused_v_sum_arm.ea.json
rm -f kernels/fused_v_sum_64.ea.json kernels/fused_v_sum_64_arm.ea.json
rm -f kernels/eakv/fused_v_sum*.ea.json
rm -f kernels/fused_v_sum_gqa.ea kernels/fused_v_sum_gqa_arm.ea
rm -f kernels/fused_v_sum_gqa.ea.json kernels/fused_v_sum_gqa_arm.ea.json
rm -f kernels/fused_v_sum_gqa_64.ea kernels/fused_v_sum_gqa_64_arm.ea
rm -f kernels/fused_v_sum_gqa_64.ea.json kernels/fused_v_sum_gqa_64_arm.ea.json

# flash_decode_attn
rm -f kernels/flash_decode_attn.ea kernels/flash_decode_attn_arm.ea
rm -f kernels/flash_decode_attn.ea.json kernels/flash_decode_attn_arm.ea.json

# fused_causal_attn
rm -f kernels/fused_causal_attn_gqa_arm.ea kernels/fused_causal_attn_gqa_arm.ea.json

# fused_attention (the old TurboQuant fused attention)
rm -f kernels/fused_attention.ea kernels/fused_attention_arm.ea
rm -f kernels/fused_attention.ea.json kernels/fused_attention_arm.ea.json
```

- [ ] **Step 2: Verify no stale kernel files remain**

```bash
# Should return nothing related to turbo/quantize_simd/fused_k/fused_v/flash_decode/fused_causal/fused_attention/dequantize
ls kernels/ | grep -E 'turbo|quantize_simd|fused_k_score|fused_v_sum|flash_decode|fused_causal|fused_attention|dequantize'
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add -A kernels/
git commit -m "delete: TurboQuant kernel files — turbo_rotate, quantize_simd, fused_k/v_score, flash_decode, dequantize"
```

---

### Task 2: Strip TurboQuant FFI from ffi.rs

**Files:**
- Modify: `src/kernels/ffi.rs`

- [ ] **Step 1: Remove turbo_rotate types, fields, loading, and wrappers from ffi.rs**

Remove these type aliases (lines 37-39):
```rust
type SignFlipFn         = unsafe extern "C" fn(*mut f32, *const f32, i32);
type FwhtFn             = unsafe extern "C" fn(*mut f32, i32);
type TurboRotateFn      = unsafe extern "C" fn(*mut f32, *const f32, i32);
```

Remove these KernelTable fields (lines 82-84):
```rust
    pub sign_flip:                SignFlipFn,
    pub fwht_inplace:             FwhtFn,
    pub turbo_rotate:             TurboRotateFn,
```

Remove library loading (lines 174-175):
```rust
    let jl_project_lib  = load("jl_project")?;
    let turbo_rotate_lib = load("turbo_rotate")?;
```

Replace with just:
```rust
    let jl_project_lib  = load("jl_project")?;
```

Remove symbol loading (lines 239-244):
```rust
            sign_flip: std::mem::transmute(
                sym(&turbo_rotate_lib, b"sign_flip\0")?),
            fwht_inplace: std::mem::transmute(
                sym(&turbo_rotate_lib, b"fwht_inplace\0")?),
            turbo_rotate: std::mem::transmute(
                sym(&turbo_rotate_lib, b"turbo_rotate\0")?),
```

Remove `turbo_rotate_lib` from libs vec (line 258-261):
```rust
            libs: vec![
                byte_classifier, leak_scanner, sanitizer, command_router,
                fused_safety, intent_router, expr_eval,
                zeroize_lib, search, jl_project_lib, turbo_rotate_lib,
                chacha20_lib, chacha20_sv2, pretokenize_lib,
                ansi_parser_lib, terminal_diff_lib,
            ],
```
Change to:
```rust
            libs: vec![
                byte_classifier, leak_scanner, sanitizer, command_router,
                fused_safety, intent_router, expr_eval,
                zeroize_lib, search, jl_project_lib,
                chacha20_lib, chacha20_sv2, pretokenize_lib,
                ansi_parser_lib, terminal_diff_lib,
            ],
```

Remove public wrappers (lines 369-379):
```rust
pub unsafe fn sign_flip(vec: *mut f32, signs: *const f32, dim: i32) {
    (k().sign_flip)(vec, signs, dim);
}

pub unsafe fn fwht_inplace(vec: *mut f32, dim: i32) {
    (k().fwht_inplace)(vec, dim);
}

pub unsafe fn turbo_rotate(vec: *mut f32, signs: *const f32, dim: i32) {
    (k().turbo_rotate)(vec, signs, dim);
}
```

- [ ] **Step 2: Verify ffi.rs compiles clean**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo check 2>&1 | head -30
```

Expected: errors only about call sites (forward_llama.rs, cache.rs etc.) — not about ffi.rs itself. Those get fixed in later tasks.

- [ ] **Step 3: Commit**

```bash
git add src/kernels/ffi.rs
git commit -m "delete: turbo_rotate/sign_flip/fwht FFI from KernelTable"
```

---

### Task 3: Strip TurboQuant FFI from ffi_inference.rs and ffi_inference_types.rs

**Files:**
- Modify: `src/kernels/ffi_inference.rs`
- Modify: `src/kernels/ffi_inference_types.rs`

- [ ] **Step 1: Remove dead types from ffi_inference_types.rs**

Delete these type aliases (lines 62-85):
```rust
pub type QuantizeSIMDFn   = unsafe extern "C" fn(*const f32, *mut i32, *mut f32, *mut f32, i32);
pub type DequantizeSIMDFn = unsafe extern "C" fn(*const u8, *const f32, *const f32, *mut f32, i32);
pub type KScoreMhaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32);
pub type KScoreGqaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32, i32);
pub type VSumMhaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32);
pub type VSumGqaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32, i32);
pub type FusedAttentionFn = unsafe extern "C" fn(
    *const f32,
    *const u8, *const f32, *const f32,
    *const u8, *const f32, *const f32,
    *mut f32, i32, i32, i32);
pub type FusedCausalAttnFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32,
    *const u8, *const f32, *const f32,
    *mut f32, *mut f32, i32, i32, i32, i32, i32);
pub type FlashDecodeAttnFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32,
    *const u8, *const f32, *const f32,
    *mut f32, *mut f32,
    i32, i32, i32, i32);
```

Add new f16 attention type aliases at the bottom:
```rust
pub type AttnDotF16Fn = unsafe extern "C" fn(
    *const f32, *const u16, *mut f32, i32, i32);
pub type AttnVsumF16Fn = unsafe extern "C" fn(
    *const f32, *const u16, *mut f32, i32, i32);
pub type F32ToF16Fn = unsafe extern "C" fn(*const f32, *mut u16, i32);
pub type F16ToF32Fn = unsafe extern "C" fn(*const u16, *mut f32, i32);
pub type SoftmaxF32Fn = unsafe extern "C" fn(*mut f32, i32, f32);
pub type SiluMulFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32);
```

- [ ] **Step 2: Strip TurboQuant fields and loading from ffi_inference.rs**

Remove from `KernelTableInference` struct (lines 46-58):
```rust
    pub quantize_simd:         QuantizeSIMDFn,
    pub dequantize_simd:       DequantizeSIMDFn,
    pub fused_k_score:         KScoreMhaFn,
    pub fused_k_score_64:      KScoreMhaFn,
    pub fused_k_score_gqa:     KScoreGqaFn,
    pub fused_k_score_gqa_64:  KScoreGqaFn,
    pub fused_v_sum:           VSumMhaFn,
    pub fused_v_sum_64:        VSumMhaFn,
    pub fused_v_sum_gqa:       VSumGqaFn,
    pub fused_v_sum_gqa_64:    VSumGqaFn,
    pub fused_attention:       FusedAttentionFn,
    pub fused_causal_attn_gqa: Option<FusedCausalAttnFn>,
    pub flash_decode_attn:     Option<FlashDecodeAttnFn>,
```

Replace with new f16 attention fields:
```rust
    pub attn_dot_f16:          AttnDotF16Fn,
    pub attn_vsum_f16:         AttnVsumF16Fn,
    pub f32_to_f16:            F32ToF16Fn,
    pub f16_to_f32:            F16ToF32Fn,
    pub softmax_f32:           SoftmaxF32Fn,
    pub silu_mul_f32:          SiluMulFn,
```

Remove library loading (lines 124-134):
```rust
    let quantize_lib      = load("quantize_simd")?;
    let k_score_lib       = load("fused_k_score")?;
    let k_score_64_lib    = load("fused_k_score_64")?;
    let k_score_gqa_lib   = load("fused_k_score_gqa")?;
    let k_score_gqa64_lib = load("fused_k_score_gqa_64")?;
    let v_sum_lib         = load("fused_v_sum")?;
    let v_sum_64_lib      = load("fused_v_sum_64")?;
    let fused_attn_lib    = load("fused_attention")?;
    let causal_attn_lib   = load("fused_causal_attn_gqa").ok();
    let flash_decode_lib  = load("flash_decode_attn").ok();
```

Replace with:
```rust
    let attn_f16_lib      = load("attn_f16")?;
    let f16_conv_lib      = load("f16_convert")?;
    let softmax_lib       = load("softmax")?;
    let silu_lib          = load("silu_mul")?;
```

Remove dequantize library loading (lines 136-158, the entire `#[cfg]` block for deq_lib/deq_sym).

Remove old symbol loading (lines 190-205) and replace with:
```rust
            attn_dot_f16:   std::mem::transmute(sym(&attn_f16_lib, b"attn_dot_f16\0")?),
            attn_vsum_f16:  std::mem::transmute(sym(&attn_f16_lib, b"attn_vsum_f16\0")?),
            f32_to_f16:     std::mem::transmute(sym(&f16_conv_lib, b"f32_to_f16\0")?),
            f16_to_f32:     std::mem::transmute(sym(&f16_conv_lib, b"f16_to_f32\0")?),
            softmax_f32:    std::mem::transmute(sym(&softmax_lib, b"softmax_f32\0")?),
            silu_mul_f32:   std::mem::transmute(sym(&silu_lib, b"silu_mul_f32\0")?),
```

Update libs vec — remove all old TurboQuant libs, add new ones:
```rust
            libs: {
                let mut v = vec![
                    i2s, quant, rms, attn, i8d, act, vadd,
                    q4kq, q4kd, q4kfg, q6kd, rope, gemm_tile,
                    attn_f16_lib, f16_conv_lib, softmax_lib, silu_lib,
                    validate_lib,
                ];
                v
            },
```

- [ ] **Step 3: Replace public wrappers (lines 390-461)**

Delete all wrappers from `quantize_simd` through `flash_decode_attn` (lines 390-461). Replace with:

```rust
// Public wrappers — f16 attention

pub unsafe fn attn_dot_f16(
    query: *const f32, k_cache: *const u16, scores_out: *mut f32,
    seq_len: i32, head_dim: i32,
) { (k().attn_dot_f16)(query, k_cache, scores_out, seq_len, head_dim) }

pub unsafe fn attn_vsum_f16(
    weights: *const f32, v_cache: *const u16, out: *mut f32,
    seq_len: i32, head_dim: i32,
) { (k().attn_vsum_f16)(weights, v_cache, out, seq_len, head_dim) }

pub unsafe fn f32_to_f16(src: *const f32, dst: *mut u16, n: i32) {
    (k().f32_to_f16)(src, dst, n)
}

pub unsafe fn f16_to_f32(src: *const u16, dst: *mut f32, n: i32) {
    (k().f16_to_f32)(src, dst, n)
}

pub unsafe fn softmax_f32(data: *mut f32, n: i32, scale: f32) {
    (k().softmax_f32)(data, n, scale)
}

pub unsafe fn silu_mul_f32(gate: *const f32, up: *const f32, out: *mut f32, n: i32) {
    (k().silu_mul_f32)(gate, up, out, n)
}
```

- [ ] **Step 4: Commit**

```bash
git add src/kernels/ffi_inference.rs src/kernels/ffi_inference_types.rs
git commit -m "refactor: replace TurboQuant FFI with f16 attention + utility kernels"
```

---

### Task 4: Write new Eä kernels

**Files:**
- Create: `kernels/attn_f16.ea` (x86), `kernels/attn_f16_arm.ea` (ARM)
- Create: `kernels/f16_convert.ea` (x86), `kernels/f16_convert_arm.ea` (ARM)
- Create: `kernels/softmax.ea` (x86), `kernels/softmax_arm.ea` (ARM)
- Create: `kernels/silu_mul.ea` (x86), `kernels/silu_mul_arm.ea` (ARM)

**Reference:** Study `/mnt/c/Users/Peter.lukka/Desktop/DEV/llama.cpp/ggml/src/ggml-cpu/ops.cpp` for the attention and softmax implementations. Study existing Eä kernels in `kernels/` for the Eä language patterns (function signatures, SIMD intrinsics).

Each kernel needs both x86 and ARM variants. The eacompute compiler handles target selection.

- [ ] **Step 1: Write attn_f16.ea (x86) and attn_f16_arm.ea (ARM)**

Two exported functions per file:
- `attn_dot_f16(query: *f32, k_cache: *u16, scores: *mut f32, seq_len: i32, head_dim: i32)` — Q·K^T with 1/√hd scaling, f16 K cache
- `attn_vsum_f16(weights: *f32, v_cache: *u16, out: *mut f32, seq_len: i32, head_dim: i32)` — weighted V sum, f16 V cache

ARM: use `fcvtl`/`fcvtl2` to widen f16→f32, `fmla` for accumulate.
x86: use `vcvtph2ps` (F16C) to widen, `vfmadd231ps` for FMA.

- [ ] **Step 2: Write f16_convert.ea (x86) and f16_convert_arm.ea (ARM)**

Two exported functions:
- `f32_to_f16(src: *f32, dst: *mut u16, n: i32)` — bulk f32→f16
- `f16_to_f32(src: *u16, dst: *mut f32, n: i32)` — bulk f16→f32

ARM: `fcvtn`/`fcvtl` (native f16 support).
x86: `vcvtps2ph`/`vcvtph2ps` (F16C extension).

- [ ] **Step 3: Write softmax.ea (x86) and softmax_arm.ea (ARM)**

One exported function:
- `softmax_f32(data: *mut f32, n: i32, scale: f32)` — in-place softmax with pre-scale: find max, subtract, scale, exp, normalize. Match llama.cpp's implementation in `ggml_compute_forward_soft_max`.

- [ ] **Step 4: Write silu_mul.ea (x86) and silu_mul_arm.ea (ARM)**

One exported function:
- `silu_mul_f32(gate: *f32, up: *f32, out: *mut f32, n: i32)` — `out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]`

- [ ] **Step 5: Build and verify kernels compile**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build 2>&1 | head -30
```

Expected: kernel compilation succeeds (build.rs auto-discovers new .ea files).

- [ ] **Step 6: Commit**

```bash
git add kernels/attn_f16*.ea* kernels/f16_convert*.ea* kernels/softmax*.ea* kernels/silu_mul*.ea*
git commit -m "feat: new Eä kernels — attn_f16, f16_convert, softmax, silu_mul (ARM + x86)"
```

---

### Task 5: Rewrite cache.rs — F16KvCache

**Files:**
- Modify: `src/inference/cache.rs` (delete all, rewrite)

- [ ] **Step 1: Delete entire cache.rs contents and write F16KvCache**

```rust
//! KV cache — f16 storage, identical to llama.cpp.
//!
//! Layout: [layer][kv_idx][head][seq * head_dim] as f16 (u16).
//! No quantization, no rotation. Direct f32↔f16 conversion via Eä kernels.

use crate::error::{Error, Result};
use crate::kernels::ffi_inference as ffi;

pub struct F16KvCache {
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
    /// Flat buffer: [layer][kv_idx][head][max_seq_len * head_dim] as u16
    data: Vec<u16>,
    /// Elements per KV slot (one layer, one of K or V): n_kv_heads * max_seq_len * head_dim
    slot_elems: usize,
}

unsafe impl Send for F16KvCache {}

impl F16KvCache {
    pub fn new(
        n_layers: usize, n_kv_heads: usize,
        head_dim: usize, max_seq_len: usize,
    ) -> Result<Self> {
        if n_layers == 0 || n_kv_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(Error::Inference("F16KvCache: invalid dimensions".into()));
        }
        let slot_elems = n_kv_heads * max_seq_len * head_dim;
        let total = slot_elems * 2 * n_layers; // K + V per layer
        Ok(Self {
            n_layers, n_kv_heads, head_dim, max_seq_len,
            seq_len: 0, data: vec![0u16; total], slot_elems,
        })
    }

    pub fn seq_len(&self) -> usize { self.seq_len }
    pub fn len(&self) -> usize { self.seq_len }
    pub fn n_layers(&self) -> usize { self.n_layers }
    pub fn n_kv_heads(&self) -> usize { self.n_kv_heads }
    pub fn head_dim(&self) -> usize { self.head_dim }
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }

    pub fn checkpoint(&self) -> usize { self.seq_len }

    pub fn restore(&mut self, seq_len: usize) -> Result<()> {
        if seq_len > self.seq_len {
            return Err(Error::Inference(format!(
                "restore: {} > current {}", seq_len, self.seq_len
            )));
        }
        self.seq_len = seq_len;
        Ok(())
    }

    pub fn advance(&mut self, n: usize) -> Result<()> {
        if self.seq_len + n > self.max_seq_len {
            return Err(Error::Inference(format!(
                "advance: {} + {} > max {}", self.seq_len, n, self.max_seq_len
            )));
        }
        self.seq_len += n;
        Ok(())
    }

    pub fn clear(&mut self) { self.seq_len = 0; }

    /// Slot offset in data[] for (layer, kv_idx). kv_idx: 0=K, 1=V.
    fn slot_offset(&self, layer: usize, kv_idx: usize) -> usize {
        (layer * 2 + kv_idx) * self.slot_elems
    }

    /// Store f32 data into KV cache as f16.
    /// Input layout: token-major [n_kv_heads * head_dim] per token.
    /// Stored as: [head][seq * head_dim].
    pub fn store(&mut self, layer: usize, kv_idx: usize, data: &[f32], n_tokens: usize) -> Result<()> {
        if self.seq_len + n_tokens > self.max_seq_len {
            return Err(Error::Inference(format!(
                "store: {} + {} > max {}", self.seq_len, n_tokens, self.max_seq_len
            )));
        }
        let hd = self.head_dim;
        let nh = self.n_kv_heads;
        let base = self.slot_offset(layer, kv_idx);
        let head_stride = self.max_seq_len * hd;
        for h in 0..nh {
            for t in 0..n_tokens {
                let src_off = t * nh * hd + h * hd;
                let dst_off = base + h * head_stride + (self.seq_len + t) * hd;
                unsafe {
                    ffi::f32_to_f16(
                        data[src_off..].as_ptr(),
                        self.data[dst_off..].as_mut_ptr(),
                        hd as i32,
                    );
                }
            }
        }
        Ok(())
    }

    /// Pointer to K cache for one head at one layer.
    /// Points to [max_seq_len * head_dim] f16 values, of which [0..seq_len*hd] are valid.
    pub fn k_head_ptr(&self, layer: usize, head: usize) -> *const u16 {
        let off = self.slot_offset(layer, 0) + head * self.max_seq_len * self.head_dim;
        unsafe { self.data.as_ptr().add(off) }
    }

    /// Pointer to V cache for one head at one layer.
    pub fn v_head_ptr(&self, layer: usize, head: usize) -> *const u16 {
        let off = self.slot_offset(layer, 1) + head * self.max_seq_len * self.head_dim;
        unsafe { self.data.as_ptr().add(off) }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/inference/cache.rs
git commit -m "feat: F16KvCache — f16 KV storage, 1:1 with llama.cpp layout"
```

---

### Task 6: Rewrite forward_llama.rs attention path

**Files:**
- Modify: `src/inference/forward_llama.rs`

- [ ] **Step 1: Update imports and LlamaState struct**

Remove:
```rust
use crate::inference::cache::{self, EakvCache};
```
Replace with:
```rust
use crate::inference::cache::F16KvCache;
```

In `LlamaState` struct, change:
```rust
    pub(crate) kv_cache: EakvCache,
    // ...
    pub(crate) flash_state: Vec<f32>,
```
To:
```rust
    pub(crate) kv_cache: F16KvCache,
```

Remove `flash_state` field entirely.

In `LlamaState::new()`, change cache creation from:
```rust
        let kt = cache::KernelTable::init()
            .expect("kv_cache kernels not found");
        let kv_cache = EakvCache::new(
            model.n_layers as i32, model.n_kv_heads as i32,
            model.head_dim as i32, max_seq_len as i32, kt,
        ).expect("failed to create EakvCache");
```
To:
```rust
        let kv_cache = F16KvCache::new(
            model.n_layers, model.n_kv_heads,
            model.head_dim, max_seq_len,
        ).expect("failed to create F16KvCache");
```

Remove `flash_state` from the constructor and the Drop impl.

- [ ] **Step 2: Rewrite process_layer attention block**

Replace lines 172-248 (K-bias handling + flash_decode + 3-pass attention) with:

```rust
        // Bias — all applied BEFORE cache (llama.cpp style)
        add_bias(&mut self.q, lw.q_bias, h);
        add_bias(&mut self.k, lw.k_bias, kv);
        add_bias(&mut self.v, lw.v_bias, kv);

        build_rope_freqs(&mut self.rope_freqs, hd, pos, model.rope_theta);
        apply_rope(&mut self.q, &self.rope_freqs, hd, nh);
        apply_rope(&mut self.k, &self.rope_freqs, hd, nkv);

        // KV store (f32 → f16)
        self.kv_cache.store(layer, 0, &self.k[..kv], 1).unwrap();
        self.kv_cache.store(layer, 1, &self.v[..kv], 1).unwrap();
        if layer == model.n_layers - 1 { self.kv_cache.advance(1).unwrap(); }

        // Attention: per-head Q·K → softmax → V sum (f16 cache)
        let seq_len = pos + 1;
        let q_per_kv = nh / nkv;
        let rsqrt_hd = 1.0 / (hd as f32).sqrt();
        for kv_h in 0..nkv {
            let k_ptr = self.kv_cache.k_head_ptr(layer, kv_h);
            let v_ptr = self.kv_cache.v_head_ptr(layer, kv_h);
            for q_off in 0..q_per_kv {
                let q_h = kv_h * q_per_kv + q_off;
                let q_slice = &self.q[q_h * hd..(q_h + 1) * hd];
                let scores = &mut self.attn_scores[q_h * seq_len..(q_h + 1) * seq_len];
                unsafe {
                    ffi::attn_dot_f16(
                        q_slice.as_ptr(), k_ptr,
                        scores.as_mut_ptr(), seq_len as i32, hd as i32,
                    );
                    ffi::softmax_f32(scores.as_mut_ptr(), seq_len as i32, rsqrt_hd);
                    ffi::attn_vsum_f16(
                        scores.as_ptr(), v_ptr,
                        self.attn_out[q_h * hd..(q_h + 1) * hd].as_mut_ptr(),
                        seq_len as i32, hd as i32,
                    );
                }
            }
        }
```

- [ ] **Step 3: Verify compile**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo check 2>&1 | head -30
```

Expected: errors only from other files (prefill_llama.rs, forward.rs, prefill.rs) — forward_llama.rs should be clean.

- [ ] **Step 4: Commit**

```bash
git add src/inference/forward_llama.rs
git commit -m "refactor: forward_llama.rs — f16 KV cache, K-bias before cache, kernel attention"
```

---

### Task 7: Rewrite prefill_llama.rs attention path

**Files:**
- Modify: `src/inference/prefill_llama.rs`

- [ ] **Step 1: Update imports**

Remove:
```rust
use crate::inference::cache;
```

(F16KvCache is accessed via `self.kv_cache` which is already on LlamaState.)

- [ ] **Step 2: Rewrite KV store (remove transpose)**

Replace lines 131-163 (bias + rope + head-major transpose + TurboQuant append) with:

```rust
            // Bias — all applied BEFORE cache (llama.cpp style)
            for t in 0..n {
                add_bias(&mut qs_all[t*h..(t+1)*h], lw.q_bias, h);
                add_bias(&mut ks_all[t*kv..(t+1)*kv], lw.k_bias, kv);
                add_bias(&mut vs_all[t*kv..(t+1)*kv], lw.v_bias, kv);
            }
            let mut rope_freqs = vec![0.0f32; hd];
            let pos_base = self.kv_cache.seq_len();
            for t in 0..n {
                build_rope_freqs(&mut rope_freqs, hd, pos_base + t, model.rope_theta);
                apply_rope(&mut qs_all[t*h..(t+1)*h], &rope_freqs, hd, nh);
                apply_rope(&mut ks_all[t*kv..(t+1)*kv], &rope_freqs, hd, nkv);
            }

            // KV store (f32 → f16) — token-major input, cache handles layout
            for t in 0..n {
                self.kv_cache.store(layer, 0, &ks_all[t*kv..(t+1)*kv], 1).unwrap();
                self.kv_cache.store(layer, 1, &vs_all[t*kv..(t+1)*kv], 1).unwrap();
                if t < n - 1 || layer == model.n_layers - 1 {
                    // Only advance seq_len once per token after all layers store
                }
            }
            // Advance seq_len for all tokens at last layer
            if layer == model.n_layers - 1 {
                self.kv_cache.advance(n).unwrap();
            }
```

Wait — `store()` uses `self.seq_len` as write position, but for N tokens we need to advance after each store-pair for the next token to land at the right position. Let me fix the store loop:

```rust
            // KV store: one token at a time (cache.store uses seq_len as write pos)
            for t in 0..n {
                self.kv_cache.store(layer, 0, &ks_all[t*kv..(t+1)*kv], 1).unwrap();
                self.kv_cache.store(layer, 1, &vs_all[t*kv..(t+1)*kv], 1).unwrap();
                if layer == model.n_layers - 1 {
                    self.kv_cache.advance(1).unwrap();
                }
            }
```

Actually, this won't work either — we need all layers to store before advancing. The cache needs a bulk store API or we handle position manually. Let me use a simpler approach — store all N tokens at once per layer using `store(layer, kv_idx, data, n_tokens)`:

```rust
            // KV store (f32 → f16) — all N tokens at once
            self.kv_cache.store(layer, 0, &ks_all[..n*kv], n).unwrap();
            self.kv_cache.store(layer, 1, &vs_all[..n*kv], n).unwrap();
            if layer == model.n_layers - 1 {
                self.kv_cache.advance(n).unwrap();
            }
```

This works because `F16KvCache::store()` takes `n_tokens` and scatters token-major data into head-major layout at `seq_len..seq_len+n_tokens`.

- [ ] **Step 3: Rewrite attention block**

Replace lines 167-238 (fused_causal_attn + 3-pass TurboQuant) with:

```rust
            // Attention: per-head Q·K → softmax → V sum (f16 cache)
            let seq_len = pos_base + n;
            let q_per_kv = nh / nkv;
            let rsqrt_hd = 1.0 / (hd as f32).sqrt();
            let max_scores = nh * seq_len;
            let mut scores = vec![0.0f32; max_scores];
            for t in 0..n {
                let causal_len = pos_base + t + 1;
                let qt = &qs_all[t*h..(t+1)*h];
                for kv_h in 0..nkv {
                    let k_ptr = self.kv_cache.k_head_ptr(layer, kv_h);
                    let v_ptr = self.kv_cache.v_head_ptr(layer, kv_h);
                    for q_off in 0..q_per_kv {
                        let q_h = kv_h * q_per_kv + q_off;
                        let q_slice = &qt[q_h * hd..(q_h + 1) * hd];
                        let s = &mut scores[..causal_len];
                        unsafe {
                            ffi::attn_dot_f16(
                                q_slice.as_ptr(), k_ptr,
                                s.as_mut_ptr(), causal_len as i32, hd as i32,
                            );
                            ffi::softmax_f32(s.as_mut_ptr(), causal_len as i32, rsqrt_hd);
                            ffi::attn_vsum_f16(
                                s.as_ptr(), v_ptr,
                                attn_all[t*h + q_h*hd..].as_mut_ptr(),
                                causal_len as i32, hd as i32,
                            );
                        }
                    }
                }
            }
```

- [ ] **Step 4: Verify compile**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo check 2>&1 | head -30
```

- [ ] **Step 5: Commit**

```bash
git add src/inference/prefill_llama.rs
git commit -m "refactor: prefill_llama.rs — f16 attention, no transpose, K-bias before cache"
```

---

### Task 8: Update BitNet forward.rs and prefill.rs

**Files:**
- Modify: `src/inference/forward.rs`
- Modify: `src/inference/prefill.rs`

- [ ] **Step 1: Update forward.rs**

Replace `EakvCache` with `F16KvCache`. Replace `cache::attention::attention_scores` and `cache::attention::attention_output` calls (lines 246-252) with per-head f16 attention loop (same pattern as Task 6 Step 2).

Update cache creation (lines 173-176) to use `F16KvCache::new(...)`.

Remove `cache::KernelTable::init()` call.

- [ ] **Step 2: Update prefill.rs**

Replace `cache::attention::attention_scores` and `cache::attention::attention_output` calls (lines 92-99) with f16 attention loop.

Replace `kv_cache.append()` calls (lines 71-72) with `kv_cache.store()`.

Replace `kv_cache.advance(n as i32)` with `kv_cache.advance(n)` (usize now, not i32).

- [ ] **Step 3: Update speculative.rs**

Update `checkpoint()` and `restore()` calls — they now return/accept `usize` instead of `i32`.

- [ ] **Step 4: Verify full build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build 2>&1 | tail -5
```

Expected: builds successfully.

- [ ] **Step 5: Commit**

```bash
git add src/inference/forward.rs src/inference/prefill.rs src/inference/speculative.rs
git commit -m "refactor: BitNet forward/prefill — f16 KV cache, remove TurboQuant attention"
```

---

### Task 9: Rewrite tests/cache.rs

**Files:**
- Modify: `tests/cache.rs`
- Modify: `tests/test_flash_decode.rs` (if exists, remove `has_flash_decode_attn` ref)

- [ ] **Step 1: Rewrite cache test**

```rust
use olorin::kernels::ffi;
use olorin::inference::cache::F16KvCache;

#[test]
fn test_f16_cache_new() {
    ffi::init().unwrap();
    let cache = F16KvCache::new(1, 5, 128, 64).unwrap();
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.n_layers(), 1);
    assert_eq!(cache.n_kv_heads(), 5);
    assert_eq!(cache.head_dim(), 128);
}

#[test]
fn test_f16_cache_store_advance() {
    ffi::init().unwrap();
    let mut cache = F16KvCache::new(1, 2, 4, 64).unwrap();
    // One token, 2 heads × 4 dim = 8 floats
    let k = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let v = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    cache.store(0, 0, &k, 1).unwrap();
    cache.store(0, 1, &v, 1).unwrap();
    cache.advance(1).unwrap();
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_f16_cache_checkpoint_restore() {
    ffi::init().unwrap();
    let mut cache = F16KvCache::new(1, 2, 4, 64).unwrap();
    let data = vec![1.0f32; 8];
    cache.store(0, 0, &data, 1).unwrap();
    cache.store(0, 1, &data, 1).unwrap();
    cache.advance(1).unwrap();
    let cp = cache.checkpoint();
    assert_eq!(cp, 1);
    cache.restore(0).unwrap();
    assert_eq!(cache.len(), 0);
}
```

- [ ] **Step 2: Fix test_flash_decode.rs**

Remove or update any `has_flash_decode_attn()` references. If the test relies entirely on flash decode, delete the test.

- [ ] **Step 3: Run tests**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/cache.rs tests/test_flash_decode.rs
git commit -m "test: rewrite cache tests for F16KvCache"
```

---

### Task 10: C benchmarks — rmsnorm, quant_q8k, rope

**Files:**
- Create: `benchmarks/rmsnorm_bench/bench.c`, `benchmarks/rmsnorm_bench/build.sh`
- Create: `benchmarks/quant_q8k_bench/bench.c`, `benchmarks/quant_q8k_bench/build.sh`
- Create: `benchmarks/rope_bench/bench.c`, `benchmarks/rope_bench/build.sh`

Each benchmark follows the exact same pattern as `benchmarks/q4k_dot_bench/bench.c`:
- `dlopen` the kernel .so
- Generate deterministic test data with xorshift32
- Scalar reference implementation
- Correctness check (abs error, rel error, PASS/FAIL)
- Benchmark N iterations, print ns/call

**Reference:** Extract the scalar reference implementations from llama.cpp:
- RMSNorm: `/mnt/c/Users/Peter.lukka/Desktop/DEV/llama.cpp/ggml/src/ggml-cpu/ops.cpp` → `ggml_compute_forward_rms_norm`
- Q8K quant: `/mnt/c/Users/Peter.lukka/Desktop/DEV/llama.cpp/ggml/src/ggml-cpu/quants.c` → `quantize_row_q8_K`
- RoPE: `/mnt/c/Users/Peter.lukka/Desktop/DEV/llama.cpp/ggml/src/ggml-cpu/ops.cpp` → `ggml_compute_forward_rope`

- [ ] **Step 1: Write rmsnorm_bench**

`bench.c`: Load `libbitnet_rmsnorm.so`, symbol `rmsnorm_f32`. Test with dim=3072 (Llama 3.2 3B hidden dim). Scalar ref: `rms = sqrt(mean(x²) + eps); out = x/rms * weight`.

`build.sh`:
```bash
#!/bin/bash
set -e
gcc -O3 -march=armv8.2-a+dotprod -o bench bench.c -ldl -lm -DNDEBUG
echo "run: ./bench ~/.olorin/lib/*/libbitnet_rmsnorm.so"
```

- [ ] **Step 2: Write quant_q8k_bench**

`bench.c`: Load `libq4k_quant.so`, symbol `quant_f32_q8k`. Test with dim=3072. Scalar ref: find abs_max per 256 block, scale = abs_max/127, quantize.

- [ ] **Step 3: Write rope_bench**

`bench.c`: Load `librope.so`, symbol `apply_rope_f32`. Test with head_dim=128, n_heads=32. Scalar ref: standard rotary position encoding.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/rmsnorm_bench/ benchmarks/quant_q8k_bench/ benchmarks/rope_bench/
git commit -m "bench: C benchmarks for rmsnorm, quant_q8k, rope — correctness + timing"
```

---

### Task 11: C benchmarks — attn_f16, f16_convert, softmax, silu_mul

**Files:**
- Create: `benchmarks/attn_f16_bench/bench.c`, `benchmarks/attn_f16_bench/build.sh`
- Create: `benchmarks/f16_convert_bench/bench.c`, `benchmarks/f16_convert_bench/build.sh`
- Create: `benchmarks/softmax_bench/bench.c`, `benchmarks/softmax_bench/build.sh`
- Create: `benchmarks/silu_bench/bench.c`, `benchmarks/silu_bench/build.sh`

- [ ] **Step 1: Write attn_f16_bench**

`bench.c`: Load `libattn_f16.so`, symbols `attn_dot_f16` + `attn_vsum_f16`. Test with seq_len=512, head_dim=128. Scalar ref:
```c
// attn_dot_f16 reference
float ref_attn_dot(const float *q, const uint16_t *k, int seq_len, int hd) {
    for (int t = 0; t < seq_len; t++) {
        float dot = 0;
        for (int d = 0; d < hd; d++)
            dot += q[d] * f16_to_f32(k[t * hd + d]);
        scores[t] = dot;
    }
}
```

Correctness threshold: rel error < 1e-3 (f16 precision).

- [ ] **Step 2: Write f16_convert_bench**

`bench.c`: Load `libf16_convert.so`, symbols `f32_to_f16` + `f16_to_f32`. Test with n=3072. Round-trip test: f32 → f16 → f32, check rel error < 1e-3.

- [ ] **Step 3: Write softmax_bench**

`bench.c`: Load `libsoftmax.so`, symbol `softmax_f32`. Test with n=512 (typical seq_len). Scalar ref: max → subtract → exp → normalize. Check sum ≈ 1.0 and values match.

- [ ] **Step 4: Write silu_bench**

`bench.c`: Load `libsilu_mul.so`, symbol `silu_mul_f32`. Test with n=8192 (typical ffn_dim). Scalar ref: `out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]`.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/attn_f16_bench/ benchmarks/f16_convert_bench/ benchmarks/softmax_bench/ benchmarks/silu_bench/
git commit -m "bench: C benchmarks for attn_f16, f16_convert, softmax, silu_mul"
```

---

### Task 12: Final cleanup and verification

**Files:**
- All files from previous tasks

- [ ] **Step 1: Check for any remaining dead code**

```bash
# Search for any remaining references to deleted functions
grep -rn 'turbo_rotate\|fwht_inplace\|sign_flip\|quantize_simd\|dequantize_simd\|fused_k_score\|fused_v_sum\|flash_decode\|fused_causal_attn\|EakvCache\|KvSlice\|jl_signs\|rotate_groups' src/ tests/
```

Expected: no output. If anything found, delete it.

- [ ] **Step 2: Check for stale kernel .ea files**

```bash
ls kernels/ | grep -E 'turbo|quantize_simd|fused_k|fused_v|flash_decode|fused_causal|fused_attention|dequantize'
```

Expected: no output.

- [ ] **Step 3: Full build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build 2>&1
```

Expected: success, no warnings about dead code.

- [ ] **Step 4: Run all tests**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test 2>&1
```

Expected: all tests pass.

- [ ] **Step 5: Verify no file exceeds 500 lines**

```bash
wc -l src/inference/cache.rs src/inference/forward_llama.rs src/inference/prefill_llama.rs src/inference/forward.rs src/inference/prefill.rs src/kernels/ffi.rs src/kernels/ffi_inference.rs src/kernels/ffi_inference_types.rs
```

Expected: all under 500 lines.

- [ ] **Step 6: Commit any cleanup**

```bash
git add -A
git commit -m "cleanup: remove all TurboQuant dead code — llama.cpp 1:1 parity complete"
```
