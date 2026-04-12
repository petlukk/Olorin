# Phase 2 — Plan 2: Unified Gemm Forward Path

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **HARD RULES (apply to ALL agents):**
> - No file exceeds 500 lines. Split before you hit the limit.
> - Every feature proven by end-to-end test. If it's not tested, it doesn't exist.
> - No fake functions. No silent fallbacks. No `// TODO`, `// HACK`, `// for now`.
> - Olorin is Ea's showcase — every SIMD op must be an Ea kernel. **Do NOT simplify kernel code to scalar Rust.**
> - Match llama.cpp **bit-exact** (per-output `to_bits()` where the recipe allows).
> - **x86 kernels target AVX2 ONLY.** No AVX-512.
> - **ARM kernels target Cortex-A76 (NEON + dotprod, NO i8mm).**
> - eacompute compiler: `$HOME/projects/eacompute/target/release/ea`
> - Build: `PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release`
> - Branch: `gemma4-batched-prompt-eval`
> - **eabrain protocol** (mandatory):
>   - Start of every task: `eabrain status` and `eabrain recall`.
>   - Before searching for any Ea symbol by name: `eabrain search <name>`.
>   - Before assuming an Ea intrinsic doesn't exist: `eabrain ref <name>` AND grep `$HOME/projects/eacompute/src/typeck/intrinsics*.rs` + `$HOME/projects/eacompute/src/codegen/simd*.rs`.
>   - After editing any `.ea` kernel: `eabrain index`.
>   - End of any task producing a non-obvious finding: `eabrain remember "..."`.

**Goal:** Replace `forward_one_graph` with a unified `forward` function that uses the Q4K 8x8 gemm kernel for all Q4K matmuls (any N), with a fused batched attention Ea kernel. Close the 4.4x prefill gap vs llama.cpp.

**Architecture:** Kernel-first TDD. Land the fused attention kernel (x86 + ARM) with standalone tests first. Then build `forward_batch` as a new function, wire it into `generate.rs`, verify bit-exact at N=1 against the existing decode path, then test N>1 against llama-eval-callback. Finally delete Path A dead code.

**Tech Stack:** Rust, Ea (eacompute), x86 AVX2 + ARM NEON+dotprod, existing `q4k_8x8_q8k_gemm` + `q8k_repack_4` kernels from Plan 1, work-stealing `GraphPool` + `SpinBarrier`.

**Spec:** `docs/superpowers/specs/2026-04-12-phase-2-plan-2-unified-gemm-forward-design.md`

**Baseline (for delta gate interpretation):**
- `cargo build --release 2>&1 | grep -c ^warning` — record at task start (currently ~10).
- `find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'` — only `chacha20_search_v2.ea` (750) and `chacha20_search_v2_arm.ea` (609) expected.
- Existing tests that must stay green: `repack_q4k`, `dual_q4k_8x8`, `gemma4_parallel_regression`, `gemma4_verify`, `gemma4_smoke`, `q8k_repack_4` (if exists), `gemm_q4k_8x8`, `bench_q4k_gemm`.

---

## Per-Task Verification Gates

Run these before every `git commit`. Delta interpretation — gates fail only if the delta vs. baseline grows.

**Gate 1 — Build clean (delta).**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
N=$(grep -c "^warning" /tmp/olorin-build.log)
echo "warnings: $N (record baseline at task start)"
```

**Gate 2 — Line limit (delta).**

```bash
find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'
```

Expected: only the two chacha search kernels. Any new file over 500 = fail.

**Gate 3 — Existing test suite.**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression 2>&1 | tail -6
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -15
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemm_q4k_8x8 2>&1 | tail -6
```

All must pass. The parallel_regression snapshot must NOT be regenerated during this plan (until the Path B replacement is proven equivalent).

---

## File Map

**New files:**
- `kernels/attn_fused_batched.ea` — x86 AVX2 fused attention kernel
- `kernels/attn_fused_batched_arm.ea` — ARM NEON+dotprod fused attention kernel
- `src/inference/forward_batch.rs` — the new unified forward function + batched layer loop
- `tests/attn_fused_batched.rs` — standalone attention kernel tests
- `tests/forward_batch_verify.rs` — bit-exact N=1 vs existing path, N>1 layer-by-layer

**Modified files:**
- `src/inference/forward.rs` — add batched buffer fields to `Gemma4State`, `store_batch` helper
- `src/inference/cache.rs` — add `store_batch` and `advance_n` methods
- `src/kernels/ffi_inference.rs` — add `attn_fused_batched` FFI wrapper
- `src/kernels/ffi_inference_types.rs` — add `AttnFusedBatchedFn` type alias
- `src/inference/generate.rs` — switch prefill to `forward_batch`, decode to `forward_batch(&[tok])`
- `src/inference/mod.rs` — add `pub mod forward_batch;`
- `build.rs` — auto-discovers new kernels (no manual change needed)

**Deleted files (Task 11):**
- `src/inference/forward_attn.rs` — Path A layer_forward
- `src/inference/forward_attn_heads.rs` — Path A attention_decode (if only used by Path A)
- Dead code in `matmul.rs` only called from Path A

---

## Task 1: Research — llama.cpp batched attention + causal mask

**Goal:** Document exactly how llama.cpp computes batched causal attention so the Ea kernel matches. Zero code changes.

**Files:**
- Create: `docs/superpowers/research/2026-04-12-batched-causal-attention.md`
- Read (no edit): `$HOME/projects/llama.cpp/src/llama-graph.cpp` — attention section (~line 1900-1960)
- Read (no edit): `$HOME/projects/llama.cpp/ggml/src/ggml-cpu/ops.cpp` — `ggml_compute_forward_soft_max` for how the causal mask is applied
- Read (no edit): existing olorin attention at `src/inference/forward_graph.rs:265-310`

- [ ] **Step 1: Read llama.cpp attention flow**

Find the batched attention computation in llama-graph.cpp. Document:
- How Q*K^T is computed for N query tokens (ggml_mul_mat shapes)
- How the causal mask is built and applied (mask shape, -inf placement)
- How softmax is applied (per-row, with scale)
- How the V multiply works (scores * V shape)

- [ ] **Step 2: Read olorin's current single-token attention**

From `forward_graph.rs:265-310`, document the exact computation:
- f16→f32 conversion of K/V from cache
- Dot product: `f32_dot(q, k, head_dim)`
- Softmax: `softmax_f32(scores, attn_len, scale=1.0)`
- V accumulate: `f32_dot_acc(out, v, score, head_dim)`
- GQA mapping: `kv_h = h / gqa_ratio`
- Cache layout: stride = `n_kv_heads * head_dim`, position p at offset `p * stride + kv_h * head_dim`

- [ ] **Step 3: Document the batched kernel contract**

Write the research note specifying:
- For N query tokens and `n_kv = seq_len_before + N` cache positions:
  - Query token `i` (0-indexed) attends to cache positions `0 .. seq_len_before + i + 1`
  - Positions `>= seq_len_before + i + 1` get `-inf` before softmax
- Score computation: `scores[i][j] = dot(q[i*hd..], k_cache[j*stride + kv_h*hd..]) * attn_scale`
- Softmax: row-wise over the `n_kv` dimension, scale = 1.0 for Gemma 4
- Output: `out[i*hd..] = sum_j(softmax_scores[i][j] * v_cache[j*stride + kv_h*hd..])`
- All K/V are f16 in cache, must be converted to f32 for computation
- Scores buffer per query: at most 512 floats (sliding window) or `seq_len + N` (global)

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/research/2026-04-12-batched-causal-attention.md
git commit -m "research: batched causal attention — kernel contract for fused Ea kernel"
```

---

## Task 2: Ea fused attention kernel — x86 AVX2

**Goal:** Write `attn_fused_batched.ea` for x86 AVX2. This kernel processes one head: takes N query vectors, the full K/V cache (f16), and produces N output vectors with causal masking built in.

**Files:**
- Create: `kernels/attn_fused_batched.ea`
- Read (no edit): `src/inference/forward_graph.rs:265-310` (reference single-token attention)
- Read (no edit): research note from Task 1

- [ ] **Step 1: eabrain baseline**

```bash
eabrain status
eabrain recall
eabrain search attn
eabrain ref f16_to_f32
eabrain ref f32_dot
eabrain ref softmax
```

- [ ] **Step 2: Write the kernel**

Create `kernels/attn_fused_batched.ea`. The kernel operates on one attention head at a time (the Rust caller dispatches heads across threads).

```
// Fused batched causal attention — one head.
// For each query token i in 0..n_batch:
//   1. dot(q[i], k_cache[j]) * attn_scale for j in 0..n_kv
//   2. mask: set scores[j] = -inf for j >= cache_start + i + 1
//   3. softmax over scores[0..n_kv]
//   4. out[i] = weighted sum of v_cache[j] by softmax scores

export func attn_fused_batched(
    q: *f32,               // [head_dim, n_batch] col-major: token i at q + i * head_dim
    k_cache: *u16,         // [stride_kv, n_kv] f16, position j at k_cache + j * stride_kv + kv_head_offset
    v_cache: *u16,         // [stride_kv, n_kv] f16, same layout
    out dst: *mut f32,     // [head_dim, n_batch] col-major
    scores_buf: *mut f32,  // [n_kv] scratch — caller provides per-thread
    kv_scratch: *mut f32,  // [head_dim] scratch for f16→f32 conversion
    head_dim: i32,
    stride_kv: i32,        // n_kv_heads * head_dim (stride between positions)
    kv_head_offset: i32,   // kv_h * head_dim (offset to this head's K/V within a position)
    n_kv: i32,             // total KV positions (seq_len_before + n_batch)
    n_batch: i32,
    cache_start: i32,      // seq_len_before
    attn_scale: f32        // 1.0 for Gemma 4
)
```

**Inner loop structure (pseudocode — implement in Ea with SIMD):**

```
let mut i: i32 = 0
while i < n_batch {
    let q_ptr = q + i * head_dim
    let causal_limit = cache_start + i + 1

    // 1. QK^T scores
    let mut j: i32 = 0
    while j < n_kv {
        // f16 → f32 for K[j]
        let k_off = j * stride_kv + kv_head_offset
        // Convert head_dim f16 values to f32 into kv_scratch
        // Compute dot product: q_ptr · kv_scratch
        // Store to scores_buf[j]
        j = j + 1
    }

    // 2. Causal mask: -inf for j >= causal_limit
    let mut j2: i32 = causal_limit
    while j2 < n_kv {
        store(scores_buf, j2, -340282346638528859811704183484516925440.0)  // -FLT_MAX or -inf
        j2 = j2 + 1
    }

    // 3. Softmax (inline: find max, subtract, exp, sum, normalize)
    // ... standard numerically-stable softmax over scores_buf[0..n_kv]

    // 4. V weighted sum
    let out_ptr = dst + i * head_dim
    // Zero out_ptr[0..head_dim]
    let mut j3: i32 = 0
    while j3 < n_kv {
        let s: f32 = load(scores_buf, j3)
        // f16 → f32 for V[j3]
        let v_off = j3 * stride_kv + kv_head_offset
        // out_ptr += s * v_scratch
        j3 = j3 + 1
    }

    i = i + 1
}
```

**Key SIMD operations needed:**
- f16→f32 conversion: use `cvt_f16_f32` (8-wide) or equivalent Ea intrinsic. Check `eabrain ref cvt_f16` first.
- f32 dot product: accumulate with `mul_f32` + horizontal add, or use `fmadd_f32` if available.
- Softmax: vectorized max-reduce, sub, exp (approximate is fine if llama.cpp uses approximate), sum-reduce, div.
- f32 dot_acc (scale-accumulate): `fmadd_f32` over head_dim chunks.

**CRITICAL:** Read the existing attention code at `forward_graph.rs:265-310` first and replicate its exact computation. The kernel must produce identical results to the existing per-position loop for N=1.

- [ ] **Step 3: Verify it compiles**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
eabrain index
```

- [ ] **Step 4: Commit**

```bash
git add kernels/attn_fused_batched.ea kernels/attn_fused_batched.ea.json
git commit -m "feat: x86 AVX2 fused batched causal attention Ea kernel"
```

---

## Task 3: Ea fused attention kernel — ARM NEON+dotprod

**Goal:** Write the ARM variant of the fused attention kernel, derived from the x86 version in Task 2.

**Files:**
- Create: `kernels/attn_fused_batched_arm.ea`
- Read (no edit): `kernels/attn_fused_batched.ea` (Task 2 output)

- [ ] **Step 1: Write the ARM kernel**

Same signature as Task 2's kernel. Adapt SIMD operations for NEON:
- f16→f32: NEON `vcvt_f32_f16` equivalent in Ea
- Dot product: use NEON float multiply-accumulate
- Softmax: same algorithm, NEON intrinsics
- All control flow identical to x86 version

- [ ] **Step 2: Verify it compiles**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
eabrain index
```

- [ ] **Step 3: Commit**

```bash
git add kernels/attn_fused_batched_arm.ea kernels/attn_fused_batched_arm.ea.json
git commit -m "feat: ARM NEON fused batched causal attention Ea kernel"
```

---

## Task 4: FFI binding for fused attention kernel

**Goal:** Wire the new kernel into olorin's kernel table and provide a safe Rust wrapper.

**Files:**
- Modify: `src/kernels/ffi_inference_types.rs` — add type alias
- Modify: `src/kernels/ffi_inference.rs` — add field to KernelTableInference, load, wrapper

- [ ] **Step 1: Add type alias**

In `src/kernels/ffi_inference_types.rs`, add:

```rust
pub type AttnFusedBatchedFn = unsafe extern "C" fn(
    *const f32,  // q [head_dim, n_batch]
    *const u16,  // k_cache [stride_kv, n_kv] f16
    *const u16,  // v_cache [stride_kv, n_kv] f16
    *mut f32,    // dst [head_dim, n_batch]
    *mut f32,    // scores_buf [n_kv] scratch
    *mut f32,    // kv_scratch [head_dim] scratch
    i32,         // head_dim
    i32,         // stride_kv
    i32,         // kv_head_offset
    i32,         // n_kv
    i32,         // n_batch
    i32,         // cache_start
    f32,         // attn_scale
);
```

- [ ] **Step 2: Add to KernelTableInference and load**

In `src/kernels/ffi_inference.rs`, add the field to the kernel table struct and the `load()` call following the existing pattern (e.g., how `q4k_8x8_q8k_gemm` is loaded). The kernel library name will be `libattn_fused_batched.so` (auto-discovered by build.rs).

- [ ] **Step 3: Add safe wrapper**

```rust
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_fused_batched(
    q: *const f32,
    k_cache: *const u16,
    v_cache: *const u16,
    dst: *mut f32,
    scores_buf: *mut f32,
    kv_scratch: *mut f32,
    head_dim: i32,
    stride_kv: i32,
    kv_head_offset: i32,
    n_kv: i32,
    n_batch: i32,
    cache_start: i32,
    attn_scale: f32,
) {
    (k().attn_fused_batched)(
        q, k_cache, v_cache, dst, scores_buf, kv_scratch,
        head_dim, stride_kv, kv_head_offset, n_kv, n_batch, cache_start, attn_scale,
    )
}
```

- [ ] **Step 4: Build clean**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "^(error)" | head -5
```

- [ ] **Step 5: Commit**

```bash
git add src/kernels/ffi_inference_types.rs src/kernels/ffi_inference.rs
git commit -m "feat(ffi): wire attn_fused_batched kernel binding"
```

---

## Task 5: Standalone test — fused attention kernel vs existing attention

**Goal:** Prove the fused kernel produces identical output to the existing per-position attention loop for both N=1 (decode) and N>1 (batched) cases.

**Files:**
- Create: `tests/attn_fused_batched.rs`

- [ ] **Step 1: Write N=1 test (must match existing attention exactly)**

```rust
//! Standalone test for the fused batched causal attention kernel.
//! Verifies bit-exact match against olorin's existing per-position attention loop.

#[test]
fn attn_n1_matches_existing_loop() {
    olorin::kernels::ffi::init().unwrap();

    let head_dim = 256usize;
    let n_kv_heads = 1usize;
    let stride_kv = n_kv_heads * head_dim;
    let n_kv = 20usize;  // 20 cached positions
    let n_batch = 1usize;
    let cache_start = (n_kv - n_batch) as i32; // 19 — this token is position 19
    let attn_scale = 1.0f32;
    let kv_head_offset = 0i32; // head 0 of kv

    // Generate deterministic Q, K_cache (f16), V_cache (f16)
    let mut q = vec![0.0f32; head_dim * n_batch];
    for i in 0..head_dim {
        q[i] = ((i as f32) * 0.0137 - 0.5).sin() * 0.3;
    }

    // K/V cache as f16 (u16 storage)
    let mut k_cache = vec![0u16; stride_kv * n_kv];
    let mut v_cache = vec![0u16; stride_kv * n_kv];
    for p in 0..n_kv {
        for i in 0..head_dim {
            let kv = ((p * head_dim + i) as f32 * 0.00731 - 0.2).cos() * 0.5;
            let vv = ((p * head_dim + i) as f32 * 0.00913 + 0.1).sin() * 0.4;
            k_cache[p * stride_kv + i] = f32_to_f16(kv);
            v_cache[p * stride_kv + i] = f32_to_f16(vv);
        }
    }

    // ── Reference: replicate forward_graph.rs:265-310 loop ──
    let mut ref_out = vec![0.0f32; head_dim];
    let mut scores = vec![0.0f32; n_kv];
    let mut kv_scratch = vec![0.0f32; head_dim];
    for p in 0..n_kv {
        let k_src = k_cache[p * stride_kv..].as_ptr();
        unsafe { olorin::kernels::ffi_inference::f16_to_f32(k_src, kv_scratch.as_mut_ptr(), head_dim as i32); }
        scores[p] = olorin::kernels::ffi_inference::f32_dot(q.as_ptr(), kv_scratch.as_ptr(), head_dim as i32);
    }
    unsafe { olorin::kernels::ffi_inference::softmax_f32(scores.as_mut_ptr(), n_kv as i32, attn_scale); }
    unsafe { std::ptr::write_bytes(ref_out.as_mut_ptr(), 0, head_dim); }
    for p in 0..n_kv {
        let v_src = v_cache[p * stride_kv..].as_ptr();
        unsafe { olorin::kernels::ffi_inference::f16_to_f32(v_src, kv_scratch.as_mut_ptr(), head_dim as i32); }
        olorin::kernels::ffi_inference::f32_dot_acc(ref_out.as_mut_ptr(), kv_scratch.as_ptr(), scores[p], head_dim as i32);
    }

    // ── Fused kernel ──
    let mut fused_out = vec![0.0f32; head_dim * n_batch];
    let mut fused_scores = vec![0.0f32; n_kv];
    let mut fused_kv_scratch = vec![0.0f32; head_dim];
    unsafe {
        olorin::kernels::ffi_inference::attn_fused_batched(
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            fused_out.as_mut_ptr(),
            fused_scores.as_mut_ptr(),
            fused_kv_scratch.as_mut_ptr(),
            head_dim as i32,
            stride_kv as i32,
            kv_head_offset,
            n_kv as i32,
            n_batch as i32,
            cache_start,
            attn_scale,
        );
    }

    // Compare bit-exact
    let mut mismatches = 0;
    for i in 0..head_dim {
        if ref_out[i].to_bits() != fused_out[i].to_bits() {
            if mismatches < 5 {
                eprintln!("MISMATCH at {i}: ref={} fused={}", ref_out[i], fused_out[i]);
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "N=1 attention: {mismatches}/{head_dim} mismatches");
    eprintln!("PASS: N=1 attention bit-exact ({head_dim} outputs)");
}

fn f32_to_f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
```

- [ ] **Step 2: Write N=4 batched test with causal masking**

```rust
#[test]
fn attn_n4_batched_causal() {
    olorin::kernels::ffi::init().unwrap();

    let head_dim = 256usize;
    let n_kv_heads = 1usize;
    let stride_kv = n_kv_heads * head_dim;
    let cache_start = 10usize; // 10 tokens already in cache
    let n_batch = 4usize;
    let n_kv = cache_start + n_batch; // 14 total positions
    let attn_scale = 1.0f32;
    let kv_head_offset = 0i32;

    // Generate Q [head_dim, 4] and K/V cache [stride_kv, 14]
    let mut q = vec![0.0f32; head_dim * n_batch];
    for b in 0..n_batch {
        for i in 0..head_dim {
            q[b * head_dim + i] = ((b * head_dim + i) as f32 * 0.0137 - 0.5).sin() * 0.3;
        }
    }
    let mut k_cache = vec![0u16; stride_kv * n_kv];
    let mut v_cache = vec![0u16; stride_kv * n_kv];
    for p in 0..n_kv {
        for i in 0..head_dim {
            k_cache[p * stride_kv + i] = f32_to_f16(((p * head_dim + i) as f32 * 0.00731 - 0.2).cos() * 0.5);
            v_cache[p * stride_kv + i] = f32_to_f16(((p * head_dim + i) as f32 * 0.00913 + 0.1).sin() * 0.4);
        }
    }

    // ── Reference: N independent calls with causal masking ──
    let mut ref_out = vec![0.0f32; head_dim * n_batch];
    let mut scores = vec![0.0f32; n_kv];
    let mut kv_scratch = vec![0.0f32; head_dim];
    for b in 0..n_batch {
        let causal_limit = cache_start + b + 1;
        let q_ptr = q[b * head_dim..].as_ptr();

        // QK^T
        for p in 0..n_kv {
            let k_src = k_cache[p * stride_kv..].as_ptr();
            unsafe { olorin::kernels::ffi_inference::f16_to_f32(k_src, kv_scratch.as_mut_ptr(), head_dim as i32); }
            scores[p] = olorin::kernels::ffi_inference::f32_dot(q_ptr, kv_scratch.as_ptr(), head_dim as i32);
        }
        // Causal mask
        for p in causal_limit..n_kv {
            scores[p] = f32::NEG_INFINITY;
        }
        // Softmax
        unsafe { olorin::kernels::ffi_inference::softmax_f32(scores.as_mut_ptr(), n_kv as i32, attn_scale); }
        // V weighted sum
        let out_ptr = ref_out[b * head_dim..].as_mut_ptr();
        unsafe { std::ptr::write_bytes(out_ptr, 0, head_dim); }
        for p in 0..n_kv {
            let v_src = v_cache[p * stride_kv..].as_ptr();
            unsafe { olorin::kernels::ffi_inference::f16_to_f32(v_src, kv_scratch.as_mut_ptr(), head_dim as i32); }
            olorin::kernels::ffi_inference::f32_dot_acc(out_ptr, kv_scratch.as_ptr(), scores[p], head_dim as i32);
        }
    }

    // ── Fused kernel ──
    let mut fused_out = vec![0.0f32; head_dim * n_batch];
    let mut fused_scores = vec![0.0f32; n_kv];
    let mut fused_kv_scratch = vec![0.0f32; head_dim];
    unsafe {
        olorin::kernels::ffi_inference::attn_fused_batched(
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            fused_out.as_mut_ptr(),
            fused_scores.as_mut_ptr(),
            fused_kv_scratch.as_mut_ptr(),
            head_dim as i32,
            stride_kv as i32,
            kv_head_offset,
            n_kv as i32,
            n_batch as i32,
            cache_start as i32,
            attn_scale,
        );
    }

    let mut mismatches = 0;
    for b in 0..n_batch {
        for i in 0..head_dim {
            let idx = b * head_dim + i;
            if ref_out[idx].to_bits() != fused_out[idx].to_bits() {
                if mismatches < 5 {
                    eprintln!("MISMATCH batch={b} dim={i}: ref={} fused={}", ref_out[idx], fused_out[idx]);
                }
                mismatches += 1;
            }
        }
    }
    assert_eq!(mismatches, 0, "N=4 batched attention: {mismatches}/{} mismatches", head_dim * n_batch);
    eprintln!("PASS: N=4 batched causal attention bit-exact ({} outputs)", head_dim * n_batch);
}
```

- [ ] **Step 3: Run the tests**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test attn_fused_batched -- --nocapture 2>&1 | tail -15
```

Expected: 2 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/attn_fused_batched.rs
git commit -m "test: fused batched attention — bit-exact vs per-position loop (N=1, N=4)"
```

---

## Task 6: Add batched buffers to Gemma4State

**Goal:** Add column-major `[dim, max_batch]` activation buffers for the batched forward path. Also add `store_batch` and `advance_n` to KV cache.

**Files:**
- Modify: `src/inference/forward.rs` — add fields to `Gemma4State`, allocate in `new()`
- Modify: `src/inference/cache.rs` — add `store_batch` and `advance_n`

- [ ] **Step 1: Add batch buffer fields to Gemma4State**

In `src/inference/forward.rs`, add these fields to the `Gemma4State` struct (after the existing single-token buffers, before `cache`):

```rust
    // ── Batched forward buffers (column-major [dim, max_batch]) ──
    pub(crate) batch_x: Vec<f32>,
    pub(crate) batch_x_norm: Vec<f32>,
    pub(crate) batch_q: Vec<f32>,
    pub(crate) batch_k: Vec<f32>,
    pub(crate) batch_v: Vec<f32>,
    pub(crate) batch_attn_out: Vec<f32>,
    pub(crate) batch_wo_out: Vec<f32>,
    pub(crate) batch_attn_res: Vec<f32>,
    pub(crate) batch_gate: Vec<f32>,
    pub(crate) batch_up: Vec<f32>,
    pub(crate) batch_down: Vec<f32>,
    pub(crate) batch_q8_qs: Vec<i8>,
    pub(crate) batch_q8_d: Vec<f32>,
    pub(crate) batch_q8_bsums: Vec<i16>,
    pub(crate) batch_ffn_q8_qs: Vec<i8>,
    pub(crate) batch_ffn_q8_d: Vec<f32>,
    pub(crate) batch_ffn_q8_bsums: Vec<i16>,
    // Q8K repacked A-side for gemm (block_q8_Kx4 tiles)
    pub(crate) batch_q8_a: Vec<u8>,
    pub(crate) batch_ffn_q8_a: Vec<u8>,
    // Per-token PLE signals for batched forward
    pub(crate) batch_ple_signal: Vec<f32>,
    pub(crate) gemm_scratch: Vec<u8>,
    pub(crate) max_batch: usize,
```

- [ ] **Step 2: Allocate in Gemma4State::new()**

Add allocation after the existing buffers. Use `max_batch = 512` (Gemma 4 sliding window cap).

```rust
        let max_batch = 512usize;
        let max_q8_dim = max_qkv.max(hd).max(max_ffn);
        let nb_max = max_q8_dim / 256;
        let block_q8_kx4_size = nb_max * 1168;
        let q8_a_groups = (max_batch + 3) / 4;
```

Then in the `Self { ... }`:

```rust
            batch_x: vec![0.0; hd * max_batch],
            batch_x_norm: vec![0.0; hd * max_batch],
            batch_q: vec![0.0; max_qkv * max_batch],
            batch_k: vec![0.0; max_kv * max_batch],
            batch_v: vec![0.0; max_kv * max_batch],
            batch_attn_out: vec![0.0; max_qkv * max_batch],
            batch_wo_out: vec![0.0; hd * max_batch],
            batch_attn_res: vec![0.0; hd * max_batch],
            batch_gate: vec![0.0; max_ffn * max_batch],
            batch_up: vec![0.0; max_ffn * max_batch],
            batch_down: vec![0.0; hd * max_batch],
            batch_q8_qs: vec![0; (max_q8_dim + 12) * max_batch],
            batch_q8_d: vec![0.0; nb_max * max_batch],
            batch_q8_bsums: vec![0; nb_max * 16 * max_batch],
            batch_ffn_q8_qs: vec![0; (max_ffn + 12) * max_batch],
            batch_ffn_q8_d: vec![0.0; n_blocks_ffn * max_batch],
            batch_ffn_q8_bsums: vec![0; n_blocks_ffn * 16 * max_batch],
            batch_q8_a: vec![0; q8_a_groups * block_q8_kx4_size],
            batch_ffn_q8_a: vec![0; q8_a_groups * block_q8_kx4_size],
            batch_ple_signal: vec![0.0; model.ple_dim * model.n_layers * max_batch],
            gemm_scratch: vec![0; 128],
            max_batch,
```

- [ ] **Step 3: Add KV cache batch methods**

In `src/inference/cache.rs`, add after `store()`:

```rust
    /// Store N tokens' K and V into the cache at sequential positions starting from seq_len.
    /// Caller must ensure k_batch/v_batch are [stride, N] column-major.
    pub fn store_batch(&mut self, layer: usize, k_batch: &[f32], v_batch: &[f32], n: usize) {
        if self.shared_source[layer].is_some() {
            return;
        }
        let hd = self.head_dim_v[layer];
        let stride = self.n_kv_heads * hd;
        debug_assert_eq!(k_batch.len(), stride * n);
        debug_assert_eq!(v_batch.len(), stride * n);

        let kb = &mut self.k[layer];
        let vb = &mut self.v[layer];

        for t in 0..n {
            let pos = match self.attn_types[layer] {
                AttnType::SlidingWindow => (self.seq_len + t) % self.window_size,
                AttnType::Global => self.seq_len + t,
            };
            let cache_off = pos * stride;
            let src_off = t * stride;
            for i in 0..stride {
                kb[cache_off + i] = f32_to_f16(k_batch[src_off + i]);
                vb[cache_off + i] = f32_to_f16(v_batch[src_off + i]);
            }
        }
    }

    /// Advance sequence position by N tokens.
    pub fn advance_n(&mut self, n: usize) {
        self.seq_len += n;
    }
```

- [ ] **Step 4: Build clean + run all gates**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "^error" | head -5
wc -l src/inference/forward.rs src/inference/cache.rs
```

Check forward.rs line count. If >500, split `Gemma4State` struct + `new()` into `src/inference/state.rs` before continuing.

- [ ] **Step 5: Commit**

```bash
git add src/inference/forward.rs src/inference/cache.rs
git commit -m "feat: batched activation buffers on Gemma4State + KV cache batch store"
```

---

## Task 7: Gemm dispatch helper

**Goal:** Write a helper function that takes N Q8K-quantized input columns, repacks them into block_q8_Kx4 tiles, and calls the gemm kernel. This encapsulates the q8k_repack_4 + gemm calling convention so forward_batch doesn't need to know the details.

**Files:**
- Create: `src/inference/matmul_batch.rs`
- Modify: `src/inference/mod.rs` — add `pub(crate) mod matmul_batch;`

- [ ] **Step 1: Write the gemm dispatch function**

Create `src/inference/matmul_batch.rs`:

```rust
//! Batched matmul dispatch: Q4K 8x8 gemm for N input columns.

use crate::kernels::ffi_inference;

/// Run Q4K 8x8 gemm: repacked_weights[nc, n_inner] × q8k_input[n_inner, N] → out[nc, N].
///
/// Input: N independently Q8K-quantized columns (qs, d, bsums arrays sized for N columns).
/// The function repacks the N Q8K columns into block_q8_Kx4 tiles (groups of 4 rows),
/// then calls the gemm kernel.
///
/// `q8_a_buf` is caller-owned scratch for the repacked A-side tiles.
/// `gemm_scratch` is caller-owned scratch (128 bytes).
///
/// # Safety
/// All pointers must be valid. nc must be multiple of 8. n_inner must be multiple of 256.
/// N must be multiple of 4 (pad with zeros if needed).
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_q4k_8x8(
    repacked_weights: *const u8,
    qs: &[i8],           // [n_inner_padded, N] — column k at qs[k * (n_inner + 12)..]
    d: &[f32],           // [nb, N] — column k at d[k * nb..]
    bsums: &[i16],       // [nb * 16, N] — column k at bsums[k * nb * 16..]
    q8_a_buf: &mut [u8], // scratch for repacked Q8K tiles
    gemm_scratch: &mut [u8],
    out: *mut f32,       // [nc, N] row-major output
    n_inner: usize,
    nc: usize,
    n: usize,            // batch size (must be multiple of 4)
) {
    let nb = n_inner / 256;
    let qs_stride = n_inner + 12;  // Q8K qs has 12-byte padding
    let block_q8_kx4_size = nb * 1168;

    // Repack Q8K inputs into groups of 4
    for group in 0..(n / 4) {
        let r0 = group * 4;
        // Interleave d values: [d_r0_b0, d_r1_b0, d_r2_b0, d_r3_b0, d_r0_b1, ...]
        // Allocate on stack (nb <= 24 for max_dim=6144, so max 96 floats = 384 bytes)
        let mut row_d = vec![0.0f32; nb * 4];
        for b in 0..nb {
            for r in 0..4 {
                row_d[b * 4 + r] = d[(r0 + r) * nb + b];
            }
        }
        let dst_off = group * block_q8_kx4_size;
        ffi_inference::q8k_repack_4(
            qs[(r0) * qs_stride..].as_ptr(),
            qs[(r0 + 1) * qs_stride..].as_ptr(),
            qs[(r0 + 2) * qs_stride..].as_ptr(),
            qs[(r0 + 3) * qs_stride..].as_ptr(),
            row_d.as_ptr(),
            bsums[(r0) * nb * 16..].as_ptr(),
            bsums[(r0 + 1) * nb * 16..].as_ptr(),
            bsums[(r0 + 2) * nb * 16..].as_ptr(),
            bsums[(r0 + 3) * nb * 16..].as_ptr(),
            q8_a_buf[dst_off..].as_mut_ptr(),
            nb as i32,
        );
    }

    // Call gemm
    ffi_inference::q4k_8x8_q8k_gemm(
        repacked_weights,
        q8_a_buf.as_ptr(),
        gemm_scratch.as_mut_ptr(),
        out,
        nc as i32,  // bs = row stride
        n_inner as i32,
        n as i32,    // nr = batch rows
        nc as i32,   // nc = output cols
    );
}
```

- [ ] **Step 2: Add module**

In `src/inference/mod.rs`, add:

```rust
pub(crate) mod matmul_batch;
```

- [ ] **Step 3: Build clean**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "^error" | head -5
```

- [ ] **Step 4: Commit**

```bash
git add src/inference/matmul_batch.rs src/inference/mod.rs
git commit -m "feat: gemm_q4k_8x8 dispatch helper — repack + call in one function"
```

---

## Task 8: forward_batch — the unified forward function

**Goal:** Write the core `forward_batch` function that processes N tokens through all layers using gemm for Q4K matmuls and the fused attention kernel. Supporting ops (rmsnorm, rope, quant) loop N times over existing single-token kernels.

**Files:**
- Create: `src/inference/forward_batch.rs`
- Modify: `src/inference/mod.rs` — add `pub mod forward_batch;`

- [ ] **Step 1: Write forward_batch skeleton**

Create `src/inference/forward_batch.rs`. This is a large task — the function mirrors `forward_one_inner` in `forward_graph.rs` but operates on `[dim, N]` column-major buffers.

```rust
//! Unified batched forward pass — gemm for all Q4K matmuls, any N.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::{compute_rope_tables, Gemma4State};
use crate::inference::matmul;
use crate::inference::matmul_batch;
use crate::inference::matmul_graph;
use crate::inference::dequant;
use crate::kernels::ffi_inference;
use crate::inference::threadpool::SpinBarrier;

/// Entry point: process N tokens. Returns slice of logits for the last token.
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

    // ── Pre-loop: embed all N tokens + PLE (thread 0) ────────
    if ith == 0 {
        let embed_scale = (hd as f32).sqrt();
        for t in 0..n {
            let dst = &mut state.batch_x[t * hd..(t + 1) * hd];
            dequant::q6k_embed_lookup(model.embed_weight, tokens[t] as usize, dst, hd);
            ffi_inference::vec_scale_f32(
                dst.as_ptr(), dst.as_mut_ptr(), embed_scale, hd as i32,
            );
        }
        // PLE: prepare for each token.
        // prepare_ple uses self.x (embedding) as input and writes to self.ple_signal.
        // For batched, we need per-token PLE signals stored in batch_ple_signal.
        // Process each token: copy its embedding into self.x, call prepare_ple,
        // then copy the resulting ple_signal into the per-token slot.
        let ple_total = model.ple_dim * model.n_layers;
        for t in 0..n {
            // prepare_ple reads self.x for the BF16 projection
            state.x[..hd].copy_from_slice(&state.batch_x[t * hd..(t + 1) * hd]);
            state.prepare_ple(model, tokens[t]);
            // Save this token's ple_signal into batch_ple_signal[t]
            state.batch_ple_signal[t * ple_total..(t + 1) * ple_total]
                .copy_from_slice(&state.ple_signal[..ple_total]);
        }
    }
    barrier.wait();

    // ── Per-layer loop ───────────────────────────────────────
    for il in 0..model.n_layers {
        layer_forward_batch(state, model, il, n, barrier, current_chunk, ith, nth);
    }

    // ── Post-loop: final norm + output matmul (last token only) ──
    if ith == 0 {
        // Only need logits for the last token
        let last = n - 1;
        let x_last = &state.batch_x[last * hd..(last + 1) * hd];
        ffi_inference::gemma4_rmsnorm(
            x_last.as_ptr(),
            model.norm_weight,
            state.x_norm.as_mut_ptr(),
            hd as i32,
            model.rms_eps,
        );
        matmul::quant_input(
            &state.x_norm,
            &mut state.q8_qs,
            &mut state.q8_d,
            &mut state.q8_bsums,
        );
    }
    barrier.wait();

    // Output matmul — Q6K, stays matvec (not repacked)
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

- [ ] **Step 2: Write per-layer function**

Add `layer_forward_batch` to the same file. This mirrors `layer_forward_graph` but with batched buffers.

The key differences from `forward_graph.rs`:
- **Matmuls use gemm** via `matmul_batch::gemm_q4k_8x8` instead of `matvec_step`
- **Q8K quant loops N times** using existing `matmul::quant_input`
- **Attention calls fused kernel** dispatched per-head across threads
- **Supporting ops loop N** using existing single-token kernels

```rust
fn layer_forward_batch(
    state: &mut Gemma4State,
    model: &Gemma4Model,
    il: usize,
    n: usize,
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
    let ffn_dim = model.ffn_dim[il];
    let qkv_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let kv_dim_v = n_kv_heads * head_dim_v;

    // Round N up to multiple of 4 for gemm
    let n_pad = (n + 3) & !3;

    // ── 1. RoPE tables + norm + quant (thread 0) ─────────────
    if ith == 0 {
        let n_rot = if model.is_swa[il] { model.rope_dim_swa } else { model.rope_dim_global };
        let rope_theta = if model.is_swa[il] { model.rope_theta_swa } else { model.rope_theta_global };
        let freq_factors = if !model.is_swa[il] { model.rope_freqs.as_deref() } else { None };
        let pos = state.cache.seq_len();

        // Norm + quant each token
        for t in 0..n {
            let x_t = &state.batch_x[t * hd..(t + 1) * hd];
            let xn_t = &mut state.batch_x_norm[t * hd..(t + 1) * hd];
            ffi_inference::gemma4_rmsnorm(
                x_t.as_ptr(), lw.attn_norm, xn_t.as_mut_ptr(), hd as i32, model.rms_eps,
            );

            let nb = hd / 256;
            let qs_stride = hd + 12;
            matmul::quant_input(
                xn_t,
                &mut state.batch_q8_qs[t * qs_stride..(t + 1) * qs_stride],
                &mut state.batch_q8_d[t * nb..(t + 1) * nb],
                &mut state.batch_q8_bsums[t * nb * 16..(t + 1) * nb * 16],
            );

            // Compute RoPE tables for each position
            compute_rope_tables(
                &mut state.cos_table, &mut state.sin_table,
                pos + t, n_rot, rope_theta, freq_factors,
            );
            // Store per-token RoPE tables into a batch scratch if needed,
            // or apply RoPE inline after Q/K projection (see step below)
        }
    }
    barrier.wait();

    // ── 2. Q projection: gemm(wq_repacked, q8k_input, N) ────
    // Thread 0 does the Q8K repack + gemm call, all threads participate via gemm's internal parallelism
    // NOTE: The gemm kernel is NOT yet work-stealing — it runs single-threaded.
    // For now, thread 0 calls it; future optimization adds WS gemm.
    if ith == 0 {
        unsafe {
            matmul_batch::gemm_q4k_8x8(
                lw.wq_repacked.as_deref().unwrap().as_ptr(),
                &state.batch_q8_qs, &state.batch_q8_d, &state.batch_q8_bsums,
                &mut state.batch_q8_a, &mut state.gemm_scratch,
                state.batch_q.as_mut_ptr(),
                hd, qkv_dim, n_pad,
            );
        }
    }
    barrier.wait();

    // ── 3. K, V projections (gemm) ───────────────────────────
    if ith == 0 && has_kv {
        unsafe {
            matmul_batch::gemm_q4k_8x8(
                lw.wk_repacked.as_deref().unwrap().as_ptr(),
                &state.batch_q8_qs, &state.batch_q8_d, &state.batch_q8_bsums,
                &mut state.batch_q8_a, &mut state.gemm_scratch,
                state.batch_k.as_mut_ptr(),
                hd, kv_dim, n_pad,
            );
            matmul_batch::gemm_q4k_8x8(
                lw.wv_repacked.as_deref().unwrap().as_ptr(),
                &state.batch_q8_qs, &state.batch_q8_d, &state.batch_q8_bsums,
                &mut state.batch_q8_a, &mut state.gemm_scratch,
                state.batch_v.as_mut_ptr(),
                hd, kv_dim_v, n_pad,
            );
        }
    }
    barrier.wait();

    // ── 4. Per-head Q norm + RoPE, K norm + RoPE (thread 0) ─
    if ith == 0 {
        let pos_base = state.cache.seq_len();
        for t in 0..n {
            // Q norm + RoPE per head
            for h in 0..n_heads {
                let q_off = t * qkv_dim + h * head_dim;
                ffi_inference::gemma4_rmsnorm(
                    state.batch_q[q_off..].as_ptr(),
                    lw.q_norm,
                    state.batch_q[q_off..].as_mut_ptr(),
                    head_dim as i32,
                    model.rms_eps,
                );
            }
            // Recompute RoPE tables for this token's position
            let n_rot = if model.is_swa[il] { model.rope_dim_swa } else { model.rope_dim_global };
            let rope_theta = if model.is_swa[il] { model.rope_theta_swa } else { model.rope_theta_global };
            let freq_factors = if !model.is_swa[il] { model.rope_freqs.as_deref() } else { None };
            compute_rope_tables(
                &mut state.cos_table, &mut state.sin_table,
                pos_base + t, n_rot, rope_theta, freq_factors,
            );
            // Apply RoPE to each Q head
            for h in 0..n_heads {
                let q_off = t * qkv_dim + h * head_dim;
                ffi_inference::rope_apply(
                    state.batch_q[q_off..].as_ptr(),
                    state.cos_table.as_ptr(),
                    state.sin_table.as_ptr(),
                    state.batch_q[q_off..].as_mut_ptr(),
                    (head_dim / 2) as i32,
                );
            }

            if has_kv {
                // K norm + RoPE per KV head
                for kh in 0..n_kv_heads {
                    let k_off = t * kv_dim + kh * head_dim;
                    ffi_inference::gemma4_rmsnorm(
                        state.batch_k[k_off..].as_ptr(),
                        lw.k_norm,
                        state.batch_k[k_off..].as_mut_ptr(),
                        head_dim as i32,
                        model.rms_eps,
                    );
                    ffi_inference::rope_apply(
                        state.batch_k[k_off..].as_ptr(),
                        state.cos_table.as_ptr(),
                        state.sin_table.as_ptr(),
                        state.batch_k[k_off..].as_mut_ptr(),
                        (head_dim / 2) as i32,
                    );
                }
            }
        }

        // ── 5. KV cache store (N positions) ──────────────────
        if has_kv {
            state.cache.store_batch(il, &state.batch_k[..kv_dim * n], &state.batch_v[..kv_dim_v * n], n);
        }
    }
    barrier.wait();

    // ── 6. Fused attention (heads split across threads) ──────
    {
        let attn_scale = 1.0f32;
        let cache_start = state.cache.seq_len() as i32; // seq_len before this batch was stored
        // Wait — cache was already advanced by store_batch conceptually, but advance_n
        // hasn't been called yet (that's at the end of forward_batch_inner).
        // So seq_len is still the pre-batch value. n_kv = seq_len + n.
        // Actually: store_batch writes at seq_len..seq_len+n but doesn't advance seq_len.
        // So cache_start = current seq_len (positions before this batch).
        // n_kv for attention = cache.attn_len(il) adjusted for the N new positions.
        // For global: n_kv = seq_len + n
        // For sliding: n_kv = min(seq_len + n, window_size)
        // But attn_len(il) returns seq_len + 1 (for single token). We need seq_len + n.
        // Compute it manually:
        let seq_len = state.cache.seq_len();
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

        // Split heads across threads
        let per = (n_heads + nth - 1) / nth;
        let h_start = ith * per;
        let h_end = ((ith + 1) * per).min(n_heads);

        for h in h_start..h_end {
            let kv_h = h / gqa_ratio;
            let kv_head_offset = (kv_h * head_dim) as i32;

            // Q for this head: batch_q column-major [qkv_dim, n]
            // Head h, token t is at batch_q[t * qkv_dim + h * head_dim]
            // But the fused kernel expects [head_dim, n_batch] contiguous.
            // We need to gather Q[h] for all N tokens into a contiguous buffer.
            // Use per-thread scratch in attn_scores (it's large enough: max_seq_len * n_thread_slots).
            let q_head_scratch = unsafe { state.batch_attn_out.as_mut_ptr().add(h * head_dim) };
            // Actually, batch_attn_out is the output buffer. We need separate scratch for Q gathering.
            // Use the per-thread kv_f32_scratch area... no, that's too small (head_dim only).
            // The simplest approach: the fused kernel takes a stride parameter for Q,
            // so it can read Q[t] at q_ptr + t * q_stride instead of t * head_dim.
            // This avoids the gather entirely.
            // BUT: the kernel signature in Task 2 uses contiguous [head_dim, n_batch].
            // Let's adjust: pass q_stride as a parameter instead.
            // ALTERNATIVE: gather into attn_out temporarily (it gets overwritten anyway).

            // Gather Q[h] for all tokens into a contiguous scratch
            let q_scratch = unsafe { state.kv_f32_scratch.as_mut_ptr().add(ith * kv_scratch_stride) };
            // kv_scratch_stride = max_head, but we need head_dim * n tokens.
            // This won't fit. We need a dedicated batch Q scratch per thread.
            // DECISION: add q_stride parameter to the fused kernel, read Q non-contiguously.
            // The kernel reads q[i * q_stride + 0..head_dim] instead of q[i * head_dim].
            // This is a one-line change to the kernel.

            // For now, call with stride = qkv_dim (the Q buffer has all heads interleaved)
            let q_head_ptr = unsafe { state.batch_q.as_ptr().add(h * head_dim) };

            let scores_buf = unsafe { state.attn_scores.as_mut_ptr().add(ith * attn_scores_stride) };
            let kv_scratch = unsafe { state.kv_f32_scratch.as_mut_ptr().add(ith * kv_scratch_stride) };

            unsafe {
                ffi_inference::attn_fused_batched(
                    q_head_ptr,
                    k_ptr,
                    v_ptr,
                    state.batch_attn_out.as_mut_ptr().add(h * head_dim),
                    scores_buf,
                    kv_scratch,
                    head_dim as i32,
                    stride_kv as i32,
                    kv_head_offset,
                    n_kv as i32,
                    n as i32,
                    seq_len as i32,
                    attn_scale,
                );
            }
        }
    }
    barrier.wait();

    // ── 7. Quant attn_out + Wo gemm (thread 0) ──────────────
    if ith == 0 {
        let attn_out_dim = n_heads * head_dim;
        let nb = attn_out_dim / 256;
        let qs_stride = attn_out_dim + 12;
        for t in 0..n {
            matmul::quant_input(
                &state.batch_attn_out[t * attn_out_dim..(t + 1) * attn_out_dim],
                &mut state.batch_q8_qs[t * qs_stride..(t + 1) * qs_stride],
                &mut state.batch_q8_d[t * nb..(t + 1) * nb],
                &mut state.batch_q8_bsums[t * nb * 16..(t + 1) * nb * 16],
            );
        }
        unsafe {
            matmul_batch::gemm_q4k_8x8(
                lw.wo_repacked.as_deref().unwrap().as_ptr(),
                &state.batch_q8_qs, &state.batch_q8_d, &state.batch_q8_bsums,
                &mut state.batch_q8_a, &mut state.gemm_scratch,
                state.batch_wo_out.as_mut_ptr(),
                attn_out_dim, hd, n_pad,
            );
        }
    }
    barrier.wait();

    // ── 8. Post-attn norm + residual (thread 0) ─────────────
    if ith == 0 {
        for t in 0..n {
            let wo_t = &state.batch_wo_out[t * hd..(t + 1) * hd];
            let x_t = &state.batch_x[t * hd..(t + 1) * hd];
            let res_t = &mut state.batch_attn_res[t * hd..(t + 1) * hd];
            if !lw.post_attn_norm.is_null() {
                ffi_inference::gemma4_rmsnorm(
                    wo_t.as_ptr(), lw.post_attn_norm, state.x_norm.as_mut_ptr(),
                    hd as i32, model.rms_eps,
                );
                ffi_inference::vec_add_f32(
                    state.x_norm.as_ptr(), x_t.as_ptr(), res_t.as_mut_ptr(), hd as i32,
                );
            } else {
                ffi_inference::vec_add_f32(
                    wo_t.as_ptr(), x_t.as_ptr(), res_t.as_mut_ptr(), hd as i32,
                );
            }
        }
    }
    barrier.wait();

    // ── 9. FFN: norm + quant + gate/up gemm + gelu_mul + down gemm ──
    if ith == 0 {
        // FFN norm + quant
        for t in 0..n {
            let res_t = &state.batch_attn_res[t * hd..(t + 1) * hd];
            let xn_t = &mut state.batch_x_norm[t * hd..(t + 1) * hd];
            ffi_inference::gemma4_rmsnorm(
                res_t.as_ptr(), lw.ffn_norm, xn_t.as_mut_ptr(), hd as i32, model.rms_eps,
            );

            let nb = hd / 256;
            let qs_stride = hd + 12;
            matmul::quant_input(
                xn_t,
                &mut state.batch_ffn_q8_qs[t * qs_stride..(t + 1) * qs_stride],
                &mut state.batch_ffn_q8_d[t * nb..(t + 1) * nb],
                &mut state.batch_ffn_q8_bsums[t * nb * 16..(t + 1) * nb * 16],
            );
        }

        // Gate gemm
        unsafe {
            matmul_batch::gemm_q4k_8x8(
                lw.w_gate_repacked.as_deref().unwrap().as_ptr(),
                &state.batch_ffn_q8_qs, &state.batch_ffn_q8_d, &state.batch_ffn_q8_bsums,
                &mut state.batch_ffn_q8_a, &mut state.gemm_scratch,
                state.batch_gate.as_mut_ptr(),
                hd, ffn_dim, n_pad,
            );
        }

        // Up gemm
        unsafe {
            matmul_batch::gemm_q4k_8x8(
                lw.w_up_repacked.as_deref().unwrap().as_ptr(),
                &state.batch_ffn_q8_qs, &state.batch_ffn_q8_d, &state.batch_ffn_q8_bsums,
                &mut state.batch_ffn_q8_a, &mut state.gemm_scratch,
                state.batch_up.as_mut_ptr(),
                hd, ffn_dim, n_pad,
            );
        }

        // GELU(gate) * up per token
        for t in 0..n {
            ffi_inference::gelu_mul(
                state.batch_gate[t * ffn_dim..].as_ptr(),
                state.batch_up[t * ffn_dim..].as_ptr(),
                state.batch_gate[t * ffn_dim..].as_mut_ptr(),
                ffn_dim as i32,
            );
        }

        // Quant GELU output for down projection
        let nb_ffn = ffn_dim / 256;
        let qs_stride_ffn = ffn_dim + 12;
        for t in 0..n {
            matmul::quant_input(
                &state.batch_gate[t * ffn_dim..(t + 1) * ffn_dim],
                &mut state.batch_ffn_q8_qs[t * qs_stride_ffn..(t + 1) * qs_stride_ffn],
                &mut state.batch_ffn_q8_d[t * nb_ffn..(t + 1) * nb_ffn],
                &mut state.batch_ffn_q8_bsums[t * nb_ffn * 16..(t + 1) * nb_ffn * 16],
            );
        }

        // Down gemm
        unsafe {
            matmul_batch::gemm_q4k_8x8(
                lw.w_down_repacked.as_deref().unwrap().as_ptr(),
                &state.batch_ffn_q8_qs, &state.batch_ffn_q8_d, &state.batch_ffn_q8_bsums,
                &mut state.batch_ffn_q8_a, &mut state.gemm_scratch,
                state.batch_down.as_mut_ptr(),
                ffn_dim, hd, n_pad,
            );
        }
    }
    barrier.wait();

    // ── 10. Post-FFN norm + residual + PLE + scale (thread 0) ──
    if ith == 0 {
        for t in 0..n {
            let down_t = &state.batch_down[t * hd..(t + 1) * hd];
            let res_t = &state.batch_attn_res[t * hd..(t + 1) * hd];
            let x_t = &mut state.batch_x[t * hd..(t + 1) * hd];

            if !lw.post_ffn_norm.is_null() {
                ffi_inference::gemma4_rmsnorm(
                    down_t.as_ptr(), lw.post_ffn_norm, state.x_norm.as_mut_ptr(),
                    hd as i32, model.rms_eps,
                );
                ffi_inference::vec_add_f32(
                    state.x_norm.as_ptr(), res_t.as_ptr(), x_t.as_mut_ptr(), hd as i32,
                );
            } else {
                ffi_inference::vec_add_f32(
                    down_t.as_ptr(), res_t.as_ptr(), x_t.as_mut_ptr(), hd as i32,
                );
            }

            // PLE — uses per-token signal from batch_ple_signal
            if model.ple_dim > 0 && !lw.inp_gate.is_null() && !lw.proj.is_null() {
                let ple_dim = model.ple_dim;
                let ple_total = ple_dim * model.n_layers;
                let ple_off = il * ple_dim;

                matmul::quant_input(x_t, &mut state.q8_qs, &mut state.q8_d, &mut state.q8_bsums);
                matmul::matvec(
                    lw.inp_gate_dtype, lw.inp_gate,
                    &state.q8_qs, &state.q8_d, &state.q8_bsums,
                    &mut state.ple_gate, &mut state.q6k_d_scratch, ple_dim, hd,
                );
                // Use this token's PLE signal from batch_ple_signal
                let sig_off = t * ple_total + ple_off;
                ffi_inference::gelu_mul(
                    state.ple_gate.as_ptr(),
                    state.batch_ple_signal[sig_off..].as_ptr(),
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
                        state.ple_out.as_ptr(), lw.post_norm, state.ple_out.as_mut_ptr(),
                        hd as i32, model.rms_eps,
                    );
                }
                ffi_inference::vec_add_f32(
                    x_t.as_ptr(), state.ple_out.as_ptr(), x_t.as_mut_ptr(), hd as i32,
                );
            }

            // Layer output scale
            let out_scale = lw.layer_output_scale;
            if out_scale != 1.0 {
                ffi_inference::vec_scale_f32(
                    x_t.as_ptr(), x_t.as_mut_ptr(), out_scale, hd as i32,
                );
            }
        }
    }
    barrier.wait();
}
```

**CRITICAL DESIGN NOTE on attention Q stride:** The fused attention kernel needs a `q_stride` parameter (not just `head_dim`) because in the batch_q buffer, head h of token t is at offset `t * qkv_dim + h * head_dim`. The stride between consecutive tokens for the same head is `qkv_dim`, not `head_dim`. **Update the kernel signature in Task 2 to include `q_stride: i32` and `out_stride: i32` parameters.** The same applies to the output buffer `batch_attn_out`.

Alternatively, gather Q into contiguous scratch before the kernel call and scatter output after. The stride approach is more efficient (no copy). **Choose stride approach — add `q_stride` and `out_stride` parameters to the kernel.**

- [ ] **Step 3: Build clean**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "^error" | head -10
wc -l src/inference/forward_batch.rs
```

If forward_batch.rs is over 500 lines, split: keep `forward_batch_inner` and post-loop in the main file, move `layer_forward_batch` to `src/inference/forward_batch_layer.rs`.

- [ ] **Step 4: Commit**

```bash
git add src/inference/forward_batch.rs src/inference/mod.rs
git commit -m "feat: forward_batch — unified gemm forward path for any N"
```

---

## Task 9: Wire forward_batch into GraphPool dispatch

**Goal:** Add a public `forward_batch` method on `Gemma4State` that uses the GraphPool (work-stealing) to call `forward_batch_inner`, matching the pattern of existing `forward_one_graph`.

**Files:**
- Modify: `src/inference/forward.rs` — add `pub fn forward_batch()` method on `Gemma4State`

- [ ] **Step 1: Find how forward_one_graph dispatches**

Read `src/inference/forward.rs` to find the `forward_one_graph` method — it uses `GraphPool::run` to dispatch `forward_one_inner` across threads.

- [ ] **Step 2: Add forward_batch method**

Add to `Gemma4State` impl block:

```rust
    /// Batched forward pass. Processes N tokens using gemm for all Q4K matmuls.
    /// Returns logits for the last token.
    pub fn forward_batch(
        &mut self,
        model: &Gemma4Model,
        tokens: &[u32],
        pool: &crate::inference::threadpool::GraphPool,
    ) -> &[f32] {
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= self.max_batch);
        let state_ptr = self as *mut Gemma4State;
        let model_ptr = model as *const Gemma4Model;
        pool.run(|barrier, current_chunk, ith, nth| {
            let state = unsafe { &mut *state_ptr };
            let model = unsafe { &*model_ptr };
            crate::inference::forward_batch::forward_batch_inner(
                state, model, tokens, barrier, current_chunk, ith, nth,
            );
        });
        &self.logits
    }
```

- [ ] **Step 3: Build clean**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "^error" | head -5
```

- [ ] **Step 4: Commit**

```bash
git add src/inference/forward.rs
git commit -m "feat: Gemma4State::forward_batch() — public API for batched forward"
```

---

## Task 10: N=1 bit-exact test — forward_batch vs forward_one_graph

**Goal:** Prove that `forward_batch(&[BOS])` produces identical logits to `forward_one_graph(BOS)`. This is the critical correctness gate: the new path must match the existing decode path exactly for N=1.

**Files:**
- Create: `tests/forward_batch_verify.rs`

- [ ] **Step 1: Write the N=1 comparison test**

```rust
//! Verify forward_batch produces identical output to forward_one_graph.

#[test]
fn forward_batch_n1_matches_forward_one_graph() {
    olorin::kernels::ffi::init().unwrap();

    let model_path = std::path::Path::new(&std::env::var("HOME").unwrap())
        .join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !model_path.exists() {
        eprintln!("SKIP: model not found at {}", model_path.display());
        return;
    }

    let model = olorin::inference::engine::Gemma4Model::from_gguf(&model_path).unwrap();
    let pool = olorin::inference::threadpool::GraphPool::new(
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
    );
    let max_seq = 2048;

    // Path A: forward_one_graph with BOS
    let mut state_a = olorin::inference::forward::Gemma4State::new(&model, max_seq, &pool);
    let logits_a = state_a.forward_one_graph(&model, model.bos_id(), &pool).to_vec();

    // Path B: forward_batch with &[BOS]
    let mut state_b = olorin::inference::forward::Gemma4State::new(&model, max_seq, &pool);
    let logits_b = state_b.forward_batch(&model, &[model.bos_id()], &pool).to_vec();

    // Compare bit-exact
    assert_eq!(logits_a.len(), logits_b.len());
    let mut mismatches = 0;
    for i in 0..logits_a.len() {
        if logits_a[i].to_bits() != logits_b[i].to_bits() {
            if mismatches < 10 {
                eprintln!("MISMATCH logit[{i}]: graph={} batch={}", logits_a[i], logits_b[i]);
            }
            mismatches += 1;
        }
    }

    if mismatches > 0 {
        // If not bit-exact, check L2 distance
        let l2: f32 = logits_a.iter().zip(&logits_b).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        eprintln!("L2 distance: {l2:.6} ({mismatches} mismatches out of {} logits)", logits_a.len());
    }
    assert_eq!(mismatches, 0, "forward_batch(N=1) must be bit-exact with forward_one_graph");
    eprintln!("PASS: forward_batch(N=1) bit-exact match ({} logits)", logits_a.len());
}
```

- [ ] **Step 2: Run the test**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test forward_batch_verify -- --nocapture 2>&1 | tail -20
```

If this fails, debug: add per-layer L2-norm prints in both paths and find the first layer where they diverge. The most likely sources of divergence are:
- Gemm vs matvec producing different accumulation order → investigate and fix
- RoPE tables computed at wrong position for batched path
- Attention kernel not matching existing loop exactly

**Do NOT proceed past this task until the test passes.** This is the critical N=1 equivalence gate.

- [ ] **Step 3: Run existing regression tests**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression 2>&1 | tail -6
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -15
```

Both must still pass — the existing forward_one_graph path is untouched.

- [ ] **Step 4: Commit**

```bash
git add tests/forward_batch_verify.rs
git commit -m "test: forward_batch(N=1) bit-exact vs forward_one_graph — critical gate"
```

---

## Task 11: Wire generate.rs to use forward_batch

**Goal:** Switch `generate.rs` to use `forward_batch` for both prefill and decode.

**Files:**
- Modify: `src/inference/generate.rs`

- [ ] **Step 1: Read current generate.rs**

```bash
grep -n "forward_one_graph\|forward_batch" src/inference/generate.rs
```

- [ ] **Step 2: Replace prefill loop with single forward_batch call**

Change the prefill section (currently lines 104-112):

```rust
        // 4. Prefill: batched forward pass
        let mut logits_snapshot = {
            let logits = self.state.forward_batch(&self.model, &tokens, &self.graph_pool);
            logits.to_vec()
        };
```

- [ ] **Step 3: Replace decode forward_one_graph with forward_batch(&[tok])**

Change the decode loop (currently line 140):

```rust
            let logits = self.state.forward_batch(&self.model, &[token_id], &self.graph_pool);
```

- [ ] **Step 4: Run end-to-end smoke test**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_smoke -- --nocapture 2>&1 | tail -20
```

- [ ] **Step 5: Run all existing tests**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release 2>&1 | tail -30
```

Everything must pass.

- [ ] **Step 6: Commit**

```bash
git add src/inference/generate.rs
git commit -m "feat: generate.rs uses forward_batch for prefill and decode"
```

---

## Task 12: Benchmark — prefill speedup

**Goal:** Measure the prefill throughput improvement. Update `bench_decode_speed.rs` to time prompt eval separately.

**Files:**
- Modify: `tests/bench_decode_speed.rs`

- [ ] **Step 1: Read the current bench**

```bash
grep -n "forward_one_graph\|prompt\|prefill\|forward_batch" tests/bench_decode_speed.rs
```

- [ ] **Step 2: Update to use forward_batch + separate timing**

Replace the prompt-eval section to use `forward_batch`:

```rust
        // Prefill — batched
        let t0 = std::time::Instant::now();
        let _ = state.forward_batch(&model, &prompt_tokens, &pool);
        let t_prompt = t0.elapsed();
        let prompt_tps = prompt_tokens.len() as f64 / t_prompt.as_secs_f64();
```

Keep the decode section using `forward_batch(&[tok])`:

```rust
        // Decode
        let t1 = std::time::Instant::now();
        for _ in 0..n_decode {
            let _ = state.forward_batch(&model, &[next_tok], &pool);
            next_tok = /* sample from logits */;
        }
        let t_decode = t1.elapsed();
```

Print both:
```
prompt eval:  {t_prompt:.2?} / {n_prompt} tok  ({prompt_tps:.2} t/s)
decode:       {t_decode:.2?} / {n_decode} tok  ({decode_tps:.2} t/s)
```

- [ ] **Step 3: Run the bench**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test bench_decode_speed -- --nocapture 2>&1 | tail -20
```

Record the numbers. Minimum acceptable prompt-eval: 15 t/s on x86 workstation (was ~8.5 t/s single-token).

- [ ] **Step 4: Commit**

```bash
git add tests/bench_decode_speed.rs
git commit -m "bench: separate prefill/decode timing via forward_batch

prompt eval:  [XX.XX] t/s  (was ~8.5 t/s token-by-token)
decode:       [XX.XX] t/s  (was ~8.7 t/s)"
```

---

## Task 13: Delete Path A dead code

**Goal:** Remove `forward_one`, `forward_attn.rs`, `forward_attn_heads.rs`, and any matmul wrappers only used by Path A.

**Files:**
- Delete or modify: `src/inference/forward_attn.rs`
- Delete or modify: `src/inference/forward_attn_heads.rs`
- Modify: `src/inference/forward.rs` — remove `forward_one` method
- Modify: `src/inference/matmul.rs` — remove functions only called from Path A
- Modify: `src/inference/mod.rs` — remove dead module declarations

- [ ] **Step 1: Find all callers of forward_one**

```bash
grep -rn "forward_one\b" src/ tests/ --include="*.rs" | grep -v "forward_one_graph\|forward_one_inner\|forward_batch"
```

If anything outside `forward.rs` and `forward_attn.rs` still calls `forward_one`, do not delete it — update the caller first.

- [ ] **Step 2: Find dead matmul functions**

```bash
grep -rn "par_matvec_maybe_repacked\|par_q4k_matvec_dual_maybe_repacked\|par_matvec\b\|par_q4k_matvec\b" src/ tests/ --include="*.rs"
```

Any function only referenced from `forward_attn.rs` is dead once Path A is removed.

- [ ] **Step 3: Delete dead code**

- Remove `forward_one` from `src/inference/forward.rs`
- Remove `src/inference/forward_attn.rs` (or gut it if shared functions remain)
- Remove `src/inference/forward_attn_heads.rs` (check if `attention_decode` is used by forward_graph.rs — if so, keep it until forward_graph.rs is also removed)
- Remove dead `par_*` functions from `matmul.rs`
- Update `mod.rs`

- [ ] **Step 4: Build clean + run all tests**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "^error" | head -10
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release 2>&1 | tail -20
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete Path A (forward_one + ThreadPool forward path)

forward_batch replaces both Path A and Path B. All matmuls go through gemm."
```

---

## Task 14: Delete forward_one_graph (Path B)

**Goal:** Now that forward_batch handles both prefill and decode, remove the old `forward_one_graph` / `forward_one_inner` / `layer_forward_graph` code.

**Files:**
- Modify: `src/inference/forward_graph.rs` — delete `forward_one_inner` and `layer_forward_graph`
- Modify: `src/inference/forward.rs` — remove `forward_one_graph` method
- Clean up any remaining dead code in `matmul_graph.rs`

- [ ] **Step 1: Find all callers**

```bash
grep -rn "forward_one_graph\|forward_one_inner\|layer_forward_graph" src/ tests/ --include="*.rs"
```

All callers should now be gone (generate.rs switched in Task 11, tests switched or removed).

- [ ] **Step 2: Delete the code**

- Remove `forward_one_inner` and `layer_forward_graph` from `forward_graph.rs`
- Remove `forward_one_graph` from `forward.rs`
- Remove `matvec_step` helper from `forward_graph.rs` if only used by the deleted functions
- Check `matmul_graph.rs` — if `matvec_ws`, `q4k_matvec_8x8_ws`, `q4k_matvec_dual_8x8_ws` are no longer called, remove them
- Keep `matmul_graph.rs` functions that are still used (e.g., output matmul `matvec_ws` is called by forward_batch for Q6K embed)

- [ ] **Step 3: Build clean + run all tests**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "^(error|warning)" | head -10
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release 2>&1 | tail -20
```

- [ ] **Step 4: Regenerate parallel_regression snapshot**

The old snapshot was for `forward_one_graph(BOS)`. Now the test should use `forward_batch(&[BOS])`. Update the test and regenerate:

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression -- --nocapture 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete forward_one_graph (Path B) — forward_batch is the sole forward path"
```

---

## Task 15: Final regression sweep + line count check

**Goal:** Full test suite pass, line count check, working tree clean.

- [ ] **Step 1: Run all tests**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release 2>&1 | tail -30
```

All pass, no new warnings.

- [ ] **Step 2: Line count**

```bash
find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'
```

Only the two chacha kernels should appear.

- [ ] **Step 3: Clean working tree**

```bash
git status
git diff --stat
```

Nothing uncommitted.

- [ ] **Step 4: eabrain remember session results**

```bash
eabrain remember "Phase 2 Plan 2 COMPLETE. forward_batch replaces both Path A and Path B. Gemm everywhere for Q4K. Fused batched attention kernel (x86+ARM). Prefill: [XX] t/s (was 8.5). Decode: [XX] t/s (was 8.7). N=1 bit-exact with old path."
```

---

## Errata: Attention Kernel Q Stride

Task 2's kernel signature must include `q_stride` and `out_stride` parameters to avoid a gather/scatter copy in Task 8. Update the signature to:

```
export func attn_fused_batched(
    q: *f32,               // q[i] at q + i * q_stride, head data at offset 0..head_dim
    k_cache: *u16,
    v_cache: *u16,
    out dst: *mut f32,     // dst[i] at dst + i * out_stride
    scores_buf: *mut f32,
    kv_scratch: *mut f32,
    head_dim: i32,
    q_stride: i32,         // stride between consecutive query tokens (= n_heads * head_dim)
    out_stride: i32,       // stride between consecutive output tokens (= n_heads * head_dim)
    stride_kv: i32,
    kv_head_offset: i32,
    n_kv: i32,
    n_batch: i32,
    cache_start: i32,
    attn_scale: f32
)
```

The kernel reads `q[i]` at `q + i * q_stride` instead of `q + i * head_dim`. For N=1, both are equivalent. The FFI type alias and wrapper in Task 4 must match this updated signature.
