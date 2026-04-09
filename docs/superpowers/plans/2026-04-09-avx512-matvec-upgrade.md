# AVX-512 Matvec Kernel Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 3.4× decode speed gap vs llama.cpp on this EPYC AVX-512 box by upgrading the existing Eä Q4K/Q5K/Q6K dot-product kernels from 256-bit (AVX2) to 512-bit (AVX-512) SIMD width.

**Architecture:** No repacking, no batched gemm, no new data layouts. The existing `q4k_dot.ea` / `q4k_dot_q8k_4row` kernel structure stays — we double the SIMD width from `u8x32`/`i8x32`/`i32x8` to `u8x64`/`i8x64`/`i32x16`. Same algorithm, wider pipes. Each kernel gets an `_avx512` variant selected at runtime via the existing `load_best` mechanism in `ffi_inference.rs`.

**Tech Stack:** Eä compiler (eacompute), AVX-512BW intrinsics (`maddubs_i16(u8x64,i8x64)`, `madd_i16(i16x32,i16x32)`), existing Olorin thread pool dispatch.

**Why this works:** llama.cpp's decode path on this machine uses `ggml_vec_dot_q4_K_q8_K` with AVX-512 — the same per-row dot product, just at 512-bit width. No repacking involved. Our `q4k_dot.ea` already has the correct algorithm (verified bit-exact via tests); the only difference is SIMD width. The 4-row variant amortizes scale extraction across 4 output rows — same structure at 512-bit.

**Why NOT batched gemm:** llama.cpp's batched gemm (8x8 repack + 16×16 tiled kernel) is a separate optimization for prompt eval (N>1). It requires a different weight layout, different Q8K input format (`block_q8_Kx4`), and architecture-specific kernels (AVX-512 on x86, SVE+I8MM on ARM). Per-token decode does NOT use it. The decode gap is purely SIMD width.

**Target hardware:** This EPYC 9354P (2-core slice, AVX-512BW/DQ). The wider kernels are selected at runtime — ARM and non-AVX-512 x86 continue using the existing AVX2/NEON kernels unchanged.

**Bench baseline (this machine, 2 threads, Gemma 4 E2B Q4_K_M):**

| | olorin | llama.cpp | gap |
|---|---|---|---|
| decode | 3.17 t/s | 10.78 t/s | 3.4× |
| prefill | 3.41 t/s | 21.18 t/s | 6.2× |

**Expected outcome:** Decode gap halved or better (SIMD width alone is 2×; reduced loop iterations may add more). Prefill gap narrowed proportionally.

---

## File Structure

| File | Role | Change |
|---|---|---|
| `kernels/q4k_dot_avx512.ea` | Create | AVX-512 Q4K dot + 4row + 4row_dual |
| `kernels/q5k_dot_avx512.ea` | Create | AVX-512 Q5K dot + 4row |
| `kernels/q6k_dot_avx512.ea` | Create | AVX-512 Q6K dot + 4row |
| `src/kernels/ffi_inference.rs` | Modify | `load_best` for x86 AVX-512 detection |
| `src/kernels/ffi_inference_types.rs` | No change | Types are pointer-width-agnostic |
| `tests/bench_decode_speed.rs` | No change | Bench picks up faster kernels automatically |

The existing `q4k_dot.ea` (AVX2) stays untouched as the fallback. Runtime kernel selection (already used for ARM I8MM variants) picks the AVX-512 variant when `cpuid` confirms AVX-512BW support.

---

### Task 1: Scaffold q4k_dot_avx512.ea with single-row dot

**Files:**
- Create: `kernels/q4k_dot_avx512.ea`

The single-row `q4k_dot_q8k` is the simplest starting point. Port `q4k_dot.ea::q4k_dot_q8k` from `u8x32`/`i8x32`/`i32x8` to `u8x64`/`i8x64`/`i32x16`.

- [ ] **Step 1: Copy q4k_dot.ea to q4k_dot_avx512.ea**

```bash
cp kernels/q4k_dot.ea kernels/q4k_dot_avx512.ea
```

- [ ] **Step 2: Update the cfg guard and vector types**

Change `#[cfg(x86_64)]` at the top (keep it — AVX-512 is x86_64 only).

In `dot_block_vec`, change the vector types:
- `u8x32` → `u8x64` (loads 64 bytes at once instead of 32)
- `i8x32` → `i8x64`
- `i16x16` → `i16x32`
- `i32x8` → `i32x16`
- `mask_lo: u8x32 = splat(15)` → `mask_lo: u8x64 = splat(15)`
- `shift4: u8x32 = splat(4)` → `shift4: u8x64 = splat(4)`

The key change in `dot_block_vec`: each `load(q4, ...)` now loads 64 bytes (two sub-blocks at once). Each `load(q8, ...)` loads 64 Q8K values. The `maddubs_i16(u8x64, i8x64) → i16x32` and `madd_i16(i16x32, i16x32) → i32x16` calls process 2× the data per instruction.

The inner loop currently processes 4 sub-block pairs (8 sub-blocks total per Q4K super-block). At 512-bit width, each iteration handles 64 quants instead of 32 — so the loop body changes from processing one sub-block to processing two sub-blocks per maddubs call.

For the single-row dot, the structure becomes:
```
// Each maddubs now covers 64 bytes = 2 sub-blocks
// So 4 sub-block pairs become 2 iterations of 2-pair-at-a-time
let p_lo: u8x64 = load(q4, nib_base)           // 64 bytes = sub-blocks 0+1
let p_hi: u8x64 = load(q4, nib_base + 64)      // sub-blocks 2+3
// ... mask, shift, maddubs, madd, accumulate
```

The exact restructuring depends on the Q4K block layout — the nibble interleaving within 64-byte loads must produce correct pairs for maddubs. Verify by checking that the first 32 bytes of a 64-byte load correspond to the same data as the existing 32-byte load.

- [ ] **Step 3: Update `reduce_add` usage**

`reduce_add(i32x16)` returns a scalar i32. The existing `reduce_add(i32x8)` in the AVX2 path does the same. Verify eacompute supports `reduce_add` on `i32x16`:

```bash
grep "reduce_add.*i32x16\|reduce_add.*16" /root/dev/eacompute/src/typeck/intrinsics*.rs
```

If not present, it needs to be added to eacompute first (STOP and report).

- [ ] **Step 4: Build and verify kernel compiles**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add kernels/q4k_dot_avx512.ea
git commit -m "feat: q4k_dot_avx512.ea scaffold — single-row dot at 512-bit width"
```

---

### Task 2: Add 4-row variant to q4k_dot_avx512.ea

**Files:**
- Modify: `kernels/q4k_dot_avx512.ea`

Port `q4k_dot_q8k_4row` to 512-bit width. This is the hot path — `par_q4k_matvec` calls it for every group of 4 output rows.

- [ ] **Step 1: Port q4k_dot_q8k_4row**

Same type widening as Task 1. The 4-row variant shares the Q8K input across 4 weight rows and amortizes scale extraction + bsums pairing. The structure stays identical:

```
for each block:
    load Q4K scales for rows 0-3 (from 4 different weight row pointers)
    load Q8K data once (shared)
    for each row 0-3:
        dot_block_vec_512(q4_row[r], q8, scales[r]) → i32x16
        reduce_add → scalar i32
        fma into per-row f32 accumulator
    row_mins for each row
```

The only change: `dot_block_vec` operates on `u8x64`/`i8x64` instead of `u8x32`/`i8x32`, and `reduce_add` takes `i32x16` instead of `i32x8`.

- [ ] **Step 2: Port q4k_dot_q8k_4row_dual (gate+up fused)**

Same widening for the dual variant that processes gate and up weights simultaneously.

- [ ] **Step 3: Build**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add kernels/q4k_dot_avx512.ea
git commit -m "feat: q4k_dot_avx512 4row + 4row_dual at 512-bit width"
```

---

### Task 3: Wire AVX-512 kernel selection in ffi_inference.rs

**Files:**
- Modify: `src/kernels/ffi_inference.rs`

The existing `load_best` function already handles architecture-specific kernel selection (used for ARM I8MM). Extend it to detect AVX-512 on x86_64 and prefer the `_avx512` variant.

- [ ] **Step 1: Add x86_64 AVX-512 detection to `load_best`**

Currently `load_best` only checks ARM I8MM. Add an x86_64 branch:

```rust
#[cfg(target_arch = "x86_64")]
{
    // Check AVX-512BW via cpuid. std::is_x86_feature_detected! works at runtime.
    if std::is_x86_feature_detected!("avx512bw") {
        let avx512_name = format!("{name}_avx512");
        if let Ok(lib) = load(&avx512_name) {
            eprintln!("olorin: {name}=avx512");
            return Ok(lib);
        }
    }
}
```

This makes `load_best("q4k_dot")` try `libq4k_dot_avx512.so` first, falling back to `libq4k_dot.so`.

- [ ] **Step 2: Change q5k and q6k loads to use `load_best`**

Currently `q5kd = load("q5k_dot")` and `q6kd = load("q6k_dot")`. Change to:
```rust
let q5kd = load_best("q5k_dot")?;
let q6kd = load_best("q6k_dot")?;
```

This enables AVX-512 variants for Q5K and Q6K once those kernels exist (Tasks 5-6).

- [ ] **Step 3: Build and run regression**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression -- --nocapture 2>&1 | tail -5
```

Expected: `olorin: q4k_dot=avx512` in stderr, `forward_one_bos_logits_bit_exact` passes.

- [ ] **Step 4: Commit**

```bash
git add src/kernels/ffi_inference.rs
git commit -m "feat: runtime AVX-512 kernel selection via load_best"
```

---

### Task 4: Verify bit-exact and benchmark Q4K

**Files:**
- No new files. Run existing tests and bench.

- [ ] **Step 1: Run full verify suite**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify --test gemma4_parallel_regression -- --test-threads=1 --nocapture 2>&1 | tail -20
```

All tests must pass. The AVX-512 kernel must produce identical logits to the AVX2 kernel (same f32 rounding — `reduce_add(i32x16)` should produce the same scalar as `reduce_add(i32x8)` when the input is zero-extended, since integer addition is associative).

If logits differ: the `reduce_add` order changed. Fix by splitting the i32x16 into two i32x8 halves, reducing each, then adding — matching the AVX2 accumulation order.

- [ ] **Step 2: Run decode bench**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test bench_decode_speed -- --nocapture 2>&1 | grep -E "prompt eval|eval time|time to first"
```

Record: decode t/s, prefill t/s, TTFT. Compare to baseline (3.17 decode, 3.41 prefill).

- [ ] **Step 3: Run llama-bench for comparison**

```bash
/root/dev/llama.cpp/build/bin/llama-bench -m ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf -p 9 -n 64 -t 2 2>&1 | tail -5
```

- [ ] **Step 4: Commit bench results in commit message**

```bash
git commit --allow-empty -m "bench: Q4K AVX-512 matvec results

decode: X.XX t/s (was 3.17, llama=10.78)
prefill: X.XX t/s (was 3.41, llama=21.18)
TTFT: XXX ms (was 250)"
```

---

### Task 5: Port Q5K dot to AVX-512

**Files:**
- Create: `kernels/q5k_dot_avx512.ea`

Same approach as Task 1-2: copy `q5k_dot.ea`, widen vector types from 256-bit to 512-bit. Q5K has an extra high-bit plane (5th bit) that requires an additional byte load and shift — this doubles naturally at 512-bit width.

- [ ] **Step 1: Copy and widen**

```bash
cp kernels/q5k_dot.ea kernels/q5k_dot_avx512.ea
```

Widen all vector types as in Task 1. Port both `q5k_dot_q8k` and `q5k_dot_q8k_4row`.

- [ ] **Step 2: Build and verify**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --nocapture 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add kernels/q5k_dot_avx512.ea
git commit -m "feat: q5k_dot_avx512 — 512-bit width Q5K dot + 4row"
```

---

### Task 6: Port Q6K dot to AVX-512

**Files:**
- Create: `kernels/q6k_dot_avx512.ea`

Same approach. Q6K has a different scale structure (16 scales per block instead of 12) but the dot product widening is mechanical.

- [ ] **Step 1: Copy and widen**

```bash
cp kernels/q6k_dot.ea kernels/q6k_dot_avx512.ea
```

- [ ] **Step 2: Build and verify**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --nocapture 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add kernels/q6k_dot_avx512.ea
git commit -m "feat: q6k_dot_avx512 — 512-bit width Q6K dot + 4row"
```

---

### Task 7: Final bench + push

**Files:**
- No changes.

- [ ] **Step 1: Full test suite**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify --test gemma4_parallel_regression -- --test-threads=1 2>&1 | tail -10
```

- [ ] **Step 2: Bench with all three quant types at AVX-512**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test bench_decode_speed -- --nocapture 2>&1 | tail -30
```

- [ ] **Step 3: Push**

```bash
git push
```
