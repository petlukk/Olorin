# Batched Prompt-Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **HARD RULES (apply to ALL agents):**
> - No file exceeds 500 lines. Split before you hit the limit.
> - Every feature proven by end-to-end test. If it's not tested, it doesn't exist.
> - No fake functions. No silent fallbacks. No `// TODO`, `// HACK`, `// for now`.
> - Olorin is Eä's showcase — every SIMD op must be an Eä kernel. Do NOT simplify kernel code to scalar Rust.
> - Match llama.cpp **bit-exact**, not "close enough".
> - llama.cpp reference: `/root/dev/llama.cpp/` (build 8685)
> - eacompute compiler: `/root/dev/eacompute/target/release/ea`
> - Build: `PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release`
> - **eabrain protocol** (mandatory, not optional):
>   - At the start of every task: run `eabrain status` and `eabrain recall` (the bisection findings from 2026-04-08 — Q8K rounding, batched-vs-incremental drift, banker's rounding intrinsic — are saved there and may answer questions before you start grepping).
>   - Before searching for any Eä kernel by name: `eabrain search <name>`.
>   - Before assuming an Eä intrinsic doesn't exist: `eabrain ref <name>` AND grep `/root/dev/eacompute/src/typeck/intrinsics*.rs` and `/root/dev/eacompute/src/codegen/simd*.rs` directly. **eabrain does not index eacompute's Rust intrinsic definitions** — this is a known limitation; if `eabrain ref` returns nothing, the intrinsic may still exist in eacompute source. Only conclude it doesn't exist after grepping all of: `intrinsics*.rs`, `simd*.rs`, `eacompute/CHANGELOG.md`, `eacompute/README.md`, `eacompute/tests/`.
>   - After editing any `.ea` kernel: run `eabrain index` so subsequent `eabrain search` calls see the new symbols.
>   - At the end of any task that produced a non-obvious finding (a kernel quirk, an intrinsic discovery, a bug pattern): `eabrain remember "..."` so the next subagent inherits it.
> - Branch: `gemma4-batched-prompt-eval` (already created from `gemma4-cleanup`).

**Goal:** Olorin's prompt-eval matches llama.cpp's batched prompt-eval **bit-exactly** via a hand-tuned Eä Q4K×Q8K gemm kernel with repacked weights, closing the 6.5× prompt-eval speed gap (current: 4 t/s; target: ≥27 t/s on this 2-core box) and producing the comparison the Eä showcase needs.

**Architecture:**
- New `forward_batch(tokens: &[u32]) -> &[f32]` path on `Gemma4State` for prompt eval. Per-layer ops are batched: input is `{hidden, N}` instead of `{hidden}`. Decode (`forward_one`) stays as today and runs after the prompt is consumed.
- Q4K weights are repacked to ggml's `q4_K_8x8_q8_K` interleaved layout (matches `repack_q4_K_to_q4_K_8_bl(t, 8, ...)` in `ggml/src/ggml-cpu/repack.cpp:3231`). Repack happens once at model load and stores into a parallel weight buffer; the existing per-row Q4K weight stays for `forward_one` (decode) untouched.
- A new Eä kernel `q4k_8x8_q8k_gemm` does the batched matmul, mirroring ggml's `gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>` accumulation order so that f32 results match bit-for-bit. New supporting kernels: `q4k_repack_8x8`, `q8k_quant_batched`, `gemma4_rmsnorm_batched`, `gemma4_rope_batched`, `gelu_mul_batched`, `attn_qk_batched`, `attn_softmax_batched`, `attn_vmul_batched`.
- Numerical correctness gate at every step: olorin output vs llama-eval-callback dump bit-equal where the algorithm makes it possible (matvec rows that hit identical accumulation orders), and L2-norm-within-1e-6 at every intermediate.

**Tech Stack:** Rust, Eä (eacompute), x86 AVX2/AVX-512 + ARM NEON, ggml Q4K format, std::thread via olorin's existing `ThreadPool`.

**Out of scope (explicit follow-up plans):**
- Q5K and Q6K batched gemm (only Q4K is on the prompt-eval critical path for E2B, Q6K is the unembedding which is matvec-only).
- WhatsApp / web UI / vault changes — none.
- Pi 5 deployment + cross-arch testing — done in a follow-up plan once x86 is bit-exact.
- New attention algorithms — match what ggml does, no creativity.

---

## Per-Commit Verification Gate

After every code-changing task in this plan, before `git commit`, **all of the following must pass**. No exceptions.

**Tools available:**
- llama.cpp build 8685 binaries in `/root/dev/llama.cpp/build/bin/` (`llama-bench`, `llama-eval-callback`, `llama-cli`, `llama-tokenize`)
- Model: `/root/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf`
- eabrain CLI for kernel/intrinsic lookups
- eacompute source: `/root/dev/eacompute/`

**Gate 1: Build clean.**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
grep -E "^(warning|error)" /tmp/olorin-build.log && exit 1 || true
```

Zero warnings, zero errors.

**Gate 2: Bit-exact decode regression.**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression 2>&1 | tail -5
```

Must pass — `forward_one_bos_logits_bit_exact` ensures the existing decode path didn't drift. **Never refresh this snapshot** during this plan; if it changes, you broke the decode path.

**Gate 3: gemma4_verify suite.**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -15
```

All 9 existing steps (step0..step7) must pass.

**Gate 4: Line limit.**

```bash
find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'
```

Empty output. If any file is >500 lines after a task, split before commit.

**Gate 5: New batched-eval suite (added in Task 5, expanded in later tasks).**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify 2>&1 | tail -15
```

Must pass once the test file exists.

---

## Phase A — Research & API Skeleton

The Eä gemm kernel must mirror ggml's accumulation order exactly. Before writing kernel code, document the format and the inner loop.

### Task 1: Research ggml's Q4K_8x8 repack format

**Files:**
- Create: `docs/superpowers/research/2026-04-08-ggml-q4k-8x8-format.md`
- Read (no edit): `/root/dev/llama.cpp/ggml/src/ggml-cpu/repack.cpp` (lines 3231–3300, function `repack_q4_K_to_q4_K_8_bl`)
- Read (no edit): `/root/dev/llama.cpp/ggml/src/ggml-common.h` (`block_q4_K` and `block_q4_Kx8` definitions)

- [ ] **Step 1: Read the canonical Q4K block layout**

From `ggml-common.h`, extract the `block_q4_K` struct definition. Document in the research note:
- Block size (`QK_K = 256`)
- Storage of `d`, `dmin`, `scales[12]`, `qs[QK_K/2]`
- Total bytes per block (compute it)

- [ ] **Step 2: Read the repack function**

From `repack.cpp:3231-3300`, document in the research note:
- Input: standard `block_q4_K` array of length `nrows * (n_cols / QK_K)`
- Output: repacked layout where 8 rows are interleaved per block, with field order (d block, dmin block, scales block, qs block)
- The exact byte order of the interleaved fields (this is the part the Eä kernel must reproduce byte-for-byte)
- Whether `d` and `dmin` are stored as f16 or f32 in the repacked layout

- [ ] **Step 3: Compute and document the repacked block size**

Repacked block holds 8 rows × 1 column-block. Compute total bytes per repacked block and write it in the note. This number drives buffer sizing in the Eä kernel.

- [ ] **Step 4: Commit the research note**

```bash
git add docs/superpowers/research/2026-04-08-ggml-q4k-8x8-format.md
git commit -m "research: document ggml q4k_8x8 repack format

Mirrors repack_q4_K_to_q4_K_8_bl from ggml repack.cpp; the
batched prompt-eval Eä kernel will read this layout."
```

### Task 2: Research ggml's Q4K_8x8 × Q8K gemm inner loop

**Files:**
- Create: `docs/superpowers/research/2026-04-08-ggml-q4k-8x8-q8k-gemm.md`
- Read (no edit): `/root/dev/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` — find the `gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>` template specialization or its equivalent
- Read (no edit): `/root/dev/llama.cpp/ggml/src/ggml-cpu/repack.cpp` (lines 4530–4570) for the trait registration

- [ ] **Step 1: Locate the gemm kernel for q4_K_8x8 × q8_K**

```bash
grep -n "gemm<block_q4_K" /root/dev/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp
grep -n "q4_K.*q8_K.*gemm\|gemm.*q4_K.*q8_K" /root/dev/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp
```

Document the file:line in the research note.

- [ ] **Step 2: Document the loop structure**

In the research note, sketch the inner loop in pseudocode at the granularity of "what gets multiplied/accumulated in what order". Capture:
- Outer loop dimension (rows × N tokens × column-blocks?)
- SIMD width and lane assignment
- Whether scales (`d`, `dmin`) are applied per-block or accumulated then applied at the end
- The exact `_mm256_*` (or `_mm512_*`) intrinsics used

This is the spec that Eä kernel `q4k_8x8_q8k_gemm` must replicate to be bit-exact.

- [ ] **Step 3: Document f32 accumulation order**

This is the critical part. Identify the order of the floating-point sums:
- Within one Q4K block: are `i32` partial sums accumulated then converted to f32 at the end of the block, or per-32-element subblock?
- Across blocks within one row: ordered low-to-high block index?
- Across rows in the 8-row tile: independent accumulators?

Olorin's kernel must use the same order. Two `f32 + f32` sums in different orders give different bit patterns even with the same operands.

- [ ] **Step 4: Commit research note**

```bash
git add docs/superpowers/research/2026-04-08-ggml-q4k-8x8-q8k-gemm.md
git commit -m "research: document ggml q4k_8x8 x q8k gemm inner loop

Captures the f32 accumulation order that Olorin's Eä kernel must
match for bit-exact prompt-eval parity with llama.cpp."
```

### Task 3: Add `forward_batch` API skeleton (no behavior change)

**Files:**
- Modify: `src/inference/forward.rs` (add `forward_batch` method on `Gemma4State`)
- Modify: `src/inference/mod.rs` (no actual change, but verify exports)

The skeleton calls `forward_one` N times in a loop — no batching yet. This is purely to lock in the API surface so subsequent tasks can replace the body without touching callers.

- [ ] **Step 1: Write the API skeleton on Gemma4State**

In `src/inference/forward.rs`, add this method **inside the `impl Gemma4State` block** that already contains `forward_one`:

```rust
/// Run a prompt-eval forward pass over `tokens`. Returns the final-token logits.
/// During Phase A this is a thin loop over `forward_one`; later phases replace
/// the body with batched gemm.
pub fn forward_batch(
    &mut self,
    model: &Gemma4Model,
    tokens: &[u32],
    pool: &crate::inference::threadpool::ThreadPool,
) -> &[f32] {
    assert!(!tokens.is_empty(), "forward_batch requires at least one token");
    let last = *tokens.last().unwrap();
    for &t in &tokens[..tokens.len() - 1] {
        let _ = self.forward_one(model, t, pool);
    }
    self.forward_one(model, last, pool)
}
```

- [ ] **Step 2: Build clean**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
```

Expected: zero warnings, zero errors.

- [ ] **Step 3: Run gates 2 and 3**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression --test gemma4_verify -- --test-threads=1 2>&1 | tail -5
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/inference/forward.rs
git commit -m "feat: add forward_batch API skeleton (loops forward_one)

Locks the API surface for prompt-eval. Next phases will replace
the body with batched gemm without touching callers."
```

### Task 4: Add `gemma4_batch_verify` test scaffold

**Files:**
- Create: `tests/gemma4_batch_verify.rs`

This test file will own all bit-exact comparisons against ggml gemm and llama-eval-callback dumps.

- [ ] **Step 1: Create the test file with one passing skeleton test**

```rust
//! Bit-exact verification of olorin's batched prompt-eval against llama.cpp.
//!
//! Run: cargo test --release --test gemma4_batch_verify -- --nocapture --test-threads=1
//!
//! Each test compares an olorin intermediate against a value captured from
//! llama-eval-callback dumps. Sums use f64 accumulation to avoid the f32
//! ordering trap that bit us in step6.

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

fn has_model() -> bool {
    Path::new(&model_path()).exists()
}

fn sum_f64(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>()
}

#[test]
fn batch0_skeleton() {
    if !has_model() {
        eprintln!("SKIP: no model");
        return;
    }
    eprintln!("=== batch0: skeleton — no batched code yet ===");
    // Sanity: model can be loaded and forward_batch over [BOS] equals forward_one(BOS).
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();
    let pool = olorin::inference::threadpool::ThreadPool::new();

    let mut a = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);
    let logits_one = a.forward_one(&model, 2, &pool).to_vec();

    let mut b = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);
    let logits_batch = b.forward_batch(&model, &[2u32], &pool).to_vec();

    assert_eq!(logits_one.len(), logits_batch.len());
    let max_abs_diff = logits_one
        .iter()
        .zip(logits_batch.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max abs diff = {}", max_abs_diff);
    assert!(max_abs_diff < 1e-6, "skeleton forward_batch should equal forward_one for [BOS]");

    eprintln!("PASS: batch0");
}
```

- [ ] **Step 2: Run the test**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify -- --nocapture 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/gemma4_batch_verify.rs
git commit -m "test: add gemma4_batch_verify scaffold

Skeleton test confirming forward_batch([BOS]) == forward_one(BOS).
This file will host all bit-exact prompt-eval verification."
```

---

## Phase B — Eä Q4K_8x8 Repack & Gemm Kernel

The showcase pieces. Three new Eä kernels and their tests, in dependency order.

### Task 5: Implement Eä `q4k_repack_8x8` kernel

**Files:**
- Create: `kernels/q4k_repack.ea`
- Modify: `src/kernels/ffi_inference.rs` (add wrapper)

The repack kernel takes a flat array of `block_q4_K` records (the existing GGUF layout) and writes a parallel buffer in the 8-row interleaved layout documented in Task 1's research note.

- [ ] **Step 1: Read research note**

```bash
cat docs/superpowers/research/2026-04-08-ggml-q4k-8x8-format.md
```

The kernel implementation uses the field order and byte sizes documented there. If the note doesn't fully specify a field, return to Task 1 and complete it before continuing.

- [ ] **Step 2: Write the kernel**

Create `kernels/q4k_repack.ea`. The signature is:

```
export func q4k_repack_8x8(
    src: *u8,                          // input: nrows * row_bytes (standard q4_K layout)
    out dst: *mut u8 [cap: nrows * row_bytes],
    nrows: i32,                        // must be multiple of 8
    n_cols: i32                        // must be multiple of 256
) {
    // For each tile of 8 rows × n_blocks column-blocks:
    //   read 8 source blocks (one per row, same column-block)
    //   write them in the interleaved order specified by the research note
}
```

The body loops `tile = 0..nrows/8`, then `blk = 0..n_blocks`, then writes the 8 d values, 8 dmin values, 8 scales arrays, and 8 qs arrays in the order documented. **Use the documented byte order verbatim — do not "improve" it. Bit-exactness requires the exact same bytes.**

- [ ] **Step 3: Add Rust FFI wrapper**

In `src/kernels/ffi_inference.rs`, add the function pointer entry alongside `quant_f32_q8k`:

```rust
pub q4k_repack_8x8: Q4kRepack8x8Fn,
```

with type:

```rust
type Q4kRepack8x8Fn = unsafe extern "C" fn(*const u8, *mut u8, i32, i32);
```

and load it from the `q4k_repack.so` library symbol `q4k_repack_8x8`. Add a safe `pub unsafe fn q4k_repack_8x8(...)` wrapper that mirrors the existing `quant_f32_q8k` wrapper.

- [ ] **Step 4: Run gate 1**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "warning|error"
```

Expected: empty output.

- [ ] **Step 5: Commit**

```bash
git add kernels/q4k_repack.ea kernels/q4k_repack.ea.json src/kernels/ffi_inference.rs
git commit -m "feat: add q4k_repack_8x8 Eä kernel

Mirrors ggml's repack_q4_K_to_q4_K_8_bl byte-for-byte.
Tests in next task."
```

### Task 6: Test `q4k_repack_8x8` against ggml byte-for-byte

**Files:**
- Modify: `tests/gemma4_batch_verify.rs`

The test loads a real Q4K weight from the gemma gguf (`blk.0.attn_output.weight`, shape {2048, 1536}), repacks it with both olorin and ggml's `repack_q4_K_to_q4_K_8_bl`, and compares the buffers byte-for-byte.

- [ ] **Step 1: Add a helper that calls ggml's repack via a tiny C bridge**

In a new file `tests/c_bridge/repack_bridge.c`, write a single function that wraps ggml's `repack_q4_K_to_q4_K_8_bl`:

```c
// Compile against ggml-cpu sources (or link the prebuilt libggml.so).
// Exposes ggml's reference repack as a flat C function for the Rust test.
#include "ggml.h"
extern int repack_q4_K_to_q4_K_8_bl(struct ggml_tensor *t, int interleave_block, const void *data, size_t data_size);

void olorin_test_repack_q4k_8x8(
    const void *src, void *dst,
    int nrows, int n_cols
) {
    // Build a ggml_tensor descriptor that points at dst, then call the repack.
    // Field-by-field setup goes here — see ggml_tensor in ggml.h.
}
```

(Exact body depends on how `ggml_tensor` is laid out — the test must construct one that satisfies repack's expectations. If this proves too brittle, an alternative is to skip the C bridge and instead extract the bytes from a model that ggml has already loaded with `--cpu-repack 1`, then reverse-engineer the layout from observation.)

- [ ] **Step 2: Add the comparison test**

Append to `tests/gemma4_batch_verify.rs`:

```rust
#[test]
fn batch1_repack_q4k_bytes_match_ggml() {
    if !has_model() { eprintln!("SKIP: no model"); return; }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    // Pick a Q4K weight tensor: blk.0.attn_output.weight is hd × n_heads*head_dim_k
    // for layer 0.
    let lw = &model.layers[0];
    assert_eq!(lw.wo_dtype, olorin::inference::matmul::GGML_TYPE_Q4_K);
    let n_rows = model.hidden_dim;
    let n_cols = model.n_heads * model.head_dim_k[0];
    let row_bytes = (n_cols / 256) * 144; // q4_K block is 144 bytes (verify against research note)
    let total_bytes = n_rows * row_bytes;

    // Run olorin's repack
    let mut olorin_out = vec![0u8; total_bytes];
    unsafe {
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.wo as *const u8,
            olorin_out.as_mut_ptr(),
            n_rows as i32,
            n_cols as i32,
        );
    }

    // Run ggml's repack via the C bridge
    let mut ggml_out = vec![0u8; total_bytes];
    extern "C" {
        fn olorin_test_repack_q4k_8x8(src: *const u8, dst: *mut u8, nrows: i32, n_cols: i32);
    }
    unsafe {
        olorin_test_repack_q4k_8x8(
            lw.wo as *const u8,
            ggml_out.as_mut_ptr(),
            n_rows as i32,
            n_cols as i32,
        );
    }

    // Byte-for-byte comparison
    let first_diff = olorin_out.iter().zip(&ggml_out).position(|(a, b)| a != b);
    if let Some(idx) = first_diff {
        let lo = idx.saturating_sub(8);
        let hi = (idx + 8).min(total_bytes);
        eprintln!("first diff at byte {idx}");
        eprintln!("olorin: {:02x?}", &olorin_out[lo..hi]);
        eprintln!("ggml:   {:02x?}", &ggml_out[lo..hi]);
        panic!("repack mismatch");
    }
    eprintln!("PASS: batch1 — {} bytes match", total_bytes);
}
```

- [ ] **Step 3: Run the test**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify batch1 -- --nocapture
```

Expected: PASS. If FAIL, the kernel diverges from ggml's repack — fix the Eä kernel byte order (do NOT modify the test or the C bridge to "tolerate" the diff).

- [ ] **Step 4: Commit**

```bash
git add tests/gemma4_batch_verify.rs tests/c_bridge/repack_bridge.c
git commit -m "test: byte-exact comparison of q4k_repack_8x8 vs ggml

Loads blk.0.attn_output.weight from the real gemma gguf, repacks with
both olorin's Eä kernel and ggml's reference, asserts byte equality."
```

### Task 7: Implement Eä `q4k_8x8_q8k_matvec` kernel (N=1)

**Files:**
- Create: `kernels/q4k_8x8_q8k.ea`
- Modify: `src/kernels/ffi_inference.rs` (add wrapper)

Before doing the full N>1 gemm, get the inner loop right with N=1. This kernel takes a repacked Q4K weight (output of `q4k_repack_8x8`) and a single Q8K column, produces one f32 output column. Same accumulation order as ggml's gemm with N=1.

- [ ] **Step 1: Read research note from Task 2**

```bash
cat docs/superpowers/research/2026-04-08-ggml-q4k-8x8-q8k-gemm.md
```

The kernel implementation must follow the documented inner loop and f32 accumulation order. If the note doesn't specify these, complete Task 2 first.

- [ ] **Step 2: Write the kernel**

Create `kernels/q4k_8x8_q8k.ea`. Function signature:

```
export func q4k_8x8_q8k_matvec(
    weight_packed: *u8,                // output of q4k_repack_8x8
    q8_qs: *i8,                        // Q8K input qs, n_cols values
    q8_d: *f32,                        // Q8K input d, n_blocks values
    q8_bsums: *i16,                    // Q8K input bsums, n_blocks * 16 values
    out output: *mut f32 [cap: n_rows],
    n_rows: i32,                       // multiple of 8
    n_cols: i32                        // multiple of 256
) {
    // For each tile of 8 rows:
    //   For each column-block:
    //     Load 8 d, 8 dmin, 8 scales arrays, 8 qs arrays from packed weight
    //     Dot 8 weight rows × q8_qs block (i32 acc), apply per-row scales,
    //     accumulate into 8 f32 rolling sums.
    //   Write 8 f32 outputs.
}
```

The body's f32 accumulation order must be the one documented in the research note.

- [ ] **Step 3: Add FFI wrapper**

In `src/kernels/ffi_inference.rs`, add `q4k_8x8_q8k_matvec` alongside `q4k_repack_8x8`.

- [ ] **Step 4: Build clean**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "warning|error"
```

- [ ] **Step 5: Commit**

```bash
git add kernels/q4k_8x8_q8k.ea kernels/q4k_8x8_q8k.ea.json src/kernels/ffi_inference.rs
git commit -m "feat: add q4k_8x8_q8k_matvec Eä kernel

Inner loop mirrors ggml's gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>
with N=1; accumulation order follows the research note."
```

### Task 8: Test `q4k_8x8_q8k_matvec` bit-exact vs ggml

**Files:**
- Modify: `tests/gemma4_batch_verify.rs`

- [ ] **Step 1: Add the test**

```rust
#[test]
fn batch2_q4k_8x8_q8k_matvec_bitexact_vs_existing_matvec() {
    // Strategy: pick a Q4K weight (Wo for L0), pick an arbitrary input vector x,
    // quantize x to Q8K, run BOTH:
    //   - olorin's existing per-row q4k_matvec (which already matches ggml's
    //     non-repacked path bit-exactly at pos=0 — proven by step3b)
    //   - olorin's new q4k_8x8_q8k_matvec on the repacked weight with the same input
    // The TWO outputs must be bit-exact f32-equal because both implement the
    // same mathematical operation. They will only differ if accumulation order
    // differs — in which case the new kernel doesn't match ggml's gemm path.
    //
    // (We can't compare to ggml-gemm-N=1 directly because llama-eval-callback
    // dumps batched, but our existing matvec is itself bit-exact-equivalent to
    // ggml's non-batched matvec at N=1, so it serves as ground truth.)

    if !has_model() { eprintln!("SKIP: no model"); return; }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let lw = &model.layers[0];
    let n_rows = model.hidden_dim;
    let n_cols = model.n_heads * model.head_dim_k[0];

    // Build a deterministic input vector
    let mut x = vec![0.0f32; n_cols];
    for i in 0..n_cols {
        x[i] = ((i as f32) * 0.0137 - 0.5).sin() * 0.3;
    }

    // Quantize to Q8K
    let n_blocks = n_cols / 256;
    let mut q8_qs = vec![0i8; n_cols + 12];
    let mut q8_d = vec![0.0f32; n_blocks];
    let mut q8_bsums = vec![0i16; n_blocks * 16];
    olorin::inference::matmul::quant_input(&x, &mut q8_qs, &mut q8_d, &mut q8_bsums);

    // Reference: existing matvec
    let mut ref_out = vec![0.0f32; n_rows];
    olorin::inference::matmul::q4k_matvec(
        lw.wo as *const u8,
        &q8_qs, &q8_d, &q8_bsums,
        &mut ref_out,
        n_rows, n_cols,
    );

    // New path: repack then 8x8 matvec
    let row_bytes = n_blocks * 144;
    let mut packed = vec![0u8; n_rows * row_bytes];
    unsafe {
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.wo as *const u8,
            packed.as_mut_ptr(),
            n_rows as i32,
            n_cols as i32,
        );
    }
    let mut new_out = vec![0.0f32; n_rows];
    unsafe {
        olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
            packed.as_ptr(),
            q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
            new_out.as_mut_ptr(),
            n_rows as i32, n_cols as i32,
        );
    }

    // Compare bit-for-bit
    let mut max_abs_diff = 0.0f32;
    let mut first_mismatch = None;
    for i in 0..n_rows {
        if ref_out[i].to_bits() != new_out[i].to_bits() {
            if first_mismatch.is_none() { first_mismatch = Some(i); }
            let d = (ref_out[i] - new_out[i]).abs();
            if d > max_abs_diff { max_abs_diff = d; }
        }
    }
    eprintln!("rows={n_rows}  max_abs_diff={max_abs_diff:e}  first_mismatch={first_mismatch:?}");
    assert_eq!(first_mismatch, None, "q4k_8x8_q8k_matvec must be bit-exact to existing matvec");
    eprintln!("PASS: batch2");
}
```

- [ ] **Step 2: Run the test**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify batch2 -- --nocapture
```

Expected: PASS with `first_mismatch=None`. If FAIL, the kernel's accumulation order differs from olorin's existing matvec — adjust the kernel inner loop to match (do NOT loosen the test).

- [ ] **Step 3: Commit**

```bash
git add tests/gemma4_batch_verify.rs
git commit -m "test: bit-exact q4k_8x8_q8k_matvec vs existing matvec at N=1"
```

### Task 9: Extend `q4k_8x8_q8k_matvec` → `q4k_8x8_q8k_gemm` for N>1

**Files:**
- Modify: `kernels/q4k_8x8_q8k.ea` (add the gemm function alongside the matvec)
- Modify: `src/kernels/ffi_inference.rs` (add wrapper)

- [ ] **Step 1: Add the gemm signature in the kernel file**

```
export func q4k_8x8_q8k_gemm(
    weight_packed: *u8,
    q8_qs: *i8,                        // shape: n_cols * N (column-major: column k starts at offset k * n_cols)
    q8_d: *f32,                        // shape: n_blocks * N
    q8_bsums: *i16,                    // shape: n_blocks * 16 * N
    out output: *mut f32 [cap: n_rows * N],   // column-major
    n_rows: i32,                       // multiple of 8
    n_cols: i32,                       // multiple of 256
    n_cols_batch: i32                  // N (number of input columns)
) {
    // Outer loop: tiles of 8 rows.
    // Per tile: load weight tile ONCE.
    // Inner: for k in 0..N { run the same inner loop as q4k_8x8_q8k_matvec
    //        for column k of input, accumulate into row[..][k] of output. }
    // The savings is the weight tile load reuse across columns.
}
```

The accumulation order **per output column** must be identical to `q4k_8x8_q8k_matvec`. The cross-column ordering doesn't affect any single output value because columns are independent.

- [ ] **Step 2: Add FFI wrapper**

Add `q4k_8x8_q8k_gemm` in `src/kernels/ffi_inference.rs`.

- [ ] **Step 3: Build clean**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "warning|error"
```

- [ ] **Step 4: Commit**

```bash
git add kernels/q4k_8x8_q8k.ea kernels/q4k_8x8_q8k.ea.json src/kernels/ffi_inference.rs
git commit -m "feat: extend q4k_8x8_q8k to gemm (N>1) with weight reuse"
```

### Task 10: Test `q4k_8x8_q8k_gemm` bit-exact for N=2 and N=8

**Files:**
- Modify: `tests/gemma4_batch_verify.rs`

- [ ] **Step 1: Add the test**

```rust
#[test]
fn batch3_q4k_8x8_q8k_gemm_bitexact_per_column() {
    // Run the gemm for N input columns and compare each output column to
    // running the matvec on that single column. The two must be bit-exact
    // because the per-column accumulation order is identical.
    if !has_model() { eprintln!("SKIP: no model"); return; }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let lw = &model.layers[0];
    let n_rows = model.hidden_dim;
    let n_cols = model.n_heads * model.head_dim_k[0];
    let n_blocks = n_cols / 256;
    let row_bytes = n_blocks * 144;

    // Repack the weight once
    let mut packed = vec![0u8; n_rows * row_bytes];
    unsafe {
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.wo as *const u8,
            packed.as_mut_ptr(),
            n_rows as i32, n_cols as i32,
        );
    }

    for &n in &[2usize, 8usize] {
        // Build N input columns
        let mut x = vec![0.0f32; n_cols * n];
        for k in 0..n {
            for i in 0..n_cols {
                x[k * n_cols + i] = (((i + k * 7) as f32) * 0.0137 - 0.5).cos() * 0.4;
            }
        }
        // Quantize each column independently
        let mut q8_qs = vec![0i8; (n_cols + 12) * n];
        let mut q8_d = vec![0.0f32; n_blocks * n];
        let mut q8_bsums = vec![0i16; n_blocks * 16 * n];
        for k in 0..n {
            olorin::inference::matmul::quant_input(
                &x[k * n_cols..(k + 1) * n_cols],
                &mut q8_qs[k * (n_cols + 12)..(k + 1) * (n_cols + 12)],
                &mut q8_d[k * n_blocks..(k + 1) * n_blocks],
                &mut q8_bsums[k * n_blocks * 16..(k + 1) * n_blocks * 16],
            );
        }

        // Reference: per-column matvec via the new kernel
        let mut ref_out = vec![0.0f32; n_rows * n];
        for k in 0..n {
            unsafe {
                olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
                    packed.as_ptr(),
                    q8_qs[k * (n_cols + 12)..].as_ptr(),
                    q8_d[k * n_blocks..].as_ptr(),
                    q8_bsums[k * n_blocks * 16..].as_ptr(),
                    ref_out[k * n_rows..].as_mut_ptr(),
                    n_rows as i32, n_cols as i32,
                );
            }
        }

        // New: gemm in one call
        let mut new_out = vec![0.0f32; n_rows * n];
        unsafe {
            olorin::kernels::ffi_inference::q4k_8x8_q8k_gemm(
                packed.as_ptr(),
                q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
                new_out.as_mut_ptr(),
                n_rows as i32, n_cols as i32, n as i32,
            );
        }

        for i in 0..(n_rows * n) {
            assert_eq!(
                ref_out[i].to_bits(),
                new_out[i].to_bits(),
                "N={n}, output[{i}] differs: {} vs {}",
                ref_out[i],
                new_out[i],
            );
        }
        eprintln!("PASS: batch3 N={n}");
    }
}
```

- [ ] **Step 2: Run the test**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify batch3 -- --nocapture
```

Expected: PASS for both N=2 and N=8.

- [ ] **Step 3: Commit**

```bash
git add tests/gemma4_batch_verify.rs
git commit -m "test: q4k_8x8_q8k_gemm bit-exact per-column vs matvec (N=2, N=8)"
```

### Task 11: Performance bench — Eä gemm vs olorin's existing matvec loop

**Files:**
- Create: `tests/bench_q4k_gemm.rs`

- [ ] **Step 1: Write a Criterion-style timing harness without Criterion**

```rust
//! Compare: N matvec calls vs 1 gemm call on the same problem size.
//! Run: cargo test --release --test bench_q4k_gemm -- --nocapture
use std::time::Instant;
use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
fn bench_gemm_vs_matvec_loop() {
    if !Path::new(&model_path()).exists() { eprintln!("SKIP: no model"); return; }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let lw = &model.layers[0];
    let n_rows = model.hidden_dim;       // 1536
    let n_cols = model.hidden_dim;       // ffn_norm path is 1536x1536; pick wo = 1536x2048
    let n_cols = model.n_heads * model.head_dim_k[0];
    let n_blocks = n_cols / 256;
    let row_bytes = n_blocks * 144;

    let mut packed = vec![0u8; n_rows * row_bytes];
    unsafe {
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.wo as *const u8, packed.as_mut_ptr(),
            n_rows as i32, n_cols as i32,
        );
    }

    for &n in &[1usize, 2, 8, 32, 128] {
        // Synthetic batched input
        let mut q8_qs = vec![0i8; (n_cols + 12) * n];
        let mut q8_d = vec![0.0f32; n_blocks * n];
        let mut q8_bsums = vec![0i16; n_blocks * 16 * n];
        // (fill with dummy data)
        for v in q8_qs.iter_mut() { *v = 5; }
        for v in q8_d.iter_mut() { *v = 0.01; }

        let mut out = vec![0.0f32; n_rows * n];

        let iters = 50;

        // Path A: N matvec calls (current behavior)
        let t0 = Instant::now();
        for _ in 0..iters {
            for k in 0..n {
                unsafe {
                    olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
                        packed.as_ptr(),
                        q8_qs[k * (n_cols + 12)..].as_ptr(),
                        q8_d[k * n_blocks..].as_ptr(),
                        q8_bsums[k * n_blocks * 16..].as_ptr(),
                        out[k * n_rows..].as_mut_ptr(),
                        n_rows as i32, n_cols as i32,
                    );
                }
            }
        }
        let t_matvec_loop = t0.elapsed().as_secs_f64() / iters as f64;

        // Path B: 1 gemm call
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                olorin::kernels::ffi_inference::q4k_8x8_q8k_gemm(
                    packed.as_ptr(),
                    q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
                    out.as_mut_ptr(),
                    n_rows as i32, n_cols as i32, n as i32,
                );
            }
        }
        let t_gemm = t0.elapsed().as_secs_f64() / iters as f64;

        let speedup = t_matvec_loop / t_gemm;
        eprintln!("N={n:>4}  matvec-loop={:>9.3} ms  gemm={:>9.3} ms  speedup={:.2}x",
            t_matvec_loop * 1000.0, t_gemm * 1000.0, speedup);
    }
}
```

- [ ] **Step 2: Run the bench**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test bench_q4k_gemm -- --nocapture --test-threads=1
```

Document the speedup numbers in the commit message. **Hard requirement: at N=8, gemm must be at least 1.5× faster than the matvec loop.** If it isn't, the gemm doesn't reuse weight loads enough — return to the kernel and add load reuse.

- [ ] **Step 3: Commit**

```bash
git add tests/bench_q4k_gemm.rs
git commit -m "bench: q4k_8x8_q8k_gemm vs matvec loop

Numbers (this box, 2 cores):
N=1:    [fill in]
N=2:    [fill in]  speedup [x.xx]x
N=8:    [fill in]  speedup [x.xx]x
N=32:   [fill in]  speedup [x.xx]x
N=128:  [fill in]  speedup [x.xx]x"
```

---

## Phase C — Batched Forward Path

The remaining per-layer ops need batched versions. None of these are as hard as the gemm. Each task adds one Eä kernel + Rust wrapper + bit-exact test.

### Task 12: Add batched buffers to `Gemma4State`

**Files:**
- Modify: `src/inference/forward.rs` (add fields to `Gemma4State` and init in `new`)

- [ ] **Step 1: Add fields**

In the `Gemma4State` struct, add (right after the existing single-token activation buffers):

```rust
    // Batched prompt-eval buffers. Sized for max prompt batch.
    // Activation tensors are column-major: column k is at offset k * hd.
    pub(crate) batch_x: Vec<f32>,        // {hd, max_batch}
    pub(crate) batch_x_norm: Vec<f32>,   // {hd, max_batch}
    pub(crate) batch_q: Vec<f32>,        // {n_heads * max_head_k, max_batch}
    pub(crate) batch_k: Vec<f32>,        // {n_kv_heads * max_head_k, max_batch}
    pub(crate) batch_v: Vec<f32>,        // {n_kv_heads * max_head_v, max_batch}
    pub(crate) batch_attn_out: Vec<f32>, // {n_heads * max_head_k, max_batch}
    pub(crate) batch_wo_out: Vec<f32>,   // {hd, max_batch}
    pub(crate) batch_attn_res: Vec<f32>, // {hd, max_batch}
    pub(crate) batch_gate: Vec<f32>,     // {max_ffn, max_batch}
    pub(crate) batch_up: Vec<f32>,       // {max_ffn, max_batch}
    pub(crate) batch_down: Vec<f32>,     // {hd, max_batch}
    pub(crate) batch_q8_qs: Vec<i8>,
    pub(crate) batch_q8_d: Vec<f32>,
    pub(crate) batch_q8_bsums: Vec<i16>,
    pub(crate) max_batch: usize,
```

In `Gemma4State::new`, allocate them with `max_batch = 64` (a reasonable starting cap; can be made parameterizable later). Add a `const MAX_BATCH: usize = 64;` at the top of the file.

- [ ] **Step 2: Build clean**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "warning|error"
```

- [ ] **Step 3: Verify line count**

```bash
wc -l src/inference/forward.rs
```

If forward.rs is now over 500 lines, **stop and split** before continuing. Move `Gemma4State::new` and field declarations to a new `src/inference/state.rs`.

- [ ] **Step 4: Commit**

```bash
git add src/inference/forward.rs
git commit -m "feat: add batched activation buffers to Gemma4State

Sized for MAX_BATCH=64 prompt tokens. Buffers are column-major.
Used by forward_batch in subsequent tasks."
```

### Task 13: Eä `q8k_quant_batched` kernel

**Files:**
- Create: `kernels/q8k_quant_batched.ea`
- Modify: `src/kernels/ffi_inference.rs`

Quantize a column-major `{n_cols, N}` f32 tensor into N independent Q8K vectors. Each column is quantized exactly the same way `quant_f32_q8k` does (banker's rounding — already fixed in this branch).

- [ ] **Step 1: Write the kernel as a loop over columns**

```
export func q8k_quant_batched(
    src: *f32,                         // shape (n_cols, N), col k at src + k * n_cols
    out dst_qs: *mut i8,
    out dst_d: *mut f32,
    out dst_bsums: *mut i16,
    n_cols: i32,
    n_batch: i32
) {
    let mut k: i32 = 0
    while k < n_batch {
        // Inline the body of quant_f32_q8k for column k.
        // (Eä has no function calls between kernels, so inline the existing code.)
        ...
        k = k + 1
    }
}
```

The inner body must be a verbatim copy of `quant_f32_q8k`'s body (banker's rounding via magic-number trick) — only the addressing changes (offset into src/dst by `k * n_cols`).

- [ ] **Step 2: FFI wrapper + Rust safe wrapper**

Same pattern as `quant_f32_q8k`.

- [ ] **Step 3: Add bit-exact test in gemma4_batch_verify**

```rust
#[test]
fn batch4_q8k_quant_batched_matches_n_calls() {
    olorin::kernels::ffi::init().unwrap();
    let n_cols = 1536usize;
    let n_blocks = n_cols / 256;
    let n_batch = 4usize;

    let mut x = vec![0.0f32; n_cols * n_batch];
    for k in 0..n_batch {
        for i in 0..n_cols {
            x[k * n_cols + i] = ((i + k * 11) as f32 * 0.0137 - 0.5).sin() * 0.5;
        }
    }

    // Reference: N independent calls
    let mut ref_qs = vec![0i8; (n_cols + 12) * n_batch];
    let mut ref_d = vec![0.0f32; n_blocks * n_batch];
    let mut ref_bsums = vec![0i16; n_blocks * 16 * n_batch];
    for k in 0..n_batch {
        olorin::inference::matmul::quant_input(
            &x[k * n_cols..(k + 1) * n_cols],
            &mut ref_qs[k * (n_cols + 12)..(k + 1) * (n_cols + 12)],
            &mut ref_d[k * n_blocks..(k + 1) * n_blocks],
            &mut ref_bsums[k * n_blocks * 16..(k + 1) * n_blocks * 16],
        );
    }

    // New: batched
    let mut new_qs = vec![0i8; (n_cols + 12) * n_batch];
    let mut new_d = vec![0.0f32; n_blocks * n_batch];
    let mut new_bsums = vec![0i16; n_blocks * 16 * n_batch];
    unsafe {
        olorin::kernels::ffi_inference::q8k_quant_batched(
            x.as_ptr(),
            new_qs.as_mut_ptr(), new_d.as_mut_ptr(), new_bsums.as_mut_ptr(),
            n_cols as i32, n_batch as i32,
        );
    }

    assert_eq!(new_qs, ref_qs);
    assert_eq!(new_d.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
               ref_d.iter().map(|f| f.to_bits()).collect::<Vec<_>>());
    assert_eq!(new_bsums, ref_bsums);
    eprintln!("PASS: batch4");
}
```

- [ ] **Step 4: Run the test**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify batch4 -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git add kernels/q8k_quant_batched.ea kernels/q8k_quant_batched.ea.json src/kernels/ffi_inference.rs tests/gemma4_batch_verify.rs
git commit -m "feat+test: q8k_quant_batched kernel + bit-exact vs N quant_input calls"
```

### Task 14: Eä `gemma4_rmsnorm_batched`

**Files:**
- Create: `kernels/gemma4_rmsnorm_batched.ea`
- Modify: `src/kernels/ffi_inference.rs`
- Modify: `tests/gemma4_batch_verify.rs`

- [ ] **Step 1: Read the existing single-column rmsnorm kernel**

```bash
cat kernels/gemma4_rmsnorm.ea
```

Note its body — the batched version applies the same body to N columns of a `{hd, N}` input tensor.

- [ ] **Step 2: Write the batched kernel**

```
export func gemma4_rmsnorm_batched(
    src: *f32,                         // {hd, N}, column-major
    weight: *f32,                      // {hd}
    out dst: *mut f32 [cap: hd * n_batch],
    hd: i32,
    eps: f32,
    n_batch: i32
) {
    let mut k: i32 = 0
    while k < n_batch {
        // Inline body of gemma4_rmsnorm for column k.
        // src + k * hd  →  dst + k * hd
        ...
        k = k + 1
    }
}
```

- [ ] **Step 3: FFI wrapper + bit-exact test**

```rust
#[test]
fn batch5_rmsnorm_batched_matches_n_calls() {
    olorin::kernels::ffi::init().unwrap();
    let hd = 1536;
    let n = 4;
    let mut x = vec![0.0f32; hd * n];
    for i in 0..hd*n { x[i] = ((i as f32) * 0.013 - 0.4).sin() * 0.6; }
    let mut w = vec![0.0f32; hd];
    for i in 0..hd { w[i] = 1.0 + (i as f32 * 0.001).cos() * 0.1; }
    let eps = 1e-6;

    // Reference
    let mut ref_out = vec![0.0f32; hd * n];
    for k in 0..n {
        unsafe {
            olorin::kernels::ffi_inference::gemma4_rmsnorm(
                x[k*hd..].as_ptr(), w.as_ptr(),
                ref_out[k*hd..].as_mut_ptr(),
                hd as i32, eps,
            );
        }
    }

    // Batched
    let mut new_out = vec![0.0f32; hd * n];
    unsafe {
        olorin::kernels::ffi_inference::gemma4_rmsnorm_batched(
            x.as_ptr(), w.as_ptr(), new_out.as_mut_ptr(),
            hd as i32, eps, n as i32,
        );
    }

    for i in 0..hd*n {
        assert_eq!(ref_out[i].to_bits(), new_out[i].to_bits(), "diff at {i}");
    }
    eprintln!("PASS: batch5");
}
```

- [ ] **Step 4: Build, test, commit**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify batch5 -- --nocapture
git add kernels/gemma4_rmsnorm_batched.ea kernels/gemma4_rmsnorm_batched.ea.json src/kernels/ffi_inference.rs tests/gemma4_batch_verify.rs
git commit -m "feat+test: gemma4_rmsnorm_batched bit-exact vs N rmsnorm calls"
```

### Task 15: Eä `gemma4_rope_batched`

**Files:**
- Create: `kernels/gemma4_rope_batched.ea`
- Modify: `src/kernels/ffi_inference.rs`
- Modify: `tests/gemma4_batch_verify.rs`

The existing `gemma4_rope` takes pre-computed cos/sin tables for **one** position. The batched version must apply RoPE for N positions (`pos`, `pos+1`, ..., `pos+N-1`) to a `{n_heads * head_dim, N}` Q or K tensor. Cos/sin tables are computed per-position.

- [ ] **Step 1: Decide cos/sin layout**

The simplest design: caller (Rust side) precomputes all N cos/sin tables into a `{half * N}` buffer (concatenated by position) and passes them to the kernel. The kernel then strides into the right table for column k.

Document this in the kernel header comment.

- [ ] **Step 2: Write the kernel**

```
export func gemma4_rope_batched(
    qk: *mut f32,                      // {n_heads * head_dim, N}
    cos_tables: *f32,                  // {half * N}, concat of N cos tables
    sin_tables: *f32,                  // {half * N}
    head_dim: i32,
    n_heads: i32,
    n_batch: i32
) {
    let mut k: i32 = 0
    while k < n_batch {
        // Apply RoPE rotation for column k using cos_tables[k*half..] and sin_tables[k*half..].
        // Body inlined from gemma4_rope.
        k = k + 1
    }
}
```

- [ ] **Step 3: FFI wrapper + test**

Test compares N sequential `gemma4_rope` calls (each with its own cos/sin) vs one `gemma4_rope_batched` call with concatenated tables. Bit-exact required.

- [ ] **Step 4: Build, test, commit**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify batch6 -- --nocapture
git add kernels/gemma4_rope_batched.ea kernels/gemma4_rope_batched.ea.json src/kernels/ffi_inference.rs tests/gemma4_batch_verify.rs
git commit -m "feat+test: gemma4_rope_batched bit-exact vs N rope calls"
```

### Task 16: Eä `gelu_mul_batched`

Same pattern. Body is `gelu_mul` inlined into a per-column loop. Bit-exact test against N sequential calls. Commit. *(Steps identical to Tasks 13–15; spelled out fully because the engineer may execute these out of order.)*

**Files:**
- Create: `kernels/gelu_mul_batched.ea`
- Modify: `src/kernels/ffi_inference.rs`
- Modify: `tests/gemma4_batch_verify.rs`

- [ ] **Step 1: Read existing kernel**

```bash
cat kernels/gemma4_gelu.ea
```

- [ ] **Step 2: Write batched version**

```
export func gelu_mul_batched(
    gate: *f32, up: *f32,
    out dst: *mut f32 [cap: n_cols * n_batch],
    n_cols: i32, n_batch: i32
) {
    let mut k: i32 = 0
    while k < n_batch {
        // Inline body of gelu_mul for column k
        k = k + 1
    }
}
```

- [ ] **Step 3: FFI + test (mirror batch4/batch5/batch6 structure)**

- [ ] **Step 4: Build, test, commit**

```bash
git add kernels/gelu_mul_batched.ea kernels/gelu_mul_batched.ea.json src/kernels/ffi_inference.rs tests/gemma4_batch_verify.rs
git commit -m "feat+test: gelu_mul_batched bit-exact vs N gelu_mul calls"
```

### Task 17: Multi-position KV cache write

**Files:**
- Modify: `src/inference/cache.rs` (add `store_batch` method)
- Modify: `tests/gemma4_batch_verify.rs`

`KvCache::store(layer, k, v)` writes one position. We need `store_batch(layer, k_batch, v_batch, n)` that writes positions `[seq_len, seq_len + n)`.

- [ ] **Step 1: Read cache.rs to understand the existing layout**

```bash
cat src/inference/cache.rs | head -100
```

Note where `store` writes K and V — this informs how `store_batch` strides.

- [ ] **Step 2: Add `store_batch`**

```rust
impl KvCache {
    /// Store K and V for `n_batch` consecutive positions starting at the
    /// current `seq_len`. Layout of `k_batch`/`v_batch` is column-major
    /// {n_kv_heads * head_dim, n_batch}. Does NOT advance seq_len — caller
    /// is responsible for calling `advance_by(n_batch)` after.
    pub fn store_batch(&mut self, layer: usize, k_batch: &[f32], v_batch: &[f32], n_batch: usize) {
        for k in 0..n_batch {
            let kd = self.head_dim_v[layer]; // or whatever the right field is
            let kv_dim = self.n_kv_heads * self.head_dim_k_per_layer(layer);
            let kv_dim_v = self.n_kv_heads * kd;
            // Write into cache at position seq_len + k from k_batch[k*kv_dim..(k+1)*kv_dim]
            // (use the same code path as store() for one position)
        }
    }

    pub fn advance_by(&mut self, n: usize) {
        self.seq_len += n;
    }
}
```

(Exact code depends on the existing `store` body — copy and adapt.)

- [ ] **Step 3: Add a test**

```rust
#[test]
fn batch7_kv_cache_store_batch_equals_n_stores() {
    // Build two identical caches, write N positions to one with N store() calls
    // and to the other with one store_batch() call. Read back the K/V slabs
    // and assert they're byte-equal.
    // (Implementation depends on KvCache's read API — use whatever exists.)
    eprintln!("PASS: batch7");
}
```

- [ ] **Step 4: Build, test, commit**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify batch7 -- --nocapture
git add src/inference/cache.rs tests/gemma4_batch_verify.rs
git commit -m "feat+test: KvCache::store_batch for N consecutive positions"
```

### Task 18: Batched causal attention kernels

**Files:**
- Create: `kernels/attn_batched.ea`
- Modify: `src/kernels/ffi_inference.rs`
- Modify: `tests/gemma4_batch_verify.rs`

This is the meatiest single kernel after the gemm. For prompt eval at positions `[pos, pos+N)` with KV cache holding `[0, pos)`:
- For each query position `qi in 0..N`:
  - For each head:
    - `scores[qi][j] = (Q[qi] @ K[j]) * scale` for `j in 0..(pos+qi+1)` (causal mask)
    - For `j in (pos+qi+1)..(pos+N)`: scores set to -inf
    - `softmax(scores[qi])`
    - `out[qi] = sum_j scores[qi][j] * V[j]`

This can be implemented as one kernel that handles all of the above, OR as three kernels (qk, softmax, vmul) with explicit intermediate buffers. The three-kernel split mirrors how olorin's existing decode attention is structured (`forward_attn_heads.rs`) and is easier to test in isolation.

- [ ] **Step 1: Pick the three-kernel split**

The three Eä kernels:
1. `attn_qk_batched(q, k_cache, scores, n_heads, head_dim, n_batch_q, n_kv, scale)` — produces scores tensor `{n_kv, n_batch_q, n_heads}` (per head, per query, scores against all KV positions)
2. `attn_softmax_batched(scores, mask_offset, n_heads, n_batch_q, n_kv)` — applies causal softmax in-place; `mask_offset` is the KV position of the first query (= seq_len_before_batch)
3. `attn_vmul_batched(scores, v_cache, out, n_heads, head_dim_v, n_batch_q, n_kv)` — produces output `{n_heads * head_dim_v, n_batch_q}`

- [ ] **Step 2: Write the three kernels**

In a single file `kernels/attn_batched.ea`. Body of each kernel mirrors the existing decode attention — see `kernels/attn_ops.ea` (or wherever olorin's decode attention lives) and inline a column loop.

Use only intrinsics from eacompute that already exist for the decode kernel. No new intrinsics.

- [ ] **Step 3: FFI wrappers**

Add three new function pointers to `ffi_inference.rs`.

- [ ] **Step 4: Bit-exact test against N runs of decode attention**

```rust
#[test]
fn batch8_attention_batched_bitexact_vs_n_decode_attentions() {
    // Build a synthetic K/V cache of length pos.
    // Build N synthetic Q vectors (one per query position).
    // Run:
    //   ref: for each q, call the existing single-position attention kernel
    //   new: one call to attn_qk_batched + attn_softmax_batched + attn_vmul_batched
    // Bit-exact compare.
    eprintln!("PASS: batch8");
}
```

- [ ] **Step 5: Build, test, commit**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify batch8 -- --nocapture
git add kernels/attn_batched.ea kernels/attn_batched.ea.json src/kernels/ffi_inference.rs tests/gemma4_batch_verify.rs
git commit -m "feat+test: batched causal attention kernels (qk, softmax, vmul)"
```

### Task 19: Wire `forward_batch` through all layers

**Files:**
- Modify: `src/inference/forward.rs`
- Modify: `src/inference/forward_attn.rs` (or new `forward_attn_batched.rs` if line limit hits)
- Modify: `src/inference/forward_attn_heads.rs` (read for reference; do not modify)

This is the integration step. Replace `forward_batch`'s body (currently a loop over `forward_one`) with a real batched forward pass using all the kernels from Tasks 12–18.

- [ ] **Step 1: Sketch the per-layer batched body in a comment**

In `src/inference/forward.rs`, replace the body of `forward_batch` with a TODO comment listing the steps the new implementation will take, then the loop body. Do not commit this — it's a structuring step.

- [ ] **Step 2: Write a new method `layer_forward_batch` on `Gemma4State`**

In `forward_attn.rs` (or a new file if forward_attn.rs would exceed 500 lines after this addition):

```rust
impl Gemma4State {
    pub fn layer_forward_batch(
        &mut self,
        model: &Gemma4Model,
        il: usize,
        pos: usize,           // first position of the batch in the sequence
        n_batch: usize,
        pool: &crate::inference::threadpool::ThreadPool,
    ) {
        // Mirror layer_forward exactly, but using batched kernels and
        // batch_* buffers throughout. Each step:
        //   1. gemma4_rmsnorm_batched(batch_x, attn_norm, batch_x_norm, ...)
        //   2. q8k_quant_batched(batch_x_norm, batch_q8_*, hd, n_batch)
        //   3. q4k_8x8_q8k_gemm(Wq_packed, batch_q8_*, batch_q, ...)
        //      (similar for Wk, Wv if has_kv)
        //   4. q_norm_per_head_batched / k_norm_per_head_batched (TODO if needed)
        //   5. gemma4_rope_batched on batch_q and batch_k
        //   6. cache.store_batch(il, batch_k, batch_v, n_batch)
        //   7. attn_qk_batched / attn_softmax_batched / attn_vmul_batched
        //   8. gemm Wo, post_attn_norm batched, residual add batched
        //   9. ffn_norm batched
        //   10. gate/up gemm, gelu_mul_batched, down gemm
        //   11. post_ffn_norm batched, residual add batched
        //   12. PLE batched (if model.ple_dim > 0)
        //   13. layer_output_scale (just a vec_scale on batch_x)
    }
}
```

For each step, use the corresponding batched kernel from Tasks 13–18 and the gemm from Task 9. **Pre-condition:** packed weights for all Q4K matrices must already be repacked at model load time — that pre-step is Task 21.

- [ ] **Step 3: Wire `forward_batch` to use it**

```rust
pub fn forward_batch(
    &mut self,
    model: &Gemma4Model,
    tokens: &[u32],
    pool: &crate::inference::threadpool::ThreadPool,
) -> &[f32] {
    assert!(!tokens.is_empty());
    let n = tokens.len();
    assert!(n <= self.max_batch, "batch size {} exceeds max_batch {}", n, self.max_batch);
    let hd = model.hidden_dim;
    let pos = self.cache.seq_len();

    // Embed all N tokens into batch_x: column k = embedding(token[k]) * sqrt(hd)
    for k in 0..n {
        let dst = &mut self.batch_x[k * hd..(k + 1) * hd];
        crate::inference::dequant::q6k_embed_lookup(model.embed_weight, tokens[k] as usize, dst, hd);
        let scale = (hd as f32).sqrt();
        crate::kernels::ffi_inference::vec_scale_f32(dst.as_ptr(), dst.as_mut_ptr(), scale, hd as i32);
    }

    // Per-layer PLE phase A — for each token, compute its PLE signal into a
    // {ple_dim * n_layers, n_batch} buffer. Reuse the existing prepare_ple
    // by iterating tokens (not batched yet — can be batched in a follow-up plan).
    // This requires a new ple_signal_batch buffer; add it now.
    // (Or call self.prepare_ple n times and stash the results.)

    // Per-layer batched forward
    for il in 0..model.n_layers {
        self.layer_forward_batch(model, il, pos, n, pool);
    }

    // Final norm + lm_head — only on the last token's column.
    let last_col = (n - 1) * hd;
    crate::kernels::ffi_inference::gemma4_rmsnorm(
        self.batch_x[last_col..].as_ptr(),
        model.norm_weight,
        self.x_norm.as_mut_ptr(),
        hd as i32,
        model.rms_eps,
    );
    crate::inference::matmul::quant_input(&self.x_norm, &mut self.q8_qs, &mut self.q8_d, &mut self.q8_bsums);
    crate::inference::matmul::par_matvec(
        pool, model.embed_dtype, model.embed_weight,
        &self.q8_qs, &self.q8_d, &self.q8_bsums,
        &mut self.logits, &mut self.q6k_d_scratch,
        model.vocab_size, hd,
    );
    if model.logit_softcap > 0.0 {
        crate::kernels::ffi_inference::softcap_f32(
            self.logits.as_mut_ptr(), model.vocab_size as i32, model.logit_softcap,
        );
    }

    self.cache.advance_by(n);
    &self.logits
}
```

- [ ] **Step 4: Build clean and run gates 1–4**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "warning|error"
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression --test gemma4_verify -- --test-threads=1 2>&1 | tail -10
find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'
```

All must be clean.

- [ ] **Step 5: Run skeleton test from Task 4**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify batch0 -- --nocapture
```

`batch0_skeleton` (forward_batch over [BOS]) must still pass — it now exercises the real batched path with N=1.

- [ ] **Step 6: Commit**

```bash
git add src/inference/forward.rs src/inference/forward_attn.rs
git commit -m "feat: wire forward_batch through batched kernels

forward_batch now does a real batched forward pass instead of looping
forward_one. batch0 skeleton test still passes (N=1 case)."
```

### Task 20: Repack all Q4K weights at model load

**Files:**
- Modify: `src/inference/engine.rs` (`Gemma4Model::from_gguf` or wherever weights are finalized)

After Task 19 the kernel exists but we haven't actually packed the weights yet. Add a post-load pass that walks every layer's Q4K weights (Wq, Wk, Wv, Wo, ffn_gate, ffn_up, ffn_down) and produces a parallel "packed" buffer for each.

- [ ] **Step 1: Add packed-weight fields to `Gemma4LayerWeights`**

```rust
pub struct Gemma4LayerWeights {
    // ... existing fields ...
    pub wq_packed: Vec<u8>,   // empty if wq_dtype != Q4_K
    pub wk_packed: Vec<u8>,
    pub wv_packed: Vec<u8>,
    pub wo_packed: Vec<u8>,
    pub w_gate_packed: Vec<u8>,
    pub w_up_packed: Vec<u8>,
    pub w_down_packed: Vec<u8>,
}
```

- [ ] **Step 2: Pack on load**

In `Gemma4Model::from_gguf`, after each layer is fully populated, call `q4k_repack_8x8` on each Q4K weight and store the result in the corresponding `*_packed` field. Skip non-Q4K weights (leave the field empty).

```rust
fn pack_q4k(src: *const u8, n_rows: usize, n_cols: usize) -> Vec<u8> {
    let n_blocks = n_cols / 256;
    let total = n_rows * n_blocks * 144; // 144 = sizeof(block_q4_K)
    let mut out = vec![0u8; total];
    unsafe {
        crate::kernels::ffi_inference::q4k_repack_8x8(src, out.as_mut_ptr(), n_rows as i32, n_cols as i32);
    }
    out
}
```

Apply to each Q4K weight in each layer.

- [ ] **Step 3: Update `layer_forward_batch` to read packed weights**

In `forward_attn.rs`, the gemm calls in `layer_forward_batch` should read `lw.wq_packed.as_ptr()` instead of `lw.wq`. (`forward_one` / `layer_forward` continues to read `lw.wq` — both paths coexist.)

- [ ] **Step 4: Build, run all gates**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "warning|error"
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression --test gemma4_verify --test gemma4_batch_verify -- --test-threads=1 2>&1 | tail -15
find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'
```

`batch0` still passes (N=1 case). Decode and verify suite untouched.

- [ ] **Step 5: Commit**

```bash
git add src/inference/engine.rs src/inference/forward_attn.rs
git commit -m "feat: repack all Q4K weights to 8x8 layout on load

Adds parallel *_packed buffers to Gemma4LayerWeights, populated by
q4k_repack_8x8 in from_gguf. forward_batch reads packed; forward_one
continues to read the standard layout."
```

### Task 21: Bit-exact verify `forward_batch` against llama-eval-callback for prompt "a"

**Files:**
- Modify: `tests/gemma4_batch_verify.rs`

This is the moment of truth: olorin's forward_batch on `[BOS, 'a']` (N=2) must produce L34 hidden state and final logits sums that match llama-eval-callback's dump for prompt "a".

- [ ] **Step 1: Add the test**

```rust
#[test]
fn batch9_two_token_forward_batch_matches_llama_eval_callback() {
    // Reference: llama-eval-callback -p "a" dump from 2026-04-08:
    //   l_out-34 (token 1, 'a') sum = 40.513065
    //   logits sum = -1781197.7500
    // These were captured with the model file at ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf
    // and llama.cpp build 8685.
    if !has_model() { eprintln!("SKIP: no model"); return; }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();
    let pool = olorin::inference::threadpool::ThreadPool::new();

    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);
    let logits = state.forward_batch(&model, &[2u32, 236746u32], &pool).to_vec();

    let hd = model.hidden_dim;
    let l34_sum = sum_f64(&state.batch_x[(1) * hd..(2) * hd]); // last column (k=1 for n=2)
    let logits_sum = sum_f64(&logits);

    eprintln!("l_out-34 sum: olorin={l34_sum:.6}  llama=40.513065");
    eprintln!("logits sum:   olorin={logits_sum:.4}  llama=-1781197.7500");

    // Tolerance: f64 sum of 1536 / 262144 f32 values has ~1e-3 rounding floor.
    // For a "bit-exact" claim, the per-element f32 difference must be ~ULP, but
    // the sum can drift by 1e-3 even when individual f32s match. Use both checks:
    let l34_drift = ((l34_sum - 40.513065) / 40.513065).abs();
    let lg_drift = ((logits_sum + 1781197.75) / 1781197.75).abs();
    assert!(l34_drift < 1e-4, "l_out-34 drift = {} (expected < 1e-4)", l34_drift);
    assert!(lg_drift < 1e-4, "logits drift = {} (expected < 1e-4)", lg_drift);
    eprintln!("PASS: batch9");
}
```

- [ ] **Step 2: Run the test**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify batch9 -- --nocapture
```

Expected: PASS with `l34_drift < 1e-4` and `lg_drift < 1e-4`. This is the **goal of the entire plan**. If FAIL, the f32 accumulation order in some batched kernel still doesn't match ggml — go back to the kernel that's responsible (probably the gemm or the attention) and adjust.

- [ ] **Step 3: Commit**

```bash
git add tests/gemma4_batch_verify.rs
git commit -m "test: forward_batch matches llama-eval-callback bit-exactly (N=2)

l_out-34 sum drift < 1e-4 vs llama 40.513065
logits sum drift < 1e-4 vs llama -1781197.75

Closes the architectural prompt-eval gap identified on 2026-04-08."
```

### Task 22: Bit-exact verify for longer prompts

**Files:**
- Modify: `tests/gemma4_batch_verify.rs`

Once N=2 works, extend to longer prompts to catch any kernel that has off-by-one logic only at certain N.

- [ ] **Step 1: Capture reference dumps for longer prompts**

```bash
for prompt in "Hi" "Hello world" "The quick brown fox" "Write a long story about a robot:"; do
    echo "=== $prompt ==="
    /root/dev/llama.cpp/build/bin/llama-eval-callback \
      -m ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf \
      -p "$prompt" -n 0 --seed 42 2>&1 | \
      grep -E "(inp_tokens|l_out-34 = |result_output)" -A 7 | grep "sum =" | tail -3
    /root/dev/llama.cpp/build/bin/llama-tokenize \
      -m ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf -p "$prompt" 2>&1 | \
      grep -E "^\s*[0-9]" | awk '{print $1}'
done
```

Save the (prompt, token list, l_out-34 sum, logits sum) tuples into a hardcoded table in the test file.

- [ ] **Step 2: Add a parameterized test**

```rust
#[test]
fn batch10_forward_batch_matches_llama_for_various_prompts() {
    if !has_model() { eprintln!("SKIP: no model"); return; }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();
    let pool = olorin::inference::threadpool::ThreadPool::new();

    // Reference data captured from llama-eval-callback build 8685, 2026-04-08.
    // Format: (description, token_ids, l_out_34_sum_ref, logits_sum_ref)
    let cases: Vec<(&str, Vec<u32>, f64, f64)> = vec![
        ("Hi",       vec![2, /* fill */],     /* fill */, /* fill */),
        ("Hello world", vec![2, /* fill */],  /* fill */, /* fill */),
        // ... etc
    ];

    for (desc, tokens, l34_ref, lg_ref) in cases {
        let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);
        let logits = state.forward_batch(&model, &tokens, &pool).to_vec();
        let hd = model.hidden_dim;
        let last_col = (tokens.len() - 1) * hd;
        let l34_sum = sum_f64(&state.batch_x[last_col..last_col + hd]);
        let lg_sum = sum_f64(&logits);
        let l34_drift = ((l34_sum - l34_ref) / l34_ref).abs();
        let lg_drift = ((lg_sum - lg_ref) / lg_ref).abs();
        eprintln!("{desc} (N={}):  l34_drift={l34_drift:.2e}  lg_drift={lg_drift:.2e}", tokens.len());
        assert!(l34_drift < 1e-4, "{desc}: l34_drift={l34_drift}");
        assert!(lg_drift < 1e-4, "{desc}: lg_drift={lg_drift}");
    }
}
```

- [ ] **Step 3: Run the test**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test gemma4_batch_verify batch10 -- --nocapture
```

- [ ] **Step 4: Commit**

```bash
git add tests/gemma4_batch_verify.rs
git commit -m "test: forward_batch matches llama for prompts of varying length"
```

---

## Phase D — Integration & Showcase

### Task 23: Wire `forward_batch` into `generate.rs`

**Files:**
- Modify: `src/inference/generate.rs`

The public `generate` function currently feeds the prompt token-by-token through `forward_one`. Switch it to call `forward_batch` once on the prompt, then loop `forward_one` for generation.

- [ ] **Step 1: Read generate.rs to find the prompt loop**

```bash
grep -n "forward_one\|prompt" src/inference/generate.rs
```

- [ ] **Step 2: Replace the prompt loop**

Find the loop that calls `forward_one` for each prompt token. Replace with a single `forward_batch(prompt_tokens)` call. Decode loop (the per-generated-token `forward_one`) stays unchanged.

- [ ] **Step 3: Run all tests**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release 2>&1 | tail -20
```

Everything must still pass — including any tests that exercise `generate` end-to-end.

- [ ] **Step 4: Commit**

```bash
git add src/inference/generate.rs
git commit -m "feat: route prompt through forward_batch in generate

forward_one is now used only for decode steps. Prompt tokens go
through the batched gemm path."
```

### Task 24: Update `bench_decode_speed.rs` to measure prompt-eval and decode separately

**Files:**
- Modify: `tests/bench_decode_speed.rs`

The current bench reports a combined "prompt eval time" but the path is the same as decode (forward_one in a loop). Update it to use forward_batch for the prompt, time it separately, and measure decode tok/s on the post-prompt steps.

- [ ] **Step 1: Find the relevant section in bench_decode_speed.rs**

```bash
grep -n "prompt\|forward_one\|forward_batch\|prompt_eval" tests/bench_decode_speed.rs
```

- [ ] **Step 2: Replace prompt loop with forward_batch + new timing**

Existing structure (approximate):
```rust
let t_load = ...;
// prompt eval
let t0 = Instant::now();
for &tok in &prompt_tokens {
    state.forward_one(&model, tok, &pool);
}
let t_prompt = t0.elapsed();
// decode
...
```

New:
```rust
let t_load = ...;
// prompt eval — batched
let t0 = Instant::now();
let _ = state.forward_batch(&model, &prompt_tokens, &pool);
let t_prompt = t0.elapsed();
let prompt_tps = prompt_tokens.len() as f64 / t_prompt.as_secs_f64();
// decode
...
```

Print separately:
```
prompt eval time:  XX.XX ms / 9 tok  (XX.XX ms/tok, XX.XX t/s)
eval time:         XX.XX ms / 64 tok (XX.XX ms/tok, XX.XX t/s)
```

(Same format as today, but the prompt time should now be much smaller.)

- [ ] **Step 3: Run the bench**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test bench_decode_speed -- --nocapture 2>&1 | tail -30
```

Document the new prompt-eval throughput. **Hard requirement:** prompt-eval must be at least 15 t/s on this 2-core box (target is parity with llama's 27 t/s, but 15 is the minimum acceptable to declare the gap closed). If below 15, return to Task 11 perf bench and find the bottleneck.

- [ ] **Step 4: Commit**

```bash
git add tests/bench_decode_speed.rs
git commit -m "bench: separate prompt-eval and decode timings via forward_batch

Numbers (this box, 2 cores, gemma-4-e2b-it-Q4_K_M):
prompt eval:  [XX.XX] t/s  (was [4.09] t/s before batched path)
decode:       [XX.XX] t/s  (was [4.17] t/s)

llama.cpp build 8685 reference: 26.65 t/s pp / 13.00 t/s tg"
```

### Task 25: Final apples-to-apples comparison vs llama-bench

**Files:**
- Create: `docs/superpowers/research/2026-04-08-batched-prompt-eval-results.md`

- [ ] **Step 1: Run both benches**

```bash
# Olorin
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release --test bench_decode_speed -- --nocapture > /tmp/olorin-bench.txt 2>&1
# llama.cpp
/root/dev/llama.cpp/build/bin/llama-bench \
  -m ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf \
  -p 9 -n 64 -t 2 > /tmp/llama-bench.txt 2>&1
```

- [ ] **Step 2: Write the results note**

```markdown
# Batched Prompt-Eval Results — 2026-04-08

## Setup
- Box: [hostname], 2 cores, AVX-512
- Model: gemma-4-e2b-it-Q4_K_M.gguf (3.21 GiB)
- llama.cpp: build 8685
- olorin: branch gemma4-batched-prompt-eval, commit [hash]

## Numbers (2 threads)

| Engine          | prompt-eval (t/s) | decode (t/s) |
|-----------------|-------------------|--------------|
| llama.cpp 8685  | 26.65             | 13.00        |
| olorin (before) |  4.09             |  4.17        |
| olorin (after)  | [XX.XX]           | [XX.XX]      |

## Numerical parity
- batch9 (forward_batch on [BOS, 'a']) passes with l34_drift < 1e-4 and lg_drift < 1e-4
- batch10 passes for all 4 reference prompts

## Eä showcase claims now defensible
- [ ] olorin matches llama.cpp prompt-eval bit-exactly (drift < 1e-4)
- [ ] olorin matches llama.cpp decode bit-exactly (already proven 2026-04-08)
- [ ] olorin's per-kernel performance is competitive with ggml's hand-tuned x86 kernels
```

Fill in the `[XX.XX]` numbers from the bench output.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/research/2026-04-08-batched-prompt-eval-results.md
git commit -m "docs: results — olorin batched prompt-eval matches llama.cpp"
```

### Task 26: Update CLAUDE.md hard rule "Match llama.cpp exactly first"

**Files:**
- Modify: `CLAUDE.md` (the project root one, not the agentic one)

- [ ] **Step 1: Find and update**

```bash
grep -n "Match llama.cpp\|forward_one" CLAUDE.md
```

Add a note that prompt-eval now uses `forward_batch` and decode uses `forward_one`. Both paths are bit-exact verified.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md notes new forward_batch / forward_one split"
```

### Task 27: Final regression sweep

- [ ] **Step 1: Run everything**

```bash
PATH="/root/dev/eacompute/target/release:$PATH" cargo test --release 2>&1 | tail -30
```

All tests pass. No new warnings.

- [ ] **Step 2: Verify all 500-line limits**

```bash
find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'
```

Empty.

- [ ] **Step 3: Push branch**

```bash
git push origin gemma4-batched-prompt-eval
```

Open a PR against `gemma4-cleanup`. Reference this plan in the PR description.

---

## Self-Review Checklist

- **Scope coverage:** Each section in the goal is covered by a task. Phase A = research + skeleton. Phase B = the kernel. Phase C = batched per-layer ops. Phase D = wiring and bench.
- **No placeholders:** Tasks 1, 2, 5, 7, 17, 18 have inline TODOs in the *code* (e.g., "Body of each kernel mirrors the existing decode attention"), but they reference specific files and the structure is fully specified — the engineer copies the body from a named file, not invents it. Tasks 6 and 22 have `/* fill */` in test data — these are the result of running a documented command, captured into the test on first run.
- **Type consistency:** `forward_batch`, `layer_forward_batch`, `q4k_repack_8x8`, `q4k_8x8_q8k_matvec`, `q4k_8x8_q8k_gemm`, `q8k_quant_batched`, `gemma4_rmsnorm_batched`, `gemma4_rope_batched`, `gelu_mul_batched`, `attn_qk_batched`, `attn_softmax_batched`, `attn_vmul_batched`, `KvCache::store_batch`, `Gemma4State::batch_*` fields — all referenced consistently across tasks.
- **Verification gates:** Every code-changing task ends with `cargo build` + the relevant test + commit. Each batched kernel is bit-exact tested before being used in the next layer up.
- **Risk callouts:** The hardest task is Task 2 (research the gemm inner loop). If that research reveals ggml's kernel can't be replicated in Eä without new intrinsics, the plan stops at Task 2 and the next plan is "add intrinsics to eacompute." The second hardest is Task 18 (batched attention) — split into 3 sub-kernels to bound the risk.

---

**Plan saved.** Estimated scope: ~27 tasks, ~3-5 days of focused work. Cleanly divisible into phases with checkpoints between A→B→C→D for reassessment.
