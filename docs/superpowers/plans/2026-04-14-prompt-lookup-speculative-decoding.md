# Prompt-Lookup Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship self-speculative decoding via n-gram prompt lookup that achieves ≥1.8× speedup on code prompts, ≤5% regression on free-form chat, and bit-identical output under greedy (temperature=0) decoding.

**Architecture:** Per decode step, sample `A_0` normally, look up a K-1 token draft from the live context, run `forward_batch` to verify, accept the longest prefix whose argmax matches the draft, rewind KV cursor, write correction. Two Ea SIMD kernels do the novel work: `ngram_lookup` (SIMD-compare over token buffer) and `verify_draft` (fused horizontal argmax + find-first-mismatch).

**Tech Stack:** Rust (Olorin binary), Ea SIMD (via eacompute, x86 SSE2 + ARM NEON), dynamic kernel loading via `libloading`. Build system auto-discovers `kernels/*.ea`.

**Spec:** `docs/superpowers/specs/2026-04-14-prompt-lookup-speculative-decoding-design.md`

**Ground rules (from `CLAUDE.md`):**
- Before writing any `.ea` code, invoke the `ea-lookup` skill and/or grep `eacompute/src/typeck/intrinsics*.rs` + `eacompute/src/codegen/simd*.rs` to verify available intrinsics. Do NOT guess.
- No file over 500 lines. No fake functions. Delete, don't comment. Every feature proven by an end-to-end test.
- Both x86 (SSE2/AVX2) and ARM (NEON) kernel variants must ship together. Build fails on one target = rollback.
- Commit after every task. Run `cargo build --release` and the relevant tests before each commit.

**Build command (used repeatedly below):**
```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release
```

**Test command base:**
```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --release
```

---

## File Structure

**New files:**
- `kernels/ngram_lookup.ea` — x86 SSE2 variant
- `kernels/ngram_lookup_arm.ea` — ARM NEON variant
- `kernels/verify_draft.ea` — x86 SSE2 variant
- `kernels/verify_draft_arm.ea` — ARM NEON variant
- `tests/ngram_lookup.rs` — kernel unit tests
- `tests/verify_draft.rs` — kernel unit tests
- `tests/speculative_parity.rs` — greedy parity E2E
- `tests/bench_speculative.rs` — speedup/accept-rate bench
- `tests/speculative_integration.rs` — six-prompt regression

**Modified files:**
- `src/kernels/ffi_inference_types.rs` — add `NgramLookupFn`, `VerifyDraftFn` typedefs
- `src/kernels/ffi_inference.rs` — load both kernels, add safe wrapper fns
- `src/inference/generate.rs` — add speculative decode branch
- `src/inference/forward.rs` — only if `forward_batch` needs a minor tweak for verify; most likely no changes
- `CLAUDE.md` / project memory — optional note on the feature

**Architectural decisions already made (from spec):**
- Reuse existing `forward_batch` for verify — no new forward path.
- KV rollback is `kv.seq_len = S + j`. No cache zeroing.
- Correction token KV is written by a second `forward_one` after rewind.

---

### Task 1: Verify Ea intrinsics available for `ngram_lookup`

**Files:** (reference only — no edits)
- Grep: `/home/peter/projects/eacompute/src/typeck/intrinsics*.rs`
- Grep: `/home/peter/projects/eacompute/src/codegen/simd*.rs`
- Run: `eabrain status && eabrain ref splat && eabrain ref load && eabrain ref i32x4`

- [ ] **Step 1: Invoke `ea-lookup` skill**

Confirm availability of: `splat`, `load` (typed, for `i32x4`), `==` (SIMD equality), broadcast-compare, mask-any, `store`, `while` / `if` / early return, pointer indexing for `*i32` and `*u32` (token buffers).

- [ ] **Step 2: Record findings**

If any intrinsic is missing, stop and report to user — do NOT fall back to scalar or invent an intrinsic. User will extend eacompute or change the design.

If all present: proceed to Task 2.

---

### Task 2: Write `ngram_lookup.ea` (x86 SSE2)

**Files:**
- Create: `kernels/ngram_lookup.ea`

**Contract (must match `ffi_inference_types.rs` in Task 4):**

```
// Inputs are u32 token IDs viewed as i32 (same bit pattern; Gemma 4 vocab is 262144 < 2^31).
// ctx_ptr:  *i32, ctx_len tokens
// key_ptr:  *i32, 3 tokens (key[0], key[1], key[2]) = last 3 tokens of ctx (or first 2
//           of key are key[1], key[2] when we do the N=2 fallback pass — see below)
// k:        i32   — max draft tokens to copy
// out_ptr:  *mut i32
// returns:  i32   — number of draft tokens written (0 = no match)
//
// Semantics:
//   Pass 1 (N=3): scan ctx right-to-left for i where ctx[i]==key[0],
//                 ctx[i+1]==key[1], ctx[i+2]==key[2]. On match, copy
//                 ctx[i+3 .. min(i+3+k, ctx_len)] to out and return the
//                 number copied.
//   Pass 2 (N=2): if pass 1 found nothing, repeat scanning for
//                 ctx[i]==key[1] && ctx[i+1]==key[2]. Copy ctx[i+2..] up
//                 to k tokens.
//   Return 0 if neither pass hits.
//
// Implementation hint:
//   Outer loop walks 4-wide chunks (SSE2 i32x4) right-to-left over ctx.
//   Broadcast key[0] as i32x4, compare-equal against load(ctx, i), collect
//   match mask. For any lane that matches, scalar-verify ctx[i+1], ctx[i+2],
//   and on full match, copy with memcpy-like loop and return.
```

- [ ] **Step 1: Write the failing test first**

See Task 5 — the test file will fail before the kernel exists. If the current TDD workflow blocks on that, write the kernel first and then the test. (Kernel code cannot be unit-tested without the FFI wrapper, so the true TDD cycle here is: kernel → FFI → Rust test.)

- [ ] **Step 2: Write the kernel**

Target: `kernels/ngram_lookup.ea`. Skeleton based on `chacha20_search_v2.ea`'s broadcast-compare pattern. Use typed `load(ctx_ptr, i): i32x4` for 4-token windows. Output kernel MUST declare an FFI entry with exact name `ngram_lookup`.

**Do not write the kernel without completing Task 1 first.** If any intrinsic is missing, stop.

- [ ] **Step 3: Build and confirm kernel compiles**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -15
```

Expected: build succeeds. `target/release/build/olorin-*/out/` contains `libngram_lookup.so`. Build.rs auto-discovers it.

- [ ] **Step 4: Commit**

```bash
git add kernels/ngram_lookup.ea
git commit -m "feat(kernel): ngram_lookup.ea x86 SSE2 — longest-match draft lookup"
```

---

### Task 3: Write `ngram_lookup_arm.ea` (ARM NEON)

**Files:**
- Create: `kernels/ngram_lookup_arm.ea`

Contract: identical to `ngram_lookup.ea`. Same entry name — eacompute build selects the `_arm` variant on aarch64 targets (see existing pattern in `chacha20_search_v2_arm.ea`, `bf16_matvec_arm.ea`, etc.).

- [ ] **Step 1: Verify NEON intrinsics for broadcast-compare on `i32x4`**

Re-run `ea-lookup` for NEON-specific forms if needed.

- [ ] **Step 2: Write `kernels/ngram_lookup_arm.ea`**

Same algorithm as x86 variant, NEON primitives. Match `chacha20_search_v2_arm.ea` shape.

- [ ] **Step 3: Build (cross-check with `cargo check` on current host first)**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -10
```

Expected: build still succeeds on x86 host (the ARM variant is picked only on aarch64 targets). If `build.rs` compiles all variants unconditionally, the ARM kernel still must compile cleanly via ea.

- [ ] **Step 4: Commit**

```bash
git add kernels/ngram_lookup_arm.ea
git commit -m "feat(kernel): ngram_lookup_arm.ea ARM NEON variant"
```

---

### Task 4: FFI typedef + wrapper + kernel-table entry for `ngram_lookup`

**Files:**
- Modify: `src/kernels/ffi_inference_types.rs` — add typedef
- Modify: `src/kernels/ffi_inference.rs` — load lib + safe wrapper

- [ ] **Step 1: Add FFI typedef**

In `src/kernels/ffi_inference_types.rs`, append:

```rust
pub type NgramLookupFn = unsafe extern "C" fn(
    ctx_ptr: *const i32,
    ctx_len: i32,
    key_ptr: *const i32,
    k: i32,
    out_ptr: *mut i32,
) -> i32;
```

- [ ] **Step 2: Register in `KernelTableInference`**

In `src/kernels/ffi_inference.rs`, add field to `KernelTableInference`:

```rust
pub ngram_lookup: NgramLookupFn,
```

Add loader line inside `load_inference_kernels`:

```rust
let lib_ngram_lookup = load_best("ngram_lookup")?;
let ngram_lookup: NgramLookupFn = unsafe {
    *lib_ngram_lookup.get(b"ngram_lookup\0").map_err(|e| format!("ngram_lookup symbol: {e}"))?
};
```

Push `lib_ngram_lookup` into the `libs` vec (mirror existing pattern) and add `ngram_lookup` to the struct init.

- [ ] **Step 3: Add safe Rust wrapper**

Bottom of `src/kernels/ffi_inference.rs`, mirroring the style of existing wrappers:

```rust
/// Longest-match n-gram lookup over context tokens.
/// `ctx` is the live sequence (prompt + generated). `key` is the last 3 tokens.
/// Writes up to `k` draft tokens to `out`. Returns the number written (0 = no match).
pub fn ngram_lookup(ctx: &[u32], key: &[u32; 3], k: usize, out: &mut [u32]) -> usize {
    assert!(out.len() >= k);
    let ctx_i32 = ctx.as_ptr() as *const i32;
    let key_i32 = key.as_ptr() as *const i32;
    let out_i32 = out.as_mut_ptr() as *mut i32;
    let n = unsafe { (k().ngram_lookup)(ctx_i32, ctx.len() as i32, key_i32, k as i32, out_i32) };
    n.max(0) as usize
}
```

- [ ] **Step 4: Build**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
```

Expected: no errors, no warnings about the new code.

- [ ] **Step 5: Commit**

```bash
git add src/kernels/ffi_inference.rs src/kernels/ffi_inference_types.rs
git commit -m "feat(ffi): wire ngram_lookup kernel"
```

---

### Task 5: Unit tests for `ngram_lookup`

**Files:**
- Create: `tests/ngram_lookup.rs`

- [ ] **Step 1: Write the tests**

```rust
use olorin::kernels::ffi;
use olorin::kernels::ffi_inference::ngram_lookup;

fn init() { ffi::init().unwrap(); }

#[test]
fn match_at_end_returns_next_tokens() {
    init();
    // ctx: [10, 11, 12, 13, 14, 15, 10, 11, 12]
    // key: [10, 11, 12]
    // Expected: match at position 0 AND position 6; right-to-left prefers 6.
    // No tokens after position 8 — so 0 tokens written.
    let ctx: Vec<u32> = vec![10, 11, 12, 13, 14, 15, 10, 11, 12];
    let key = [10, 11, 12];
    let mut out = vec![0u32; 4];
    let n = ngram_lookup(&ctx, &key, 4, &mut out);
    assert_eq!(n, 0);
}

#[test]
fn match_prefers_recent_not_earliest() {
    init();
    // Two matches: positions 0 and 5. Recent (position 5) is preferred.
    // After position 5's "10 11 12", ctx has nothing — 0 tokens.
    // Add some trailing so we can tell which match was picked.
    // ctx: [10, 11, 12, 99, 98, 10, 11, 12, 77, 88]
    let ctx: Vec<u32> = vec![10, 11, 12, 99, 98, 10, 11, 12, 77, 88];
    let key = [10, 11, 12];
    let mut out = vec![0u32; 4];
    let n = ngram_lookup(&ctx, &key, 4, &mut out);
    assert_eq!(n, 2);
    assert_eq!(&out[..2], &[77, 88]);
}

#[test]
fn no_match_returns_zero() {
    init();
    let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let key = [99, 99, 99];
    let mut out = vec![0u32; 4];
    assert_eq!(ngram_lookup(&ctx, &key, 4, &mut out), 0);
}

#[test]
fn n3_miss_n2_hit() {
    init();
    // No 3-gram match. But 2-gram [20,30] appears at position 3.
    // Key: [10, 20, 30] (N=3 key; kernel tries N=3 first, then N=2 on last-2)
    let ctx: Vec<u32> = vec![5, 20, 30, 20, 30, 42, 43];
    let key = [10, 20, 30]; // N=3 miss; N=2 on [20,30] matches at 3 (most recent)
    let mut out = vec![0u32; 4];
    let n = ngram_lookup(&ctx, &key, 4, &mut out);
    assert_eq!(n, 2);
    assert_eq!(&out[..2], &[42, 43]);
}

#[test]
fn context_shorter_than_key() {
    init();
    let ctx: Vec<u32> = vec![1, 2]; // only 2 tokens
    let key = [1, 2, 3];
    let mut out = vec![0u32; 4];
    assert_eq!(ngram_lookup(&ctx, &key, 4, &mut out), 0);
}

#[test]
fn respects_k_limit() {
    init();
    let ctx: Vec<u32> = vec![1, 2, 3, 10, 11, 12, 13, 14, 15];
    // Pattern [1,2,3] at 0; tail is [10,11,12,13,14,15].
    let key = [1, 2, 3];
    let mut out = vec![0u32; 3];
    let n = ngram_lookup(&ctx, &key, 3, &mut out);
    assert_eq!(n, 3);
    assert_eq!(&out[..3], &[10, 11, 12]);
}
```

- [ ] **Step 2: Run tests — expect pass if kernel is correct**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --release --test ngram_lookup -- --nocapture
```

Expected: all 6 tests pass.

If a test fails: debug the kernel against the specific case, fix, rebuild, re-run. Do not weaken the test.

- [ ] **Step 3: Commit**

```bash
git add tests/ngram_lookup.rs
git commit -m "test(kernel): ngram_lookup — 6 cases covering match priority, misses, fallback"
```

---

### Task 6: Verify Ea intrinsics for `verify_draft`

Same approach as Task 1. Critical intrinsics:
- Horizontal reduce-max-with-index over a chunk of f32 lanes
- Running best-value + best-index pattern (keep in two SIMD registers across the vocab loop)
- Argmax-finalize (horizontal max over final register)
- Scalar compare + branch for mismatch detection

- [ ] **Step 1: Invoke `ea-lookup`** — check `argmax`-shape primitives exist. The sampler uses a scalar argmax; see if a SIMD `horizontal_max_index` or equivalent is already present. If not, build argmax out of `.>`, blend/select, and manual horizontal reduction — all standard SIMD.

- [ ] **Step 2: If missing critical intrinsics, stop and report.** Otherwise proceed.

---

### Task 7: Write `verify_draft.ea` (x86)

**Files:**
- Create: `kernels/verify_draft.ea`

**Contract:**

```
// logits_ptr: *f32  (k rows, each of `vocab` f32 values)
// vocab:      i32
// drafts_ptr: *i32  (k-1 draft token ids; index j compares to row j for j=1..k-1)
// k:          i32   (number of rows; k-1 drafts + final unconstrained row)
// out_argmax: *mut i32  (k entries; kernel writes A_1..A_K)
// returns:    i32   first j in 1..k-1 where argmax(row[j-1]) != drafts[j-1];
//                    k if all accepted (full-accept path).
//
// Per-row argmax loop: iterate vocab in SIMD-width chunks, maintain running
// (best_vec, idx_vec) pair. Final horizontal reduce to scalar argmax.
// Early-exit: on mismatch, still write argmaxes for remaining rows until j,
// then return j (engine needs A_j for the correction forward_one).
```

- [ ] **Step 1: Write kernel** following pattern from `softmax_f32.ea` for per-row sweeps and `q4k_dot_q8k.ea` for horizontal reductions.

- [ ] **Step 2: Build**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add kernels/verify_draft.ea
git commit -m "feat(kernel): verify_draft.ea x86 — fused argmax + first-mismatch"
```

---

### Task 8: Write `verify_draft_arm.ea` (ARM NEON)

**Files:**
- Create: `kernels/verify_draft_arm.ea`

Mirror Task 7 with NEON intrinsics. Same contract.

- [ ] **Step 1: Write kernel.**

- [ ] **Step 2: Build and commit.**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
git add kernels/verify_draft_arm.ea
git commit -m "feat(kernel): verify_draft_arm.ea ARM NEON variant"
```

---

### Task 9: FFI wrapper + kernel-table entry for `verify_draft`

**Files:**
- Modify: `src/kernels/ffi_inference_types.rs`
- Modify: `src/kernels/ffi_inference.rs`

- [ ] **Step 1: Add typedef**

```rust
pub type VerifyDraftFn = unsafe extern "C" fn(
    logits_ptr: *const f32,
    vocab: i32,
    drafts_ptr: *const i32,
    k: i32,
    out_argmax: *mut i32,
) -> i32;
```

- [ ] **Step 2: Add field + loader (mirror Task 4, Step 2)**

`pub verify_draft: VerifyDraftFn,` in the struct; `load_best("verify_draft")?` in the loader; push into libs; add to init.

- [ ] **Step 3: Add safe wrapper**

```rust
/// Verify K-1 drafts against per-row argmax of K logits rows.
/// `logits` is row-major K × vocab. `drafts` has K-1 tokens.
/// `out_argmax` receives K argmax values. Returns first mismatch index in 1..K-1,
/// or K for full accept.
pub fn verify_draft(
    logits: &[f32],
    vocab: usize,
    drafts: &[u32],
    k: usize,
    out_argmax: &mut [u32],
) -> usize {
    assert_eq!(logits.len(), k * vocab);
    assert_eq!(drafts.len(), k - 1);
    assert_eq!(out_argmax.len(), k);
    let j = unsafe {
        (k().verify_draft)(
            logits.as_ptr(),
            vocab as i32,
            drafts.as_ptr() as *const i32,
            k as i32,
            out_argmax.as_mut_ptr() as *mut i32,
        )
    };
    j as usize
}
```

- [ ] **Step 4: Build + commit**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
git add src/kernels/ffi_inference.rs src/kernels/ffi_inference_types.rs
git commit -m "feat(ffi): wire verify_draft kernel"
```

---

### Task 10: Unit tests for `verify_draft`

**Files:**
- Create: `tests/verify_draft.rs`

- [ ] **Step 1: Write tests**

```rust
use olorin::kernels::ffi;
use olorin::kernels::ffi_inference::verify_draft;

fn init() { ffi::init().unwrap(); }

/// Build a K×V logits tensor where row r has its max at column `peaks[r]`.
fn make_logits(peaks: &[usize], vocab: usize) -> Vec<f32> {
    let k = peaks.len();
    let mut logits = vec![-1.0f32; k * vocab];
    for (r, &p) in peaks.iter().enumerate() {
        logits[r * vocab + p] = 10.0;
    }
    logits
}

#[test]
fn full_accept_returns_k() {
    init();
    let vocab = 128;
    // K=4: argmaxes [7, 8, 9, 99]. Drafts (K-1=3): [8, 9, 99]. All match.
    let logits = make_logits(&[8, 9, 99, 42], vocab);
    let drafts: Vec<u32> = vec![8, 9, 99];
    let mut out = vec![0u32; 4];
    let j = verify_draft(&logits, vocab, &drafts, 4, &mut out);
    assert_eq!(j, 4);
    assert_eq!(out, vec![8, 9, 99, 42]);
}

#[test]
fn immediate_reject_returns_one() {
    init();
    let vocab = 128;
    // Row 0 argmax 7, draft expected 8 — mismatch at j=1.
    let logits = make_logits(&[7, 20, 30, 40], vocab);
    let drafts: Vec<u32> = vec![8, 20, 30];
    let mut out = vec![0u32; 4];
    let j = verify_draft(&logits, vocab, &drafts, 4, &mut out);
    assert_eq!(j, 1);
    assert_eq!(out[0], 7); // correction token
}

#[test]
fn partial_accept_returns_middle_index() {
    init();
    let vocab = 128;
    // Argmaxes [8, 9, 77, 99]. Drafts [8, 9, 30].
    // j=1: 8 == 8 ✓. j=2: 9 == 9 ✓. j=3: 77 != 30 — mismatch at j=3.
    let logits = make_logits(&[8, 9, 77, 99], vocab);
    let drafts: Vec<u32> = vec![8, 9, 30];
    let mut out = vec![0u32; 4];
    let j = verify_draft(&logits, vocab, &drafts, 4, &mut out);
    assert_eq!(j, 3);
    assert_eq!(&out[..3], &[8, 9, 77]);
}

#[test]
fn realistic_vocab_size() {
    init();
    let vocab = 262144;
    let peaks = [100_000, 200_000, 50_000, 150_000];
    let logits = make_logits(&peaks, vocab);
    let drafts: Vec<u32> = vec![200_000, 50_000, 150_000];
    let mut out = vec![0u32; 4];
    let j = verify_draft(&logits, vocab, &drafts, 4, &mut out);
    assert_eq!(j, 4);
    assert_eq!(out, vec![100_000, 200_000, 50_000, 150_000]);
}
```

- [ ] **Step 2: Run tests**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --release --test verify_draft -- --nocapture
```

Expected: all 4 pass.

- [ ] **Step 3: Commit**

```bash
git add tests/verify_draft.rs
git commit -m "test(kernel): verify_draft — accept/reject/partial + realistic vocab"
```

---

### Task 11: Refactor `generate.rs` decode loop to extract single-token path (no behavior change)

**Files:**
- Modify: `src/inference/generate.rs`

This prepares a clean branch point for the speculative path without changing observable behavior yet.

- [ ] **Step 1: Extract a helper**

Inside `Engine::generate`, pull the per-token body of the decode `for _ in 0..self.max_tokens` loop into a local closure or a `self`-method so the speculative branch can be added beside it without duplicating streaming / timing / stop-id bookkeeping. Keep the exact same behavior.

Specifically, ensure:
- The stop-id / EOS check runs once per emitted token.
- `on_token` streams every non-control token exactly once per emission.
- Timing counters (`t_sample_total`, `t_forward_total`, etc.) still attribute correctly.

- [ ] **Step 2: Build + run existing parity tests**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_smoke -- --nocapture
```

Expected: smoke test still passes with identical output.

- [ ] **Step 3: Commit**

```bash
git add src/inference/generate.rs
git commit -m "refactor(generate): extract single-token decode step (no behavior change)"
```

---

### Task 12: Implement speculative decode branch

**Files:**
- Modify: `src/inference/generate.rs`

- [ ] **Step 1: Add the speculative path**

Inside `Engine::generate`, after sampling `A_0` from `logits_snapshot`, branch on `self.draft_k > 0 && context_tokens.len() >= 3`:

```rust
// Context buffer for n-gram lookup — prompt ids + tokens generated so far.
// Build this incrementally: start as a Vec<u32> from prompt tokens;
// append every emitted token.
//
// Sketch (integrate into your refactored loop):
//
// 1. sample A_0 from current logits_snapshot
// 2. if draft_k <= 1 || context_tokens.len() < 3:
//        emit A_0, forward_one(A_0), continue
// 3. let key = [ctx[last-2], ctx[last-1], A_0]   (see note below on key construction)
//    let mut drafts = [0u32; K-1];
//    let n = ngram_lookup(&context_tokens, &key, K-1, &mut drafts);
//    if n == 0:
//        emit A_0, forward_one(A_0), continue
// 4. build batch = [A_0, drafts[0..n]]   (n+1 inputs)
//    call self.state.forward_batch(&self.model, &batch, &self.graph_pool)
//    this writes n+1 KV positions and advances seq_len by n+1;
//    returns (n+1) × vocab logits.
// 5. let mut out_argmax = vec![0u32; n+1];
//    let j = verify_draft(&logits_batch, vocab, &drafts[..n], n+1, &mut out_argmax);
//    // j in 1..n accepted-drafts; j == n+1 means full accept
// 6. accepted = j - 1 drafts; correction = out_argmax[j-1].
//    Rewind: self.state.kv.seq_len = S + j
//    (add a seq_len setter on KvCache if not exposed — name it `rewind_to(n)`)
//    forward_one_graph(correction) writes correction KV and gives next logits_snapshot.
// 7. Emit A_0, then drafts[0..j-1], then correction, via on_token.
//    Append them all to context_tokens and check stop_ids per token.
//    Break the outer loop if any emitted token is a stop id.
```

**Key construction note:** the 3-token key is the last 3 tokens of `context_tokens ++ A_0`. Edge case: at the start of generation when we've only sampled `A_0` and prompt has ≥2 tokens, key is `[prompt[-2], prompt[-1], A_0]`. The spec does not require special handling — just ensure the key always comes from real tokens already in the context buffer.

- [ ] **Step 2: Add a `KvCache::rewind_to(n: usize)` method if needed**

In `src/inference/cache.rs`, add:

```rust
/// Rewind the cache cursor to `n`. Caller is responsible for ensuring future
/// writes overwrite stale positions. Safe because `attn_len` caps reads at
/// `seq_len`.
pub fn rewind_to(&mut self, n: usize) {
    debug_assert!(n <= self.seq_len);
    self.seq_len = n;
}
```

This must live on `KvCache` itself; if the cache is accessed through `Gemma4State`, expose a matching `Gemma4State::rewind_to(n)` that forwards to `KvCache::rewind_to` — do not leak the cache field.

- [ ] **Step 3: Build**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -10
```

Expected: builds cleanly. No warnings.

- [ ] **Step 4: Quick behavioral check**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo run --release -- --serve --port 8787 &
# In another terminal or via curl:
curl -N -X POST http://127.0.0.1:8787/api/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"Write the numbers 1 to 5.","max_tokens":50,"temperature":0}'
# Kill server when done
```

Expected: output appears. No hang. No crash.

- [ ] **Step 5: Commit**

```bash
git add src/inference/generate.rs src/inference/cache.rs src/inference/forward.rs
git commit -m "feat(generate): prompt-lookup speculative decode branch"
```

---

### Task 13: Greedy parity test (the correctness gate)

**Files:**
- Create: `tests/speculative_parity.rs`

- [ ] **Step 1: Write the test**

```rust
//! Greedy parity: speculative decoding must produce bit-identical token
//! streams to non-speculative decoding when temperature = 0.

use olorin::inference::generate::{Engine, resolve_model};
use std::sync::Mutex;

fn load_engine(draft_k: usize) -> Engine {
    let model_path = resolve_model(Some("gemma4"))
        .expect("gemma-4-e2b-it-Q4_K_M.gguf required under ~/.olorin/models/");
    let mut engine = Engine::load(&model_path, 2048).expect("engine load");
    engine.temperature = 0.0;
    engine.max_tokens = 128;
    engine.draft_k = draft_k;
    engine
}

fn capture_tokens(engine: &mut Engine, prompt: &str) -> String {
    let buf = Mutex::new(String::new());
    let on_token = |t: &str| { buf.lock().unwrap().push_str(t); };
    engine.generate(prompt, "", &on_token).expect("generate ok");
    buf.into_inner().unwrap()
}

fn parity_for_prompt(prompt: &str) {
    let baseline = capture_tokens(&mut load_engine(0), prompt);
    let spec4 = capture_tokens(&mut load_engine(4), prompt);
    let spec8 = capture_tokens(&mut load_engine(8), prompt);
    assert_eq!(baseline, spec4, "parity failure draft_k=4 on {prompt:?}");
    assert_eq!(baseline, spec8, "parity failure draft_k=8 on {prompt:?}");
}

#[test]
fn parity_code_prompt() {
    parity_for_prompt("Write a Python hello world script.");
}

#[test]
fn parity_prose_prompt() {
    parity_for_prompt("In two sentences, what is Rust?");
}

#[test]
fn parity_json_prompt() {
    parity_for_prompt("Return a JSON object with keys a, b, c set to 1, 2, 3.");
}
```

- [ ] **Step 2: Run it**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --release --test speculative_parity -- --test-threads=1 --nocapture
```

Expected: all three pass. If any fail, the speculative branch is violating the argmax-emit invariant — debug before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/speculative_parity.rs
git commit -m "test(speculative): greedy parity across 3 prompts (code/prose/json)"
```

---

### Task 14: Extend `GEMMA4_TIMING=1` output with speculative stats

**Files:**
- Modify: `src/inference/generate.rs`

- [ ] **Step 1: Add counters**

In the decode loop, track:
- `n_spec_steps: usize` — number of speculative steps attempted (draft_k > 0 and n>0)
- `n_spec_accepted: usize` — total accepted drafts across all steps

- [ ] **Step 2: Emit one line in the timing block**

```rust
if self.draft_k > 0 && n_spec_steps > 0 {
    let total_drafts = n_spec_steps * (self.draft_k - 1);
    let accept_rate = n_spec_accepted as f64 / total_drafts.max(1) as f64;
    eprintln!("[timing] speculative: K={} steps={} accepted={} accept_rate={:.2}",
        self.draft_k, n_spec_steps, n_spec_accepted, accept_rate);
}
```

- [ ] **Step 3: Verify**

```bash
GEMMA4_TIMING=1 PATH="/home/peter/projects/eacompute/target/release:$PATH" \
    ./target/release/olorin --serve --port 8787 --draft-k 4
# Make a request, check stderr for [timing] speculative: line
```

- [ ] **Step 4: Commit**

```bash
git add src/inference/generate.rs
git commit -m "feat(timing): speculative accept-rate stats under GEMMA4_TIMING=1"
```

---

### Task 15: Benchmark harness

**Files:**
- Create: `tests/bench_speculative.rs`

- [ ] **Step 1: Write bench**

```rust
//! Wall-clock speedup comparison across three workloads.

use olorin::inference::generate::{Engine, resolve_model};
use std::sync::Mutex;
use std::time::Instant;

fn run(engine: &mut Engine, prompt: &str) -> (String, u128) {
    let buf = Mutex::new(String::new());
    let cb = |t: &str| { buf.lock().unwrap().push_str(t); };
    let t0 = Instant::now();
    engine.generate(prompt, "", &cb).expect("generate");
    (buf.into_inner().unwrap(), t0.elapsed().as_millis())
}

fn bench_prompt(label: &str, prompt: &str) {
    let model_path = resolve_model(Some("gemma4")).expect("model");

    let mut e0 = Engine::load(&model_path, 2048).unwrap();
    e0.temperature = 0.0; e0.max_tokens = 128; e0.draft_k = 0;
    let (base_out, base_ms) = run(&mut e0, prompt);

    let mut e4 = Engine::load(&model_path, 2048).unwrap();
    e4.temperature = 0.0; e4.max_tokens = 128; e4.draft_k = 4;
    let (spec_out, spec_ms) = run(&mut e4, prompt);

    assert_eq!(base_out, spec_out, "{label}: parity broke");

    let speedup = base_ms as f64 / spec_ms.max(1) as f64;
    eprintln!("[bench] {label}: baseline {base_ms}ms, spec(K=4) {spec_ms}ms, speedup {speedup:.2}x");
}

#[test]
#[ignore] // run explicitly with --ignored
fn bench_all() {
    bench_prompt("code",
        "Write a Python function that reverses a linked list.");
    bench_prompt("chat",
        "What's your favorite color and why?");
    bench_prompt("json",
        "Produce a JSON array of 5 objects, each with id and name fields.");
}
```

- [ ] **Step 2: Run bench**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" \
    cargo test --release --test bench_speculative -- --ignored --nocapture
```

Record the three speedup numbers.

- [ ] **Step 3: Decide on defaults**

Target thresholds (from spec):
- code ≥ 1.8×
- chat ≥ 0.95× (no worse than 5% regression)
- json ≥ 1.5×

If code + json clear their bars AND chat is within the 5% window, Task 16 flips the default. If chat regresses more than 5%, leave `draft_k` default at 0 (opt-in via `--draft-k` flag) and document.

- [ ] **Step 4: Commit**

```bash
git add tests/bench_speculative.rs
git commit -m "bench: speculative speedup harness for code/chat/json"
```

---

### Task 16: Integration test — six-prompt regression

**Files:**
- Create: `tests/speculative_integration.rs`

Guard against recurrence of the ChatML-hang pattern (silent decodes, blank responses).

- [ ] **Step 1: Write test**

```rust
//! Replay the six-prompt sequence that surfaced the <|turn> bug.
//! Every prompt must produce non-empty output with draft_k=4.

use olorin::inference::generate::{Engine, resolve_model};
use std::sync::Mutex;

fn run(engine: &mut Engine, prompt: &str) -> String {
    let buf = Mutex::new(String::new());
    let cb = |t: &str| { buf.lock().unwrap().push_str(t); };
    engine.generate(prompt, "", &cb).unwrap();
    buf.into_inner().unwrap()
}

#[test]
#[ignore] // heavy; run explicitly
fn six_prompt_regression() {
    let model_path = resolve_model(Some("gemma4")).unwrap();
    let mut engine = Engine::load(&model_path, 2048).unwrap();
    engine.temperature = 0.7;
    engine.max_tokens = 200;
    engine.draft_k = 4;

    let prompts = [
        "the capital of France is?",
        "tell me a joke",
        "weather Haparanda",
        "write a Python hello world script",
        "what is the time?",
        "hello",
    ];

    for p in prompts {
        let out = run(&mut engine, p);
        assert!(!out.trim().is_empty(), "empty response on prompt: {p:?}");
    }
}
```

- [ ] **Step 2: Run**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" \
    cargo test --release --test speculative_integration -- --ignored --nocapture
```

Expected: all six produce non-empty output.

- [ ] **Step 3: Commit**

```bash
git add tests/speculative_integration.rs
git commit -m "test: six-prompt integration regression for speculative decode"
```

---

### Task 17: Conditional default flip + docs

**Files:**
- Modify: `src/inference/generate.rs` (line 66) — default `draft_k` value
- Modify: `CLAUDE.md` (optional, one-liner under a "Performance" section)

- [ ] **Step 1: Decision**

If Task 15 confirmed code ≥1.8×, json ≥1.5×, chat ≥0.95×: change `draft_k: 0` default to `draft_k: 4` in `Engine::load`.

Otherwise: skip the flip, document the decision and thresholds observed.

- [ ] **Step 2: If flipping default, verify**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --release --test speculative_parity -- --test-threads=1 --nocapture
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --release --test speculative_integration -- --ignored --nocapture
```

Expected: both green.

- [ ] **Step 3: Commit**

```bash
git add src/inference/generate.rs CLAUDE.md
git commit -m "feat(generate): enable draft_k=4 by default"
# or, if not flipping:
git commit -m "docs: record speculative decode benchmark results (draft_k opt-in)"
```

---

## Self-review checklist (performed during plan authoring)

- **Spec coverage:** All spec sections mapped to tasks — ngram_lookup kernel (T2,T3), verify_draft kernel (T7,T8), FFI (T4,T9), unit tests (T5,T10), engine integration (T11,T12), greedy parity (T13), timing (T14), bench (T15), integration (T16), rollout (T17). ✓
- **Placeholder scan:** No TBDs, TODOs, or "implement later" strings. Kernel bodies deferred to Tasks 2/3/7/8 because honest kernel code can only be written after `ea-lookup` confirms intrinsics — Tasks 1 and 6 are gates. ✓
- **Type consistency:** `ngram_lookup` signature uses `i32` in Ea / `u32` in Rust wrapper consistently. `verify_draft` uses `K` rows, `K-1` drafts, `K` argmaxes throughout — matches spec section `verify_draft.ea`. `KvCache::rewind_to(n)` used consistently in Task 12. ✓
- **Scope check:** One coherent subsystem. Does not leak into unrelated refactors. ✓
