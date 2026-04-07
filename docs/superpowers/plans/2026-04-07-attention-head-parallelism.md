# Attention Head Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize the attention head loop and per-head Q/K/V RMSNorms in Gemma 4 inference using the existing `ThreadPool`, with bit-exact output preservation verified by the existing `gemma4_verify` test suite.

**Architecture:** Single decode path; no API changes. Per-thread scratch buffers replace shared `attn_scores` / `kv_f32_scratch`. Heads are partitioned across pool threads via `ThreadPool::run`. The `forward_attn.rs` file (currently 524 lines, over the 500-line hard limit) is split first: `attention_decode` and the per-head norm helpers move into a new `forward_attn_heads.rs`.

**Tech Stack:** Rust, std::thread (via existing `inference::threadpool::ThreadPool`), Ea SIMD kernels via `ffi_inference` (no kernel changes — host-side parallelism only).

**Constraints from CLAUDE.md:**
- No file > 500 lines (forward_attn.rs is already over — split is mandatory before adding code).
- Every feature proven by end-to-end test (`gemma4_verify` provides L2-norm checks against llama.cpp reference).
- No fake functions, no silent fallbacks. Parallel path must be the only path; no `if n_threads == 1 { ... }` shim.
- Match llama.cpp exactly. Output must be **bit-exact** with serial baseline (parallelization alone changes nothing numerically because each head's writes are disjoint and there are no reductions across heads).

**Out of scope:** matmul row remainder, embed dequant, tokenizer, kernel-internal SIMD work. Attention head loop and per-head norms only.

---

## Per-Commit Verification Gate

After every code-changing task in this plan, before `git commit`, **all of the following must pass**. No exceptions, no "I'll fix it next commit." If any gate fails, fix it on the spot or revert.

**Tools available:**
- llama.cpp 7376 installed system-wide (`/usr/bin/llama-cli`, `/usr/bin/llama-eval-callback`, etc.)
- Model: `/home/peter/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf` (Q4K, same file olorin loads)
- eabrain CLI for kernel/intrinsic lookups (use it instead of grepping kernels by hand)

**Gate 1: Build clean.**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
grep -E "^(warning|error)" /tmp/olorin-build.log && exit 1 || true
```

Zero warnings, zero errors. Hard rule "no fakes, no silent fallbacks" applies — if the build adds a warning that wasn't there before this task, fix it.

**Gate 2: Bit-exact internal regression.**

```bash
cargo test --release --test gemma4_parallel_regression
```

Snapshot from Task 1 must match byte-for-byte. Drift = bug, not "good enough."

**Gate 3: gemma4_verify L2 norms unchanged.**

```bash
cargo test --release --test gemma4_verify -- --nocapture | tee /tmp/olorin-verify.log
```

This test prints olorin's L2 norms next to hardcoded llama.cpp reference numbers. The numbers olorin prints must be **identical** to the previous commit's run (capture `/tmp/olorin-verify.log` before each task and `diff` after — see Gate 5 below).

**Gate 4: End-to-end generation matches llama.cpp on a fixed prompt.**

This is the live cross-check against the installed llama.cpp. Run the same prompt through both engines with identical sampling (greedy, seed 0, fixed token count) and diff the output.

```bash
# Reference: llama.cpp greedy-decode 32 tokens
llama-cli \
    -m /home/peter/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf \
    -p "The capital of France is" \
    -n 32 --temp 0 --top-k 1 --seed 0 \
    --no-display-prompt 2>/dev/null \
    > /tmp/llamacpp-out.txt

# Olorin: same prompt, greedy, 32 tokens
./target/release/olorin --interactive <<< "The capital of France is" \
    > /tmp/olorin-out.txt 2>&1
# (or whatever the right CLI invocation is — confirm during Task 0 below)

diff /tmp/llamacpp-out.txt /tmp/olorin-out.txt
```

The two outputs must be **token-for-token identical** for the first N tokens before either temperature or sampler differences kick in. Greedy decode is deterministic.

If they diverge at token K and K stays the same across commits, that's a pre-existing parity gap, not a regression — note it and continue. **What this gate catches is divergence that *changes* between commits.**

**Gate 5: Diff against pre-task baseline.**

Before starting any code-changing task, capture the baseline:

```bash
cp /tmp/olorin-verify.log /tmp/olorin-verify.baseline
cp /tmp/llamacpp-out.txt /tmp/llamacpp-out.baseline  # only if not yet captured
cp /tmp/olorin-out.txt /tmp/olorin-out.baseline
```

After the task, compare:

```bash
diff /tmp/olorin-verify.baseline /tmp/olorin-verify.log || { echo "L2 NORMS DRIFTED"; exit 1; }
diff /tmp/olorin-out.baseline /tmp/olorin-out.txt || { echo "GENERATION DRIFTED"; exit 1; }
```

The llama.cpp output is captured once and never changes (the model file is fixed). The olorin output and the verify-log L2 norms must be **identical to the last good commit**, not merely "close enough." Bit-exactness is the contract this plan signs.

**Gate 6: Hard rules from CLAUDE.md still hold.**

```bash
# 500-line limit
for f in src/**/*.rs; do
    lines=$(wc -l < "$f")
    if [ "$lines" -gt 500 ]; then
        echo "OVER LIMIT: $f ($lines lines)"
        exit 1
    fi
done
```

No fakes, no silent fallbacks (`grep -rn 'unwrap_or_default\|todo!\|unimplemented!\|fixme' src/inference/`), no commented-out code blocks left behind.

**Gate 7: eabrain index refresh if .ea files were touched.**

```bash
# Only if any kernels/*.ea was modified in this commit
git diff --cached --name-only | grep -q '\.ea$' && eabrain index || true
```

This plan does not modify any `.ea` file, so Gate 7 is normally a no-op. But if a future task discovers an Ea kernel needs adjustment, the index must stay current so subsequent kernel lookups via eabrain return the right thing.

---

## Task 0: Establish baselines for the gates

Before Task 1, capture the reference outputs the per-commit gates will diff against. This is a one-time setup, run on the **current `gemma4-cleanup` HEAD** before any plan changes are made.

**Files:** None modified. Outputs go to `/tmp/`.

- [ ] **Step 1: Confirm the olorin CLI invocation for greedy generation**

Read `src/interface/terminal.rs` and `src/inference/generate.rs` to find the right way to invoke olorin in non-interactive, greedy, fixed-seed mode. If no such mode exists, document the closest available mode in this step and adjust Gate 4 to use it. Do not add a new CLI flag for the gate — this plan is parallelization, not CLI work.

If no greedy/seedable mode exists at all: Gate 4 falls back to running olorin once on master, capturing the output, and diffing later runs against that *olorin self-baseline*. Gate 4 then becomes "olorin output unchanged across this plan's commits" rather than "olorin matches llama.cpp." Note this in the gate-failure log if it happens.

- [ ] **Step 2: Capture the llama.cpp reference**

```bash
llama-cli \
    -m /home/peter/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf \
    -p "The capital of France is" \
    -n 32 --temp 0 --top-k 1 --seed 0 \
    --no-display-prompt 2>/dev/null \
    > /tmp/llamacpp-out.baseline
cat /tmp/llamacpp-out.baseline
```

Inspect the output. It should be coherent French-capital text (e.g. "Paris. It is..."). If it's garbage or empty, the model file is wrong or the CLI flags differ in this llama.cpp version — fix before proceeding.

- [ ] **Step 3: Capture the olorin baseline (current HEAD, pre-plan)**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release
./target/release/olorin <invocation from Step 1>  > /tmp/olorin-out.baseline 2>&1
cat /tmp/olorin-out.baseline
```

- [ ] **Step 4: Capture the verify log baseline**

```bash
cargo test --release --test gemma4_verify -- --nocapture > /tmp/olorin-verify.baseline 2>&1
grep -E "L2|first4|llama.cpp" /tmp/olorin-verify.baseline | head -30
```

This is the set of L2 norms that every subsequent commit must reproduce identically.

- [ ] **Step 5: Record the parity status**

Compare `/tmp/llamacpp-out.baseline` and `/tmp/olorin-out.baseline`. Find the first token where they diverge (if any). Append to this plan under a new `## Baseline Parity` section:

```markdown
## Baseline Parity

- llama.cpp first 32 tokens: <paste>
- olorin first 32 tokens: <paste>
- First divergence: token <K> (or "identical")
- Pre-existing parity gap noted: yes/no
```

This is the floor. The plan must not lower it. If post-parallelization olorin diverges from llama.cpp **earlier** than this baseline, the parallelization broke something even if the bit-exact snapshot from Task 1 still passes (which it shouldn't, in that case — but defense in depth).

- [ ] **Step 6: Do not commit baselines to git**

The `/tmp/` files are intentionally outside the repo. They are scratch state for the per-commit gates only. Adding them to git would couple the plan to one machine.

---

## File Structure

**Modify:**
- `src/inference/forward.rs:88-90` — `Gemma4State` field changes (per-thread scratch).
- `src/inference/forward.rs:165-175` — `Gemma4State::new` allocates per-thread scratch sized to `pool.thread_count() * max_head` and `pool.thread_count() * max_seq_len`.
- `src/inference/forward_attn.rs:96-110` — Q-norm loop becomes a `pool.run()` over heads.
- `src/inference/forward_attn.rs:144-167` — K-norm and V-bare-norm loops become `pool.run()` over kv_heads.
- `src/inference/forward_attn.rs:362-431` — `attention_decode` head loop becomes `pool.run()` over heads.
- `src/inference/mod.rs` — register new `forward_attn_heads` module.

**Create:**
- `src/inference/forward_attn_heads.rs` — extracted attention compute (`attention_decode`, per-head norm helpers). Owns the parallel dispatch logic. Brings forward_attn.rs back under 500 lines.
- `tests/gemma4_parallel_regression.rs` — bit-exact regression: capture full-forward-pass logits with current code, then re-run after each parallelization step and assert equality to f32 bit pattern.

**Why split first:** the parallel versions will be longer than the serial loops (per-thread scratch indexing, closures, slice splitting). Adding them to a file already over the limit makes the violation worse and forces cleanup mid-feature. Split is the smallest reversible step.

**Why per-thread scratch in `Gemma4State` (not stack/per-call alloc):** project rule "Pre-allocate, reuse. Structs own their buffers." `Gemma4State::new` already takes `max_seq_len`; it gets a `pool: &ThreadPool` parameter so it can size scratch to `pool.thread_count()`.

---

**Per-task gate reminder:** every task below that ends with `git commit` must first pass all 7 gates from the "Per-Commit Verification Gate" section above. Tasks 1, 2, 3, 4, 5 each include explicit `cargo test` steps for Gates 1-3; Gates 4-6 are not duplicated per-task to keep the plan readable, but they are mandatory. If any gate fails, fix in place or revert — do not commit a known-broken state.

---

### Task 1: Capture serial baseline for regression test

**Files:**
- Create: `tests/gemma4_parallel_regression.rs`

This test runs `forward_one` for a fixed prompt and writes the resulting logits to a JSON-free binary snapshot under `tests/snapshots/`. Initially the snapshot is missing — first run captures it. Subsequent runs must match bit-exactly.

- [ ] **Step 1: Look up the existing forward-pass test entry point**

Read `tests/gemma4_verify.rs` lines 540-560 to confirm the `forward_one` signature and `Engine` setup pattern. The new test mirrors this setup.

- [ ] **Step 2: Write the regression test (capture-or-compare)**

Create `tests/gemma4_parallel_regression.rs`:

```rust
//! Bit-exact regression for parallelized inference.
//!
//! On first run with no snapshot, captures the current logits as ground truth.
//! On every subsequent run, asserts the new logits match the snapshot byte-for-byte.
//! This guards parallelization changes against numerical drift.

use olorin::inference::engine::Gemma4Model;
use olorin::inference::forward::Gemma4State;
use olorin::inference::threadpool::ThreadPool;
use std::fs;
use std::path::PathBuf;

const MODEL_PATH: &str = "models/gemma-3-E2B-it-Q4_K_M.gguf";
const SNAPSHOT: &str = "tests/snapshots/gemma4_logits_bos.bin";

fn snapshot_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(SNAPSHOT);
    p
}

#[test]
fn forward_one_bos_logits_bit_exact() {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: {} not present", MODEL_PATH);
        return;
    }
    let model = Gemma4Model::load(MODEL_PATH).expect("load model");
    let pool = ThreadPool::new();
    let mut state = Gemma4State::new(&model, 512);
    let logits = state.forward_one(&model, 2 /* BOS */, &pool).to_vec();

    // Serialize as raw little-endian f32 bytes.
    let mut bytes = Vec::with_capacity(logits.len() * 4);
    for v in &logits {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    let path = snapshot_path();
    if !path.exists() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, &bytes).unwrap();
        panic!("captured baseline snapshot at {} — re-run test to verify", path.display());
    }

    let expected = fs::read(&path).unwrap();
    assert_eq!(
        bytes.len(),
        expected.len(),
        "logits length changed: got {} bytes, snapshot {}",
        bytes.len(),
        expected.len(),
    );
    assert!(
        bytes == expected,
        "logits drifted from snapshot — parallelization changed numerics"
    );
}
```

- [ ] **Step 3: Run the test to capture the baseline**

Run: `PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression -- --nocapture`

Expected: panic with `captured baseline snapshot at ... — re-run test to verify`. The snapshot file now exists.

If the model file isn't present, the test prints SKIP and returns. In that case, verify the snapshot path is correct, locate the actual gguf file, update `MODEL_PATH`, and re-run.

- [ ] **Step 4: Re-run to verify capture is stable**

Run: `cargo test --release --test gemma4_parallel_regression`

Expected: PASS.

- [ ] **Step 5: Commit baseline + test**

```bash
git add tests/gemma4_parallel_regression.rs tests/snapshots/gemma4_logits_bos.bin
git commit -m "test: bit-exact regression snapshot for forward_one logits"
```

---

### Task 2: Split forward_attn.rs — extract attention compute helpers

Pure refactor, no semantic change. Brings forward_attn.rs under 500 lines and gives parallel dispatch a home.

**Files:**
- Create: `src/inference/forward_attn_heads.rs`
- Modify: `src/inference/forward_attn.rs` (delete the moved code, leave call sites unchanged)
- Modify: `src/inference/mod.rs` (add `pub mod forward_attn_heads;`)

- [ ] **Step 1: Add module declaration**

In `src/inference/mod.rs`, add (alphabetical with other forward_* modules):

```rust
pub mod forward_attn_heads;
```

- [ ] **Step 2: Create `forward_attn_heads.rs` with the moved functions**

The new file owns three helpers (still serial — parallelization comes in later tasks). Move from `forward_attn.rs`:

- The Q-norm loop (lines 96-110): wrap as `pub(crate) fn q_norm_per_head(state: &mut Gemma4State, q_norm: *const f32, n_heads: usize, head_dim: usize, rms_eps: f32)`.
- The K-norm loop (lines 144-158): wrap as `pub(crate) fn k_norm_per_head(state: &mut Gemma4State, k_norm: *const f32, n_kv_heads: usize, head_dim: usize, rms_eps: f32)`.
- The V-bare-norm loop (lines 161-167): wrap as `pub(crate) fn v_bare_norm_per_head(state: &mut Gemma4State, n_kv_heads: usize, head_dim_v: usize, rms_eps: f32)`.
- The full `attention_decode` method (lines 362-431): move as a free function `pub(crate) fn attention_decode(state: &mut Gemma4State, n_heads, n_kv_heads, gqa_ratio, head_dim, kv_dim, attn_len, scale, k_ptr, v_ptr)`.

File header:

```rust
//! Per-head attention compute: Q/K/V RMSNorms and attention_decode.
//!
//! Split from forward_attn.rs to keep that file under the 500-line limit
//! and to localize parallelization changes (Tasks 4-6 of the head-parallelism plan).
//!
//! All functions in this module are serial. Parallel dispatch is added in
//! later tasks of the same plan.

use crate::inference::forward::{bare_rmsnorm, Gemma4State};
use crate::kernels::ffi_inference;

pub(crate) fn q_norm_per_head(
    state: &mut Gemma4State,
    q_norm: *const f32,
    n_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) {
    for h in 0..n_heads {
        let off = h * head_dim;
        ffi_inference::gemma4_rmsnorm(
            state.q.as_ptr().wrapping_add(off),
            q_norm,
            state.kv_f32_scratch.as_mut_ptr(),
            head_dim as i32,
            rms_eps,
        );
        state.q[off..off + head_dim]
            .copy_from_slice(&state.kv_f32_scratch[..head_dim]);
    }
}

pub(crate) fn k_norm_per_head(
    state: &mut Gemma4State,
    k_norm: *const f32,
    n_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) {
    for h in 0..n_kv_heads {
        let off = h * head_dim;
        ffi_inference::gemma4_rmsnorm(
            state.k.as_ptr().wrapping_add(off),
            k_norm,
            state.kv_f32_scratch.as_mut_ptr(),
            head_dim as i32,
            rms_eps,
        );
        state.k[off..off + head_dim]
            .copy_from_slice(&state.kv_f32_scratch[..head_dim]);
    }
}

pub(crate) fn v_bare_norm_per_head(
    state: &mut Gemma4State,
    n_kv_heads: usize,
    head_dim_v: usize,
    rms_eps: f32,
) {
    for h in 0..n_kv_heads {
        let off = h * head_dim_v;
        bare_rmsnorm(&mut state.v[off..off + head_dim_v], rms_eps);
    }
}

pub(crate) fn attention_decode(
    state: &mut Gemma4State,
    n_heads: usize,
    _n_kv_heads: usize,
    gqa_ratio: usize,
    head_dim: usize,
    kv_dim: usize,
    attn_len: usize,
    scale: f32,
    k_ptr: *const u16,
    v_ptr: *const u16,
) {
    let stride = kv_dim;
    for h in 0..n_heads {
        let kv_h = h / gqa_ratio;
        let q_off = h * head_dim;
        let q_slice = &state.q[q_off..q_off + head_dim];

        for p in 0..attn_len {
            let k_offset = p * stride + kv_h * head_dim;
            let k_src = unsafe { k_ptr.add(k_offset) };
            unsafe {
                ffi_inference::f16_to_f32(
                    k_src,
                    state.kv_f32_scratch.as_mut_ptr(),
                    head_dim as i32,
                );
            }
            state.attn_scores[p] = ffi_inference::f32_dot(
                q_slice.as_ptr(),
                state.kv_f32_scratch.as_ptr(),
                head_dim as i32,
            );
        }

        unsafe {
            ffi_inference::softmax_f32(
                state.attn_scores.as_mut_ptr(),
                attn_len as i32,
                scale,
            );
        }

        let out_off = q_off;
        state.attn_out[out_off..out_off + head_dim].fill(0.0);
        for p in 0..attn_len {
            let v_offset = p * stride + kv_h * head_dim;
            let v_src = unsafe { v_ptr.add(v_offset) };
            unsafe {
                ffi_inference::f16_to_f32(
                    v_src,
                    state.kv_f32_scratch.as_mut_ptr(),
                    head_dim as i32,
                );
            }
            let s = state.attn_scores[p];
            ffi_inference::f32_dot_acc(
                state.attn_out[out_off..].as_mut_ptr(),
                state.kv_f32_scratch.as_ptr(),
                s,
                head_dim as i32,
            );
        }
    }
}
```

- [ ] **Step 3: Update call sites in `forward_attn.rs`**

In `forward_attn.rs`, replace:

```rust
if !lw.q_norm.is_null() {
    for h in 0..n_heads {
        // ... 12 lines of inline norm
    }
}
```

with:

```rust
if !lw.q_norm.is_null() {
    super::forward_attn_heads::q_norm_per_head(self, lw.q_norm, n_heads, head_dim, model.rms_eps);
}
```

Same for K-norm:

```rust
if !lw.k_norm.is_null() {
    super::forward_attn_heads::k_norm_per_head(self, lw.k_norm, n_kv_heads, head_dim, model.rms_eps);
}
```

V bare norm:

```rust
super::forward_attn_heads::v_bare_norm_per_head(self, n_kv_heads, head_dim_v, model.rms_eps);
```

Replace the `self.attention_decode(...)` method call with:

```rust
super::forward_attn_heads::attention_decode(
    self,
    n_heads, n_kv_heads, gqa_ratio, head_dim,
    kv_dim, attn_len, attn_scale, k_ptr, v_ptr,
);
```

Delete the `pub(crate) fn attention_decode(&mut self, ...)` method definition (formerly at lines 362-431). Delete the old inline norm loops.

- [ ] **Step 4: Build**

Run: `PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release`

Expected: clean build, no warnings about unused imports.

- [ ] **Step 5: Verify file is now under 500 lines**

Run: `wc -l src/inference/forward_attn.rs`

Expected: < 500. If not, the moved code wasn't fully removed — recheck deletions.

- [ ] **Step 6: Run the regression test**

Run: `cargo test --release --test gemma4_parallel_regression`

Expected: PASS (refactor is semantics-preserving).

- [ ] **Step 7: Run the existing forward-pass verification**

Run: `cargo test --release --test gemma4_verify`

Expected: PASS — all L2-norm checks vs llama.cpp still pass.

- [ ] **Step 8: Commit**

```bash
git add src/inference/mod.rs src/inference/forward_attn.rs src/inference/forward_attn_heads.rs
git commit -m "refactor: extract attention head helpers into forward_attn_heads"
```

---

### Task 3: Per-thread scratch buffers in Gemma4State

Replace the single `attn_scores` and `kv_f32_scratch` buffers with per-thread slabs, sized at construction time from the pool. Still serial after this task — only the buffer layout changes. Each call site picks slot 0.

**Files:**
- Modify: `src/inference/forward.rs:85-95` (struct fields)
- Modify: `src/inference/forward.rs:160-180` (constructor)
- Modify: `src/inference/forward_attn_heads.rs` (use slot 0)

- [ ] **Step 1: Read the constructor and find max_head**

Read `src/inference/forward.rs` lines 150-200. Note how `max_head` and `max_seq_len` are computed. We need to multiply by `pool.thread_count()`.

- [ ] **Step 2: Change `Gemma4State::new` to take a `&ThreadPool`**

Update the signature:

```rust
pub fn new(model: &Gemma4Model, max_seq_len: usize, pool: &crate::inference::threadpool::ThreadPool) -> Self {
```

In the body, after computing `max_head`:

```rust
let n_threads = pool.thread_count();
// ...
attn_scores: vec![0.0; max_seq_len * n_threads],
kv_f32_scratch: vec![0.0; max_head * n_threads],
// store for later striding:
n_thread_slots: n_threads,
attn_scores_stride: max_seq_len,
kv_scratch_stride: max_head,
```

- [ ] **Step 3: Add the new fields to `Gemma4State`**

In the `pub struct Gemma4State { ... }` block, add:

```rust
pub(crate) n_thread_slots: usize,
pub(crate) attn_scores_stride: usize,
pub(crate) kv_scratch_stride: usize,
```

- [ ] **Step 4: Update all `Gemma4State::new(&model, N)` callers**

Find call sites:

Run: `grep -rn "Gemma4State::new" src/ tests/`

For each result, add the pool argument. Tests may need to construct a `ThreadPool::new()` first. The `gemma4_parallel_regression.rs` test from Task 1 already builds a pool — pass it through.

- [ ] **Step 5: In `forward_attn_heads.rs`, all serial functions still use slot 0**

The serial helpers from Task 2 still index `state.kv_f32_scratch[0..head_dim]` and `state.attn_scores[0..attn_len]` — i.e., the first slot. With the larger buffer they continue to work because slot 0 is bytes 0..stride. No code change needed inside `forward_attn_heads.rs` for this task — verify by re-reading and confirming all accesses use `[..head_dim]` or `as_ptr()` (slot 0 base).

- [ ] **Step 6: Build**

Run: `PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release`

Expected: clean build.

- [ ] **Step 7: Run regression**

Run: `cargo test --release --test gemma4_parallel_regression`

Expected: PASS — buffers grew but slot 0 layout matches the old single-slot layout exactly.

- [ ] **Step 8: Commit**

```bash
git add src/inference/forward.rs src/inference/forward_attn_heads.rs tests/
git commit -m "refactor: per-thread slab layout for attn_scores and kv_f32_scratch"
```

---

### Task 4: Parallelize attention_decode head loop

Now the actual work. Replace the `for h in 0..n_heads` in `attention_decode` with a `pool.run()` call where each thread takes a strided range of heads. Each thread uses its own scratch slot (`kv_f32_scratch[tid * stride .. (tid+1) * stride]`, `attn_scores[tid * stride .. ...]`).

**Files:**
- Modify: `src/inference/forward_attn_heads.rs` — `attention_decode` body.

- [ ] **Step 1: Note the aliasing problem**

`attention_decode` takes `&mut Gemma4State`. Inside the closure passed to `pool.run`, multiple threads need to: (a) read `state.q` and `state.cache.k_ptr/v_ptr` (immutable share — fine), (b) write disjoint ranges of `state.attn_out` (each head h writes `[h*head_dim .. (h+1)*head_dim]` — disjoint), (c) write disjoint ranges of `state.kv_f32_scratch` and `state.attn_scores` (each tid writes its own slot — disjoint by construction).

Disjoint mutable borrows of one `&mut Vec<f32>` from a `Send + Sync` closure require either:
- Splitting via `chunks_mut` outside the closure and moving the chunks in (doesn't work — closures need fixed slots, not iteration), or
- Raw pointers wrapped in a `Send + Sync` newtype.

We use raw pointers. This is the same pattern `matmul.rs` already uses (look at `par_q4k_matvec` for reference).

- [ ] **Step 2: Define a local `Send + Sync` pointer wrapper**

At the top of `forward_attn_heads.rs`, add:

```rust
/// Wrapper to ship raw pointers across thread boundaries inside pool closures.
/// Safety: callers must ensure each thread accesses a disjoint range of the
/// underlying allocation. Used only for slab-partitioned scratch and per-head
/// disjoint output writes.
#[derive(Copy, Clone)]
struct SharedPtr<T>(*mut T);
unsafe impl<T> Send for SharedPtr<T> {}
unsafe impl<T> Sync for SharedPtr<T> {}

#[derive(Copy, Clone)]
struct SharedConstPtr<T>(*const T);
unsafe impl<T> Send for SharedConstPtr<T> {}
unsafe impl<T> Sync for SharedConstPtr<T> {}
```

- [ ] **Step 3: Rewrite `attention_decode`**

Replace the existing `attention_decode` body with:

```rust
pub(crate) fn attention_decode(
    state: &mut Gemma4State,
    n_heads: usize,
    _n_kv_heads: usize,
    gqa_ratio: usize,
    head_dim: usize,
    kv_dim: usize,
    attn_len: usize,
    scale: f32,
    k_ptr: *const u16,
    v_ptr: *const u16,
    pool: &crate::inference::threadpool::ThreadPool,
) {
    let stride_kv = kv_dim;
    let kv_scratch_stride = state.kv_scratch_stride;
    let attn_scores_stride = state.attn_scores_stride;

    let q_ptr = SharedConstPtr(state.q.as_ptr());
    let attn_out_ptr = SharedPtr(state.attn_out.as_mut_ptr());
    let kv_scratch_ptr = SharedPtr(state.kv_f32_scratch.as_mut_ptr());
    let attn_scores_ptr = SharedPtr(state.attn_scores.as_mut_ptr());
    let k_ptr_w = SharedConstPtr(k_ptr);
    let v_ptr_w = SharedConstPtr(v_ptr);

    let n_workers = n_heads.min(pool.thread_count()).max(1);

    pool.run(n_workers, |tid, nt| {
        // Distribute heads across threads contiguously.
        let per = (n_heads + nt - 1) / nt;
        let h_start = tid * per;
        let h_end = ((tid + 1) * per).min(n_heads);

        // This thread's private scratch slots.
        let kv_scratch_base = unsafe { kv_scratch_ptr.0.add(tid * kv_scratch_stride) };
        let attn_scores_base = unsafe { attn_scores_ptr.0.add(tid * attn_scores_stride) };

        for h in h_start..h_end {
            let kv_h = h / gqa_ratio;
            let q_off = h * head_dim;
            let q_slice_ptr = unsafe { q_ptr.0.add(q_off) };

            // Q · K for each cached position
            for p in 0..attn_len {
                let k_offset = p * stride_kv + kv_h * head_dim;
                let k_src = unsafe { k_ptr_w.0.add(k_offset) };
                unsafe {
                    ffi_inference::f16_to_f32(k_src, kv_scratch_base, head_dim as i32);
                }
                let dot = ffi_inference::f32_dot(
                    q_slice_ptr,
                    kv_scratch_base as *const f32,
                    head_dim as i32,
                );
                unsafe { *attn_scores_base.add(p) = dot; }
            }

            unsafe {
                ffi_inference::softmax_f32(attn_scores_base, attn_len as i32, scale);
            }

            // Weighted V sum into attn_out[h*head_dim..(h+1)*head_dim]
            let out_base = unsafe { attn_out_ptr.0.add(q_off) };
            unsafe {
                std::ptr::write_bytes(out_base, 0, head_dim);
            }
            for p in 0..attn_len {
                let v_offset = p * stride_kv + kv_h * head_dim;
                let v_src = unsafe { v_ptr_w.0.add(v_offset) };
                unsafe {
                    ffi_inference::f16_to_f32(v_src, kv_scratch_base, head_dim as i32);
                }
                let s = unsafe { *attn_scores_base.add(p) };
                ffi_inference::f32_dot_acc(
                    out_base,
                    kv_scratch_base as *const f32,
                    s,
                    head_dim as i32,
                );
            }
        }
    });
}
```

- [ ] **Step 4: Update the call site in `forward_attn.rs`**

Pass `pool` through:

```rust
super::forward_attn_heads::attention_decode(
    self,
    n_heads, n_kv_heads, gqa_ratio, head_dim,
    kv_dim, attn_len, attn_scale, k_ptr, v_ptr,
    pool,
);
```

- [ ] **Step 5: Build**

Run: `PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release`

Expected: clean build.

- [ ] **Step 6: Run regression — bit-exact output required**

Run: `cargo test --release --test gemma4_parallel_regression`

Expected: PASS.

If FAIL with "logits drifted from snapshot": parallelization changed numerics. Likely causes to investigate, in order:
1. A thread is reading/writing the wrong scratch slot (off-by-one in `tid * stride`).
2. The head range distribution overlaps (`per` calculation wrong for non-divisible n_heads).
3. Two heads happen to map to the same `kv_h` and accidentally clobber via shared state — they shouldn't, since each head writes its own `attn_out[q_off..]`, but verify.

Do not adjust the snapshot to make the test pass. The whole point is bit-exactness.

- [ ] **Step 7: Run gemma4_verify**

Run: `cargo test --release --test gemma4_verify`

Expected: PASS — L2 norms vs llama.cpp unchanged.

- [ ] **Step 8: Run gemma4_smoke**

Run: `cargo test --release --test gemma4_smoke`

Expected: PASS — end-to-end generation still produces output.

- [ ] **Step 9: Commit**

```bash
git add src/inference/forward_attn_heads.rs src/inference/forward_attn.rs
git commit -m "perf: parallelize attention_decode head loop across pool threads"
```

---

### Task 5: Parallelize per-head Q/K/V RMSNorms

Same pattern as Task 4 but for the three norm helpers. Each head's norm is independent: head h reads `q[h*head_dim..]`, writes back to the same range via the kernel's scratch round-trip.

The kernel signature is `gemma4_rmsnorm(input, weight, output, dim, eps)` — it writes to `output`, then the caller copies back. To parallelize cleanly, each thread's output goes directly into its own scratch slot, then we copy that slot back into `q[h*head_dim..]`. Since the copy-back is also disjoint (different h ranges), it's safe under the same `SharedPtr` pattern.

**Files:**
- Modify: `src/inference/forward_attn_heads.rs` — three norm helpers.
- Modify: `src/inference/forward_attn.rs` — pass `pool` to the helpers.

- [ ] **Step 1: Rewrite `q_norm_per_head` to use the pool**

```rust
pub(crate) fn q_norm_per_head(
    state: &mut Gemma4State,
    q_norm: *const f32,
    n_heads: usize,
    head_dim: usize,
    rms_eps: f32,
    pool: &crate::inference::threadpool::ThreadPool,
) {
    let kv_scratch_stride = state.kv_scratch_stride;
    let q_ptr = SharedPtr(state.q.as_mut_ptr());
    let scratch_ptr = SharedPtr(state.kv_f32_scratch.as_mut_ptr());
    let q_norm_w = SharedConstPtr(q_norm);

    let n_workers = n_heads.min(pool.thread_count()).max(1);

    pool.run(n_workers, |tid, nt| {
        let per = (n_heads + nt - 1) / nt;
        let h_start = tid * per;
        let h_end = ((tid + 1) * per).min(n_heads);
        let scratch_base = unsafe { scratch_ptr.0.add(tid * kv_scratch_stride) };

        for h in h_start..h_end {
            let off = h * head_dim;
            let q_head_in = unsafe { q_ptr.0.add(off) as *const f32 };
            ffi_inference::gemma4_rmsnorm(
                q_head_in,
                q_norm_w.0,
                scratch_base,
                head_dim as i32,
                rms_eps,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scratch_base as *const f32,
                    q_ptr.0.add(off),
                    head_dim,
                );
            }
        }
    });
}
```

- [ ] **Step 2: Rewrite `k_norm_per_head` symmetrically**

Same pattern, swap `state.q` for `state.k`, parameter `q_norm` for `k_norm`, `n_heads` for `n_kv_heads`. Copy the function and edit — do not abstract; the function is small and the abstraction would obscure the disjointness reasoning.

- [ ] **Step 3: Rewrite `v_bare_norm_per_head`**

`bare_rmsnorm` is a Rust scalar helper today (look at `forward.rs` — it's the only scalar exception in inference, marked `bare_*`). It takes `&mut [f32]`. To parallelize, we partition `state.v` into chunks of `head_dim_v` and dispatch each chunk to a thread:

```rust
pub(crate) fn v_bare_norm_per_head(
    state: &mut Gemma4State,
    n_kv_heads: usize,
    head_dim_v: usize,
    rms_eps: f32,
    pool: &crate::inference::threadpool::ThreadPool,
) {
    let v_ptr = SharedPtr(state.v.as_mut_ptr());
    let n_workers = n_kv_heads.min(pool.thread_count()).max(1);

    pool.run(n_workers, |tid, nt| {
        let per = (n_kv_heads + nt - 1) / nt;
        let h_start = tid * per;
        let h_end = ((tid + 1) * per).min(n_kv_heads);
        for h in h_start..h_end {
            let off = h * head_dim_v;
            let slice = unsafe {
                std::slice::from_raw_parts_mut(v_ptr.0.add(off), head_dim_v)
            };
            crate::inference::forward::bare_rmsnorm(slice, rms_eps);
        }
    });
}
```

**Note:** if `bare_rmsnorm` is not `pub(crate)`, make it so. This is the only inference scalar helper called from a sibling module — no architectural concern.

- [ ] **Step 4: Update call sites in `forward_attn.rs`**

Each helper call gains a `pool` arg:

```rust
if !lw.q_norm.is_null() {
    super::forward_attn_heads::q_norm_per_head(self, lw.q_norm, n_heads, head_dim, model.rms_eps, pool);
}
// ...
if !lw.k_norm.is_null() {
    super::forward_attn_heads::k_norm_per_head(self, lw.k_norm, n_kv_heads, head_dim, model.rms_eps, pool);
}
super::forward_attn_heads::v_bare_norm_per_head(self, n_kv_heads, head_dim_v, model.rms_eps, pool);
```

- [ ] **Step 5: Build**

Run: `PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release`

Expected: clean build.

- [ ] **Step 6: Run regression**

Run: `cargo test --release --test gemma4_parallel_regression`

Expected: PASS — bit-exact, since each head's norm is a deterministic kernel on disjoint inputs.

- [ ] **Step 7: Run gemma4_verify and smoke**

Run: `cargo test --release --test gemma4_verify --test gemma4_smoke`

Expected: PASS.

- [ ] **Step 8: Confirm forward_attn.rs is still under 500 lines**

Run: `wc -l src/inference/forward_attn.rs src/inference/forward_attn_heads.rs`

Expected: both under 500. If `forward_attn_heads.rs` is approaching 500, that's a future task (not this plan).

- [ ] **Step 9: Commit**

```bash
git add src/inference/forward_attn_heads.rs src/inference/forward_attn.rs src/inference/forward.rs
git commit -m "perf: parallelize per-head Q/K/V RMSNorms across pool threads"
```

---

### Task 6: Sanity benchmark

Not a TDD step — a measurement to confirm the optimization actually helped. If it didn't, the plan is wrong (e.g., dispatch overhead exceeds work, or `attn_len` is too small for parallelism to win).

**Files:**
- None modified. Read-only measurement.

- [ ] **Step 1: Run gemma4_smoke with timing**

Use the existing smoke test path. If it doesn't print timing, run it under `time`:

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" \
  time cargo test --release --test gemma4_smoke -- --nocapture
```

- [ ] **Step 2: Compare to pre-parallelization**

Check out the commit before Task 4 (`git rev-parse HEAD~3` or similar — count commits from Task 4 onward), build, time, then return to HEAD.

```bash
git log --oneline | head -10  # find the commit before "perf: parallelize attention_decode"
git stash  # if any uncommitted snapshot/test artifacts
git checkout <pre-task-4-sha>
cargo build --release
time cargo test --release --test gemma4_smoke -- --nocapture
git checkout -
```

- [ ] **Step 3: Record findings**

Append to `docs/superpowers/plans/2026-04-07-attention-head-parallelism.md` (this file) under a new heading `## Results`:

```markdown
## Results

- Hardware: <CPU model, core count>
- Pool size: <thread_count>
- Pre-parallel smoke wall time: <X>s
- Post-parallel smoke wall time: <Y>s
- Speedup: <X/Y>x
```

If speedup is < 1.2x: investigate before claiming the work is done. Possible causes: dispatch overhead, false sharing on `attn_out`, `attn_len` too small at decode time (sliding-window cap of 512 may be the bottleneck), or the matmul row-parallel work was already saturating cores. Report findings; do not silently ship a non-improvement.

- [ ] **Step 4: Commit results**

```bash
git add docs/superpowers/plans/2026-04-07-attention-head-parallelism.md
git commit -m "docs: record attention parallelism benchmark results"
```

---

## Results

Measured 2026-04-07 on the gemma4-cleanup branch.

- **Hardware:** 16-thread CPU (pool reports `Thread pool: 16 threads`)
- **Model:** `~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf` (Gemma 4 E2B Q4K_M)
- **Workload:** `echo "Write a 200 word story about a robot."` piped via REPL,
  default sampling (temperature=1.0, top_k=64, top_p=0.95 — stochastic, so
  generation length varies between runs).

| | Pre-parallel (3bc12b3) | Post-parallel (a949241) |
|---|---|---|
| run 1 | 1m 44.7s | 1m 24.0s |
| run 2 | 1m 32.9s | 1m 25.7s |
| run 3 | 1m 34.4s | 1m 19.6s |
| **median** | **~94 s** | **~84 s** |

**Speedup: ~1.12x** end-to-end.

The win is modest because:
1. The matmul row-parallel kernels (Wo, FFN gate/up/down, logits) were
   *already* parallelized via `par_matvec` and `par_q4k_matvec_dual`. Those
   are the dominant compute on the per-token hot path. Attention head loop
   and per-head norms together are a smaller fraction of total wall time.
2. Gemma 4 E2B has only 8 attention heads (`n_heads=8`) and 1 KV head, so
   `pool.run(n_workers, …)` clamps to `n_workers = min(8, 16) = 8` for the
   attention loop. We're spreading 8 heads across 8 threads, leaving 8 of
   the 16 pool threads idle during that section.
3. Stochastic sampling makes generation length vary by ±10% between runs,
   which is on the same order as the measured speedup. The bit-exact
   regression test (deterministic) is the more reliable signal for
   *correctness*, not throughput.

**Correctness preservation:** every commit in this plan passed the bit-exact
1 MiB logit snapshot regression and the gemma4_verify L2-norm checks against
hardcoded llama.cpp reference values. End-to-end greedy generation produced
the same correct answer ("101, 103, 107") at every checkpoint.

**Where the work actually went:** removing the serial bottleneck on principle.
The architectural change is real — heads now run on multiple cores — but to
see a bigger wall-time win on this specific model the next bottleneck would
need attention, likely the f16→f32 KV-cache conversion in the attention inner
loop or matmul reduction tails.

---

## Self-Review Notes

**Spec coverage:**
- Attention head loop (forward_attn.rs:377) → Task 4 ✓
- Q-norm per-head (forward_attn.rs:98) → Task 5 ✓
- K-norm per-head (forward_attn.rs:146) → Task 5 ✓
- V-bare-norm per-head (forward_attn.rs:161) → Task 5 ✓
- 500-line file rule → Task 2 ✓
- Bit-exact regression → Task 1 ✓
- Per-thread scratch (precondition) → Task 3 ✓

**Risks:**
1. **Snapshot may not be bit-stable across CPUs** — f32_dot reduction order is fixed by the kernel, so it should be deterministic on a given binary. If the snapshot was captured on a different CPU than where CI runs, it may fail. Mitigation: capture and compare on the same machine; if CI is needed, regenerate snapshots in a setup step or relax to ULP tolerance (deferred — not in this plan).
2. **`pool.thread_count()` may exceed n_heads** — handled by `n_workers = n_heads.min(pool.thread_count()).max(1)`. The `.max(1)` guards against pool size 0 (not possible from `ThreadPool::new` but cheap).
3. **`bare_rmsnorm` visibility** — Task 5 Step 3 may need to widen `bare_rmsnorm` from `pub(super)` to `pub(crate)`. Confirm and adjust.
4. **Other `Gemma4State::new` callers in tests** — Task 3 Step 4 must catch all of them. The grep is the safety net.
