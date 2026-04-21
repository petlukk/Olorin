# Path B — Cauchy-Schwarz Output-Head Skip — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip computing cold-vocab logit rows when a Cauchy-Schwarz bound proves the hot top-1 is the global argmax. Correctness-preserving by construction. Targets 24.9% of Pi decode (output head).

**Architecture:** Two phases with a hard gate between them. Phase 1 is a *diagnostic* that runs the full matmul as-today but ALSO records whether the Cauchy-Schwarz bound would have succeeded. Produces a safe-skip rate. Phase 2 is the actual implementation — only starts if Phase 1 shows ≥20% safe-skip rate. Math guarantees correctness regardless of rate; the rate only decides whether Phase 2 is worth building.

**Tech Stack:** Rust (Olorin), existing Ea kernels (`q6k_repacked_batch_ws_pre_d`, `matvec_ws`), gemma-4-e2b-it-Q4_K_M.gguf. No new kernels.

**Branch:** create `path-b-hot-vocab` from `origin/gemma4-specialize` (commit `ba8c045`). Do NOT branch from plain `supertools` — we want to carry forward the RoPE pre-bake win.

**CRITICAL context read-first:**
- `project_path_b_hot_vocab.md` (memory) — mechanism, why-now, Path-A-vs-Path-B
- `project_decode_profile_2026-04-21.md` (memory) — proves output head is 24.9% of Pi decode
- `project_hot_vocab_path_a_null.md` (memory) — what failed before, what not to redo
- `CLAUDE.md` at repo root — hard rules (500-line files, no fake functions, etc.)

## The math (summary — see memory for full derivation)

For output head row i: `logit_i = embed_weight_row_i · hidden`. Vocab size 262144, hidden dim 1536.

Define:
- `row_norm[i] = ‖embed_weight_row_i‖₂` for i in 0..262144 (precomputed once at load)
- "hot set" H = top-N indices by `row_norm` (NOT by frequency — that was Path A's mistake)
- `max_cold_norm = max(row_norm[i] for i not in H)`

Bound: `|logit_cold_i| ≤ ‖hidden‖ × row_norm[i] ≤ ‖hidden‖ × max_cold_norm`

If `hot_top_1 > ‖hidden‖ × max_cold_norm`, then `hot_top_1 > max logit_cold_i` → `hot_top_1` is global argmax. Safe to skip cold matmul.

Else: fall back to full matmul (bit-exact to today).

Softcap is monotonic; argmax of softcapped logits = argmax of raw. Apply the bound pre-softcap.

## File Structure

**Create (Phase 1):**
- `src/inference/path_b.rs` (≤200 lines) — `RowNormCache` struct storing `row_norm[i]` array + `max_cold_norm` + `hot_indices` (contiguous top-N by norm).
- `tests/path_b_diagnostic.rs` (≤150 lines) — runs N prompts, for each step records `(hot_top_1, ‖hidden‖, max_cold_norm, bound_held)`, reports aggregate safe-skip rate. Opt-in via `OLORIN_PATH_B_DIAGNOSTIC=1`.

**Modify (Phase 1):**
- `src/inference/engine.rs` — add `pub row_norm_cache: Option<RowNormCache>` to `Gemma4Model`. Populate from the Q6K embed weight at `from_gguf` time (dequant each row, compute ‖·‖₂, cache the scalars).
- `src/inference/forward_graph.rs` — add diagnostic instrumentation gated on `OLORIN_PATH_B_DIAGNOSTIC=1`. After the existing output matmul + softcap, compute `‖hidden‖`, compare `max(logits[hot_indices]) > ‖hidden‖ × max_cold_norm`, emit a stat line per step.

**Create (Phase 2, GATED on Phase 1 pass):**
- `tests/path_b_parity.rs` — ensures Path B enabled with hot_size = vocab_size produces bit-identical output stream to today's path.
- `tests/path_b_fallback.rs` — artificially forces fallback (e.g., sets bound threshold to infinity) and asserts identical output to today.

**Modify (Phase 2, GATED):**
- `src/inference/forward_graph.rs` — when `OLORIN_PATH_B=<N>`: compute only rows 0..N in the output matmul initially; check bound; on success, argmax hot set and emit sampled position; on failure, complete the cold matmul then argmax the full logit vector.
- `src/inference/engine.rs` — permute output-head rows at load time so hot rows are contiguous 0..N (needed so the existing matvec kernels can work on a prefix with no kernel changes). Also store the permutation + inverse-permutation tables on `Gemma4Model`.
- `src/inference/generate.rs` — sampling path: translate sampled position through perm table → original vocab id. (Identical pattern to what Path A used; reuse the code structure if it helps.)

---

## Phase 1 — Diagnostic

### Task 1: RowNormCache — precompute at load time

**Files:**
- Create: `src/inference/path_b.rs`
- Modify: `src/inference/engine.rs` (add field + initialization)
- Modify: `src/inference/mod.rs` (`pub mod path_b;`)
- Test: `tests/path_b_norm_cache.rs`

The output head is Q6K-quantized with 262144 rows of 1536 elements each. To compute `row_norm[i]`, we need to dequantize each row, compute its L2 norm, and store the scalar. One-time cost at load.

- [ ] **Step 1: Write the failing test**

```rust
// tests/path_b_norm_cache.rs
use olorin::inference::path_b::RowNormCache;
use olorin::inference::gguf::GgufFile;
use olorin::inference::engine::Gemma4Model;
use std::path::Path;

fn model_path() -> String {
    std::env::var("OLORIN_MODEL").unwrap_or_else(|_|
        format!("{}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf",
                std::env::var("HOME").unwrap()))
}

#[test]
fn row_norm_cache_shape_and_sanity() {
    if std::env::var("OLORIN_PATH_B_TESTS").is_err() {
        eprintln!("skipping — needs the 3 GB model; set OLORIN_PATH_B_TESTS=1");
        return;
    }
    let gguf = GgufFile::open(Path::new(&model_path())).unwrap();
    let model = Gemma4Model::from_gguf(&gguf).unwrap();
    let cache = RowNormCache::from_embed_q6k(&model).unwrap();
    assert_eq!(cache.len(), model.vocab_size);
    // Sanity: all norms finite, non-negative.
    for (i, &n) in cache.row_norms().iter().enumerate() {
        assert!(n.is_finite(), "row {i} norm {n}");
        assert!(n >= 0.0, "row {i} norm {n} < 0");
    }
    // Sanity: at least a few rows are meaningfully large (not all zeros).
    let max_norm = cache.row_norms().iter().copied().fold(0.0f32, f32::max);
    assert!(max_norm > 0.1, "max row norm suspiciously small: {max_norm}");
    // Sanity: hot subset is sorted top-down by norm.
    let hot_n = 30_000;
    let hot_indices = cache.select_hot_indices(hot_n);
    assert_eq!(hot_indices.len(), hot_n);
    let max_cold = cache.max_cold_norm_for(&hot_indices);
    let min_hot = hot_indices.iter().map(|&i| cache.row_norms()[i as usize])
        .fold(f32::INFINITY, f32::min);
    assert!(min_hot >= max_cold,
            "min hot norm {min_hot} < max cold norm {max_cold} — selection inverted");
}
```

- [ ] **Step 2: Run test to see it fail**

```
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
    cargo test --test path_b_norm_cache 2>&1 | tail -5
```
Expected: compile error (`path_b` module not found).

- [ ] **Step 3: Implement `RowNormCache`**

```rust
// src/inference/path_b.rs
//! Cauchy-Schwarz bound infrastructure for output-head-row skip.
//!
//! `row_norm[i] = ‖embed_weight_row_i‖₂` for i in 0..vocab_size.
//! At inference time, hot top-1 > ‖hidden‖ × max_cold_norm proves no
//! cold row can exceed the hot top-1 (Cauchy-Schwarz), so we can skip
//! the cold-row matmul on that step.

use crate::error::{Error, Result};
use crate::inference::engine::Gemma4Model;

pub struct RowNormCache {
    row_norms: Vec<f32>,          // len = vocab_size
}

impl RowNormCache {
    /// Dequantize the Q6K output-head rows and compute L2 norms. One-time
    /// cost at model load. For Gemma 4 E2B: 262144 rows × 1536 floats =
    /// ~1.6 GB of transient dequantized data processed one row at a time,
    /// storing just 1 MB of row-norm scalars.
    pub fn from_embed_q6k(model: &Gemma4Model) -> Result<Self> {
        let vocab_size = model.vocab_size;
        let hd = model.hidden_dim; // 1536 for E2B
        let mut row_norms = vec![0.0f32; vocab_size];
        // Scratch for dequantizing one row at a time.
        let mut scratch = vec![0.0f32; hd];
        for i in 0..vocab_size {
            // Use existing dequant path. If the embed is Q6K (usual case):
            //   crate::inference::dequant::q6k_dequant_row(&model.embed_weight, i, &mut scratch)
            // If the embed is f16/f32 for some reason, handle that too.
            Self::dequant_row_into(&mut scratch, model, i)?;
            let n2: f32 = scratch.iter().map(|&x| x * x).sum();
            row_norms[i] = n2.sqrt();
        }
        Ok(Self { row_norms })
    }

    fn dequant_row_into(
        dst: &mut [f32], model: &Gemma4Model, row: usize,
    ) -> Result<()> {
        // TODO: call the existing dequant helper. Look at where
        // engine_helpers::populate_embed_q6k_repacked consumes the Q6K
        // bytes and mirror the per-row slice access.
        // If no public per-row dequant exists, add a helper in dequant.rs:
        //   pub fn q6k_dequant_single_row(data: &[u8], row: usize, hd: usize, out: &mut [f32])
        // using the same math as existing block decoders (see q6k_dot.ea
        // or its C reference in the GGML source for the exact formula).
        unimplemented!("dequant one row of the Q6K embed weight");
    }

    pub fn len(&self) -> usize { self.row_norms.len() }
    pub fn row_norms(&self) -> &[f32] { &self.row_norms }

    /// Top-N row indices by norm, sorted descending.
    pub fn select_hot_indices(&self, n: usize) -> Vec<u32> {
        let mut pairs: Vec<(u32, f32)> = self.row_norms.iter().enumerate()
            .map(|(i, &v)| (i as u32, v)).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.into_iter().take(n).map(|(i, _)| i).collect()
    }

    /// Max norm over indices NOT in `hot`. O(vocab_size); ~1 MB scan, runs once.
    pub fn max_cold_norm_for(&self, hot: &[u32]) -> f32 {
        let hot_set: std::collections::HashSet<u32> = hot.iter().copied().collect();
        self.row_norms.iter().enumerate()
            .filter(|(i, _)| !hot_set.contains(&(*i as u32)))
            .map(|(_, &n)| n)
            .fold(0.0f32, f32::max)
    }
}
```

Register in `src/inference/mod.rs`:
```rust
pub mod path_b;
```

Add to `Gemma4Model` in `src/inference/engine.rs`:
```rust
pub row_norm_cache: Option<crate::inference::path_b::RowNormCache>,
```

Populate in `from_gguf` only when `OLORIN_PATH_B_DIAGNOSTIC=1` or `OLORIN_PATH_B=<N>` is set:
```rust
let row_norm_cache = if std::env::var("OLORIN_PATH_B_DIAGNOSTIC").is_ok()
                     || std::env::var("OLORIN_PATH_B").is_ok() {
    Some(crate::inference::path_b::RowNormCache::from_embed_q6k(&partial_model)?)
} else { None };
```

Note: `from_embed_q6k` is called AFTER `embed_weight` is populated on the model. Verify ordering in the current `from_gguf`.

- [ ] **Step 4: Find/write the Q6K per-row dequant helper**

Check if `src/inference/dequant.rs` already exposes a per-row dequant. If yes, use it. If no, add:
```rust
pub fn q6k_dequant_single_row(data: &[u8], row: usize, hd: usize, out: &mut [f32]) {
    // Q6K row stride is hd/256 * 210 bytes (block_size = 256 elements, 210 bytes/block).
    // Walk the blocks, decode each, write to out.
    // Cross-check against ggml-quants.c `dequantize_row_q6_K` for the exact math.
    // Bit-exact parity is required — this feeds row norms used for a
    // correctness-preserving bound, so drift here could make the bound
    // either too loose (safe, wasted work) or TOO TIGHT (UNSAFE — wrong
    // argmax sometimes accepted). Loose is fine; tight is a correctness bug.
    unimplemented!();
}
```

If implementing from scratch, reference the Ea kernel at `kernels/q6k_dot.ea` for the math.

- [ ] **Step 5: Run the test to verify PASS**

```
OLORIN_PATH_B_TESTS=1 OLORIN_PATH_B_DIAGNOSTIC=1 OLORIN_MODEL=/home/peterlukka/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf \
    PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
    cargo test --release --test path_b_norm_cache 2>&1 | tail -20
```
Expected: PASS. Cache load will take 10-60 seconds (dequantizing 262144 rows) but that's one-time.

- [ ] **Step 6: Commit**

```
git add src/inference/path_b.rs src/inference/mod.rs src/inference/engine.rs \
        src/inference/dequant.rs tests/path_b_norm_cache.rs
git commit -m "feat(path-b): RowNormCache — precompute embed row L2 norms at load"
```

---

### Task 2: Diagnostic mode — "would the bound have succeeded here?"

**Files:**
- Modify: `src/inference/forward_graph.rs` (add diagnostic after existing output matmul)
- Create: `tests/path_b_diagnostic.rs`

Goal: WITHOUT changing today's behavior, instrument every decode step to record whether the Cauchy-Schwarz bound would have allowed a skip. This gives us the safe-skip rate distribution before we commit to building the partial-matmul path.

- [ ] **Step 1: Wire diagnostic instrumentation into forward_graph**

Find the output-matmul-then-softcap block in `src/inference/forward_graph.rs` (around `logit_rows = model.hot_size.unwrap_or(model.vocab_size)`). AFTER softcap is applied (logits are now the final output), add:

```rust
if let Some(ref cache) = model.row_norm_cache {
    if std::env::var("OLORIN_PATH_B_DIAGNOSTIC").is_ok() {
        let hot_n = std::env::var("OLORIN_PATH_B_DIAGNOSTIC_HOT_N")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(30_000usize);
        let hot_indices = cache.select_hot_indices(hot_n);
        let max_cold_norm = cache.max_cold_norm_for(&hot_indices);
        let hidden_norm = {
            // state.x_norm is the post-final-rmsnorm hidden state used in
            // the matmul. Use it here — but read-only, don't perturb.
            let s: f32 = state.x_norm.iter().map(|&x| x * x).sum();
            s.sqrt()
        };
        let hot_top_1 = hot_indices.iter()
            .map(|&i| state.logits[i as usize])
            .fold(f32::NEG_INFINITY, f32::max);
        let bound = hidden_norm * max_cold_norm;
        let bound_held = hot_top_1 > bound;
        eprintln!("[PATH-B-DIAG] step_pos={} hot_n={} hot_top_1={:.3} hidden_norm={:.3} \
                   max_cold_norm={:.3} bound={:.3} held={}",
                  pos, hot_n, hot_top_1, hidden_norm, max_cold_norm, bound, bound_held);
    }
}
```

Important: IF softcap was applied, hot_top_1 is post-softcap. Post-softcap values are bounded in `(-30, 30)`. The bound `‖hidden‖ × max_cold_norm` is pre-softcap. **These are NOT directly comparable** — comparing post-softcap to pre-softcap loses the bound.

Fix: capture `hot_top_1` BEFORE softcap is applied. Restructure the code to:
1. Run matmul.
2. Capture hot top-1 from raw logits.
3. Run softcap.
4. Continue normally.

Or: capture the full pre-softcap logits before softcap runs in place. Check how softcap is called — if it's `ffi_inference::softcap_f32(state.logits.as_mut_ptr(), ...)`, you can record the hot max before that line.

- [ ] **Step 2: Create the diagnostic test**

```rust
// tests/path_b_diagnostic.rs
use std::path::PathBuf;
use olorin::inference::generate::Engine;

const DIAG_PROMPTS: &[&str] = &[
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Write a haiku about autumn.",
    "How do I sort a list in Python?",
    "Förklara vad en transformer är.",
    "Count from 1 to 5.",
    "Write a Python function that reverses a string.",
    "Implement binary search in Rust.",
    // Add ~20 more drawn from diverse categories; structural_prefix_check
    // history (see branch gemma4-specialize history ba8c045) has a good list.
];

fn model_path() -> PathBuf {
    std::env::var("OLORIN_MODEL").map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(
            format!("{}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf",
                    std::env::var("HOME").unwrap())))
}

/// Generates a fixed number of tokens per prompt and relies on the
/// [PATH-B-DIAG] eprintln for the data. Parse the log afterwards for
/// aggregation.
#[test]
fn path_b_diagnostic_run() {
    if std::env::var("OLORIN_PATH_B_DIAGNOSTIC").is_err() {
        eprintln!("skipping — set OLORIN_PATH_B_DIAGNOSTIC=1 to run.");
        return;
    }
    let mut engine = Engine::load(&model_path(), 2048).expect("engine load");
    engine.temperature = 0.0;
    engine.top_k = 1;
    engine.max_tokens = 64;
    for (i, p) in DIAG_PROMPTS.iter().enumerate() {
        eprintln!("=== prompt {}/{} === {:?}", i + 1, DIAG_PROMPTS.len(),
                  p.chars().take(40).collect::<String>());
        let _ = engine.generate(p, "", &|_| {}).unwrap();
    }
    eprintln!("\n=== DONE. Pipe output through `grep PATH-B-DIAG | awk ...` \
               to compute safe-skip rate. ===");
}
```

- [ ] **Step 3: Run diagnostic on Pi** (primary signal source)

Cross-compile + scp + run as established by `reference_pi_deploy_workflow.md`:

```bash
# Local WSL:
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
    RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
    cargo test --release --target aarch64-unknown-linux-gnu --no-run \
    --test path_b_diagnostic

scp -i ~/.ssh/id_ed25519_pi \
    target/aarch64-unknown-linux-gnu/release/deps/path_b_diagnostic-* \
    peter@10.46.0.27:~/path_b_diag

# Pi:
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 \
    'pgrep -af "olorin|llama" | grep -v pgrep || echo clean'
# ensure clean before running — see feedback_check_shells.md

ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 \
    'OLORIN_PATH_B_DIAGNOSTIC=1 OLORIN_THREADS=3 ~/path_b_diag --nocapture 2>&1 | tee /tmp/path_b_log'

# Analysis:
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 \
    'grep PATH-B-DIAG /tmp/path_b_log | awk -F"held=" "{print \$2}" | sort | uniq -c'
```

Expected log analysis produces two counts: `N_true` and `N_false`. Safe-skip rate = `N_true / (N_true + N_false)`.

- [ ] **Step 4: The Phase 1 GATE**

If safe-skip rate **≥ 20%**: Phase 2 is worth building. Write an empty-commit note with the rate and proceed.
If safe-skip rate **< 20%**: Phase 2 is NOT worth building at the Cauchy-Schwarz looseness. Stop. Options: (a) try multiple hot-sizes (larger N shrinks max_cold_norm; smaller has less work), (b) try a tighter bound (e.g., precomputed per-row norm cached, check top-K cold rows individually), (c) abort Path B and move to AQ.

- [ ] **Step 5: Commit Phase 1 outcome**

```
git add src/inference/forward_graph.rs tests/path_b_diagnostic.rs
git commit -m "feat(path-b): diagnostic mode — measure safe-skip rate per step"

# After running + measuring:
git commit --allow-empty -m "notes(path-b): Phase 1 safe-skip rate = XX.X% at hot_n=30000 (PASS|FAIL)

  N_true: ...
  N_false: ...
  hot_n=10000: YY.Y%   (scan a few hot sizes if time allows)
  hot_n=30000: XX.X%
  hot_n=50000: ZZ.Z%

Decision: Phase 2 GO|STOP.
"
```

---

## Phase 2 — Actual skip (GATED on Phase 1 ≥ 20%)

### Task 3: Row permutation at load

Permute output-head rows at load time so that hot indices (top-N by norm) land at 0..N. Makes the partial matmul trivial — kernel just processes a prefix. Same pattern Path A used — carry the implementation sketch forward but keyed on row_norm instead of frequency.

(Skip template details — structure matches Task 6 of the adaptive-quant branch plan `docs/superpowers/plans/2026-04-21-hot-vocab-empirical.md` Task 6 almost line-for-line, with `OLORIN_HOT_VOCAB_N` replaced by `OLORIN_PATH_B` and hot-set derived from norms not frequency. Read that plan for the permutation + inverse-permutation details, copy the general approach, change the selection criterion.)

### Task 4: Forward pass — partial matmul then bound check then conditional cold matmul

The CORE change. In `forward_graph.rs`:

1. First phase: matvec rows 0..N (hot) into `state.logits[0..N]`.
2. Compute `‖hidden‖` (thread 0).
3. Compute `hot_top_1 = max(state.logits[0..N])`.
4. Branch: if `hot_top_1 > ‖hidden‖ × max_cold_norm`, skip to sampling.
5. Else: continue matvec rows N..vocab_size into `state.logits[N..]`, then proceed normally.

The kernel doesn't change — we're just calling it twice (once for hot, conditionally once for cold).

### Task 5: Sampler mapping

Same as Path A's Task 8. Sampled-position → original id via perm table.

### Task 6: Parity tests

- `tests/path_b_parity.rs`: with `OLORIN_PATH_B=<vocab_size>` (all rows hot, never skip), output stream must be byte-identical to today.
- `tests/path_b_fallback.rs`: force bound to always fail (set `OLORIN_PATH_B_FORCE_FALLBACK=1` — gate the bound check on that var), output must still be byte-identical.

### Task 7: Measured correctness on adversarial prompts

Run the `structural_prefix_check` prompts from gemma4-specialize history (including "Count from 1 to 5" which is a known divergence) under Path B enabled, verify argmax stream matches today's path token-for-token.

### Task 8: Pi bench

Cross-compile, scp, bench at `OLORIN_THREADS=3` with `OLORIN_PATH_B=30000` (or whatever Phase 1 identified as best). Compare to `ba8c045` baseline (6.81 t/s Pi decode). Empty-commit the result.

### Task 9: Empty-commit the measurement and update memory

Write the Pi decode delta, the observed safe-skip rate at final hot_n, the wall-clock comparison. Update `project_path_b_hot_vocab.md` memory with the outcome.

---

## Self-Review

**Spec coverage:**
- Cauchy-Schwarz bound: Task 1 (norm cache) + Task 4 (bound check) ✓
- Correctness ≠ empirical: bound is always mathematically safe ✓
- Diagnostic-before-implementation gate: Phase 1 Task 2 Step 4 ✓
- Pi as primary signal: Task 2 Step 3 + Task 8 ✓
- Parity tests: Task 6 ✓

**Known open items:**
- `q6k_dequant_single_row` may or may not exist already. Task 1 Step 4 tells the implementer to check first, add if missing. The math reference is `kernels/q6k_dot.ea` or llama.cpp's `ggml-quants.c`.
- Softcap-vs-bound comparison issue: captured in Task 2 Step 1 with explicit fix.
- Hot-set size N is parameterized (`OLORIN_PATH_B_DIAGNOSTIC_HOT_N`, `OLORIN_PATH_B`) so the final N can be tuned based on Phase 1 data.
- The permutation step in Phase 2 requires owning a copy of `embed_q6k_repacked` (330 MB). Same memory-cost tradeoff as Path A noted. Acceptable on Pi 8GB.

**Things to NOT do:**
- Don't use Path A's frequency calibration data for hot-set selection. Path B uses norms. This is the key distinction from Path A.
- Don't push Phase 2 tasks before Phase 1 gate passes.
- Don't trust WSL for < 5% claims. Pi always.
- Don't kill the Pi kiosk chromium. Don't run multiple benches concurrently.
- If softcap ordering trips you up, stare at it until you're sure hot_top_1 is pre-softcap. Bound comparison is pre-softcap.
