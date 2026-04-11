# Q4K Repack — Phase B.2 Design

**Status:** design approved, plan to be written via `superpowers:writing-plans`.
**Branch:** `gemma4-batched-prompt-eval`.
**Prerequisites:** Phase A (`2026-04-11-q4k-repack-phase-a-ffi.md`) and Phase B.1 (`2026-04-11-q4k-repack-phase-b-core.md`) are landed (commits `5c6f9ca`, `bc26fae`, `72a7769`, `bb71265`).
**Supersedes:** the Path B.2 deferral in Phase B.1's "Not in scope" list.

---

## 1. Goal

Extend the Q4K 8×8 repack dispatch from Path A (ThreadPool, used only by
`forward_one` and `tests/gemma4_parallel_regression.rs`) into **Path B**
(work-stealing `forward_graph` / `matmul_graph`, used by production
`forward_one_graph` and `generate.rs`), so the production decode path
actually benefits from the repacked weights instead of the existing 4-row
kernel.

At the same time, land a new **fused dual 8×8 matvec Ea kernel**
(`q4k_8x8_q8k_matvec_dual`) that processes the `ffn_gate` and `ffn_up`
matrices in a single kernel call with shared Q8 input loads and shared
broadcast operands, and use it on **both** Path A (retrofit — replacing
B.1's deferred "two separate 8×8 calls" compromise) and Path B.

After this plan lands:
- `generate.rs` — production decode — uses the repacked weight path on
  every Q4K matmul, via Path B's work-stealing dispatch.
- The gate+up pair, the biggest single memory-bandwidth consumer in a
  forward pass, runs through one fused kernel call with the Q8K broadcasts
  held in registers across both weight streams.
- Path A and Path B use **identical kernels**, symmetric dispatch, and the
  same bit-exact guarantees.
- Phase B.1's "measure at Phase B.3" TODO on the dual path is deleted,
  not updated.

## 2. Non-goals

- **Prompt-eval throughput.** Prompt-eval still runs per-token after B.2.
  The actual batched GEMM kernel (`q4k_8x8_q8k_gemm`) is Phase 2 —
  a separate plan, still unwritten. The branch name `gemma4-batched-prompt-eval`
  is aspirational; B.2 is on the critical path to that name but does not
  land the final piece.
- **ARM NEON run-time validation.** ARM kernel compiles and cross-links;
  Pi 5 validation lives in a follow-up.
- **Selective per-weight repack gating.** B.1 repacks every eligible
  weight; B.2 inherits that and does not measure whether skipping small
  matrices (e.g. `wk`, `wv`) would pay off.
- **Flash attention.** Phase 3, future.

## 3. Architectural context

### 3.1 Paths A and B

Olorin1's forward pass has two parallelism backends, and every matmul call
site exists in both:

1. **Path A — ThreadPool.** `par_*` helpers in `matmul.rs` / `matmul_par.rs`,
   called from `forward_attn.rs`, reached via `forward_one` in
   `forward.rs:281`. Used by `gemma4_parallel_regression.rs` (the
   bit-exact gate) and `gemma4_verify.rs step5_logits`. Parallelism via
   `pool.run(n_threads, closure)` with contiguous tile slicing inside each
   closure. Phase B.1 routed this path through the 8×8 kernel for single
   matvecs and **left the dual (gate+up) case on two separate 8×8 calls**.

2. **Path B — work-stealing `GraphPool`.** `*_ws` helpers in
   `matmul_graph.rs`, called from `forward_graph.rs`, reached via
   `forward_one_graph` in `forward.rs:365`. Used by **production**
   (`generate.rs:107,110,140`) and `tests/bench_decode_speed.rs`.
   Parallelism via a shared `AtomicI32` current_chunk counter and a
   `SpinBarrier` between ops. Each op lifecycle: thread 0 resets chunk
   counter → barrier → all threads run ws loop fetching chunks → barrier.
   **Phase B.1 did not touch this path at all.** Production decode still
   dispatches to `q4k_dot_q8k_4row`.

### 3.2 Why both paths must update simultaneously

Keeping Path A on the old "two separate 8×8 calls" while Path B uses a
fused kernel creates an asymmetry where the *test* path and the
*production* path run different code for the same call site. The snapshot
that guards correctness (`gemma4_parallel_regression`) runs on Path A;
production runs on Path B. That's a footgun.

Retrofitting Path A is a ~13-line net simplification of `matmul.rs`
(4-case match → 2-case match + debug_assert), costs one snapshot
regeneration (which B.1 already did once, establishing the precedent),
and leaves a single kernel to maintain.

### 3.3 Fusion analysis of `q4k_8x8_q8k_matvec`

From `kernels/q4k_dot_8x8.ea`, per `(tile, super-block, sub-block)`
iteration when running two weight matrices against the same Q8K input:

**Shared (computed once, used for both A and B):**
- `row_sc = splat(q8_d[b])` — one f32 splat per super-block.
- Q8K bsums → `hadd_i16` → `q8s_half` — one scratch store per super-block.
- Q8K qs loads `la, lb, lc, ld` (64 bytes total from `q8_qs`) per sb.
- `concat_i8x16` broadcasts `v00, v01, v10, v11` per sb — feed the
  16 `maddubs` ops in the dot loop. These are the most expensive shared
  operands; holding them in registers across both weight streams is the
  primary fusion win.

**Weight-specific per (sub-block, matrix):**
- 8 × 32-byte packed weight loads (256 B per sb per matrix).
- 16 low/high nibble extract ops.
- Scale decode (utmp) + `scales_0`, `scales_1`, `mins_01` i16x16 literal
  construction.
- 16 `maddubs_i16` ops (consuming shared `v**` broadcasts).
- Integer accumulation into matrix-local `iacc`, `iacc_min`.

**Weight-specific per super-block:**
- `col_d`, `col_dmin` load from the packed header (16 bytes from f16 d/dmin).
- One FMA into `acc_row`, one FMA into `acc_min` — each matrix owns its
  own `f32x8` accumulator lanes.

**Consequence for correctness.** `acc_row_a`'s reduction sequence is
identical to calling `q4k_8x8_q8k_matvec` once on `packed_a` alone:
interleaving B-side integer work inside the same `sb` loop does not touch
A's accumulator lanes, and rows are independent. Per-output `to_bits()`
equality holds vs. "run the single kernel twice."

**Consequence for scratch.** The bsums hadd scratch depends only on Q8K
input — one 128-byte scratch is enough for the dual kernel, not two.

### 3.4 The "hybrid dual" case is unreachable

`populate_q4k_repacked` (`src/inference/engine_helpers.rs:115-134`)
repacks `w_gate` and `w_up` with the **same** `(ffn_dim, hidden_dim)`
dimensions and the same CPU-feature gate. Their dtypes always match in
every Gemma 4 quant. Therefore `(w_gate_repacked, w_up_repacked)` is
always `(Some, Some)` or `(None, None)`, never mixed. Phase B.1's
4-case match in `par_q4k_matvec_dual_maybe_repacked` (`matmul.rs:250-267`)
is defensive YAGNI. B.2 collapses it to 2 cases + a `debug_assert!` that
fires in debug builds if a future model breaks the invariant.

## 4. Scope (the commit list)

B.2 lands in **6 commits**, kernel-first TDD order. Each commit passes
per-commit gates (§6.1) before the next begins.

| # | Title | Files touched |
|---|---|---|
| 1 | Research note: dual fusion shared/weight-specific split | `docs/superpowers/research/2026-04-11-q4k-8x8-dual-fusion.md` |
| 2 | x86 AVX2 dual kernel | `kernels/q4k_dot_8x8_dual.ea` (new), `kernels/q4k_dot_8x8_dual.ea.json` |
| 3 | ARM NEON dual kernel | `kernels/q4k_dot_8x8_dual_arm.ea` (new), `.json` sibling |
| 4 | FFI binding + standalone bit-exact test (correctness gate) | `src/kernels/ffi_inference_types.rs`, `src/kernels/ffi_inference.rs`, `tests/dual_q4k_8x8.rs` (new) |
| 5 | Path A retrofit + Path B wire-up | `src/inference/matmul_par.rs`, `src/inference/matmul.rs`, `src/inference/matmul_graph.rs`, `src/inference/forward_graph.rs` |
| 6 | Snapshot regeneration + full regression sweep | `tests/snapshots/gemma4_logits_bos.bin`, commit message records new decode tok/s |

## 5. Component design

### 5.1 New Ea kernel: `kernels/q4k_dot_8x8_dual.ea` (x86 AVX2)

```ea
#[cfg(x86_64)]

export func q4k_8x8_q8k_matvec_dual(
    packed_a:  *restrict u8,     // first weight (e.g. ffn_gate), 8-row tile stride
    packed_b:  *restrict u8,     // second weight (e.g. ffn_up), same shape as packed_a
    q8_qs:     *restrict i8,     // shared Q8K input
    q8_d:      *restrict f32,    // shared
    q8_bsums:  *restrict i16,    // shared
    pow2:      *restrict f32,    // shared (unused in kernel body, API parity only)
    scratch:   *mut u8,          // 128 bytes, shared bsums hadd
    out_a:     *mut f32,         // first output
    out_b:     *mut f32,         // second output
    n_rows:    i32,              // must be multiple of 8
    n_cols:    i32               // must be multiple of 256
)
```

**Inner-loop template** (structurally derived from `q4k_dot_8x8.ea`):

```
while x < n_rows / 8:                           # tile
    acc_row_a = 0; acc_min_a = 0
    acc_row_b = 0; acc_min_b = 0

    while b < nb:                               # super-block
        row_sc = splat(q8_d[b])                             # SHARED
        col_d_a, col_dmin_a  = f16->f32 from packed_a[b]    # A-specific
        col_d_b, col_dmin_b  = f16->f32 from packed_b[b]    # B-specific

        iacc_a = 0; iacc_min_a = 0
        iacc_b = 0; iacc_min_b = 0

        q8s_half = hadd_i16(bsums_lo, bsums_hi)             # SHARED
        store(scratch_i16, 0, q8s_half)

        while sb < 4:                           # sub-block pair
            la, lb, lc, ld      = load q8_qs[b*256 + sb*64, ..]   # SHARED
            v00, v01, v10, v11  = concat broadcasts               # SHARED

            # ── A-side block (mirror of lines 87–190 of q4k_dot_8x8.ea) ──
            #   packed_a loads, low/high nibble extract,
            #   utmp/scales_0_a/scales_1_a/mins_01_a construction,
            #   16 maddubs using shared v00..v11,
            #   madd_i16 → iacc_a += iacc_0_a + iacc_1_a,
            #   iacc_min_a += madd_i16(q8s_sb, mins_01_a)

            # ── B-side block (same body, packed_b + scales_*_b + mins_01_b) ──
            #   reuses the SAME v00..v11 from registers

            sb += 1

        acc_row_a = fma(to_f32(iacc_a),     col_d_a    .* row_sc, acc_row_a)
        acc_min_a = fma(to_f32(iacc_min_a), col_dmin_a .* row_sc, acc_min_a)
        acc_row_b = fma(to_f32(iacc_b),     col_d_b    .* row_sc, acc_row_b)
        acc_min_b = fma(to_f32(iacc_min_b), col_dmin_b .* row_sc, acc_min_b)

        b += 1

    row_nat_a = shuffle(acc_row_a, [0,2,4,6,1,3,5,7])
    store(out_a, x*8, row_nat_a .- acc_min_a)
    row_nat_b = shuffle(acc_row_b, [0,2,4,6,1,3,5,7])
    store(out_b, x*8, row_nat_b .- acc_min_b)

    x += 1
```

Estimate: ~360 lines. Flat body, no helper factoring — matches the
existing single kernel's style. Under the 500-line limit.

### 5.2 New Ea kernel: `kernels/q4k_dot_8x8_dual_arm.ea` (ARM NEON)

Same public signature, mirror of `kernels/q4k_dot_8x8_arm.ea` with
interleaved a/b streams. Estimate: ~380 lines.

### 5.3 FFI binding

**`src/kernels/ffi_inference_types.rs`** — add:

```rust
pub type Q4k8x8MatvecDualFn = unsafe extern "C" fn(
    packed_a: *const u8,
    packed_b: *const u8,
    q8_qs:    *const i8,
    q8_d:     *const f32,
    q8_bsums: *const i16,
    pow2:     *const f32,
    scratch:  *mut u8,
    out_a:    *mut f32,
    out_b:    *mut f32,
    n_rows:   i32,
    n_cols:   i32,
);
```

**`src/kernels/ffi_inference.rs`** — add:
- Field `pub q4k_8x8_q8k_matvec_dual: Q4k8x8MatvecDualFn` next to the
  existing 8×8 fields (currently lines 34–35).
- Symbol load from `libq4k_dot_8x8_dual.so` next to the existing 8×8
  loads (currently lines 139–140).
- Public wrapper `pub unsafe fn q4k_8x8_q8k_matvec_dual(...)` next to the
  existing 8×8 wrappers (currently around line 285).

### 5.4 Standalone bit-exact test: `tests/dual_q4k_8x8.rs` (new)

```rust
#[test]
fn dual_matches_two_single_calls_bitexact() {
    if !has_model() { eprintln!("SKIP: no model"); return; }
    let gguf  = GgufFile::open(Path::new(&model_path())).unwrap();
    let model = Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    // ffn_gate + ffn_up from layer 0, both Q4K, identical shape.
    let lw      = &model.layers[0];
    let n_rows  = model.ffn_dim[0];
    let n_cols  = model.hidden_dim;
    let n_blocks = n_cols / 256;
    let tile_bytes = n_blocks * 1152;  // 1152 bytes per 8-row tile group
    let n_tiles = n_rows / 8;
    let total   = n_tiles * tile_bytes;

    let mut packed_gate = vec![0u8; total];
    let mut packed_up   = vec![0u8; total];
    unsafe {
        q4k_repack_8x8(lw.w_gate, packed_gate.as_mut_ptr(), n_rows as i32, n_cols as i32);
        q4k_repack_8x8(lw.w_up,   packed_up.as_mut_ptr(),   n_rows as i32, n_cols as i32);
    }

    // Non-trivial synthetic Q8K input (same pattern as tests/repack_q4k.rs).
    let (q8_qs, q8_d, q8_bsums) = make_q8k_input(n_cols);
    let pow2 = pow2_table();

    // Reference: two separate single 8×8 calls.
    let mut ref_gate = vec![0f32; n_rows];
    let mut ref_up   = vec![0f32; n_rows];
    let mut scratch_ref = [0u8; 128];
    unsafe {
        q4k_8x8_q8k_matvec(packed_gate.as_ptr(), q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
                           pow2.as_ptr(), scratch_ref.as_mut_ptr(),
                           ref_gate.as_mut_ptr(), n_rows as i32, n_cols as i32);
        q4k_8x8_q8k_matvec(packed_up.as_ptr(),   q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
                           pow2.as_ptr(), scratch_ref.as_mut_ptr(),
                           ref_up.as_mut_ptr(),   n_rows as i32, n_cols as i32);
    }

    // Test: one fused dual call.
    let mut fused_gate = vec![0f32; n_rows];
    let mut fused_up   = vec![0f32; n_rows];
    let mut scratch_fused = [0u8; 128];
    unsafe {
        q4k_8x8_q8k_matvec_dual(
            packed_gate.as_ptr(), packed_up.as_ptr(),
            q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
            pow2.as_ptr(), scratch_fused.as_mut_ptr(),
            fused_gate.as_mut_ptr(), fused_up.as_mut_ptr(),
            n_rows as i32, n_cols as i32,
        );
    }

    for i in 0..n_rows {
        assert_eq!(ref_gate[i].to_bits(), fused_gate[i].to_bits(), "gate[{i}]");
        assert_eq!(ref_up[i].to_bits(),   fused_up[i].to_bits(),   "up[{i}]");
    }
}
```

This is the correctness gate for commits 2–4. Commits 5–6 run only after
`to_bits()` equality passes on every output element of both channels.

### 5.5 Path A retrofit

**New in `src/inference/matmul_par.rs`:**

```rust
#[allow(clippy::too_many_arguments)]
pub(super) fn par_q4k_8x8_matvec_dual(
    pool: &ThreadPool,
    packed_a: *const u8,
    packed_b: *const u8,
    input_qs: &[i8], input_d: &[f32], input_bsums: &[i16],
    output_a: &mut [f32],
    output_b: &mut [f32],
    n_rows: usize, n_cols: usize,
)
```

Mirror of `par_q4k_8x8_matvec` (lines 346–411): tile-slice `n_tiles`
across threads, each thread calls `q4k_8x8_q8k_matvec_dual` on its
tile slice with its own stack-allocated 128-byte scratch, writing into
both output slices via `SendMutPtr.add(...)`. ~50 lines.

**Rewrite in `src/inference/matmul.rs`:**

```rust
#[allow(clippy::too_many_arguments)]
pub fn par_q4k_matvec_dual_maybe_repacked(
    pool: &ThreadPool,
    gate_weight: *const u8,
    up_weight: *const u8,
    gate_repacked: Option<&[u8]>,
    up_repacked: Option<&[u8]>,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert!(
        gate_repacked.is_some() == up_repacked.is_some(),
        "ffn_gate and ffn_up always repack together on Gemma-family models; \
         hybrid repacking is unreachable and unsupported"
    );
    match (gate_repacked, up_repacked) {
        (Some(g), Some(u)) => par_q4k_8x8_matvec_dual(
            pool, g.as_ptr(), u.as_ptr(),
            input_qs, input_d, input_bsums,
            gate_output, up_output, n_rows, n_cols,
        ),
        _ => par_q4k_matvec_dual(
            pool, gate_weight, up_weight,
            input_qs, input_d, input_bsums,
            gate_output, up_output, n_rows, n_cols,
        ),
    }
}
```

Deletes the current 4-case match (`matmul.rs:250-267`) and its
"measure at Phase B.3" comment block (`matmul.rs:230-234`). Net line
change: −13.

### 5.6 Path B additions

**New in `src/inference/matmul_graph.rs`:**

```rust
pub fn q4k_matvec_8x8_ws(
    packed: *const u8,
    q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
)
```

Mirrors `q4k_matvec_ws` structure: initial chunk = `ith`,
`fetch_add(1, Relaxed)` on advance, one chunk = one 8-row tile, one FFI
call per chunk with `n_rows=8`, no remainder handling (repack gate
ensures `n_rows % 8 == 0`). Stack-allocated 128-byte scratch. ~40 lines.

```rust
pub fn q4k_matvec_dual_8x8_ws(
    gate_w: *const u8, up_w: *const u8,
    q8: *const i8, q8_d: *const f32, bsums: *const i16,
    gate_out: *mut f32, up_out: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
)
```

Same work-stealing loop shape; each chunk calls
`q4k_8x8_q8k_matvec_dual` once with both packed pointers and both
output pointers. One shared 128-byte scratch. ~48 lines.

**No new dispatch wrapper function.** Unlike Path A, Path B cannot hide
dispatch behind a single helper: the `current_chunk.store(nth, ...)`
reset and the surrounding `barrier.wait()` calls live in the graph
loop, not in matmul helpers. The 9 call sites in `forward_graph.rs`
dispatch inline (§5.7).

### 5.7 Call-site rewiring in `src/inference/forward_graph.rs`

9 existing ws call sites (`forward_graph.rs:85, 143, 182, 194, 288,
327 [dual], 338, 347, 370`). Two patterns.

**Pattern A — single matvec (8 sites).** Currently:

```rust
matmul_graph::matvec_ws(
    lw.wq_dtype, lw.wq,
    state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
    state.q.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
    q_rows, hd,
    current_chunk, ith, nth,
);
```

Becomes a call to a new private helper `matvec_step` (§5.8):

```rust
matvec_step(
    lw.wq_dtype, lw.wq, lw.wq_repacked.as_deref(),
    state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
    state.q.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
    q_rows, hd,
    current_chunk, ith, nth,
);
```

**Pattern B — dual matvec (1 site, lines 322–334).** Becomes:

```rust
if lw.w_gate_dtype == matmul::GGML_TYPE_Q4_K
    && lw.w_up_dtype == matmul::GGML_TYPE_Q4_K
{
    debug_assert!(
        lw.w_gate_repacked.is_some() == lw.w_up_repacked.is_some(),
        "ffn_gate/ffn_up repack invariant violated"
    );
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    match (lw.w_gate_repacked.as_deref(), lw.w_up_repacked.as_deref()) {
        (Some(g), Some(u)) => matmul_graph::q4k_matvec_dual_8x8_ws(
            g.as_ptr(), u.as_ptr(),
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.gate.as_mut_ptr(), state.up.as_mut_ptr(),
            ffn_dim, hd,
            current_chunk, ith, nth,
        ),
        _ => matmul_graph::q4k_matvec_dual_ws(
            lw.w_gate, lw.w_up,
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.gate.as_mut_ptr(), state.up.as_mut_ptr(),
            ffn_dim, hd,
            current_chunk, ith, nth,
        ),
    }
    barrier.wait();
}
```

### 5.8 Line-limit pressure and the `matvec_step` helper

Inlining the pattern-A match at 8 sites would grow `forward_graph.rs`
from 442 to ~490 lines — under 500 but uncomfortably close and
brittle against follow-up edits. Factor the repeated match into a
private helper inside `forward_graph.rs`:

```rust
#[inline]
#[allow(clippy::too_many_arguments)]
fn matvec_step(
    dtype: u32,
    weight: *const u8,
    repacked: Option<&[u8]>,
    q8: *const i8,
    q8_d: *const f32,
    bsums: *const i16,
    output: *mut f32,
    d_scratch: *mut f32,
    n_rows: usize,
    n_cols: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    nth: usize,
) {
    // Not `unsafe fn` — `matmul_graph::matvec_ws` and `q4k_matvec_8x8_ws`
    // are both safe public fns that take raw pointers directly, matching
    // the style of the existing call sites in forward_graph.rs.
    match repacked {
        Some(p) => matmul_graph::q4k_matvec_8x8_ws(
            p.as_ptr(), q8, q8_d, bsums, output,
            n_rows, n_cols, current_chunk, ith, nth,
        ),
        None => matmul_graph::matvec_ws(
            dtype, weight, q8, q8_d, bsums, output, d_scratch,
            n_rows, n_cols, current_chunk, ith, nth,
        ),
    }
}
```

Each Pattern A call site becomes a single-line `matvec_step(...)` call.
`forward_graph.rs` stays around 450 lines with headroom. The helper is
private and has zero impact on dispatch semantics.

## 6. Verification plan

### 6.1 Per-commit gates (run before `git commit`)

1. **Build clean.**
   ```bash
   PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
   ```
   Zero warnings, zero errors.
2. **Line limit.**
   ```bash
   find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l \
     | awk '$1 > 500 && $2 != "total" {print}'
   ```
   Empty output.
3. **Phase A smoke.**
   ```bash
   cargo test --release --test repack_q4k -- --test-threads=1
   ```
   All 3 green.

### 6.2 Per-commit additional gates

| Commit | Additional gate |
|---|---|
| 1 research note | None (pure doc). |
| 2 x86 dual kernel | `build.rs` produces `libq4k_dot_8x8_dual.so` without errors. No Rust-side test yet. |
| 3 ARM dual kernel | `build.rs` produces `libq4k_dot_8x8_dual_arm.so` without errors and the `.ea.json` sibling is generated. Runtime validation (actually loading the ARM lib on the Pi 5) is a follow-up plan, not a B.2 gate. |
| 4 FFI + dual test | **`cargo test --release --test dual_q4k_8x8`** — `to_bits()` equality on every `n_rows` element of both channels. **Correctness gate for the whole plan.** |
| 5 retrofit + wire-up | (a) `cargo test --release --test gemma4_verify step5_logits` — layer-by-layer L2 norms within existing tolerance (same bar as B.1). (b) `cargo test --release --test gemma4_smoke` — end-to-end sentence completion still coherent. (c) `cargo test --release --test gemma4_parallel_regression` is **expected to FAIL** on this commit because Path A's dual path now fuses (~ULP snapshot drift). Commit 5's gate is: the only failing test is `gemma4_parallel_regression`, and its failure is restricted to the logits snapshot binary — nothing else regresses. Commit message calls this out explicitly. |
| 6 snapshot regen | `cargo test --release --test gemma4_parallel_regression -- --nocapture` green after regeneration. Re-run commit 5's three tests to confirm nothing else moved. Record new decode tok/s in commit message. |

### 6.3 Full regression sweep (before pushing the branch)

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release \
  --test repack_q4k \
  --test dual_q4k_8x8 \
  --test gemma4_verify \
  --test gemma4_parallel_regression \
  --test gemma4_smoke \
  -- --test-threads=1 2>&1 | tail -40
```

All five suites green (model-gated skips allowed only if the `.gguf` is
absent on the runner).

### 6.4 Performance gate

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release \
  --test bench_decode_speed -- --nocapture --test-threads=1 \
  2>&1 | tee /tmp/b2-bench.log
```

**Hard requirement:** decode tok/s on Gemma 4 E2B Q4_K_M, 16 threads,
this workstation must **improve** vs. the B.1 baseline
(~8.70 tok/s olorin1, vs. 8.9 tok/s llama.cpp — recorded in the
branch's eabrain session note from 2026-04-11). The B.2 target is
directional ("must move upward"); commit 6's message records the new
number. Prompt-eval throughput is **not** expected to change — batched
prompt-eval is Phase 2 and is still unwritten.

### 6.5 Diagnostic step if the bench doesn't move

If commit 5 passes correctness but commit 6's bench shows no improvement
(or a regression) vs. B.1:

- Profile with `perf stat -e L1-dcache-loads,L1-dcache-load-misses` on
  a single-thread run of just the gate+up step, before and after.
  The fusion hypothesis says L1-dcache loads should drop.
- If atomic overhead dominates, try increasing the work-stealing chunk
  step in `q4k_matvec_dual_8x8_ws` from 1 tile (8 rows) to 2 tiles
  (16 rows) per chunk.
- If neither: re-verify against `dual_q4k_8x8` and inspect the generated
  assembly for unexpected spills of `v00..v11`. Fall back to filing a
  B.3 follow-up plan rather than blocking B.2 on the perf claim.

Commits 1–4 are not gated on perf — they only need correctness. Commits
5–6 are where the perf claim lives.

## 7. What this plan does NOT do

- Prompt-eval speedup. Phase 2 (unwritten, still pending a plan).
- ARM NEON runtime validation. Pi 5 follow-up plan.
- Selective per-weight repack gating. B.1 inherited "repack everything
  eligible"; B.2 does too.
- Flash attention / online softmax. Phase 3, future.
- Touching `forward_attn.rs` (Path A call sites). Phase B.1 already
  routed those through `par_matvec_maybe_repacked` and
  `par_q4k_matvec_dual_maybe_repacked` — B.2's retrofit lives inside
  those wrapper bodies, not at the call sites.

## 8. Rollback plan

If B.2 lands and something downstream breaks (decode quality drift,
unforeseen thread-safety issue, Pi 5 runtime failure that blocks the
ARM path), rollback is:

```bash
git revert <commit6>..<commit1>
```

The revert is clean because Phase B.1 is not touched (only extended) and
because the one thing that changes destructively — the
`gemma4_parallel_regression` snapshot — is reverted alongside the code.

---

## 9. Handoff to writing-plans

This spec is the input to `superpowers:writing-plans`. The plan file
produced from it should live at
`docs/superpowers/plans/2026-04-11-q4k-repack-phase-b2.md` and expand
each of the 6 commits in §4 into explicit tasks with step-level
checkboxes, matching the format of
`2026-04-11-q4k-repack-phase-b-core.md`.
