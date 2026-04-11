# Phase 2 — Plan 1: Q4K 8×8 × Q8K batched gemm kernel

**Status:** design approved, plan to be written via `superpowers:writing-plans`.
**Branch:** `gemma4-batched-prompt-eval`.
**Prerequisites:**
- Phase B.2 landed (commits `a33de79` … `98b6b1e`). The repacked `block_q4_Kx8` weight format, FFI plumbing, `par_q4k_8x8_matvec` / `par_q4k_8x8_matvec_dual`, Path A + Path B 8×8 dispatch are all in place.
- `bc1fd87` (Phase B.2 session scope creep) must be in tree — it fixes 7 latent type errors in `kernels/q4k_dot_8x8_arm.ea` so the NEON baseline cross-compiles cleanly. Plan 1's ARM gemm derivation starts from that repaired baseline, not the original broken one.

**Part of:** Phase 2 of the batched prompt-eval effort. Phase 2 is decomposed into three independently-landing plans:
- **Plan 1 (this spec):** gemm kernel + isolated bench. ~9 commits. Zero changes to the forward pass.
- **Plan 2 (future spec):** 6 batched helper kernels (`q8k_quant_batched`, `gemma4_rmsnorm_batched`, `gemma4_rope_batched`, `gelu_mul_batched`, batched causal attention trio) + `forward_batch` method + `tests/gemma4_batch_verify.rs`. Test-only surface; `generate.rs` untouched.
- **Plan 3 (future spec):** `generate.rs` prefill wire-up + `bench_decode_speed` split + bit-exact vs `llama-eval-callback` + recorded prefill tok/s.

---

## 1. Goal

Land a bit-exact, standalone Q4K 8×8 × Q8K batched gemm Eä kernel that processes N input columns per call against a repacked weight tile, sharing weight unpack + scale decode across all N columns. Deliver both an x86 AVX2 implementation matching llama.cpp's AVX2 path at `ggml/src/ggml-cpu/arch/x86/repack.cpp:2816-3487` and an ARM NEON+dotprod implementation derived from olorin's own existing matvec kernel. Add a `q8k_repack_4` helper kernel that packs 4 rows of regular Q8K into llama.cpp's `block_q8_Kx4` interleaved layout, which the gemm consumes as its A-side input.

Plan 1 ends when:
- `kernels/q4k_dot_8x8_gemm.ea` (x86 AVX2) and `kernels/q4k_dot_8x8_gemm_arm.ea` (ARM NEON+dotprod) compile cleanly via build.rs (x86 host) and `ea --emit-asm --target-triple=aarch64-unknown-linux-gnu` (cross).
- `tests/gemm_q4k_8x8.rs` passes `to_bits()` per-output equality vs. running the existing single-column `q4k_8x8_q8k_matvec` N times, for each N in {4, 8, 16, 32}.
- `tests/bench_q4k_gemm.rs` compiles (it's currently dead code on the branch) and its existing hard gate "at N=8 the gemm is ≥ 1.5× faster than a matvec loop on the same problem" passes.
- `q8k_repack_4` has its own standalone byte-layout test proving the output struct matches `block_q8_Kx4`.

Plan 1 does not land any change that is observable from `generate.rs` or `forward_one_graph`. Production decode stays exactly where Phase B.2 left it.

## 2. Non-goals

- **Prefill speedup in `generate.rs`.** Plan 3 wires `forward_batch` into the prefill loop. Plan 1's showcase is the isolated kernel bench, not user-visible prefill.
- **Dual gemm (gate+up fused).** Phase B.2's `q4k_matvec_dual` fusion saved Q8 input broadcasts across two weight matrices. At gemm granularity the picture is different — weight reuse already dominates, so fusing two weights per gemm call may not pay. Deferred to Plan 2 or later; Plan 2's forward_batch initially issues two separate gemm calls for `ffn_gate` / `ffn_up`.
- **Bit-exact verify vs `llama-eval-callback`.** Plan 3 does that. Plan 1 verifies gemm vs. matvec-loop, which is sufficient because the matvec kernel is already bit-exact vs. llama.cpp on a per-output basis (Phase B.2 locked that).
- **ARM runtime validation.** `aarch64-linux-gnu-gcc` is not installed on this workstation; ARM verification is cross-compile + `--emit-asm` only. Pi 5 runtime deferred to Plan 3 or a later plan.
- **Selective repack gating, flash attention, Q5K/Q6K gemm.** Future work.
- **N < 4 support.** llama.cpp's gemm asserts `nr % 4 == 0`. Plan 1 inherits the constraint — batch sizes < 4 are handled by the existing matvec upstream (Plan 2's `forward_batch` decides when to fall through).

## 3. Architectural context

### 3.1 Why an A-side repack (`q8k_repack_4`) is in Plan 1

llama.cpp's x86 AVX2 gemm at `repack.cpp:2816–3487` reads its A-side input from `a_ptrs[rp][b].qs` where `a_ptrs` is an array of 4 `block_q8_Kx4 *`. Each `block_q8_Kx4` holds 4 interleaved Q8K rows (1024 quant bytes + 4 f32 deltas + 64 i16 bsums), giving 4 A-rows per struct. The outer loop steps `y` by 4 (16 rows per pass via 4 `a_ptrs` pointers), and the inner maddubs sequence reads all 4 rows at once from one struct via `_mm256_loadu_si256` + permute shuffles.

Olorin's existing Q8K pipeline (`matmul::quant_input` and the matvec kernels) uses **per-row** Q8K storage — each row's `qs / d / bsums` lives in a separate contiguous array. To feed llama.cpp's hot loop without fighting its load pattern, Plan 1 adds a `q8k_repack_4` Eä kernel that packs 4 per-row Q8K inputs into one `block_q8_Kx4` struct. The gemm kernel then consumes the interleaved struct directly.

Amortization: a forward_batch call runs ~35 layers × ~4 Q4K matmuls per layer = ~140 gemm calls per prefill step. The `q8k_repack_4` cost is paid once per (input, layer) — the same Q8K input is reused across the layer's 4 matmuls (Wq, Wk, Wv, Wo then again for FFN gate, up, down via a second quantization). Effectively ~35 × 2 = 70 repacks per prefill step. Each repack is a simple byte-copy + permutation with no arithmetic, so the per-repack cost is small (~10 µs for 1024 bytes per row × 4 rows at modern memory bandwidth). Negligible vs. the gemm work.

### 3.2 AVX-512 constraint — hard-coded

Olorin kernels target x86 AVX2 only. Never use `__m512` / AVX-512 / `evex` intrinsics (`_mm512_*`, `vpermi2d`, `vcompress`, `vpermt2*`, `_kmask*`, etc.) in any olorin kernel. An earlier AVX-512 attempt on this project was a trap that had to be deleted wholesale. When porting algorithms from llama.cpp, always target the AVX2 fallback branch, not the AVX-512 branch, even when the AVX-512 code is denser and better documented.

This rule drove the A-side choice for Plan 1: llama.cpp's AVX-512 path is a 16×16 tile using 16 `__m512` accumulators (16 row × 16 col lanes). The AVX2 fallback is a **16×8 tile** using 16 `__m256` accumulators × 8 col lanes, with a second 16-vector mins accumulator stack that the compiler spills (32 `__m256` locals exceed AVX2's 16 YMM registers). Olorin's gemm matches the AVX2 tile shape and accepts the spill.

### 3.3 Pi 5 reality — no I8MM

Olorin's deployment target for ARM is the Raspberry Pi 5 (Cortex-A76). Cortex-A76 has NEON + dotprod but **no I8MM** (`__ARM_FEATURE_MATMUL_INT8`). Both of llama.cpp's ARM NEON gemm paths for `ggml_gemm_q4_K_8x8_q8_K` require I8MM (lines 3773–4081 SVE2+I8MM, 4083–4268 NEON+I8MM). On Cortex-A76 both guards fail and the function falls through to the **generic scalar fallback** at `repack.cpp:4269` — `ggml_gemm_q4_K_8x8_q8_K_generic`. Pi 5 prefill via llama.cpp is scalar.

Consequences:
- Plan 1's ARM NEON+dotprod gemm has **no llama.cpp reference to match** — there is no dotprod-only ARM gemm in llama.cpp for this quant type.
- The kernel is derived by analogy from olorin's own `kernels/q4k_dot_8x8_arm.ea` (the matvec), extending the per-output `acc_row` + `acc_min_rows` state to hold N-col accumulators per row.
- The ARM kernel is original work — Phase B.2's pattern of "line-for-line port from llama.cpp" does not apply.
- On Pi 5, beating llama.cpp on prefill is a low bar. Even a naive NEON+dotprod gemm should crush a scalar fallback. The hard target is x86 prefill matching llama.cpp's AVX2 path.

### 3.4 Relationship to Phase B.2's repack and matvec

Phase B.2 landed `q4k_repack_8x8` (weight-side) and `q4k_8x8_q8k_matvec` (single-column matvec). The Plan 1 gemm reads the **same repacked weight format** — `block_q4_Kx8` tiles, 1152 bytes per 8-row tile — that B.2's matvec reads. No new weight repack is needed; `try_repack_q4k` / `populate_q4k_repacked` / `LayerWeights.*_repacked` continue to carry the weight side unchanged.

The only new repack on Plan 1 is the A-side `q8k_repack_4`, which is a runtime-per-call operation inside `forward_batch`, not a load-time operation like the weight repack.

## 4. Scope (the commit list)

Plan 1 lands in **9 commits**, kernel-first TDD. Each commit passes the per-task verification gates (§7.1) before the next begins.

| # | Title | Files |
|---|---|---|
| 1 | Research note | `docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md` (new) |
| 2 | `q8k_repack_4.ea` x86 | `kernels/q8k_repack_4.ea` |
| 3 | `q8k_repack_4_arm.ea` ARM NEON | `kernels/q8k_repack_4_arm.ea` |
| 4 | FFI binding for `q8k_repack_4` + bit-exact layout test | `src/kernels/ffi_inference_types.rs`, `src/kernels/ffi_inference.rs`, `tests/q8k_repack_4.rs` (new) |
| 5 | `q4k_dot_8x8_gemm.ea` x86 AVX2 kernel (16×8 tile, 32 accumulators with compiler spill) | `kernels/q4k_dot_8x8_gemm.ea` |
| 6 | `q4k_dot_8x8_gemm_arm.ea` ARM NEON+dotprod kernel (derived from existing matvec) | `kernels/q4k_dot_8x8_gemm_arm.ea` |
| 7 | FFI binding for gemm + bit-exact test vs. matvec loop (**correctness gate**) | `src/kernels/ffi_inference_types.rs`, `src/kernels/ffi_inference.rs`, `tests/gemm_q4k_8x8.rs` (new) |
| 8 | Rewrite `tests/bench_q4k_gemm.rs` Q8K input setup to use `block_q8_Kx4` + confirm the ≥1.5× gate at N=8 (**perf gate**) | `tests/bench_q4k_gemm.rs` (currently dead code on branch) |
| 9 | eabrain memory update + self-review | no files; commit is the eabrain entry + optional README touch |

Commits 2–4 land the Q8K A-side repack as a self-contained unit — the new kernel is dead code until Commit 7's FFI binding, but the repack is its own provable layer. Commits 5–7 land the gemm kernel and prove correctness. Commit 8 delivers the performance claim. Commit 9 is optional; if the bench numbers fit cleanly in Commit 8's message, Commit 9 collapses.

## 5. Component design

### 5.1 `block_q8_Kx4` struct (llama.cpp layout, locked)

From `ggml/src/ggml-cpu/repack.h:96-100` in llama.cpp:

```c
struct block_q8_Kx4 {
    float   d[4];              // 4 row-scales, one per packed row
    int8_t  qs[QK_K * 4];      // 4 rows × 256 quant bytes = 1024 bytes, interleaved
    int16_t bsums[QK_K / 4];   // 64 i16 bsums total (16 per row × 4 rows)
};
static_assert(sizeof(block_q8_Kx4) == sizeof(float) * 4 + QK_K * 4 + (QK_K / 4) * sizeof(int16_t));
```

Total size: `4*4 + 256*4 + 64*2 = 16 + 1024 + 128 = 1168 bytes` per `block_q8_Kx4`. Covers 4 rows × 1 super-block.

**Rust-side struct, for `q8k_repack_4` output + gemm input:**

```rust
#[repr(C)]
pub struct BlockQ8Kx4 {
    pub d: [f32; 4],
    pub qs: [i8; 1024],
    pub bsums: [i16; 64],
}
const _: () = assert!(std::mem::size_of::<BlockQ8Kx4>() == 1168);
```

Exposed from `src/inference/matmul.rs` or a new `src/inference/gemm_layout.rs` module.

### 5.2 `q8k_repack_4` Eä kernel

**Signature (both arches):**

```ea
export func q8k_repack_4(
    row0_qs: *restrict i8,    // row 0 quants (256 bytes)
    row1_qs: *restrict i8,    // row 1 quants
    row2_qs: *restrict i8,    // row 2 quants
    row3_qs: *restrict i8,    // row 3 quants
    row_d: *restrict f32,     // 4 row deltas, contiguous
    row0_bsums: *restrict i16, // row 0 bsums (16 i16)
    row1_bsums: *restrict i16,
    row2_bsums: *restrict i16,
    row3_bsums: *restrict i16,
    dst: *mut u8,             // block_q8_Kx4 output, 1168 bytes
    nb: i32                   // number of super-blocks (one block_q8_Kx4 per super-block)
)
```

The kernel walks `nb` super-blocks. For each super-block:
- Writes `dst[0..16]` = the four row deltas (one f32 per row).
- Writes `dst[16..1040]` = the four rows' quants interleaved per llama.cpp's layout (see §5.3 below).
- Writes `dst[1040..1168]` = the four rows' bsums interleaved (16 i16 per row × 4 rows = 64 i16 total).

**Interleaving pattern for `qs`:** llama.cpp's `block_q8_Kx4` loads 4 rows at once via `_mm256_loadu_si256` at `a_ptrs[rp][b].qs + N*32`. The natural reading is that the 1024-byte `qs` array holds row 0 bytes 0–255, then row 1, then row 2, then row 3 — **not** interleaved in the lane sense. Let me verify by checking how llama.cpp's gemm loads from this struct.

(Verification item for Task 1 of the plan: read `repack.cpp` ~2820+ and confirm the exact byte layout inside `qs[1024]`. The Plan 1 research note at Commit 1 pins this definitively before any kernel code is written. If llama.cpp's layout is "row 0 contiguous, row 1 contiguous, ..." then the repack kernel is a straight copy of 256 bytes × 4 rows. If it's lane-interleaved (e.g., groups of 8 bytes per row cycled), the repack is a shuffle. Either way the kernel is ~50 lines of `load + store` with no arithmetic.)

**Bsums layout:** `bsums[64]` is 16 per row × 4 rows. Same question — contiguous or interleaved. Research note pins this.

**x86 kernel (`q8k_repack_4.ea`):** SIMD loads + stores, probably 60–80 lines. `#[cfg(x86_64)]` gated.

**ARM kernel (`q8k_repack_4_arm.ea`):** Similarly small, ~70–90 lines. `#[cfg(aarch64)]` gated. Uses `vld1q_i8` + `vst1q_i8` plus whatever permutations the layout demands.

### 5.3 x86 AVX2 gemm kernel — `q4k_dot_8x8_gemm.ea`

**Reference:** llama.cpp `x86/repack.cpp:2816–3487` (the AVX2 fallback, inside the `#if defined(__AVX2__) || defined(__AVX512F__)` outer guard, `#else` branch of the inner AVX-512 guard).

**Signature:**

```ea
export func q4k_8x8_q8k_gemm(
    packed: *restrict u8,          // block_q4_Kx8 weights (same layout as matvec)
    a_ptrs: *restrict u8,          // block_q8_Kx4 pointers, laid out as a byte array
                                   // of n_rows_a/4 * n_blocks * 1168 bytes
    pow2: *restrict f32,           // shared scale LUT
    scratch: *mut u8,              // 144 bytes (bsums hadd + intra-tile state)
    acc_scratch: *mut f32,         // 2 * nc f32 (per-col mins spill, matches bench scaffold)
    out: *mut f32,                 // row-major, bs float stride
    bs: i32,                       // output row stride in f32s (matches llama.cpp's `bs`)
    n_rows_a: i32,                 // nr, # A-rows, must be % 4 == 0
    n_cols_b: i32,                 // nc, # B-cols, must be % 8 == 0
    n_cols_inner: i32              // n (inner K dim), must be % QK_K == 0
)
```

**Tile shape:** 16 A-rows × 8 B-cols per tile.

- Outer y loop: `y = 0; y < (nr / 4) - ((nr / 4) % 4); y += 4` stepping through groups of 4 `block_q8_Kx4` (16 A-rows per iteration).
- Inner x loop: `x = 0; x < nc / 8; x++` stepping through single `block_q4_Kx8` tiles (8 B-cols per iteration).
- b loop inside x: `b = 0; b < n / QK_K; b++` walking super-blocks.
- sb loop inside b: `sb = 0; sb < 4; sb++` walking sub-block pairs (each covers two 32-quant sub-blocks).

**Accumulator layout per tile:**

```ea
// 16 f32x8 for output rows × 8 output cols (lanes)
let mut acc_rows: f32x8 × 16 = [splat(0.0); 16]
// 16 f32x8 for mins correction, subtracted at end of b loop per row
let mut acc_min_rows: f32x8 × 16 = [splat(0.0); 16]
```

AVX2 has 16 YMM registers; 32 live `f32x8` locals force ~16 spills to stack / L1. This matches llama.cpp's AVX2 path register behavior. Compiler-managed, no manual spill code needed.

**Per-super-block body (sketch, not the full kernel):**

```ea
// Load 8 packed-weight AVX2 vectors (shared across all N cols)
let r0_mat_0 = load u8x32 packed[b, qs + sb*256 + 0..31]
let r1_mat_0 = load u8x32 packed[b, qs + sb*256 + 32..63]
... // 8 raw loads total

// blend + permutevar8x32 shuffle into rhs_raw_mat_0145_* and rhs_raw_mat_2367_*
// (llama.cpp lines ~2866-2873)

// nibble extract: low nibbles → rhs_mat_0145_00 .. _03, high → _10 .. _13
// (eight `and` + four `srli_epi16` + four `and`)

// Decode scales from packed[b].scales (utmp dance)
let scales_0_8cols: i16x16 = ...
let scales_1_8cols: i16x16 = ...
let mins_01_8cols: i16x16 = ...

// A-side: 4 row-pair iterations rp in 0..4
// Each rp reads one block_q8_Kx4 (4 rows) and feeds 4 row-accumulators
for rp in 0..4:
    // Load 4 × 256-byte rows of q8 quants from a_ptrs[rp][b].qs
    let lhs_mat_ymm_0123_0 = load i8x32 a_ptrs[rp][b].qs + 0
    let lhs_mat_ymm_01_0 = permute2f128(lhs_mat_ymm_0123_0, 0)  # rows 0,1
    let lhs_mat_ymm_23_0 = permute2f128(lhs_mat_ymm_0123_0, 17) # rows 2,3
    ... // same for _1, _2, _3 offsets 32, 64, 96

    // 16 maddubs_i16 operations producing 4 row × 2 nibble-half iacc vectors
    // (pattern from llama.cpp lines 788-990, adapted to AVX2 register widths)

    // madd_i16 applies the scale_* i16x16 vectors, giving i32x8 per row
    let iacc_row_0 = madd_i16(...) # 8 B-col lanes of int32 dot partials
    let iacc_row_1 = madd_i16(...)
    let iacc_row_2 = madd_i16(...)
    let iacc_row_3 = madd_i16(...)

    // FMA into acc_rows[rp*4 + k] for k in 0..4
    acc_rows[rp*4 + 0] = fma(to_f32(iacc_row_0), col_d * row_d_0, acc_rows[rp*4 + 0])
    acc_rows[rp*4 + 1] = fma(to_f32(iacc_row_1), col_d * row_d_1, acc_rows[rp*4 + 1])
    acc_rows[rp*4 + 2] = fma(to_f32(iacc_row_2), col_d * row_d_2, acc_rows[rp*4 + 2])
    acc_rows[rp*4 + 3] = fma(to_f32(iacc_row_3), col_d * row_d_3, acc_rows[rp*4 + 3])

    // Mins correction runs in parallel:
    let iacc_min_row_0 = madd_i16(bsums_row_0_packed, mins_01)
    ... // same for rows 1-3
    acc_min_rows[rp*4 + k] = fma(to_f32(iacc_min_row_k), col_dmin * row_d_k, acc_min_rows[rp*4 + k])
```

At end of b loop: `out[(y*4 + i)*bs + x*8 .. +8] = acc_rows[i] - acc_min_rows[i]` for each i in 0..16.

**Per-output bit-exactness invariant:** For any fixed (row_i, col_k) in the 16×8 tile, the integer reduction sequence inside a super-block (maddubs → i16 → add_epi16 → madd_epi16 scale → add_epi32 across 2 sub-blocks → cvt_epi32_ps) and the outer f32 FMA chain (per-super-block add-and-multiply by `col_d * row_d`, looped b low→high) are **identical** to running `q4k_8x8_q8k_matvec` on a single-row input for that (row_i, col_k) combination. Rows are independent across the tile. Cols are independent across the 8 lanes.

Therefore the `tests/gemm_q4k_8x8.rs` bit-exact gate (gemm output vs. N matvec calls on N separate columns) holds at `to_bits()` equality.

**Line budget for this file:**

llama.cpp's AVX2 path for the gemm is ~670 lines of C++. Olorin's Eä translation will likely run 500–800 lines because:
- Eä doesn't have macros, so `rhs_mat_0145_00` / `rhs_mat_2367_00` patterns expand explicitly.
- Nibble extract + scale decode is ~80 lines of declarative code in Eä.
- 16 maddubs × 4 rp iterations = 64 maddubs calls.
- 16 FMA stores per sb × 4 sb per b = 64 FMAs per super-block iteration.

Estimated: **~680 lines**. This **breaches the 500-line hard rule**. Two mitigation options:

**Option α: helper funcs inside one `.ea` file.** Extract per-sub-block weight unpack as a `func`, per-rp-iteration A-side load + maddubs as a `func`. Shrinks the main exported func to ~400 lines with ~150 lines of helpers in the same file. Still one file; still under 500 lines if helpers are split smart.

**Option β: split across two `.ea` files.** `q4k_dot_8x8_gemm.ea` (main kernel, 400 lines) and `q4k_dot_8x8_gemm_inner.ea` (helper funcs, 250 lines). Both `.so` get linked into one by build.rs. Loses cross-file inlining visibility for the LLVM optimizer; may cost a few % perf.

**Pick α.** β loses inlining; α keeps the kernel monolithic at the LLVM level. Task 1 of the plan (the research note) documents the helper-func decomposition so Task 5 (x86 kernel authoring) has a clear outline before the first line of Eä is written.

**Scratch sizing:** 144 bytes + `2 * n_rows_a` f32. The sizing matches the existing Phase B.2-committed `tests/bench_q4k_gemm.rs` scaffold, which allocates `scratch: vec![0u8; 144]` and `acc_scratch: vec![0f32; 2 * n]` where `n` is the batch size (= `n_rows_a` in the spec's naming). The 144 bytes are plausibly the bsums hadd intermediate state (8 sub-blocks × 2 i16 stored per tile = 32 bytes, plus per-rp row-pair temporaries rounded up to 144). The `2 * n_rows_a` f32 array is plausibly per-A-row state — perhaps running accumulator spill or low/high nibble partial results indexed by A-row. Final sizing and semantics are determined during Task 5 when the kernel is actually written. If the kernel needs more than 144 bytes or more than 2× `n_rows_a` f32s, Task 8 grows the bench buffers; the scaffolding signature is not load-bearing, only the FFI type declaration is.

### 5.4 ARM NEON+dotprod gemm kernel — `q4k_dot_8x8_gemm_arm.ea`

**No direct llama.cpp reference.** Both llama.cpp ARM gemm paths require `__ARM_FEATURE_MATMUL_INT8` (i8mm), which Cortex-A76 doesn't have. Plan 1 derives the kernel from olorin's own `kernels/q4k_dot_8x8_arm.ea` (the matvec, now repaired via commit `bc1fd87`).

**Derivation strategy:**

The existing matvec already has the per-tile loop structure (8 rows × 1 col) with:
- `acc0, acc1: f32x4` — 8-row accumulators split into two halves (rows 0..3 and 4..7)
- `bias0, bias1: i32x4` — parallel mins accumulators
- per-sb body loading weight tile, extracting nibbles, decoding scales, running `vdot_i32`-based dot loop, updating bias

The gemm extends this by holding **8 parallel column-accumulators per row-half**:
- `acc_row_0[8]: f32x4 × 8` — rows 0..3, one f32x4 per output col lane-group
- `acc_row_1[8]: f32x4 × 8` — rows 4..7
- `bias_0[8]: i32x4 × 8` and `bias_1[8]: i32x4 × 8` — mins state per col

Total: 32 `f32x4` + 32 `i32x4` = 64 vector registers. NEON has 32 vector registers; the compiler spills half. Spill pressure is similar to AVX2 but NEON's wider register file handles it slightly better.

Alternatively: process the tile as **8 rows × 4 cols** (f32x4 wide — lanes = cols), halving N's inner chunk but fitting more cleanly into NEON's native vector width. Then loop over 4-col chunks for N > 4. This matches how olorin's existing matvec uses f32x4 accumulators already.

**Pick f32x4 × 8 row + 4-col chunking.** Register pressure: 8 acc + 8 acc_min = 16 vectors, comfortable. N in {4, 8, 16, 32} decomposes to {1, 2, 4, 8} chunks of 4 cols.

**Per-output bit-exactness invariant:** Same argument as x86 — per (row, col), the integer dot + f32 FMA chain matches `q4k_8x8_q8k_matvec_arm` on a single-row input. The `tests/gemm_q4k_8x8.rs` gate catches drift.

**Line budget:** ~500 lines. The existing matvec is 260 lines (post-B.1-fix), and the gemm extension duplicates the cp loop into an 8-col-chunk sweep. Helper funcs as needed to stay under 500.

**Cross-compile verification:** `ea --emit-asm --target-triple=aarch64-unknown-linux-gnu --target=cortex-a76 --dotprod` must emit valid aarch64 assembly without errors, per Phase B.2's Task 3 precedent. No `.so` link on this workstation.

### 5.5 FFI types and wrappers

**`src/kernels/ffi_inference_types.rs`:**

```rust
pub type Q8kRepack4Fn = unsafe extern "C" fn(
    row0_qs:    *const i8,
    row1_qs:    *const i8,
    row2_qs:    *const i8,
    row3_qs:    *const i8,
    row_d:      *const f32,
    row0_bsums: *const i16,
    row1_bsums: *const i16,
    row2_bsums: *const i16,
    row3_bsums: *const i16,
    dst:        *mut u8,
    nb:         i32,
);

pub type Q4k8x8GemmFn = unsafe extern "C" fn(
    packed:       *const u8,
    a_ptrs:       *const u8,
    pow2:         *const f32,
    scratch:      *mut u8,
    acc_scratch:  *mut f32,
    out:          *mut f32,
    bs:           i32,
    n_rows_a:     i32,
    n_cols_b:     i32,
    n_cols_inner: i32,
);
```

**`src/kernels/ffi_inference.rs`:** Add two fields to `KernelTableInference`, two library loads (`q8k_repack_4_lib`, `q4k_dot_8x8_gemm_lib`), two symbol transmutes, update the `libs` vec. Add two public `pub unsafe fn` wrappers mirroring the Phase B.2 style.

### 5.6 Test 1: `tests/q8k_repack_4.rs` (new)

One test function: given 4 rows of synthetic Q8K input (per-row `qs`, `d`, `bsums`), call `q8k_repack_4` and verify the resulting `block_q8_Kx4` matches the expected byte layout.

```rust
#[test]
fn q8k_repack_4_matches_block_q8_Kx4_layout() {
    olorin::kernels::ffi::init().unwrap();

    let n_blocks = 3; // arbitrary, multi-super-block to catch stride bugs
    let qk = 256;

    // Build 4 rows of Q8K input with non-constant values per (row, block, position).
    let mut row_qs: [[i8; 3 * 256]; 4] = [[0; 768]; 4];
    let mut row_d:  [f32; 4 * 3] = [0.0; 12];
    let mut row_bsums: [[i16; 3 * 16]; 4] = [[0; 48]; 4];
    for row in 0..4 {
        for b in 0..n_blocks {
            row_d[row * n_blocks + b] = 0.01 + (row as f32) * 0.001 + (b as f32) * 0.0001;
            for i in 0..qk {
                row_qs[row][b * qk + i] = ((row * 7 + b * 11 + i) as i32 % 127 - 63) as i8;
            }
            for j in 0..16 {
                row_bsums[row][b * 16 + j] = ((row * 3 + b * 5 + j) as i16 % 31) - 15;
            }
        }
    }

    // Call q8k_repack_4 to produce n_blocks × block_q8_Kx4 output (1168 bytes each).
    let mut dst = vec![0u8; n_blocks * 1168];
    unsafe {
        olorin::kernels::ffi_inference::q8k_repack_4(
            row_qs[0].as_ptr(),
            row_qs[1].as_ptr(),
            row_qs[2].as_ptr(),
            row_qs[3].as_ptr(),
            row_d.as_ptr(),
            row_bsums[0].as_ptr(),
            row_bsums[1].as_ptr(),
            row_bsums[2].as_ptr(),
            row_bsums[3].as_ptr(),
            dst.as_mut_ptr(),
            n_blocks as i32,
        );
    }

    // For each super-block, verify:
    // - dst[bloff + 0..16] = f32s [row_d[0], row_d[1], row_d[2], row_d[3]]
    // - dst[bloff + 16..1040] = interleaved quants per llama.cpp layout
    //   (exact pattern pinned in Task 1's research note)
    // - dst[bloff + 1040..1168] = interleaved bsums per llama.cpp layout
    for b in 0..n_blocks {
        let bloff = b * 1168;
        // ... per-byte assertions against expected layout ...
    }
}
```

(The exact assertions are written in Task 4 of the plan, after Task 1's research note pins the interleaving pattern.)

### 5.7 Test 2: `tests/gemm_q4k_8x8.rs` (new) — **correctness gate**

```rust
#[test]
fn gemm_matches_matvec_loop_bitexact() {
    if !Path::new(&model_path()).exists() { eprintln!("SKIP: no model"); return; }
    let gguf  = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    // Layer 0 ffn_gate: Q4K, shape [ffn_dim=6144, hidden_dim=1536].
    let lw = &model.layers[0];
    assert_eq!(lw.w_gate_dtype, olorin::inference::matmul::GGML_TYPE_Q4_K);
    let n_rows = model.ffn_dim[0];
    let n_cols = model.hidden_dim;
    let n_blocks = n_cols / 256;
    let tile_bytes = n_blocks * 1152;
    let n_tiles = n_rows / 8;

    // Repack weight once (reuse Phase B.2's q4k_repack_8x8).
    let mut packed = vec![0u8; n_tiles * tile_bytes];
    unsafe {
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.w_gate, packed.as_mut_ptr(), n_rows as i32, n_cols as i32,
        );
    }

    let pow2 = olorin::inference::matmul::pow2_table();

    for &n in &[4usize, 8, 16, 32] {
        // Build n input columns of Q8K (non-trivial synthetic data).
        let (per_col_qs, per_col_d, per_col_bsums) = build_synthetic_q8k(n, n_cols);

        // Reference: run q4k_8x8_q8k_matvec n times.
        let mut ref_out = vec![0f32; n_rows * n];
        let mut scratch_ref = [0u8; 128];
        for k in 0..n {
            unsafe {
                olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
                    packed.as_ptr(),
                    per_col_qs[k].as_ptr(),
                    per_col_d[k].as_ptr(),
                    per_col_bsums[k].as_ptr(),
                    pow2.as_ptr(), scratch_ref.as_mut_ptr(),
                    ref_out[k * n_rows..].as_mut_ptr(),
                    n_rows as i32, n_cols as i32,
                );
            }
        }

        // Build block_q8_Kx4 a_ptrs input for the gemm via q8k_repack_4.
        // The exact packing loop depends on n / 4.
        let a_ptrs_buf = build_a_ptrs_for_gemm(n, n_blocks, &per_col_qs, &per_col_d, &per_col_bsums);

        // Candidate: one gemm call.
        //
        // Naming map (olorin matvec ↔ llama.cpp gemm):
        //   olorin `n_rows` (= ffn_dim = 6144)  ↔  llama.cpp `nc` = n_cols_b (B-cols)
        //   olorin `n_cols` (= hidden_dim)      ↔  llama.cpp `n`  = n_cols_inner (K)
        //   test's `n`     (= batch size)       ↔  llama.cpp `nr` = n_rows_a (A-rows)
        //
        // Output is row-major [n_rows_a × n_cols_b] with stride `bs = n_cols_b`.
        // For the layer-0 ffn_gate test that's stride 6144 floats per A-row,
        // so `gemm_out[a_row * n_rows + b_col]` addresses output (a_row, b_col).
        // Total size = n × n_rows floats (matches the ref_out allocation below).
        let mut gemm_out = vec![0f32; n_rows * n];
        let mut scratch = vec![0u8; 144];
        let mut acc_scratch = vec![0f32; 2 * n];
        unsafe {
            olorin::kernels::ffi_inference::q4k_8x8_q8k_gemm(
                packed.as_ptr(),
                a_ptrs_buf.as_ptr(),
                pow2.as_ptr(),
                scratch.as_mut_ptr(),
                acc_scratch.as_mut_ptr(),
                gemm_out.as_mut_ptr(),
                n_rows as i32,       // bs = n_cols_b (row stride in floats)
                n as i32,            // n_rows_a (= batch N, llama.cpp's nr)
                n_rows as i32,       // n_cols_b  (= olorin's n_rows, llama.cpp's nc)
                n_cols as i32,       // n_cols_inner (= olorin's n_cols, llama.cpp's n / K)
            );
        }

        // Per-output to_bits() equality. The reference (ref_out) is produced
        // by running q4k_8x8_q8k_matvec n times with per-column input, which
        // stores each column's n_rows scores contiguously at offset `k*n_rows`.
        // That's col-major in (col, row) space but numerically equivalent to
        // the gemm's row-major (a_row, b_col) storage because a_row == col
        // and b_col == row under the naming map above.
        for a_row in 0..n {
            for b_col in 0..n_rows {
                let ref_idx  = a_row * n_rows + b_col; // ref_out[col * n_rows + row]
                let gemm_idx = a_row * n_rows + b_col; // gemm row-major
                assert_eq!(
                    ref_out[ref_idx].to_bits(),
                    gemm_out[gemm_idx].to_bits(),
                    "N={n}, a_row={a_row}, b_col={b_col}"
                );
            }
        }
        eprintln!("PASS: N={n}");
    }
}
```

(Helper functions `build_synthetic_q8k` and `build_a_ptrs_for_gemm` are defined in the same test file; `build_a_ptrs_for_gemm` calls `q8k_repack_4` for each 4-row group.)

N sweep: `{4, 8, 16, 32}`. N=4 is the minimum llama.cpp allows. N=8 is the bench gate point. N=16 and 32 stress the outer y-loop iteration count and the weight reuse across larger batches. If N=4 fails, it's a baseline kernel bug. If only N > 4 fails, it's an outer-loop or accumulator-reset bug.

### 5.8 Bench rewrite: `tests/bench_q4k_gemm.rs` (Commit 8)

The file currently exists on the branch as dead code (doesn't compile because `ffi_inference::q4k_8x8_q8k_gemm` doesn't exist pre-Plan-1). Its input setup allocates per-col `q8_qs / q8_d / q8_bsums` arrays. Commit 8 rewrites the setup to build `block_q8_Kx4` `a_ptrs` buffers using the now-working `q8k_repack_4`, then calls the gemm with the new signature.

**Existing gate (preserved verbatim):**

```rust
if n == 8 {
    assert!(speedup >= 1.5,
        "N=8 gemm speedup {:.2}x is below the 1.5x acceptance gate", speedup);
}
```

The gate stays. The rewrite only changes how the inputs are built, not the measurement or the gate.

**Expected numbers (informed by llama.cpp's AVX2 characteristics):**
- N=1: cannot be benched (gemm requires N ≥ 4). Remove this row from the sweep.
- N=2: same — remove.
- N=4: the minimum legal case. Expected 1.2–1.4× vs. matvec loop — weight reuse wins but overhead dominates.
- N=8: **≥ 1.5× required**. Expected 1.8–2.5×.
- N=16: expected 2.5–3.2×.
- N=32: expected 3–4×.
- N=128: expected 3.5–4.5× (diminishing returns — weight tile no longer dominates, memory bandwidth catches up).

If the N=8 gate misses, diagnostic path is in §7.4.

## 6. Architectural context recap — Phase 2 delivery arc

Plan 1 ends with the gemm kernel proven in isolation. It is **not** on the forward_batch critical path yet; production `generate.rs` still loops `forward_one_graph` for prefill. The user-visible prefill win is Plan 3's delivery.

Plan 2 consumes Plan 1's `q4k_8x8_q8k_gemm` by:
1. Adding batched state buffers to `Gemma4State` (`batch_x`, `batch_q`, `batch_k`, `batch_v`, batched Q8K, etc.).
2. Writing 6 new Eä kernels for the per-token ops that need to go batched (`q8k_quant_batched`, `gemma4_rmsnorm_batched`, `gemma4_rope_batched`, `gelu_mul_batched`, `attn_qk_batched`, `attn_softmax_batched`, `attn_vmul_batched`).
3. Adding a multi-position `cache.store_batch` method to `inference/cache.rs` that respects sliding-window wrap.
4. Writing `forward_batch` on `Gemma4State` tying the above plus the Plan 1 gemm into a full batched forward pass.
5. Adding `tests/gemma4_batch_verify.rs` that asserts `forward_batch(tokens)` bit-matches `forward_one(tokens[i])` for each `i`.

**Plan 2 does not touch `generate.rs`.** Plan 3 wires `generate.rs` to call `forward_batch` in the prefill loop and verifies the numerical output matches `llama-eval-callback`.

Plan 1 → Plan 2 → Plan 3 is the hard dependency chain. Plan 2 cannot start until Plan 1's gemm is bit-exact-verified. Plan 3 cannot start until Plan 2's `forward_batch` is bit-exact-verified against `forward_one` loop.

## 7. Verification plan

### 7.1 Per-commit gates

Run before every `git commit`. Delta interpretation against the session baseline (currently 9 actual warnings + summary line, only chacha20 files over 500):

1. **Build clean** — `cargo build --release` — warning count must not grow beyond baseline.
2. **Line limit** — `find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'` — only chacha20 files allowed.
3. **Phase A smoke** — `tests/repack_q4k` (3 PASS).
4. **Phase B.2 smoke** — `tests/dual_q4k_8x8` (1 PASS, guards bit-exact dual matvec).
5. **Decode regression** — `tests/gemma4_parallel_regression` (1 PASS, guards Path A snapshot — Plan 1 does not touch the forward pass, should stay bit-exact).

### 7.2 Per-commit additional gates

| Commit | Additional gate |
|---|---|
| 1 research note | None — doc only. |
| 2 q8k_repack_4.ea x86 | build.rs produces `libq8k_repack_4.so`. No Rust-side test yet. |
| 3 q8k_repack_4_arm.ea | `ea --emit-asm --target-triple=aarch64-unknown-linux-gnu --target=cortex-a76 --dotprod` emits valid aarch64 asm for the new kernel. Cross-compile only. |
| 4 FFI + q8k_repack_4 test | `cargo test --release --test q8k_repack_4` PASS — byte-exact layout vs. expected `block_q8_Kx4`. |
| 5 x86 gemm kernel | build.rs produces `libq4k_dot_8x8_gemm.so`. Dead .so until Commit 7. |
| 6 ARM gemm kernel | `ea --emit-asm --target-triple=aarch64-unknown-linux-gnu --target=cortex-a76 --dotprod` emits valid asm. Cross-compile only. |
| 7 FFI + gemm test | **`cargo test --release --test gemm_q4k_8x8`** PASS across N ∈ {4, 8, 16, 32} with per-output `to_bits()` equality vs. matvec loop. **Correctness gate for the whole plan.** |
| 8 bench rewrite + perf gate | `cargo test --release --test bench_q4k_gemm -- --nocapture --test-threads=1` PASS and **N=8 speedup ≥ 1.5×**. Commit message records the full speedup table (N=4, 8, 16, 32, 128). |
| 9 wrap-up | None — optional commit. |

### 7.3 Full regression sweep (before considering Plan 1 done)

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release \
  --test repack_q4k \
  --test dual_q4k_8x8 \
  --test q8k_repack_4 \
  --test gemm_q4k_8x8 \
  --test gemma4_verify \
  --test gemma4_parallel_regression \
  --test gemma4_smoke \
  --test bench_q4k_gemm \
  -- --test-threads=1 2>&1 | tail -60
```

All 8 suites green.

### 7.4 If the perf gate misses at Commit 8

**Stop.** Do not commit. Investigate before retrying. The gate-miss diagnostic path:

1. **Profile with `perf stat`** — `perf stat -e L1-dcache-loads,L1-dcache-load-misses,cycles,instructions` on a single-thread gemm vs. a matvec-loop run, at N=8. Compare the dcache load count per FLOP. If gemm dcache loads > matvec dcache loads × 0.6 (i.e., weight reuse isn't saving 40%), the fusion isn't working.
2. **Inspect generated assembly** — `ea --emit-asm` on the gemm kernel, look for register spills of `rhs_mat_*` or `acc_rows[*]` that shouldn't be happening. The compiler is allowed to spill the 32 accumulators but shouldn't spill the shared weight state.
3. **Try a different scratch layout** — if `acc_scratch` sizing is constraining the compiler, grow it temporarily and re-bench.
4. **Reshape the tile** — drop to 8 rows × 8 cols if 16 × 8 is register-bound. Loses some weight reuse but might unblock perf.

None of these are planned tasks; they're diagnostic steps. If they all fail, Plan 1 stops and I surface to the user rather than hack the gate down.

### 7.5 Rollback plan

Each of Plan 1's 9 commits is independently revertible:
- Commits 2, 3, 5, 6 add new `.ea` files. Revert = `git rm`.
- Commit 4 adds FFI types + loader + one test. Revert = Edit undo.
- Commit 7 adds gemm FFI + one test. Same shape.
- Commit 8 rewrites `bench_q4k_gemm.rs` (already on branch as dead code). Revert = `git checkout HEAD~ tests/bench_q4k_gemm.rs`.
- Commit 9 is memory + maybe a README line. No functional change.

No commit touches `generate.rs`, `forward_*.rs`, `matmul*.rs`, or `engine*.rs`. The forward pass is side-effect-free across the whole plan. Worst case, Plan 1 fails and we revert 9 commits with zero downstream cleanup.

## 8. Rollback plan (repeated for emphasis)

If Plan 1 lands and something downstream surfaces (e.g., a Plan 2 subagent discovers the gemm has a subtle bug the Plan 1 test missed), rollback is:

```bash
git revert <commit9>..<commit1>
```

The revert is clean because nothing between the 9 commits touches Phase B.2's landed code. Phase B.2 (commits up through `98b6b1e`) is the rollback floor.

## 9. Handoff to writing-plans

This spec is the input to `superpowers:writing-plans`. The plan file produced from it should live at `docs/superpowers/plans/2026-04-11-phase-2-plan-1-q4k-8x8-gemm.md` and expand each of the 9 commits in §4 into step-level tasks with checkboxes, matching the format of `docs/superpowers/plans/2026-04-11-q4k-repack-phase-b2.md` from Phase B.2.
