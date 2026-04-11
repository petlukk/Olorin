# Phase 2 — Plan 1: Q4K 8×8 × Q8K batched gemm kernel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **HARD RULES (apply to ALL agents):**
> - No file exceeds 500 lines. Split before you hit the limit.
> - Every feature proven by end-to-end test. If it's not tested, it doesn't exist.
> - No fake functions. No silent fallbacks. No `// TODO`, `// HACK`, `// for now`.
> - Olorin is Eä's showcase — every SIMD op must be an Eä kernel. **Do NOT simplify kernel code to scalar Rust.**
> - Match llama.cpp **bit-exact** (per-output `to_bits()` where the recipe allows).
> - **x86 kernels target AVX2 ONLY. Never use `__m512` / AVX-512 / `_mm512_*` / `vpermi2d` / `vcompress` / `_kmask*` / `evex` intrinsics.** An earlier AVX-512 attempt on olorin was a trap that had to be deleted wholesale. When porting from llama.cpp, reference the AVX2 fallback branch, not the `__AVX512BW__ && __AVX512DQ__` guard block.
> - **ARM kernels target Cortex-A76 (NEON + dotprod, NO i8mm).** Both llama.cpp ARM gemm paths require `__ARM_FEATURE_MATMUL_INT8` which Cortex-A76 lacks — llama.cpp falls through to scalar on Pi 5. This means the ARM gemm kernel has **no llama.cpp reference to copy**; derive it from olorin's own `kernels/q4k_dot_8x8_arm.ea` matvec instead.
> - eacompute compiler: `$HOME/projects/eacompute/target/release/ea`
> - Build: `PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release`
> - Branch: `gemma4-batched-prompt-eval`
> - **eabrain protocol** (mandatory):
>   - Start of every task: `eabrain status` and `eabrain recall` (previous findings may answer questions before you start grepping).
>   - Before searching for any Eä symbol by name: `eabrain search <name>`.
>   - Before assuming an Eä intrinsic doesn't exist: `eabrain ref <name>` AND grep `$HOME/projects/eacompute/src/typeck/intrinsics*.rs` + `$HOME/projects/eacompute/src/codegen/simd*.rs` directly. **eabrain does not index eacompute's Rust intrinsic definitions** — if `eabrain ref` returns nothing, the intrinsic may still exist.
>   - After editing any `.ea` kernel: `eabrain index`.
>   - End of any task producing a non-obvious finding: `eabrain remember "..."`.

**Goal:** Land a bit-exact Q4K 8×8 × Q8K batched gemm Eä kernel that processes N input columns per call against a repacked weight tile, sharing weight unpack + scale decode across all N columns. Deliver x86 AVX2 and ARM NEON+dotprod implementations, plus a new `q8k_repack_4` A-side helper kernel that packs 4 rows of regular Q8K into llama.cpp's `block_q8_Kx4` interleaved format.

**Architecture:** Kernel-first TDD. Commit 1 is a research note pinning the `block_q8_Kx4` byte layout and the x86 AVX2 kernel's helper-func decomposition. Commits 2-4 land the `q8k_repack_4` helper kernel with a standalone byte-layout test. Commits 5-6 write the gemm kernels (x86 first, then ARM derived from olorin's own matvec). Commit 7 ships the FFI binding + bit-exact correctness gate (`tests/gemm_q4k_8x8.rs`). Commit 8 rewrites `tests/bench_q4k_gemm.rs` to use the new input layout and verifies the ≥1.5× speedup gate at N=8.

**Tech Stack:** Rust, Eä (eacompute), x86 AVX2 + ARM NEON+dotprod, `libloading`, existing `q4k_dot_8x8` / `q4k_repack_8x8` infrastructure from Phase A+B.2. Reference: llama.cpp AVX2 gemm at `~/projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp:2816-3487`.

**Spec:** `docs/superpowers/specs/2026-04-11-phase-2-plan-1-q4k-8x8-gemm-design.md` (committed as `6a43ec1`).

**Baseline (for delta gate interpretation):**
- `cargo build --release 2>&1 | grep -c ^warning` = **10** (9 actual warnings + the "generated 9 warnings" summary line).
- `find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'` = `{chacha20_search_v2.ea: 750, chacha20_search_v2_arm.ea: 609}`.
- All 7 test suites (`repack_q4k`, `dual_q4k_8x8`, `gemma4_verify`, `gemma4_parallel_regression`, `gemma4_smoke`, + Plan 1's new `q8k_repack_4` and `gemm_q4k_8x8` once they exist, + `bench_q4k_gemm` once it compiles) must be green at end of every commit.

---

## Per-Task Verification Gates

Run these before every `git commit`. Delta interpretation — gates fail only if the delta vs. baseline grows.

**Gate 1 — Build clean (delta).**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
N=$(grep -c "^warning" /tmp/olorin-build.log)
echo "warnings: $N (baseline 10)"
test "$N" -le 10 || { echo "FAIL: warning count grew"; exit 1; }
```

**Gate 2 — Line limit (delta).**

```bash
find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l \
  | awk '$1 > 500 && $2 != "total" {print}'
```

Expected: only `kernels/chacha20_search_v2.ea` (750) and `kernels/chacha20_search_v2_arm.ea` (609). Any new file over 500 = fail.

**Gate 3 — Phase A smoke.**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test repack_q4k -- --test-threads=1 2>&1 | tail -6
```

Expected: 3 PASS.

**Gate 4 — Phase B.2 smoke (bit-exact dual matvec).**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test dual_q4k_8x8 -- --test-threads=1 2>&1 | tail -6
```

Expected: 1 PASS.

**Gate 5 — Decode regression (Path A snapshot).**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression 2>&1 | tail -6
```

Expected: 1 PASS. This plan touches nothing in the forward pass, so this gate should never move.

When a task says "run all gates," it means run Gates 1–5 in order, plus any per-task additional gate.

---

## Task 1: Research note — lock `block_q8_Kx4` layout and x86 helper-func decomposition

**Goal:** Read llama.cpp's AVX2 gemm body, confirm the exact byte layout of `block_q8_Kx4`, and document the helper-func decomposition that Task 5's x86 kernel author will follow. Zero code changes in this task — doc only.

**Files:**
- Create: `docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md`
- Read (no edit): `$HOME/projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` — the gemm function at line 2042, specifically the AVX2 `#else` fallback branch starting around line 2816 and ending around 3487.
- Read (no edit): `$HOME/projects/llama.cpp/ggml/src/ggml-cpu/repack.h:96-100` — the `block_q8_Kx4` struct definition.
- Read (no edit): `kernels/q4k_dot_8x8.ea` (olorin's existing AVX2 matvec, 228 lines, structural precedent).
- Read (no edit): `kernels/q4k_dot_8x8_arm.ea` (olorin's existing ARM matvec, post-B.1-fix, structural template for Task 6).

- [ ] **Step 1: eabrain baseline**

```bash
eabrain status
eabrain search q4k_8x8_q8k_matvec
eabrain ref concat_i8x16
eabrain ref shuffle_bytes
eabrain ref maddubs_i16
eabrain ref madd_i16
```

Expected: matches for all existing names. If any intrinsic is missing from eabrain, grep `$HOME/projects/eacompute/src/typeck/intrinsics*.rs` to confirm existence.

- [ ] **Step 2: Confirm `block_q8_Kx4` struct layout**

Read `$HOME/projects/llama.cpp/ggml/src/ggml-cpu/repack.h` at lines 96–100:

```c
struct block_q8_Kx4 {
    float d[4];              // delta, one per packed row
    int8_t qs[QK_K * 4];     // QK_K = 256, so 1024 bytes of quants
    int16_t bsums[QK_K / 4]; // 64 i16 bsums total
};
```

Size check: `4 × 4 + 256 × 4 + 64 × 2 = 16 + 1024 + 128 = 1168 bytes`. Record this number in the research note — the `q8k_repack_4` kernel's output stride is 1168 bytes per super-block.

- [ ] **Step 3: Confirm the `qs[1024]` internal byte order**

The critical question for Task 2's repack kernel: is `block_q8_Kx4.qs[1024]` stored as **row-major** (row 0's 256 bytes, then row 1, then row 2, then row 3) or **interleaved** (chunks of some size cycling through rows)?

Read `$HOME/projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` around line 788 (the AVX-512 path) or line 2820+ (the AVX2 path we care about). Look for how `a_ptrs[rp][b].qs` is accessed:

```bash
grep -n "a_ptrs\[rp\]\[b\]\.qs" ~/projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp | head -20
```

If the accesses look like `a_ptrs[rp][b].qs + 0`, `a_ptrs[rp][b].qs + 32`, `a_ptrs[rp][b].qs + 64`, `a_ptrs[rp][b].qs + 96` — that is 4 × 32-byte loads starting at `qs`, covering 128 bytes. For a 4-row interleaved layout, one reasonable pattern is: 8 bytes from row 0, 8 bytes from row 1, 8 bytes from row 2, 8 bytes from row 3, repeat. Verify by reading the subsequent `permute2f128_si256` calls — if they split the loaded `__m256i` into `row01` / `row23` halves via `permute2f128(x, x, 0)` / `permute2f128(x, x, 17)`, that confirms the interleave granularity is 16 bytes (two rows per `__m128i` half).

**Document your findings in the research note.** Sketch:

```
block_q8_Kx4.qs[1024] byte layout (as read from repack.cpp:2820+):

For each super-block, the 1024 qs bytes are laid out as:
  Offset    Content
  0..15     row 0 quants bytes 0..7  + row 1 quants bytes 0..7   (interleaved 8+8)
  16..31    row 2 quants bytes 0..7  + row 3 quants bytes 0..7
  32..47    row 0 quants bytes 8..15 + row 1 quants bytes 8..15
  ...

OR (alternative, confirm by reading code):

  0..255    row 0 quants (contiguous)
  256..511  row 1 quants
  512..767  row 2 quants
  768..1023 row 3 quants
```

Write the version you confirmed. This pins the `q8k_repack_4` implementation in Task 2.

- [ ] **Step 4: Confirm the `bsums[64]` internal i16 order**

Same question for bsums. Find how `a_ptrs[rp][b].bsums` is accessed in the gemm body. The bsums array is `QK_K / 4 = 64` i16 values; for 4 rows × 16 bsums per row, verify whether the 16-per-row groups are contiguous or interleaved. Document in the research note.

- [ ] **Step 5: Map llama.cpp's AVX2 gemm body to helper-func decomposition**

Read `$HOME/projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` from line 2816 (start of AVX2 fallback) to ~3487. The body is dense — focus on identifying 3 logical sections:

**Section A — Per-super-block weight unpack (shared across all rp iterations):**

The `b` loop at around line 2844 loads 8 packed weight vectors (`rhs_raw_mat_0123_0` through `rhs_raw_mat_4567_3`), applies `permutevar8x32_epi32 + blend_epi32` to reshape into `rhs_raw_mat_0145_*` / `rhs_raw_mat_2367_*`, and does nibble extract into `rhs_mat_0145_00..03` (low) and `rhs_mat_0145_10..13` (high). This block runs ONCE per (tile, super-block). It is pure function of `b_ptr[b].qs`. **Candidate for an Eä helper func** taking `packed` and `b` as arguments, returning the nibble-extracted output.

**Section B — Per-super-block scale decode (shared across all rp iterations):**

Decodes the 12-byte `scales` array from the `block_q4_Kx8` header into `scale_0*`, `scale_1*`, `mins_01*` i16x16 vectors via the `utmp_*` dance. Runs ONCE per (tile, super-block). **Candidate for an Eä helper func.**

**Section C — Per-(super-block, rp) A-side load + dot + FMA:**

The `for (int rp = 0; rp < 4; rp++)` loop at around line 2942 loads 4 `block_q8_Kx4` quants via `_mm256_loadu_si256 + permute2f128_si256 + inserti32x8`, runs 16 `maddubs_epi16` calls, combines with `add_epi16 + madd_epi16(scale_*)`, sums to i32 via `add_epi32`, converts to f32 via `cvtepi32_ps`, and FMAs into 4 row accumulators. Runs 4 times per (tile, super-block) — once per rp. **Each rp iteration is another helper func candidate** — it's ~80 lines of AVX2 in llama.cpp, translates to ~120 lines of Eä.

**In the research note**, document the decomposition as:

```
q4k_dot_8x8_gemm.ea structure:

func unpack_weight_b(packed, b) -> (nibble_vecs[16])
  // Section A, ~90 lines of Eä

func decode_scales_b(packed, b) -> (scale_0x, scale_1x, mins_01x)
  // Section B, ~60 lines of Eä

func acc_rp_b(rp_q8_ptr, b, nibble_vecs, scale_*, mins_01,
              acc_rows_rp0, acc_rows_rp1, acc_rows_rp2, acc_rows_rp3,
              acc_min_rows_rp0, ..., col_d, col_dmin)
  // Section C, ~120 lines of Eä, called 4× per super-block

export func q4k_8x8_q8k_gemm(...)
  // Outer y/x/b loop skeleton, calls the helpers, ~150 lines of Eä

Total: ~420 lines, comfortably under the 500-line limit.
```

- [ ] **Step 6: Sketch the ARM NEON+dotprod gemm derivation**

The ARM gemm has no llama.cpp reference. Document the derivation path:

```
ARM gemm (q4k_dot_8x8_gemm_arm.ea) is derived from olorin's existing
kernels/q4k_dot_8x8_arm.ea (matvec) by extending the per-row accumulators
(acc0: f32x4 for rows 0..3, acc1: f32x4 for rows 4..7) into per-(row, col-chunk)
accumulators:

  acc_row[i][cc]  for i in 0..8, cc = col-chunk in 0..(N/4)

where each acc_row[i][cc] is f32x4 (4 col lanes).

The existing matvec's per-sb body processes one input column's q8 data via
vdot_i32 into alo0..alo3 + ahi0..ahi3. The gemm extends this by:

  1. Loading the per-sb weight tile ONCE (shared across all col chunks).
  2. Looping over col chunks: for each cc in 0..(N/4):
     a. Load 4 columns' q8 data from the block_q8_Kx4 struct.
     b. Run the 8 vdot_i32 calls with those 4 Q8 inputs.
     c. Accumulate into acc_row[0..7][cc] via hadd_i32 + to_f32 + fma.
     d. Bias accumulation for the cc chunk.
  3. End-of-super-block: subtract bias[cc] * sb_mn from acc_row[*][cc] for each cc.

N=4 degenerates to one cc=0 iteration (matches matvec). N=8 → 2 chunks.
N=16 → 4 chunks. N=32 → 8 chunks. The chunk loop amortizes weight unpack
across the N input columns.

NEON has 32 vector registers; the cc loop body uses:
  8 acc_row[i][cc] (per cc) — f32x4
  8 bias[i][cc]    (per cc) — i32x4
  4 q8 register tiles       — i8x16
  8 vdot_i32 results        — i32x4
  + weight unpack state     — ~10 vectors

For a single cc iteration that's ~38 live vectors, which fits NEON's 32
after compiler spill. Spill pressure is manageable.

Line estimate: ~460 lines (existing matvec is 263, gemm extension adds
~200 for the cc loop body and outer y-loop).
```

- [ ] **Step 7: Write the research note file**

Consolidate Steps 2–6 into `docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md`. Target length: ~300-400 lines, roughly the density of `docs/superpowers/research/2026-04-08-ggml-q4k-8x8-q8k-gemm.md` from Phase A.

Include:
- Source citations (llama.cpp line numbers with the caveat "at the time of writing").
- `block_q8_Kx4` byte layout confirmed in Step 3.
- Bsums byte layout confirmed in Step 4.
- x86 helper-func decomposition from Step 5.
- ARM derivation sketch from Step 6.
- A one-paragraph "what Task 5 writes first" and "what Task 6 writes first" checklist.

- [ ] **Step 8: Run all gates**

Gates 1–5. All pass. This task only adds a doc file; no code changes.

- [ ] **Step 9: Commit**

```bash
git add docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md
git commit -m "research(phase-2): q4k_8x8_q8k_gemm Ea kernel template

Pins block_q8_Kx4 byte layout (qs interleaving + bsums interleaving)
confirmed by reading llama.cpp's AVX2 gemm body at arch/x86/repack.cpp:
2816-3487 and repack.h:96-100.

Documents the x86 kernel's helper-func decomposition: unpack_weight_b
and decode_scales_b as per-super-block helpers (shared across all rp
iterations), acc_rp_b as a per-rp helper called 4 times per super-block.
Expected Eä line count ~420 for the full file, under the 500-line
limit without cross-file splits.

Documents the ARM NEON+dotprod gemm derivation from olorin's own
kernels/q4k_dot_8x8_arm.ea: extend per-row acc0/acc1 f32x4 pairs into
per-(row, col-chunk) accumulators, loop over col chunks to amortize
weight unpack across N input columns. No llama.cpp reference for ARM
(both llama.cpp ARM gemm paths require i8mm which Cortex-A76 lacks;
llama.cpp falls through to scalar on Pi 5).

Input to Task 2 (q8k_repack_4) and Task 5/6 (x86 / ARM gemm kernels)."
```

---

## Task 2: `q8k_repack_4.ea` x86 kernel — pack 4 Q8K rows into `block_q8_Kx4`

**Goal:** Write an Eä kernel that takes 4 per-row Q8K inputs (`qs`, `d`, `bsums`) and writes a contiguous `block_q8_Kx4` output buffer in the exact byte layout confirmed in Task 1's research note. x86 host build only — ARM mirror is Task 3.

**Files:**
- Create: `kernels/q8k_repack_4.ea`
- Read (no edit): `docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md` (Task 1 output — **required to know the target byte layout**)
- Read (no edit): `kernels/q4k_repack.ea` (existing weight repack, structural reference for an Eä kernel that does shuffle + store with zero arithmetic)

- [ ] **Step 1: eabrain lookup**

```bash
eabrain ref store
eabrain ref load
eabrain ref shuffle_bytes
eabrain ref ptr_as_i8
```

Verify the load/store/cast intrinsics for the kernel.

- [ ] **Step 2: Create the new kernel file with the signature**

Create `kernels/q8k_repack_4.ea` with:

```ea
// q8k_repack_4.ea — Pack 4 rows of Q8K into llama.cpp's block_q8_Kx4 layout
// (x86 AVX2 SIMD).
//
// Given 4 parallel Q8K streams (row0..row3 qs, d, bsums), write a
// contiguous block_q8_Kx4 output buffer with the interleave pattern
// documented in docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md.
//
// Output stride: 1168 bytes per super-block. Total output: nb * 1168 bytes.
// Pure shuffle + store; no arithmetic.

#[cfg(x86_64)]

export func q8k_repack_4(
    row0_qs:    *restrict i8,
    row1_qs:    *restrict i8,
    row2_qs:    *restrict i8,
    row3_qs:    *restrict i8,
    row_d:      *restrict f32,
    row0_bsums: *restrict i16,
    row1_bsums: *restrict i16,
    row2_bsums: *restrict i16,
    row3_bsums: *restrict i16,
    dst:        *mut u8,
    nb:         i32
) {
    // Loop body written in Step 3.
}
```

- [ ] **Step 3: Write the loop body for one super-block**

Add this loop body inside the export func. **The exact interleave pattern below is a placeholder — replace it with the one from Task 1's research note.** If Task 1's research confirmed "row-major: row0 contiguous, then row1, row2, row3," then this body stores 256 bytes per row in order. If it confirmed "interleaved 8-byte groups," the body cycles.

Example body for the **row-major** case (the simpler one — confirm first):

```ea
    let mut b: i32 = 0
    while b < nb {
        // dst[0..16] = four row deltas as f32 — 16 bytes
        let dst_f32: *mut f32 = ptr_as_f32(ptr_as_i8(dst))
        let dst_f32_off: i32 = b * 1168 / 4
        dst_f32[dst_f32_off + 0] = row_d[b * 4 + 0]
        dst_f32[dst_f32_off + 1] = row_d[b * 4 + 1]
        dst_f32[dst_f32_off + 2] = row_d[b * 4 + 2]
        dst_f32[dst_f32_off + 3] = row_d[b * 4 + 3]

        // dst[16..1040] = 4 rows × 256 quants = 1024 bytes (row-major)
        // Copy via 32-byte SIMD loads/stores, 8 iterations × 32 bytes = 256 bytes per row.
        let mut i: i32 = 0
        while i < 8 {
            let off_qs: i32 = b * 256 + i * 32
            let dst_off: i32 = b * 1168 + 16 + i * 32

            let v0: i8x32 = load(row0_qs, off_qs)
            store(dst, dst_off + 0 * 256, v0)
            let v1: i8x32 = load(row1_qs, off_qs)
            store(dst, dst_off + 1 * 256, v1)
            let v2: i8x32 = load(row2_qs, off_qs)
            store(dst, dst_off + 2 * 256, v2)
            let v3: i8x32 = load(row3_qs, off_qs)
            store(dst, dst_off + 3 * 256, v3)

            i = i + 1
        }

        // dst[1040..1168] = 4 rows × 16 bsums × 2 bytes = 128 bytes (row-major)
        // Copy via 16-byte SIMD loads/stores.
        let dst_i16: *mut i16 = ptr_as_i16(ptr_as_i8(dst))
        let dst_i16_off: i32 = b * 1168 / 2 + 520 // 1040 bytes / 2 = 520 i16 offset
        let br0: i16x8 = load(row0_bsums, b * 16)
        let br1: i16x8 = load(row0_bsums, b * 16 + 8)
        store(dst_i16, dst_i16_off + 0, br0)
        store(dst_i16, dst_i16_off + 8, br1)
        let br2: i16x8 = load(row1_bsums, b * 16)
        let br3: i16x8 = load(row1_bsums, b * 16 + 8)
        store(dst_i16, dst_i16_off + 16, br2)
        store(dst_i16, dst_i16_off + 24, br3)
        let br4: i16x8 = load(row2_bsums, b * 16)
        let br5: i16x8 = load(row2_bsums, b * 16 + 8)
        store(dst_i16, dst_i16_off + 32, br4)
        store(dst_i16, dst_i16_off + 40, br5)
        let br6: i16x8 = load(row3_bsums, b * 16)
        let br7: i16x8 = load(row3_bsums, b * 16 + 8)
        store(dst_i16, dst_i16_off + 48, br6)
        store(dst_i16, dst_i16_off + 56, br7)

        b = b + 1
    }
```

**If Task 1 confirmed the interleave pattern instead of row-major**, rewrite the two SIMD loops above to match. The interleaved pattern typically stores 8 bytes from row 0, 8 bytes from row 1, etc., which you'd express as smaller loads + `concat_i8x16` stores.

- [ ] **Step 4: Compile via build.rs**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -20
```

Expected: zero errors, no new warnings. The `build.rs` auto-discovers the new `.ea` file and produces `libq8k_repack_4.so` in `target/release/build/olorin-*/out/`.

```bash
find target/release/build -name "libq8k_repack_4.so" 2>&1
```

Expected: one path reported.

- [ ] **Step 5: Line limit check**

```bash
wc -l kernels/q8k_repack_4.ea
```

Expected: ~80–120 lines, well under 500.

- [ ] **Step 6: Run all gates**

Gates 1–5. All pass. The new kernel is compiled as a dead `.so` until Task 4 adds the FFI wrapper; there should be no new warnings and no tests change.

- [ ] **Step 7: eabrain index**

```bash
eabrain index
eabrain search q8k_repack_4
```

Expected: new symbol appears.

- [ ] **Step 8: Commit**

```bash
git add kernels/q8k_repack_4.ea
git commit -m "feat(phase-2): x86 q8k_repack_4 kernel — pack 4 rows into block_q8_Kx4

Pure shuffle + store kernel, zero arithmetic. Takes 4 per-row Q8K streams
(qs/d/bsums) and writes a contiguous block_q8_Kx4 output buffer per
llama.cpp's layout (sizeof(block_q8_Kx4) = 1168 bytes per super-block).

Output byte layout confirmed against
repack.h:96-100 and arch/x86/repack.cpp A-side loads; documented in
docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md.

Will feed Plan 1's gemm kernel as the A-side input format in Task 7.
Dead .so until FFI wrapper lands in Task 4."
```

---

## Task 3: `q8k_repack_4_arm.ea` — ARM NEON mirror

**Goal:** Write the ARM NEON version of `q8k_repack_4`. Since the repack is pure shuffle + store with no arithmetic, the x86 and ARM versions differ only in intrinsic names (u8x32 → i8x16 pairs on NEON, different load/store types). Cross-compile verification on the x86 host via `ea --emit-asm`.

**Files:**
- Create: `kernels/q8k_repack_4_arm.ea`
- Read (no edit): `kernels/q8k_repack_4.ea` (just written in Task 2)
- Read (no edit): `kernels/q4k_repack_arm.ea` (existing ARM repack kernel, for NEON load/store precedent)

- [ ] **Step 1: Copy the x86 baseline**

```bash
cp kernels/q8k_repack_4.ea kernels/q8k_repack_4_arm.ea
```

- [ ] **Step 2: Switch the cfg guard + adapt SIMD types**

Replace the header:

```ea
// q8k_repack_4_arm.ea — ARM NEON mirror of q8k_repack_4.ea.
//
// Same function, same byte layout, NEON load/store instead of AVX2.
// Since this kernel is pure shuffle + store with no arithmetic, the ARM
// version differs from the x86 only in SIMD widths (i8x16 instead of
// i8x32) and slightly more iterations in the copy loop.

#[cfg(aarch64)]

export func q8k_repack_4(
```

Change the body's i8x32 loads to **pairs of i8x16 loads** (NEON vectors are 128-bit). For the quant-copy loop:

```ea
        // dst[16..1040] = 4 rows × 256 quants = 1024 bytes (row-major)
        // Copy via 16-byte NEON loads/stores, 16 iterations × 16 bytes = 256 bytes per row.
        let mut i: i32 = 0
        while i < 16 {
            let off_qs: i32 = b * 256 + i * 16
            let dst_off: i32 = b * 1168 + 16 + i * 16

            let v0: i8x16 = load(row0_qs, off_qs)
            store(dst, dst_off + 0 * 256, v0)
            let v1: i8x16 = load(row1_qs, off_qs)
            store(dst, dst_off + 1 * 256, v1)
            let v2: i8x16 = load(row2_qs, off_qs)
            store(dst, dst_off + 2 * 256, v2)
            let v3: i8x16 = load(row3_qs, off_qs)
            store(dst, dst_off + 3 * 256, v3)

            i = i + 1
        }
```

Bsums loop stays the same (i16x8 is native on both arches).

- [ ] **Step 3: Cross-compile via direct `ea` invocation**

```bash
$HOME/projects/eacompute/target/release/ea \
    kernels/q8k_repack_4_arm.ea \
    --emit-asm --opt-level=3 \
    --target-triple=aarch64-unknown-linux-gnu \
    --target=cortex-a76 --dotprod 2>&1 | tail -20
```

Expected: `wrote q8k_repack_4_arm.s` message. Check:

```bash
ls -la q8k_repack_4_arm.s
```

Expected: non-zero file size. Clean up the scratch file:

```bash
rm -f q8k_repack_4_arm.s
```

**If cross-compile fails:** the ARM version has a type-inference issue. The most common cause is `let v: i8x16 = load(row0_qs, off_qs)` being ambiguous — eacompute's load intrinsic is type-directed, but the `row0_qs: *restrict i8` parameter type should make the load type unambiguous. If ea complains, add an explicit cast helper: `let row0_qs_i8: *restrict i8 = ptr_as_i8(row0_qs)` and load from `row0_qs_i8`. See `bc1fd87` (Phase B.1 ARM fix) for precedent.

- [ ] **Step 4: Line limit check**

```bash
wc -l kernels/q8k_repack_4_arm.ea
```

Expected: ~100–140 lines.

- [ ] **Step 5: Run all gates (x86 build)**

Gates 1–5 on x86 host. The ARM file is filtered out by `build.rs` on x86 targets, so the warning/test state should be identical to Task 2's end state.

- [ ] **Step 6: eabrain index**

```bash
eabrain index
```

- [ ] **Step 7: Commit**

```bash
git add kernels/q8k_repack_4_arm.ea
git commit -m "feat(phase-2): ARM NEON q8k_repack_4 kernel

Mirror of kernels/q8k_repack_4.ea adapted to NEON load/store widths
(i8x16 instead of i8x32, 16 iterations per row instead of 8). Same
function, same output byte layout, zero arithmetic.

Cross-compile verified on this x86 workstation via direct ea
invocation:
  ea kernels/q8k_repack_4_arm.ea --emit-asm
     --target-triple=aarch64-unknown-linux-gnu
     --target=cortex-a76 --dotprod

Final .so link requires aarch64-linux-gnu-gcc which is not installed
on this workstation (per B.2 precedent). Runtime validation on Pi 5
deferred to Plan 3 or a later follow-up.

Not yet wired into Rust. FFI binding + test lands in Task 4."
```

---

## Task 4: FFI binding + byte-layout test for `q8k_repack_4`

**Goal:** Wire `q8k_repack_4` through `ffi_inference.rs` and prove it produces the correct `block_q8_Kx4` layout byte-for-byte via a standalone Rust test.

**Files:**
- Modify: `src/kernels/ffi_inference_types.rs` (+1 type)
- Modify: `src/kernels/ffi_inference.rs` (+1 field, +1 library load, +1 public wrapper)
- Create: `tests/q8k_repack_4.rs`

- [ ] **Step 1: Add the FFI type**

In `src/kernels/ffi_inference_types.rs`, after the existing `Q4k8x8MatvecDualFn` type (the last one added in Phase B.2), add:

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
```

- [ ] **Step 2: Wire into `KernelTableInference`**

In `src/kernels/ffi_inference.rs`, inside the `KernelTableInference` struct (currently around lines 8–37), add a new field right after `q4k_8x8_q8k_matvec_dual`:

```rust
    pub q4k_8x8_q8k_matvec_dual: Q4k8x8MatvecDualFn,
    pub q8k_repack_4:            Q8kRepack4Fn,
}
```

- [ ] **Step 3: Add the library load**

Find the `load_inference_kernels` body. After the existing `let q4k_dot_8x8_dual_lib = load("q4k_dot_8x8_dual")?;` line, add:

```rust
    let q4k_dot_8x8_dual_lib = load("q4k_dot_8x8_dual")?;
    let q8k_repack_4_lib     = load("q8k_repack_4")?;
```

- [ ] **Step 4: Add the symbol transmute and update libs vec**

In the `KernelTableInference { ... }` struct literal, after the `q4k_8x8_q8k_matvec_dual` transmute line, add:

```rust
            q4k_8x8_q8k_matvec_dual: std::mem::transmute(sym(&q4k_dot_8x8_dual_lib, b"q4k_8x8_q8k_matvec_dual\0")?),
            q8k_repack_4:            std::mem::transmute(sym(&q8k_repack_4_lib,     b"q8k_repack_4\0")?),
```

And add `q8k_repack_4_lib` to the `libs: vec![...]` line at the end of the struct literal.

- [ ] **Step 5: Add the public wrapper**

At the end of `src/kernels/ffi_inference.rs` (after the last `pub unsafe fn`), add:

```rust
#[allow(clippy::too_many_arguments)]
pub unsafe fn q8k_repack_4(
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
) {
    (k().q8k_repack_4)(
        row0_qs, row1_qs, row2_qs, row3_qs,
        row_d,
        row0_bsums, row1_bsums, row2_bsums, row3_bsums,
        dst, nb,
    )
}
```

- [ ] **Step 6: Write the byte-layout test**

Create `tests/q8k_repack_4.rs`:

```rust
//! Byte-layout test for q8k_repack_4.
//!
//! Builds 4 rows of synthetic Q8K input with non-constant values at every
//! (row, block, position) combination, runs the kernel, and checks every
//! output byte against the expected block_q8_Kx4 layout from llama.cpp's
//! repack.h:96-100 as confirmed in
//! docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md.

#[test]
fn q8k_repack_4_matches_block_q8_Kx4_layout() {
    olorin::kernels::ffi::init().unwrap();

    // 3 super-blocks to catch stride bugs.
    let n_blocks = 3;
    let qk = 256;

    // Build 4 rows of Q8K input. Use distinct non-constant patterns per
    // (row, block, position) so any mis-copy surfaces as a byte mismatch.
    let mut row_qs: [Vec<i8>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut row_d: Vec<f32> = Vec::new();
    let mut row_bsums: [Vec<i16>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    for row in 0..4 {
        row_qs[row] = vec![0i8; n_blocks * qk];
        row_bsums[row] = vec![0i16; n_blocks * 16];
        for b in 0..n_blocks {
            for i in 0..qk {
                row_qs[row][b * qk + i] = (((row * 7 + b * 11 + i) as i32) % 127 - 63) as i8;
            }
            for j in 0..16 {
                row_bsums[row][b * 16 + j] = (((row * 3 + b * 5 + j) as i16) % 31) - 15;
            }
        }
    }

    // Row deltas: 4 × n_blocks floats, in (block, row) order per llama.cpp's
    // block_q8_Kx4.d[4] layout (one delta per row, and the block stride comes
    // from stepping through consecutive block_q8_Kx4 structs).
    for b in 0..n_blocks {
        for row in 0..4 {
            row_d.push(0.01 + (row as f32) * 0.001 + (b as f32) * 0.0001);
        }
    }

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

    // For each super-block, assert:
    //   dst[bloff + 0..16]    = row_d[b*4 .. b*4+4] as 4 little-endian f32s
    //   dst[bloff + 16..1040] = 4 rows × 256 quants, row-major or interleaved
    //                           per the layout confirmed in Task 1
    //   dst[bloff + 1040..1168] = 4 rows × 16 bsums, each as 2 little-endian i16 bytes
    //
    // The exact quant-layout assertions below assume the ROW-MAJOR layout
    // (row 0 contiguous, then row 1, then row 2, then row 3). If Task 1's
    // research confirmed a different interleaved layout, rewrite the quant
    // loop to match.
    for b in 0..n_blocks {
        let bloff = b * 1168;

        // d[0..4]: four row deltas as f32
        for row in 0..4 {
            let expected = row_d[b * 4 + row];
            let off = bloff + row * 4;
            let actual = f32::from_le_bytes([dst[off], dst[off+1], dst[off+2], dst[off+3]]);
            assert_eq!(
                actual.to_bits(), expected.to_bits(),
                "b={b}, row={row}: d mismatch. expected {expected}, got {actual}",
            );
        }

        // qs[0..1024]: 4 rows × 256 quants, row-major assumption
        for row in 0..4 {
            for i in 0..qk {
                let expected = row_qs[row][b * qk + i];
                let off = bloff + 16 + row * 256 + i;
                assert_eq!(
                    dst[off] as i8, expected,
                    "b={b}, row={row}, i={i}: qs byte mismatch",
                );
            }
        }

        // bsums[0..64]: 4 rows × 16 i16, row-major assumption
        for row in 0..4 {
            for j in 0..16 {
                let expected = row_bsums[row][b * 16 + j];
                let off = bloff + 1040 + row * 32 + j * 2;
                let actual = i16::from_le_bytes([dst[off], dst[off+1]]);
                assert_eq!(
                    actual, expected,
                    "b={b}, row={row}, j={j}: bsum mismatch",
                );
            }
        }
    }

    eprintln!("PASS: n_blocks={n_blocks}, byte-exact block_q8_Kx4 layout");
}
```

- [ ] **Step 7: Run the new test**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test q8k_repack_4 -- --nocapture --test-threads=1 2>&1 | tail -20
```

Expected: `PASS: n_blocks=3, byte-exact block_q8_Kx4 layout`, exit code 0.

**If the test fails:** the most likely cause is the quant-layout assumption (row-major vs. interleaved) being wrong. Check Task 1's research note and the llama.cpp source at `repack.cpp:2820+`. Either fix the kernel to match what llama.cpp reads, or fix the test's `bloff + 16 + row * 256 + i` offset formula to match what the kernel writes. **The source of truth is llama.cpp's read pattern** — the kernel must match whatever llama.cpp's gemm consumes, not the other way around.

- [ ] **Step 8: Run all gates**

Gates 1–5. All pass. The new FFI wrapper now has a caller (the test), so no dead-code warnings should remain.

- [ ] **Step 9: eabrain remember**

```bash
eabrain remember "Phase 2 Plan 1 Task 4 complete: q8k_repack_4 FFI wired and byte-exact layout verified via tests/q8k_repack_4.rs against llama.cpp block_q8_Kx4 (sizeof=1168 bytes per super-block). Quant layout confirmed as <ROW-MAJOR | INTERLEAVED — fill in the one that passed>. Gate for gemm kernel's A-side consumption in Task 7."
```

- [ ] **Step 10: Commit**

```bash
git add src/kernels/ffi_inference_types.rs src/kernels/ffi_inference.rs tests/q8k_repack_4.rs
git commit -m "feat(phase-2): FFI binding + byte-layout test for q8k_repack_4

Wires q8k_repack_4 through ffi_inference:
  - Q8kRepack4Fn type in ffi_inference_types.rs
  - KernelTableInference.q8k_repack_4 field
  - Library load from libq8k_repack_4.so
  - Public unsafe wrapper

tests/q8k_repack_4.rs builds 3 super-blocks of synthetic Q8K input
with distinct non-constant patterns per (row, block, position), runs
the kernel, and asserts every output byte against the expected
block_q8_Kx4 layout (sizeof 1168 bytes per super-block).

This locks the A-side input format that Plan 1's gemm kernel consumes.
Task 7 reuses q8k_repack_4 to build block_q8_Kx4 A-side buffers from
per-column Q8K input for the gemm bit-exact test."
```

---

## Task 5: x86 AVX2 gemm kernel — `q4k_dot_8x8_gemm.ea`

**Goal:** Write the x86 AVX2 Q4K 8×8 × Q8K gemm kernel. Structurally port llama.cpp's AVX2 fallback at `arch/x86/repack.cpp:2816-3487` with the helper-func decomposition pinned in Task 1. **This is the biggest task in Plan 1 — expect 400-500 lines of Eä code.**

**Files:**
- Create: `kernels/q4k_dot_8x8_gemm.ea`
- Read (no edit): `docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md` (Task 1)
- Read (no edit): `$HOME/projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` lines 2816-3487 (AVX2 fallback of `ggml_gemm_q4_K_8x8_q8_K`)
- Read (no edit): `kernels/q4k_dot_8x8.ea` (olorin's existing AVX2 matvec — same weight unpack, different accumulator strategy)
- Read (no edit): `kernels/q4k_dot_8x8_dual.ea` (Phase B.2 dual matvec — precedent for helper-func decomposition and multi-accumulator lane usage in Eä)

**Line budget:** 500 lines HARD. If the kernel approaches 500, split per Task 1's helper-func plan (section A unpack, section B scale decode, section C per-rp body). Multiple `func` definitions inside one `.ea` file is the preferred split — not multiple files.

- [ ] **Step 1: eabrain baseline**

```bash
eabrain status
eabrain ref permute_bytes
eabrain ref blend_epi32
eabrain ref permutevar8x32
eabrain ref inserti32x8
eabrain ref maddubs_i16
eabrain ref madd_i16
eabrain ref add_epi32
eabrain ref cvtepi32_ps
```

Expected: most match (they're in existing kernels). `permutevar8x32` and `inserti32x8` may not exist by those names in Eä — they may be exposed as `shuffle` / `permute` / `concat_*`. **Grep eacompute source if a name doesn't match**:

```bash
grep -n "permute\|shuffle\|blend\|inserti32" ~/projects/eacompute/src/typeck/intrinsics_simd.rs | head -20
grep -n "permute\|shuffle\|blend\|inserti32" ~/projects/eacompute/src/codegen/simd.rs | head -20
```

If an intrinsic llama.cpp uses genuinely doesn't exist in Eä, stop and surface to the user. Do not fake it.

- [ ] **Step 2: Create the kernel file with the signature**

Create `kernels/q4k_dot_8x8_gemm.ea` starting with:

```ea
// q4k_dot_8x8_gemm.ea — Q4K 8x8 × Q8K batched gemm (x86 AVX2 SIMD).
//
// Line-for-structure port of ggml_gemm_q4_K_8x8_q8_K's AVX2 fallback
// at ~/projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp:2816-3487.
//
// Tile: 16 A-rows × 8 B-cols × nb super-blocks of K.
// Accumulators: 16 f32x8 for acc_rows (lane k = B-col k for row i) +
//               16 f32x8 for acc_min_rows. AVX2 has only 16 YMM registers,
// so the compiler spills ~half. This matches llama.cpp's AVX2 path behavior.
//
// A-side input: block_q8_Kx4 structs, produced by q8k_repack_4 upstream.
// Accessed via 4 a_ptrs[rp] pointers (rp in 0..4), each stepping through
// nb super-blocks.
//
// Weight-side input: block_q4_Kx8 tiles (same repacked layout as the matvec
// at kernels/q4k_dot_8x8.ea). One b_ptr per tile, stepping through nb
// super-blocks.
//
// Helper-func decomposition (see research note
// docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md):
//   - unpack_weight_b:   per-super-block weight load + nibble extract
//   - decode_scales_b:   per-super-block scale utmp decode
//   - acc_rp_b:          per-rp A-side load + dot + FMA
//
// Per-output to_bits() bit-exactness against N matvec calls holds because
// rows are independent and the per-(row, col) f32 FMA chain is identical.

#[cfg(x86_64)]

// ── Helper func: decode utmp scales ────────────────────────────────
// (full body written in Step 4)
// func decode_scales_b(...) -> (scale_0, scale_1, mins_01) { ... }

// ── Helper func: unpack weight tile for one super-block ────────────
// (full body written in Step 3)
// func unpack_weight_b(...) -> (rhs_mat vectors) { ... }

// ── Helper func: per-rp A-side load + dot + FMA ────────────────────
// (full body written in Step 5)
// func acc_rp_b(...) -> () { ... }

// ── Main entry point ───────────────────────────────────────────────
export func q4k_8x8_q8k_gemm(
    packed:       *restrict u8,
    a_ptrs:       *restrict u8,
    pow2:         *restrict f32,
    scratch:      *mut u8,
    acc_scratch:  *mut f32,
    out:          *mut f32,
    bs:           i32,
    n_rows_a:     i32,
    n_cols_b:     i32,
    n_cols_inner: i32
) {
    // Outer skeleton written in Step 6.
}
```

- [ ] **Step 3: Write `unpack_weight_b` helper**

Translate llama.cpp's AVX2 fallback lines ~2856–2905 into an Eä helper func. The input is `(packed_weight_base, b_tile_offset)`; output is the nibble-extracted weight vectors needed for the per-sub-block dot loop.

The llama.cpp structure is:
1. 8 raw `_mm256_loadu_si256` loads from `b_ptr[b].qs + sb * 256 + {0, 32, 64, 96, 128, 160, 192, 224}`.
2. 8 `_mm256_permutevar8x32_epi32` + `_mm256_blend_epi32` combos to reshape into `rhs_raw_mat_0145_0..3` and `rhs_raw_mat_2367_0..3`.
3. 8 `_mm256_and_si256(x, m4b)` for low nibbles (→ `rhs_mat_0145_00..03`, `rhs_mat_2367_00..03`).
4. 8 `_mm256_and_si256(_mm256_srli_epi16(x, 4), m4b)` for high nibbles (→ `rhs_mat_0145_10..13`, `rhs_mat_2367_10..13`).

**Eä equivalent** (sketch — match the existing matvec `kernels/q4k_dot_8x8.ea` line 87-114 for the same pattern at smaller scale):

```ea
func unpack_weight_b(
    packed: *restrict u8,
    tile_base: i32,  // byte offset of this tile's start in packed
    sb: i32,          // sub-block index in 0..4
    m4b: u8x32,       // splat(15)
    shift4: u8x32,    // splat(4)
    bea: u8x32,       // blend_even mask
    beb: u8x32,
    boa: u8x32,       // blend_odd mask
    bob: u8x32
) {
    // ... 8 packed loads ...
    // ... 8 permute+blend for rhs_raw_mat_0145/2367 ...
    // ... 8 nibble extract for low ...
    // ... 8 nibble extract for high ...

    // Return 16 u8x32 vectors as a struct, or write to an output struct
    // pointer, or accept them as output parameters. Ea's return type
    // system determines which pattern works.

    // Approach: use a struct passed by mutable pointer to return all 16
    // vectors, since Ea probably can't return 16 u8x32 values from a func.
}
```

**Note on Eä return conventions:** Ea funcs return at most one value. For helpers that need to produce 16 vectors, use one of:
- An output struct pointer parameter: `out: *mut u8 [cap: ...]`.
- A scratch buffer the caller provides.
- Inline the helper body into the main export func (no helper).

Check `kernels/q4k_dot_8x8_dual.ea` for how the B.2 kernel handled large local state — if it declared all locals inline without a helper func, consider doing the same here if the helper-func approach hits Eä limitations.

**If the helper-func pattern works**: write `unpack_weight_b` now, targeting ~90 lines of body.

**If helper funcs don't work with Ea's return model**: skip this step and inline the unpack into the main export func in Step 6. The line budget gets tighter; you may need to split across multiple export funcs or factor differently.

- [ ] **Step 4: Write `decode_scales_b` helper**

Translate llama.cpp's AVX2 fallback lines ~2907–2970 (the utmp dance + `scale_0145_0`, `scale_0145_1`, `mins_01`). Same considerations as Step 3 re: helper-func return values. Target ~60 lines of body.

- [ ] **Step 5: Write `acc_rp_b` helper (per-rp A-side + FMA)**

Translate llama.cpp's AVX2 fallback lines ~2972–3120 (the `for (int rp = 0; rp < 4; rp++)` body — 4 row-pair accumulator updates). This is the most complex helper; target ~120 lines.

Inputs:
- `a_ptrs[rp][b]` reader (i.e., one `block_q8_Kx4 *` at super-block b)
- The unpacked weight vectors from `unpack_weight_b`
- The decoded scales from `decode_scales_b`
- The 4 row accumulators for this rp iteration (`acc_rows[rp*4 + 0..3]`, `acc_min_rows[rp*4 + 0..3]`)
- `col_d`, `col_dmin`, `row_sc` from the outer b loop

The body performs:
1. Load 4 × `__m256i` of Q8 quants from `a_ptr[b].qs + {0, 32, 64, 96}`.
2. `permute2f128_si256(x, x, 0)` and `permute2f128_si256(x, x, 17)` to get 4 `lhs_mat_01_*` and 4 `lhs_mat_23_*` halves.
3. 16 `maddubs_epi16` calls producing intermediate `__m256i` vectors for 4 row-pair combos × 2 nibble halves × 2 sub-blocks.
4. `add_epi16` reductions into `iacc_mat_01_0`, `iacc_mat_01_1`, `iacc_mat_23_0`, `iacc_mat_23_1`, same for sub-block 1.
5. `madd_epi16` with the `scale_*` i16x16 vectors → 4 `iacc_row_k_0` + 4 `iacc_row_k_1` i32 vectors.
6. `add_epi32(iacc_row_k_0, iacc_row_k_1)` combining two sub-blocks into 4 `iacc_row_k` i32 vectors.
7. `cvtepi32_ps` → 4 f32 vectors, `mul_ps(col_d, row_d_k)`, `fmadd_ps` into `acc_rows[rp*4 + k]`.
8. Mins correction: `madd_epi16(bsums_row, mins_01)` → 4 i32 vectors, `cvtepi32_ps`, `mul_ps(col_dmin, row_d_k)`, `fmadd_ps` into `acc_min_rows[rp*4 + k]`.

**This is the most challenging block to get bit-exact.** Follow llama.cpp line-by-line. Name locals with the same suffixes (`_0_sp1`, `_2_sp1`, `_0_sp2`, etc.) for easier comparison.

- [ ] **Step 6: Write the outer y/x/b loop skeleton**

The main `export func q4k_8x8_q8k_gemm` body orchestrates the helpers:

```ea
{
    let nb: i32 = n_cols_inner / 256

    // Masks + constants (shared across all iterations)
    let m4b: u8x32 = splat(15)
    let shift4: u8x32 = splat(4)
    let km1: i32 = 0x3f3f3f3f
    let km2: i32 = 0x0f0f0f0f
    let km3: i32 = 0x03030303
    // ... blend masks bea, beb, boa, bob (same as matvec) ...

    let mut y: i32 = 0
    while y < n_rows_a / 4 {
        // 4 block_q8_Kx4 pointers for 16 A-rows
        let a0_off: i32 = (y + 0) * nb * 1168
        let a1_off: i32 = (y + 1) * nb * 1168
        let a2_off: i32 = (y + 2) * nb * 1168
        let a3_off: i32 = (y + 3) * nb * 1168

        let mut x: i32 = 0
        while x < n_cols_b / 8 {
            // 16 acc_rows + 16 acc_min_rows
            let mut acc_rows_00: f32x8 = splat(0.0)
            let mut acc_rows_01: f32x8 = splat(0.0)
            // ... through acc_rows_15 (16 total)
            let mut acc_min_00: f32x8 = splat(0.0)
            // ... through acc_min_15

            let mut b: i32 = 0
            while b < nb {
                let tile_base: i32 = x * nb * 1152 + b * 1152

                // Section A: unpack weight for this super-block
                let (rhs_lo_00, rhs_lo_01, ...) = unpack_weight_b(packed, tile_base, 0, m4b, shift4, bea, beb, boa, bob)
                // If helper doesn't work, inline here.

                // Section B: decode scales for sub-block 0
                let (scale_0145_0_0, scale_2367_0_0, scale_0145_1_0, scale_2367_1_0, mins_01_0) = decode_scales_b(packed, tile_base, 0, km1, km2, km3)

                // col_d + col_dmin for this super-block
                let col_d: f32x8 = load f16->f32 at packed[tile_base + d_offset] ...
                let col_dmin: f32x8 = load f16->f32 at packed[tile_base + dmin_offset] ...

                // Section C: 4 rp iterations
                // rp = 0:
                let row_sc_0..3: f32x8 = splat(a_d_rp0[0..3])
                acc_rp_b(a_ptr_rp0, ..., &mut acc_rows_00..03, &mut acc_min_00..03, col_d, col_dmin, row_sc_0..3)
                // rp = 1, 2, 3 similarly

                b = b + 1
            }

            // Finalize: store acc_rows - acc_min_rows to out[row, col] for 16 rows × 8 cols
            // out[(y*4 + i) * bs + x * 8 + k] for i in 0..16, k in 0..8
            // (16 stores of f32x8 vectors)
            let store_off_0: i32 = (y * 4 + 0) * bs + x * 8
            store(out, store_off_0, acc_rows_00 .- acc_min_00)
            // ... 15 more stores ...

            x = x + 1
        }

        y = y + 1
    }
}
```

- [ ] **Step 7: Compile via build.rs**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -40
```

Expected: zero errors. If the compile fails, the most likely causes are:
- Missing or miss-typed intrinsic (check eacompute source per Step 1).
- Helper-func return value limitation — inline the helper body if so.
- Array literal in an operator slot — fix with a typed `let` intermediate (B.2 precedent).
- Load width inference — add explicit type annotations (B.2 precedent).

**If the build succeeds**: confirm the `.so` is produced:

```bash
find target/release/build -name "libq4k_dot_8x8_gemm.so" 2>&1
```

Expected: one path.

- [ ] **Step 8: Line limit check**

```bash
wc -l kernels/q4k_dot_8x8_gemm.ea
```

**Must be ≤ 500.** If it's over:
- Move the per-rp body into a helper func if you didn't already.
- Move the weight unpack + scale decode into a second helper func.
- If still over 500, re-examine the body for repetitive sections that can be collapsed.
- **Do not split into multiple `.ea` files** — keep the kernel monolithic at the LLVM level.

- [ ] **Step 9: Run all gates**

Gates 1–5. All pass. The new kernel is a dead `.so` until Task 7 adds the FFI wrapper.

- [ ] **Step 10: eabrain index**

```bash
eabrain index
eabrain search q4k_8x8_q8k_gemm
```

- [ ] **Step 11: Commit**

```bash
git add kernels/q4k_dot_8x8_gemm.ea
git commit -m "feat(phase-2): x86 AVX2 q4k_8x8_q8k_gemm kernel

Line-for-structure port of ggml_gemm_q4_K_8x8_q8_K's AVX2 fallback at
llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp:2816-3487.

Tile: 16 A-rows × 8 B-cols × nb super-blocks of K.
Accumulators: 16 f32x8 acc_rows + 16 f32x8 acc_min_rows per tile.
AVX2 has 16 YMM registers, compiler spills ~half of the 32 live
vectors to stack/L1 (matches llama.cpp's AVX2 path behavior).

A-side input is block_q8_Kx4 via a_ptrs pointer, produced by the
q8k_repack_4 helper kernel (landed in Task 4). Weight-side is the
existing block_q4_Kx8 format from Phase A.

Helper-func decomposition:
  - unpack_weight_b: per-super-block weight load + nibble extract
  - decode_scales_b: per-super-block scale utmp decode
  - acc_rp_b: per-rp A-side load + dot + FMA (called 4× per super-block)

Per-output to_bits() bit-exactness against N matvec calls on the same
input is the correctness claim, verified in Task 7 via
tests/gemm_q4k_8x8.rs.

Not yet wired into Rust. Dead .so until Task 7."
```

---

## Task 6: ARM NEON+dotprod gemm kernel — `q4k_dot_8x8_gemm_arm.ea`

**Goal:** Write the ARM NEON+dotprod gemm, derived from olorin's own `kernels/q4k_dot_8x8_arm.ea` matvec. **No llama.cpp reference** — both llama.cpp ARM gemm paths require i8mm which Cortex-A76 lacks.

**Files:**
- Create: `kernels/q4k_dot_8x8_gemm_arm.ea`
- Read (no edit): `kernels/q4k_dot_8x8_arm.ea` (post-B.1-fix matvec, ~263 lines, structural template)
- Read (no edit): `docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md` Step 6 (ARM derivation sketch)

- [ ] **Step 1: eabrain baseline**

```bash
eabrain ref vdot_i32
eabrain ref addp_i32
eabrain ref addp_i16
eabrain search q4k_dot_8x8_arm
```

Confirm NEON intrinsics are available. Recall from B.2: `hadd_i16`/`hadd_i32` are **x86-only** — use `addp_i16`/`addp_i32` for ARM.

- [ ] **Step 2: Copy the matvec as starting point**

```bash
cp kernels/q4k_dot_8x8_arm.ea kernels/q4k_dot_8x8_gemm_arm.ea
```

Do NOT copy `kernels/q4k_dot_8x8_gemm.ea` (the x86 gemm from Task 5) — x86 and ARM kernel structures are too different. The ARM matvec is the cleaner starting point.

- [ ] **Step 3: Update the header and signature**

Replace the header with:

```ea
// q4k_dot_8x8_gemm_arm.ea — Q4K 8x8 × Q8K batched gemm (ARM NEON+dotprod).
//
// Derived from olorin's own kernels/q4k_dot_8x8_arm.ea (matvec) by
// extending the per-row acc0/acc1 f32x4 pair into per-(row, col-chunk)
// accumulators, with an inner col-chunk loop that amortizes weight
// unpack across N input columns.
//
// No llama.cpp reference: both llama.cpp ARM gemm paths require i8mm
// (__ARM_FEATURE_MATMUL_INT8), which Cortex-A76 (Pi 5) lacks. On Pi 5,
// llama.cpp falls through to ggml_gemm_q4_K_8x8_q8_K_generic (scalar).
//
// Tile: 8 rows × (N % 4) col-chunks of 4 cols each.
// N in {4, 8, 16, 32} → col-chunks in {1, 2, 4, 8}.
//
// Accumulators per col-chunk:
//   acc0_cc: f32x4 (rows 0..3, 4 col lanes)
//   acc1_cc: f32x4 (rows 4..7, 4 col lanes)
//   bias0_cc: i32x4 (rows 0..3 mins)
//   bias1_cc: i32x4 (rows 4..7 mins)

#[cfg(aarch64)]

export func q4k_8x8_q8k_gemm(
    packed:       *restrict u8,
    a_ptrs:       *restrict u8,    // block_q8_Kx4 array
    pow2:         *restrict f32,
    scratch:      *mut u8,
    acc_scratch:  *mut f32,
    out:          *mut f32,
    bs:           i32,
    n_rows_a:     i32,
    n_cols_b:     i32,
    n_cols_inner: i32
) {
    // Body written in steps 4-7.
}
```

- [ ] **Step 4: Outer loop skeleton**

ARM's gemm is organized differently from x86's 16×8 tile — we use **8 rows × 4 col-chunks-of-4**. Skeleton:

```ea
    let nb: i32 = n_cols_inner / 256
    let n_col_chunks: i32 = n_rows_a / 4  // N/4 chunks of 4 cols

    // Constants from existing matvec
    let m4b: i8x16 = splat(15)
    let shift4: i8x16 = splat(4)
    let km1: i32 = 0x3f3f3f3f
    let km2: i32 = 0x0f0f0f0f
    let km3: i32 = 0x03030303
    let dup8: u8x16 = [0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7]

    let d16: *restrict i16 = ptr_as_i16(ptr_as_i8(packed))
    let packed_i8: *restrict i8 = ptr_as_i8(packed)
    let out_f32: *mut f32 = ptr_as_f32(ptr_as_i8(out))
    let scratch_i16: *mut i16 = ptr_as_i16(ptr_as_i8(scratch))

    // Outer loop over weight tiles (8-row groups)
    let mut tile_x: i32 = 0
    while tile_x < n_cols_b / 8 {
        // Acc_rows per col-chunk: allocate enough locals for max N
        // Allocate MAX 8 chunks worth (supports up to N=32)
        let mut acc0_cc0: f32x4 = splat(0.0)  // rows 0..3, cols 0..3
        let mut acc1_cc0: f32x4 = splat(0.0)  // rows 4..7, cols 0..3
        let mut acc0_cc1: f32x4 = splat(0.0)  // rows 0..3, cols 4..7
        let mut acc1_cc1: f32x4 = splat(0.0)  // rows 4..7, cols 4..7
        // ... through acc0_cc7, acc1_cc7 for N=32

        let mut b: i32 = 0
        while b < nb {
            // Per-super-block body written in Step 5
            b = b + 1
        }

        // Store acc_row[i][cc] - bias[i][cc] * sb_mn[cc] for each row × col-chunk
        // (see Step 6 for exact store pattern)

        tile_x = tile_x + 1
    }
```

**Note on the dynamic col-chunk count:** Ea doesn't support arrays of vectors indexed at runtime. Declare separate locals `acc0_cc0`, `acc0_cc1`, ..., `acc0_cc7` for the max supported col-chunks (N=32 → 8 chunks). For smaller N, the unused chunks stay zero and don't affect correctness.

- [ ] **Step 5: Per-super-block body**

Mirror the matvec's per-sb body structure, but wrap the cp loop + scale apply + bias accumulation in an **outer col-chunk loop**. Eä can't do a runtime-counted loop over distinct locals, so unroll the col-chunk loop at source level using `if n_col_chunks > 0 { ... acc_cc0 body ... } if n_col_chunks > 1 { ... acc_cc1 body ... } ...`.

Sketch:

```ea
        let bp: i32 = (tile_x * nb + b) * 1152

        // Col scales (shared across col-chunks — this is the WEIGHT col, not input col)
        let q4d0: f32x4 = cvt_f16_f32(load(d16, bp / 2))
        let q4d1: f32x4 = cvt_f16_f32(load(d16, bp / 2 + 4))
        let q4dm0: f32x4 = cvt_f16_f32(load(d16, bp / 2 + 8))
        let q4dm1: f32x4 = cvt_f16_f32(load(d16, bp / 2 + 12))

        // For each col-chunk (= group of 4 input cols from a_ptrs):
        if n_col_chunks > 0 {
            // a_ptrs[cc=0][b] points to block_q8_Kx4 #0 at super-block b
            let a_cc0_off: i32 = 0 * nb * 1168 + b * 1168
            // Load this block_q8_Kx4's d[4] as f32x4
            let a_d_cc0: *restrict f32 = ptr_as_f32(a_ptrs.add(a_cc0_off))
            let row_sc_cc0: f32x4 = load_f32x4(a_d_cc0)

            // sb_sc / sb_mn for this col-chunk's 4 input rows
            let sb_sc0_cc0: f32x4 = q4d0 .* row_sc_cc0
            // ... etc, following the matvec pattern at lines ~57-66 ...

            // Run the cp loop for this col-chunk's 4 input cols
            // Reuses the matvec's cp loop structure: for cp in 0..4, load packed,
            // extract nibbles, vdot with the A-side q8 of the current cc.
            // Accumulate into acc0_cc0 and acc1_cc0.

            // (Full body: ~100 lines — mirror matvec lines ~75-230, with _cc0
            // suffixes on iacc/bias locals)
        }

        if n_col_chunks > 1 {
            // Same body but cc=1, using acc0_cc1 / acc1_cc1 / bias_cc1 / etc.
            // The A-side offset is 1 * nb * 1168 + b * 1168.
        }

        // Same for cc=2..7 up to the max
```

**Line budget check**: unrolled 8 col-chunks × ~100 lines per chunk = 800 lines. **Over the 500 limit.** Reduce by:
- Factoring the col-chunk body into a helper `func acc_for_cc(cc: i32, a_ptrs_cc_base: *i8, ...)` that takes cc as a parameter and returns void, mutating output references. This lets one helper body serve all 8 col-chunks, but the caller must pass 8+8 mutable accumulator references...
- Or: reduce the max col-chunk count to 4 (N ≤ 16), and add an assertion that N ≤ 16 at the top of the kernel. N=32 gets handled by the caller iterating twice.

**Simpler path: support only N ≤ 16** (max 4 col-chunks). The kernel asserts `n_rows_a <= 16`. N=32 is handled upstream by calling the gemm twice with n_rows_a=16 each. Loses some perf at the outer boundary but keeps the kernel under 500 lines.

Update the `n_col_chunks` max to 4, declare only `acc_cc0..cc3` locals (8 total vs 16), unroll only 4 if-branches. Line estimate drops to ~450 lines.

- [ ] **Step 6: Output store**

At end of the outer b loop, subtract mins and store. For each (row i in 0..8, col-chunk cc in 0..n_col_chunks):

```ea
        // Subtract mins and prepare final acc values
        let final_cc0_lo: f32x4 = acc0_cc0 .- to_f32(bias0_cc0) .* sb_mn0_cc0
        let final_cc0_hi: f32x4 = acc1_cc0 .- to_f32(bias1_cc0) .* sb_mn1_cc0

        // Store 8 rows × 4 cols for cc0
        // Output layout: row-major [n_rows_a × n_cols_b], out[a_row * bs + b_col]
        // Here tile_x*8 is the b_col base and a_row = cc*4 + row_in_chunk
        if n_col_chunks > 0 {
            let a_row_base: i32 = 0 * 4  // cc=0 → A-rows 0..3
            let store_off_0: i32 = (a_row_base + 0) * bs + tile_x * 8
            store(out, store_off_0 + 0, final_cc0_lo[0 lane])  // row 0, cols 0..3
            // ... 4 × 4 = 16 individual stores per cc? Or can we store f32x4 chunks?
        }
```

**Note:** The store pattern depends on whether the output is row-major (standard) or col-major. Per the spec, it's row-major `[n_rows_a × n_cols_b]` with stride `bs`. Each store is `out[(a_row) * bs + (b_col_base) + k]` for k in the lane.

Storing 4 lanes at once as f32x4 is possible if the destination stride matches. Check `kernels/q4k_dot_8x8.ea` line 223–224 for the existing matvec's f32x8 store pattern, then adapt.

- [ ] **Step 7: Cross-compile via direct `ea` invocation**

```bash
$HOME/projects/eacompute/target/release/ea \
    kernels/q4k_dot_8x8_gemm_arm.ea \
    --emit-asm --opt-level=3 \
    --target-triple=aarch64-unknown-linux-gnu \
    --target=cortex-a76 --dotprod 2>&1 | tail -30
```

Expected: `wrote q4k_dot_8x8_gemm_arm.s`. If type errors surface, apply Phase B.2 learnings (explicit `load` type annotations, `addp_i16`/`addp_i32` instead of `hadd_*`, `packed_i8` ptr cast, typed array literal contexts). Clean up the scratch file:

```bash
rm -f q4k_dot_8x8_gemm_arm.s
```

- [ ] **Step 8: Line limit check**

```bash
wc -l kernels/q4k_dot_8x8_gemm_arm.ea
```

Must be ≤ 500.

- [ ] **Step 9: Run all gates (x86 build)**

Gates 1–5. The ARM kernel is filtered out on x86 targets, so all x86 tests should be unchanged.

- [ ] **Step 10: eabrain index**

```bash
eabrain index
```

- [ ] **Step 11: Commit**

```bash
git add kernels/q4k_dot_8x8_gemm_arm.ea
git commit -m "feat(phase-2): ARM NEON+dotprod q4k_8x8_q8k_gemm kernel

Derived from olorin's own kernels/q4k_dot_8x8_arm.ea (matvec) by
extending per-row acc0/acc1 f32x4 pair into per-(row, col-chunk)
accumulators. Inner col-chunk loop (unrolled as 4 if-branches, one
per cc in 0..4) amortizes weight unpack across N input columns.

Tile: 8 rows × 4 col-chunks of 4 cols each. Max N = 16 per call;
N=32 handled upstream by calling the gemm twice.

No llama.cpp reference: both llama.cpp ARM gemm paths require i8mm
which Cortex-A76 (Pi 5) lacks; llama.cpp falls through to scalar
on Pi 5.

Cross-compile verified on this x86 workstation via ea --emit-asm
with --target-triple=aarch64-unknown-linux-gnu --target=cortex-a76
--dotprod. Runtime validation on Pi 5 deferred.

Not yet wired into Rust. FFI binding + bit-exact test in Task 7."
```

---

## Task 7: FFI binding + bit-exact test — correctness gate

**Goal:** Wire the gemm kernel through `ffi_inference.rs` and prove per-output `to_bits()` equality vs. running `q4k_8x8_q8k_matvec` N times for N ∈ {4, 8, 16, 32}. **This is the correctness gate for the whole plan.**

**Files:**
- Modify: `src/kernels/ffi_inference_types.rs` (+1 type)
- Modify: `src/kernels/ffi_inference.rs` (+1 field, +1 library load, +1 wrapper)
- Create: `tests/gemm_q4k_8x8.rs`

- [ ] **Step 1: Add the FFI type**

In `src/kernels/ffi_inference_types.rs`, after `Q8kRepack4Fn`, add:

```rust
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

- [ ] **Step 2: Wire into `KernelTableInference`**

In `src/kernels/ffi_inference.rs`, add the field:

```rust
    pub q8k_repack_4:            Q8kRepack4Fn,
    pub q4k_8x8_q8k_gemm:        Q4k8x8GemmFn,
}
```

Add the library load after `q8k_repack_4_lib`:

```rust
    let q8k_repack_4_lib     = load("q8k_repack_4")?;
    let q4k_dot_8x8_gemm_lib = load("q4k_dot_8x8_gemm")?;
```

Add the symbol transmute after `q8k_repack_4`:

```rust
            q8k_repack_4:     std::mem::transmute(sym(&q8k_repack_4_lib,     b"q8k_repack_4\0")?),
            q4k_8x8_q8k_gemm: std::mem::transmute(sym(&q4k_dot_8x8_gemm_lib, b"q4k_8x8_q8k_gemm\0")?),
```

And add `q4k_dot_8x8_gemm_lib` to the `libs: vec![...]` line.

- [ ] **Step 3: Add the public wrapper**

At the end of `src/kernels/ffi_inference.rs`, after the `q8k_repack_4` wrapper, add:

```rust
#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_8x8_q8k_gemm(
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
) {
    (k().q4k_8x8_q8k_gemm)(
        packed, a_ptrs, pow2, scratch, acc_scratch, out,
        bs, n_rows_a, n_cols_b, n_cols_inner,
    )
}
```

- [ ] **Step 4: Write the bit-exact test**

Create `tests/gemm_q4k_8x8.rs`:

```rust
//! Bit-exact correctness gate for q4k_8x8_q8k_gemm.
//!
//! For each N in {4, 8, 16, 32}, runs the fused gemm on layer-0 ffn_gate
//! against N synthetic input columns and asserts per-output to_bits()
//! equality vs. running q4k_8x8_q8k_matvec N times with per-column input.
//!
//! Naming map (olorin matvec ↔ llama.cpp gemm):
//!   olorin `n_rows` (= ffn_dim = 6144)  ↔  llama.cpp `nc` = n_cols_b
//!   olorin `n_cols` (= hidden_dim)      ↔  llama.cpp `n`  = n_cols_inner
//!   test's `n`     (= batch size)       ↔  llama.cpp `nr` = n_rows_a
//!
//! Output layout: row-major [n_rows_a × n_cols_b] with stride bs = n_cols_b.

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
fn gemm_matches_matvec_loop_bitexact() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model at {}", model_path());
        return;
    }
    let gguf  = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    // Layer 0 ffn_gate: Q4K, shape [ffn_dim=6144, hidden_dim=1536].
    let lw = &model.layers[0];
    assert_eq!(
        lw.w_gate_dtype,
        olorin::inference::matmul::GGML_TYPE_Q4_K,
        "test requires Q4K ffn_gate"
    );
    let n_rows = model.ffn_dim[0];
    let n_cols = model.hidden_dim;
    let n_blocks = n_cols / 256;
    let tile_bytes = n_blocks * 1152;
    let n_tiles = n_rows / 8;

    // Repack weight once (Phase B.2's q4k_repack_8x8).
    let mut packed = vec![0u8; n_tiles * tile_bytes];
    unsafe {
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.w_gate, packed.as_mut_ptr(), n_rows as i32, n_cols as i32,
        );
    }
    let pow2 = olorin::inference::matmul::pow2_table();

    for &n in &[4usize, 8, 16, 32] {
        eprintln!("---- N={n} ----");

        // Build n per-column Q8K inputs with non-constant values per (col, pos).
        let mut per_col_qs: Vec<Vec<i8>> = vec![Vec::new(); n];
        let mut per_col_d: Vec<Vec<f32>> = vec![Vec::new(); n];
        let mut per_col_bsums: Vec<Vec<i16>> = vec![Vec::new(); n];
        for k in 0..n {
            per_col_qs[k] = vec![0i8; n_cols];
            per_col_d[k] = vec![0.0f32; n_blocks];
            per_col_bsums[k] = vec![0i16; n_blocks * 16];
            for i in 0..n_cols {
                per_col_qs[k][i] = (((k * 7 + i) as i32) % 127 - 63) as i8;
            }
            for i in 0..n_blocks {
                per_col_d[k][i] = 0.01 + (k as f32) * 0.0013 + (i as f32) * 0.0001;
            }
            for j in 0..(n_blocks * 16) {
                per_col_bsums[k][j] = (((k * 3 + j) as i16) % 31) - 15;
            }
        }

        // Reference: n sequential matvec calls.
        let mut ref_out = vec![0f32; n_rows * n]; // col-major (k * n_rows + row)
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

        // Build the block_q8_Kx4 A-side buffer for the gemm call.
        // Layout: (n/4) groups of block_q8_Kx4, each containing 4 consecutive
        // input columns. Each block_q8_Kx4 is n_blocks × 1168 bytes.
        assert_eq!(n % 4, 0, "N must be % 4 == 0 for gemm");
        let n_groups = n / 4;
        let mut a_ptrs_buf = vec![0u8; n_groups * n_blocks * 1168];
        for g in 0..n_groups {
            // Four rows for this group: cols g*4 + 0..3
            let row0 = g * 4 + 0;
            let row1 = g * 4 + 1;
            let row2 = g * 4 + 2;
            let row3 = g * 4 + 3;

            // Assemble a per-group row_d: 4 × n_blocks f32s in (block, row) order.
            let mut row_d = vec![0f32; 4 * n_blocks];
            for b in 0..n_blocks {
                row_d[b * 4 + 0] = per_col_d[row0][b];
                row_d[b * 4 + 1] = per_col_d[row1][b];
                row_d[b * 4 + 2] = per_col_d[row2][b];
                row_d[b * 4 + 3] = per_col_d[row3][b];
            }

            let group_dst_off = g * n_blocks * 1168;
            unsafe {
                olorin::kernels::ffi_inference::q8k_repack_4(
                    per_col_qs[row0].as_ptr(),
                    per_col_qs[row1].as_ptr(),
                    per_col_qs[row2].as_ptr(),
                    per_col_qs[row3].as_ptr(),
                    row_d.as_ptr(),
                    per_col_bsums[row0].as_ptr(),
                    per_col_bsums[row1].as_ptr(),
                    per_col_bsums[row2].as_ptr(),
                    per_col_bsums[row3].as_ptr(),
                    a_ptrs_buf[group_dst_off..].as_mut_ptr(),
                    n_blocks as i32,
                );
            }
        }

        // Candidate: one gemm call producing [n_rows_a × n_cols_b] row-major output.
        //
        // Output is row-major with stride bs = n_cols_b = n_rows (6144 for ffn_gate).
        // gemm_out[a_row * bs + b_col] for a_row in 0..n, b_col in 0..n_rows.
        // Total size = n × n_rows floats.
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
                n_rows as i32,       // bs = n_cols_b (row stride in f32s)
                n as i32,            // n_rows_a (= batch N, llama.cpp's nr)
                n_rows as i32,       // n_cols_b (= olorin's n_rows, llama.cpp's nc)
                n_cols as i32,       // n_cols_inner (= olorin's n_cols, K)
            );
        }

        // Per-output to_bits() equality.
        // ref_out is stored as [col-k, row-b_col] with ref_out[k * n_rows + b_col].
        // gemm_out is stored as [a_row, b_col] row-major with
        //   gemm_out[a_row * n_rows + b_col].
        // Under the naming map, k == a_row and b_col indexes n_rows.
        // So both formulas reduce to index = a_row * n_rows + b_col.
        let mut mismatch_count = 0usize;
        for a_row in 0..n {
            for b_col in 0..n_rows {
                let ref_v = ref_out[a_row * n_rows + b_col];
                let gemm_v = gemm_out[a_row * n_rows + b_col];
                if ref_v.to_bits() != gemm_v.to_bits() {
                    if mismatch_count < 5 {
                        eprintln!(
                            "N={n}, a_row={a_row}, b_col={b_col}: MISMATCH. \
                             ref={ref_v} ({:#x}) gemm={gemm_v} ({:#x})",
                            ref_v.to_bits(), gemm_v.to_bits(),
                        );
                    }
                    mismatch_count += 1;
                }
            }
        }
        if mismatch_count > 0 {
            panic!("N={n}: {mismatch_count} mismatches out of {} outputs", n * n_rows);
        }
        eprintln!("PASS: N={n}, {} outputs bit-exact", n * n_rows);
    }
}
```

- [ ] **Step 5: Run the new test**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemm_q4k_8x8 -- --nocapture --test-threads=1 2>&1 | tail -30
```

Expected: `PASS: N=4`, `PASS: N=8`, `PASS: N=16`, `PASS: N=32`, then `test result: ok`.

**If any N fails:** the first 5 mismatches per N are printed. Root causes in descending likelihood:
1. **Accumulator initialization bug in gemm kernel** (e.g., `acc_rows_cc1` not zeroed at tile start). Shows up as N=8+ failing with N=4 passing. Fix in Task 5/6 and re-run.
2. **Scale decode bug** (utmp dance off-by-one). Shows up as every output drifting by a small ULP factor but consistently across N. Cross-check scale decode code against llama.cpp.
3. **A-side layout mismatch** — the gemm is reading `block_q8_Kx4` but expecting a different byte order than what `q8k_repack_4` produced. Re-verify Task 1's layout decision and Task 4's `q8k_repack_4` test. If the byte layout was wrong in Task 1, fix it in both the kernel and the test.
4. **Row-major vs. col-major output storage mismatch** — gemm stores col-major but test indexes row-major, or vice versa. Check the stride `bs` and the store formulas at the end of the gemm kernel.
5. **Load stride wrong in per-rp loop** — kernel reads `a_ptrs[rp] + wrong_offset` and grabs the wrong row group. Compare the `a_ptrs_buf` layout built in the test with what the kernel indexes into.

Do not commit until all 4 N values pass.

- [ ] **Step 6: Run all gates**

Gates 1–5. Plus the new `q8k_repack_4` and `gemm_q4k_8x8` tests. All pass.

- [ ] **Step 7: eabrain remember**

```bash
eabrain remember "Phase 2 Plan 1 Task 7 complete: q4k_8x8_q8k_gemm verified bit-exact via tests/gemm_q4k_8x8.rs (to_bits equality on N in {4,8,16,32} for Gemma 4 E2B layer-0 ffn_gate = 6144×1536). FFI wired via Q4k8x8GemmFn. A-side input path: synthetic Q8K per-column → q8k_repack_4 → block_q8_Kx4 → gemm. Correctness gate for Plan 1 is now locked; perf gate in Task 8."
```

- [ ] **Step 8: Commit**

```bash
git add src/kernels/ffi_inference_types.rs src/kernels/ffi_inference.rs tests/gemm_q4k_8x8.rs
git commit -m "feat(phase-2): FFI binding + bit-exact test for q4k_8x8_q8k_gemm

Wires q4k_8x8_q8k_gemm through ffi_inference:
  - Q4k8x8GemmFn type
  - KernelTableInference.q4k_8x8_q8k_gemm field
  - Library load from libq4k_dot_8x8_gemm.so
  - Public unsafe wrapper

tests/gemm_q4k_8x8.rs::gemm_matches_matvec_loop_bitexact builds N ∈
{4, 8, 16, 32} synthetic Q8K input columns, packs them via q8k_repack_4
into block_q8_Kx4 A-side buffers, runs one gemm call, and asserts
per-output to_bits() equality vs. running q4k_8x8_q8k_matvec N times.

Total assertions per N: n_rows × N = 6144 × N.
  N=4:  24576 bit-exact outputs
  N=8:  49152 bit-exact outputs
  N=16: 98304 bit-exact outputs
  N=32: 196608 bit-exact outputs

All green on this workstation (AVX2).

This is the Plan 1 correctness gate. Task 8 adds the perf gate."
```

---

## Task 8: `tests/bench_q4k_gemm.rs` rewrite + ≥1.5× perf gate

**Goal:** Make the existing dead-code `tests/bench_q4k_gemm.rs` compile against the new gemm signature, rewire its Q8K input setup to use `q8k_repack_4` + `block_q8_Kx4`, run it, and verify the N=8 speedup is ≥ 1.5× vs. the matvec loop.

**Files:**
- Modify: `tests/bench_q4k_gemm.rs` (currently committed as dead code on this branch; references `ffi_inference::q4k_8x8_q8k_gemm` which didn't exist pre-Plan-1)

- [ ] **Step 1: Read the existing bench file**

```bash
cat tests/bench_q4k_gemm.rs
```

Verify the current state: it allocates per-col `q8_qs / q8_d / q8_bsums` arrays with strides `n_cols + 12` / `n_blocks` / `n_blocks * 16`, loops N ∈ {1, 2, 8, 32, 128}, and asserts N=8 speedup ≥ 1.5×.

- [ ] **Step 2: Rewrite the Q8K input setup**

Replace the Q8K allocation and initialization block with a `q8k_repack_4`-based A-side buffer builder. Also remove N=1 and N=2 from the sweep (gemm requires N ≥ 4).

```rust
    // New N sweep: 4, 8, 16, 32, 128 (all % 4 == 0).
    for &n in &[4usize, 8, 16, 32, 128] {
        assert_eq!(n % 4, 0, "gemm requires n % 4 == 0");
        let n_groups = n / 4;

        // Build per-column Q8K input (for the matvec reference loop).
        let mut per_col_qs: Vec<Vec<i8>>  = vec![vec![5i8;  n_cols]; n];
        let mut per_col_d:  Vec<Vec<f32>> = vec![vec![0.01f32; n_blocks]; n];
        let mut per_col_bsums: Vec<Vec<i16>> = vec![vec![17i16; n_blocks * 16]; n];

        // Make the data non-constant per (col, pos) so the compiler can't
        // constant-fold anything across cols.
        for k in 0..n {
            per_col_qs[k][0] = ((k as i32) % 127 - 63) as i8;
            per_col_d[k][0] = 0.01 + (k as f32) * 0.0013;
            per_col_bsums[k][0] = ((k as i16) % 31) - 15;
        }

        // Build the block_q8_Kx4 A-side buffer for the gemm (same as test).
        let mut a_ptrs_buf = vec![0u8; n_groups * n_blocks * 1168];
        for g in 0..n_groups {
            let row0 = g * 4 + 0;
            let row1 = g * 4 + 1;
            let row2 = g * 4 + 2;
            let row3 = g * 4 + 3;
            let mut row_d = vec![0f32; 4 * n_blocks];
            for b in 0..n_blocks {
                row_d[b * 4 + 0] = per_col_d[row0][b];
                row_d[b * 4 + 1] = per_col_d[row1][b];
                row_d[b * 4 + 2] = per_col_d[row2][b];
                row_d[b * 4 + 3] = per_col_d[row3][b];
            }
            let group_dst_off = g * n_blocks * 1168;
            unsafe {
                olorin::kernels::ffi_inference::q8k_repack_4(
                    per_col_qs[row0].as_ptr(),
                    per_col_qs[row1].as_ptr(),
                    per_col_qs[row2].as_ptr(),
                    per_col_qs[row3].as_ptr(),
                    row_d.as_ptr(),
                    per_col_bsums[row0].as_ptr(),
                    per_col_bsums[row1].as_ptr(),
                    per_col_bsums[row2].as_ptr(),
                    per_col_bsums[row3].as_ptr(),
                    a_ptrs_buf[group_dst_off..].as_mut_ptr(),
                    n_blocks as i32,
                );
            }
        }

        let mut out = vec![0f32; n_rows * n];
        let mut scratch = vec![0u8; 144];
        let mut acc_scratch = vec![0f32; 2 * n];

        // Iteration count — total work should be ~100 ms so timer noise is small.
        let iters: usize = std::cmp::max(5, 200 / std::cmp::max(1, n));

        // Warm-up.
        for k in 0..n {
            unsafe {
                olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
                    packed.as_ptr(),
                    per_col_qs[k].as_ptr(),
                    per_col_d[k].as_ptr(),
                    per_col_bsums[k].as_ptr(),
                    pow2.as_ptr(), scratch.as_mut_ptr(),
                    out[k * n_rows..].as_mut_ptr(),
                    n_rows as i32, n_cols as i32,
                );
            }
        }
        unsafe {
            olorin::kernels::ffi_inference::q4k_8x8_q8k_gemm(
                packed.as_ptr(),
                a_ptrs_buf.as_ptr(),
                pow2.as_ptr(),
                scratch.as_mut_ptr(),
                acc_scratch.as_mut_ptr(),
                out.as_mut_ptr(),
                n_rows as i32,
                n as i32,
                n_rows as i32,
                n_cols as i32,
            );
        }

        // Path A: N matvec calls.
        let t0 = Instant::now();
        for _ in 0..iters {
            for k in 0..n {
                unsafe {
                    olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
                        packed.as_ptr(),
                        per_col_qs[k].as_ptr(),
                        per_col_d[k].as_ptr(),
                        per_col_bsums[k].as_ptr(),
                        pow2.as_ptr(), scratch.as_mut_ptr(),
                        out[k * n_rows..].as_mut_ptr(),
                        n_rows as i32, n_cols as i32,
                    );
                }
            }
        }
        let t_matvec_loop = t0.elapsed().as_secs_f64() / iters as f64;

        // Path B: 1 gemm call.
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                olorin::kernels::ffi_inference::q4k_8x8_q8k_gemm(
                    packed.as_ptr(),
                    a_ptrs_buf.as_ptr(),
                    pow2.as_ptr(),
                    scratch.as_mut_ptr(),
                    acc_scratch.as_mut_ptr(),
                    out.as_mut_ptr(),
                    n_rows as i32,
                    n as i32,
                    n_rows as i32,
                    n_cols as i32,
                );
            }
        }
        let t_gemm = t0.elapsed().as_secs_f64() / iters as f64;

        let speedup = t_matvec_loop / t_gemm;
        eprintln!(
            "{:>6}  {:>14.3}  {:>14.3}  {:>9.2}x",
            n,
            t_matvec_loop * 1000.0,
            t_gemm * 1000.0,
            speedup,
        );

        // Acceptance gate: at N=8, gemm must be >= 1.5x faster than matvec loop.
        if n == 8 {
            assert!(
                speedup >= 1.5,
                "N=8 gemm speedup {:.2}x is below the 1.5x acceptance gate",
                speedup
            );
        }
    }
```

- [ ] **Step 3: Run the bench**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test bench_q4k_gemm -- --nocapture --test-threads=1 2>&1 | tail -30
```

Expected output: a table showing matvec-loop ms, gemm ms, and speedup for N ∈ {4, 8, 16, 32, 128}. N=8 speedup must be ≥ 1.5× or the test panics.

**If the N=8 speedup is below 1.5×:**

1. **Profile first.** Run `perf stat -e L1-dcache-loads,L1-dcache-load-misses,cycles,instructions` on a single-thread gemm vs. matvec-loop at N=8, compare dcache loads per FLOP. If gemm's dcache loads are within 60% of matvec-loop's, weight reuse isn't saving enough.
2. **Check compiler spill behavior** — `ea --emit-asm` on the gemm, look for repeated loads of `rhs_mat_*` that should have stayed in registers.
3. **Retry scratch sizing** — if the kernel's acc_scratch usage is constraining the compiler, grow it and re-run the bench.
4. **Halt and surface** if none of the above closes the gap. Do not lower the 1.5× gate to make the test pass.

- [ ] **Step 4: Run all gates**

Gates 1–5 + `q8k_repack_4` + `gemm_q4k_8x8` + `bench_q4k_gemm`. All pass.

- [ ] **Step 5: eabrain remember the bench numbers**

```bash
eabrain remember "Phase 2 Plan 1 Task 8 complete: q4k_8x8_q8k_gemm perf gate passed on Gemma 4 E2B layer-0 ffn_gate (6144×1536). N=8 speedup vs matvec loop = <FILL_IN>x (≥1.5× gate). Full speedup table: N=4: <x>, N=8: <x>, N=16: <x>, N=32: <x>, N=128: <x>. Plan 1 correctness gate (Task 7) and perf gate (Task 8) both green. Plan 2 (batched helper kernels + forward_batch test-only) is now unblocked."
```

Substitute the actual numbers from the bench output before running.

- [ ] **Step 6: Commit**

```bash
git add tests/bench_q4k_gemm.rs
git commit -m "$(cat <<'EOF'
bench(phase-2): q4k_8x8_q8k_gemm vs matvec loop — N=8 ≥1.5× gate passes

Rewrites tests/bench_q4k_gemm.rs's Q8K input setup to use q8k_repack_4
and block_q8_Kx4 A-side buffers, matching the new gemm signature.
Removes N=1 and N=2 (gemm requires N % 4 == 0); keeps N ∈ {4, 8, 16,
32, 128}.

Measured on Gemma 4 E2B layer-0 ffn_gate (n_rows=6144, n_cols=1536),
this workstation, single-threaded:

  N      matvec-loop ms       gemm ms      speedup
  4      <FILL_IN>            <FILL_IN>    <FILL_IN>x
  8      <FILL_IN>            <FILL_IN>    <FILL_IN>x   (≥1.5 gate)
  16     <FILL_IN>            <FILL_IN>    <FILL_IN>x
  32     <FILL_IN>            <FILL_IN>    <FILL_IN>x
  128    <FILL_IN>            <FILL_IN>    <FILL_IN>x

This is the Plan 1 perf gate. Plan 1 is now complete — the gemm
kernel exists, is bit-exact (Task 7), and hits its speedup gate.
Plan 2 (batched helper kernels + forward_batch test-only surface)
is unblocked.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Replace `<FILL_IN>` placeholders with the actual bench numbers before pasting the heredoc.

---

## Task 9: Plan 1 wrap-up + self-review

**Goal:** Run the full regression sweep one more time, record the Plan 1 completion in eabrain, optionally touch the vault memory, and confirm all 9 tasks' checkboxes are ticked.

**Files:** No file changes unless the self-review surfaces something. This task is mostly a ceremony commit or a no-op commit.

- [ ] **Step 1: Run the full regression sweep**

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

Expected: all 8 suites green, no skipped tests except possibly `gemma4_verify`'s llama-ref subset if the llama.cpp eval dumps aren't present.

- [ ] **Step 2: Plan 1 self-review checklist**

Tick off each item. If any is false, fix it and re-run Step 1:

- [ ] `kernels/q8k_repack_4.ea` exists, ≤ 150 lines.
- [ ] `kernels/q8k_repack_4_arm.ea` exists, cross-compiles via `ea --emit-asm`.
- [ ] `kernels/q4k_dot_8x8_gemm.ea` exists, ≤ 500 lines.
- [ ] `kernels/q4k_dot_8x8_gemm_arm.ea` exists, ≤ 500 lines, cross-compiles.
- [ ] `tests/q8k_repack_4.rs` passes byte-layout assertions.
- [ ] `tests/gemm_q4k_8x8.rs` passes `to_bits()` equality for N ∈ {4, 8, 16, 32}.
- [ ] `tests/bench_q4k_gemm.rs` N=8 speedup ≥ 1.5×.
- [ ] `src/kernels/ffi_inference_types.rs` contains `Q8kRepack4Fn` and `Q4k8x8GemmFn`.
- [ ] `src/kernels/ffi_inference.rs` wires both new libs with matching symbol names, public wrappers, and `libs:` vec entries.
- [ ] `cargo build --release` has no more warnings than baseline (10).
- [ ] Line limit gate empty except for the two pre-existing chacha20 files.
- [ ] No `// TODO`, `// HACK`, or `// for now` introduced by Plan 1's commits.
- [ ] All 9 tasks of this plan have every checkbox ticked.

- [ ] **Step 3: eabrain final memory entry**

```bash
eabrain remember "Phase 2 Plan 1 COMPLETE (2026-04-11): q4k_8x8_q8k_gemm kernel landed and verified. 9 commits: 1 research, 2 q8k_repack_4 kernels (x86+ARM), 1 repack FFI+test, 2 gemm kernels (x86+ARM), 1 gemm FFI+bit-exact test, 1 bench perf gate, 1 wrap-up. Correctness: to_bits() equality for N ∈ {4,8,16,32} on 6144×1536 ffn_gate. Perf: N=8 speedup <FILL_IN>x vs matvec loop (gate: ≥1.5x). Phase 2 Plan 2 (batched helper kernels + forward_batch test-only) is unblocked. No generate.rs changes yet — that's Plan 3."
```

- [ ] **Step 4: Optional — no-op commit for the eabrain update**

If Task 9 produces no file changes, skip the commit and let Task 8's commit stand as the last commit of Plan 1. If you updated `project_olorin1.md` or any other memory file, commit here:

```bash
git add <any-memory-or-doc-files>
git commit -m "chore(phase-2): Plan 1 wrap-up — eabrain + vault memory updates"
```

---

## Self-Review Checklist (run after completing all 9 tasks)

- [ ] Every `- [ ]` checkbox in this plan is ticked.
- [ ] Branch is 9 commits ahead of `6a43ec1` (the spec commit).
- [ ] Full regression sweep green.
- [ ] N=8 gemm speedup ≥ 1.5× recorded in Task 8's commit message with actual numbers.
- [ ] No new warnings, no new files over 500 lines, no `// TODO` / `// HACK` / `// for now`.
- [ ] `generate.rs` untouched. Production decode unchanged.

## What's unblocked after Plan 1

- **Plan 2 — batched helper kernels + forward_batch test-only.** Spec + plan to be written after Plan 1 lands. Builds the 6 batched helper kernels (q8k_quant_batched, gemma4_rmsnorm_batched, dual-rope batched, gelu_mul_batched, batched causal attention trio), adds `forward_batch` to `Gemma4State`, adds `tests/gemma4_batch_verify.rs` bit-matching vs. `forward_one` loop. Does NOT touch `generate.rs`.
- **Plan 3 — integration + bit-exact verify vs llama.cpp.** Wires `generate.rs` prefill to `forward_batch`, splits `bench_decode_speed`, runs `llama-eval-callback` comparison, records new prefill tok/s.

Do NOT start Plan 2 or Plan 3 during this plan. Their preconditions are "Plan 1 landed" (and for Plan 3, "Plan 2 landed").
