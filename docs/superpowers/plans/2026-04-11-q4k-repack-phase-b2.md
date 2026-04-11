# Q4K Repack — Phase B.2: Fused Dual 8×8 Matvec + Path B Wire-Up

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **HARD RULES (apply to ALL agents):**
> - No file exceeds 500 lines. Split before you hit the limit.
> - Every feature proven by end-to-end test. If it's not tested, it doesn't exist.
> - No fake functions. No silent fallbacks.
> - Olorin is Ea's showcase — never replace an Ea kernel with Rust scalar.
> - Match llama.cpp **bit-exact** (per-output `to_bits()` where the recipe allows).
> - eacompute compiler: `$HOME/projects/eacompute/target/release/ea`
> - Build: `PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release`
> - Branch: `gemma4-batched-prompt-eval`
> - **eabrain protocol**: run `eabrain status` and `eabrain search <name>` before grepping for any Ea symbol. Run `eabrain ref <name>` before assuming an Ea intrinsic doesn't exist (and grep `$HOME/projects/eacompute/src/typeck/intrinsics*.rs` + `$HOME/projects/eacompute/src/codegen/simd*.rs` as fallback — eabrain does not index eacompute's Rust intrinsic definitions). After editing any `.ea` kernel: run `eabrain index`. After producing a non-obvious finding: `eabrain remember "..."`.

**Goal:** Port the Q4K 8×8 repack dispatch into the work-stealing forward path (`forward_graph.rs` / `matmul_graph.rs`) so that production decode via `forward_one_graph` (called by `generate.rs`) uses the repacked weights instead of the 4-row kernel, and simultaneously land a new `q4k_8x8_q8k_matvec_dual` Ea kernel that fuses the `ffn_gate` + `ffn_up` matmul pair with shared Q8K input loads and broadcast operands. Retrofit Path A to use the same fused kernel, giving both paths an identical dispatch surface.

**Architecture:** Kernel-first TDD. Land the new dual Ea kernel (x86 + ARM) and prove it bit-exact against "two separate `q4k_8x8_q8k_matvec` calls" in a standalone Rust test *before* wiring anything into the forward pass. Then add Path B work-stealing variants (`q4k_matvec_8x8_ws`, `q4k_matvec_dual_8x8_ws`) as dead code, rewire `forward_graph.rs` call sites to use them, then retrofit Path A to use the fused dual kernel. Snapshot regeneration is the final commit.

**Tech Stack:** Rust, Ea (eacompute), x86 AVX2 + ARM NEON, `libloading` for kernel dispatch, work-stealing `GraphPool` + `SpinBarrier`, existing `q4k_dot_8x8.ea` / `q4k_dot_8x8_arm.ea` kernels as structural templates.

**Spec:** `docs/superpowers/specs/2026-04-11-q4k-repack-phase-b2-design.md` (committed as `aff7518`).

**Scope note vs. spec:** the spec described a 6-commit landing with commit 5 bundled as "Path A retrofit + Path B wire-up." This plan refines the split into **7 commits** for better bisection: Path B's dead-code kernels land first (c5), then the active `forward_graph.rs` wire-up turns them on (c6), then the Path A retrofit + snapshot regeneration lands last (c7). Between c5 and c7 every commit leaves the tree green on both test paths; only c7 regenerates the snapshot. The end state matches the spec exactly.

---

## Per-Task Verification Gates

Run these **before every `git commit`** in this plan. Any failure = do not commit, fix first.

**Gate 1 — Build clean.**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
grep -E "^(warning|error)" /tmp/olorin-build.log && exit 1 || true
```

Expected: exit 0, no warnings or errors.

**Gate 2 — Line limit.**

```bash
find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'
```

Expected: empty output.

**Gate 3 — Phase A smoke (protects B.1 functionality).**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test repack_q4k -- --test-threads=1 2>&1 | tail -6
```

Expected: 3 tests, all PASS (or all SKIP if model missing — accept skip on a fresh runner only).

**Gate 4 — Bit-exact decode regression (Path A snapshot).**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression 2>&1 | tail -6
```

Expected: PASS on Tasks 1–6. **Expected to FAIL on Task 7** (snapshot drift by ~ULP because Path A's dual path now fuses). Task 8 regenerates and re-verifies.

**Gate 5 — End-to-end smoke (guards production path).**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_smoke 2>&1 | tail -10
```

Expected: PASS on Tasks 1–8. This is the test that actually exercises `forward_one_graph` (production path).

When a task says "run all gates," it means run Gates 1–5 in order. Gate 4 gets a pass on Task 7 only if its *only* failure is the snapshot binary diff (Task 8 will confirm).

---

## Task 1: Research note — dual fusion shared/weight-specific split

**Goal:** Pin the shared vs. weight-specific work decomposition of `q4k_dot_8x8.ea` in a dedicated research note, so the kernel author in Task 2 isn't re-deriving it from the inner loop.

**Files:**
- Create: `docs/superpowers/research/2026-04-11-q4k-8x8-dual-fusion.md`
- Read (no edit): `kernels/q4k_dot_8x8.ea` (lines 53–228) — the kernel body whose fusion properties this note documents.

- [ ] **Step 1: Create the research note**

Write the full file content below to `docs/superpowers/research/2026-04-11-q4k-8x8-dual-fusion.md`:

````markdown
# q4k_8x8 dual fusion — shared vs. weight-specific split

Companion to `2026-04-08-ggml-q4k-8x8-q8k-gemm.md`. This note documents
what can be shared when running two weight matrices against the same
Q8K input column — i.e., the structural template for a new fused
`q4k_8x8_q8k_matvec_dual` kernel that processes `ffn_gate` + `ffn_up`
in one call.

## Baseline: `kernels/q4k_dot_8x8.ea` (x86 AVX2, 228 lines)

Inner loop shape, from the source:

```
while x < n_rows / 8:               # tile
    acc_row = 0; acc_min = 0
    while b < nb:                   # super-block (b loops low → high)
        row_sc = splat(q8_d[b])
        col_d, col_dmin = f16→f32 loads from packed[b]
        iacc_b = 0; iacc_min_b = 0
        q8s_half = hadd_i16(bsums_lo, bsums_hi)
        store(scratch_i16, 0, q8s_half)
        while sb < 4:               # sub-block pair
            la, lb, lc, ld = load q8_qs[b*256 + sb*64 ..]
            v00, v01, v10, v11 = concat broadcasts from la..ld
            # 8 packed weight loads from packed[bp, qs + sb*256 + 0..224]
            # Low + high nibble extract
            # utmp decode → scales_0, scales_1, mins_01
            # 16 maddubs using v00..v11 + scale madd_i16
            # iacc_b += iacc_0 + iacc_1
            # iacc_min_b += madd_i16(q8s_sb, mins_01)
            sb += 1
        acc_row = fma(to_f32(iacc_b), col_d .* row_sc, acc_row)
        acc_min = fma(to_f32(iacc_min_b), col_dmin .* row_sc, acc_min)
        b += 1
    row_nat = shuffle(acc_row, [0,2,4,6,1,3,5,7])
    store(out, x*8, row_nat .- acc_min)
    x += 1
```

## Shared when running two weight matrices (A, B) against the same Q8K

Per super-block `b`:
- `row_sc = splat(q8_d[b])` — single f32 splat, shared.
- `q8s_half` from `hadd_i16` on Q8 bsums — **scratch store is shared**,
  one 128-byte scratch is enough for the dual kernel.

Per `sb` iteration within a super-block:
- Q8 qs loads `la, lb, lc, ld` (64 bytes from `q8_qs[b*256 + sb*64]`).
- `concat_i8x16` broadcasts `v00, v01, v10, v11` (256 bytes of register
  state holding the broadcasted Q8K input columns).

**These broadcasts are the primary fusion win.** `v00..v11` feed the
16 `maddubs_i16` ops in the dot loop; with two weight matrices they'd
otherwise be rebuilt twice from scratch. Holding them in registers
across both weight streams is what saves memory bandwidth and
uop pressure.

## Weight-specific per (sub-block, matrix)

- 8 × 32-byte packed weight loads (256 B).
- 16 low/high nibble extract ops.
- Scale decode (utmp) → `scales_0`, `scales_1`, `mins_01` i16x16
  literal construction.
- 16 `maddubs_i16` ops consuming shared `v**` broadcasts.
- `madd_i16` scale multiplications + int32 accumulation into
  matrix-local `iacc`, `iacc_min`.

## Weight-specific per super-block

- `col_d`, `col_dmin` f16→f32 loads (16 bytes from the packed header).
- One FMA into `acc_row`, one FMA into `acc_min`.

## Correctness claim for the dual kernel

Per output element, the integer reduction order and the f32 FMA chain
on `acc_row_a` are **identical** to calling `q4k_8x8_q8k_matvec` once
on `packed_a` alone: interleaving B-side integer work inside the
same `sb` loop does not touch A's accumulator lanes, and rows are
independent. The per-output `to_bits()` equality holds against
"two separate calls to the single kernel."

**Consequence for the scratch:** one shared 128-byte scratch is enough
for the dual kernel; the bsums hadd only depends on Q8K input.

## Consequence for the dual kernel body

```
while x < n_rows / 8:
    acc_row_a, acc_min_a = 0, 0
    acc_row_b, acc_min_b = 0, 0
    while b < nb:
        row_sc = splat(q8_d[b])                       # SHARED
        col_d_a, col_dmin_a = from packed_a[b]        # A-specific
        col_d_b, col_dmin_b = from packed_b[b]        # B-specific
        iacc_a, iacc_min_a = 0, 0
        iacc_b, iacc_min_b = 0, 0
        q8s_half = hadd_i16(...)                      # SHARED
        store(scratch_i16, 0, q8s_half)
        while sb < 4:
            la, lb, lc, ld = load q8_qs[...]          # SHARED
            v00, v01, v10, v11 = concat broadcasts    # SHARED
            # ── A-side block (mirror of lines 87–210 of q4k_dot_8x8.ea) ──
            # ── B-side block (same body, packed_b + scales_*_b + mins_01_b) ──
            sb += 1
        # Four FMAs (vs. two in single kernel)
        acc_row_a = fma(to_f32(iacc_a),     col_d_a    .* row_sc, acc_row_a)
        acc_min_a = fma(to_f32(iacc_min_a), col_dmin_a .* row_sc, acc_min_a)
        acc_row_b = fma(to_f32(iacc_b),     col_d_b    .* row_sc, acc_row_b)
        acc_min_b = fma(to_f32(iacc_min_b), col_dmin_b .* row_sc, acc_min_b)
        b += 1
    store(out_a, x*8, shuffle(acc_row_a, [0,2,4,6,1,3,5,7]) .- acc_min_a)
    store(out_b, x*8, shuffle(acc_row_b, [0,2,4,6,1,3,5,7]) .- acc_min_b)
    x += 1
```

Estimated line count: ~360 (roughly 228 × 1.6 — body duplication plus
distinct scales/mins for the two matrices). Under the 500-line limit.

## ARM NEON

Same structural argument applies to `kernels/q4k_dot_8x8_arm.ea` (248
lines). The NEON kernel uses different intrinsic names but the same
per-`(tile, b, sb)` decomposition. Task 3 produces the NEON mirror.
````

- [ ] **Step 2: Run Gates 1–5**

All must pass. This task only adds a doc file, so everything should be green from the pre-task state.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/research/2026-04-11-q4k-8x8-dual-fusion.md
git commit -m "research(phase-b.2): dual 8x8 matvec fusion decomposition

Documents what can be shared when running two weight matrices against
the same Q8K input column. Shared: row_sc splat, bsums hadd scratch,
Q8 qs loads + concat broadcasts (v00..v11). Weight-specific: packed
loads, nibble extract, scales/mins, maddubs loop, iacc accumulator.
Per-output bit-exactness vs. two separate single-kernel calls holds
because rows are independent and A-side accumulation order is
unchanged by interleaving B-side integer work inside the sb loop.

Input to Task 2 (x86 dual kernel) and Task 3 (ARM dual kernel)."
```

---

## Task 2: New x86 AVX2 Ea kernel — `q4k_dot_8x8_dual.ea`

**Goal:** Produce a new Ea kernel that implements `q4k_8x8_q8k_matvec_dual` on x86 AVX2, structurally derived from `q4k_dot_8x8.ea`.

**Files:**
- Create: `kernels/q4k_dot_8x8_dual.ea`
- Read (no edit): `kernels/q4k_dot_8x8.ea` (full — this is the structural template)
- Read (no edit): `docs/superpowers/research/2026-04-11-q4k-8x8-dual-fusion.md` (from Task 1)

**Line budget:** ~360 lines. If you hit 500, stop and re-factor before committing.

- [ ] **Step 1: eabrain baseline**

Run, to confirm nothing in eacompute has changed since the spec was written:

```bash
eabrain search q4k_8x8_q8k_matvec
eabrain search q4k_dot_8x8
eabrain ref concat_i8x16
eabrain ref shuffle_bytes
eabrain ref maddubs_i16
```

Expected: matches for all existing names from `q4k_dot_8x8.ea`. No surprises. If any intrinsic name is missing, grep `$HOME/projects/eacompute/src/typeck/intrinsics*.rs` and `$HOME/projects/eacompute/src/codegen/simd*.rs` before concluding anything is unavailable.

- [ ] **Step 2: Copy the baseline kernel as the starting point**

```bash
cp kernels/q4k_dot_8x8.ea kernels/q4k_dot_8x8_dual.ea
```

This gives you the exact 228-line body to transform. **Do not start from scratch.**

- [ ] **Step 3: Update the header comment and function signature**

Replace the header (lines 1–7 of the copy) with:

```ea
// q4k_dot_8x8_dual.ea — Q4Kx8 repacked fused dual matvec kernel (x86 AVX2 SIMD).
//
// Fuses two weight matrices (e.g. ffn_gate + ffn_up) against the same Q8K
// input column. Shares Q8 qs loads + v00..v11 broadcasts across both weight
// streams; keeps separate iacc_a/iacc_b accumulators and separate FMAs into
// acc_row_a/acc_row_b. Per-output to_bits() equality holds against calling
// q4k_8x8_q8k_matvec twice on (packed_a, q8k) and (packed_b, q8k) because
// rows are independent and the A-side reduction order is unchanged by
// interleaving B-side integer work.
//
// See docs/superpowers/research/2026-04-11-q4k-8x8-dual-fusion.md for the
// shared vs. weight-specific split.

#[cfg(x86_64)]

func ubyte(p: *restrict u8, off: i32) -> i32 {
    return to_i32(p[off]) & 255
}
```

Change the export signature from:

```ea
export func q4k_8x8_q8k_matvec(
    packed: *restrict u8,
    q8_qs: *restrict i8,
    q8_d: *restrict f32,
    q8_bsums: *restrict i16,
    pow2: *restrict f32,
    scratch: *mut u8,
    out: *mut f32,
    n_rows: i32,
    n_cols: i32
) {
```

to:

```ea
export func q4k_8x8_q8k_matvec_dual(
    packed_a: *restrict u8,
    packed_b: *restrict u8,
    q8_qs: *restrict i8,
    q8_d: *restrict f32,
    q8_bsums: *restrict i16,
    pow2: *restrict f32,
    scratch: *mut u8,
    out_a: *mut f32,
    out_b: *mut f32,
    n_rows: i32,
    n_cols: i32
) {
```

- [ ] **Step 4: Duplicate the packed-header state for A and B**

In the body, the existing kernel has a single `d16 = ptr_as_i16(ptr_as_i8(packed))` and a single `out_f32 = ptr_as_f32(ptr_as_i8(out))`. Duplicate them:

```ea
    let d16_a: *restrict i16 = ptr_as_i16(ptr_as_i8(packed_a))
    let d16_b: *restrict i16 = ptr_as_i16(ptr_as_i8(packed_b))
    let out_a_f32: *mut f32 = ptr_as_f32(ptr_as_i8(out_a))
    let out_b_f32: *mut f32 = ptr_as_f32(ptr_as_i8(out_b))
    let scratch_i16: *mut i16 = ptr_as_i16(ptr_as_i8(scratch))
```

The `scratch_i16` stays single — it's shared.

- [ ] **Step 5: Double the per-tile accumulators**

Inside the outer `while x < n_rows / 8:` loop, replace:

```ea
        let mut acc_row: f32x8 = splat(0.0)
        let mut acc_min: f32x8 = splat(0.0)
```

with:

```ea
        let mut acc_row_a: f32x8 = splat(0.0)
        let mut acc_min_a: f32x8 = splat(0.0)
        let mut acc_row_b: f32x8 = splat(0.0)
        let mut acc_min_b: f32x8 = splat(0.0)
```

- [ ] **Step 6: Double the per-super-block col_d + iacc state**

Inside the `while b < nb:` loop, the existing kernel loads `col_d`, `col_dmin`, and initializes `iacc_b` and `iacc_min_b`. Replace:

```ea
            let d_raw: i16x8 = load(d16, bp / 2)
            let col_d: f32x8 = shuffle(cvt_f16_f32(d_raw), [0,4,1,5,2,6,3,7])
            let dm_raw: i16x8 = load(d16, bp / 2 + 8)
            let col_dmin: f32x8 = cvt_f16_f32(dm_raw)
            let row_sc: f32x8 = splat(q8_d[b])

            let mut iacc_b: i32x8 = splat(0)
            let mut iacc_min_b: i32x8 = splat(0)
```

with:

```ea
            let d_raw_a: i16x8 = load(d16_a, bp / 2)
            let col_d_a: f32x8 = shuffle(cvt_f16_f32(d_raw_a), [0,4,1,5,2,6,3,7])
            let dm_raw_a: i16x8 = load(d16_a, bp / 2 + 8)
            let col_dmin_a: f32x8 = cvt_f16_f32(dm_raw_a)

            let d_raw_b: i16x8 = load(d16_b, bp / 2)
            let col_d_b: f32x8 = shuffle(cvt_f16_f32(d_raw_b), [0,4,1,5,2,6,3,7])
            let dm_raw_b: i16x8 = load(d16_b, bp / 2 + 8)
            let col_dmin_b: f32x8 = cvt_f16_f32(dm_raw_b)

            // Shared across both weight matrices
            let row_sc: f32x8 = splat(q8_d[b])

            let mut iacc_a: i32x8 = splat(0)
            let mut iacc_min_a: i32x8 = splat(0)
            let mut iacc_b_acc: i32x8 = splat(0)
            let mut iacc_min_b_acc: i32x8 = splat(0)
```

Note on naming: `iacc_b` already exists in the single kernel as the per-super-block integer accumulator. In the dual we have both a super-block index `b` and a matrix name `b`, which collides. Use `iacc_a` / `iacc_b_acc` (or `iacc_A` / `iacc_B` — pick one convention and stay consistent). This plan uses `iacc_a` + `iacc_b_acc` to avoid collision with the super-block loop variable.

The `q8s_half` bsums scratch store (4 lines of existing code) stays single:

```ea
            let bs_lo: i16x8 = load(q8_bsums, b * 16)
            let bs_hi: i16x8 = load(q8_bsums, b * 16 + 8)
            let q8s_half: i16x8 = hadd_i16(bs_lo, bs_hi)
            store(scratch_i16, 0, q8s_half)
```

- [ ] **Step 7: Duplicate the sub-block body (A-side + B-side)**

Inside the `while sb < 4:` loop, the existing body has three phases:

1. **Shared loads** (existing): `qs = bp + 128 + sb * 256`, load `r03_0..r47_3` from `packed` at `qs + 0..224`, nibble extract into `l03_*`, `l47_*`, `h03_*`, `h47_*`. → **Now A-only.** Rename all the locals to `*_a` and load from `packed_a`:

   ```ea
                let qs_a: i32 = bp + 128 + sb * 256
                let r03_0_a: u8x32 = load(packed_a, qs_a)
                let r47_0_a: u8x32 = load(packed_a, qs_a + 32)
                // ... (all 8 raw loads, 16 nibble extracts, all *_a)
   ```

2. **Scale decode (utmp)** (existing uses `sp = ptr_as_i32(ptr_as_i8(packed))`). → **Now A-only.** Replace with `sp_a = ptr_as_i32(ptr_as_i8(packed_a))` and `scales_0_a`, `scales_1_a`, `mins_01_a`.

3. **Q8K load + broadcast** (existing): `q8b = b * 256 + sb * 64`, load `la..ld`, concat into `v00..v11`. → **Keep shared** (no `_a` suffix). Place this block **between** the A-side nibble extracts and the A-side maddubs.

   The ordering for clarity is:
   - A-side packed loads + nibble extracts + scale decode + mins_01 construction
   - **Shared Q8K loads + broadcasts** (one block)
   - A-side 16 maddubs + `iacc_a` accumulation + mins correction into `iacc_min_a`
   - B-side packed loads + nibble extracts + scale decode + mins_01 construction
   - B-side 16 maddubs (reusing the already-in-register `v00..v11`) + `iacc_b_acc` + mins correction into `iacc_min_b_acc`

4. Copy the A-side maddubs pattern (two `mut ia0, ia1` blocks with 16 total maddubs, both `madd_i16` scale applications, both sub-block `iacc` adds, mins correction) from lines 180–213 of `q4k_dot_8x8.ea`. Rename all `_a` suffix and accumulate into `iacc_a` / `iacc_min_a`.

5. Duplicate the entire A-side block for B, suffixing `_b` on all locals and accumulating into `iacc_b_acc` / `iacc_min_b_acc`. **Use `packed_b` and `sp_b`, but reuse `v00..v11`** from the shared block — do not reload or rebroadcast them.

The sub-block body roughly doubles in line count from ~130 lines to ~220 lines. That puts the whole kernel around 350–370 lines.

- [ ] **Step 8: Add the four FMAs (doubled)**

Replace the existing two FMAs:

```ea
            acc_row = fma(to_f32(iacc_b), col_d .* row_sc, acc_row)
            acc_min = fma(to_f32(iacc_min_b), col_dmin .* row_sc, acc_min)
```

with four:

```ea
            acc_row_a = fma(to_f32(iacc_a),        col_d_a    .* row_sc, acc_row_a)
            acc_min_a = fma(to_f32(iacc_min_a),    col_dmin_a .* row_sc, acc_min_a)
            acc_row_b = fma(to_f32(iacc_b_acc),    col_d_b    .* row_sc, acc_row_b)
            acc_min_b = fma(to_f32(iacc_min_b_acc),col_dmin_b .* row_sc, acc_min_b)
```

- [ ] **Step 9: Double the final store**

Replace the existing single store:

```ea
        let row_nat: f32x8 = shuffle(acc_row, [0,2,4,6,1,3,5,7])
        store(out_f32, x * 8, row_nat .- acc_min)
```

with two:

```ea
        let row_nat_a: f32x8 = shuffle(acc_row_a, [0,2,4,6,1,3,5,7])
        store(out_a_f32, x * 8, row_nat_a .- acc_min_a)

        let row_nat_b: f32x8 = shuffle(acc_row_b, [0,2,4,6,1,3,5,7])
        store(out_b_f32, x * 8, row_nat_b .- acc_min_b)
```

- [ ] **Step 10: Compile via build.rs**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
grep -E "^(warning|error)" /tmp/olorin-build.log
```

Expected: zero warnings, zero errors. The `build.rs` auto-discovers the new `.ea` file and produces `libq4k_dot_8x8_dual.so` in `target/release/build/olorin-*/out/`. Confirm with:

```bash
find target/release/build -name "libq4k_dot_8x8_dual.so" | head -1
```

Expected: one path.

- [ ] **Step 11: Line limit check**

```bash
wc -l kernels/q4k_dot_8x8_dual.ea
```

Expected: ~350 lines, and **must be <= 500**. If ≥ 500, stop and refactor — the skill-level hard rule wins over any other consideration.

- [ ] **Step 12: Run all gates**

Gates 1 (build) + 2 (line limit) + 3 (repack_q4k) + 4 (parallel_regression) + 5 (smoke). All must pass — this task adds a new kernel but doesn't wire it anywhere, so every existing test should still be green.

- [ ] **Step 13: eabrain index**

```bash
eabrain index
eabrain search q4k_8x8_q8k_matvec_dual
```

Expected: new symbol appears in the index.

- [ ] **Step 14: Commit**

```bash
git add kernels/q4k_dot_8x8_dual.ea
git commit -m "feat(phase-b.2): x86 AVX2 q4k_8x8_q8k_matvec_dual kernel

Fuses two weight matrices (e.g. ffn_gate + ffn_up) against the same Q8K
input column by sharing the bsums hadd scratch, the Q8 qs loads, and
the v00..v11 concat broadcasts across both weight streams. Keeps
separate iacc/acc_row accumulators and separate FMAs per matrix.

Per-output accumulation order is identical to calling the single 8x8
matvec twice — verified in Task 4 via tests/dual_q4k_8x8.rs.

Not yet wired into Rust. Kernel is loadable but unused until Task 4
adds the FFI binding."
```

---

## Task 3: New ARM NEON Ea kernel — `q4k_dot_8x8_dual_arm.ea`

**Goal:** Mirror Task 2's x86 kernel for ARM NEON, same export signature, same structural split. Cross-compile only — runtime validation on Pi 5 is a follow-up plan.

**Files:**
- Create: `kernels/q4k_dot_8x8_dual_arm.ea`
- Read (no edit): `kernels/q4k_dot_8x8_arm.ea` (structural template — the NEON mirror of `q4k_dot_8x8.ea`)

- [ ] **Step 1: eabrain lookup for ARM-specific intrinsics**

```bash
eabrain ref vld1q_u8
eabrain ref vdotq_s32
eabrain search q4k_dot_8x8_arm
```

Expected: entries for NEON intrinsics used in `q4k_dot_8x8_arm.ea`. If any intrinsic name is missing from eabrain, grep `$HOME/projects/eacompute/src/codegen/simd_arm*.rs` before concluding.

- [ ] **Step 2: Copy the ARM baseline**

```bash
cp kernels/q4k_dot_8x8_arm.ea kernels/q4k_dot_8x8_dual_arm.ea
```

- [ ] **Step 3: Update header + function signature**

Replace the header with a version that calls out "dual fusion, ARM NEON, mirror of `q4k_dot_8x8_dual.ea`." Function signature must be **byte-identical** to the x86 dual kernel (same arg names, same types, same order):

```ea
export func q4k_8x8_q8k_matvec_dual(
    packed_a: *restrict u8,
    packed_b: *restrict u8,
    q8_qs: *restrict i8,
    q8_d: *restrict f32,
    q8_bsums: *restrict i16,
    pow2: *restrict f32,
    scratch: *mut u8,
    out_a: *mut f32,
    out_b: *mut f32,
    n_rows: i32,
    n_cols: i32
) {
```

- [ ] **Step 4: Apply the same A/B split as Task 2**

The transformations from Task 2 (steps 4–9) apply mechanically to the ARM kernel too:
- `d16 → d16_a, d16_b`
- `out → out_a, out_b` (and `out_a_f32`, `out_b_f32`)
- `acc_row → acc_row_a, acc_row_b`
- `acc_min → acc_min_a, acc_min_b`
- `col_d → col_d_a, col_d_b`
- `col_dmin → col_dmin_a, col_dmin_b`
- `iacc_b → iacc_a, iacc_b_acc` (rename to avoid super-block `b` collision)
- `iacc_min_b → iacc_min_a, iacc_min_b_acc`
- Sub-block body: A-side packed loads/extracts/scale decode → shared Q8K loads/broadcasts → A-side dotprod → B-side packed loads/extracts/scale decode → B-side dotprod (reusing shared Q8 registers).
- Four FMAs (doubled), two stores (doubled).

The NEON intrinsic names differ from AVX2 (`vld1q_u8`, `vdotq_s32`, `vfmaq_f32`, etc.) but the control flow and the shared/weight-specific split are identical. Consult `kernels/q4k_dot_8x8_arm.ea` line-by-line for the template.

- [ ] **Step 5: Cross-compile for aarch64 via build.rs**

The native build is still x86 on this workstation, so `cargo build --release` compiles the new file only if `build.rs` targets ARM. **However**, `build.rs` filters `.ea` files by the `_arm` suffix + `#[cfg(aarch64)]` and only compiles ARM variants when `TARGET` is aarch64. On an x86 build, the ARM kernel is *skipped*, which means "compiles without error" is trivially true — there's no compilation for it to fail on.

To actually verify the NEON kernel typechecks, invoke the Ea compiler directly:

```bash
$HOME/projects/eacompute/target/release/ea \
    kernels/q4k_dot_8x8_dual_arm.ea \
    --lib --opt-level=3 \
    --target-triple=aarch64-unknown-linux-gnu \
    --target=cortex-a76 --dotprod \
    -o /tmp/libq4k_dot_8x8_dual_arm.so 2>&1 | tee /tmp/ea-arm.log
```

Expected: exit 0, no errors in `/tmp/ea-arm.log`. This is the Task 3 gate — it's the same command `build.rs` runs on an aarch64 target. If it fails, fix the kernel before proceeding.

- [ ] **Step 6: Line limit check**

```bash
wc -l kernels/q4k_dot_8x8_dual_arm.ea
```

Expected: ~380 lines, **must be <= 500**.

- [ ] **Step 7: Run Gates 1–5 (x86 build)**

All must pass. The x86 build ignores the ARM kernel.

- [ ] **Step 8: eabrain index**

```bash
eabrain index
```

- [ ] **Step 9: Commit**

```bash
git add kernels/q4k_dot_8x8_dual_arm.ea
git commit -m "feat(phase-b.2): ARM NEON q4k_8x8_q8k_matvec_dual kernel

Mirror of kernels/q4k_dot_8x8_dual.ea adapted to NEON intrinsics.
Same export signature, same shared/weight-specific split, same A/B
interleaved dot loop structure. Structurally cloned from
kernels/q4k_dot_8x8_arm.ea.

Cross-compile verified on this x86 workstation via direct Ea compiler
invocation (aarch64-unknown-linux-gnu + cortex-a76 + dotprod).
Runtime validation on the Pi 5 is a follow-up plan — out of scope
for B.2."
```

---

## Task 4: FFI binding + standalone bit-exact test (correctness gate)

**Goal:** Wire the new dual kernel into Rust via `ffi_inference.rs` and prove `to_bits()` equality against "two separate single-kernel calls." This is the gate for everything in Tasks 5–8 — commits that follow trust this test.

**Files:**
- Modify: `src/kernels/ffi_inference_types.rs` (+1 type)
- Modify: `src/kernels/ffi_inference.rs` (+1 field, +1 symbol load, +1 public wrapper)
- Create: `tests/dual_q4k_8x8.rs`

- [ ] **Step 1: Add the FFI type**

Open `src/kernels/ffi_inference_types.rs`. Add this type definition below the existing `Q4k8x8MatvecFn` (currently the last type in the file):

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

- [ ] **Step 2: Add the field to `KernelTableInference`**

In `src/kernels/ffi_inference.rs`, find the struct `KernelTableInference` (around line 8). Add a new field `q4k_8x8_q8k_matvec_dual` right after `q4k_8x8_q8k_matvec` at line 35:

```rust
    pub q4k_8x8_q8k_matvec:      Q4k8x8MatvecFn,
    pub q4k_8x8_q8k_matvec_dual: Q4k8x8MatvecDualFn,
```

- [ ] **Step 3: Add the library load**

Find the `load_inference_kernels` body. Around line 103, after `let q4k_dot_8x8_lib = load("q4k_dot_8x8")?;`, add:

```rust
    let q4k_dot_8x8_dual_lib = load("q4k_dot_8x8_dual")?;
```

- [ ] **Step 4: Add the symbol transmute**

Inside the `KernelTableInference { ... }` struct-literal (around line 140), after the `q4k_8x8_q8k_matvec:` line, add:

```rust
            q4k_8x8_q8k_matvec:      std::mem::transmute(sym(&q4k_dot_8x8_lib,      b"q4k_8x8_q8k_matvec\0")?),
            q4k_8x8_q8k_matvec_dual: std::mem::transmute(sym(&q4k_dot_8x8_dual_lib, b"q4k_8x8_q8k_matvec_dual\0")?),
```

And add the new library handle to the `libs` vec at line 141:

```rust
            libs: vec![q4kq, q4kd, q5kd, q6kd, f16_conv_lib, softmax_lib, gemma4_rmsnorm_lib, gemma4_gelu_lib, gemma4_rope_lib, bf16_matvec_lib, vec_ops_lib, attn_ops_lib, bare_rmsnorm_lib, softcap_lib, q4k_repack_lib, q4k_dot_8x8_lib, q4k_dot_8x8_dual_lib],
```

- [ ] **Step 5: Add the public wrapper**

At the end of `src/kernels/ffi_inference.rs` (after the existing `q4k_8x8_q8k_matvec` wrapper ending around line 299), add:

```rust
#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_8x8_q8k_matvec_dual(
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
) {
    (k().q4k_8x8_q8k_matvec_dual)(
        packed_a, packed_b, q8_qs, q8_d, q8_bsums, pow2, scratch,
        out_a, out_b, n_rows, n_cols,
    )
}
```

- [ ] **Step 6: Verify the FFI compiles**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
grep -E "^(warning|error)" /tmp/olorin-build.log
```

Expected: zero warnings, zero errors. The new wrapper + field will show as "unused" only if nothing references them — `tests/dual_q4k_8x8.rs` in the next step references the wrapper, so the warnings should be absent once the test file exists. Order these steps as: (a) build after FFI changes (expect "never used" warning on the new wrapper), (b) write the test file, (c) build again (warning gone).

Actually, to avoid the warning on step 6, add `#[allow(dead_code)]` temporarily? No — easier: skip the intermediate build and go straight to writing the test, then build once. Revising:

(Proceed to Step 7 without intermediate build.)

- [ ] **Step 7: Create the bit-exact test file**

Create `tests/dual_q4k_8x8.rs` with this full content:

```rust
//! Bit-exact regression for q4k_8x8_q8k_matvec_dual.
//!
//! Runs the dual kernel on (packed_gate, packed_up, q8k) and compares
//! each output f32 bit pattern against running the single kernel twice.
//! This is the correctness gate for Phase B.2 — subsequent commits trust
//! this equality.

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
fn dual_matches_two_single_calls_bitexact() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model at {}", model_path());
        return;
    }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    // ffn_gate + ffn_up from layer 0. Both Q4K in Q4_K_M, identical shape.
    let lw = &model.layers[0];
    assert_eq!(
        lw.w_gate_dtype,
        olorin::inference::matmul::GGML_TYPE_Q4_K,
        "test requires Q4K ffn_gate"
    );
    assert_eq!(
        lw.w_up_dtype,
        olorin::inference::matmul::GGML_TYPE_Q4_K,
        "test requires Q4K ffn_up"
    );

    let n_rows = model.ffn_dim[0];
    let n_cols = model.hidden_dim;
    let n_blocks = n_cols / 256;
    let tile_bytes = n_blocks * 1152; // 1152 B per 8-row tile group
    let n_tiles = n_rows / 8;
    let total = n_tiles * tile_bytes;

    let mut packed_gate = vec![0u8; total];
    let mut packed_up = vec![0u8; total];
    unsafe {
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.w_gate, packed_gate.as_mut_ptr(), n_rows as i32, n_cols as i32,
        );
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.w_up, packed_up.as_mut_ptr(), n_rows as i32, n_cols as i32,
        );
    }

    // Non-trivial Q8K input. Pattern from tests/repack_q4k.rs:
    // non-zero entries in every field, non-constant across blocks so the
    // compiler can't hoist anything constant.
    let mut q8_qs = vec![5i8; n_cols + 12];
    let mut q8_d = vec![0.01f32; n_blocks];
    let mut q8_bsums = vec![17i16; n_blocks * 16];
    for i in 0..n_cols {
        q8_qs[i] = ((i as i32) % 127 - 63) as i8;
    }
    for i in 0..n_blocks {
        q8_d[i] = 0.01 + (i as f32) * 0.0013;
    }
    for i in 0..(n_blocks * 16) {
        q8_bsums[i] = ((i as i16) % 31) - 15;
    }

    let pow2 = olorin::inference::matmul::pow2_table();

    // Reference: two separate calls to q4k_8x8_q8k_matvec.
    let mut ref_gate = vec![0f32; n_rows];
    let mut ref_up = vec![0f32; n_rows];
    let mut scratch_ref = [0u8; 128];
    unsafe {
        olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
            packed_gate.as_ptr(),
            q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
            pow2.as_ptr(), scratch_ref.as_mut_ptr(),
            ref_gate.as_mut_ptr(),
            n_rows as i32, n_cols as i32,
        );
        olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
            packed_up.as_ptr(),
            q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
            pow2.as_ptr(), scratch_ref.as_mut_ptr(),
            ref_up.as_mut_ptr(),
            n_rows as i32, n_cols as i32,
        );
    }

    // Candidate: one fused call.
    let mut fused_gate = vec![0f32; n_rows];
    let mut fused_up = vec![0f32; n_rows];
    let mut scratch_fused = [0u8; 128];
    unsafe {
        olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec_dual(
            packed_gate.as_ptr(), packed_up.as_ptr(),
            q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
            pow2.as_ptr(), scratch_fused.as_mut_ptr(),
            fused_gate.as_mut_ptr(), fused_up.as_mut_ptr(),
            n_rows as i32, n_cols as i32,
        );
    }

    // Per-output bit-exact equality on both channels.
    for i in 0..n_rows {
        assert_eq!(
            ref_gate[i].to_bits(),
            fused_gate[i].to_bits(),
            "gate[{i}]: ref={} fused={}",
            ref_gate[i], fused_gate[i],
        );
        assert_eq!(
            ref_up[i].to_bits(),
            fused_up[i].to_bits(),
            "up[{i}]: ref={} fused={}",
            ref_up[i], fused_up[i],
        );
    }
    eprintln!("PASS: n_rows={n_rows}, n_cols={n_cols}, bit-exact on both channels");
}
```

- [ ] **Step 8: Run the new test**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test dual_q4k_8x8 -- --nocapture --test-threads=1 2>&1 | tail -20
```

Expected: `PASS: n_rows=..., n_cols=..., bit-exact on both channels`. Test exit status 0.

**If any element mismatches**, the kernel's A-side or B-side accumulation order diverged from the single kernel. Compare the dual kernel line-by-line against `q4k_dot_8x8.ea` for the A-side and ensure the A-side code block is unchanged (other than `_a` renames). The single-kernel test in `tests/repack_q4k.rs::repack_q4k_matvec_roundtrip` provides the ground truth for A-side — if it was within tolerance there and fails here, the bug is in the interleaving of B-side integer work, not in the A-side math.

- [ ] **Step 9: Run all gates**

Gates 1 (build) + 2 (line limit) + 3 (repack_q4k) + 4 (parallel_regression) + 5 (smoke). All pass. The new FFI wrapper is now referenced by `tests/dual_q4k_8x8.rs`, so no dead-code warnings should remain.

- [ ] **Step 10: eabrain remember**

```bash
eabrain remember "Phase B.2 Task 4: q4k_8x8_q8k_matvec_dual kernel verified bit-exact (to_bits equality) against two separate q4k_8x8_q8k_matvec calls on ffn_gate+ffn_up of Gemma 4 E2B Q4_K_M layer 0. Test file: tests/dual_q4k_8x8.rs. Correctness gate for the rest of Phase B.2."
```

- [ ] **Step 11: Commit**

```bash
git add src/kernels/ffi_inference_types.rs src/kernels/ffi_inference.rs tests/dual_q4k_8x8.rs
git commit -m "feat(phase-b.2): FFI binding + bit-exact test for dual 8x8 matvec

Wires q4k_8x8_q8k_matvec_dual through ffi_inference:
  - Q4k8x8MatvecDualFn type in ffi_inference_types.rs
  - KernelTableInference field + symbol load from libq4k_dot_8x8_dual.so
  - Public unsafe wrapper q4k_8x8_q8k_matvec_dual

tests/dual_q4k_8x8.rs runs the fused kernel on (ffn_gate, ffn_up,
synthetic Q8K) and asserts to_bits() equality per-output element vs.
running the single 8x8 kernel twice. Correctness gate for the Path B
wire-up in Task 6 and the Path A retrofit in Task 7."
```

---

## Task 5: Path B — new work-stealing kernels in `matmul_graph.rs`

**Goal:** Add `q4k_matvec_8x8_ws` (single) and `q4k_matvec_dual_8x8_ws` (dual) to `matmul_graph.rs`. Both are dead code at the end of this task — they're wired into `forward_graph.rs` in Task 6.

**Files:**
- Modify: `src/inference/matmul_graph.rs` (currently 243 lines, budget +90 lines to ~335)
- Read (no edit): `src/inference/matmul_par.rs` lines 346–411 (`par_q4k_8x8_matvec` — structural reference for the single)
- Read (no edit): `src/inference/matmul_graph.rs` lines 16–111 (`q4k_matvec_ws` + `q4k_matvec_dual_ws` — work-stealing structural reference)

- [ ] **Step 1: Add `q4k_matvec_8x8_ws`**

Open `src/inference/matmul_graph.rs`. After the existing `q4k_matvec_dual_ws` function (ending around line 111), insert:

```rust
// ---------------------------------------------------------------------------
// Phase B.2: Q4K 8x8 repacked work-stealing matvec (single + dual)
// ---------------------------------------------------------------------------

/// Q4K 8×8 repacked matvec (single weight): work-stealing via atomic
/// current_chunk. Each chunk = one 8-row tile, one FFI call per chunk.
///
/// Requirements:
/// - `n_rows % 8 == 0`  (enforced by the repack gate in try_repack_q4k)
/// - `n_cols % 256 == 0`
/// - `packed` points to `(n_rows / 8) * n_blocks * 1152` bytes of repacked
///   weight, laid out per `q4k_repack_8x8`.
///
/// `current_chunk` must be reset to `nth` before calling (by the preceding
/// op, inside the graph-loop barrier lifecycle).
pub fn q4k_matvec_8x8_ws(
    packed: *const u8,
    q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    debug_assert!(n_rows % 8 == 0, "q4k_matvec_8x8_ws: n_rows must be multiple of 8");
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0, "q4k_matvec_8x8_ws: n_cols must be multiple of 256");
    let _ = nth; // unused; atomic counter carries the assignment state

    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let tile_bytes = n_blocks * 1152;
    let n_tiles = n_rows / 8;
    let pow2 = pow2_table();
    let mut scratch = [0u8; 128];

    let mut chunk = ith as i32;
    while (chunk as usize) < n_tiles {
        let tile = chunk as usize;
        unsafe {
            ffi_inference::q4k_8x8_q8k_matvec(
                packed.add(tile * tile_bytes),
                q8, q8_d, bsums,
                pow2.as_ptr(),
                scratch.as_mut_ptr(),
                output.add(tile * 8),
                8i32,
                n_cols as i32,
            );
        }
        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}
```

- [ ] **Step 2: Add `q4k_matvec_dual_8x8_ws`**

Immediately after `q4k_matvec_8x8_ws`, add:

```rust
/// Q4K 8×8 repacked fused dual matvec (gate + up): work-stealing.
/// Each chunk = one 8-row tile, one fused FFI call per chunk.
///
/// Processes both weight matrices against a shared Q8K input column in one
/// pass, reusing Q8 broadcasts across both streams. Bit-exact against two
/// separate `q4k_matvec_8x8_ws` calls per Task 4's test.
///
/// Requirements same as `q4k_matvec_8x8_ws`, applied to both gate_w and up_w.
/// Both weights must have identical (n_rows, n_cols) — the repack invariant
/// for ffn_gate/ffn_up on Gemma-family models guarantees this.
pub fn q4k_matvec_dual_8x8_ws(
    gate_w: *const u8, up_w: *const u8,
    q8: *const i8, q8_d: *const f32, bsums: *const i16,
    gate_out: *mut f32, up_out: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    debug_assert!(n_rows % 8 == 0, "q4k_matvec_dual_8x8_ws: n_rows must be multiple of 8");
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0, "q4k_matvec_dual_8x8_ws: n_cols must be multiple of 256");
    let _ = nth;

    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let tile_bytes = n_blocks * 1152;
    let n_tiles = n_rows / 8;
    let pow2 = pow2_table();
    let mut scratch = [0u8; 128];

    let mut chunk = ith as i32;
    while (chunk as usize) < n_tiles {
        let tile = chunk as usize;
        unsafe {
            ffi_inference::q4k_8x8_q8k_matvec_dual(
                gate_w.add(tile * tile_bytes),
                up_w.add(tile * tile_bytes),
                q8, q8_d, bsums,
                pow2.as_ptr(),
                scratch.as_mut_ptr(),
                gate_out.add(tile * 8),
                up_out.add(tile * 8),
                8i32,
                n_cols as i32,
            );
        }
        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}
```

- [ ] **Step 3: Verify imports**

The top of `matmul_graph.rs` currently imports:

```rust
use std::sync::atomic::{AtomicI32, Ordering};
use crate::kernels::ffi_inference;
use super::matmul::*;
```

`super::matmul::*` re-exports `Q4K_BLOCK_SIZE` and `pow2_table` (verified: the existing `q4k_matvec_ws` uses both via the glob). No new imports needed.

- [ ] **Step 4: Build clean**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
grep -E "^(warning|error)" /tmp/olorin-build.log
```

Expected: one "never used" warning for each new function (`q4k_matvec_8x8_ws`, `q4k_matvec_dual_8x8_ws`) — they're dead code until Task 6. To keep Gate 1's "zero warnings" rule, add `#[allow(dead_code)]` on both functions temporarily.

Revise Step 1's function opening:

```rust
#[allow(dead_code)] // wired into forward_graph.rs in Task 6
pub fn q4k_matvec_8x8_ws(
```

And Step 2's:

```rust
#[allow(dead_code)] // wired into forward_graph.rs in Task 6
pub fn q4k_matvec_dual_8x8_ws(
```

After the revise, re-run the build. Expected: zero warnings.

- [ ] **Step 5: Line limit check**

```bash
wc -l src/inference/matmul_graph.rs
```

Expected: ~335 lines, <= 500.

- [ ] **Step 6: Run all gates**

Gates 1–5. All pass. New functions are unused but present.

- [ ] **Step 7: Commit**

```bash
git add src/inference/matmul_graph.rs
git commit -m "feat(phase-b.2): work-stealing 8x8 matvec variants for Path B

Adds q4k_matvec_8x8_ws and q4k_matvec_dual_8x8_ws to matmul_graph.rs.
Both use the atomic current_chunk work-stealing pattern with chunk=1
tile (8 rows), stack-allocated 128-byte scratch, and no remainder
handling (the repack gate ensures n_rows % 8 == 0).

Dead code at this commit — wired into forward_graph.rs in the next
task. #[allow(dead_code)] carries the unused-function warning until
then."
```

---

## Task 6: Path B — wire the new work-stealing kernels into `forward_graph.rs`

**Goal:** Route 7 of 9 `matvec_ws` / `q4k_matvec_dual_ws` call sites in `forward_graph.rs` through the new 8×8 work-stealing kernels, via a new `matvec_step` helper for the single-matvec pattern. After this task, production decode via `forward_one_graph` uses the repacked path on every Q4K weight where it's supported.

**Files:**
- Modify: `src/inference/forward_graph.rs` (currently 442 lines)

**Call site inventory** (from the current source — verify before editing):

| # | Line | Kind | Weight | Where |
|---|------|------|--------|-------|
| 1 | 85   | single | `model.embed_weight` | output logits (Q6K, no repack possible — **stays as-is**) |
| 2 | 143  | single | `lw.wq` | Wq projection |
| 3 | 182  | single | `lw.wk` | Wk projection (if `has_kv`) |
| 4 | 194  | single | `lw.wv` | Wv projection (if `has_kv`) |
| 5 | 288  | single | `lw.wo` | Wo projection |
| 6 | 327  | dual   | `lw.w_gate` + `lw.w_up` | FFN gate+up fused dispatch |
| 7 | 338  | single | `lw.w_gate` | FFN gate fallback (else branch, non-Q4K) |
| 8 | 347  | single | `lw.w_up` | FFN up fallback (else branch, non-Q4K) |
| 9 | 370  | single | `lw.w_down` | FFN down projection |

Sites 2, 3, 4, 5, 7, 8, 9 are single matvecs on per-layer Q4K-eligible weights → route through `matvec_step`. Site 1 is the embed matmul (Q6K, never repacked) → stays as direct `matvec_ws`. Site 6 is the dual dispatch → inline match on the repacked pair.

- [ ] **Step 1: Add the `matvec_step` private helper**

Open `src/inference/forward_graph.rs`. After the `unsafe impl Sync for FwdCtx<'a> {}` line (around line 31) but before `pub(crate) fn forward_one_inner`, insert:

```rust
/// Dispatch a single matvec_ws call through either the repacked 8x8 path
/// or the standard 4-row matvec_ws fallback, depending on whether the
/// weight has been repacked at model load time.
///
/// Not `unsafe fn` — `matmul_graph::q4k_matvec_8x8_ws` and
/// `matmul_graph::matvec_ws` are both safe public fns that take raw
/// pointers, matching the calling style used elsewhere in this file.
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

- [ ] **Step 2: Rewire Wq (site 2, line 143)**

Replace the existing block:

```rust
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matmul_graph::matvec_ws(
        lw.wq_dtype, lw.wq,
        state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
        state.q.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        n_heads * head_dim, hd,
        current_chunk, ith, nth,
    );
    barrier.wait();
```

with:

```rust
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matvec_step(
        lw.wq_dtype, lw.wq, lw.wq_repacked.as_deref(),
        state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
        state.q.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        n_heads * head_dim, hd,
        current_chunk, ith, nth,
    );
    barrier.wait();
```

- [ ] **Step 3: Rewire Wk (site 3, line 182)**

Replace:

```rust
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matmul_graph::matvec_ws(
            lw.wk_dtype, lw.wk,
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.k.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            kv_dim, hd,
            current_chunk, ith, nth,
        );
        barrier.wait();
```

with:

```rust
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matvec_step(
            lw.wk_dtype, lw.wk, lw.wk_repacked.as_deref(),
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.k.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            kv_dim, hd,
            current_chunk, ith, nth,
        );
        barrier.wait();
```

- [ ] **Step 4: Rewire Wv (site 4, line 194)**

Same pattern, substituting `lw.wv_dtype`, `lw.wv`, `lw.wv_repacked.as_deref()`, `state.v.as_mut_ptr()`, `kv_dim_v`.

- [ ] **Step 5: Rewire Wo (site 5, line 288)**

Replace:

```rust
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matmul_graph::matvec_ws(
        lw.wo_dtype, lw.wo,
        state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
        state.wo_out.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, n_heads * head_dim,
        current_chunk, ith, nth,
    );
    barrier.wait();
```

with:

```rust
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matvec_step(
        lw.wo_dtype, lw.wo, lw.wo_repacked.as_deref(),
        state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
        state.wo_out.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, n_heads * head_dim,
        current_chunk, ith, nth,
    );
    barrier.wait();
```

- [ ] **Step 6: Rewire the dual FFN gate+up site (site 6, lines 322–334)**

Replace the existing `if lw.w_gate_dtype == matmul::GGML_TYPE_Q4_K && lw.w_up_dtype == matmul::GGML_TYPE_Q4_K { ... }` body:

```rust
    if lw.w_gate_dtype == matmul::GGML_TYPE_Q4_K && lw.w_up_dtype == matmul::GGML_TYPE_Q4_K {
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matmul_graph::q4k_matvec_dual_ws(
            lw.w_gate, lw.w_up,
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.gate.as_mut_ptr(), state.up.as_mut_ptr(),
            ffn_dim, hd,
            current_chunk, ith, nth,
        );
        barrier.wait();
    } else {
```

with:

```rust
    if lw.w_gate_dtype == matmul::GGML_TYPE_Q4_K && lw.w_up_dtype == matmul::GGML_TYPE_Q4_K {
        debug_assert!(
            lw.w_gate_repacked.is_some() == lw.w_up_repacked.is_some(),
            "ffn_gate/ffn_up repack invariant violated in layer {il}"
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
    } else {
```

- [ ] **Step 7: Rewire the gate fallback (site 7, line 338)**

Inside the `else` branch (non-Q4K gate/up path), replace the first `matvec_ws` call:

```rust
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matmul_graph::matvec_ws(
            lw.w_gate_dtype, lw.w_gate,
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.gate.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            ffn_dim, hd, current_chunk, ith, nth,
        );
        barrier.wait();
```

with:

```rust
        current_chunk.store(nth as i32, Ordering::Relaxed);
        barrier.wait();
        matvec_step(
            lw.w_gate_dtype, lw.w_gate, lw.w_gate_repacked.as_deref(),
            state.q8_qs.as_ptr(), state.q8_d.as_ptr(), state.q8_bsums.as_ptr(),
            state.gate.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
            ffn_dim, hd, current_chunk, ith, nth,
        );
        barrier.wait();
```

- [ ] **Step 8: Rewire the up fallback (site 8, line 347)**

Same transformation for `lw.w_up_dtype`, `lw.w_up`, `lw.w_up_repacked.as_deref()`, `state.up.as_mut_ptr()`.

- [ ] **Step 9: Rewire w_down (site 9, line 370)**

Replace:

```rust
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matmul_graph::matvec_ws(
        lw.w_down_dtype, lw.w_down,
        state.ffn_q8_qs.as_ptr(), state.ffn_q8_d.as_ptr(), state.ffn_q8_bsums.as_ptr(),
        state.down.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, ffn_dim,
        current_chunk, ith, nth,
    );
    barrier.wait();
```

with:

```rust
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matvec_step(
        lw.w_down_dtype, lw.w_down, lw.w_down_repacked.as_deref(),
        state.ffn_q8_qs.as_ptr(), state.ffn_q8_d.as_ptr(), state.ffn_q8_bsums.as_ptr(),
        state.down.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, ffn_dim,
        current_chunk, ith, nth,
    );
    barrier.wait();
```

- [ ] **Step 10: Do NOT touch site 1 (output logits, line 85)**

The output matmul uses `model.embed_weight` (Q6K in Gemma 4), which is **not** a per-layer weight and has no `_repacked` buffer. Leave that call site unchanged. Routing it through `matvec_step` with `repacked = None` would work but adds a layer of indirection for no benefit.

- [ ] **Step 11: Remove the `#[allow(dead_code)]` from Task 5**

Now that `q4k_matvec_8x8_ws` and `q4k_matvec_dual_8x8_ws` have live callers, remove the `#[allow(dead_code)]` attributes added in Task 5 Step 4 from `src/inference/matmul_graph.rs`.

- [ ] **Step 12: Build clean**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
grep -E "^(warning|error)" /tmp/olorin-build.log
```

Expected: zero warnings, zero errors.

- [ ] **Step 13: Line limit check**

```bash
wc -l src/inference/forward_graph.rs src/inference/matmul_graph.rs
```

Expected: `forward_graph.rs` ~460 lines, `matmul_graph.rs` ~335 lines. Both <= 500.

- [ ] **Step 14: Run all gates**

- Gate 1 (build): pass.
- Gate 2 (line limit): pass.
- Gate 3 (`repack_q4k`): pass.
- Gate 4 (`gemma4_parallel_regression`): **pass** — Path A is not touched by this task, so the snapshot is still valid.
- Gate 5 (`gemma4_smoke`): **pass** — production decode via `forward_one_graph` now uses the repacked path on every eligible Q4K matmul + the fused dual on ffn_gate/ffn_up, and the end-to-end sentence completion should still be coherent.

**If Gate 5 fails**, the likely cause is a mismatched pointer shape in one of the call-site rewrites. The `dual_q4k_8x8` test from Task 4 already proves the kernel is right, so a failure here is a wiring bug. Re-check that every `matvec_step` call passes the correct `n_rows`/`n_cols` pair, and that the dual site passes `ffn_dim, hd` in the same order both branches.

- [ ] **Step 15: Run `gemma4_verify` step5_logits explicitly**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -20
```

Expected: all steps pass, L2 norms within existing tolerance. This cross-checks that layer-by-layer logits haven't drifted outside the documented bar.

- [ ] **Step 16: eabrain remember**

```bash
eabrain remember "Phase B.2 Task 6: production forward_one_graph now uses the 8x8 repacked path on wq/wk/wv/wo/w_gate/w_up/w_down via new matvec_step helper in forward_graph.rs and new q4k_matvec_8x8_ws + q4k_matvec_dual_8x8_ws in matmul_graph.rs. ffn_gate+ffn_up go through the fused dual kernel. Path A (forward_one) still uses the old 'two separate 8x8 calls' for the dual case — retrofit in Task 7."
```

- [ ] **Step 17: Commit**

```bash
git add src/inference/forward_graph.rs src/inference/matmul_graph.rs
git commit -m "feat(phase-b.2): wire Path B forward_graph through 8x8 repacked kernels

Adds a private matvec_step helper to forward_graph.rs that dispatches
to q4k_matvec_8x8_ws when the weight has been repacked at load time,
falling through to the standard matvec_ws otherwise. Routes 7 single
matvec call sites (Wq/Wk/Wv/Wo/w_gate fallback/w_up fallback/w_down)
through the helper.

The ffn_gate+ffn_up dual dispatch in layer_forward_graph now branches
on the repacked pair: both Some -> q4k_matvec_dual_8x8_ws (fused),
otherwise -> existing q4k_matvec_dual_ws. Debug-asserts the repack
invariant (both repack together or neither does).

Production decode via forward_one_graph + generate.rs now uses the
repacked path on every eligible Q4K matmul. Path A (forward_one,
gemma4_parallel_regression) untouched — snapshot stays valid. Path A
retrofit with fused dual kernel is in Task 7."
```

---

## Task 7: Path A retrofit — fused dual in `matmul_par.rs` + `matmul.rs`

**Goal:** Retrofit Path A (`par_q4k_matvec_dual_maybe_repacked` in `matmul.rs`) to use the new fused dual kernel instead of two separate 8×8 calls. Add `par_q4k_8x8_matvec_dual` to `matmul_par.rs`. Collapses the 4-case match to 2 cases + `debug_assert!`. **This task intentionally breaks `gemma4_parallel_regression`** (~ULP snapshot drift); Task 8 regenerates the snapshot.

**Files:**
- Modify: `src/inference/matmul_par.rs` (+50 lines, to ~461)
- Modify: `src/inference/matmul.rs` (net −13 lines on the dual wrapper body)

- [ ] **Step 1: Add `par_q4k_8x8_matvec_dual` to `matmul_par.rs`**

Open `src/inference/matmul_par.rs`. Immediately after the existing `par_q4k_8x8_matvec` function (ends around line 411), add:

```rust
/// Phase B.2: Parallel fused dual Q4K 8×8 matvec on Path A.
///
/// Mirrors `par_q4k_8x8_matvec` structure: tile-slice `n_tiles` across
/// pool workers, each thread calls `q4k_8x8_q8k_matvec_dual` on its
/// slice with its own stack-allocated 128-byte scratch. Writes into both
/// output slices. Bit-exact against two separate `par_q4k_8x8_matvec`
/// calls per Task 4's standalone test.
///
/// Requirements:
/// - `n_rows % 8 == 0`
/// - `n_cols % 256 == 0`
/// - `packed_a`, `packed_b` each point to `(n_rows / 8) * n_blocks * 1152`
///   bytes, identical shape.
#[allow(clippy::too_many_arguments)]
pub(super) fn par_q4k_8x8_matvec_dual(
    pool: &ThreadPool,
    packed_a: *const u8,
    packed_b: *const u8,
    input_qs: &[i8], input_d: &[f32], input_bsums: &[i16],
    output_a: &mut [f32],
    output_b: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    debug_assert!(n_rows % 8 == 0, "par_q4k_8x8_matvec_dual: n_rows must be multiple of 8");
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0, "par_q4k_8x8_matvec_dual: n_cols must be multiple of 256");

    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let tile_bytes = n_blocks * 1152;
    let n_tiles = n_rows / 8;
    let n_threads = pool.thread_count().min(n_tiles.max(1));

    let pow2 = pow2_table();

    // Single-thread fast path
    if n_threads <= 1 {
        let mut scratch = [0u8; 128];
        unsafe {
            ffi_inference::q4k_8x8_q8k_matvec_dual(
                packed_a, packed_b,
                input_qs.as_ptr(),
                input_d.as_ptr(),
                input_bsums.as_ptr(),
                pow2.as_ptr(),
                scratch.as_mut_ptr(),
                output_a.as_mut_ptr(),
                output_b.as_mut_ptr(),
                n_rows as i32,
                n_cols as i32,
            );
        }
        return;
    }

    // Multi-thread: slice n_tiles across n_threads.
    let q8 = SendPtr(input_qs.as_ptr());
    let bsums = SendPtr(input_bsums.as_ptr());
    let q8_d = SendPtr(input_d.as_ptr());
    let pow2_ptr = SendPtr(pow2.as_ptr());
    let out_a_ptr = SendMutPtr(output_a.as_mut_ptr());
    let out_b_ptr = SendMutPtr(output_b.as_mut_ptr());
    let wa = SendPtr(packed_a);
    let wb = SendPtr(packed_b);

    pool.run(n_threads, move |tid, nt| {
        let start_tile = tid * n_tiles / nt;
        let end_tile = (tid + 1) * n_tiles / nt;
        let tile_count = end_tile - start_tile;
        if tile_count == 0 { return; }
        let slice_rows = tile_count * 8;
        let mut scratch = [0u8; 128];
        unsafe {
            ffi_inference::q4k_8x8_q8k_matvec_dual(
                wa.add(start_tile * tile_bytes),
                wb.add(start_tile * tile_bytes),
                q8.ptr(),
                q8_d.ptr(),
                bsums.ptr(),
                pow2_ptr.ptr(),
                scratch.as_mut_ptr(),
                out_a_ptr.add(start_tile * 8),
                out_b_ptr.add(start_tile * 8),
                slice_rows as i32,
                n_cols as i32,
            );
        }
    });
}
```

- [ ] **Step 2: Rewrite `par_q4k_matvec_dual_maybe_repacked` in `matmul.rs`**

Open `src/inference/matmul.rs`. Find the existing `par_q4k_matvec_dual_maybe_repacked` function (currently at lines 227–267, with a 4-case match). Replace the whole function **and** the "Phase B.1 dispatch wrappers" comment block above it (lines 227–267) with:

```rust
/// Parallel Q4K dual (gate+up) matvec with optional repacked buffers.
///
/// Phase B.2: when both gate and up are repacked, routes through the
/// fused `par_q4k_8x8_matvec_dual`, which shares Q8 input loads + v**
/// broadcasts across both weight streams. When neither is repacked,
/// falls through to the existing `par_q4k_matvec_dual` (4-row kernel).
///
/// The (Some, None) / (None, Some) cases are unreachable on
/// Gemma-family models: `populate_q4k_repacked` in engine_helpers.rs
/// repacks ffn_gate and ffn_up with the same (ffn_dim, hidden_dim)
/// gate + dtype, so they're always both repacked or both not. A
/// debug_assert catches future models that break the invariant.
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

Also delete the preamble comment block at lines 227–234 (the old "Phase B.1: dispatch wrappers" header + the "// - Both repacked: two separate 8x8 calls (loses dual's Q8-input-sharing bandwidth win, gains 8x8 throughput; net TBD, measure at Phase B.3)." comment).

**Note:** `par_q4k_matvec_dual` (the non-`_maybe_repacked` variant) is the pre-existing 4-row fallback. It lives in `matmul_par.rs`. Do not remove it — it's still used for the `(None, None)` fall-through.

- [ ] **Step 3: Add the new fn to the import list in `matmul.rs`**

Near the top of `matmul.rs`, find the `use super::matmul_par::{...}` statement (or equivalent) that currently imports `par_q4k_8x8_matvec`. Add `par_q4k_8x8_matvec_dual` to the import list. If the existing `par_matvec_maybe_repacked` was already importing from `matmul_par`, extend that; otherwise add a new import line consistent with existing style.

- [ ] **Step 4: Build clean**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tee /tmp/olorin-build.log
grep -E "^(warning|error)" /tmp/olorin-build.log
```

Expected: zero warnings, zero errors.

- [ ] **Step 5: Line limit check**

```bash
wc -l src/inference/matmul.rs src/inference/matmul_par.rs
```

Expected: `matmul.rs` smaller by ~13 lines (4-case → 2-case), `matmul_par.rs` larger by ~50 lines (new `par_q4k_8x8_matvec_dual`). Both <= 500.

- [ ] **Step 6: Run Gates 1, 2, 3**

All pass.

- [ ] **Step 7: Run Gate 4 (parallel_regression)**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression 2>&1 | tail -20
```

**Expected: FAIL.** The single test in that suite (`forward_one_bos_logits_bit_exact`) compares `forward_one`'s output against `tests/snapshots/gemma4_logits_bos.bin`. Path A's dual path now goes through the fused 8x8 kernel, which accumulates in a different f32 order than "two separate 8x8 calls" (per-output `to_bits()` drift is ≤ 1 ULP because the integer reductions are identical but the outer `acc_row_a += fma(...); acc_row_b += fma(...)` interleaving is new). The snapshot binary is byte-wise stale.

**Verify the failure mode** before continuing: the only failing test must be the snapshot comparison, and the reported diff should be small-magnitude floating point drift (not structural, not NaN, not large). If the test fails for any other reason — e.g., n_rows mismatch, panic, NaN output — **stop**. That indicates a wiring bug in the Path A retrofit, not the expected snapshot drift. Diagnose and fix before proceeding to Task 8.

Useful diagnostic: re-run `cargo test --release --test dual_q4k_8x8` — if that test still passes (`to_bits()` equality, synthetic Q8K), the kernel is fine and the issue is Path A dispatch glue. If that test also fails, the retrofit broke the kernel.

- [ ] **Step 8: Run Gate 5 (smoke) + `gemma4_verify`**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_smoke 2>&1 | tail -10
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -20
```

Both must pass. `gemma4_smoke` runs production decode (Path B) end-to-end and should be unaffected. `gemma4_verify` runs layer-by-layer L2-norm comparisons (Path A, but tolerance-based not bit-exact) and should stay within the existing tolerance bar.

- [ ] **Step 9: Do NOT commit yet**

Task 7's code changes and Task 8's snapshot regeneration ship as a single commit so that reviewers see the cause and the fix together, and so that bisection across the branch lands on a green tree at every intermediate point.

Leave the working tree dirty. Proceed to Task 8.

---

## Task 8: Snapshot regeneration + performance gate + commit

**Goal:** Regenerate the `gemma4_parallel_regression` snapshot, confirm all tests are green, measure decode throughput, and land Task 7 + Task 8 as a single commit with the new bench numbers in the message.

**Files:**
- Modify: `tests/snapshots/gemma4_logits_bos.bin` (byte-replace)

- [ ] **Step 1: Identify the regeneration entry point**

Grep to find how the snapshot was produced:

```bash
grep -rn "gemma4_logits_bos.bin\|regenerate\|OLORIN_REGEN" tests/ src/ | head -20
```

Expected: `tests/gemma4_parallel_regression.rs` contains the test `forward_one_bos_logits_bit_exact` that reads the snapshot. It may have an env-var gate for regeneration (e.g. `OLORIN_REGEN_SNAPSHOT=1`), or a separate `#[ignore]` test that writes the file. Find whichever mechanism B.1 used when it regenerated this same snapshot on commit `bb71265`.

- [ ] **Step 2: Run the regeneration step**

Use the mechanism found in Step 1. Most common patterns:

If an env-var gate exists:

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" OLORIN_REGEN_SNAPSHOT=1 \
    cargo test --release --test gemma4_parallel_regression -- --nocapture 2>&1 | tail -20
```

If a separate ignored test exists:

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" \
    cargo test --release --test gemma4_parallel_regression regenerate -- --nocapture --ignored 2>&1 | tail -20
```

If there is **no** regeneration mechanism in the test file, add one based on B.1's pattern — look at git log for `tests/gemma4_parallel_regression.rs` in commit `bb71265`:

```bash
git show bb71265 -- tests/gemma4_parallel_regression.rs | head -80
```

If the regeneration code was removed after the B.1 snapshot landed (i.e., it was a one-shot), re-add it now in the same shape. Do not invent a new mechanism — mirror whatever B.1 used.

- [ ] **Step 3: Verify the snapshot regenerated**

```bash
git status tests/snapshots/gemma4_logits_bos.bin
git diff --stat tests/snapshots/gemma4_logits_bos.bin
```

Expected: the file is modified (shown in `git status`), and the binary diff is non-zero but small (not a structural rewrite — same number of bytes).

- [ ] **Step 4: Re-run Gate 4 without regeneration**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Run all Gates 1–5**

All five pass.

- [ ] **Step 6: Run the performance bench**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release \
    --test bench_decode_speed -- --nocapture --test-threads=1 2>&1 | tee /tmp/b2-bench.log
```

Look at the output for the decode tok/s line. Compare against the B.1 baseline recorded in memory (`~8.70 tok/s olorin1 vs. 8.9 tok/s llama.cpp` per the 2026-04-11 eabrain note).

**Hard requirement:** decode tok/s must **move upward** vs. B.1.

**If it moves upward:** record the new number for the commit message. Proceed to Step 8.

**If it stays flat or regresses:** **stop** — do not commit. The correctness gates have passed (the kernel is right), so the problem is in wiring or scheduling, not in the kernel math. Run the diagnostic steps from Task 7 Step 7's "verify failure mode" section:

- Profile with `perf stat -e L1-dcache-loads,L1-dcache-load-misses -e cycles,instructions` on a single-thread decode run to check if L1 dcache loads dropped at the gate+up step.
- Check whether atomic overhead dominates at chunk=1 tile by temporarily doubling the chunk step to 2 tiles in both `q4k_matvec_8x8_ws` and `q4k_matvec_dual_8x8_ws` — retry bench.
- Inspect the compiled assembly for `libq4k_dot_8x8_dual.so` for register spills of `v00..v11` — the fusion win depends on them staying live across both streams.

None of these are in-scope for this plan as an execution step; they're diagnostic paths to file a Phase B.3 follow-up if the bench doesn't move. A bench regression is a hard stop; a bench that stays flat is grounds for a conversation with the plan owner about whether to land as-is or defer.

- [ ] **Step 7: Run the full regression sweep**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release \
    --test repack_q4k \
    --test dual_q4k_8x8 \
    --test gemma4_verify \
    --test gemma4_parallel_regression \
    --test gemma4_smoke \
    -- --test-threads=1 2>&1 | tail -40
```

Expected: all 5 suites green.

- [ ] **Step 8: eabrain remember the bench result**

```bash
eabrain remember "Phase B.2 complete (Task 8): decode tok/s on Gemma 4 E2B Q4_K_M, 16 threads, this workstation = <NEW_NUMBER> (vs. B.1 baseline ~8.70). Production forward_one_graph now uses q4k_8x8_q8k_matvec + q4k_8x8_q8k_matvec_dual via matvec_step helper + q4k_matvec_8x8_ws / q4k_matvec_dual_8x8_ws in matmul_graph.rs. Path A retrofitted to match (par_q4k_matvec_dual_maybe_repacked now 2-case + debug_assert). Snapshot regenerated. All 5 test suites green. Next: Phase 2 plan (batched prompt eval, still unwritten) for the q4k_8x8_q8k_gemm kernel that closes the prefill gap."
```

Substitute `<NEW_NUMBER>` with the actual measured tok/s from Step 6.

- [ ] **Step 9: Final commit**

Stage every file touched by Tasks 7 and 8 and commit as one:

```bash
git add src/inference/matmul_par.rs \
        src/inference/matmul.rs \
        tests/snapshots/gemma4_logits_bos.bin
git commit -m "$(cat <<'EOF'
feat(phase-b.2): retrofit Path A dual matmul to fused 8x8 + regen snapshot

Replaces Phase B.1's "two separate 8x8 calls" compromise on Path A with
the new par_q4k_8x8_matvec_dual wrapper (matmul_par.rs), which routes
ffn_gate + ffn_up through the fused q4k_8x8_q8k_matvec_dual kernel in
a single pool.run dispatch. Collapses par_q4k_matvec_dual_maybe_repacked
from 4 cases to 2 + debug_assert — the (Some, None) / (None, Some)
"hybrid dual" cases are unreachable on Gemma-family models because
populate_q4k_repacked gates gate and up on the same dims + dtype.

Deletes the "measure at Phase B.3" TODO block from matmul.rs.

Snapshot regeneration: the 8x8 accumulation order for dual gate+up
differs from "two separate 8x8 calls" by <= 1 ULP because the outer
f32 FMA chain interleaves differently even though the per-row integer
reduction is identical. forward_one_bos_logits_bit_exact's snapshot
byte-drifts accordingly. Regenerated tests/snapshots/gemma4_logits_bos.bin.

Verification:
  - tests/dual_q4k_8x8 (Task 4 correctness gate): PASS
  - tests/repack_q4k: PASS
  - tests/gemma4_parallel_regression (post-regen): PASS
  - tests/gemma4_verify step5_logits: PASS (within L2 tolerance)
  - tests/gemma4_smoke: PASS

Performance (decode, Gemma 4 E2B Q4_K_M, 16 threads, this workstation):
  B.1 baseline: ~8.70 tok/s
  B.2:          <NEW_NUMBER> tok/s

Path A and Path B now use identical kernels. Phase B.1's deferred
"measure at B.3" TODO is deleted, not updated.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Substitute `<NEW_NUMBER>` with the tok/s value from Step 6 before pasting the heredoc.

- [ ] **Step 10: Verify the branch state**

```bash
git log --oneline -10
git status
```

Expected: 8 new commits (Tasks 1–8) on top of `bb71265`, clean working tree (modulo the pre-existing untracked junk from the branch state). The branch is ready to push.

---

## Self-Review Checklist (run after completing all 8 tasks)

Before considering B.2 done:

- [ ] Every `- [ ]` checkbox in this plan is ticked.
- [ ] `kernels/q4k_dot_8x8_dual.ea` exists, ~350 lines, < 500.
- [ ] `kernels/q4k_dot_8x8_dual_arm.ea` exists, ~380 lines, < 500.
- [ ] `tests/dual_q4k_8x8.rs` exists and passes.
- [ ] `src/kernels/ffi_inference_types.rs` has `Q4k8x8MatvecDualFn`.
- [ ] `src/kernels/ffi_inference.rs` has the field, symbol load, and public wrapper.
- [ ] `src/inference/matmul_graph.rs` has `q4k_matvec_8x8_ws` and `q4k_matvec_dual_8x8_ws`, both actively called.
- [ ] `src/inference/forward_graph.rs` has the private `matvec_step` helper and uses it at sites 2, 3, 4, 5, 7, 8, 9. Site 1 (embed) stays as `matvec_ws`. Site 6 (dual) uses the inline match on `(gate_repacked, up_repacked)`.
- [ ] `src/inference/matmul_par.rs` has `par_q4k_8x8_matvec_dual`.
- [ ] `src/inference/matmul.rs`'s `par_q4k_matvec_dual_maybe_repacked` is the 2-case + debug_assert version. The "Phase B.3" TODO comment is gone.
- [ ] `tests/snapshots/gemma4_logits_bos.bin` is updated.
- [ ] All 5 gates (build, line limit, repack_q4k, gemma4_parallel_regression, gemma4_smoke) pass on the final commit.
- [ ] `bench_decode_speed` shows tok/s improvement vs. B.1 baseline, recorded in the Task 8 commit message.
- [ ] No file > 500 lines.
- [ ] No `TODO`, `HACK`, `FIXME`, `for now` markers introduced in new code.
- [ ] `eabrain remember` entries exist for Tasks 4, 6, and 8.
- [ ] Branch is 8 commits ahead of `bb71265`.

## What's next after this plan

- **Phase 2** — batched prompt eval. Still unwritten. Needs a new plan file at `docs/superpowers/plans/2026-0X-XX-phase-2-batched-prompt-eval.md` whose sole deliverable is a `q4k_8x8_q8k_gemm` Ea kernel + a `forward_batch` path that consumes it. Phase B.2 unblocks this because all the repack/matvec/FFI/engine-load plumbing is done — Phase 2 just authors one kernel and wires it.
- **Phase B.3** — optional: re-measure whether selectively skipping the 8x8 repack on small weights (wk, wv) is worth it on Pi 5. Not a speedup plan, a measurement plan. Lowest priority.
- **Phase 3** — flash attention / online softmax. Future.

Do NOT start any of the above during this plan. Their preconditions are all "B.2 landed."
