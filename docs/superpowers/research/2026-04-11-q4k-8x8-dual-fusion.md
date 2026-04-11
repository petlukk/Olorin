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
  matrix-local `iacc_mat`, `iacc_min_mat`.

## Weight-specific per super-block

- `col_d`, `col_dmin` f16→f32 loads (16 bytes from the packed header).
- One FMA into `acc_row`, one FMA into `acc_min`.

## Correctness claim for the dual kernel

Per output element, the integer reduction order and the f32 FMA chain
on `acc_row_mat_a` are **identical** to calling `q4k_8x8_q8k_matvec`
once on `packed_a` alone: interleaving B-side integer work inside the
same `sb` loop does not touch A's accumulator lanes, and rows are
independent. The per-output `to_bits()` equality holds against
"two separate calls to the single kernel."

**Consequence for the scratch:** one shared 128-byte scratch is enough
for the dual kernel; the bsums hadd only depends on Q8K input.

## Naming convention for the dual kernel

The single kernel uses `iacc_b` as the per-super-block integer
accumulator — the `_b` in `iacc_b` originally meant "per-b-super-block,"
not "matrix B." To avoid confusion when adding a second weight matrix,
this note and the dual kernel use suffix `_mat_a` / `_mat_b` for the
per-matrix accumulators:

- `iacc_mat_a` / `iacc_mat_b` — per-super-block i32x8 integer accumulators
- `iacc_min_mat_a` / `iacc_min_mat_b` — per-super-block mins accumulators
- `acc_row_mat_a` / `acc_row_mat_b` — per-tile f32x8 row accumulators
- `acc_min_mat_a` / `acc_min_mat_b` — per-tile f32x8 mins accumulators
- `col_d_mat_a` / `col_d_mat_b` — per-super-block col scales
- `col_dmin_mat_a` / `col_dmin_mat_b` — per-super-block col mins
- `d16_mat_a` / `d16_mat_b` — packed-header i16 bases
- `out_mat_a_f32` / `out_mat_b_f32` — output f32 bases
- `sp_mat_a` / `sp_mat_b` — packed-header i32 bases for utmp decode

The super-block loop counter stays as `b` (matches the single kernel).
The sub-block loop counter stays as `sb`.

## Consequence for the dual kernel body

```
while x < n_rows / 8:
    acc_row_mat_a, acc_min_mat_a = 0, 0
    acc_row_mat_b, acc_min_mat_b = 0, 0
    while b < nb:
        row_sc = splat(q8_d[b])                               # SHARED
        col_d_mat_a, col_dmin_mat_a = from packed_a[b]        # A-specific
        col_d_mat_b, col_dmin_mat_b = from packed_b[b]        # B-specific
        iacc_mat_a, iacc_min_mat_a = 0, 0
        iacc_mat_b, iacc_min_mat_b = 0, 0
        q8s_half = hadd_i16(...)                              # SHARED
        store(scratch_i16, 0, q8s_half)
        while sb < 4:
            # ── Shared Q8K loads + broadcasts ──
            la, lb, lc, ld = load q8_qs[...]                  # SHARED
            v00, v01, v10, v11 = concat broadcasts            # SHARED

            # ── A-side block (mirror of lines 87–210 of q4k_dot_8x8.ea) ──
            # packed_a loads + nibble extract + utmp decode +
            # scales_0_mat_a/scales_1_mat_a/mins_01_mat_a +
            # 16 maddubs using shared v00..v11 +
            # iacc_mat_a accumulation + iacc_min_mat_a accumulation

            # ── B-side block (same body, packed_b + *_mat_b locals) ──
            # Reuses the same v00..v11 from registers — no reload.

            sb += 1
        # Four FMAs (vs. two in single kernel)
        acc_row_mat_a = fma(to_f32(iacc_mat_a),     col_d_mat_a    .* row_sc, acc_row_mat_a)
        acc_min_mat_a = fma(to_f32(iacc_min_mat_a), col_dmin_mat_a .* row_sc, acc_min_mat_a)
        acc_row_mat_b = fma(to_f32(iacc_mat_b),     col_d_mat_b    .* row_sc, acc_row_mat_b)
        acc_min_mat_b = fma(to_f32(iacc_min_mat_b), col_dmin_mat_b .* row_sc, acc_min_mat_b)
        b += 1
    store(out_mat_a, x*8, shuffle(acc_row_mat_a, [0,2,4,6,1,3,5,7]) .- acc_min_mat_a)
    store(out_mat_b, x*8, shuffle(acc_row_mat_b, [0,2,4,6,1,3,5,7]) .- acc_min_mat_b)
    x += 1
```

Estimated line count: ~360 (roughly 228 × 1.6 — body duplication plus
distinct scales/mins for the two matrices). Under the 500-line limit.

## ARM NEON

Same structural argument applies to `kernels/q4k_dot_8x8_arm.ea` (248
lines). The NEON kernel uses different intrinsic names but the same
per-`(tile, b, sb)` decomposition. Task 3 produces the NEON mirror.
