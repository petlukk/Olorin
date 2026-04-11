# q4k_8x8_q8k_gemm — Eä kernel template (AVX2 path + ARM derivation)

Input to Phase 2 Plan 1, Tasks 2, 5, and 6. This note pins down:

1. The byte layout of `block_q8_Kx4` — both the `qs[1024]` interleave and
   the `bsums[64]` interleave — so that Task 2's `q8k_repack_4` kernel
   produces the exact layout the gemm expects.
2. The helper-func decomposition of llama.cpp's AVX2 gemm body so that
   Task 5's x86 Eä kernel author has a line-bounded skeleton to fill.
3. The ARM NEON+dotprod derivation path for Task 6, which has no
   llama.cpp reference (Pi 5 Cortex-A76 lacks `i8mm`; llama.cpp's ARM
   gemm paths all require `__ARM_FEATURE_MATMUL_INT8` and fall through
   to scalar on Pi 5).

This note stands alone from the earlier Phase A note
`2026-04-08-ggml-q4k-8x8-q8k-gemm.md`, which documented the AVX-512
path. **Olorin1 does not use AVX-512.** An earlier AVX-512 attempt was
deleted wholesale after it proved to be a trap. All code paths referenced
below are the AVX2 fallback only.

## Sources read (no edits)

All line numbers are "at the time of writing" anchors. The symbols are
the load-bearing references.

- `$HOME/projects/llama.cpp/ggml/src/ggml-cpu/repack.h`
  - `struct block_q8_Kx4` at lines 96–100.
- `$HOME/projects/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp`
  - free function `ggml_gemm_q4_K_8x8_q8_K` at line 2042.
  - inside it, the **AVX-512 path** is `#if __AVX512BW__ && __AVX512DQ__`
    from 2077 to 2815. **IGNORED by this note.**
  - the **AVX2 fallback** is the body following `#endif // __AVX512BW__ …`
    from 2816 to 3487, ending at `#else` (generic scalar) on 3488.
    **This is the only path the Eä x86 kernel translates.**
  - Q4Kx8 matvec (`ggml_gemv_q4_K_8x8_q8_K`) AVX2 body at 1194–1446,
    structurally identical to the gemm rp=1 case — olorin's existing
    `kernels/q4k_dot_8x8.ea` is a line-for-line port of this.
- `$HOME/projects/olorin1/kernels/q4k_dot_8x8.ea` (228 lines) —
  existing x86 AVX2 matvec, structural precedent for Task 5.
- `$HOME/projects/olorin1/kernels/q4k_dot_8x8_arm.ea` (263 lines) —
  existing ARM NEON+dotprod matvec, structural precedent for Task 6.

## `block_q8_Kx4` struct — confirmed layout

From `repack.h:96-100`:

```c
struct block_q8_Kx4 {
    float d[4];              // delta, one per packed row
    int8_t qs[QK_K * 4];     // QK_K = 256 → 1024 quant bytes
    int16_t bsums[QK_K / 4]; // 64 i16 bsums total
};
```

With the static_assert at `repack.h:102`:

```
sizeof(block_q8_Kx4) = 4*sizeof(float) + QK_K*4 + (QK_K/4)*sizeof(int16_t)
                     = 16 + 1024 + 128
                     = 1168 bytes
```

**`q8k_repack_4` kernel output stride: 1168 bytes per super-block.**

No header padding on x86-64; the struct is naturally 4-byte aligned by
`d[4]`. `qs` starts at offset 16, `bsums` starts at offset 16 + 1024 =
1040.

## `qs[1024]` byte interleave — confirmed

### The critical question

Task 2 must produce `qs[1024]` in a specific order. The gemm at AVX2
line 2993 reads via 8 consecutive 32-byte loads per sb-iter at offsets
`256*sb + {0, 32, 64, 96, 128, 160, 192, 224}`, where `sb = 0..4`
(each sb-iter covers two sub blocks of 32 quants × 4 rows = 256 bytes).
Total: 4 sb-iters × 256 bytes = 1024 bytes = full qs.

Each 32-byte load is then split by `_mm256_permute2f128_si256(x, x, 0)`
(low 128 duplicated) and `_mm256_permute2f128_si256(x, x, 17)` (high 128
duplicated). So each load's low 16 bytes become `lhs_mat_01_*` (rows
0,1) and its high 16 bytes become `lhs_mat_23_*` (rows 2,3).

The `_mm256_shuffle_epi32(..., 160)` + `_mm256_shuffle_epi32(..., 245)`
sp1/sp2 pair that follows reconstructs the per-row 8-byte slices. The
inline comments at lines 3024–3045 are definitive:

```
lhs_mat_01_00_sp1 shuffle(160): A00(0-3) A00(0-3) A01(0-3) A01(0-3) …
lhs_mat_23_00_sp1 shuffle(160): A02(0-3) A03(0-3) A02(0-3) A03(0-3) …
```

Reading `A00` as "row-0, sub-block-0, bytes 0..3" and `A01` as "row-1,
sub-block-0, bytes 0..3": the low 16 bytes of the 32-byte load at offset
0 hold `{row-0 bytes 0..7, row-1 bytes 0..7}`, and the high 16 bytes
hold `{row-2 bytes 0..7, row-3 bytes 0..7}`. **One 32-byte load = one
8-byte slice from each of the 4 rows.**

### The layout

For each super-block, `qs[1024]` is laid out as:

```
For sb-iter in 0..4 (outer):
  each sb-iter covers 2 consecutive sub-blocks (sb0, sb1) per row
  each sub-block is 32 quant bytes per row
  one sb-iter occupies 256 bytes of qs = 8 consecutive 32-byte chunks

Within each sb-iter, the 8 chunks of 32 bytes are:

  chunk 0 (offset   0): row-0[ 0..7], row-1[ 0..7], row-2[ 0..7], row-3[ 0..7]   (sub-block sb0 of that row, quant bytes  0.. 7)
  chunk 1 (offset  32): row-0[ 8..15], row-1[ 8..15], row-2[ 8..15], row-3[ 8..15] (sub-block sb0, quant bytes  8..15)
  chunk 2 (offset  64): row-0[16..23], row-1[16..23], row-2[16..23], row-3[16..23] (sub-block sb0, quant bytes 16..23)
  chunk 3 (offset  96): row-0[24..31], row-1[24..31], row-2[24..31], row-3[24..31] (sub-block sb0, quant bytes 24..31)
  chunk 4 (offset 128): row-0[ 0..7], row-1[ 0..7], row-2[ 0..7], row-3[ 0..7]   (sub-block sb1 of that row, quant bytes  0.. 7)
  chunk 5 (offset 160): row-0[ 8..15], …                                          (sub-block sb1, quant bytes  8..15)
  chunk 6 (offset 192): …                                                          (sub-block sb1, quant bytes 16..23)
  chunk 7 (offset 224): …                                                          (sub-block sb1, quant bytes 24..31)
```

Concretely, for a given `(row r, sub-block s, quant position p)` with
`r ∈ 0..4`, `s ∈ 0..8`, `p ∈ 0..32`, the destination byte in
`qs[1024]` is:

```
  sb_iter   = s / 2                // 0..4
  sb_within = s % 2                // 0 or 1
  qchunk    = p / 8                // 0..4 — which 8-byte slice in the sub-block
  qbyte     = p % 8                // 0..8 — byte within the slice

  offset = sb_iter * 256
         + sb_within * 128         // chunks 0..3 for sb0, 4..7 for sb1
         + qchunk * 32             // chunks 0/1/2/3 or 4/5/6/7
         + r * 8                   // row-r's 8-byte slice inside the 32-byte chunk
         + qbyte
```

**Granularity:** 8-byte slices. Four rows × 8 bytes = 32 bytes per
chunk. No shuffling inside the 8-byte slices — each slice is a
contiguous copy of 8 consecutive quant bytes from one source row.

This is **not row-major** (rows are not contiguous) and **not fully
interleaved at the byte level** (the smallest unit is 8 bytes, not 1).
It is **8-byte-granular row interleave** with outer sub-block ordering.

Source citations: load pattern at `repack.cpp:2993-3014`; per-lane
comment confirmation at `repack.cpp:3024-3045`; sb loop at
`repack.cpp:2853`.

## `bsums[64]` byte interleave — confirmed

From `repack.cpp:3019`, per sb-iter:

```c
__m256i lhs_bsums_0123_01 = _mm256_loadu_si256(
    (const __m256i *)(a_ptrs[rp][b].bsums + 16 * sb));
```

One 32-byte load = 16 i16 values per sb-iter. Four sb-iters × 16 =
64 i16 = full `bsums[QK_K/4]`. The sb-iter advances by 16 i16 (32
bytes), so the 64 bsums are split into 4 contiguous groups of 16, one
group per sb-iter.

The variable name `lhs_bsums_0123_01` is "rows 0..3, sub-blocks 0 and
1" — the `01` refers to the two sub-blocks the sb-iter handles. It is
then hadd-reduced:

```c
__m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(
    _mm_hadd_epi16(
        _mm256_castsi256_si128(lhs_bsums_0123_01),
        _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(
    lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);
```

`_mm_hadd_epi16(lo128, hi128)` sums adjacent i16 pairs within each
128-bit half and concatenates, producing 8 i16 outputs. The subsequent
`shuffle_epi32(hsum, 0)`, `85`, `170`, `255` calls at 3139-3142 broadcast
4-i32-at-a-time lanes to feed `_mm256_madd_epi16(·, mins_01)` four
times (once per row). This tells us the hadd output layout must be:

```
  i16[0..1]: row-0's two bsum-sums (one per sub-block sb0, sb1)
  i16[2..3]: row-1's two bsum-sums
  i16[4..5]: row-2's two bsum-sums
  i16[6..7]: row-3's two bsum-sums
```

Working backwards from `_mm_hadd_epi16(lo, hi)`, which produces:

```
  out[0] = lo[0] + lo[1]
  out[1] = lo[2] + lo[3]
  out[2] = lo[4] + lo[5]
  out[3] = lo[6] + lo[7]
  out[4] = hi[0] + hi[1]
  out[5] = hi[2] + hi[3]
  out[6] = hi[4] + hi[5]
  out[7] = hi[6] + hi[7]
```

and given out[0..1] must be (row-0 sb0 total, row-0 sb1 total) the
16-i16 load layout is:

```
  lo (i16[0..7], from the low 128 of the 32-byte load):
    i16[0..1] : row-0 sb0 → 2 bsums covering position-groups 0..15 and 16..31
    i16[2..3] : row-0 sb1 → 2 bsums
    i16[4..5] : row-1 sb0
    i16[6..7] : row-1 sb1

  hi (i16[8..15], from the high 128 of the 32-byte load):
    i16[ 8.. 9] : row-2 sb0
    i16[10..11] : row-2 sb1
    i16[12..13] : row-3 sb0
    i16[14..15] : row-3 sb1
```

Concretely, for `(row r, sub-block s, bsum-half h)` with `r ∈ 0..4`,
`s ∈ 0..8`, `h ∈ 0..2` (each sub-block has 2 bsums, covering the first
16 and second 16 quant positions):

```
  sb_iter   = s / 2                 // 0..4
  sb_within = s % 2                 // 0 or 1
  rgroup    = r / 2                 // 0 for rows 0,1 ; 1 for rows 2,3
  rwithin   = r % 2                 // 0 or 1

  i16_offset = sb_iter * 16          // 16 i16 per sb-iter
             + rgroup * 8            // rows 0,1 in low half; rows 2,3 in high half
             + rwithin * 4           // 4 i16 per row (2 sub-blocks × 2 halves)
             + sb_within * 2         // 2 i16 per sub-block
             + h
```

So `bsums[64]` is **row-group-ordered with granularity 4 i16 per row**,
grouped two-rows-at-a-time for 128-bit hadd lanes. This is again **not
row-major** (row 2 sits in the second half of each sb-iter's 16-i16
group, not after row 0's full 16 i16). `q8k_repack_4` must emit
bsums in this exact order.

Source citations: load at `repack.cpp:3019`; hadd at 3020; broadcast
at 3139-3142.

## AVX2 gemm body — helper-func decomposition

llama.cpp's AVX2 fallback body at 2816-3487 is one ~670-line block. It
will blow olorin's 500-line rule if translated line-for-line into one
Eä function. The body decomposes cleanly into three independent
sections that map to three Eä helper funcs:

### Section A — per-(sb-iter) weight unpack (shared across all rp)

`repack.cpp:2855-2951`. Loads 8 × 32-byte packed weight vectors from
`b_ptr[b].qs + 256*sb + {0..224 step 32}`, applies
`_mm256_permutevar8x32_epi32 + _mm256_blend_epi32` to reshape them into
`rhs_raw_mat_0145_*` / `rhs_raw_mat_2367_*` (8 vectors), nibble-extracts
low (`& m4b`) and high (`srli_epi16(4) & m4b`) into 16 `rhs_mat_*_*` i8
vectors, and then applies `_mm256_shuffle_epi32(..., 136)` and
`_mm256_shuffle_epi32(..., 221)` to produce 32 `rhs_mat_*_sp{1,2}`
vectors ready for `maddubs_epi16`.

**Shared:** this runs once per `(tile, super-block, sb-iter)` and is
reused by all 4 `rp` iterations.

**Eä helper:**

```
func unpack_weight_sb(
    packed: *restrict u8,
    b: i32,
    sb: i32,
    // output: 32 shuffled nibble vectors as i8x32
    out_rhs_mat_0145_00_sp1: *mut i8, …
) -> ()
```

Actually, Eä functions can return tuples / arrays of vectors directly
or write into a caller-allocated scratch buffer. The existing olorin
matvec `q4k_dot_8x8.ea` lines 60-150 do this inline — for the gemm the
same 90 lines become a helper called once per sb-iter per super-block.
**Estimated Eä size: ~90 lines.**

### Section B — per-(sb-iter) scale decode (shared across all rp)

`repack.cpp:2953-2987`. The utmp_0/utmp_1 scalar dance at 2953-2970
(`memcpy` from `b_ptr[b].scales + 24*sb`, the kmask1/2/3 bit
manipulation) produces 8 scale bytes and 8 min bytes for the two sub
blocks. Then lines 2973-2987 build `scales_0`, `scales_1`, `mins_01`,
`scale_0145_0/1`, `scale_2367_0/1` as i16x16 vectors.

**Shared:** per `(tile, super-block, sb-iter)`, reused by all 4 rp.

**Eä helper:**

```
func decode_scales_sb(
    packed: *restrict u8,
    b: i32,
    sb: i32,
    // output scalars/vectors — scratch slots or tuple
) -> ()
```

The utmp scalar block already exists in olorin's matvec at
`q4k_dot_8x8.ea` ~line 100 — re-use the same integer-bit manipulation.
The vector construction uses `_mm_set_epi32` → `cvtepu8_epi16` + a few
`_mm256_shuffle_epi32` ops. **Estimated Eä size: ~55 lines.**

### Section C — per-(sb-iter, rp) A-side load + dot + FMA

`repack.cpp:2989-3148`. The `for (int rp = 0; rp < 4; rp++)` loop at
2989. Inside:

1. **A-load (2993-3016):** 8 × 32-byte loads from
   `a_ptrs[rp][b].qs + 256*sb + {0..224 step 32}`, each split via
   `_mm256_permute2f128_si256(., ., 0/17)` into `lhs_mat_01_*` /
   `lhs_mat_23_*`. → 16 i8x32 vectors.
2. **Bsums load (3019-3021):** 32-byte load at
   `a_ptrs[rp][b].bsums + 16*sb`, hadd-reduced to `lhs_bsums_hsum_0123_01`.
3. **A-shuffle (3024-3071):** 32 `lhs_mat_*_sp{1,2}` vectors via
   `_mm256_shuffle_epi32(.., 160/245)` — these are A-side counterparts
   to the B-side shuffles produced in Section A.
4. **Dot (3074-3090):** 16 `_mm256_maddubs_epi16` calls inside nested
   `_mm256_add_epi16` chains, producing 8 `iacc_mat_*_*_sp{1,2}` i16x16.
5. **Combine (3092-3101):** 8 `_mm256_add_epi16` merging sp1+sp2 pairs
   into 8 `iacc_mat_*` i16x16.
6. **Scale (3103-3112):** 8 `_mm256_madd_epi16(·, scale_*)` calls →
   8 i32x8 outputs.
7. **Row straighten (3114-3127):** 8 `_mm256_blend_epi32 + shuffle_epi32`
   calls combining `iacc_mat_*` pairs into 4 i32x8 row accumulators for
   sb0 + 4 for sb1, then 4 `_mm256_add_epi32` to sum sb0+sb1.
8. **D scale load + FMA (3129-3137):** load
   `row_scale_f32 = a_ptrs[rp][b].d`, broadcast each f32 lane, FMA into
   `acc_rows[rp*4 .. rp*4+3]`.
9. **Min FMA (3139-3147):** 4 `_mm256_madd_epi16` calls on
   `shuffle_epi32(lhs_bsums_hsum_0123_01, {0,85,170,255})` × `mins_01`,
   then FMA into `acc_min_rows[rp*4 .. rp*4+3]`.

**Per-rp, 4 times per `(tile, super-block, sb-iter)`.**

**Eä helper:**

```
func acc_rp_sb(
    q8_a_ptr: *restrict u8,      // a_ptrs[rp][b] base
    b: i32,
    sb: i32,
    rp: i32,                     // 0..4 — only affects where acc_rows[rp*4+*] writes back
    // weight unpack results from Section A (32 nibble vectors)
    // scale decode results from Section B (scale_0145_*, scale_2367_*, mins_01)
    // accumulator slots
    acc_rows:     *mut f32,      // 16 f32x8 slots — tile accumulators
    acc_min_rows: *mut f32,      // 16 f32x8 slots
    col_scale_f32: f32x8,        // b_ptr[b].d broadcast
    col_dmin_f32:  f32x8         // b_ptr[b].dmin broadcast
) -> ()
```

Note: `col_scale_f32` and `col_dmin_f32` are loaded once per super-block
(2847, 2850) so they are passed in as args, not recomputed. The
Section-A results are either a scratch region pointer or (if Eä
supports it) a large tuple.

**Estimated Eä size: ~125 lines.** Allocating scratch helps keep the
argument list manageable.

### Outer skeleton

```
export func q4k_8x8_q8k_gemm(
    packed:        *restrict u8,     // block_q4_Kx8 tiles
    q8_a:          *restrict u8,     // block_q8_Kx4 tiles (output of q8k_repack_4)
    out:           *mut f32,
    bs:            i32,              // output row stride in floats
    n:             i32,              // K inner, must be % 256 == 0
    nr:            i32,              // A rows, must be % 4 == 0
    nc:            i32               // B cols, must be % 8 == 0
) {
    // nb = n / 256
    // y = 0
    // while y < (nr - nr%16) / 4, y += 4:
    //     // a_ptrs[0..3] at y*nb stride
    //     let mut x = 0
    //     while x < nc / 8:
    //         // zero acc_rows[16], acc_min_rows[16]
    //         let mut b = 0
    //         while b < nb:
    //             // load col_scale_f32, col_dmin_f32 (once per super-block)
    //             let mut sb = 0
    //             while sb < 4:
    //                 unpack_weight_sb(packed, b, sb, …)
    //                 decode_scales_sb(packed, b, sb, …)
    //                 let mut rp = 0
    //                 while rp < 4:
    //                     acc_rp_sb(a_ptrs[rp], b, sb, rp, …)
    //                     rp += 1
    //                 sb += 1
    //             b += 1
    //         // store acc_rows[i] - acc_min_rows[i] → out[(y*4+i)*bs + x*8]
    //         x += 1
    //     y += 4
    // // tail y loop for (nr % 16) / 4 — 1 rp instead of 4
}
```

**Estimated Eä size (skeleton only): ~150 lines** including the tail-y
loop (for `nr % 16` unaligned rows, rp = 1 instead of 4 — see
`repack.cpp:3158` onward). The tail path is the existing matvec loop
body wrapped in its own outer x loop.

**Total file line estimate:**

| Section               | Lines |
|-----------------------|------:|
| Header + helpers      |    15 |
| `unpack_weight_sb`    |    90 |
| `decode_scales_sb`    |    55 |
| `acc_rp_sb`           |   125 |
| outer `q4k_8x8_q8k_gemm` | 150 |
| tail-y single-rp path |    80 |
| **Total**             | **~515** |

**This is over the 500-line limit by ~15 lines.** Mitigations the Task 5
implementer should try first, in order:

1. Inline `decode_scales_sb` back into the outer loop (55 → 0 in helper,
   +55 at call site, net ~0 — but removes one helper's overhead).
2. Fold the tail-y path into the main loop with a runtime `if rp_max`
   branch (saves ~60 lines of duplication; adds ~10 of branching).
3. Hoist the bsums hadd computation into a separate helper called once
   per rp (currently ~6 lines inside `acc_rp_sb`, split across both
   paths — saves ~6 per path).

If all three applied, estimate drops to ~440 lines. The first is the
simplest and alone brings the file to under 500 comfortably.

## `block_q8_Kx4` field offsets (for the Eä kernel)

For the Eä kernel that addresses bytes inside a packed `q8_a_ptr`:

```
  struct offset (bytes)   | field
  ------------------------|-----------------------------------
    0                     | d[0] (f32)
    4                     | d[1] (f32)
    8                     | d[2] (f32)
   12                     | d[3] (f32)
   16                     | qs[0]   (i8)
   16 + 1024 = 1040       | bsums[0] (i16)
   16 + 1024 + 128 = 1168 | (next block_q8_Kx4)
```

The Eä kernel for the gemm receives a single base pointer into the
`block_q8_Kx4` array, strides by 1168 per super-block, and uses
compile-time-known field offsets for `d`, `qs`, and `bsums`.

## ARM NEON+dotprod gemm derivation (no llama.cpp reference)

llama.cpp's ARM gemm paths at `arch/arm/repack.cpp` all require
`__ARM_FEATURE_MATMUL_INT8` (i8mm); on Cortex-A76 / Pi 5 there is no
i8mm feature flag, and llama.cpp falls through to the scalar generic
path. This means **Task 6 has no line-for-line reference** — the ARM
NEON+dotprod gemm must be derived from olorin's own single-column
matvec `kernels/q4k_dot_8x8_arm.ea`.

### Derivation: matvec → gemm

The existing matvec (263 lines) uses two f32x4 row accumulators:

```
acc0: f32x4  // rows 0..3
acc1: f32x4  // rows 4..7
```

Its per-super-block body produces 8 `vdot_i32` outputs (i32x4) against
one Q8_K column's qs and folds them into `acc0`, `acc1` via `hadd_i32
→ to_f32 → fma(col_d, row_d)`.

The gemm extends the matvec by **amortizing weight unpack across the
4 input rows of one `block_q8_Kx4`.** The naive extension: read the
weight tile once per super-block, then run the 8 vdot sequence 4 times
(once per row r ∈ 0..4) against each row's qs data. Per-row accumulators
become:

```
acc_row[r][cc]  for r in 0..8, cc in 0..(rp_count)
```

where `cc` is the A-row index (here 0..4, matching rp). Each
`acc_row[r][cc]` is f32x4 covering the 4 output cols in the row
(matching the existing matvec's acc shape).

The weight tile is loaded once per `(x, b)` pair (x = output-col-tile,
b = super-block). For each of the 4 input rows (cc = 0..4), the qs and
bsums are drawn from the interleaved `block_q8_Kx4` layout documented
above. Since ARM NEON has 16-byte-wide registers vs. AVX2's 32-byte,
the 8-byte interleave granularity maps cleanly: one `ldr q` load gets
two rows' 8-byte slices in one go — use `vget_low` / `vget_high` (or
the Eä `split_i8x16_low/high` equivalent) to extract per-row slices.

### Per-cc iteration body

```
for cc in 0..4:
    // base of row-cc's qs inside the q8_a block
    // for sb-iter in 0..4:
    //   for each of 4 quant-chunks in the sb-iter:
    //     - load one 8-byte slice from the cc-th row at the interleaved offset
    //     - apply the 8 vdot_i32 dot products against the
    //       (already-unpacked) weight tile
    //     - accumulate into i32x4 per-row partials
    // for each of 8 output rows r:
    //   folded = hadd_i32(partial[r][0], partial[r][1])   // pair-reduce
    //   folded = hadd_i32(folded_lo, folded_hi)           // i32x4 → scalar sum
    //   acc_row[r][cc] = fma(to_f32(folded), col_d[r], row_d_cc * acc_row[r][cc])
    // // bsums subtract:
    //   acc_row[r][cc] -= bias[r][cc] * sb_min_cc
```

The cc loop can be unrolled (4 iterations) or kept as a runtime while;
the existing matvec is a while loop over super-blocks and the gemm
follows the same style. Unrolling cc gives more register pressure but
keeps weight-tile state hot.

### NEON register pressure

NEON has 32 vector registers (v0..v31). The cc-loop body's per-
iteration working set:

- 8 `acc_row[r][cc]` f32x4 — 8 regs (live across cc iterations)
- 8 `bias[r][cc]` i32x4     — 8 regs (live across cc iterations)
- 4 q8 register tiles i8x16 — 4 regs (per cc, reloaded)
- 8 vdot_i32 results i32x4  — 8 regs (per cc, reused after fold)
- weight-unpack state       — ~10 regs

Static live count per cc: 8 + 8 + 10 = 26 regs (acc + bias + weight
state). Plus 4 q8 loads and 8 vdot outputs during the inner dot phase:
peak = 26 + 12 = 38 regs. This exceeds the 32 register file; the
compiler will spill ~6 regs. Acceptable — the spill set is short-lived
(vdot results die after fold), so the spill cost is local.

If spill pressure turns out to be painful in practice, the mitigation
is to split the cc loop into two passes (cc=0,1 then cc=2,3) so the
working set halves. The existing dual matvec `q4k_dot_8x8_dual_arm.ea`
uses this pattern already.

### Line budget

Existing matvec: 263 lines. The gemm extension adds:

- cc outer loop (~8 lines)
- 4× the per-row fold+fma block at the end of the sb loop (~30 lines, partially replaces existing matvec fold)
- bsums subtract per cc (~10 lines)
- outer y/b skeleton adjustments (~20 lines)
- additional weight-unpack hoisting (~20 lines to keep tile-shared state cleaner)

Net: +150–180 lines. **Estimated total: ~430 lines**, comfortably under
the 500 limit.

## Intrinsic availability note

The following intrinsics named in Task 1's eabrain baseline
(`concat_i8x16`, `shuffle_bytes`, `maddubs_i16`, `madd_i16`) returned
no matches from `eabrain ref` — eabrain does not index eacompute's
Rust intrinsic definitions, only `.ea` kernel source. The intrinsics
exist in eacompute:

- `shuffle_bytes` — present in `eacompute/src/typeck/intrinsics.rs`
  (both u8x16 and u8x32 variants).
- `maddubs_i16`, `madd_i16` — present in the same file alongside
  `pmaddubsw` / `pmaddwd` bindings.
- `concat_i8x16` / `concat_u8x16` — present (the existing
  `q4k_dot_8x8.ea` uses `concat_u8x16` at line 35).

The existing matvec `q4k_dot_8x8.ea` uses all four — Task 5 inherits
the same intrinsic set. No new intrinsics are required by the gemm.

For the ARM path, `vdot_i32` / `hadd_i32` are the dotprod-backed
intrinsics used by the existing matvec; no new NEON intrinsics are
required for the gemm either.

## What Task 5 writes first (x86 AVX2 gemm)

1. Copy `kernels/q4k_dot_8x8.ea` to `kernels/q4k_dot_8x8_gemm.ea` as
   a starting skeleton. Rename the export to `q4k_8x8_q8k_gemm`.
2. Add a second pointer arg for the output stride `bs: i32` and the
   tile dimensions `nr, nc: i32`.
3. Extract Section A (weight unpack) into helper `unpack_weight_sb`
   — lift lines 60-150 of the existing matvec.
4. Extract Section B (scale decode) into helper `decode_scales_sb`
   — lift lines ~100-130.
5. Write Section C (`acc_rp_sb`) from scratch using the matvec's
   inner dot+FMA block as a template; multiply accumulator slots by 4
   (rp=4 rows of A at once).
6. Wrap in the `y / x / b / sb / rp` outer loop skeleton.
7. Add the `nr % 16 != 0` tail-y path (single-rp — functionally
   equivalent to the existing matvec wrapped in an x loop).
8. Run the Task 7 bit-exact test against `q4k_8x8_q8k_matvec` run N
   times.

## What Task 6 writes first (ARM NEON+dotprod gemm)

1. Copy `kernels/q4k_dot_8x8_arm.ea` to
   `kernels/q4k_dot_8x8_gemm_arm.ea` as a starting skeleton. Rename
   the export to `q4k_8x8_q8k_gemm`.
2. Expand the 2 f32x4 accumulators (`acc0`, `acc1`) into an
   `acc_row[8][4]` 2-D structure (8 output rows × 4 input rows).
3. Add the `cc ∈ 0..4` input-row loop inside the super-block body.
   Keep the 8 vdot_i32 sequence unchanged — only the A-side q8 loads
   change per cc.
4. Update the `hadd_i32 → fma` fold path to address all 4 cc slots
   (4× the existing fold, one per cc).
5. Update the bsums subtract to address all 4 cc slots.
6. Run the Task 7 bit-exact test on ARM.

## Appendix — key AVX2 line-number anchors

Everything below is "at the time of writing" against the olorin-checked
llama.cpp tree at `/home/peter/projects/llama.cpp`.

```
repack.cpp:2042   ggml_gemm_q4_K_8x8_q8_K (function entry)
repack.cpp:2077   AVX-512 path starts (IGNORED)
repack.cpp:2815   AVX-512 path #endif
repack.cpp:2816   AVX2 fallback body starts
repack.cpp:2818   main y loop (y < anr/4, y += 4) — 16 rows / 4 rp
repack.cpp:2828   x loop (x < nc/8)
repack.cpp:2833   acc_rows[16] f32x8
repack.cpp:2838   acc_min_rows[16] f32x8
repack.cpp:2844   b loop (super-block)
repack.cpp:2847   col_scale_f32 load
repack.cpp:2850   col_dmin_f32 load
repack.cpp:2853   sb loop (sb < 4 — two sub-blocks per iter)
repack.cpp:2856   Section A starts — weight qs loads
repack.cpp:2866   blend_epi32 + permutevar8x32 reshape
repack.cpp:2877   nibble extract (& m4b)
repack.cpp:2890   nibble extract high (srli_epi16(4) & m4b)
repack.cpp:2903   shuffle_epi32(·, 136) — sp1 shuffles
repack.cpp:2929   shuffle_epi32(·, 221) — sp2 shuffles
repack.cpp:2953   Section B starts — utmp decode
repack.cpp:2957   memcpy(utmp_0, b_ptr[b].scales + 24*sb, 12)
repack.cpp:2973   mins_and_scales_0 / scales_0 construction
repack.cpp:2981   mins_01 construction
repack.cpp:2983   scale_0145_0 / scale_2367_0 / scale_0145_1 / scale_2367_1
repack.cpp:2989   Section C starts — rp loop
repack.cpp:2993   lhs_mat_0123_00 = loadu(a_ptrs[rp][b].qs + 256*sb)
repack.cpp:2994   permute2f128(x, x, 0)  → lhs_mat_01_00 (rows 0,1 share lo128)
repack.cpp:2995   permute2f128(x, x, 17) → lhs_mat_23_00 (rows 2,3 share hi128)
repack.cpp:3019   lhs_bsums_0123_01 = loadu(a_ptrs[rp][b].bsums + 16*sb)
repack.cpp:3020   hadd_epi16(lo128, hi128)
repack.cpp:3024   shuffle_epi32(·, 160) → sp1
repack.cpp:3049   shuffle_epi32(·, 245) → sp2
repack.cpp:3074   maddubs_epi16 × 16 → iacc_mat_*_sp{1,2} i16x16
repack.cpp:3093   add_epi16 merge sp1+sp2 → iacc_mat_*
repack.cpp:3104   madd_epi16(·, scale_*) → i32x8
repack.cpp:3115   blend_epi32 + shuffle_epi32 → row-straighten
repack.cpp:3124   add_epi32 → iacc_row_* (sb0 + sb1)
repack.cpp:3130   row_scale_f32 = load(a_ptrs[rp][b].d)
repack.cpp:3134   acc_rows[rp*4 + 0..3] fmadd
repack.cpp:3139   iacc_row_min_* = madd_epi16(shuffle(bsums_hsum, …), mins_01)
repack.cpp:3144   acc_min_rows[rp*4 + 0..3] fmadd
repack.cpp:3149   end sb loop
repack.cpp:3150   end b loop
repack.cpp:3153   store acc_rows[i] - acc_min_rows[i] → s[...]
repack.cpp:3158   tail-y loop (rp=1 path)
repack.cpp:3487   AVX2 fallback body ends
repack.cpp:3488   #else generic scalar (IGNORED)
```
