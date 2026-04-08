# ggml Q4K_8x8 × Q8K gemm — inner loop spec

Companion to `2026-04-08-ggml-q4k-8x8-format.md` (byte layout). This note
documents the **inner loop** that consumes the repacked layout. Its sole
purpose is to pin down the exact f32 reduction order so the Eä kernel
`q4k_8x8_q8k_gemm` can be bit-exact with llama.cpp on x86 AVX-512.

## Sources read (no edits)

All line numbers below are parenthetical references to llama.cpp build
8685 — the symbols are the load-bearing anchors.

- `/root/dev/llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp`
  - free function **`ggml_gemm_q4_K_8x8_q8_K`** (at the time of writing,
    line 2042) — not a template specialization; the
    `tensor_traits<block_q4_K,8,8,Q8_K>` trait dispatches to this named
    function via its `.gemm` member.
  - Inside that function, the **`#if defined(__AVX512BW__) && defined(__AVX512DQ__)`**
    guard block is the AVX-512 path (at the time of writing, lines
    2077–2815).
  - The **`#else` AVX2 fallback** in the same function (at the time of
    writing, lines ~2816–3487) runs the same per-output recipe at
    256-bit lane width.
  - The generic scalar fallback **`ggml_gemm_q4_K_8x8_q8_K_generic`** (at
    the time of writing, line 3492) is reference-only.
- `/root/dev/llama.cpp/ggml/src/ggml-cpu/repack.cpp`
  - trait registration **`q4_K_8x8_q8_K`** (at the time of writing,
    line 4536).

Olorin runs on AVX-512 → the spec below describes the AVX-512 path. See
"Footnote: AVX2 and generic fallbacks" for why the AVX2 path is
per-output bit-exact with AVX-512.

## Signature and tiling

```c
void ggml_gemm_q4_K_8x8_q8_K(
    int n,               // K (inner dim, must be multiple of QK_K=256)
    float* s,            // C output, row-major
    size_t bs,           // C row stride (in floats)
    const void* vx,      // B = block_q4_Kx8*  (ncols_interleaved=8, blocklen=8)
    const void* vy,      // A = block_q8_Kx4*  (4 rows of Q8_K interleaved)
    int nr,              // number of A rows  (must be % 4 == 0)
    int nc);             // number of B cols  (must be % 8 == 0)
```

- `block_q4_Kx8` packs **8 output columns** of Q4_K (super-block of 256).
- `block_q8_Kx4` packs **4 input rows**  of Q8_K (super-block of 256)
  — with `.qs[1024]`, `.bsums[16*4]`, `.d[4]`.
- Outer loops:
  - `y` steps rows in groups of 16 (`anr = nr - nr%16`; inner `y += 4`
    with `a_ptrs[0..3]` each separated by `nb` super-blocks → 16 rows).
  - `x` steps cols in groups of 16 (`anc = nc - nc%16`; inner `x += 2`
    uses `b_ptr_0 = x`, `b_ptr_1 = x+1` → 16 cols).
  - So the registered tile is **16 rows × 16 cols** per AVX-512 iteration.
  - `b` steps over super-blocks (`nb = n/QK_K`, low→high).
  - `sb` steps over sub-block *pairs* inside a super-block
    (`QK_K/64 = 4` iterations; each covers two Q4K 32-element sub-blocks).

### Master accumulators (per 16×16 tile)

```c
__m512 acc_rows    [16];  // one f32 lane per output column (16 cols)
__m512 acc_min_rows[16];  // parallel "mins" (dmin) accumulator
for (i=0; i<16; i++) { acc_rows[i] = 0.0f; acc_min_rows[i] = 0.0f; }
```

`acc_rows[i]` is a `__m512` holding **all 16 output columns** for row `i`
in the tile. Lanes = columns. A separate `__m512` per row keeps each
row's reduction independent (see "f32 reduction order" below).

## Inner loop (pseudocode, AVX-512 path)

```
for b in 0..nb:                                       # super-blocks, low→high
    col_scale_f32 = f16->f32(b_ptr_0[b].d  ‖ b_ptr_1[b].d )   # 16 col d
    col_dmin_f32  = f16->f32(b_ptr_0[b].dmin‖b_ptr_1[b].dmin) # 16 col dmin

    for sb in 0..4:                                   # sub-block *pairs*
        # ---- B side: load 8×(256 bytes) of packed 4-bit weights ----
        rhs_raw[0..3] from b_ptr_0, rhs_raw[4..7] from b_ptr_1
        # low-nibble  = sub-block (2*sb + 0)  weights "0"
        # high-nibble = sub-block (2*sb + 1)  weights "1"
        rhs_mat_*_0x = raw & 0x0F              # even sub-block
        rhs_mat_*_1x = (raw >> 4) & 0x0F       # odd sub-block
        rhs_mat_..._sp1, _sp2 = shuffle patterns for 4-elt groups

        # ---- scales/mins extraction ----
        utmp_00..11  = 6-bit unpack via kmask1/2/3 from b_ptr_0/1[b].scales
        scales_0/1   = zero-extend utmp  as u8->i16
        mins_01      = shuffled mins (for Q4K mins subtraction)
        scale_014589CD_0/1, scale_2367ABEF_0/1 = broadcast layouts

        # ---- A side: 4 "row-pair" iterations rp in 0..4 ----
        for rp in 0..4:
            load 8 × 32 int8 from a_ptrs[rp][b].qs + 256*sb
            permute into lhs_mat_01_0x, lhs_mat_23_0x  (x=0..3 for sub-block 0,
                                                         x=10..13 for sub-block 1)
            # Bsums for Q4K min term (two sub-blocks combined)
            lhs_bsums_hsum_0123_01 = hadd_epi16 on a_ptrs[rp][b].bsums+16*sb

            # ============ int16 dot accumulation ============
            # For each output pair (01, 23) × col group (0145..., 2367...)
            # × sub-block (0, 1) × shuffle pattern (sp1, sp2):
            #
            # _mm512_maddubs_epi16(rhs_mat, lhs_mat) -> i16
            # reduce across 4 "sub-sub-block" chunks with three add_epi16:
            iacc_mat_00_0_sp1 =
                ((maddubs(rhs_03, lhs_03) + maddubs(rhs_02, lhs_02))
                 + maddubs(rhs_01, lhs_01))
                + maddubs(rhs_00, lhs_00)                 # fixed order
            # same for 01, 10, 11 outputs and 0_sp2, 1_sp1, 1_sp2 variants

            # ============ int16 → int32 via scale madd ============
            # Combine the two shuffle-pattern halves (sp1 + sp2), still int16:
            iacc_mat_XY_S = add_epi16(iacc_mat_XY_S_sp1, iacc_mat_XY_S_sp2)

            # Multiply by int16 scales and horizontally add pairs to int32:
            iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0)
            iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0)
            iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0)
            iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0)
            iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1)
            iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1)
            iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1)
            iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1)

            # Reshuffle to per-row int32 accumulators for the 4 output rows of
            # this row-pair iteration (row indices rp*4 + {0,1,2,3}):
            iacc_row_{0..3}_0 = mask_blend(iacc_mat_* for sub-block 0)
            iacc_row_{0..3}_1 = mask_blend(iacc_mat_* for sub-block 1)

            # Combine the two sub-blocks of this sb iteration — still int32:
            iacc_row_0 = add_epi32(iacc_row_0_0, iacc_row_0_1)
            iacc_row_1 = add_epi32(iacc_row_1_0, iacc_row_1_1)
            iacc_row_2 = add_epi32(iacc_row_2_0, iacc_row_2_1)
            iacc_row_3 = add_epi32(iacc_row_3_0, iacc_row_3_1)

            # ============ CONVERT i32 → f32 and accumulate ============
            row_scale_f32 = broadcast(a_ptrs[rp][b].d)  # 4 f32 (per row of pair)
            acc_rows[rp*4 + k] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(iacc_row_k),
                _mm512_mul_ps(col_scale_f32, broadcast_row_d_lane_k),
                acc_rows[rp*4 + k])                      # k = 0..3

            # ============ Q4K "mins" correction (bsums · mins) ============
            iacc_row_min_k = _mm512_madd_epi16(shuffle(lhs_bsums_hsum, k), mins_01)
            acc_min_rows[rp*4 + k] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(iacc_row_min_k),
                _mm512_mul_ps(col_dmin_f32, broadcast_row_d_lane_k),
                acc_min_rows[rp*4 + k])
        # end rp
    # end sb
# end b

# After all super-blocks: store (acc_rows - acc_min_rows) to C
for i in 0..16:
    _mm512_storeu_ps(&s[(y*4+i)*bs + x*8],
                     _mm512_sub_ps(acc_rows[i], acc_min_rows[i]))
```

### SIMD width and lane assignment

- **Width:** `__m512` (16 × f32 / 16 × i32 / 32 × i16 / 64 × i8).
- **Lane assignment in `acc_rows[i]`:** the 16 f32 lanes = 16 output
  columns of the current tile (cols `x*8 .. x*8+15`). There are 16
  separate `acc_rows[]` registers, one per output row (rows
  `y*4 .. y*4+15`). Rows are fully independent accumulators — no
  cross-row reductions happen inside the kernel.

### Scales applied per-block (NOT at the end)

**Critical:** scales are applied **inside** each super-block `b`, not
accumulated-then-scaled. The flow within one `b` is:

1. int8×int8 → int16 dot products (`maddubs_epi16`).
2. Accumulated to int16 with 3 fixed-order `add_epi16`s, per sub-block.
3. Multiplied by 6-bit int16 **Q4K sub-block scales** via `madd_epi16`,
   which also horizontally adds adjacent i16 pairs → int32.
4. Two sub-blocks combined via **int32** `add_epi32`.
5. Converted i32 → f32 via `cvtepi32_ps`.
6. Multiplied by `col_scale_f32 * row_scale_f32` (both f16→f32
   super-block `d` values) and FMA'd into the f32 accumulator.

The Q4K "mins" correction runs in parallel: `bsums(Q8K row) · mins(Q4K)`
as int32, converted to f32, scaled by `col_dmin_f32 * row_scale_f32`,
FMA'd into `acc_min_rows`, and subtracted **once at the very end** via
`_mm512_sub_ps(acc_rows[i], acc_min_rows[i])` before the store.

### Exact AVX-512 intrinsics used in the hot path

- Loads: `_mm256_loadu_si256`, `_mm512_inserti32x8`, `_mm512_castsi256_si512`
- Nibble split: `_mm512_and_si512`, `_mm512_srli_epi16`
- Shuffles: `_mm512_shuffle_epi32`, `_mm512_mask_blend_epi32`
- Dot: `_mm512_maddubs_epi16`  (signed i8 × u8 → i16, pairwise add)
- i16 reduce: `_mm512_add_epi16`
- Scale: `_mm512_madd_epi16`   (i16 × i16 → i32, pairwise add)
- i32 reduce: `_mm512_add_epi32`
- i32 → f32: `_mm512_cvtepi32_ps`
- f32 scale + accumulate: `_mm512_mul_ps`, `_mm512_fmadd_ps`
- f32 shuffles (broadcast row `d` lanes): `_mm512_shuffle_ps`
- Final: `_mm512_sub_ps`, `_mm512_storeu_ps`

## THE CRITICAL PART — f32 reduction order

This is the whole reason this note exists. The 8.8% L34 drift bisected
earlier in this branch was purely a f32 sum-order difference between
incremental matvec and batched gemm (see memory: `project_gemma4_parity`
and the eabrain note on batched-vs-incremental drift).

**Per-row f32 reduction order in `acc_rows[i]` (lane-wise, i.e. per
output column):**

1. **Within one Q4K 32-element sub-block:** there is **no f32 addition**.
   All reductions are integer (`maddubs`, `add_epi16`, `madd_epi16` with
   per-sub-block scale, `add_epi32`). The int32 result is the **dot of
   the full 32 elements scaled by that sub-block's 6-bit scale**.

2. **Within one Q4K 64-element sub-block pair (one `sb` iteration):**
   still integer. The two 32-element sub-blocks are combined via
   `_mm512_add_epi32(iacc_row_k_0, iacc_row_k_1)` — this is the inner
   accumulation loop in `ggml_gemm_q4_K_8x8_q8_K`'s AVX-512 branch — and
   the result is converted to f32 *once per (`sb`, row)*.

3. **Within one Q4K super-block (`b` fixed, all 4 `sb` iterations):**
   this is where f32 addition **starts**. For each row `i`:
   ```
   acc_rows[i]  +=  f32(sb=0 iacc_pair) * (col_d * row_d)
   acc_rows[i]  +=  f32(sb=1 iacc_pair) * (col_d * row_d)
   acc_rows[i]  +=  f32(sb=2 iacc_pair) * (col_d * row_d)
   acc_rows[i]  +=  f32(sb=3 iacc_pair) * (col_d * row_d)
   ```
   Order: **`sb` low → high** (0,1,2,3), four FMAs per super-block per row.
   Note that `col_d` and `row_d` are the **same** for all 4 `sb`
   iterations within a super-block (they are super-block scales), so the
   4 FMAs differ only in the integer operand.

4. **Across super-blocks (`b` in `0..nb`):** FMA'd into the same
   `acc_rows[i]` in **ascending `b` order**. This is the dominant
   ordering constraint: `nb = K/256` super-blocks, added low→high.

5. **Across rows in the 16×16 tile:** **fully independent**. Each row
   owns its own `__m512` accumulator. There is no cross-row reduction
   anywhere in the kernel. Row `i`'s result bit pattern is unaffected
   by row `j`'s operands.

6. **Across columns (lanes) within a row:** **fully independent**. The
   16 columns of a tile sit in 16 parallel lanes of one `__m512`. No
   lane ever sees another lane's value.

7. **Mins correction:** `acc_min_rows[i]` follows the **same** order
   (per `sb`, per `b`, per row, low→high), and the subtraction
   `acc_rows[i] - acc_min_rows[i]` happens **exactly once, after the
   `b` loop has completed** — not interleaved.

### What Olorin's Eä kernel must match

For a given (row, col) output, the bit-exact recipe is:

```
acc      = 0.0_f32
acc_min  = 0.0_f32
for b in 0..nb:                              # ASCENDING
    col_d    = f16->f32(B[b].d    [col])
    col_dmin = f16->f32(B[b].dmin [col])
    row_d    = A[b].d[row]                   # already f32 in block_q8_Kx4
    for sb in 0..4:                          # ASCENDING, sb = 0,1,2,3
        # two 32-elt sub-blocks per sb iter, fully integer:
        i32_0 = i32_dot_scaled(B[b], A[b], col, row, sub=2*sb+0)
        i32_1 = i32_dot_scaled(B[b], A[b], col, row, sub=2*sb+1)
        i32   = i32_0 + i32_1                # i32 add, wraps
        acc   = fma(cvt(i32), col_d * row_d, acc)   # f32 FMA
        # mins term (one bsum per sb, spans both sub-blocks):
        i32_m = bsums_hsum(A[b], row, sb) * mins(B[b], col, sb) (i32)
        acc_min = fma(cvt(i32_m), col_dmin * row_d, acc_min)   # f32 FMA
out[row, col] = acc - acc_min                # single sub at the end
```

Notes:
- The `col_d * row_d` and `col_dmin * row_d` products are computed as a
  separate `_mm512_mul_ps` feeding the FMA's second operand — they are
  not fused with the outer FMA. This matters because `(a*b)+c` via FMA
  is single-rounded, but `mul_ps` + `fmadd_ps` rounds the `mul` first.
  The Eä kernel must match: **round col_scale * row_scale first**, then
  FMA into `acc`.
- `i32_0 + i32_1` is plain `add_epi32` (modular wrap in i32) before the
  single `cvtepi32_ps` — **do not** convert each sub-block to f32
  separately. Doing so would change the rounding boundary.
- `row_d` is already stored as f32 in `block_q8_Kx4.d[4]` — the AVX-512
  branch of `ggml_gemm_q4_K_8x8_q8_K` loads it via
  `_mm_load_ps(a_ptrs[rp][b].d)` inside the `rp` loop. No f16→f32 on the
  Q8K side.
- `col_d` and `col_dmin` are f16 in the `block_q4_Kx8` header and
  converted via `GGML_F32Cx8x2_LOAD` (two halves into one `__m512`).
  The Eä kernel must use the same f16→f32 conversion semantics
  (IEEE 754 half → float, no flush-to-zero, ties-to-even).

## Trait registration (where the dispatch lives)

In `/root/dev/llama.cpp/ggml/src/ggml-cpu/repack.cpp`, the
`q4_K_8x8_q8_K` trait instance (at the time of writing, line 4536):

```cpp
static const ggml::cpu::repack::tensor_traits<block_q4_K, 4, 8, GGML_TYPE_Q8_K> q4_K_8x4_q8_K;
static const ggml::cpu::repack::tensor_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;
```

The `8x8` variant is selected for AVX2/AVX-512 capable CPUs when
`ne[1] % 8 == 0`. Its `.gemm` member points at
`ggml_gemm_q4_K_8x8_q8_K` (the function documented above).

## What this means for Tasks 5–10

- Task 5 (`q4k_repack_8x8`): produces the layout consumed by the loads at
  `b_ptr_0[b].qs`, `.scales`, `.d`, `.dmin`. Task 1's note pins the byte
  layout; this note pins how each byte is used.
- Task 7 (`q4k_8x8_q8k_matvec`, N=1): the N=1 case is exactly **one
  row-pair iteration with `rp` fixed**, 4 rows collapsing to 1. Same
  per-`b`, per-`sb` ascending order.
- Task 9 (`q4k_8x8_q8k_gemm`, N>1): multiple row-pair iterations.
  `rp` order does not affect bit-exactness because rows are independent,
  but keeping ggml's `rp = 0..3` pair-of-pairs order simplifies the
  verify-vs-ggml test in Task 10.
- Task 10 verify: diff per-output f32 bit pattern against ggml's
  `ggml_gemm_q4_K_8x8_q8_K` output for N=2 and N=8 prompts — if the
  recipe above is followed, the bit patterns must match exactly.

## Footnote: AVX2 and generic fallbacks

- The AVX2 fallback — the `#else` branch of `ggml_gemm_q4_K_8x8_q8_K`
  (at the time of writing, lines ~2816–3487) — performs the same
  per-`b`, per-`sb`, ascending f32 accumulation in 256-bit lanes. Its
  outer tile differs from the AVX-512 path: instead of stepping
  `x += 2` to build a 16×16 tile across two `block_q4_Kx8` columns, it
  steps `x++` and keeps a 16×8 tile (one `block_q4_Kx8` at a time). The
  per-output bit-exactness claim nevertheless holds, and here is why:

  > AVX-512 has 16 i32 / f32 lanes, AVX2 has 8. Each output element of
  > the gemm — `s[(y*4 + rp)*bs + (x*8 + col)]` for some fixed
  > `(rp, col)` — owns **exactly one accumulator lane** in either
  > implementation (one lane of `acc_rows[rp*4 + k]` in the AVX-512
  > `__m512` path, and one lane of the AVX2 `__m256` counterpart). The
  > f32 reduction tree for that single output is therefore independent
  > of how many *other* lanes happen to share the SIMD register. The
  > AVX2 branch applies exactly the same per-output sequence per
  > `(b, sb, rp, k)`: integer dot via `maddubs_epi16` → i16
  > add-reductions in the same fixed order → `madd_epi16` with the
  > 6-bit sub-block scale → `add_epi32` of the two 32-elt sub-blocks →
  > `cvtepi32_ps` → `mul_ps(col_scale, broadcast(row_d_k))` →
  > `fmadd_ps` into `acc_rows[rp*4 + k]`. The `mul_ps`-then-`fmadd_ps`
  > decomposition (separate round on `col * row`, then single-rounded
  > FMA) is preserved verbatim. End-of-block `sub_ps(acc_rows,
  > acc_min_rows)` runs once after the `b` loop in both paths. Because
  > lanes never cross, narrowing from 16 to 8 lanes does not perturb
  > any per-output reduction — every per-output bit pattern is
  > identical to the AVX-512 path.

  Olorin's critical path is AVX-512 only, so the AVX2 path is not a
  verification target. Task 10 verifies bit-exactness against whichever
  ggml path the test host selects; on an AVX-512 machine that is the
  AVX-512 branch, and the AVX2 argument above is documented here only so
  future maintainers do not have to re-derive it.
- The scalar generic fallback `ggml_gemm_q4_K_8x8_q8_K_generic` (at the
  time of writing, line 3492) is reference-only and is **not** used on
  x86 when AVX2 or AVX-512 is available. Olorin does not need to match
  the generic path.
