# Q4K Performance Parity with llama.cpp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the ~35% decode and ~50% prefill gap between Olorin and llama.cpp on Q4K models through three independent kernel optimizations.

**Architecture:** Three sequential optimizations to `kernels/q4k_dot.ea`, each independently measurable. Phase 1 (vector accumulation) is the biggest win (~15-20% decode). Phase 2 (pshufb scale unpacking) saves ~8 cycles/block. Phase 3 (prefill tiling) targets the prefill gap.

**Tech Stack:** Eä compiler (eacompute feat/i8mm-intrinsics), AVX2 intrinsics (maddubs_i32, shuffle_bytes), SSSE3

**Baseline (2026-04-01, 16 threads):**

| Model | Olorin prefill | Olorin decode | llama.cpp prefill | llama.cpp decode |
|-------|---------------|--------------|-------------------|-----------------|
| Qwen 2.5 1.5B Q4K | 32.3 tok/s | 10.3 tok/s | 69.6 tok/s | 17.0 tok/s |
| Llama 3.2 3B Q4K | 17.6 tok/s | 7.2 tok/s | 35.0 tok/s | 10.8 tok/s |

---

## Phase 1: Vector-Domain Accumulation

### Problem

Current kernel does `reduce_add(maddubs_i32(...))` per j-iteration, collapsing `i32x8` to scalar i32 four times per block. llama.cpp keeps partial sums in `i32x8` and reduces once per block. Each `reduce_add` costs ~4 cycles (vextracti128 + vpaddd + vpshufd × 2), so 4 extra reduces = ~12 wasted cycles per block.

For 32 blocks × 4 extra reduces × 4 cycles = ~512 cycles per row. At ~2500 cycles/row total, that's ~20%.

### Target

Accumulate `maddubs_i32` results in `i32x8 vacc`, apply scalar scales via `splat(scale) .* dot`, reduce once per block.

---

### Task 1: Rewrite q4k_dot_q8k with vector accumulation

**Files:**
- Modify: `kernels/q4k_dot.ea:47-85`

- [ ] **Step 1: Replace q4k_dot_q8k inner loop**

Replace the current `q4k_dot_q8k` function (lines 47-85) with:

```
// Single-row Q4_K × Q8_K dot product with vector-domain accumulation.
export func q4k_dot_q8k(
    q4: *restrict u8,
    q8: *restrict i8,
    bsums: *restrict i32,
    n_blocks: i32,
    d_arr: *restrict f32,
    dmin_arr: *restrict f32
) -> f32 {
    let mask_lo: u8x32 = splat(15)
    let shift4: u8x32 = splat(4)
    let mut result: f32 = 0.0

    let mut blk: i32 = 0
    while blk < n_blocks {
        let bp: i32 = blk * 144
        let nib: i32 = bp + 16
        let q8_off: i32 = blk * 256
        let sp: i32 = bp + 4

        let mut vacc: i32x8 = splat(0)

        let mut j: i32 = 0
        while j < 4 {
            let p: u8x32 = load(q4, nib + j * 32)
            let q8_lo: i8x32 = load(q8, q8_off + j * 64)
            let q8_hi: i8x32 = load(q8, q8_off + j * 64 + 32)
            let dot_lo: i32x8 = maddubs_i32(p .& mask_lo, q8_lo)
            let dot_hi: i32x8 = maddubs_i32(p .>> shift4, q8_hi)
            vacc = vacc .+ dot_lo .* splat(get_scale(q4, sp, j)) .+ dot_hi .* splat(get_scale_hi(q4, sp, j))
            j = j + 1
        }

        let sumi: i32 = reduce_add(vacc)
        let summs: i32 = row_mins(q4, sp, bsums, blk * 16)
        result = result + d_arr[blk] * to_f32(sumi) - dmin_arr[blk] * to_f32(summs)
        blk = blk + 1
    }

    return result
}
```

Key changes:
- `vacc: i32x8` replaces `sumi: i32`
- `dot_lo`/`dot_hi` stay as `i32x8` (no reduce_add per iteration)
- `splat(scale) .* dot` broadcasts scalar scale to vector
- Single `reduce_add(vacc)` per block instead of 4

- [ ] **Step 2: Build and verify compilation**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1
```

Expected: Successful build.

- [ ] **Step 3: Verify correctness**

```bash
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model Qwen2.5-1.5B-Instruct-Q4_K_M 2>&1
```

Expected: Coherent answer with "Paris". Check decode tok/s.

- [ ] **Step 4: Commit**

```bash
git add kernels/q4k_dot.ea
git commit -m "perf: vector-domain accumulation in q4k_dot_q8k — reduce once per block"
```

### Task 2: Rewrite q4k_dot_q8k_4row with vector accumulation

**Files:**
- Modify: `kernels/q4k_dot.ea:87-157`

- [ ] **Step 1: Replace q4k_dot_q8k_4row inner loop**

Replace the current `q4k_dot_q8k_4row` function (lines 87-157) with:

```
// 4-row Q4_K × Q8_K dot product with vector-domain accumulation.
export func q4k_dot_q8k_4row(
    rw0: *restrict u8, rw1: *restrict u8, rw2: *restrict u8, rw3: *restrict u8,
    q8: *restrict i8,
    bsums: *restrict i32,
    out scores: *mut f32 [cap: 4],
    n_blocks: i32,
    d0_arr: *restrict f32, d1_arr: *restrict f32, d2_arr: *restrict f32, d3_arr: *restrict f32,
    dm0_arr: *restrict f32, dm1_arr: *restrict f32, dm2_arr: *restrict f32, dm3_arr: *restrict f32
) {
    let mask_lo: u8x32 = splat(15)
    let shift4: u8x32 = splat(4)
    let mut res0: f32 = 0.0
    let mut res1: f32 = 0.0
    let mut res2: f32 = 0.0
    let mut res3: f32 = 0.0

    let mut blk: i32 = 0
    while blk < n_blocks {
        let q4_off: i32 = blk * 144 + 16
        let q8_off: i32 = blk * 256
        let sp: i32 = blk * 144 + 4
        let bs: i32 = blk * 16

        let mut v0: i32x8 = splat(0)
        let mut v1: i32x8 = splat(0)
        let mut v2: i32x8 = splat(0)
        let mut v3: i32x8 = splat(0)

        let mut j: i32 = 0
        while j < 4 {
            let q8_lo: i8x32 = load(q8, q8_off + j * 64)
            let q8_hi: i8x32 = load(q8, q8_off + j * 64 + 32)
            let sc_lo: i32x8 = splat(get_scale(rw0, sp, j))
            let sc_hi: i32x8 = splat(get_scale_hi(rw0, sp, j))

            let p0: u8x32 = load(rw0, q4_off + j * 32)
            v0 = v0 .+ maddubs_i32(p0 .& mask_lo, q8_lo) .* sc_lo .+ maddubs_i32(p0 .>> shift4, q8_hi) .* sc_hi

            let p1: u8x32 = load(rw1, q4_off + j * 32)
            v1 = v1 .+ maddubs_i32(p1 .& mask_lo, q8_lo) .* splat(get_scale(rw1, sp, j)) .+ maddubs_i32(p1 .>> shift4, q8_hi) .* splat(get_scale_hi(rw1, sp, j))

            let p2: u8x32 = load(rw2, q4_off + j * 32)
            v2 = v2 .+ maddubs_i32(p2 .& mask_lo, q8_lo) .* splat(get_scale(rw2, sp, j)) .+ maddubs_i32(p2 .>> shift4, q8_hi) .* splat(get_scale_hi(rw2, sp, j))

            let p3: u8x32 = load(rw3, q4_off + j * 32)
            v3 = v3 .+ maddubs_i32(p3 .& mask_lo, q8_lo) .* splat(get_scale(rw3, sp, j)) .+ maddubs_i32(p3 .>> shift4, q8_hi) .* splat(get_scale_hi(rw3, sp, j))

            j = j + 1
        }

        res0 = res0 + d0_arr[blk] * to_f32(reduce_add(v0)) - dm0_arr[blk] * to_f32(row_mins(rw0, sp, bsums, bs))
        res1 = res1 + d1_arr[blk] * to_f32(reduce_add(v1)) - dm1_arr[blk] * to_f32(row_mins(rw1, sp, bsums, bs))
        res2 = res2 + d2_arr[blk] * to_f32(reduce_add(v2)) - dm2_arr[blk] * to_f32(row_mins(rw2, sp, bsums, bs))
        res3 = res3 + d3_arr[blk] * to_f32(reduce_add(v3)) - dm3_arr[blk] * to_f32(row_mins(rw3, sp, bsums, bs))

        blk = blk + 1
    }

    scores[0] = res0
    scores[1] = res1
    scores[2] = res2
    scores[3] = res3
}
```

Note: row 0 pre-extracts `sc_lo`/`sc_hi` to reuse for its own scale calls. Rows 1-3 call `get_scale` inline since each row has different scales.

- [ ] **Step 2: Build and verify**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1
```

- [ ] **Step 3: Test all three models**

```bash
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model Llama-3.2-3B-Instruct-Q4_K_M 2>&1
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model Qwen2.5-1.5B-Instruct-Q4_K_M 2>&1
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model ggml-model-i2_s 2>&1
```

Expected: All three produce coherent output. Record decode/prefill tok/s.

- [ ] **Step 4: Commit**

```bash
git add kernels/q4k_dot.ea
git commit -m "perf: vector-domain accumulation in q4k_dot_q8k_4row"
```

### Task 3: Rewrite q4k_dot_q8k_4row_dual with vector accumulation

**Files:**
- Modify: `kernels/q4k_dot.ea:159-265`

- [ ] **Step 1: Replace q4k_dot_q8k_4row_dual inner loop**

Same pattern as Task 2 but for 8 accumulators (4 gate + 4 up). Replace lines 159-265 with:

```
// Dual 4-row Q4_K × Q8_K with vector-domain accumulation.
export func q4k_dot_q8k_4row_dual(
    gw0: *restrict u8, gw1: *restrict u8, gw2: *restrict u8, gw3: *restrict u8,
    uw0: *restrict u8, uw1: *restrict u8, uw2: *restrict u8, uw3: *restrict u8,
    q8: *restrict i8,
    bsums: *restrict i32,
    out gate_scores: *mut f32 [cap: 4],
    out up_scores: *mut f32 [cap: 4],
    n_blocks: i32,
    gd0: *restrict f32, gd1: *restrict f32, gd2: *restrict f32, gd3: *restrict f32,
    gdm0: *restrict f32, gdm1: *restrict f32, gdm2: *restrict f32, gdm3: *restrict f32,
    ud0: *restrict f32, ud1: *restrict f32, ud2: *restrict f32, ud3: *restrict f32,
    udm0: *restrict f32, udm1: *restrict f32, udm2: *restrict f32, udm3: *restrict f32
) {
    let mask_lo: u8x32 = splat(15)
    let shift4: u8x32 = splat(4)
    let mut gr0: f32 = 0.0
    let mut gr1: f32 = 0.0
    let mut gr2: f32 = 0.0
    let mut gr3: f32 = 0.0
    let mut ur0: f32 = 0.0
    let mut ur1: f32 = 0.0
    let mut ur2: f32 = 0.0
    let mut ur3: f32 = 0.0

    let mut blk: i32 = 0
    while blk < n_blocks {
        let q4_off: i32 = blk * 144 + 16
        let q8_off: i32 = blk * 256
        let sp: i32 = blk * 144 + 4
        let bs: i32 = blk * 16

        let mut gv0: i32x8 = splat(0)
        let mut gv1: i32x8 = splat(0)
        let mut gv2: i32x8 = splat(0)
        let mut gv3: i32x8 = splat(0)
        let mut uv0: i32x8 = splat(0)
        let mut uv1: i32x8 = splat(0)
        let mut uv2: i32x8 = splat(0)
        let mut uv3: i32x8 = splat(0)

        let mut j: i32 = 0
        while j < 4 {
            let q8_lo: i8x32 = load(q8, q8_off + j * 64)
            let q8_hi: i8x32 = load(q8, q8_off + j * 64 + 32)

            let gp0: u8x32 = load(gw0, q4_off + j * 32)
            let up0: u8x32 = load(uw0, q4_off + j * 32)
            let gp0_lo: i32x8 = maddubs_i32(gp0 .& mask_lo, q8_lo)
            let up0_lo: i32x8 = maddubs_i32(up0 .& mask_lo, q8_lo)
            let gp0_hi: i32x8 = maddubs_i32(gp0 .>> shift4, q8_hi)
            let up0_hi: i32x8 = maddubs_i32(up0 .>> shift4, q8_hi)
            gv0 = gv0 .+ gp0_lo .* splat(get_scale(gw0, sp, j)) .+ gp0_hi .* splat(get_scale_hi(gw0, sp, j))
            uv0 = uv0 .+ up0_lo .* splat(get_scale(uw0, sp, j)) .+ up0_hi .* splat(get_scale_hi(uw0, sp, j))

            let gp1: u8x32 = load(gw1, q4_off + j * 32)
            let up1: u8x32 = load(uw1, q4_off + j * 32)
            let gp1_lo: i32x8 = maddubs_i32(gp1 .& mask_lo, q8_lo)
            let up1_lo: i32x8 = maddubs_i32(up1 .& mask_lo, q8_lo)
            let gp1_hi: i32x8 = maddubs_i32(gp1 .>> shift4, q8_hi)
            let up1_hi: i32x8 = maddubs_i32(up1 .>> shift4, q8_hi)
            gv1 = gv1 .+ gp1_lo .* splat(get_scale(gw1, sp, j)) .+ gp1_hi .* splat(get_scale_hi(gw1, sp, j))
            uv1 = uv1 .+ up1_lo .* splat(get_scale(uw1, sp, j)) .+ up1_hi .* splat(get_scale_hi(uw1, sp, j))

            let gp2: u8x32 = load(gw2, q4_off + j * 32)
            let up2: u8x32 = load(uw2, q4_off + j * 32)
            let gp2_lo: i32x8 = maddubs_i32(gp2 .& mask_lo, q8_lo)
            let up2_lo: i32x8 = maddubs_i32(up2 .& mask_lo, q8_lo)
            let gp2_hi: i32x8 = maddubs_i32(gp2 .>> shift4, q8_hi)
            let up2_hi: i32x8 = maddubs_i32(up2 .>> shift4, q8_hi)
            gv2 = gv2 .+ gp2_lo .* splat(get_scale(gw2, sp, j)) .+ gp2_hi .* splat(get_scale_hi(gw2, sp, j))
            uv2 = uv2 .+ up2_lo .* splat(get_scale(uw2, sp, j)) .+ up2_hi .* splat(get_scale_hi(uw2, sp, j))

            let gp3: u8x32 = load(gw3, q4_off + j * 32)
            let up3: u8x32 = load(uw3, q4_off + j * 32)
            let gp3_lo: i32x8 = maddubs_i32(gp3 .& mask_lo, q8_lo)
            let up3_lo: i32x8 = maddubs_i32(up3 .& mask_lo, q8_lo)
            let gp3_hi: i32x8 = maddubs_i32(gp3 .>> shift4, q8_hi)
            let up3_hi: i32x8 = maddubs_i32(up3 .>> shift4, q8_hi)
            gv3 = gv3 .+ gp3_lo .* splat(get_scale(gw3, sp, j)) .+ gp3_hi .* splat(get_scale_hi(gw3, sp, j))
            uv3 = uv3 .+ up3_lo .* splat(get_scale(uw3, sp, j)) .+ up3_hi .* splat(get_scale_hi(uw3, sp, j))

            j = j + 1
        }

        gr0 = gr0 + gd0[blk] * to_f32(reduce_add(gv0)) - gdm0[blk] * to_f32(row_mins(gw0, sp, bsums, bs))
        gr1 = gr1 + gd1[blk] * to_f32(reduce_add(gv1)) - gdm1[blk] * to_f32(row_mins(gw1, sp, bsums, bs))
        gr2 = gr2 + gd2[blk] * to_f32(reduce_add(gv2)) - gdm2[blk] * to_f32(row_mins(gw2, sp, bsums, bs))
        gr3 = gr3 + gd3[blk] * to_f32(reduce_add(gv3)) - gdm3[blk] * to_f32(row_mins(gw3, sp, bsums, bs))
        ur0 = ur0 + ud0[blk] * to_f32(reduce_add(uv0)) - udm0[blk] * to_f32(row_mins(uw0, sp, bsums, bs))
        ur1 = ur1 + ud1[blk] * to_f32(reduce_add(uv1)) - udm1[blk] * to_f32(row_mins(uw1, sp, bsums, bs))
        ur2 = ur2 + ud2[blk] * to_f32(reduce_add(uv2)) - udm2[blk] * to_f32(row_mins(uw2, sp, bsums, bs))
        ur3 = ur3 + ud3[blk] * to_f32(reduce_add(uv3)) - udm3[blk] * to_f32(row_mins(uw3, sp, bsums, bs))

        blk = blk + 1
    }

    gate_scores[0] = gr0
    gate_scores[1] = gr1
    gate_scores[2] = gr2
    gate_scores[3] = gr3
    up_scores[0] = ur0
    up_scores[1] = ur1
    up_scores[2] = ur2
    up_scores[3] = ur3
}
```

- [ ] **Step 2: Build, test all three models, record perf**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model Qwen2.5-1.5B-Instruct-Q4_K_M 2>&1
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model Llama-3.2-3B-Instruct-Q4_K_M 2>&1
```

Expected: Coherent output. Decode should improve ~15-20% vs pre-Phase-1 baseline.

- [ ] **Step 3: Commit**

```bash
git add kernels/q4k_dot.ea
git commit -m "perf: vector-domain accumulation in q4k_dot_q8k_4row_dual"
```

---

## Phase 2: Scale Unpacking via pshufb

### Problem

Current inline scale unpacking uses 4 scalar bitwise ops per scale (ubyte + mask + shift + or). 8 scales per block × ~4 cycles each = ~32 cycles. llama.cpp pre-unpacks scales to 16×i16 via bitwise on u32 (~8 cycles), then broadcasts each pair with `pshufb` (1 cycle per group).

### Approach

Pre-unpack all 8 scales + 8 mins from 12-byte header to i32 array at block start (scalar bitwise, ~16 cycles). Then `splat()` from array instead of calling `get_scale()` per group. This eliminates repeated byte loads and conditional branches (j < 2 check).

### Task 4: Pre-unpack scales at block level

**Files:**
- Modify: `kernels/q4k_dot.ea` — add `unpack_scales` helper, update all three functions

- [ ] **Step 1: Add unpack_scales helper**

Add after `row_mins` function:

```
// Pre-unpack all 8 scales from 12-byte packed header into array.
// sc[0..7] = scales for lo nibble groups, sc[8..15] = scales for hi nibble groups.
// Avoids repeated conditional byte extraction in inner loop.
func unpack_scales(p: *restrict u8, sp: i32, out sc: *mut i32 [cap: 8]) {
    // Groups 0,1: direct 6-bit extraction
    sc[0] = ubyte(p, sp) & 63
    sc[1] = ubyte(p, sp + 1) & 63
    sc[2] = ubyte(p, sp + 2) & 63
    sc[3] = ubyte(p, sp + 3) & 63
    // Groups 2,3: reconstruct from upper nibble + high bits
    sc[4] = (ubyte(p, sp + 8) & 15) | ((ubyte(p, sp) >> 6) << 4)
    sc[5] = (ubyte(p, sp + 9) & 15) | ((ubyte(p, sp + 1) >> 6) << 4)
    sc[6] = (ubyte(p, sp + 10) & 15) | ((ubyte(p, sp + 2) >> 6) << 4)
    sc[7] = (ubyte(p, sp + 11) & 15) | ((ubyte(p, sp + 3) >> 6) << 4)
}

func unpack_mins(p: *restrict u8, sp: i32, out mn: *mut i32 [cap: 8]) {
    mn[0] = ubyte(p, sp + 4) & 63
    mn[1] = ubyte(p, sp + 5) & 63
    mn[2] = ubyte(p, sp + 6) & 63
    mn[3] = ubyte(p, sp + 7) & 63
    mn[4] = (ubyte(p, sp + 8) >> 4) | ((ubyte(p, sp + 4) >> 6) << 4)
    mn[5] = (ubyte(p, sp + 9) >> 4) | ((ubyte(p, sp + 5) >> 6) << 4)
    mn[6] = (ubyte(p, sp + 10) >> 4) | ((ubyte(p, sp + 6) >> 6) << 4)
    mn[7] = (ubyte(p, sp + 11) >> 4) | ((ubyte(p, sp + 7) >> 6) << 4)
}

// Mins correction using pre-unpacked mins array
func row_mins_fast(mn: *restrict i32, bsums: *restrict i32, bs: i32) -> i32 {
    let mut s: i32 = 0
    let mut k: i32 = 0
    while k < 8 {
        s = s + mn[k] * (bsums[bs + k*2] + bsums[bs + k*2 + 1])
        k = k + 1
    }
    return s
}
```

- [ ] **Step 2: Update q4k_dot_q8k to use pre-unpacked scales**

Replace inner loop to use array indexing instead of `get_scale`/`get_scale_hi`:

```
        let mut sc: i32[8]
        unpack_scales(q4, sp, sc)

        let mut j: i32 = 0
        while j < 4 {
            let p: u8x32 = load(q4, nib + j * 32)
            let q8_lo: i8x32 = load(q8, q8_off + j * 64)
            let q8_hi: i8x32 = load(q8, q8_off + j * 64 + 32)
            let dot_lo: i32x8 = maddubs_i32(p .& mask_lo, q8_lo)
            let dot_hi: i32x8 = maddubs_i32(p .>> shift4, q8_hi)
            vacc = vacc .+ dot_lo .* splat(sc[j*2]) .+ dot_hi .* splat(sc[j*2 + 1])
            j = j + 1
        }
```

Note: `sc[j*2]` maps to: j=0→sc[0] (lo scale group 0), j=1→sc[2] (lo scale group 1), etc. This matches the original `get_scale(q4, sp, 0)=sc[0]`, `get_scale_hi(q4, sp, 0)=sc[1]`, `get_scale(q4, sp, 1)=sc[2]`, etc.

**IMPORTANT**: Verify the mapping. Current code:
- `get_scale(p, sp, j)` for j=0: `ubyte(p, sp + 0) & 63` = first packed byte & 63
- `get_scale_hi(p, sp, j)` for j=0: `ubyte(p, sp + 1) & 63` = second packed byte & 63

So scale ordering is: [scale_lo_0, scale_hi_0, scale_lo_1, scale_hi_1, scale_lo_2, scale_hi_2, scale_lo_3, scale_hi_3]. The `unpack_scales` function should produce this exact ordering.

Adjust `unpack_scales` to output in interleaved order:
```
func unpack_scales(p: *restrict u8, sp: i32, out sc: *mut i32 [cap: 8]) {
    // Interleaved: [lo_0, hi_0, lo_1, hi_1, lo_2, hi_2, lo_3, hi_3]
    sc[0] = ubyte(p, sp) & 63         // get_scale(j=0)
    sc[1] = ubyte(p, sp + 1) & 63     // get_scale_hi(j=0)
    sc[2] = ubyte(p, sp + 2) & 63     // get_scale(j=1)
    sc[3] = ubyte(p, sp + 3) & 63     // get_scale_hi(j=1)
    sc[4] = (ubyte(p, sp + 8) & 15) | ((ubyte(p, sp) >> 6) << 4)       // get_scale(j=2)
    sc[5] = (ubyte(p, sp + 9) & 15) | ((ubyte(p, sp + 1) >> 6) << 4)   // get_scale_hi(j=2)
    sc[6] = (ubyte(p, sp + 10) & 15) | ((ubyte(p, sp + 2) >> 6) << 4)  // get_scale(j=3)
    sc[7] = (ubyte(p, sp + 11) & 15) | ((ubyte(p, sp + 3) >> 6) << 4)  // get_scale_hi(j=3)
}
```

Then inner loop becomes:
```
vacc = vacc .+ dot_lo .* splat(sc[j*2]) .+ dot_hi .* splat(sc[j*2 + 1])
```

- [ ] **Step 3: Update 4row and dual functions similarly**

Same pattern: `unpack_scales` per row at block start, index `sc[j*2]`/`sc[j*2+1]` in inner loop. `unpack_mins` + `row_mins_fast` replaces `row_mins`.

- [ ] **Step 4: Build, test all models, record perf, commit**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model Qwen2.5-1.5B-Instruct-Q4_K_M 2>&1
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model Llama-3.2-3B-Instruct-Q4_K_M 2>&1
git add kernels/q4k_dot.ea
git commit -m "perf: pre-unpack scales at block level — eliminate per-group branches"
```

---

## Phase 3: Prefill GEMM Tiling

### Problem

Current prefill GEMM in `gemm_q4k.rs` tiles 4 weight rows × N tokens (weight-stationary). llama.cpp tiles 16×16 with repacked interleaved weights. We don't want to repack (extra memory), but can improve tiling.

### Approach

Increase weight tile from 4 to 8 rows (using two `q4k_dot_q8k_4row` calls per tile), and add token-dimension tiling to keep Q8 activations in L1. Target: 4-8 tokens × 8 rows per tile ≈ 8×(128×32) = 32KB Q4 + 8×256×8 = 16KB Q8 = fits L1.

### Task 5: Add L1-aware token tiling to GEMM

**Files:**
- Modify: `src/inference/gemm_q4k.rs`

- [ ] **Step 1: Add token tiling constant and restructure inner loop**

In `q4k_gemm_mt` and `q4k_fused_silu_gemm_mt`, wrap the token loop with a tile:

```rust
const TOKEN_TILE: usize = 4; // Process 4 tokens per weight-load

// Current: for t in 0..nt { ... load weights ... kernel(t) ... }
// New:     for t_base in (0..nt).step_by(TOKEN_TILE) {
//              let t_end = (t_base + TOKEN_TILE).min(nt);
//              ... load weights once ...
//              for t in t_base..t_end { kernel(t) }
//          }
```

This is already roughly what the code does (weight-outer, token-inner), but verify the f16→f32 weight scales are cached ONCE per weight group, not per token. The current code already does this via `da_w`/`dma_w` arrays.

The actual improvement is to process 8 rows instead of 4 per tile, doubling Q8 reuse:

```rust
while r + 8 <= count {
    let row = start + r;
    // First 4 rows
    let ws0 = [weight.add(row * rs), ...+1, ...+2, ...+3];
    // Second 4 rows
    let ws1 = [weight.add((row+4) * rs), ...+5, ...+6, ...+7];
    
    // Cache f16→f32 for all 8 rows
    for i in 0..4 { unpack_d_cached(ws0[i], ...); }
    for i in 0..4 { unpack_d_cached(ws1[i], ...); }
    
    for t in 0..nt {
        // Multiply cached d by per-token q8_d (8 rows)
        for i in 0..4 { da0[i][blk] = da_w0[i][blk] * q; }
        for i in 0..4 { da1[i][blk] = da_w1[i][blk] * q; }
        
        // Two 4-row kernel calls sharing same Q8 (still in L1)
        ffi::q4k_dot_q8k_4row(ws0[0..3], q8, ...);
        ffi::q4k_dot_q8k_4row(ws1[0..3], q8, ...);
    }
    r += 8;
}
```

- [ ] **Step 2: Apply same pattern to fused SiLU GEMM**

Same 8-row tiling for `q4k_fused_silu_gemm_mt`: process 8 gate + 8 up rows per tile. Use two `q4k_dot_q8k_4row_dual` calls.

- [ ] **Step 3: Build, test, benchmark, commit**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model Qwen2.5-1.5B-Instruct-Q4_K_M 2>&1
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model Llama-3.2-3B-Instruct-Q4_K_M 2>&1
git add src/inference/gemm_q4k.rs
git commit -m "perf: 8-row GEMM tiling for better Q8 cache reuse in prefill"
```

---

## Verification

After all three phases, benchmark against llama.cpp:

```bash
# Olorin
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model Qwen2.5-1.5B-Instruct-Q4_K_M 2>&1
echo "the capital of France is?" | timeout 30 ./target/release/olorin --model Llama-3.2-3B-Instruct-Q4_K_M 2>&1

# llama.cpp
timeout 60 /home/peter/projects/llama.cpp/build/bin/llama-cli -m ~/.olorin/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf -n 64 -t 16 -p "the capital of France is?" -no-cnv --single-turn 2>&1
timeout 60 /home/peter/projects/llama.cpp/build/bin/llama-cli -m ~/.olorin/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -n 64 -t 16 -p "the capital of France is?" -no-cnv --single-turn 2>&1
```

**Target:** Decode within 85% of llama.cpp. Prefill within 70% of llama.cpp.
