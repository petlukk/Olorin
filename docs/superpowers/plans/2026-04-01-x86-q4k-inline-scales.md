# x86 Q4K Inline Scale Unpacking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `kernels/q4k_dot.ea` (x86) to unpack scales/mins inline from Q4K block headers, matching the ARM kernel signature so that Rust FFI calls work correctly on x86.

**Architecture:** Port the ARM kernel's inline scale-unpacking helpers (`ubyte`, `get_scale`, `get_scale_hi`, `row_mins`) to x86, adapting for `u8` pointers and `maddubs_i32` (SSSE3) instead of `vdot_i32` (NEON). The three export functions (`q4k_dot_q8k`, `q4k_dot_q8k_4row`, `q4k_dot_q8k_4row_dual`) get new signatures matching the Rust FFI wrappers.

**Tech Stack:** Eä compiler (eacompute feat/i8mm-intrinsics branch), SSSE3 intrinsics (`maddubs_i32`), scalar bitwise ops (`&`, `>>`, `<<`)

---

### Task 1: Write inline scale helpers in q4k_dot.ea

**Files:**
- Modify: `kernels/q4k_dot.ea:1-15`

The ARM kernel uses `i8` pointers with `ubyte()` to mask to unsigned. The x86 kernel uses `u8` pointers, so `ubyte()` just needs `to_i32(p[off])` without the `& 255` mask. The scale-unpacking logic is identical.

- [ ] **Step 1: Replace the file header and add helper functions**

Replace lines 1-15 of `kernels/q4k_dot.ea` with:

```
// q4k_dot.ea — Q4_K × Q8_K dot product kernel (SSSE3/AVX2)
//
// Inline 6-bit scale unpacking from Q4K block headers.
// d_arr/dmin_arr pre-computed in Rust (f16→f32 + q8_d multiply).
// bsums are i32 (not i16) because our Q8_K quant kernel produces i32 bsums.

#[cfg(x86_64)]

// Read byte from u8 pointer as i32
func ubyte(p: *restrict u8, off: i32) -> i32 {
    return to_i32(p[off])
}

// Unpack 6-bit scale at group j (0..3) from 12-byte packed header at p+sp
func get_scale(p: *restrict u8, sp: i32, j: i32) -> i32 {
    if j < 2 {
        return ubyte(p, sp + j*2) & 63
    }
    return (ubyte(p, sp + j*2 + 4) & 15) | ((ubyte(p, sp + j*2 - 4) >> 6) << 4)
}

func get_scale_hi(p: *restrict u8, sp: i32, j: i32) -> i32 {
    if j < 2 {
        return ubyte(p, sp + j*2 + 1) & 63
    }
    return (ubyte(p, sp + j*2 + 5) & 15) | ((ubyte(p, sp + j*2 - 3) >> 6) << 4)
}

// Mins correction for one row: sum(mins[k] * bsums_pair[k]) for k=0..7
func row_mins(p: *restrict u8, sp: i32, bsums: *restrict i32, bs: i32) -> i32 {
    let mut s: i32 = 0
    let mut k: i32 = 0
    while k < 4 {
        s = s + (ubyte(p, sp + 4 + k) & 63) * (bsums[bs + k*2] + bsums[bs + k*2 + 1])
        k = k + 1
    }
    k = 0
    while k < 4 {
        s = s + ((ubyte(p, sp + 8 + k) >> 4) | ((ubyte(p, sp + 4 + k) >> 6) << 4)) * (bsums[bs + 8 + k*2] + bsums[bs + 8 + k*2 + 1])
        k = k + 1
    }
    return s
}
```

- [ ] **Step 2: Verify helpers compile**

Run:
```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" ea kernels/q4k_dot.ea --check 2>&1 || true
```

This will fail because the export functions still have old signatures — that's expected. We just want to confirm the helpers parse without syntax errors. If the Eä compiler doesn't support `--check`, skip this step and verify after Task 2.

### Task 2: Rewrite q4k_dot_q8k (single-row)

**Files:**
- Modify: `kernels/q4k_dot.ea:18-66` (the `q4k_dot_q8k` function)

- [ ] **Step 1: Replace q4k_dot_q8k with inline-scale version**

Replace the `q4k_dot_q8k` function (after the helpers) with:

```
// Single-row Q4_K × Q8_K dot product with inline scale unpacking.
// d_arr/dmin_arr: per-block pre-multiplied scale arrays (length n_blocks).
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

        let mut sumi: i32 = 0

        let mut j: i32 = 0
        while j < 4 {
            let p: u8x32 = load(q4, nib + j * 32)
            let q8_lo: i8x32 = load(q8, q8_off + j * 64)
            let q8_hi: i8x32 = load(q8, q8_off + j * 64 + 32)
            let dot_lo: i32 = reduce_add(maddubs_i32(p .& mask_lo, q8_lo))
            let dot_hi: i32 = reduce_add(maddubs_i32(p .>> shift4, q8_hi))
            sumi = sumi + dot_lo * get_scale(q4, sp, j) + dot_hi * get_scale_hi(q4, sp, j)
            j = j + 1
        }

        let summs: i32 = row_mins(q4, sp, bsums, blk * 16)
        result = result + d_arr[blk] * to_f32(sumi) - dmin_arr[blk] * to_f32(summs)
        blk = blk + 1
    }

    return result
}
```

Key changes from old version:
- Removed `scales` and `mins` parameters
- `sp = bp + 4` reads scales from block header (bytes 4-15)
- `get_scale`/`get_scale_hi` replace `scales[sc_off + 2*j]` / `scales[sc_off + 2*j + 1]`
- `row_mins` replaces the manual mins loop

### Task 3: Rewrite q4k_dot_q8k_4row

**Files:**
- Modify: `kernels/q4k_dot.ea:68-154` (the `q4k_dot_q8k_4row` function)

- [ ] **Step 1: Replace q4k_dot_q8k_4row with inline-scale version**

Replace the function with:

```
// 4-row Q4_K × Q8_K dot product: 4 weight rows × shared activations.
// d0..d3 / dm0..dm3: per-block d/dmin arrays per row.
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

        let mut sumi0: i32 = 0
        let mut sumi1: i32 = 0
        let mut sumi2: i32 = 0
        let mut sumi3: i32 = 0

        let mut j: i32 = 0
        while j < 4 {
            let q8_lo: i8x32 = load(q8, q8_off + j * 64)
            let q8_hi: i8x32 = load(q8, q8_off + j * 64 + 32)

            let p0: u8x32 = load(rw0, q4_off + j * 32)
            let d0_lo: i32 = reduce_add(maddubs_i32(p0 .& mask_lo, q8_lo))
            let d0_hi: i32 = reduce_add(maddubs_i32(p0 .>> shift4, q8_hi))
            sumi0 = sumi0 + d0_lo * get_scale(rw0, sp, j) + d0_hi * get_scale_hi(rw0, sp, j)

            let p1: u8x32 = load(rw1, q4_off + j * 32)
            let d1_lo: i32 = reduce_add(maddubs_i32(p1 .& mask_lo, q8_lo))
            let d1_hi: i32 = reduce_add(maddubs_i32(p1 .>> shift4, q8_hi))
            sumi1 = sumi1 + d1_lo * get_scale(rw1, sp, j) + d1_hi * get_scale_hi(rw1, sp, j)

            let p2: u8x32 = load(rw2, q4_off + j * 32)
            let d2_lo: i32 = reduce_add(maddubs_i32(p2 .& mask_lo, q8_lo))
            let d2_hi: i32 = reduce_add(maddubs_i32(p2 .>> shift4, q8_hi))
            sumi2 = sumi2 + d2_lo * get_scale(rw2, sp, j) + d2_hi * get_scale_hi(rw2, sp, j)

            let p3: u8x32 = load(rw3, q4_off + j * 32)
            let d3_lo: i32 = reduce_add(maddubs_i32(p3 .& mask_lo, q8_lo))
            let d3_hi: i32 = reduce_add(maddubs_i32(p3 .>> shift4, q8_hi))
            sumi3 = sumi3 + d3_lo * get_scale(rw3, sp, j) + d3_hi * get_scale_hi(rw3, sp, j)

            j = j + 1
        }

        res0 = res0 + d0_arr[blk] * to_f32(sumi0) - dm0_arr[blk] * to_f32(row_mins(rw0, sp, bsums, bs))
        res1 = res1 + d1_arr[blk] * to_f32(sumi1) - dm1_arr[blk] * to_f32(row_mins(rw1, sp, bsums, bs))
        res2 = res2 + d2_arr[blk] * to_f32(sumi2) - dm2_arr[blk] * to_f32(row_mins(rw2, sp, bsums, bs))
        res3 = res3 + d3_arr[blk] * to_f32(sumi3) - dm3_arr[blk] * to_f32(row_mins(rw3, sp, bsums, bs))

        blk = blk + 1
    }

    scores[0] = res0
    scores[1] = res1
    scores[2] = res2
    scores[3] = res3
}
```

Key changes:
- Removed `sc0..sc3`, `mn0..mn3` parameters (was 8 extra pointers)
- Each row reads its own scales via `get_scale(rwN, sp, j)` where `sp = blk * 144 + 4`
- `row_mins(rwN, sp, bsums, bs)` replaces the shared mins loop

### Task 4: Rewrite q4k_dot_q8k_4row_dual

**Files:**
- Modify: `kernels/q4k_dot.ea:156-287` (the `q4k_dot_q8k_4row_dual` function)

- [ ] **Step 1: Replace q4k_dot_q8k_4row_dual with inline-scale version**

Replace the function with:

```
// Dual 4-row Q4_K × Q8_K: 4 gate rows + 4 up rows × shared activations.
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

        let mut gs0: i32 = 0
        let mut gs1: i32 = 0
        let mut gs2: i32 = 0
        let mut gs3: i32 = 0
        let mut us0: i32 = 0
        let mut us1: i32 = 0
        let mut us2: i32 = 0
        let mut us3: i32 = 0

        let mut j: i32 = 0
        while j < 4 {
            let q8_lo: i8x32 = load(q8, q8_off + j * 64)
            let q8_hi: i8x32 = load(q8, q8_off + j * 64 + 32)

            let gp0: u8x32 = load(gw0, q4_off + j * 32)
            let up0: u8x32 = load(uw0, q4_off + j * 32)
            let gd0_lo: i32 = reduce_add(maddubs_i32(gp0 .& mask_lo, q8_lo))
            let ud0_lo: i32 = reduce_add(maddubs_i32(up0 .& mask_lo, q8_lo))
            let gd0_hi: i32 = reduce_add(maddubs_i32(gp0 .>> shift4, q8_hi))
            let ud0_hi: i32 = reduce_add(maddubs_i32(up0 .>> shift4, q8_hi))
            gs0 = gs0 + gd0_lo * get_scale(gw0, sp, j) + gd0_hi * get_scale_hi(gw0, sp, j)
            us0 = us0 + ud0_lo * get_scale(uw0, sp, j) + ud0_hi * get_scale_hi(uw0, sp, j)

            let gp1: u8x32 = load(gw1, q4_off + j * 32)
            let up1: u8x32 = load(uw1, q4_off + j * 32)
            let gd1_lo: i32 = reduce_add(maddubs_i32(gp1 .& mask_lo, q8_lo))
            let ud1_lo: i32 = reduce_add(maddubs_i32(up1 .& mask_lo, q8_lo))
            let gd1_hi: i32 = reduce_add(maddubs_i32(gp1 .>> shift4, q8_hi))
            let ud1_hi: i32 = reduce_add(maddubs_i32(up1 .>> shift4, q8_hi))
            gs1 = gs1 + gd1_lo * get_scale(gw1, sp, j) + gd1_hi * get_scale_hi(gw1, sp, j)
            us1 = us1 + ud1_lo * get_scale(uw1, sp, j) + ud1_hi * get_scale_hi(uw1, sp, j)

            let gp2: u8x32 = load(gw2, q4_off + j * 32)
            let up2: u8x32 = load(uw2, q4_off + j * 32)
            let gd2_lo: i32 = reduce_add(maddubs_i32(gp2 .& mask_lo, q8_lo))
            let ud2_lo: i32 = reduce_add(maddubs_i32(up2 .& mask_lo, q8_lo))
            let gd2_hi: i32 = reduce_add(maddubs_i32(gp2 .>> shift4, q8_hi))
            let ud2_hi: i32 = reduce_add(maddubs_i32(up2 .>> shift4, q8_hi))
            gs2 = gs2 + gd2_lo * get_scale(gw2, sp, j) + gd2_hi * get_scale_hi(gw2, sp, j)
            us2 = us2 + ud2_lo * get_scale(uw2, sp, j) + ud2_hi * get_scale_hi(uw2, sp, j)

            let gp3: u8x32 = load(gw3, q4_off + j * 32)
            let up3: u8x32 = load(uw3, q4_off + j * 32)
            let gd3_lo: i32 = reduce_add(maddubs_i32(gp3 .& mask_lo, q8_lo))
            let ud3_lo: i32 = reduce_add(maddubs_i32(up3 .& mask_lo, q8_lo))
            let gd3_hi: i32 = reduce_add(maddubs_i32(gp3 .>> shift4, q8_hi))
            let ud3_hi: i32 = reduce_add(maddubs_i32(up3 .>> shift4, q8_hi))
            gs3 = gs3 + gd3_lo * get_scale(gw3, sp, j) + gd3_hi * get_scale_hi(gw3, sp, j)
            us3 = us3 + ud3_lo * get_scale(uw3, sp, j) + ud3_hi * get_scale_hi(uw3, sp, j)

            j = j + 1
        }

        gr0 = gr0 + gd0[blk] * to_f32(gs0) - gdm0[blk] * to_f32(row_mins(gw0, sp, bsums, bs))
        gr1 = gr1 + gd1[blk] * to_f32(gs1) - gdm1[blk] * to_f32(row_mins(gw1, sp, bsums, bs))
        gr2 = gr2 + gd2[blk] * to_f32(gs2) - gdm2[blk] * to_f32(row_mins(gw2, sp, bsums, bs))
        gr3 = gr3 + gd3[blk] * to_f32(gs3) - gdm3[blk] * to_f32(row_mins(gw3, sp, bsums, bs))
        ur0 = ur0 + ud0[blk] * to_f32(us0) - udm0[blk] * to_f32(row_mins(uw0, sp, bsums, bs))
        ur1 = ur1 + ud1[blk] * to_f32(us1) - udm1[blk] * to_f32(row_mins(uw1, sp, bsums, bs))
        ur2 = ur2 + ud2[blk] * to_f32(us2) - udm2[blk] * to_f32(row_mins(uw2, sp, bsums, bs))
        ur3 = ur3 + ud3[blk] * to_f32(us3) - udm3[blk] * to_f32(row_mins(uw3, sp, bsums, bs))

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

Key changes:
- Removed 16 scale/mins parameters (`gsc0..gsc3`, `gmn0..gmn3`, `usc0..usc3`, `umn0..umn3`)
- Each row reads its own scales via `get_scale(gwN/uwN, sp, j)`
- `row_mins(gwN/uwN, sp, bsums, bs)` replaces the 8 separate mins accumulators

### Task 5: Build and fix compilation errors

**Files:**
- Modify: `kernels/q4k_dot.ea` (if compile errors)

- [ ] **Step 1: Build olorin release**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1
```

Expected: Successful compilation. The Eä compiler should emit a new `libq4k_dot.so` with the updated signatures. The Rust FFI types (`Q4kDotQ8kFn`, `Q4kDot4RowFn`, `Q4kDot4RowDualFn` in `ffi_inference_types.rs:18-35`) already match the new signatures.

If errors: fix the Eä kernel syntax. Common issues:
- `maddubs_i32` requires SSSE3 — make sure `u8x32` loads work (AVX2 type)
- Scalar `&`, `>>`, `<<` operators on i32 — available on eacompute feat/i8mm-intrinsics branch
- `to_i32(p[off])` for u8 pointer — should work since eacompute has u8 → i32 conversion

- [ ] **Step 2: Fix any errors and rebuild**

If step 1 had errors, fix them in `kernels/q4k_dot.ea` and rebuild.

- [ ] **Step 3: Commit**

```bash
git add kernels/q4k_dot.ea
git commit -m "fix: x86 Q4K inline scale unpacking — match ARM kernel signature

Port ARM's inline 6-bit scale/mins unpacking to x86 kernels.
Removes separate scales/mins parameters, reads directly from
Q4K block headers. Fixes segfault on Qwen and garbage output
on Llama caused by x86/ARM signature mismatch."
```

### Task 6: Verify with all three models

- [ ] **Step 1: Run olorin and test BitNet**

```bash
./target/release/olorin --serve
```

Load BitNet (ggml-model-i2_s) via web UI, send a test prompt. Expect: coherent output, decode ~16 tok/s (regression check — BitNet uses I2S kernels, not Q4K).

- [ ] **Step 2: Test Qwen**

Reload Qwen2.5-1.5B-Instruct-Q4_K_M via web UI. Expect: NO segfault, coherent output, decode in range of baseline 8.6 tok/s.

- [ ] **Step 3: Test Llama**

Reload Llama-3.2-3B-Instruct-Q4_K_M. Expect: coherent output (not gibberish), decode tok/s reported.

- [ ] **Step 4: Record performance numbers**

Compare against baseline:

| Model | Baseline decode | Target |
|-------|----------------|--------|
| BitNet I2S | 16.9 tok/s | >= 16 tok/s |
| Qwen Q4K | 8.6 tok/s | >= 8 tok/s, no crash |
| Llama Q4K | 5.8 tok/s | >= 5.5 tok/s, coherent |

### Task 7: Update C test to match new kernel signature

**Files:**
- Modify: `tests/test_q4k_kernel.c`

The C test (`tests/test_q4k_kernel.c`) uses the old kernel signature with separate `scales`/`mins` parameters. It needs updating to match the new inline-scale signature, or it should be deleted since it can't test inline unpacking from outside the kernel.

- [ ] **Step 1: Update the C test typedef and reference function**

The kernel now reads scales from the Q4K block data itself. The C test needs to:
1. Pack scales into the Q4K block header (bytes 4-15) instead of passing separate arrays
2. Update the kernel function typedef to remove scales/mins params
3. Update the reference function to unpack scales inline

Replace the kernel typedef (line 17-18):

```c
// Kernel function type — new signature: inline scale unpacking
typedef float (*q4k_dot_fn)(
    const uint8_t* q4, const int8_t* q8, const int32_t* bsums,
    int32_t n_blocks, const float* d_arr, const float* dmin_arr);
```

Replace `ref_q4k_dot` (lines 64-115) with a version that reads scales from q4 block header:

```c
// Pure-C reference: Q4_K × Q8_K dot product with inline scale unpacking
// Q4_K layout per block (144 bytes):
//   [0..1]   f16 d           (handled externally as d_arr)
//   [2..3]   f16 dmin        (handled externally as dmin_arr)
//   [4..15]  12 bytes packed scales/mins
//   [16..143] 128 bytes: packed nibbles
float ref_q4k_dot(
    const uint8_t* q4, const int8_t* q8, const int32_t* bsums,
    int n_blocks, const float* d_arr, const float* dmin_arr)
{
    float result = 0.0f;
    for (int blk = 0; blk < n_blocks; blk++) {
        int bp = blk * 144;
        int sp = bp + 4;
        int nib = bp + 16;
        int q8_off = blk * 256;
        int bs_off = blk * 16;

        // Unpack scales from block header
        uint8_t scales[8], mins[8];
        unpack_scales(&q4[sp], scales, mins);

        int sumi = 0;
        for (int j = 0; j < 4; j++) {
            int dot_lo = 0;
            for (int i = 0; i < 32; i++)
                dot_lo += (q4[nib + j * 32 + i] & 0x0F) * q8[q8_off + j * 64 + i];
            int dot_hi = 0;
            for (int i = 0; i < 32; i++)
                dot_hi += (q4[nib + j * 32 + i] >> 4) * q8[q8_off + j * 64 + 32 + i];
            sumi += dot_lo * scales[2 * j] + dot_hi * scales[2 * j + 1];
        }

        int summs = 0;
        for (int j = 0; j < 8; j++) {
            summs += mins[j] * (bsums[bs_off + 2 * j] + bsums[bs_off + 2 * j + 1]);
        }

        result += d_arr[blk] * (float)sumi - dmin_arr[blk] * (float)summs;
    }
    return result;
}
```

Update test data to use full 144-byte Q4K blocks (pack scales into bytes 4-15, nibbles into bytes 16-143, d/dmin into d_arr/dmin_arr arrays). Update all test calls from:
```c
kernel(q4, q8, bsums, scales, mins, n_blocks, d, dmin)
```
to:
```c
kernel(q4_blocks, q8, bsums, n_blocks, d_arr, dmin_arr)
```

Where `q4_blocks` is a properly laid out 144-byte-per-block array with packed scales at offset 4 and nibbles at offset 16.

- [ ] **Step 2: Build and run the C test**

```bash
gcc -O2 -o test_q4k_kernel tests/test_q4k_kernel.c -ldl -lm && ./test_q4k_kernel
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_q4k_kernel.c
git commit -m "test: update Q4K C test for inline scale kernel signature"
```

### Task 8: Run existing test suite

- [ ] **Step 1: Run cargo test**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test 2>&1
```

Expected: All existing tests pass. The inference tests should work since they go through the same FFI path.

- [ ] **Step 2: Fix any failures**

If tests fail, investigate and fix. Common issues:
- `test_gguf_parse` might fail if kernel loading checks signature mismatches
- Any test that directly calls `q4k_dot_q8k` would need signature update (but all calls go through `ffi_inference.rs` which already has the right signature)
