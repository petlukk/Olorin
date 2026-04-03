# x86 Q4K Inline Scale Unpacking

## Problem

ARM Q4K kernels were optimized to read scales/mins inline from block headers (6 params).
Rust FFI wrappers were updated to match the ARM signature.
x86 Q4K kernels still expect pre-unpacked scales/mins as separate buffers (8 params).

Result: on x86, kernels receive wrong arguments — Llama produces garbage, Qwen segfaults.

## Baseline Performance (v0.6.0, pre-branch, 16 threads)

| Model | Quant | Prefill tok/s | Decode tok/s | ms/tok |
|-------|-------|---------------|-------------|--------|
| Qwen 2.5 1.5B | Q4K_M | 22.6 | 8.6 | 116.6 |
| BitNet 3B | I2_S | 19.0 | 16.9 | 59.2 |
| Llama 3.2 3B | Q4K_M | 15.2 | 5.8 | 172.4 |

## Current State (terminal-config-pipeline branch)

| Model | Status |
|-------|--------|
| BitNet I2S | OK — 16.2 tok/s decode |
| Qwen Q4K | Segfault |
| Llama Q4K | 14.9 tok/s decode but incoherent output |

## Solution

Rewrite `kernels/q4k_dot.ea` (x86) to unpack scales/mins inline from Q4K block headers,
matching the ARM kernel signature. SSSE3 baseline (pshufb, maddubs available).

## Scope

### Changed

- `kernels/q4k_dot.ea` — rewrite all three functions with inline scale unpacking

### Unchanged

- `kernels/q4k_dot_arm.ea`
- `kernels/q6k_dot.ea` (already inline scales)
- `src/inference/matmul_q4k.rs` (already ARM-compatible signatures)
- `src/inference/gemm_q4k.rs` (already ARM-compatible signatures)
- `src/kernels/ffi_inference.rs` (already ARM-compatible signatures)

## New Signature (all three functions)

### q4k_dot_q8k (single-row)

```
func q4k_dot_q8k(
    q4: *restrict u8,        // Raw Q4K block data (144 bytes/block)
    q8: *restrict i8,        // Q8K quantized activations
    bsums: *restrict i32,    // Block sums
    n_blocks: i32,
    d_arr: *restrict f32,    // Pre-computed: f16_to_f32(d) * q8_d[blk]
    dmin_arr: *restrict f32  // Pre-computed: f16_to_f32(dmin) * q8_d[blk]
) -> f32
```

### q4k_dot_q8k_4row

```
func q4k_dot_q8k_4row(
    rw0..rw3: *restrict u8,  // 4 weight rows
    q8: *restrict i8,
    bsums: *restrict i32,
    out scores: *mut f32,
    n_blocks: i32,
    d0..d3: *restrict f32,
    dm0..dm3: *restrict f32
)
```

### q4k_dot_q8k_4row_dual

```
func q4k_dot_q8k_4row_dual(
    gw0..gw3: *restrict u8,  // Gate weight rows
    uw0..uw3: *restrict u8,  // Up weight rows
    q8: *restrict i8,
    bsums: *restrict i32,
    out gate_scores: *mut f32,
    out up_scores: *mut f32,
    n_blocks: i32,
    gd0..gd3, gdm0..gdm3: *restrict f32,
    ud0..ud3, udm0..udm3: *restrict f32
)
```

## Helper Functions

Ported from ARM kernel, adapted for u8 pointers:

```
func ubyte(p: *restrict u8, off: i32) -> i32
    // Read p[off] as unsigned i32

func get_scale(p: *restrict u8, sp: i32, j: i32) -> i32
    // Extract 6-bit scale for group j (low nibble)
    // j < 2: packed[j*2] & 63
    // j >= 2: (packed[j*2+4] & 15) | ((packed[j*2-4] >> 6) << 4)

func get_scale_hi(p: *restrict u8, sp: i32, j: i32) -> i32
    // Extract 6-bit scale for group j (high nibble)
    // j < 2: packed[j*2+1] & 63
    // j >= 2: (packed[j*2+5] & 15) | ((packed[j*2-3] >> 6) << 4)

func row_mins(p: *restrict u8, sp: i32, bsums: *restrict i32, bs: i32) -> i32
    // Compute mins correction term for one row
    // Reads 8 mins from packed header, multiplies by bsums pairs
```

## Q4K Block Layout (144 bytes)

```
[0-1]:    d (f16)         — handled in Rust (unpack_d)
[2-3]:    dmin (f16)      — handled in Rust (unpack_d)
[4-15]:   12 bytes packed scales/mins (8 × 6-bit each)
[16-143]: 128 bytes packed 4-bit nibbles (256 elements)
```

Scale packing format (12 bytes at offset 4):
- Bytes 0-3: scales[0..3] low 6 bits + scales[4..7] high 2 bits
- Bytes 4-7: mins[0..3] low 6 bits + mins[4..7] high 2 bits
- Bytes 8-11: scales[4..7] low 4 bits | mins[4..7] low 4 bits

## Core Algorithm (single-row)

```
for blk in 0..n_blocks:
    bp = blk * 144
    sp = bp + 4           // Packed scales/mins
    nib = bp + 16         // Nibble data

    sumi = 0
    for j in 0..4:
        lo = maddubs_i32(nibbles & 0x0F, q8_lo)   // SSSE3
        hi = maddubs_i32(nibbles >> 4, q8_hi)      // SSSE3
        sumi += reduce(lo) * get_scale(q4, sp, j)
        sumi += reduce(hi) * get_scale_hi(q4, sp, j)

    summs = row_mins(q4, sp, bsums, blk * 16)
    result += d_arr[blk] * sumi - dmin_arr[blk] * summs
```

## Verification

1. Build with eacompute feat/i8mm-intrinsics branch
2. Run all three models: BitNet (regression check), Qwen (no segfault), Llama (coherent output)
3. Compare decode tok/s against baseline
4. Qwen and Llama must produce coherent text
5. Run existing e2e tests
