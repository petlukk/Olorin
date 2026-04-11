# Weight Repacking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **HARD RULES (apply to ALL agents):**
> - No file exceeds 500 lines. Split before you hit the limit.
> - Every feature proven by end-to-end test. If it's not tested, it doesn't exist.
> - No fake functions. No silent fallbacks.
> - Olorin is Ea's showcase — every SIMD op must be an Ea kernel. Do NOT simplify kernel code to scalar Rust.
> - Match llama.cpp **bit-exact**, not "close enough".
> - eacompute compiler: `~/projects/eacompute/target/release/ea`
> - Build: `PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release`
> - Branch: `gemma4-batched-prompt-eval`

**Goal:** Repack Q4K weight matrices from standard GGUF row-major layout into 8-row interleaved layout matching llama.cpp's `block_q4_Kx8`, then implement Ea SIMD matvec kernels that consume the repacked layout for 8-row parallel dot products. This is Phase 1 of matching llama.cpp's remaining performance features (repack → batch → flash).

**Architecture:**
- At model load, Q4K weights are repacked into `block_q4_Kx8` layout: 8 consecutive rows' blocks interleaved so that one SIMD pass computes 8 dot products simultaneously.
- A new Ea kernel `q4k_8x8_q8k_matvec` consumes the repacked layout and outputs 8 scores per tile.
- The repack itself is an Ea kernel (`q4k_repack_8x8`) — not Rust scalar code.
- Q5K/Q6K repack follows the same pattern but is a separate follow-up (Q5K/Q6K are not on the critical path for prompt eval — Q4K dominates weight matrices).
- Decode path (`forward_one` / `forward_one_graph`) switches to repacked weights. No separate weight buffer — repack is done in-place at load time.

**Tech Stack:** Rust, Ea (eacompute), x86 AVX2 + ARM NEON, ggml Q4K format.

**Scope:**
- Q4K repack and repacked matvec only (Q5K/Q6K follow-up).
- Single-token decode path only (batched prompt eval is Phase 2).
- Both x86 and ARM kernels.

---

## Per-Commit Verification Gate

After every code-changing task, before `git commit`, **all of the following must pass**:

**Gate 1: Build clean.**
```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
```
Zero errors.

**Gate 2: Bit-exact decode regression.**
```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression -- --nocapture 2>&1 | tail -5
```
Must pass — `forward_one_bos_logits_bit_exact` ensures the decode path didn't drift.

**Gate 3: gemma4_verify suite.**
```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --test-threads=1 --nocapture 2>&1 | tail -15
```
All steps must pass.

**Gate 4: Line limit.**
```bash
find src/ kernels/ -name "*.rs" -o -name "*.ea" | xargs wc -l | awk '$1 > 500 && $2 != "total" {print}'
```
Empty output.

---

## How llama.cpp Does It (Reference)

### block_q4_K (standard, 144 bytes)

```
Offset 0-1:    d     (f16) — super-block scale
Offset 2-3:    dmin  (f16) — super-block min scale
Offset 4-15:   scales[12]  — 8 scales + 8 mins, 6-bit packed
Offset 16-143: qs[128]     — 256 x 4-bit quants (2 per byte)
```

### block_q4_Kx8 (repacked, 1152 bytes = 8 x 144)

Interleaves 8 blocks from 8 consecutive rows at the same column position:

```
Offset 0-15:     d[8]      (f16 x 8)  — scale from each of the 8 rows
Offset 16-31:    dmin[8]   (f16 x 8)  — min from each of the 8 rows
Offset 32-127:   scales[96]           — 8 rows' 6-bit scales repacked
Offset 128-1151: qs[1024]             — 8 rows' quants interleaved
```

**Quant interleaving (blck_size=8):** Take 8 bytes from row 0, then 8 bytes from row 1, ..., row 7, then next 8 bytes from row 0, etc. This means a single 256-bit AVX2 load grabs quants from 4 rows simultaneously.

**Scale repacking:** Each 12-byte scale group covers one sub-block across all 8 rows. 6-bit values are re-packed with high bits shifted: `(s[j] & 63) + ((s[j+4] & 48) << 2)` pattern, with remaining 4-bit pairs in bytes 8-11.

Source: `llama.cpp/ggml/src/ggml-cpu/repack.cpp:2836-2911` (make_block_q4_Kx8), `repack.cpp:3231-3260` (repack_q4_K_to_q4_K_8_bl).

### x86 AVX2 consumption

`ggml_gemv_q4_K_8x8_q8_K` in `arch/x86/repack.cpp:1464-1685`:
- Outer loop: per Q8K block
- Inner loop: per sub-block (QK_K/64 = 4 sub-blocks)
- Loads 256 bytes of interleaved quants via `_mm256_loadu_si256`
- Extracts low/high nibbles via `_mm256_and_si256` + `_mm256_srli_epi16`
- `_mm256_maddubs_epi16` for u8 x i8 multiply-accumulate
- Scale decode from 12-byte groups using `kmask1/2/3` bit extraction
- Accumulates 8 separate f32 results

### ARM NEON consumption

`ggml_gemv_q4_K_8x8_q8_K` in `arch/arm/repack.cpp:709-861`:
- Same structure but with NEON intrinsics
- `vld1q_u8` for quant loading
- `ggml_vdotq_s32` for dot products
- `decode_q_Kx8_6bit_scales()` helper for scale extraction

---

## Task 1: Ea Repack Kernel — x86

**Files:**
- Create: `kernels/q4k_repack.ea`
- Test: `tests/repack_q4k.rs`

### What this kernel does

Takes the standard Q4K weight buffer (n_rows x row_bytes) and writes it into q4Kx8 interleaved format. For every group of 8 rows at the same column block position, it:
1. Copies d[0..7] and dmin[0..7] from the 8 source blocks
2. Interleaves quants in 8-byte chunks (round-robin across 8 rows)
3. Repacks the 6-bit scales into the grouped format

### Steps

- [ ] **Step 1: Write the repack test**

Create `tests/repack_q4k.rs`:

```rust
//! Test: Q4K repack matches llama.cpp golden reference.

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

/// Repack a small slice (8 rows, n_blocks columns) and verify
/// against the golden reference from llama.cpp.
#[test]
fn repack_q4k_matches_golden() {
    let input = std::fs::read("tests/fixtures/q4k_repack/input.bin")
        .expect("missing input.bin fixture");
    let golden = std::fs::read("tests/fixtures/q4k_repack/golden.bin")
        .expect("missing golden.bin fixture");

    olorin::kernels::ffi::init().unwrap();

    // input.bin: 8 rows of Q4K blocks. 13824 bytes = 8 rows x 12 blocks x 144 bytes
    // (12 blocks = 3072 / 256 cols, but check actual size)
    let n_rows = 8;
    let row_bytes = input.len() / n_rows;
    let n_blocks_per_row = row_bytes / 144;
    let n_cols = n_blocks_per_row * 256;

    let mut output = vec![0u8; input.len()];
    unsafe {
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            input.as_ptr(),
            output.as_mut_ptr(),
            n_rows as i32,
            n_cols as i32,
        );
    }

    assert_eq!(output.len(), golden.len(), "size mismatch");
    assert_eq!(output, golden, "repack output doesn't match llama.cpp golden");
}

/// Repack full weight matrix from model, verify round-trip via
/// repacked matvec producing same results as standard matvec.
#[test]
fn repack_q4k_roundtrip_matvec() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model");
        return;
    }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    // Layer 0 wq — Q4K
    let lw = &model.layers[0];
    let n_rows = model.n_heads[0] * model.head_dim_k[0];
    let n_cols = model.hidden_dim;
    let n_blocks = n_cols / 256;
    let row_bytes = n_blocks * 144;

    // Standard matvec on un-repacked weights
    let input = vec![0.01f32; n_cols];
    let pow2 = olorin::inference::matmul::pow2_table();
    let mut q8_qs = vec![0i8; n_cols + 12];
    let mut q8_d = vec![0.0f32; n_blocks];
    let mut q8_bsums = vec![0i16; n_blocks * 16];
    unsafe {
        olorin::kernels::ffi_inference::quant_f32_q8k(
            input.as_ptr(), q8_qs.as_mut_ptr(), q8_d.as_mut_ptr(),
            q8_bsums.as_mut_ptr(), n_cols as i32,
        );
    }

    let mut standard_out = vec![0.0f32; n_rows];
    olorin::inference::matmul::q4k_matvec(
        lw.wq, &q8_qs, &q8_d, &q8_bsums, &pow2,
        &mut standard_out, n_rows, n_cols,
    );

    // Repack
    let mut packed = vec![0u8; n_rows * row_bytes];
    unsafe {
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.wq, packed.as_mut_ptr(), n_rows as i32, n_cols as i32,
        );
    }

    // Repacked matvec
    let mut repacked_out = vec![0.0f32; n_rows];
    unsafe {
        olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
            packed.as_ptr(), q8_qs.as_ptr(), q8_d.as_ptr(),
            q8_bsums.as_ptr(), pow2.as_ptr(),
            std::ptr::null_mut(), // scratch
            repacked_out.as_mut_ptr(), n_rows as i32, n_cols as i32,
        );
    }

    // Compare — must be bit-exact since we're doing same math in same order
    for i in 0..n_rows {
        assert_eq!(
            standard_out[i].to_bits(), repacked_out[i].to_bits(),
            "row {i}: standard={} repacked={}",
            standard_out[i], repacked_out[i],
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test repack_q4k -- --nocapture 2>&1 | tail -10
```

Expected: compilation error — `q4k_repack_8x8` and `q4k_8x8_q8k_matvec` don't exist in ffi_inference yet.

- [ ] **Step 3: Write the Ea repack kernel (x86)**

Create `kernels/q4k_repack.ea`:

```ea
// q4k_repack.ea — Repack standard Q4K blocks into 8-row interleaved layout.
//
// Matches llama.cpp make_block_q4_Kx8() with blck_size_interleave=8.
// Input:  n_rows * (n_cols/256) standard block_q4_K (144 bytes each)
// Output: (n_rows/8) * (n_cols/256) block_q4_Kx8 (1152 bytes each)
//
// The kernel repacks in groups of 8 rows. For each column block position,
// it gathers the 8 source blocks and interleaves them.

#[cfg(x86_64)]

// Repack one group of 8 Q4K blocks (from 8 rows, same column) into one q4Kx8 block.
//
// src: pointer to first row's block (stride = row_bytes between consecutive rows)
// dst: pointer to output q4Kx8 block (1152 bytes)
// row_bytes: byte stride between rows in the source weight matrix
func repack_one(src: *restrict u8, dst: *mut u8, row_bytes: i32) {
    // Part A: Copy d[0..7] and dmin[0..7] (2 bytes each, 8 rows)
    // d is at offset 0, dmin at offset 2 in each 144-byte block
    let mut i: i32 = 0
    while i < 8 {
        let blk: *restrict u8 = src + i * row_bytes
        // d: 2 bytes at offset 0 → dst offset i*2
        dst[i * 2]     = blk[0]
        dst[i * 2 + 1] = blk[1]
        // dmin: 2 bytes at offset 2 → dst offset 16 + i*2
        dst[16 + i * 2]     = blk[2]
        dst[16 + i * 2 + 1] = blk[3]
        i = i + 1
    }

    // Part B: Interleave quants (offset 16..143 in source = 128 bytes per block)
    // blck_size_interleave = 8: take 8 bytes from row 0, 8 from row 1, ..., row 7, repeat
    // Total: 8 * 128 = 1024 bytes output at dst offset 128
    // Source quant offset within block: 16
    let qout: *mut u8 = dst + 128  // quants start at offset 128 in q4Kx8
    let n_chunks: i32 = 128 / 8    // 16 chunks of 8 bytes per row
    let mut chunk: i32 = 0
    while chunk < n_chunks {
        let mut row: i32 = 0
        while row < 8 {
            let src_blk: *restrict u8 = src + row * row_bytes
            let src_off: i32 = 16 + chunk * 8  // offset within block (16 = quant start)
            let dst_off: i32 = (chunk * 8 + row) * 8
            // Copy 8 bytes using a 64-bit load/store
            let v: i64 = load(src_blk, src_off)
            store(qout, dst_off, v)
            row = row + 1
        }
        chunk = chunk + 1
    }

    // Part C: Repack scales (offset 4..15 in source = 12 bytes per block)
    // Output scales at dst offset 32, 96 bytes total
    // This follows make_block_q4_Kx8 exactly.
    let sout: *mut u8 = dst + 32
    // Sub-blocks 0..3 (first loop in llama.cpp)
    let mut sb: i32 = 0
    while sb < 4 {
        // Extract 6-bit scales and mins from each of the 8 blocks
        let mut s: [i32; 8] = [0; 8]
        let mut m: [i32; 8] = [0; 8]
        let mut j: i32 = 0
        while j < 8 {
            let blk: *restrict u8 = src + j * row_bytes
            s[j] = to_i32(blk[4 + sb]) & 63
            m[j] = to_i32(blk[4 + sb + 4]) & 63
            j = j + 1
        }
        // Pack: low 6 bits of s[0..3] + high 2 bits of s[4..7]
        sout[sb * 12 + 0] = to_u8((s[0] & 63) + ((s[4] & 48) << 2))
        sout[sb * 12 + 1] = to_u8((s[1] & 63) + ((s[5] & 48) << 2))
        sout[sb * 12 + 2] = to_u8((s[2] & 63) + ((s[6] & 48) << 2))
        sout[sb * 12 + 3] = to_u8((s[3] & 63) + ((s[7] & 48) << 2))
        // Pack mins
        sout[sb * 12 + 4] = to_u8((m[0] & 63) + ((m[4] & 48) << 2))
        sout[sb * 12 + 5] = to_u8((m[1] & 63) + ((m[5] & 48) << 2))
        sout[sb * 12 + 6] = to_u8((m[2] & 63) + ((m[6] & 48) << 2))
        sout[sb * 12 + 7] = to_u8((m[3] & 63) + ((m[7] & 48) << 2))
        // Remaining 4-bit pairs
        sout[sb * 12 + 8]  = to_u8((s[4] & 15) + ((m[4] & 15) << 4))
        sout[sb * 12 + 9]  = to_u8((s[5] & 15) + ((m[5] & 15) << 4))
        sout[sb * 12 + 10] = to_u8((s[6] & 15) + ((m[6] & 15) << 4))
        sout[sb * 12 + 11] = to_u8((s[7] & 15) + ((m[7] & 15) << 4))
        sb = sb + 1
    }
    // Sub-blocks 4..7 (second loop — upper bits)
    sb = 0
    while sb < 4 {
        let mut s: [i32; 8] = [0; 8]
        let mut m: [i32; 8] = [0; 8]
        let mut j: i32 = 0
        while j < 8 {
            let blk: *restrict u8 = src + j * row_bytes
            let i_idx: i32 = sb + 4
            s[j] = ((to_i32(blk[4 + i_idx]) & 192) >> 2) | (to_i32(blk[4 + i_idx + 8]) & 15)
            m[j] = ((to_i32(blk[4 + i_idx + 4]) & 192) >> 2) | ((to_i32(blk[4 + i_idx + 8]) & 240) >> 4)
            j = j + 1
        }
        let off: i32 = (sb + 4) * 12
        sout[off + 0] = to_u8((s[0] & 63) + ((s[4] & 48) << 2))
        sout[off + 1] = to_u8((s[1] & 63) + ((s[5] & 48) << 2))
        sout[off + 2] = to_u8((s[2] & 63) + ((s[6] & 48) << 2))
        sout[off + 3] = to_u8((s[3] & 63) + ((s[7] & 48) << 2))
        sout[off + 4] = to_u8((m[0] & 63) + ((m[4] & 48) << 2))
        sout[off + 5] = to_u8((m[1] & 63) + ((m[5] & 48) << 2))
        sout[off + 6] = to_u8((m[2] & 63) + ((m[6] & 48) << 2))
        sout[off + 7] = to_u8((m[3] & 63) + ((m[7] & 48) << 2))
        sout[off + 8]  = to_u8((s[4] & 15) + ((m[4] & 15) << 4))
        sout[off + 9]  = to_u8((s[5] & 15) + ((m[5] & 15) << 4))
        sout[off + 10] = to_u8((s[6] & 15) + ((m[6] & 15) << 4))
        sout[off + 11] = to_u8((s[7] & 15) + ((m[7] & 15) << 4))
        sb = sb + 1
    }
}

// Public entry: repack entire weight matrix.
// src: standard Q4K weight matrix, row-major
// dst: output buffer (same total size)
// n_rows: must be multiple of 8
// n_cols: must be multiple of 256
export func q4k_repack_8x8(
    src: *restrict u8, dst: *mut u8,
    n_rows: i32, n_cols: i32
) {
    let n_blocks: i32 = n_cols / 256
    let row_bytes: i32 = n_blocks * 144
    let block_8_bytes: i32 = 1152  // 8 * 144

    let mut row8: i32 = 0
    while row8 < n_rows / 8 {
        let mut blk: i32 = 0
        while blk < n_blocks {
            let src_ptr: *restrict u8 = src + row8 * 8 * row_bytes + blk * 144
            let dst_ptr: *mut u8 = dst + (row8 * n_blocks + blk) * block_8_bytes
            repack_one(src_ptr, dst_ptr, row_bytes)
            blk = blk + 1
        }
        row8 = row8 + 1
    }
}
```

**Note:** This is a data-layout kernel (memcpy-like), not a compute kernel. SIMD helps with the 8-byte copy but the scale repacking is inherently scalar-ish. The key is getting the byte layout exactly right.

- [ ] **Step 4: Wire up FFI for repack kernel**

Add to `src/kernels/ffi_inference_types.rs`:
```rust
pub type Q4kRepack8x8Fn = unsafe extern "C" fn(
    src: *const u8, dst: *mut u8, n_rows: i32, n_cols: i32,
);
```

Add to `KernelTableInference` in `src/kernels/ffi_inference.rs`:
```rust
pub q4k_repack_8x8: Q4kRepack8x8Fn,
```

Load in `load_inference_kernels`:
```rust
let q4k_repack_lib = load("q4k_repack")?;
// ...
q4k_repack_8x8: std::mem::transmute(sym(&q4k_repack_lib, b"q4k_repack_8x8\0")?),
```

Add public wrapper:
```rust
pub unsafe fn q4k_repack_8x8(
    src: *const u8, dst: *mut u8, n_rows: i32, n_cols: i32,
) {
    (k().q4k_repack_8x8)(src, dst, n_rows, n_cols)
}
```

- [ ] **Step 5: Run golden test**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test repack_q4k repack_q4k_matches_golden -- --nocapture 2>&1 | tail -10
```

Expected: PASS (byte-exact match against llama.cpp golden reference).
If FAIL: compare byte-by-byte to find which section (d, dmin, scales, quants) diverges.

- [ ] **Step 6: Commit repack kernel**

```bash
git add kernels/q4k_repack.ea src/kernels/ffi_inference.rs src/kernels/ffi_inference_types.rs tests/repack_q4k.rs
git commit -m "feat: Q4K repack kernel matching llama.cpp block_q4_Kx8 layout"
```

---

## Task 2: Ea Repacked Matvec Kernel — x86

**Files:**
- Create: `kernels/q4k_dot_8x8.ea`
- Modify: `src/kernels/ffi_inference.rs`
- Modify: `src/kernels/ffi_inference_types.rs`

### What this kernel does

Given repacked Q4K weights (block_q4_Kx8 layout) and a Q8K-quantized input vector, compute 8 dot products simultaneously — one per interleaved row. This replaces the current 4-row-at-a-time approach with 8-row-at-a-time using the cache-friendly interleaved layout.

The kernel mirrors `ggml_gemv_q4_K_8x8_q8_K` from `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp:1464-1685`.

### Steps

- [ ] **Step 1: Write the Ea kernel**

Create `kernels/q4k_dot_8x8.ea`:

```ea
// q4k_dot_8x8.ea — Q4Kx8 repacked matvec kernel (x86 AVX2).
//
// Processes one block_q4_Kx8 tile (1152 bytes = 8 interleaved Q4K blocks)
// against one Q8K input block, producing 8 f32 dot products.
//
// Matches llama.cpp ggml_gemv_q4_K_8x8_q8_K (arch/x86/repack.cpp:1464).

#[cfg(x86_64)]

// Decode 6-bit scales from the repacked 12-byte group.
// Returns 8 scales and 8 mins as i32 values.
// sptr: pointer to 12-byte scale group within block_q4_Kx8.scales
func decode_scales(
    sptr: *restrict u8,
    out_scales: *mut i32,
    out_mins: *mut i32
) {
    // Bytes 0-3: low 6 bits of scales[0..3] + high 2 bits of scales[4..7]
    // Bytes 4-7: low 6 bits of mins[0..3] + high 2 bits of mins[4..7]
    // Bytes 8-11: remaining 4 bits of scales[4..7] and mins[4..7]
    let mut i: i32 = 0
    while i < 4 {
        out_scales[i] = to_i32(sptr[i]) & 63
        out_mins[i] = to_i32(sptr[4 + i]) & 63
        i = i + 1
    }
    // Reconstruct upper scales/mins from packed bytes
    i = 0
    while i < 4 {
        let hi_s: i32 = (to_i32(sptr[i]) >> 6) << 4
        let lo_s: i32 = to_i32(sptr[8 + i]) & 15
        out_scales[4 + i] = hi_s | lo_s
        let hi_m: i32 = (to_i32(sptr[4 + i]) >> 6) << 4
        let lo_m: i32 = (to_i32(sptr[8 + i]) >> 4) & 15
        out_mins[4 + i] = hi_m | lo_m
        i = i + 1
    }
}

// Process the full weight matrix (n_rows/8 tiles, n_blocks_per_row blocks each).
//
// packed: repacked Q4Kx8 weight buffer
// q8_qs: Q8K quantized input (n_cols + 12 bytes)
// q8_d: Q8K per-block scales (n_blocks floats)
// q8_bsums: Q8K per-block subset sums (n_blocks * 16 i16s)
// pow2: f16→f32 exponent table (32 floats)
// scratch: 144 bytes scratch (unused, kept for API compat)
// out: output scores (n_rows floats)
// n_rows: total rows (must be multiple of 8)
// n_cols: total columns (must be multiple of 256)
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
    let n_blocks: i32 = n_cols / 256
    let tile_bytes: i32 = 1152  // block_q4_Kx8 size
    let m4b: u8x32 = splat_u8x32(15)  // 0x0f mask for nibble extraction
    let n_tiles: i32 = n_rows / 8

    let mut tile: i32 = 0
    while tile < n_tiles {
        // Accumulate 8 f32 sums for this tile's 8 rows
        let mut sums: [f32; 8] = [0.0; 8]

        let mut blk: i32 = 0
        while blk < n_blocks {
            let bp: *restrict u8 = packed + (tile * n_blocks + blk) * tile_bytes

            // d[0..7] at offset 0, dmin[0..7] at offset 16
            // Convert f16 scales to f32
            let mut d_vals: [f32; 8] = [0.0; 8]
            let mut dmin_vals: [f32; 8] = [0.0; 8]
            let mut r: i32 = 0
            while r < 8 {
                let raw_d: i32 = to_i32(bp[r * 2]) | (to_i32(bp[r * 2 + 1]) << 8)
                let exp_d: i32 = (raw_d >> 10) & 31
                if exp_d == 0 {
                    d_vals[r] = pow2[1] * to_f32(raw_d & 1023) * 0.0009765625
                } else {
                    d_vals[r] = pow2[exp_d] * (1.0 + to_f32(raw_d & 1023) * 0.0009765625)
                }
                let raw_m: i32 = to_i32(bp[16 + r * 2]) | (to_i32(bp[16 + r * 2 + 1]) << 8)
                let exp_m: i32 = (raw_m >> 10) & 31
                if exp_m == 0 {
                    dmin_vals[r] = pow2[1] * to_f32(raw_m & 1023) * 0.0009765625
                } else {
                    dmin_vals[r] = pow2[exp_m] * (1.0 + to_f32(raw_m & 1023) * 0.0009765625)
                }
                r = r + 1
            }

            let q8_d_blk: f32 = q8_d[blk]
            let q8_base: *restrict i8 = q8_qs + blk * 256

            // Process 4 sub-blocks (each 64 elements = 256/4)
            let mut sb: i32 = 0
            while sb < 4 {
                // Decode scales for this sub-block across 8 rows
                let mut sc: [i32; 8] = [0; 8]
                let mut mn: [i32; 8] = [0; 8]
                decode_scales(bp + 32 + sb * 24, &sc, &mn)

                // Load Q8K input for this sub-block (64 i8 values)
                let q8_0: i8x32 = load(q8_base, sb * 64)
                let q8_1: i8x32 = load(q8_base, sb * 64 + 32)

                // Load interleaved Q4K quants for this sub-block (256 bytes = 8 rows x 32 bytes)
                // In the q4Kx8 layout, quants for sub-block sb start at offset 128 + sb*256
                let qbase: *restrict u8 = bp + 128 + sb * 256

                // Process 8 rows — quants are interleaved in 8-byte chunks
                // Each row contributes 32 bytes of quants per sub-block
                // Load all 256 bytes (8 x 256-bit loads)
                let raw0: u8x32 = load(qbase, 0)      // bytes 0-31: rows 0-3 low
                let raw1: u8x32 = load(qbase, 32)     // bytes 32-63: rows 4-7 low
                let raw2: u8x32 = load(qbase, 64)     // bytes 64-95: rows 0-3 high
                let raw3: u8x32 = load(qbase, 96)     // bytes 96-127: rows 4-7 high
                let raw4: u8x32 = load(qbase, 128)    // second half...
                let raw5: u8x32 = load(qbase, 160)
                let raw6: u8x32 = load(qbase, 192)
                let raw7: u8x32 = load(qbase, 224)

                // Extract low and high nibbles
                let lo0: u8x32 = raw0 & m4b
                let hi0: u8x32 = (raw0 >> 4) & m4b
                let lo1: u8x32 = raw1 & m4b
                let hi1: u8x32 = (raw1 >> 4) & m4b
                let lo2: u8x32 = raw2 & m4b
                let hi2: u8x32 = (raw2 >> 4) & m4b
                let lo3: u8x32 = raw3 & m4b
                let hi3: u8x32 = (raw3 >> 4) & m4b
                let lo4: u8x32 = raw4 & m4b
                let hi4: u8x32 = (raw4 >> 4) & m4b
                let lo5: u8x32 = raw5 & m4b
                let hi5: u8x32 = (raw5 >> 4) & m4b
                let lo6: u8x32 = raw6 & m4b
                let hi6: u8x32 = (raw6 >> 4) & m4b
                let lo7: u8x32 = raw7 & m4b
                let hi7: u8x32 = (raw7 >> 4) & m4b

                // maddubs: u8 x i8 → i16 accumulate, then reduce per row
                // Each maddubs gives us the raw dot product for a pair of rows
                let prod_lo0: i16x16 = maddubs_i16(lo0, q8_0)
                let prod_hi0: i16x16 = maddubs_i16(hi0, q8_0)
                let prod_lo1: i16x16 = maddubs_i16(lo1, q8_1)
                let prod_hi1: i16x16 = maddubs_i16(hi1, q8_1)

                // For each row r, the raw dot = scale * sum - dmin * bsum_correction
                // This is simplified — the actual accumulation per row requires
                // separating the interleaved per-row contributions.
                //
                // NOTE: The exact SIMD reduction pattern here is a placeholder.
                // The implementor must follow the llama.cpp x86 kernel exactly,
                // matching the maddubs → madd → hadd reduction chain.
                // See arch/x86/repack.cpp lines 1620-1660 for the precise pattern.

                // Accumulate per-row sums with scales and mins correction
                r = 0
                while r < 8 {
                    // bsums correction: sum of Q8K values for this sub-block x min
                    let bs_idx: i32 = blk * 16 + sb * 2
                    let bsum_pair: i32 = to_i32(q8_bsums[bs_idx]) + to_i32(q8_bsums[bs_idx + 1])
                    sums[r] = sums[r] + d_vals[r] * q8_d_blk * to_f32(sc[r]) * to_f32(0 /* raw_dot_r */)
                                       - dmin_vals[r] * q8_d_blk * to_f32(mn[r]) * to_f32(bsum_pair)
                    r = r + 1
                }

                sb = sb + 1
            }
            blk = blk + 1
        }

        // Write 8 results
        let mut r: i32 = 0
        while r < 8 {
            out[tile * 8 + r] = sums[r]
            r = r + 1
        }
        tile = tile + 1
    }
}
```

**CRITICAL NOTE:** The kernel above is a structural skeleton showing the data flow. The exact SIMD reduction (how maddubs results map back to per-row dot products given the interleaved quant layout) must be implemented by studying `arch/x86/repack.cpp:1620-1660` line by line. The interleaving pattern means each 256-bit register contains data from multiple rows, and the reduction must de-interleave correctly. **Do not ship this skeleton — fill in the exact SIMD chain matching llama.cpp.**

- [ ] **Step 2: Wire up FFI**

Add to `src/kernels/ffi_inference_types.rs`:
```rust
pub type Q4k8x8Q8kMatvecFn = unsafe extern "C" fn(
    packed: *const u8, q8_qs: *const i8, q8_d: *const f32,
    q8_bsums: *const i16, pow2: *const f32, scratch: *mut u8,
    out: *mut f32, n_rows: i32, n_cols: i32,
);
```

Add to `KernelTableInference`, load from `q4k_dot_8x8` library, add public wrapper.

- [ ] **Step 3: Run roundtrip test**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test repack_q4k repack_q4k_roundtrip_matvec -- --nocapture 2>&1 | tail -10
```

Expected: PASS — repacked matvec produces identical f32 output to standard matvec.

- [ ] **Step 4: Run verification gates**

Run Gate 1-4 from the verification gate section.

- [ ] **Step 5: Commit**

```bash
git add kernels/q4k_dot_8x8.ea src/kernels/ffi_inference.rs src/kernels/ffi_inference_types.rs
git commit -m "feat: Q4Kx8 repacked matvec kernel (x86 AVX2) matching llama.cpp"
```

---

## Task 3: Ea Repack + Matvec Kernels — ARM NEON

**Files:**
- Create: `kernels/q4k_repack_arm.ea`
- Create: `kernels/q4k_dot_8x8_arm.ea`
- Modify: `src/kernels/ffi_inference.rs` (load ARM variants)

### Steps

- [ ] **Step 1: Write ARM repack kernel**

Create `kernels/q4k_repack_arm.ea` — identical logic to x86 version but with `#[cfg(aarch64)]`. The repack is data movement (not compute), so SIMD differences are minimal.

- [ ] **Step 2: Write ARM matvec kernel**

Create `kernels/q4k_dot_8x8_arm.ea` matching `llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp:709-861`:
- Use `vld1q_u8` for quant loading
- Use `ggml_vdotq_s32` equivalent (`vdot_i32` in Ea) for dot products
- Use `decode_q_Kx8_6bit_scales()` pattern for scale extraction

- [ ] **Step 3: Update FFI to load ARM variants**

In `load_inference_kernels`, use `load_best("q4k_repack")` and `load_best("q4k_dot_8x8")` to pick ARM variants on aarch64.

- [ ] **Step 4: Test on ARM (if available) or cross-compile check**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add kernels/q4k_repack_arm.ea kernels/q4k_dot_8x8_arm.ea src/kernels/ffi_inference.rs
git commit -m "feat: Q4Kx8 repack + matvec ARM NEON kernels"
```

---

## Task 4: Integrate Repacked Weights into Engine

**Files:**
- Modify: `src/inference/engine.rs`
- Modify: `src/inference/matmul.rs`
- Modify: `src/inference/matmul_graph.rs`

### What changes

Model loading repacks all Q4K weight matrices at startup. The matmul wrappers switch to the 8x8 kernel for Q4K. The work-stealing graph ops also use the repacked path.

### Steps

- [ ] **Step 1: Add repacked weight storage to Gemma4Model**

In `engine.rs`, after loading weights from GGUF, repack Q4K weights:

```rust
// In Gemma4Model::from_gguf(), after loading all LayerWeights:

// Repack Q4K weights for 8x8 matvec
let mut repacked_bufs: Vec<Vec<u8>> = Vec::new();
for (i, lw) in layers.iter_mut().enumerate() {
    // Repack each Q4K weight matrix
    for (ptr, dtype, n_rows, n_cols) in [
        (&mut lw.wq, lw.wq_dtype, n_heads[i] * head_dim_k[i], hidden_dim),
        (&mut lw.wk, lw.wk_dtype, n_kv_heads[i] * head_dim_k[i], hidden_dim),
        // ... wv, wo, w_gate, w_up, w_down
    ] {
        if dtype == GGML_TYPE_Q4_K && n_rows % 8 == 0 {
            let row_bytes = (n_cols / 256) * 144;
            let mut buf = vec![0u8; n_rows * row_bytes];
            unsafe {
                ffi_inference::q4k_repack_8x8(*ptr, buf.as_mut_ptr(), n_rows as i32, n_cols as i32);
            }
            *ptr = buf.as_ptr();
            repacked_bufs.push(buf);
        }
    }
}
```

Store `repacked_bufs` in `Gemma4Model` so the buffers live as long as the model.

- [ ] **Step 2: Update q4k_matvec to use repacked kernel**

In `matmul.rs`, replace the 4-row loop in `q4k_matvec` with 8-row tiles calling `q4k_8x8_q8k_matvec`:

```rust
pub fn q4k_matvec(
    weight: *const u8, q8_qs: &[i8], q8_d: &[f32], q8_bsums: &[i16],
    pow2: &[f32], out: &mut [f32], n_rows: usize, n_cols: usize,
) {
    // Weight is already repacked at this point
    unsafe {
        ffi_inference::q4k_8x8_q8k_matvec(
            weight, q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
            pow2.as_ptr(), std::ptr::null_mut(),
            out.as_mut_ptr(), n_rows as i32, n_cols as i32,
        );
    }
}
```

Handle remainder rows (n_rows % 8 != 0) with the existing single-row kernel.

- [ ] **Step 3: Update work-stealing matmul**

In `matmul_graph.rs`, update `q4k_matvec_ws` to use 8-row tiles with the repacked kernel. Work-stealing granularity becomes 8 rows instead of 4.

- [ ] **Step 4: Run full verification**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression -- --nocapture
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --test-threads=1 --nocapture
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_smoke -- --nocapture
```

All must pass. The bit-exact regression test is the most critical — if the repacked path produces different bits, something is wrong in the kernel.

- [ ] **Step 5: Commit**

```bash
git add src/inference/engine.rs src/inference/matmul.rs src/inference/matmul_graph.rs
git commit -m "feat: integrate Q4K repacked weights into model loading and matmul"
```

---

## Task 5: Benchmark Repacked vs Standard

**Files:**
- Modify: `tests/bench_q4k_gemm.rs` (or create `tests/bench_repack.rs`)

### Steps

- [ ] **Step 1: Run existing bench**

The bench in `tests/bench_q4k_gemm.rs` already references the repacked APIs. Verify it compiles and runs:

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test bench_q4k_gemm -- --nocapture --test-threads=1
```

- [ ] **Step 2: Run decode speed bench**

```bash
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test bench_decode_speed -- --nocapture
```

Record tok/s before and after. The repack should improve decode speed because:
- 8 rows per tile instead of 4 → fewer kernel calls
- Better cache locality (interleaved quants are accessed sequentially)

Expected: measurable improvement on both x86 and ARM (10-30% for decode).

- [ ] **Step 3: Compare against llama.cpp**

```bash
~/projects/llama.cpp/build/bin/llama-batched \
    -m ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf \
    -p "Count from 1 to 20" -n 64 --temp 0 -t 16 --ignore-eos
```

Record llama.cpp's generation tok/s. Compare with Olorin's bench output.

- [ ] **Step 4: Save results via eabrain**

```bash
eabrain remember "Q4K repack benchmark (DATE): Olorin decode X tok/s (was Y), llama.cpp Z tok/s. Gap: A.Bx (was C.Dx)"
```

- [ ] **Step 5: Commit bench results in commit message**

```bash
git add tests/
git commit -m "bench: Q4K repacked matvec — X tok/s decode (Yx improvement)"
```

---

## Task 6: Update Regression Snapshot

**Files:**
- Modify: `tests/snapshots/gemma4_logits_bos.bin` (regenerated)

### Steps

**Only do this task if the repack changes the bit-exact output** (which it should NOT — same math, same accumulation order). If the regression test passes without snapshot update, skip this task entirely.

- [ ] **Step 1: Investigate why bits changed**

If the regression test fails, the repacked kernel is computing differently. Debug:
- Compare standard vs repacked output for a single block
- Check accumulation order matches llama.cpp exactly
- Fix the kernel, don't update the snapshot

- [ ] **Step 2: Only if accumulation order intentionally changed**

If the repack genuinely changes accumulation order (8-row tile vs 4-row), the snapshot must be updated. Delete old snapshot, run test to regenerate, run again to verify:

```bash
rm tests/snapshots/gemma4_logits_bos.bin
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression -- --nocapture
PATH="$HOME/projects/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression -- --nocapture
```

Second run must pass.

---

## Summary

| Task | What | New Files | Key Verification |
|------|------|-----------|-----------------|
| 1 | Ea repack kernel (x86) | `q4k_repack.ea`, `repack_q4k.rs` | Golden binary match |
| 2 | Ea repacked matvec (x86) | `q4k_dot_8x8.ea` | Roundtrip bit-exact vs standard |
| 3 | ARM variants | `q4k_repack_arm.ea`, `q4k_dot_8x8_arm.ea` | Cross-compile clean |
| 4 | Engine integration | engine.rs, matmul.rs changes | Full test suite pass |
| 5 | Benchmark | bench output | tok/s improvement measured |
| 6 | Snapshot update | Only if needed | Regression suite green |

**Known follow-ups within repack scope:**
- `q4k_dot_8x8_dual` — fused gate+up repacked matvec (currently `q4k_dot_q8k_4row_dual` does this for standard layout). Add after the base kernel is proven correct.
- Q5K/Q6K repack — same pattern, lower priority (Q6K used only for embeddings, Q5K mixed usage).

**After this plan:** Phase 2 (batched prompt eval) builds on the repacked weights, adding mat×mat GEMM kernels. Phase 3 (flash attention) adds online softmax. Each gets its own plan.
