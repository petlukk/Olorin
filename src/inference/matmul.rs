//! Matrix-vector multiplication wrappers for Q4K and Q6K Ea kernels.
//!
//! All dot-product compute goes through Ea SIMD kernels via FFI.
//! This module provides safe-ish wrappers that handle block layout,
//! row stride, and 4-row batching.

use crate::kernels::ffi_inference;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const Q4K_BLOCK_SIZE: usize = 256; // elements per Q4K superblock
pub const Q4K_BLOCK_BYTES: usize = 144; // bytes per Q4K superblock
pub const Q6K_BLOCK_SIZE: usize = 256; // elements per Q6K superblock
pub const Q6K_BLOCK_BYTES: usize = 210; // bytes per Q6K superblock

// GGUF dtype codes
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q6_K: u32 = 14;

// ---------------------------------------------------------------------------
// pow2 table for Q4K f16→f32 inline conversion
// ---------------------------------------------------------------------------

/// Precomputed 2^(exp-15) for f16 exponent fields 0..31.
/// The Q4K kernel uses this to convert f16 scale/dmin without F16C.
fn build_pow2_table() -> [f32; 32] {
    let mut t = [0.0f32; 32];
    let mut i = 1u32;
    while i < 32 {
        // 2^(i - 15)
        let bits: u32 = (i.wrapping_add(127 - 15)) << 23;
        t[i as usize] = f32::from_bits(bits);
        i += 1;
    }
    t
}

fn pow2_table() -> &'static [f32; 32] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[f32; 32]> = OnceLock::new();
    TABLE.get_or_init(build_pow2_table)
}

// ---------------------------------------------------------------------------
// f16 scalar helper (for extracting d from Q6K blocks)
// ---------------------------------------------------------------------------

#[inline]
fn f16_to_f32_scalar(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: normalize
        let mut m = mant;
        let mut e = 0i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        let f32_exp = ((127 - 15 + 1) as i32 + e) as u32;
        let f32_mant = (m & 0x3ff) << 13;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant);
    }
    if exp == 31 {
        // Inf / NaN
        let f32_mant = mant << 13;
        return f32::from_bits((sign << 31) | (0xff << 23) | f32_mant);
    }
    // Normal
    let f32_exp = (exp as i32 + 127 - 15) as u32;
    let f32_mant = mant << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
}

// ---------------------------------------------------------------------------
// Quantize input: f32 → Q8K
// ---------------------------------------------------------------------------

/// Quantize f32 input vector to Q8K format for use with dot-product kernels.
///
/// - `src`: input f32 values, length `n` (must be multiple of 256)
/// - `dst_qs`: output i8 quantized values, length `n + 12` (padding)
/// - `dst_d`: output per-block scale, length `n / 256`
/// - `dst_bsums`: output per-block sums, length `n / 256 * 16`
pub fn quant_input(
    src: &[f32],
    dst_qs: &mut [i8],
    dst_d: &mut [f32],
    dst_bsums: &mut [i16],
) {
    let n = src.len();
    debug_assert!(n % Q4K_BLOCK_SIZE == 0, "input length must be multiple of 256");
    let n_blocks = n / Q4K_BLOCK_SIZE;
    debug_assert!(dst_qs.len() >= n + 12);
    debug_assert!(dst_d.len() >= n_blocks);
    debug_assert!(dst_bsums.len() >= n_blocks * 16);

    unsafe {
        ffi_inference::quant_f32_q8k(
            src.as_ptr(),
            dst_qs.as_mut_ptr(),
            dst_d.as_mut_ptr(),
            dst_bsums.as_mut_ptr(),
            n as i32,
        );
    }
}

// ---------------------------------------------------------------------------
// Q4K matrix-vector multiply
// ---------------------------------------------------------------------------

/// Q4K matrix-vector: weight (n_rows × n_cols, Q4K) × input (Q8K) → output (f32).
///
/// - `weight`: pointer to packed Q4K weight data (n_rows * row_bytes)
/// - `input_qs`: Q8K quantized input, length n_cols + 12
/// - `input_d`: per-block input scale, length n_blocks
/// - `input_bsums`: per-block input sums, length n_blocks * 16
/// - `output`: result buffer, length >= n_rows
/// - `n_rows`: number of output rows
/// - `n_cols`: number of input columns (must be multiple of 256)
pub fn q4k_matvec(
    weight: *const u8,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0);
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let pow2 = pow2_table();

    let q8 = input_qs.as_ptr();
    let bsums = input_bsums.as_ptr();
    let q8_d = input_d.as_ptr();
    let pow2_ptr = pow2.as_ptr();

    // Process 4 rows at a time
    let full_quads = n_rows / 4;
    let remainder = n_rows % 4;

    unsafe {
        for quad in 0..full_quads {
            let base_row = quad * 4;
            let rw0 = weight.add(base_row * row_bytes);
            let rw1 = weight.add((base_row + 1) * row_bytes);
            let rw2 = weight.add((base_row + 2) * row_bytes);
            let rw3 = weight.add((base_row + 3) * row_bytes);

            ffi_inference::q4k_dot_q8k_4row(
                rw0, rw1, rw2, rw3,
                q8, bsums,
                output.as_mut_ptr().add(base_row),
                n_blocks as i32,
                q8_d, pow2_ptr,
            );
        }

        // Remainder rows (1-3)
        for i in 0..remainder {
            let row = full_quads * 4 + i;
            let rw = weight.add(row * row_bytes);
            output[row] = ffi_inference::q4k_dot_q8k(
                rw, q8, bsums,
                n_blocks as i32,
                q8_d, pow2_ptr,
            );
        }
    }
}

/// Q4K dual matrix-vector: gate + up projection in one pass.
///
/// Computes gate_weight × input and up_weight × input simultaneously,
/// sharing Q8K input across both. 4 rows at a time for each pair.
pub fn q4k_matvec_dual(
    gate_weight: *const u8,
    up_weight: *const u8,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0);
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let pow2 = pow2_table();

    let q8 = input_qs.as_ptr();
    let bsums = input_bsums.as_ptr();
    let q8_d = input_d.as_ptr();
    let pow2_ptr = pow2.as_ptr();

    let full_quads = n_rows / 4;
    let remainder = n_rows % 4;

    unsafe {
        for quad in 0..full_quads {
            let base_row = quad * 4;
            let gw0 = gate_weight.add(base_row * row_bytes);
            let gw1 = gate_weight.add((base_row + 1) * row_bytes);
            let gw2 = gate_weight.add((base_row + 2) * row_bytes);
            let gw3 = gate_weight.add((base_row + 3) * row_bytes);
            let uw0 = up_weight.add(base_row * row_bytes);
            let uw1 = up_weight.add((base_row + 1) * row_bytes);
            let uw2 = up_weight.add((base_row + 2) * row_bytes);
            let uw3 = up_weight.add((base_row + 3) * row_bytes);

            ffi_inference::q4k_dot_q8k_4row_dual(
                gw0, gw1, gw2, gw3,
                uw0, uw1, uw2, uw3,
                q8, bsums,
                gate_output.as_mut_ptr().add(base_row),
                up_output.as_mut_ptr().add(base_row),
                n_blocks as i32,
                q8_d, pow2_ptr,
            );
        }

        // Remainder rows — fall back to separate single-row calls
        for i in 0..remainder {
            let row = full_quads * 4 + i;
            let gw = gate_weight.add(row * row_bytes);
            let uw = up_weight.add(row * row_bytes);
            gate_output[row] = ffi_inference::q4k_dot_q8k(
                gw, q8, bsums, n_blocks as i32, q8_d, pow2_ptr,
            );
            up_output[row] = ffi_inference::q4k_dot_q8k(
                uw, q8, bsums, n_blocks as i32, q8_d, pow2_ptr,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Q6K matrix-vector multiply
// ---------------------------------------------------------------------------

/// Extract per-block d values from Q6K weight data and pre-multiply with Q8K d.
///
/// Q6K block layout: d (f16) is at offset 208 within each 210-byte block.
/// The kernel expects `d_arr[blk] = d_q6k * d_q8k`.
#[inline]
fn q6k_extract_d(weight: *const u8, n_blocks: usize, q8_d: &[f32], d_arr: &mut [f32]) {
    for blk in 0..n_blocks {
        let block_ptr = unsafe { weight.add(blk * Q6K_BLOCK_BYTES + 208) };
        let raw = unsafe { u16::from_le_bytes([*block_ptr, *block_ptr.add(1)]) };
        d_arr[blk] = f16_to_f32_scalar(raw) * q8_d[blk];
    }
}

/// Q6K matrix-vector: weight (n_rows × n_cols, Q6K) × input (Q8K) → output (f32).
///
/// - `weight`: pointer to packed Q6K weight data
/// - `input_qs`: Q8K quantized input
/// - `input_d`: per-block input scale
/// - `input_bsums`: per-block input sums
/// - `output`: result buffer
/// - `d_scratch`: scratch buffer for pre-multiplied d values, length >= n_blocks * 4
///   (needs 4× for the 4-row variant which needs 4 independent d arrays)
/// - `n_rows`, `n_cols`: dimensions
pub fn q6k_matvec(
    weight: *const u8,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    d_scratch: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert!(n_cols % Q6K_BLOCK_SIZE == 0);
    let n_blocks = n_cols / Q6K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    debug_assert!(d_scratch.len() >= n_blocks * 4);

    let q8 = input_qs.as_ptr();
    let bsums = input_bsums.as_ptr();

    let full_quads = n_rows / 4;
    let remainder = n_rows % 4;

    unsafe {
        for quad in 0..full_quads {
            let base_row = quad * 4;
            let w0 = weight.add(base_row * row_bytes);
            let w1 = weight.add((base_row + 1) * row_bytes);
            let w2 = weight.add((base_row + 2) * row_bytes);
            let w3 = weight.add((base_row + 3) * row_bytes);

            // Extract d arrays for each row into scratch
            let (d0, rest) = d_scratch.split_at_mut(n_blocks);
            let (d1, rest) = rest.split_at_mut(n_blocks);
            let (d2, d3) = rest.split_at_mut(n_blocks);
            q6k_extract_d(w0, n_blocks, input_d, d0);
            q6k_extract_d(w1, n_blocks, input_d, d1);
            q6k_extract_d(w2, n_blocks, input_d, d2);
            q6k_extract_d(w3, n_blocks, input_d, d3);

            ffi_inference::q6k_dot_q8k_4row(
                w0, w1, w2, w3,
                q8, bsums,
                output.as_mut_ptr().add(base_row),
                n_blocks as i32,
                d0.as_ptr(), d1.as_ptr(), d2.as_ptr(), d3.as_ptr(),
            );
        }

        // Remainder rows
        let d0 = &mut d_scratch[..n_blocks];
        for i in 0..remainder {
            let row = full_quads * 4 + i;
            let w = weight.add(row * row_bytes);
            q6k_extract_d(w, n_blocks, input_d, d0);
            output[row] = ffi_inference::q6k_dot_q8k(
                w, q8, bsums, n_blocks as i32, d0.as_ptr(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Q6K embedding dequantization
// ---------------------------------------------------------------------------

/// Dequantize a single row from a Q6K embedding table to f32.
///
/// Q6K block layout (210 bytes per 256 elements):
///   ql[128]@0  qh[64]@128  scales[16]@192  d(f16)@208
///
/// Each element is 6-bit: low 4 bits from ql, high 2 bits from qh.
/// Value = d * scale[group] * (q6 - 32).
pub fn q6k_embed_lookup(
    weight: *const u8,
    token_id: usize,
    output: &mut [f32],
    hidden_dim: usize,
) {
    debug_assert!(hidden_dim % Q6K_BLOCK_SIZE == 0);
    let n_blocks = hidden_dim / Q6K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    let row_base = unsafe { weight.add(token_id * row_bytes) };

    for blk in 0..n_blocks {
        let block = unsafe { row_base.add(blk * Q6K_BLOCK_BYTES) };
        let out_base = blk * Q6K_BLOCK_SIZE;

        // Extract d (f16 at offset 208)
        let d_raw = unsafe { u16::from_le_bytes([*block.add(208), *block.add(209)]) };
        let d = f16_to_f32_scalar(d_raw);

        // Q6K layout per block (256 elements):
        //   ql[128]@0  qh[64]@128  scales[16]@192  d(f16)@208
        //
        // Processed in 2 halves of 128 elements each.
        // Each half has 4 groups of 32 elements:
        //   Group 0: ql[0..32] low nibble  + qh bits [0..1]
        //   Group 1: ql[0..32] high nibble + qh bits [2..3]
        //   Group 2: ql[32..64] low nibble + qh bits [4..5]
        //   Group 3: ql[32..64] high nibble + qh bits [6..7]
        //
        // Each group's 32 elements split into 2 subgroups of 16
        // with separate scale entries.
        // Q8K input is consumed sequentially: 32 elements per group.

        for half in 0..2usize {
            let ql_ptr = unsafe { block.add(half * 64) };
            let qh_ptr = unsafe { block.add(128 + half * 32) };
            let sc_ptr = unsafe { block.add(192 + half * 8) };
            let elem_base = out_base + half * 128;

            // Group 0: ql[0..32] low nibble, qh bits 0-1
            let sc0a = d * unsafe { *sc_ptr.add(0) as i8 as f32 };
            let sc0b = d * unsafe { *sc_ptr.add(1) as i8 as f32 };
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(j) };
                let qh = unsafe { *qh_ptr.add(j) };
                let q6 = (ql & 0x0f) as i32 | (((qh & 0x03) as i32) << 4);
                output[elem_base + j] = sc0a * ((q6 - 32) as f32);
            }
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(16 + j) };
                let qh = unsafe { *qh_ptr.add(16 + j) };
                let q6 = (ql & 0x0f) as i32 | (((qh & 0x03) as i32) << 4);
                output[elem_base + 16 + j] = sc0b * ((q6 - 32) as f32);
            }

            // Group 1: ql[0..32] high nibble, qh bits 2-3
            let sc1a = d * unsafe { *sc_ptr.add(2) as i8 as f32 };
            let sc1b = d * unsafe { *sc_ptr.add(3) as i8 as f32 };
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(j) };
                let qh = unsafe { *qh_ptr.add(j) };
                let q6 = ((ql >> 4) & 0x0f) as i32 | ((((qh >> 2) & 0x03) as i32) << 4);
                output[elem_base + 32 + j] = sc1a * ((q6 - 32) as f32);
            }
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(16 + j) };
                let qh = unsafe { *qh_ptr.add(16 + j) };
                let q6 = ((ql >> 4) & 0x0f) as i32 | ((((qh >> 2) & 0x03) as i32) << 4);
                output[elem_base + 32 + 16 + j] = sc1b * ((q6 - 32) as f32);
            }

            // Group 2: ql[32..64] low nibble, qh bits 4-5
            let sc2a = d * unsafe { *sc_ptr.add(4) as i8 as f32 };
            let sc2b = d * unsafe { *sc_ptr.add(5) as i8 as f32 };
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(32 + j) };
                let qh = unsafe { *qh_ptr.add(j) };
                let q6 = (ql & 0x0f) as i32 | ((((qh >> 4) & 0x03) as i32) << 4);
                output[elem_base + 64 + j] = sc2a * ((q6 - 32) as f32);
            }
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(32 + 16 + j) };
                let qh = unsafe { *qh_ptr.add(16 + j) };
                let q6 = (ql & 0x0f) as i32 | ((((qh >> 4) & 0x03) as i32) << 4);
                output[elem_base + 64 + 16 + j] = sc2b * ((q6 - 32) as f32);
            }

            // Group 3: ql[32..64] high nibble, qh bits 6-7
            let sc3a = d * unsafe { *sc_ptr.add(6) as i8 as f32 };
            let sc3b = d * unsafe { *sc_ptr.add(7) as i8 as f32 };
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(32 + j) };
                let qh = unsafe { *qh_ptr.add(j) };
                let q6 = ((ql >> 4) & 0x0f) as i32 | ((((qh >> 6) & 0x03) as i32) << 4);
                output[elem_base + 96 + j] = sc3a * ((q6 - 32) as f32);
            }
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(32 + 16 + j) };
                let qh = unsafe { *qh_ptr.add(16 + j) };
                let q6 = ((ql >> 4) & 0x0f) as i32 | ((((qh >> 6) & 0x03) as i32) << 4);
                output[elem_base + 96 + 16 + j] = sc3b * ((q6 - 32) as f32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Row byte helpers (for external callers computing offsets)
// ---------------------------------------------------------------------------

/// Bytes per row for Q4K weight matrix with given column count.
#[inline]
pub fn q4k_row_bytes(n_cols: usize) -> usize {
    (n_cols / Q4K_BLOCK_SIZE) * Q4K_BLOCK_BYTES
}

/// Bytes per row for Q6K weight matrix with given column count.
#[inline]
pub fn q6k_row_bytes(n_cols: usize) -> usize {
    (n_cols / Q6K_BLOCK_SIZE) * Q6K_BLOCK_BYTES
}
