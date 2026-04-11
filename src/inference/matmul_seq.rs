//! Sequential (single-threaded) dtype kernels for quantized matmul.
//!
//! Each function processes `n_rows × n_cols` on the calling thread with
//! 4-row batching via the `_4row` Ea kernels, and a 1-3 row remainder loop.
//! Parallel ThreadPool wrappers live in `matmul_par.rs`; dispatch entry
//! points live in `matmul.rs`.

use crate::kernels::ffi_inference;
use super::matmul::{
    Q4K_BLOCK_SIZE, Q4K_BLOCK_BYTES,
    Q5K_BLOCK_SIZE, Q5K_BLOCK_BYTES,
    Q6K_BLOCK_SIZE, Q6K_BLOCK_BYTES,
    pow2_table, f16_to_f32_scalar,
};

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
// Q5K matrix-vector multiply
// ---------------------------------------------------------------------------

/// Q5K matrix-vector: weight (n_rows × n_cols, Q5K) × input (Q8K) → output (f32).
///
/// Same interface as q4k_matvec. Q5K has same scale/mins format as Q4K,
/// just 5 bits per weight instead of 4 (extra bit in qh field).
pub fn q5k_matvec(
    weight: *const u8,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert!(n_cols % Q5K_BLOCK_SIZE == 0);
    let n_blocks = n_cols / Q5K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
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
            let rw0 = weight.add(base_row * row_bytes);
            let rw1 = weight.add((base_row + 1) * row_bytes);
            let rw2 = weight.add((base_row + 2) * row_bytes);
            let rw3 = weight.add((base_row + 3) * row_bytes);

            ffi_inference::q5k_dot_q8k_4row(
                rw0, rw1, rw2, rw3,
                q8, bsums,
                output.as_mut_ptr().add(base_row),
                n_blocks as i32,
                q8_d, pow2_ptr,
            );
        }

        for i in 0..remainder {
            let row = full_quads * 4 + i;
            let rw = weight.add(row * row_bytes);
            output[row] = ffi_inference::q5k_dot_q8k(
                rw, q8, bsums,
                n_blocks as i32,
                q8_d, pow2_ptr,
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
pub(super) fn q6k_extract_d(weight: *const u8, n_blocks: usize, q8_d: &[f32], d_arr: &mut [f32]) {
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
// BF16 matrix-vector multiply
// ---------------------------------------------------------------------------

/// BF16 matrix-vector: weight (n_rows x n_cols, BF16) x input (f32) -> output (f32).
///
/// BF16 weight data is 2 bytes per element. Row stride = n_cols * 2 bytes.
pub fn bf16_matvec(
    weight: *const u8,
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    let row_bytes = n_cols * 2;
    let mut scratch = [0i32; 8]; // scratch for BF16->f32 reinterpretation

    let full_quads = n_rows / 4;
    let remainder = n_rows % 4;

    unsafe {
        for quad in 0..full_quads {
            let base_row = quad * 4;
            let w0 = weight.add(base_row * row_bytes) as *const u16;
            let w1 = weight.add((base_row + 1) * row_bytes) as *const u16;
            let w2 = weight.add((base_row + 2) * row_bytes) as *const u16;
            let w3 = weight.add((base_row + 3) * row_bytes) as *const u16;

            ffi_inference::bf16_dot_f32_4row(
                w0, w1, w2, w3,
                input.as_ptr(),
                output.as_mut_ptr().add(base_row),
                scratch.as_mut_ptr(),
                n_cols as i32,
            );
        }

        for i in 0..remainder {
            let row = full_quads * 4 + i;
            let w = weight.add(row * row_bytes) as *const u16;
            output[row] = ffi_inference::bf16_dot_f32(
                w, input.as_ptr(), scratch.as_mut_ptr(), n_cols as i32,
            );
        }
    }
}
