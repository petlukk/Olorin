//! Sequential (single-threaded) dtype kernels for quantized matmul.
//!
//! Kept as the scalar reference used by `matmul::matvec` and for the
//! `repack_q4k_matvec_roundtrip` test (which calls `q4k_matvec` directly).
//! The `bf16_matvec` helper backs PLE phase-A decode. Parallel variants
//! live in `matmul_graph.rs` and are what the live forward paths use.

use crate::kernels::ffi_inference;
use super::matmul::{
    Q3K_BLOCK_SIZE, Q3K_BLOCK_BYTES,
    Q4K_BLOCK_SIZE, Q4K_BLOCK_BYTES,
    Q5K_BLOCK_SIZE, Q5K_BLOCK_BYTES,
    Q6K_BLOCK_SIZE, Q6K_BLOCK_BYTES,
    pow2_table, f16_to_f32_scalar,
};

// ---------------------------------------------------------------------------
// Q3K matrix-vector multiply
// ---------------------------------------------------------------------------

/// Q3K matrix-vector: weight (n_rows × n_cols, Q3K) × input (Q8K) → output (f32).
pub fn q3k_matvec(
    weight: *const u8,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert!(n_cols % Q3K_BLOCK_SIZE == 0);
    let n_blocks = n_cols / Q3K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q3K_BLOCK_BYTES;
    let pow2 = pow2_table();
    let q8 = input_qs.as_ptr();
    let bsums = input_bsums.as_ptr();
    let q8_d = input_d.as_ptr();
    let pow2_ptr = pow2.as_ptr();
    unsafe {
        for row in 0..n_rows {
            output[row] = ffi_inference::q3k_dot_q8k(
                weight.add(row * row_bytes), q8, bsums,
                n_blocks as i32, q8_d, pow2_ptr,
            );
        }
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

// ---------------------------------------------------------------------------
// Q5K matrix-vector multiply
// ---------------------------------------------------------------------------

/// Q5K matrix-vector: weight (n_rows × n_cols, Q5K) × input (Q8K) → output (f32).
/// Same interface as q4k_matvec; 5 bits per weight (extra bit in qh field).
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
            ffi_inference::q5k_dot_q8k_4row(
                weight.add(base_row * row_bytes),
                weight.add((base_row + 1) * row_bytes),
                weight.add((base_row + 2) * row_bytes),
                weight.add((base_row + 3) * row_bytes),
                q8, bsums,
                output.as_mut_ptr().add(base_row),
                n_blocks as i32, q8_d, pow2_ptr,
            );
        }
        for i in 0..remainder {
            let row = full_quads * 4 + i;
            output[row] = ffi_inference::q5k_dot_q8k(
                weight.add(row * row_bytes), q8, bsums,
                n_blocks as i32, q8_d, pow2_ptr,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Q6K matrix-vector multiply
// ---------------------------------------------------------------------------

/// Extract per-block d values from Q6K weight data and pre-multiply with Q8K d.
/// Q6K block layout: d (f16) is at offset 208 within each 210-byte block.
#[inline]
fn q6k_extract_d(weight: *const u8, n_blocks: usize, q8_d: &[f32], d_arr: &mut [f32]) {
    for blk in 0..n_blocks {
        let block_ptr = unsafe { weight.add(blk * Q6K_BLOCK_BYTES + 208) };
        let raw = unsafe { u16::from_le_bytes([*block_ptr, *block_ptr.add(1)]) };
        d_arr[blk] = f16_to_f32_scalar(raw) * q8_d[blk];
    }
}

/// Q6K matrix-vector. `d_scratch` must be length >= n_blocks * 4 (4 rows × per-block d).
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
            let (d0, rest) = d_scratch.split_at_mut(n_blocks);
            let (d1, rest) = rest.split_at_mut(n_blocks);
            let (d2, d3) = rest.split_at_mut(n_blocks);
            q6k_extract_d(w0, n_blocks, input_d, d0);
            q6k_extract_d(w1, n_blocks, input_d, d1);
            q6k_extract_d(w2, n_blocks, input_d, d2);
            q6k_extract_d(w3, n_blocks, input_d, d3);
            ffi_inference::q6k_dot_q8k_4row(
                w0, w1, w2, w3, q8, bsums,
                output.as_mut_ptr().add(base_row),
                n_blocks as i32,
                d0.as_ptr(), d1.as_ptr(), d2.as_ptr(), d3.as_ptr(),
            );
        }
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
