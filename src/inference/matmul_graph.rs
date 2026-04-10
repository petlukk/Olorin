//! Work-stealing matmul ops for graph-loop threading.
//!
//! Each function takes (ith, nth, current_chunk) and uses atomic fetch_add
//! to grab work chunks, matching llama.cpp's matmul work distribution.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::kernels::ffi_inference;
use super::matmul::*;

// ---------------------------------------------------------------------------
// Q4K work-stealing matvec
// ---------------------------------------------------------------------------

/// Q4K matvec: work-stealing via atomic current_chunk.
/// current_chunk must be reset to nth before calling (by the preceding op).
pub fn q4k_matvec_ws(
    weight: *const u8, q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32, n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let pow2 = pow2_table();

    let mut chunk = ith as i32;
    while (chunk as usize) < full_quads {
        let base_row = (chunk as usize) * 4;
        unsafe {
            ffi_inference::q4k_dot_q8k_4row(
                weight.add(base_row * row_bytes),
                weight.add((base_row + 1) * row_bytes),
                weight.add((base_row + 2) * row_bytes),
                weight.add((base_row + 3) * row_bytes),
                q8, bsums,
                output.add(base_row),
                n_blocks as i32, q8_d, pow2.as_ptr(),
            );
        }
        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }

    // Remainder: only thread 0
    if ith == 0 {
        let base = full_quads * 4;
        for i in 0..(n_rows % 4) {
            let row = base + i;
            unsafe {
                *output.add(row) = ffi_inference::q4k_dot_q8k(
                    weight.add(row * row_bytes), q8, bsums,
                    n_blocks as i32, q8_d, pow2.as_ptr(),
                );
            }
        }
    }
}

/// Q4K dual matvec (gate + up): work-stealing.
pub fn q4k_matvec_dual_ws(
    gate_w: *const u8, up_w: *const u8,
    q8: *const i8, q8_d: *const f32, bsums: *const i16,
    gate_out: *mut f32, up_out: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let pow2 = pow2_table();

    let mut chunk = ith as i32;
    while (chunk as usize) < full_quads {
        let base_row = (chunk as usize) * 4;
        unsafe {
            ffi_inference::q4k_dot_q8k_4row_dual(
                gate_w.add(base_row * row_bytes),
                gate_w.add((base_row + 1) * row_bytes),
                gate_w.add((base_row + 2) * row_bytes),
                gate_w.add((base_row + 3) * row_bytes),
                up_w.add(base_row * row_bytes),
                up_w.add((base_row + 1) * row_bytes),
                up_w.add((base_row + 2) * row_bytes),
                up_w.add((base_row + 3) * row_bytes),
                q8, bsums,
                gate_out.add(base_row),
                up_out.add(base_row),
                n_blocks as i32, q8_d, pow2.as_ptr(),
            );
        }
        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }

    // Remainder: only thread 0
    if ith == 0 {
        let base = full_quads * 4;
        let pow2_ptr = pow2.as_ptr();
        for i in 0..(n_rows % 4) {
            let row = base + i;
            unsafe {
                *gate_out.add(row) = ffi_inference::q4k_dot_q8k(
                    gate_w.add(row * row_bytes), q8, bsums,
                    n_blocks as i32, q8_d, pow2_ptr,
                );
                *up_out.add(row) = ffi_inference::q4k_dot_q8k(
                    up_w.add(row * row_bytes), q8, bsums,
                    n_blocks as i32, q8_d, pow2_ptr,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q5K work-stealing matvec
// ---------------------------------------------------------------------------

pub fn q5k_matvec_ws(
    weight: *const u8, q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32, n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    let n_blocks = n_cols / Q5K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let pow2 = pow2_table();

    let mut chunk = ith as i32;
    while (chunk as usize) < full_quads {
        let base_row = (chunk as usize) * 4;
        unsafe {
            ffi_inference::q5k_dot_q8k_4row(
                weight.add(base_row * row_bytes),
                weight.add((base_row + 1) * row_bytes),
                weight.add((base_row + 2) * row_bytes),
                weight.add((base_row + 3) * row_bytes),
                q8, bsums,
                output.add(base_row),
                n_blocks as i32, q8_d, pow2.as_ptr(),
            );
        }
        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }

    if ith == 0 {
        let base = full_quads * 4;
        let pow2_ptr = pow2.as_ptr();
        for i in 0..(n_rows % 4) {
            let row = base + i;
            unsafe {
                *output.add(row) = ffi_inference::q5k_dot_q8k(
                    weight.add(row * row_bytes), q8, bsums,
                    n_blocks as i32, q8_d, pow2_ptr,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q6K work-stealing matvec
// ---------------------------------------------------------------------------

pub fn q6k_matvec_ws(
    weight: *const u8, q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    let n_blocks = n_cols / Q6K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    let full_quads = n_rows / 4;

    // Each thread needs its own d_scratch slice (n_blocks * 4 floats per row set)
    let scratch_per = n_blocks * 4;
    let my_scratch = unsafe { d_scratch.add(ith * scratch_per) };

    let mut chunk = ith as i32;
    while (chunk as usize) < full_quads {
        let base_row = (chunk as usize) * 4;
        unsafe {
            // Extract d_arr for 4 rows
            let d0 = my_scratch;
            let d1 = my_scratch.add(n_blocks);
            let d2 = my_scratch.add(n_blocks * 2);
            let d3 = my_scratch.add(n_blocks * 3);
            for blk in 0..n_blocks {
                let off = 208; // d at byte 208 in Q6K block
                for (row_off, d_ptr) in [(0, d0), (1, d1), (2, d2), (3, d3)] {
                    let w = weight.add((base_row + row_off) * row_bytes + blk * Q6K_BLOCK_BYTES + off);
                    let raw = u16::from_le_bytes([*w, *w.add(1)]);
                    *d_ptr.add(blk) = f16_to_f32_scalar(raw) * *q8_d.add(blk);
                }
            }
            ffi_inference::q6k_dot_q8k_4row(
                weight.add(base_row * row_bytes),
                weight.add((base_row + 1) * row_bytes),
                weight.add((base_row + 2) * row_bytes),
                weight.add((base_row + 3) * row_bytes),
                q8, bsums,
                output.add(base_row),
                n_blocks as i32, d0, d1, d2, d3,
            );
        }
        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }

    if ith == 0 {
        let base = full_quads * 4;
        for i in 0..(n_rows % 4) {
            let row = base + i;
            unsafe {
                let d0 = my_scratch;
                for blk in 0..n_blocks {
                    let w = weight.add(row * row_bytes + blk * Q6K_BLOCK_BYTES + 208);
                    let raw = u16::from_le_bytes([*w, *w.add(1)]);
                    *d0.add(blk) = f16_to_f32_scalar(raw) * *q8_d.add(blk);
                }
                *output.add(row) = ffi_inference::q6k_dot_q8k(
                    weight.add(row * row_bytes), q8, bsums,
                    n_blocks as i32, d0,
                );
            }
        }
    }
}

/// Dispatch to correct dtype. Resets current_chunk to nth before work-stealing.
pub fn matvec_ws(
    dtype: u32, weight: *const u8,
    q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    // Reset chunk counter — thread 0 does this, others wait at barrier
    // (caller must barrier after this op)
    match dtype {
        GGML_TYPE_Q4_K => q4k_matvec_ws(weight, q8, q8_d, bsums, output, n_rows, n_cols, current_chunk, ith, nth),
        GGML_TYPE_Q5_K => q5k_matvec_ws(weight, q8, q8_d, bsums, output, n_rows, n_cols, current_chunk, ith, nth),
        GGML_TYPE_Q6_K => q6k_matvec_ws(weight, q8, q8_d, bsums, output, d_scratch, n_rows, n_cols, current_chunk, ith, nth),
        _ => panic!("unsupported weight dtype {dtype}"),
    }
}
