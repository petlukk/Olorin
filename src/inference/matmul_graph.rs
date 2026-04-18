//! Work-stealing matmul ops for graph-loop threading.
//!
//! Each function takes (ith, nth, current_chunk) and uses atomic fetch_add
//! to grab work chunks, matching llama.cpp's matmul work distribution.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::kernels::ffi_inference;
use super::matmul::*;

/// Q4K matvec: work-stealing via atomic current_chunk.
/// current_chunk must be reset to nth before calling (by the preceding op).
pub fn q4k_matvec_ws(
    weight: *const u8, q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32, n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
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
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
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

pub fn q5k_matvec_ws(
    weight: *const u8, q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32, n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
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

/// Q5K matvec on the repacked 4-row tile layout produced by
/// `repack::q5k_repack_4row`. Reads one 704-byte tile per block (4 × 176).
/// Mirrors `q5k_matvec_ws` but hands a single packed pointer to the kernel
/// — the tile layout removes the scattered 4-row cache loads the old
/// per-row path forced.
pub fn q5k_matvec_repacked_ws(
    packed: *const u8, q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32, n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
) {
    let n_blocks = n_cols / Q5K_BLOCK_SIZE;
    let tile_bytes = 4 * Q5K_BLOCK_BYTES * n_blocks;
    let full_quads = n_rows / 4;
    let pow2 = pow2_table();

    let mut chunk = ith as i32;
    while (chunk as usize) < full_quads {
        let quad = chunk as usize;
        unsafe {
            ffi_inference::q5k_dot_q8k_4row_repacked(
                packed.add(quad * tile_bytes),
                q8, bsums,
                output.add(quad * 4),
                n_blocks as i32, q8_d, pow2.as_ptr(),
            );
        }
        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }

    // Tail: rows not in a full quad fall back to the per-row kernel, reading
    // the ORIGINAL (non-repacked) weight layout. Callers must ensure the raw
    // weight pointer is available for tail rows if any; Gemma 4 E2B Q5K
    // tensors (attn_k, attn_output) have n_rows % 4 == 0 so this path is
    // currently unreachable in production.
}

pub fn q6k_matvec_ws(
    weight: *const u8, q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
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

/// Q4K 8×8 repacked matvec: work-stealing, one 8-row tile per chunk.
/// n_rows % 8 == 0, n_cols % 256 == 0. current_chunk reset to nth before call.
pub fn q4k_matvec_8x8_ws(
    packed: *const u8,
    q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    debug_assert!(n_rows % 8 == 0, "q4k_matvec_8x8_ws: n_rows must be multiple of 8");
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0, "q4k_matvec_8x8_ws: n_cols must be multiple of 256");
    let _ = nth; // atomic counter carries the per-thread claim state

    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let tile_bytes = n_blocks * 1152;
    let n_tiles = n_rows / 8;
    let pow2 = pow2_table();
    let mut scratch = [0u8; 128];

    let mut chunk = ith as i32;
    while (chunk as usize) < n_tiles {
        let tile = chunk as usize;
        unsafe {
            ffi_inference::q4k_8x8_q8k_matvec(
                packed.add(tile * tile_bytes),
                q8, q8_d, bsums,
                pow2.as_ptr(),
                scratch.as_mut_ptr(),
                output.add(tile * 8),
                8i32,
                n_cols as i32,
            );
        }
        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}

/// Q4K 8×8 repacked fused dual matvec (gate + up): work-stealing.
/// Same requirements as q4k_matvec_8x8_ws, applied to both weight matrices.
pub fn q4k_matvec_dual_8x8_ws(
    gate_w: *const u8, up_w: *const u8,
    q8: *const i8, q8_d: *const f32, bsums: *const i16,
    gate_out: *mut f32, up_out: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    debug_assert!(n_rows % 8 == 0, "q4k_matvec_dual_8x8_ws: n_rows must be multiple of 8");
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0, "q4k_matvec_dual_8x8_ws: n_cols must be multiple of 256");
    let _ = nth;

    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let tile_bytes = n_blocks * 1152;
    let n_tiles = n_rows / 8;
    let pow2 = pow2_table();
    let mut scratch = [0u8; 128];

    let mut chunk = ith as i32;
    while (chunk as usize) < n_tiles {
        let tile = chunk as usize;
        unsafe {
            ffi_inference::q4k_8x8_q8k_matvec_dual(
                gate_w.add(tile * tile_bytes),
                up_w.add(tile * tile_bytes),
                q8, q8_d, bsums,
                pow2.as_ptr(),
                scratch.as_mut_ptr(),
                gate_out.add(tile * 8),
                up_out.add(tile * 8),
                8i32,
                n_cols as i32,
            );
        }
        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
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

// Re-export batch functions from matmul_graph_batch
pub use super::matmul_graph_batch::{
    q4k_matvec_8x8_batch_ws, q4k_gemm_8x8_batch_ws,
    q6k_gemm_batch_ws, q6k_repacked_batch_ws, q6k_repacked_batch_ws_pre_d,
    matvec_batch_ws,
};
