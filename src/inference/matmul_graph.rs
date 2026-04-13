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

/// Q4K 8×8 repacked batch matvec: 2D work-stealing across (row_tiles × token_chunks).
/// Matches llama.cpp's ggml_compute_forward_mul_mat chunking: flattened 2D grid,
/// each thread claims (tile, token_range) pairs via atomic fetch_add.
#[allow(clippy::too_many_arguments)]
pub fn q4k_matvec_8x8_batch_ws(
    packed: *const u8, batch_q8_qs: *const i8, batch_q8_d: *const f32,
    batch_q8_bsums: *const i16, output: *mut f32,
    n_rows: usize, n_cols: usize, n_tokens: usize, output_stride: usize,
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
) {
    debug_assert!(n_rows % 8 == 0);
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0);

    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let tile_bytes = n_blocks * 1152;
    let n_tiles = n_rows / 8;
    let qs_stride = n_cols + 12;
    let pow2 = pow2_table();
    let mut scratch = [0u8; 128];

    // 2D chunking: dim0 = row tiles (8 rows each), dim1 = token chunks
    let tok_chunk_size = 16usize;
    let n_tok_chunks = (n_tokens + tok_chunk_size - 1) / tok_chunk_size;
    let total_chunks = n_tiles * n_tok_chunks;

    let mut chunk = ith as i32;
    while (chunk as usize) < total_chunks {
        let tile = (chunk as usize) % n_tiles;
        let tok_idx = (chunk as usize) / n_tiles;
        let t_start = tok_idx * tok_chunk_size;
        let t_end = (t_start + tok_chunk_size).min(n_tokens);

        let w_ptr = unsafe { packed.add(tile * tile_bytes) };
        let out_col_off = tile * 8;

        for t in t_start..t_end {
            let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
            let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
            let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
            let out_ptr = unsafe { output.add(t * output_stride + out_col_off) };
            unsafe {
                ffi_inference::q4k_8x8_q8k_matvec(
                    w_ptr, q8, q8_d, bsums,
                    pow2.as_ptr(), scratch.as_mut_ptr(),
                    out_ptr, 8i32, n_cols as i32,
                );
            }
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}

/// Batch matvec fallback (any dtype): 2D work-stealing across (4-row chunks × token chunks).
#[allow(clippy::too_many_arguments)]
pub fn matvec_batch_ws(
    dtype: u32, weight: *const u8, batch_q8_qs: *const i8,
    batch_q8_d: *const f32, batch_q8_bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize, n_tokens: usize, output_stride: usize,
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
) {
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let qs_stride = n_cols + 12;
    let full_quads = n_rows / 4;
    let pow2 = pow2_table();

    let scratch_per = n_blocks * 4;
    let my_scratch = unsafe { d_scratch.add(ith * scratch_per) };

    // 2D chunking: dim0 = row quads (4 rows each), dim1 = token chunks
    let tok_chunk_size = 16usize;
    let n_tok_chunks = (n_tokens + tok_chunk_size - 1) / tok_chunk_size;
    let total_chunks = full_quads * n_tok_chunks;

    let mut chunk = ith as i32;
    while (chunk as usize) < total_chunks {
        let quad = (chunk as usize) % full_quads;
        let tok_idx = (chunk as usize) / full_quads;
        let t_start = tok_idx * tok_chunk_size;
        let t_end = (t_start + tok_chunk_size).min(n_tokens);
        let base_row = quad * 4;

        for t in t_start..t_end {
            let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
            let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
            let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
            let out_ptr = unsafe { output.add(t * output_stride + base_row) };

            match dtype {
                GGML_TYPE_Q4_K => {
                    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
                    unsafe {
                        ffi_inference::q4k_dot_q8k_4row(
                            weight.add(base_row * row_bytes),
                            weight.add((base_row + 1) * row_bytes),
                            weight.add((base_row + 2) * row_bytes),
                            weight.add((base_row + 3) * row_bytes),
                            q8, bsums, out_ptr,
                            n_blocks as i32, q8_d, pow2.as_ptr(),
                        );
                    }
                }
                GGML_TYPE_Q5_K => {
                    let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
                    unsafe {
                        ffi_inference::q5k_dot_q8k_4row(
                            weight.add(base_row * row_bytes),
                            weight.add((base_row + 1) * row_bytes),
                            weight.add((base_row + 2) * row_bytes),
                            weight.add((base_row + 3) * row_bytes),
                            q8, bsums, out_ptr,
                            n_blocks as i32, q8_d, pow2.as_ptr(),
                        );
                    }
                }
                GGML_TYPE_Q6_K => {
                    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
                    unsafe {
                        let d0 = my_scratch;
                        let d1 = my_scratch.add(n_blocks);
                        let d2 = my_scratch.add(n_blocks * 2);
                        let d3 = my_scratch.add(n_blocks * 3);
                        for blk in 0..n_blocks {
                            let off = 208;
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
                            q8, bsums, out_ptr,
                            n_blocks as i32, d0, d1, d2, d3,
                        );
                    }
                }
                _ => panic!("unsupported weight dtype {dtype}"),
            }
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }

    // Remainder rows (n_rows % 4), thread 0 only
    if ith == 0 {
        let base = full_quads * 4;
        for r in 0..(n_rows % 4) {
            let row = base + r;
            for t in 0..n_tokens {
                let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
                let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
                let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
                let out_ptr = unsafe { output.add(t * output_stride + row) };
                match dtype {
                    GGML_TYPE_Q4_K => {
                        let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
                        unsafe {
                            *out_ptr = ffi_inference::q4k_dot_q8k(
                                weight.add(row * row_bytes), q8, bsums,
                                n_blocks as i32, q8_d, pow2.as_ptr(),
                            );
                        }
                    }
                    GGML_TYPE_Q5_K => {
                        let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
                        unsafe {
                            *out_ptr = ffi_inference::q5k_dot_q8k(
                                weight.add(row * row_bytes), q8, bsums,
                                n_blocks as i32, q8_d, pow2.as_ptr(),
                            );
                        }
                    }
                    GGML_TYPE_Q6_K => {
                        let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
                        unsafe {
                            let d0 = my_scratch;
                            for blk in 0..n_blocks {
                                let w = weight.add(row * row_bytes + blk * Q6K_BLOCK_BYTES + 208);
                                let raw = u16::from_le_bytes([*w, *w.add(1)]);
                                *d0.add(blk) = f16_to_f32_scalar(raw) * *q8_d.add(blk);
                            }
                            *out_ptr = ffi_inference::q6k_dot_q8k(
                                weight.add(row * row_bytes), q8, bsums,
                                n_blocks as i32, d0,
                            );
                        }
                    }
                    _ => panic!("unsupported weight dtype {dtype}"),
                }
            }
        }
    }
}
